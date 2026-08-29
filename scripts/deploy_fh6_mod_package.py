#!/usr/bin/env python3
"""Validate and deploy a complete FH6 character Mod package.

The script is dry-run by default. --apply backs up and replaces the declared
ZIPs, then applies the declared characters.zip swatch patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from add_fh6_aligned_zip import build as build_aligned_zip
from fh6_xml_patch import apply_xml_patch, validate_xml_patch_spec
from rebuild_fh6_aligned_zip import rebuild as rebuild_aligned_zip
from rebuild_fh6_aligned_zip import verify as verify_rebuilt_zip


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "mod.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported mod.json schema_version")
    contract = manifest.get("payload_contract", {})
    replacement_count = contract.get("replacement_count")
    injection_count = contract.get("characters_injection_count", 1)
    metadata_count = contract.get("metadata_count", 1)
    total_payload_files = contract.get("total_payload_files")
    if (
        not isinstance(replacement_count, int)
        or replacement_count < 1
        or injection_count != 1
        or metadata_count != 1
        or total_payload_files != replacement_count + injection_count + metadata_count
    ):
        raise ValueError("Invalid package payload contract")
    xml_patches = manifest.get("xml_patches", [])
    xml_patch_count = contract.get("xml_patch_count", 0)
    if (
        not isinstance(xml_patches, list)
        or not isinstance(xml_patch_count, int)
        or xml_patch_count != len(xml_patches)
    ):
        raise ValueError("Invalid XML patch contract")
    seen_xml_targets: set[str] = set()
    for spec in xml_patches:
        validate_xml_patch_spec(spec)
        target = spec["game_target"]
        if target in seen_xml_targets:
            raise ValueError(f"Duplicate XML patch target: {target}")
        seen_xml_targets.add(target)
    return manifest


def target_path(game_root: Path, game_target: str) -> Path:
    relative = PurePosixPath(game_target)
    if not relative.parts or relative.parts[0] != "media" or ".." in relative.parts:
        raise ValueError(f"Unsafe game target: {game_target}")
    target = (game_root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(game_root)
    except ValueError as exc:
        raise ValueError(f"Game target escapes supplied game root: {game_target}") from exc
    return target


def validate_replacements(package_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    replacements = manifest["replacements"]
    expected_count = manifest["payload_contract"]["replacement_count"]
    if len(replacements) != expected_count:
        raise ValueError(f"mod.json must describe exactly {expected_count} replacements")
    validated = []
    seen_files: set[str] = set()
    seen_targets: set[str] = set()
    for item in replacements:
        relative = item["file"]
        if relative in seen_files or item["game_target"] in seen_targets:
            raise ValueError("Duplicate replacement file or target")
        seen_files.add(relative)
        seen_targets.add(item["game_target"])
        source = (package_root / relative).resolve(strict=True)
        try:
            source.relative_to(package_root)
        except ValueError as exc:
            raise ValueError(f"Replacement escapes package root: {relative}") from exc
        actual = sha256_file(source)
        if actual != item["sha256"]:
            raise ValueError(f"Replacement hash mismatch: {relative}")
        validated.append({**item, "source": source})
    return validated


def validate_injection(
    package_root: Path, manifest: dict[str, Any], temporary_root: Path
) -> tuple[dict[str, Path], dict[str, Any]]:
    injection = manifest["characters_zip_injection"]
    payload = (package_root / injection["payload_file"]).resolve(strict=True)
    try:
        payload.relative_to(package_root)
    except ValueError as exc:
        raise ValueError("Swatch payload escapes package root") from exc
    if sha256_file(payload) != injection["sha256"]:
        raise ValueError("Swatch payload hash mismatch")
    entries = injection["entries"]
    if injection["entry_count"] != len(entries) or not entries:
        raise ValueError("Swatch injection entry_count is invalid")
    dependencies = manifest["material_dependencies"]
    replacement_count = injection.get("replacement_count")
    addition_count = injection.get("addition_count")
    if (
        not isinstance(replacement_count, int)
        or replacement_count < 0
        or not isinstance(addition_count, int)
        or addition_count < 0
        or len(entries) != replacement_count + addition_count
        or dependencies.get("referenced_entry_count")
        != dependencies.get("unchanged_base_game_entry_count", 0)
        + dependencies.get("patch_entry_count", 0)
        or dependencies.get("patch_entry_count")
        != dependencies.get("replacement_entry_count", 0)
        + dependencies.get("addition_entry_count", 0)
        or replacement_count != dependencies.get("replacement_entry_count", 0)
        or addition_count != dependencies.get("addition_entry_count", 0)
    ):
        raise ValueError("Swatch injection counts do not match the declared material dependency contract")
    operations = [item["operation"] for item in entries]
    if (
        operations.count("replace-existing") != replacement_count
        or operations.count("append-missing") != addition_count
    ):
        raise ValueError("Swatch patch operations do not match the declared counts")

    additions: dict[str, Path] = {}
    with zipfile.ZipFile(payload) as archive:
        expected_names = [item["archive_entry"] for item in entries]
        if archive.namelist() != expected_names:
            raise ValueError("Swatch payload entry list differs from mod.json")
        for item in entries:
            name = item["archive_entry"]
            if name in additions:
                raise ValueError(f"Duplicate swatch entry: {name}")
            data = archive.read(name)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Swatch payload hash mismatch: {name}")
            destination = temporary_root / Path(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            additions[name] = destination
    return additions, injection


def existing_entry_states(
    target: Path, patches: dict[str, Path], entries: list[dict[str, Any]]
) -> tuple[list[str], list[str], list[str]]:
    entry_by_name = {item["archive_entry"]: item for item in entries}
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        matching = []
        replacements = []
        additions = []
        for name, path in sorted(patches.items()):
            item = entry_by_name[name]
            operation = item["operation"]
            if name not in names:
                if operation == "replace-existing":
                    raise ValueError(f"Required retail swatch is missing: {name}")
                additions.append(name)
                continue
            actual_hash = hashlib.sha256(archive.read(name)).hexdigest()
            if actual_hash == sha256_file(path):
                matching.append(name)
            elif operation == "replace-existing" and actual_hash == item["clean_sha256"]:
                replacements.append(name)
            else:
                raise ValueError(
                    f"Target characters.zip has an unknown conflicting swatch: {name} "
                    f"({actual_hash})"
                )
    return matching, replacements, additions


def validate_base_game_dependencies(
    target: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    dependencies = manifest["material_dependencies"]["base_game_entries"]
    declared_count = manifest["material_dependencies"].get("unchanged_base_game_entry_count")
    if not isinstance(declared_count, int) or len(dependencies) != declared_count:
        raise ValueError("mod.json base-game dependency count is inconsistent")
    with zipfile.ZipFile(target) as archive:
        by_name = {entry.filename: entry for entry in archive.infolist()}
        missing = [
            item["archive_entry"]
            for item in dependencies
            if item["archive_entry"] not in by_name
        ]
        if missing:
            raise ValueError(
                "Target characters.zip is missing base-game swatches: "
                + ", ".join(missing)
            )
        content_mismatches = []
        for item in dependencies:
            name = item["archive_entry"]
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                content_mismatches.append(name)
    if content_mismatches:
        raise ValueError(
            "Target characters.zip has changed base-game swatches: "
            + ", ".join(content_mismatches)
        )
    return {
        "required": len(dependencies),
        "present": len(dependencies),
        "content_mismatches_against_tested_baseline": content_mismatches,
    }


def deploy_copy(source: Path, target: Path, stamp: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(target.name + f".bak-{stamp}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup}")
    shutil.copy2(target, backup)
    temporary = target.with_name(target.name + f".tmp-{stamp}")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise ValueError(f"Temporary copy hash mismatch: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(backup)


def prepare_xml_patches(
    game_root: Path,
    manifest: dict[str, Any],
    temporary_root: Path,
) -> list[tuple[Path, Path, dict[str, Any]]]:
    prepared = []
    for index, spec in enumerate(manifest.get("xml_patches", [])):
        target = target_path(game_root, spec["game_target"])
        if not target.is_file():
            raise FileNotFoundError(f"Missing target customization XML: {target}")
        original = target.read_bytes()
        patched, operations = apply_xml_patch(original, spec)
        output = temporary_root / f"xml-patch-{index}.xml"
        output.write_bytes(patched)
        record = {
            "target": str(target),
            "target_sha256_before": hashlib.sha256(original).hexdigest(),
            "candidate_sha256": hashlib.sha256(patched).hexdigest(),
            "already_patched": patched == original,
            "operations": operations,
        }
        prepared.append((output, target, record))
    return prepared


def deploy_package(
    package_root: Path,
    game_root: Path,
    *,
    apply: bool,
    report_path: Path,
) -> dict[str, Any]:
    """Validate and optionally install an extracted complete Mod package."""
    package_root = package_root.resolve(strict=True)
    game_root = game_root.resolve(strict=True)
    report_path = report_path.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    manifest = load_manifest(package_root)
    replacements = validate_replacements(package_root, manifest)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report: dict[str, Any] = {
        "schema_version": 1,
        "package": {"path": str(package_root), "id": manifest["mod"]["id"]},
        "game_root": str(game_root),
        "apply": apply,
        "replacements": [],
        "xml_patches": [],
        "characters_zip": {},
    }

    resolved_replacements = []
    for item in replacements:
        target = target_path(game_root, item["game_target"])
        if not target.is_file():
            raise FileNotFoundError(f"Missing target game archive: {target}")
        target_hash = sha256_file(target)
        record = {
            "file": item["file"],
            "target": str(target),
            "source_sha256": item["sha256"],
            "target_sha256_before": target_hash,
            "already_installed": target_hash == item["sha256"],
        }
        report["replacements"].append(record)
        resolved_replacements.append((item, target, record))

    with tempfile.TemporaryDirectory(prefix="fh6-mod-deploy-") as temporary:
        temporary_root = Path(temporary)
        prepared_xml = prepare_xml_patches(game_root, manifest, temporary_root)
        report["xml_patches"] = [record for _output, _target, record in prepared_xml]
        additions, injection = validate_injection(package_root, manifest, temporary_root)
        target = target_path(game_root, injection["game_target"])
        if not target.is_file():
            raise FileNotFoundError(f"Missing target characters.zip: {target}")
        base_game = validate_base_game_dependencies(target, manifest)
        patch_entries = injection["entries"]
        matching, replacement_names, addition_names = existing_entry_states(
            target, additions, patch_entries
        )
        replacement_paths = {name: additions[name] for name in replacement_names}
        addition_paths = {name: additions[name] for name in addition_names}
        zip_record: dict[str, Any] = {
            "target": str(target),
            "target_sha256_before": sha256_file(target),
            "entry_count": len(additions),
            "already_present": not replacement_paths and not addition_paths,
            "matching_before": len(matching),
            "replace_before": len(replacement_paths),
            "append_before": len(addition_paths),
            "replaced": 0,
            "added": 0,
            "base_game_dependencies": base_game,
        }
        patched_output = None
        patch_result = None
        intermediate_output = None
        if apply and (replacement_paths or addition_paths):
            patch_source = target
            validations: dict[str, Any] = {}
            if replacement_paths:
                intermediate_output = target.with_name(target.name + f".rebuilt-{stamp}")
                replacement_report, entry_count = rebuild_aligned_zip(
                    target, intermediate_output, replacement_paths
                )
                validations["replacement"] = {
                    "replacements": replacement_report,
                    "archive": verify_rebuilt_zip(
                        intermediate_output, replacement_paths, entry_count
                    ),
                }
                patch_source = intermediate_output
            if addition_paths:
                patched_output = target.with_name(target.name + f".patched-{stamp}")
                patch_result = build_aligned_zip(patch_source, patched_output, addition_paths)
                validations["addition"] = patch_result["archive"]
            else:
                patched_output = intermediate_output
            patch_result = {"validations": validations}

        if apply:
            changed_targets = [
                replacement_target
                for _item, replacement_target, record in resolved_replacements
                if not record["already_installed"]
            ]
            changed_targets.extend(
                xml_target
                for _xml_output, xml_target, xml_record in prepared_xml
                if not xml_record["already_patched"]
            )
            if patched_output is not None:
                changed_targets.append(target)
            for changed_target in changed_targets:
                backup = changed_target.with_name(changed_target.name + f".bak-{stamp}")
                if backup.exists():
                    raise FileExistsError(f"Refusing to overwrite backup: {backup}")

            for item, replacement_target, record in resolved_replacements:
                if record["already_installed"]:
                    continue
                record["backup"] = deploy_copy(item["source"], replacement_target, stamp)
                record["target_sha256_after"] = sha256_file(replacement_target)

            for xml_output, xml_target, xml_record in prepared_xml:
                if xml_record["already_patched"]:
                    continue
                xml_record["backup"] = deploy_copy(xml_output, xml_target, stamp)
                xml_record["target_sha256_after"] = sha256_file(xml_target)

            if patched_output is not None and patch_result is not None:
                try:
                    zip_record["backup"] = deploy_copy(patched_output, target, stamp)
                finally:
                    if patched_output.exists():
                        patched_output.unlink()
                    if intermediate_output is not None and intermediate_output.exists():
                        intermediate_output.unlink()
                zip_record.update(
                    {
                        "replaced": len(replacement_paths),
                        "added": len(addition_paths),
                        "target_sha256_after": sha256_file(target),
                        "alignment_validation": patch_result["validations"],
                    }
                )
        report["characters_zip"] = zip_record

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    parser.add_argument("game_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = deploy_package(
        args.package_root,
        args.game_root,
        apply=args.apply,
        report_path=args.report,
    )
    print("FH6_MOD_DEPLOY=" + json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
