#!/usr/bin/env python3
"""Build the three-archive offline Si Display LOD0 package without deploying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from add_fh6_aligned_zip import build as add_aligned_entries
from rebuild_fh6_aligned_zip import rebuild, verify


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
MILESTONE = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-06-display-package-lod0-v001"
)
MATERIAL_ROOT = WORKSPACE / "work" / "si" / "fbx-source" / "material-pipeline-v002"
DEFAULT_CHARACTERS_BASELINE = (
    WORKSPACE
    / "work"
    / "si"
    / "components"
    / "baselines"
    / "textures-characters"
    / "characters.original.zip"
)
DEFAULT_HELMET_ARCHIVE = (
    WORKSPACE
    / "work"
    / "si"
    / "components"
    / "baselines"
    / "helmet"
    / "Helmet_Race_Modern.zip"
)
DEFAULT_OUTFIT_ARCHIVE = (
    WORKSPACE
    / "work"
    / "donors"
    / "Outfit_Race_Suit_Modern_F"
    / "original"
    / "Outfit_Race_Suit_Modern_F.zip"
)
DEFAULT_HELMET_MODELBIN = (
    MATERIAL_ROOT
    / "candidates"
    / "Helmet_Race_Modern.lod0-materials-v002.modelbin"
)
DEFAULT_OUTFIT_MODELBIN = (
    MATERIAL_ROOT
    / "candidates"
    / "Outfit_Race_Suit_Modern_F.lod0-materials-v002.modelbin"
)
DEFAULT_SWATCH_MANIFEST = (
    MATERIAL_ROOT / "swatches" / "si-display-swatches-v002.json"
)
DEFAULT_OUTPUT = MILESTONE / "archives"
EXPECTED_CHARACTERS_SHA256 = (
    "b54507011707db4f4037d8f63d54519fd9d8d45ea527c3cf93ea36ad1d16e77e"
)
EXPECTED_CHARACTERS_ENTRIES = 1990


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--characters-baseline", type=Path, default=DEFAULT_CHARACTERS_BASELINE
    )
    parser.add_argument("--helmet-archive", type=Path, default=DEFAULT_HELMET_ARCHIVE)
    parser.add_argument("--outfit-archive", type=Path, default=DEFAULT_OUTFIT_ARCHIVE)
    parser.add_argument("--helmet-modelbin", type=Path, default=DEFAULT_HELMET_MODELBIN)
    parser.add_argument("--outfit-modelbin", type=Path, default=DEFAULT_OUTFIT_MODELBIN)
    parser.add_argument("--swatch-manifest", type=Path, default=DEFAULT_SWATCH_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def write_rebuilt_archive(
    source: Path,
    output: Path,
    replacements: dict[str, Path],
    report_path: Path,
) -> dict[str, Any]:
    source_record = file_record(source)
    replacement_report, entry_count = rebuild(source, output, replacements)
    validation = verify(output, replacements, entry_count)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": source_record,
        "output": file_record(output),
        "replacements": replacement_report,
        "validation": validation,
        "deployment": {"applied": False, "backup": None},
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def require_offline_evidence(milestone: Path) -> dict[str, Any]:
    verification_dir = milestone / "verification"
    preview_dir = milestone / "previews"
    paths = {
        "helmet_structure_material": verification_dir
        / "helmet.final.structure-material.report.json",
        "outfit_structure_material": verification_dir
        / "outfit.final.structure-material.report.json",
        "helmet_inspect": verification_dir / "helmet.final.inspect.json",
        "outfit_inspect": verification_dir / "outfit.final.inspect.json",
        "material_preview": preview_dir / "material-preview.report.json",
    }
    reports = {
        key: json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    for key in ("helmet_structure_material", "outfit_structure_material"):
        if reports[key]["summary"]["failed"]:
            raise ValueError(f"Offline structure/material gate failed: {key}")
    if reports["helmet_inspect"]["parsed"]["errors"]:
        raise ValueError("Final Helmet inspection contains parser errors")
    if reports["outfit_inspect"]["parsed"]["errors"]:
        raise ValueError("Final Outfit inspection contains parser errors")
    if reports["material_preview"]["validation"]["hard_error_count"]:
        raise ValueError("Final material preview contains hard errors")
    return {
        key: {"path": str(paths[key].resolve()), "sha256": sha256_path(paths[key])}
        for key in paths
    }


def main() -> int:
    args = arguments()
    characters_baseline = args.characters_baseline.resolve(strict=True)
    helmet_archive = args.helmet_archive.resolve(strict=True)
    outfit_archive = args.outfit_archive.resolve(strict=True)
    helmet_modelbin = args.helmet_modelbin.resolve(strict=True)
    outfit_modelbin = args.outfit_modelbin.resolve(strict=True)
    swatch_manifest_path = args.swatch_manifest.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse package archive directory {output}")

    characters_hash = sha256_path(characters_baseline)
    if characters_hash != EXPECTED_CHARACTERS_SHA256:
        raise ValueError(
            "Characters baseline hash differs from the locked clean retail backup: "
            f"{characters_hash}"
        )
    with zipfile.ZipFile(characters_baseline) as zipped:
        names = zipped.namelist()
        custom = [
            name for name in names if "si_" in name.lower() or "jsspsi" in name.lower()
        ]
        if len(names) != EXPECTED_CHARACTERS_ENTRIES or custom:
            raise ValueError(
                "Characters baseline is not clean: "
                f"entries={len(names)}, custom_entries={custom[:10]}"
            )
        if zipped.testzip() is not None:
            raise ValueError("Characters baseline failed ZIP CRC validation")

    evidence = require_offline_evidence(output.parent)
    output.mkdir(parents=True)

    helmet_output = output / "Helmet_Race_Modern.si-display-lod0-v001.zip"
    helmet_report_path = output / "Helmet_Race_Modern.si-display-lod0-v001.zip.report.json"
    helmet_report = write_rebuilt_archive(
        helmet_archive,
        helmet_output,
        {"Helmet_Race_Modern.modelbin": helmet_modelbin},
        helmet_report_path,
    )

    outfit_output = output / "Outfit_Race_Suit_Modern_F.si-display-lod0-v001.zip"
    outfit_report_path = (
        output / "Outfit_Race_Suit_Modern_F.si-display-lod0-v001.zip.report.json"
    )
    outfit_report = write_rebuilt_archive(
        outfit_archive,
        outfit_output,
        {"Outfit_Race_Suit_Modern_F.modelbin": outfit_modelbin},
        outfit_report_path,
    )

    swatch_manifest = json.loads(swatch_manifest_path.read_text(encoding="utf-8"))
    additions: dict[str, Path] = {}
    for key, record in sorted(swatch_manifest["swatches"].items()):
        swatch = Path(record["path"]).resolve(strict=True)
        entry = f"Swatches/{record['filename']}"
        if entry in additions:
            raise ValueError(f"Duplicate swatch archive entry: {entry}")
        additions[entry] = swatch
    if len(additions) != 20:
        raise ValueError(f"Expected 20 final swatches, found {len(additions)}")

    characters_output = output / "characters.si-display-lod0-v001.zip"
    characters_report_path = output / "characters.si-display-lod0-v001.zip.report.json"
    characters_result = add_aligned_entries(
        characters_baseline, characters_output, additions
    )
    characters_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": file_record(characters_baseline),
        "output": file_record(characters_output),
        "validation": characters_result,
        "deployment": {"applied": False, "backup": None},
    }
    characters_report_path.write_text(
        json.dumps(characters_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    package_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "scope": "FH6 Si Display LOD0 offline package; _Driver excluded; not deployed",
        "inputs": {
            "helmet_archive": file_record(helmet_archive),
            "outfit_archive": file_record(outfit_archive),
            "characters_archive": file_record(characters_baseline),
            "helmet_modelbin": file_record(helmet_modelbin),
            "outfit_modelbin": file_record(outfit_modelbin),
            "swatch_manifest": file_record(swatch_manifest_path),
        },
        "archives": {
            "helmet": {
                "game_relative_path": "media/Cinematic_Assets/Characters/Garments/Hat/Helmet_Race_Modern.zip",
                **file_record(helmet_output),
                "report": str(helmet_report_path),
                "entries": helmet_report["validation"]["entries"],
            },
            "outfit": {
                "game_relative_path": "media/Cinematic_Assets/Characters/Garments/Outfit/Outfit_Race_Suit_Modern_F.zip",
                **file_record(outfit_output),
                "report": str(outfit_report_path),
                "entries": outfit_report["validation"]["entries"],
            },
            "characters": {
                "game_relative_path": "media/_library/TexturesPG/characters.zip",
                **file_record(characters_output),
                "report": str(characters_report_path),
                "entries_before": characters_result["entries_before"],
                "entries_after": characters_result["entries_after"],
                "swatches_added": len(additions),
            },
        },
        "offline_evidence": evidence,
        "validation": {
            "modelbin_structure_material_gates": True,
            "modelbin_inspection_parse_errors": 0,
            "material_preview_hard_errors": 0,
            "zip_crc": True,
            "zip_payload_alignment": 4096,
            "zip_extra_fields_verified": True,
            "characters_clean_baseline": True,
            "characters_swatches_added": len(additions),
            "game_directory_modified": False,
            "game_validated": False,
        },
        "limitations": {
            "lod": "LOD0 geometry is currently shared by donor-enabled LOD flags; native LOD1-L3 are not yet packaged.",
            "unsupported_material_channels": swatch_manifest["coverage"][
                "contract_source_files_not_supported_by_current_donor_mtpr"
            ],
            "claim": "Offline structural and decoded-pixel evidence does not substitute for the in-game Display gate.",
        },
        "license_guard": "Local technical validation only; do not redistribute source-model-derived archives.",
    }
    package_report_path = output / "si-display-lod0-v001.package.json"
    package_report_path.write_text(
        json.dumps(package_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "SI_DISPLAY_LOD0_PACKAGE="
        + json.dumps(
            {
                "output": str(output),
                "manifest": str(package_report_path),
                "archives": len(package_report["archives"]),
                "swatches": len(additions),
                "deployed": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
