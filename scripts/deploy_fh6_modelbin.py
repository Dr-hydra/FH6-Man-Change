#!/usr/bin/env python3
"""Safely replace one FH6 character ZIP's modelbin with a validated candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from inspect_modelbin import inspect


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_asset(game_root: Path, asset_name: str) -> Path:
    matches = sorted(game_root.rglob(asset_name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {asset_name!r} below {game_root}, found {len(matches)}"
        )
    return matches[0]


def local_extra(data: bytes, header_offset: int) -> bytes:
    if data[header_offset : header_offset + 4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid ZIP local header at 0x{header_offset:X}")
    name_length, extra_length = struct.unpack_from("<HH", data, header_offset + 26)
    extra_offset = header_offset + 30 + name_length
    return data[extra_offset : extra_offset + extra_length]


def restore_single_entry_local_extra(source_zip: Path, output_zip: Path) -> dict[str, int]:
    source_data = source_zip.read_bytes()
    output_data = bytearray(output_zip.read_bytes())
    with zipfile.ZipFile(source_zip, "r") as source, zipfile.ZipFile(output_zip, "r") as output:
        source_entries = source.infolist()
        output_entries = output.infolist()
        if len(source_entries) != 1 or len(output_entries) != 1:
            raise RuntimeError("Aligned FH6 replacement currently requires a single-entry ZIP")
        source_entry = source_entries[0]
        output_entry = output_entries[0]
        source_extra = local_extra(source_data, source_entry.header_offset)
        output_extra = local_extra(output_data, output_entry.header_offset)
        name_length = struct.unpack_from("<H", output_data, output_entry.header_offset + 26)[0]
        extra_offset = output_entry.header_offset + 30 + name_length
    delta = len(source_extra) - len(output_extra)
    output_data[extra_offset : extra_offset + len(output_extra)] = source_extra
    struct.pack_into("<H", output_data, output_entry.header_offset + 28, len(source_extra))
    eocd_offset = output_data.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise RuntimeError("ZIP EOCD not found after local-extra restoration")
    central_offset = struct.unpack_from("<I", output_data, eocd_offset + 16)[0]
    struct.pack_into("<I", output_data, eocd_offset + 16, central_offset + delta)
    output_zip.write_bytes(output_data)
    payload_offset = output_entry.header_offset + 30 + name_length + len(source_extra)
    return {
        "source_local_extra_bytes": len(source_extra),
        "rebuilt_local_extra_bytes": len(output_extra),
        "payload_offset": payload_offset,
        "payload_alignment": payload_offset % 4096,
    }


def build_replacement(source_zip: Path, candidate: bytes, output_zip: Path) -> str:
    with zipfile.ZipFile(source_zip, "r") as source:
        entries = source.infolist()
        modelbins = [entry for entry in entries if entry.filename.lower().endswith(".modelbin")]
        if len(modelbins) != 1:
            raise RuntimeError(f"Expected one modelbin entry, found {len(modelbins)}")
        target_entry = modelbins[0]
        with zipfile.ZipFile(output_zip, "w", allowZip64=True) as output:
            output.comment = source.comment
            for entry in entries:
                payload = candidate if entry is target_entry else source.read(entry)
                # writestr preserves the entry name, timestamps, extra field and attributes.
                output.writestr(entry, payload)
    alignment = restore_single_entry_local_extra(source_zip, output_zip)
    if alignment["payload_alignment"] != 0:
        raise RuntimeError(f"Rebuilt modelbin payload is not 4096-byte aligned: {alignment}")
    return target_entry.filename


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--asset-name", default="Upper_Shirt_Tucked_F_Driver.zip")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true", help="Replace the game asset; default is dry-run")
    args = parser.parse_args()

    game_root = args.game_root.resolve(strict=True)
    candidate_path = args.candidate.resolve(strict=True)
    candidate = candidate_path.read_bytes()
    candidate_inspection = inspect(candidate_path)
    if candidate_inspection["parsed"]["errors"]:
        raise RuntimeError("Candidate modelbin has parser errors")
    asset = find_asset(game_root, args.asset_name).resolve(strict=True)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    backup = asset.with_name(asset.name + f".bak-{timestamp}")
    report_path = args.report.resolve() if args.report else candidate_path.parent / "deploy" / f"deployment-{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(asset, "r") as archive:
        original_entries = archive.infolist()
        original_modelbins = [entry for entry in original_entries if entry.filename.lower().endswith(".modelbin")]
        if len(original_modelbins) != 1:
            raise RuntimeError(f"Target archive has {len(original_modelbins)} modelbin entries")
        original_payload = archive.read(original_modelbins[0])
        if len(original_payload) == len(candidate) and sha256_bytes(original_payload) == sha256_bytes(candidate):
            raise RuntimeError("Candidate is byte-identical to the target payload")

    report = {
        "schema_version": 1,
        "created_local": datetime.now(timezone.utc).astimezone().isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "game_root": str(game_root),
        "asset": str(asset),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256_bytes(candidate),
        "original_archive_sha256": sha256_file(asset),
        "original_modelbin_sha256": sha256_bytes(original_payload),
        "candidate_modelbin_sha256": sha256_bytes(candidate),
        "entry": original_modelbins[0].filename,
        "backup": str(backup) if args.apply else None,
        "candidate_parse_errors": candidate_inspection["parsed"]["errors"],
    }

    if args.apply:
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite existing backup {backup}")
        shutil.copy2(asset, backup)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(prefix=asset.name + ".", suffix=".tmp", dir=asset.parent, delete=False) as handle:
                temp_name = Path(handle.name)
            entry_name = build_replacement(asset, candidate, temp_name)
            with zipfile.ZipFile(temp_name, "r") as verify:
                if verify.testzip() is not None:
                    raise RuntimeError("Rebuilt ZIP failed CRC validation")
                deployed = verify.read(entry_name)
            if sha256_bytes(deployed) != sha256_bytes(candidate):
                raise RuntimeError("Rebuilt ZIP payload hash does not match candidate")
            os.replace(temp_name, asset)
            report["deployed_archive_sha256"] = sha256_file(asset)
            report["deployed_modelbin_sha256"] = sha256_bytes(deployed)
        except Exception:
            if temp_name is not None and temp_name.exists():
                temp_name.unlink()
            if backup.exists() and not asset.exists():
                os.replace(backup, asset)
            raise

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
