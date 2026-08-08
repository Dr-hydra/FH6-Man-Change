#!/usr/bin/env python3
"""Transplant and patch MatI payloads while preserving a target modelbin layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle, rebuild_with_blob_data
from patch_fh6_racesuit_materials import (
    decode_7bit,
    encode_7bit,
    material_id,
    material_name,
    parameter_end,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--transplant",
        action="append",
        required=True,
        metavar="TARGET_ID=DONOR_ID",
    )
    parser.add_argument(
        "--texture",
        action="append",
        default=[],
        metavar="TARGET_ID=PARAMETER_HASH=GAME_PATH",
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="TARGET_ID=ASCII_NAME",
        help="Replace outer MatI Name metadata; the new ASCII name must have the same byte length.",
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_transplants(values: list[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        try:
            raw_target, raw_donor = value.split("=", 1)
            target_id, donor_id = int(raw_target, 0), int(raw_donor, 0)
        except ValueError as exc:
            raise ValueError(f"Invalid transplant mapping {value!r}") from exc
        if target_id in result:
            raise ValueError(f"Duplicate target material ID {target_id}")
        result[target_id] = donor_id
    return result


def parse_textures(values: list[str]) -> dict[int, dict[int, str]]:
    result: dict[int, dict[int, str]] = {}
    for value in values:
        try:
            raw_target, raw_hash, path = value.split("=", 2)
            target_id, parameter_hash = int(raw_target, 0), int(raw_hash, 16)
        except ValueError as exc:
            raise ValueError(f"Invalid texture patch {value!r}") from exc
        if not path:
            raise ValueError(f"Texture patch has an empty path: {value!r}")
        patches = result.setdefault(target_id, {})
        if parameter_hash in patches:
            raise ValueError(f"Duplicate texture parameter 0x{parameter_hash:08X} for material {target_id}")
        patches[parameter_hash] = path
    return result


def parse_renames(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        try:
            raw_target, name = value.split("=", 1)
            target_id = int(raw_target, 0)
            name.encode("ascii")
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError(f"Invalid material rename {value!r}") from exc
        if not name or target_id in result:
            raise ValueError(f"Invalid or duplicate material rename {value!r}")
        result[target_id] = name
    return result


def patch_texture_parameters(data: bytes, patches: dict[int, str]) -> tuple[bytes, list[dict]]:
    if not patches:
        return data, []
    count = data[0]
    offset = 1
    records: list[bytes] = []
    changes: list[dict] = []
    remaining = dict(patches)
    for _ in range(count):
        start = offset
        offset, name_hash, parameter_type, value_offset = parameter_end(data, offset)
        record = data[start:offset]
        if name_hash in remaining:
            if parameter_type != 6:
                raise ValueError(f"Parameter 0x{name_hash:08X} is not a Texture2D")
            old_length, old_path_offset = decode_7bit(data, value_offset)
            old_path = data[old_path_offset : old_path_offset + old_length].decode("utf-8")
            new_path = remaining.pop(name_hash)
            path_bytes = new_path.encode("utf-8")
            path_hash = zlib.crc32(new_path.lower().encode("utf-8")) & 0xFFFFFFFF
            record = (
                data[start:value_offset]
                + encode_7bit(len(path_bytes))
                + path_bytes
                + struct.pack("<I", path_hash)
            )
            changes.append(
                {
                    "parameter_hash": f"{name_hash:08x}",
                    "old_path": old_path,
                    "new_path": new_path,
                }
            )
        records.append(record)
    if remaining:
        missing = ", ".join(f"0x{value:08X}" for value in sorted(remaining))
        raise ValueError(f"MTPR is missing requested texture parameters: {missing}")
    return bytes([count]) + b"".join(records) + data[offset:], changes


def patch_material_payload(payload: bytes, patches: dict[int, str]) -> tuple[bytes, list[dict]]:
    nested = parse_bundle(payload)
    shaders = [blob for blob in nested.blobs if blob.tag == "MTPR"]
    if len(shaders) != 1 or shaders[0].version != (2, 0):
        raise ValueError("Expected one nested MTPR 2.0 shader parameter blob")
    shader = shaders[0]
    patched_shader, changes = patch_texture_parameters(shader.data, patches)
    result = rebuild_with_blob_data(nested, {shader.index: patched_shader})
    parse_bundle(result)
    return result, changes


def materials_by_id(bundle) -> dict[int, object]:
    result = {material_id(blob): blob for blob in bundle.blobs if blob.tag == "MatI"}
    if len(result) != sum(blob.tag == "MatI" for blob in bundle.blobs):
        raise ValueError("Duplicate MatI material IDs")
    return result


def main() -> int:
    args = arguments()
    target_path = args.target.resolve(strict=True)
    donor_path = args.donor.resolve(strict=True)
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    transplants = parse_transplants(args.transplant)
    textures = parse_textures(args.texture)
    renames = parse_renames(args.rename)
    if set(textures) - set(transplants):
        raise ValueError("Texture patches must target a transplanted material ID")
    if set(renames) - set(transplants):
        raise ValueError("Material renames must target a transplanted material ID")

    target_data = target_path.read_bytes()
    donor_data = donor_path.read_bytes()
    target_bundle = parse_bundle(target_data)
    donor_bundle = parse_bundle(donor_data)
    target_materials = materials_by_id(target_bundle)
    donor_materials = materials_by_id(donor_bundle)
    if set(transplants) - set(target_materials):
        raise ValueError("A target material ID does not exist")
    if set(transplants.values()) - set(donor_materials):
        raise ValueError("A donor material ID does not exist")

    replacements: dict[int, bytes] = {}
    changes = []
    for target_id, donor_id in sorted(transplants.items()):
        target_blob = target_materials[target_id]
        donor_blob = donor_materials[donor_id]
        payload, texture_changes = patch_material_payload(
            donor_blob.data,
            textures.get(target_id, {}),
        )
        replacements[target_blob.index] = payload
        changes.append(
            {
                "target_id": target_id,
                "target_name": material_name(target_blob),
                "donor_id": donor_id,
                "donor_name": material_name(donor_blob),
                "old_payload_sha256": sha256_bytes(target_blob.data),
                "new_payload_sha256": sha256_bytes(payload),
                "texture_changes": texture_changes,
                "new_target_name": renames.get(target_id, material_name(target_blob)),
            }
        )

    output_data = rebuild_with_blob_data(target_bundle, replacements)
    if renames:
        renamed_data = bytearray(output_data)
        rebuilt_bundle = parse_bundle(output_data)
        rebuilt_materials = materials_by_id(rebuilt_bundle)
        for target_id, name in renames.items():
            entries = [entry for entry in rebuilt_materials[target_id].metadata if entry.tag == "Name"]
            if len(entries) != 1:
                raise ValueError(f"Material {target_id} does not have one Name metadata entry")
            entry = entries[0]
            encoded = name.encode("ascii")
            if len(encoded) != len(entry.value):
                raise ValueError(
                    f"Material {target_id} rename must remain {len(entry.value)} bytes, got {len(encoded)}"
                )
            renamed_data[entry.value_offset : entry.value_offset + len(encoded)] = encoded
        output_data = bytes(renamed_data)
    output_path.write_bytes(output_data)
    output_bundle = parse_bundle(output_data)
    for blob in output_bundle.blobs:
        if blob.tag == "MatI":
            parse_bundle(blob.data)
    parsed = inspector.inspect(output_path)
    if parsed["parsed"]["errors"]:
        raise ValueError(f"Output parser errors: {parsed['parsed']['errors']}")
    output_materials = materials_by_id(output_bundle)
    for target_id, name in renames.items():
        if material_name(output_materials[target_id]) != name:
            raise ValueError(f"Material {target_id} rename verification failed")

    for original, rebuilt in zip(target_bundle.blobs, output_bundle.blobs):
        if original.index not in replacements and original.data != rebuilt.data:
            raise ValueError(f"Unexpected change to blob {original.index}:{original.tag}")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "target": {
            "path": str(target_path),
            "bytes": len(target_data),
            "sha256": sha256(target_path),
        },
        "donor": {
            "path": str(donor_path),
            "bytes": len(donor_data),
            "sha256": sha256(donor_path),
        },
        "output": {
            "path": str(output_path),
            "bytes": len(output_data),
            "sha256": sha256(output_path),
        },
        "changes": changes,
        "validation": {
            "outer_bundle": True,
            "nested_material_bundles": True,
            "modelbin_parser": True,
            "non_target_blob_payloads_preserved": True,
            "material_names_verified": True,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_MATERIAL_TRANSPLANTS=" + json.dumps(report["output"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
