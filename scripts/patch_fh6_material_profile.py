#!/usr/bin/env python3
"""Patch donor-template FH6 MatI payloads from an explicit JSON profile.

This writer never invents a MatI layout.  Each target material is replaced by
a verified donor MatI payload, then every Texture2D path in that payload must be
explicitly replaced.  Optional fixed-size scalar/vector edits are keyed by the
same CRC32 parameter hashes stored in MTPR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle, rebuild_with_blob_data
from fh6_material_codec import (
    decode_7bit,
    encode_7bit,
    material_id,
    material_name,
    parameter_end,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE / candidate


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hash(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value, 16) if not value.lower().startswith("0x") else int(value, 0)


def shader_info(payload: bytes) -> tuple[str, str, object, object]:
    nested = parse_bundle(payload)
    mati = next((blob for blob in nested.blobs if blob.tag == "MATI"), None)
    mtpr = next((blob for blob in nested.blobs if blob.tag == "MTPR"), None)
    if mati is None or mtpr is None or mtpr.version != (2, 0):
        raise ValueError("Template MatI must contain MATI and MTPR 2.0")
    shader = next(
        (entry.value.decode("utf-8") for entry in mati.metadata if entry.tag == "Name"),
        None,
    )
    atst = next((entry.value.hex() for entry in mati.metadata if entry.tag == "ATST"), None)
    if shader is None or atst is None:
        raise ValueError("Template MATI lacks Name or ATST metadata")
    return shader, atst, nested, mtpr


def parameter_name_hash(name: str) -> int:
    """Forza material parameter hashes are case-sensitive CRC32 values."""

    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def normalize_patches(spec: dict[str, Any], key: str) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for raw_hash, value in spec.get(key, {}).items():
        name_hash = parse_hash(raw_hash)
        if name_hash in result:
            raise ValueError(f"Duplicate {key} hash 0x{name_hash:08X}")
        result[name_hash] = value
    return result


def scalar_bytes(parameter_type: int, patch: dict[str, Any]) -> bytes:
    expected_type = int(patch["type"])
    if parameter_type != expected_type:
        raise ValueError(
            f"Scalar patch expects type {expected_type}, template uses {parameter_type}"
        )
    value = patch["value"]
    if parameter_type in (0, 1, 5, 9):
        values = [float(item) for item in value]
        if len(values) != 4:
            raise ValueError("Vector parameter patches require four values")
        return struct.pack("<4f", *values)
    if parameter_type == 2:
        return struct.pack("<f", float(value))
    if parameter_type in (3, 4):
        return struct.pack("<I", int(value))
    raise ValueError(f"Fixed-size patching does not support parameter type {parameter_type}")


def patch_mtpr(
    data: bytes,
    texture_patches: dict[int, str],
    value_patches: dict[int, dict[str, Any]],
    *,
    require_all_textures: bool,
) -> tuple[bytes, list[dict[str, Any]], set[int]]:
    count = data[0]
    offset = 1
    records: list[bytes] = []
    changes: list[dict[str, Any]] = []
    remaining_textures = dict(texture_patches)
    remaining_values = dict(value_patches)
    template_texture_hashes: set[int] = set()
    for _ in range(count):
        start = offset
        offset, name_hash, parameter_type, value_offset = parameter_end(data, offset)
        record = data[start:offset]
        if parameter_type == 6:
            template_texture_hashes.add(name_hash)
        if name_hash in remaining_textures:
            if parameter_type != 6:
                raise ValueError(f"Texture patch 0x{name_hash:08X} is not Texture2D")
            old_length, old_path_offset = decode_7bit(data, value_offset)
            old_path = data[old_path_offset : old_path_offset + old_length].decode("utf-8")
            new_path = remaining_textures.pop(name_hash)
            encoded = new_path.encode("utf-8")
            path_hash = zlib.crc32(new_path.lower().encode("utf-8")) & 0xFFFFFFFF
            record = (
                data[start:value_offset]
                + encode_7bit(len(encoded))
                + encoded
                + struct.pack("<I", path_hash)
            )
            changes.append(
                {
                    "kind": "texture",
                    "parameter_hash": f"{name_hash:08x}",
                    "old": old_path,
                    "new": new_path,
                }
            )
        elif name_hash in remaining_values:
            patch = remaining_values.pop(name_hash)
            encoded = scalar_bytes(parameter_type, patch)
            old = data[value_offset : value_offset + len(encoded)]
            record = data[start:value_offset] + encoded + data[value_offset + len(encoded) : offset]
            changes.append(
                {
                    "kind": "value",
                    "parameter_hash": f"{name_hash:08x}",
                    "parameter_name": patch.get("name"),
                    "type": parameter_type,
                    "old_hex": old.hex(),
                    "new_hex": encoded.hex(),
                    "new_value": patch["value"],
                }
            )
        records.append(record)
    missing = sorted(set(remaining_textures) | set(remaining_values))
    if missing:
        raise ValueError(
            "Template is missing requested MTPR hashes: "
            + ", ".join(f"0x{item:08X}" for item in missing)
        )
    if require_all_textures and set(texture_patches) != template_texture_hashes:
        missing_patches = template_texture_hashes - set(texture_patches)
        extra_patches = set(texture_patches) - template_texture_hashes
        raise ValueError(
            "Every template texture must be patched; missing="
            + ",".join(f"0x{item:08X}" for item in sorted(missing_patches))
            + " extra="
            + ",".join(f"0x{item:08X}" for item in sorted(extra_patches))
        )
    return bytes([count]) + b"".join(records) + data[offset:], changes, template_texture_hashes


def materials_by_id(bundle: object) -> dict[int, object]:
    materials = {material_id(blob): blob for blob in bundle.blobs if blob.tag == "MatI"}
    if len(materials) != sum(blob.tag == "MatI" for blob in bundle.blobs):
        raise ValueError("Duplicate material IDs")
    return materials


def main() -> int:
    args = arguments()
    input_path = args.input.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported material profile schema")

    source_data = input_path.read_bytes()
    source_bundle = parse_bundle(source_data)
    targets = materials_by_id(source_bundle)
    replacements: dict[int, bytes] = {}
    report_materials = []
    for spec in profile["materials"]:
        target_id = int(spec["target_material_id"])
        if target_id not in targets:
            raise ValueError(f"Target material {target_id} does not exist")
        template_path = resolve(spec["template_modelbin"]).resolve(strict=True)
        template_bundle = parse_bundle(template_path.read_bytes())
        template_materials = materials_by_id(template_bundle)
        template_id = int(spec["template_material_id"])
        if template_id not in template_materials:
            raise ValueError(f"Template material {template_id} does not exist in {template_path}")
        template = template_materials[template_id]
        shader, atst, nested, mtpr = shader_info(template.data)
        if shader != spec["expected_shader"] or atst != spec["expected_atst"]:
            raise ValueError(
                f"Template shader/ATST {shader}/{atst} != "
                f"{spec['expected_shader']}/{spec['expected_atst']}"
            )
        texture_patches = normalize_patches(spec, "texture_patches")
        value_patches = normalize_patches(spec, "value_patches")
        patched_mtpr, changes, template_texture_hashes = patch_mtpr(
            mtpr.data,
            texture_patches,
            value_patches,
            require_all_textures=bool(spec.get("require_all_template_textures_patched", True)),
        )
        payload = rebuild_with_blob_data(nested, {mtpr.index: patched_mtpr})
        parse_bundle(payload)
        replacements[targets[target_id].index] = payload
        report_materials.append(
            {
                "target_material_id": target_id,
                "target_name": material_name(targets[target_id]),
                "role": spec.get("role"),
                "template": {
                    "path": str(template_path),
                    "sha256": sha256_path(template_path),
                    "material_id": template_id,
                    "material_name": material_name(template),
                    "shader": shader,
                    "atst": atst,
                    "texture_hashes": [
                        f"{item:08x}" for item in sorted(template_texture_hashes)
                    ],
                },
                "changes": changes,
                "payload_sha256": sha256_bytes(payload),
            }
        )

    output_data = rebuild_with_blob_data(source_bundle, replacements)
    output_path.write_bytes(output_data)
    output_bundle = parse_bundle(output_data)
    parsed = inspector.inspect(output_path)
    if parsed["parsed"]["errors"]:
        raise ValueError(f"Output parser errors: {parsed['parsed']['errors']}")
    for original, rebuilt in zip(source_bundle.blobs, output_bundle.blobs):
        if original.index not in replacements and original.data != rebuilt.data:
            raise ValueError(f"Unexpected edit to blob {original.index}:{original.tag}")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "input": {
            "path": str(input_path),
            "bytes": len(source_data),
            "sha256": sha256_path(input_path),
        },
        "profile": {"path": str(profile_path), "sha256": sha256_path(profile_path)},
        "output": {
            "path": str(output_path),
            "bytes": len(output_data),
            "sha256": sha256_path(output_path),
        },
        "materials": report_materials,
        "validation": {
            "outer_bundle": True,
            "nested_material_bundles": True,
            "modelbin_parser": True,
            "all_template_texture_slots_explicitly_patched": True,
            "non_target_blob_payloads_preserved": True,
            "game_validated": False,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_MATERIAL_PROFILE=" + json.dumps(report["output"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
