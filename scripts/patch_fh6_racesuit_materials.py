#!/usr/bin/env python3
"""Replace race-suit MatI payloads with source-material texture bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

from modelbin_bundle import parse_bundle, rebuild_with_blob_data


DIFFUSE_PARAMETER_HASH = 0xEE34B08B
MATERIAL_PLAN = {
    0: (2, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_diff_a346c884-e0a3-4ffd-af3f-5e32ac65d8f2.swatchbin", "Cloth1"),
    1: (5, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_gloves_diff_ec74c04d-10a7-40d4-be29-16ddc8c0e638.swatchbin", "肌"),
    2: (2, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_diffuse_blue_25f544a6-e713-4a3d-9e65-a4447f541daf.swatchbin", "Cloth2"),
    3: (2, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_diff_a346c884-e0a3-4ffd-af3f-5e32ac65d8f2.swatchbin", "Cloth1"),
    4: (2, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_diff_a346c884-e0a3-4ffd-af3f-5e32ac65d8f2.swatchbin", "Cloth1"),
    5: (5, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_gloves_diff_ec74c04d-10a7-40d4-be29-16ddc8c0e638.swatchbin", "肌"),
    6: (2, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_gloves_diffuse_hwhite_ccaeb9fb-15dd-41f0-8320-7ad7a78521e8.swatchbin", "Cloth1Alpha"),
    7: (2, r"Game:\Media\_library\texturespg\characters\swatches\outfit_race_suit_modern_shoes_diff_35038698-cd09-4a0d-b340-ea966105ae09.swatchbin", "Cloth1"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_7bit(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(5):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("Invalid 7-bit string length")


def encode_7bit(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def parameter_end(data: bytes, offset: int) -> tuple[int, int, int, int]:
    start = offset
    major, minor = data[offset], data[offset + 1]
    name_hash = struct.unpack_from("<I", data, offset + 2)[0]
    offset += 6
    if major > 3 or (major == 3 and minor >= 1):
        has_extra = data[offset]
        offset += 1
        if has_extra:
            offset += 4
    parameter_type = data[offset]
    offset += 1
    if major >= 3:
        offset += 16
    value_offset = offset

    if parameter_type in (0, 1, 5, 9):
        offset += 16
    elif parameter_type in (2, 3, 4):
        offset += 4
    elif parameter_type == 6:
        length, offset = decode_7bit(data, offset)
        offset += length
        if major >= 2:
            offset += 4
    elif parameter_type == 7:
        offset += 8
        if major >= 1 and minor >= 1:
            offset += 4
    elif parameter_type == 8:
        count = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + count * 16
    elif parameter_type == 11:
        offset += 8
        if major == 1:
            offset += 8
    else:
        raise ValueError(f"Unsupported MTPR parameter type {parameter_type} at 0x{start:X}")
    if offset > len(data):
        raise ValueError("MTPR parameter extends beyond its payload")
    return offset, name_hash, parameter_type, value_offset


def patch_diffuse_parameter(data: bytes, path: str) -> tuple[bytes, str]:
    count = data[0]
    offset = 1
    records: list[bytes] = []
    replaced_path: str | None = None
    path_bytes = path.encode("utf-8")
    path_hash = zlib.crc32(path.lower().encode("utf-8")) & 0xFFFFFFFF
    for _ in range(count):
        start = offset
        offset, name_hash, parameter_type, value_offset = parameter_end(data, offset)
        record = data[start:offset]
        if name_hash == DIFFUSE_PARAMETER_HASH:
            if parameter_type != 6:
                raise ValueError("Diffuse parameter is not Texture2D")
            old_length, old_path_offset = decode_7bit(data, value_offset)
            replaced_path = data[old_path_offset : old_path_offset + old_length].decode("utf-8")
            record = (
                data[start:value_offset]
                + encode_7bit(len(path_bytes))
                + path_bytes
                + struct.pack("<I", path_hash)
            )
        records.append(record)
    if replaced_path is None:
        raise ValueError("Template MTPR has no diffuse Texture2D parameter")
    return bytes([count]) + b"".join(records) + data[offset:], replaced_path


def material_id(blob) -> int:
    metadata = next((entry for entry in blob.metadata if entry.tag == "Id  "), None)
    if metadata is None or len(metadata.value) != 4:
        raise ValueError(f"MatI blob {blob.index} has no four-byte material ID")
    return struct.unpack("<I", metadata.value)[0]


def material_name(blob) -> str:
    metadata = next((entry for entry in blob.metadata if entry.tag == "Name"), None)
    return metadata.value.decode("utf-8") if metadata else f"MatI_{blob.index}"


def patched_material_payload(template_payload: bytes, path: str) -> tuple[bytes, str]:
    nested = parse_bundle(template_payload)
    shader = next((blob for blob in nested.blobs if blob.tag == "MTPR"), None)
    if shader is None or shader.version != (2, 0):
        raise ValueError("Expected one nested MTPR 2.0 shader parameter blob")
    patched_shader, old_path = patch_diffuse_parameter(shader.data, path)
    return rebuild_with_blob_data(nested, {shader.index: patched_shader}), old_path


def main() -> None:
    args = arguments()
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source = source_path.read_bytes()
    bundle = parse_bundle(source)
    materials = {material_id(blob): blob for blob in bundle.blobs if blob.tag == "MatI"}
    if set(materials) != set(MATERIAL_PLAN):
        raise ValueError(f"Unexpected donor material IDs: {sorted(materials)}")

    replacements: dict[int, bytes] = {}
    report_materials = []
    for target_id, (template_id, texture_path, source_material) in MATERIAL_PLAN.items():
        target_blob = materials[target_id]
        template_blob = materials[template_id]
        payload, old_path = patched_material_payload(template_blob.data, texture_path)
        replacements[target_blob.index] = payload
        report_materials.append(
            {
                "material_id": target_id,
                "outer_blob_index": target_blob.index,
                "outer_name": material_name(target_blob),
                "template_material_id": template_id,
                "template_name": material_name(template_blob),
                "source_material": source_material,
                "old_template_diffuse": old_path,
                "new_diffuse": texture_path,
                "payload_bytes": len(payload),
                "payload_sha256": sha256_bytes(payload),
            }
        )

    output = rebuild_with_blob_data(bundle, replacements)
    parsed_output = parse_bundle(output)
    if len(parsed_output.blobs) != len(bundle.blobs):
        raise ValueError("Material patch changed the outer blob count")
    output_path.write_bytes(output)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "input": {"path": str(source_path), "bytes": len(source), "sha256": sha256_bytes(source)},
        "output": {"path": str(output_path), "bytes": len(output), "sha256": sha256_bytes(output)},
        "materials": report_materials,
        "policy": "Each racesuit8 draw uses one source material; texture-capable donor MatI templates are retained without branding overlays.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_RACESUIT_MATERIALS=" + json.dumps(report["output"], separators=(",", ":")))


if __name__ == "__main__":
    main()
