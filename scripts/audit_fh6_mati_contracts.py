#!/usr/bin/env python3
"""Write a compact material-contract inventory for one or more FH6 modelbins."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelbin_bundle import parse_bundle
from patch_fh6_material_profile import shader_info
from fh6_material_codec import (
    decode_7bit,
    material_id,
    material_name,
    parameter_end,
)


KNOWN_NAMES = (
    "HairGradientBlack",
    "HairGradientMid",
    "HairGradientWhite",
    "PrimarySpecColour",
    "SecondarySpecColour",
    "PrimarySpecIntensity",
    "SecondarySpecIntensity",
    "PeachfuzzIntensity",
    "ScleraPrimaryColour",
    "ScleraSecondaryColour",
)
KNOWN_HASHES = {
    zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF: name for name in KNOWN_NAMES
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Modelbin path or LABEL=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_spec(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, value = raw.split("=", 1)
        if label and value:
            return label, Path(value)
    path = Path(raw)
    return path.stem, path


def decode_parameter(
    payload: bytes,
    name_hash: int,
    parameter_type: int,
    value_offset: int,
    end: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hash": f"{name_hash:08x}",
        "name": KNOWN_HASHES.get(name_hash),
        "type": parameter_type,
    }
    if parameter_type == 6:
        length, path_offset = decode_7bit(payload, value_offset)
        result["texture_path"] = payload[path_offset : path_offset + length].decode("utf-8")
    elif parameter_type in (0, 1, 5, 9):
        result["value"] = list(struct.unpack_from("<4f", payload, value_offset))
    elif parameter_type == 2:
        result["value"] = struct.unpack_from("<f", payload, value_offset)[0]
    elif parameter_type in (3, 4):
        result["value"] = struct.unpack_from("<I", payload, value_offset)[0]
    else:
        result["raw_value_hex"] = payload[value_offset:end].hex()
    return result


def inspect_modelbin(label: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes()
    outer = parse_bundle(data)
    materials = []
    for blob in outer.blobs:
        if blob.tag != "MatI":
            continue
        shader, atst, _nested, mtpr = shader_info(blob.data)
        payload = mtpr.data
        offset = 1
        parameters = []
        for _ in range(payload[0]):
            end, name_hash, parameter_type, value_offset = parameter_end(payload, offset)
            parameters.append(
                decode_parameter(payload, name_hash, parameter_type, value_offset, end)
            )
            offset = end
        materials.append(
            {
                "id": material_id(blob),
                "name": material_name(blob),
                "shader": shader,
                "atst": atst,
                "payload_sha256": sha256_bytes(blob.data),
                "parameter_count": int(payload[0]),
                "texture_count": sum("texture_path" in item for item in parameters),
                "parameters": parameters,
            }
        )
    return {
        "label": label,
        "path": str(resolved),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "materials": sorted(materials, key=lambda item: int(item["id"])),
    }


def main() -> int:
    args = arguments()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [inspect_modelbin(*input_spec(raw)) for raw in args.inputs]
    report = {
        "schema_version": 1,
        "created_local": datetime.now(timezone.utc).astimezone().isoformat(),
        "inputs": records,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_MATI_CONTRACT_AUDIT="
        + json.dumps(
            {"output": str(output), "modelbins": len(records)},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
