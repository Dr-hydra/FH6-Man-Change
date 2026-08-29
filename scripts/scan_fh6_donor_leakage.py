#!/usr/bin/env python3
"""Fail closed when FH6 MatI texture paths still reference donor responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from modelbin_bundle import parse_bundle
from fh6_material_codec import (
    decode_7bit,
    material_id,
    material_name,
    parameter_end,
)


DEFAULT_TOKENS = (
    "alice",
    "average_kim",
    "race_suit",
    "race_helmet",
    "branding",
    "logo",
    "stitch",
    "detailnormal",
    "micronormal",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--forbidden-token", action="append", default=[])
    parser.add_argument(
        "--require-generated-prefix",
        default=None,
        help="Require every texture basename to begin with this prefix (case-insensitive).",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shader_name(payload: bytes) -> str | None:
    nested = parse_bundle(payload)
    mati = next((blob for blob in nested.blobs if blob.tag == "MATI"), None)
    if mati is None:
        return None
    return next(
        (entry.value.decode("utf-8") for entry in mati.metadata if entry.tag == "Name"),
        None,
    )


def texture_parameters(payload: bytes) -> list[dict[str, Any]]:
    nested = parse_bundle(payload)
    mtpr = next((blob for blob in nested.blobs if blob.tag == "MTPR"), None)
    if mtpr is None:
        raise ValueError("MatI lacks MTPR")
    data = mtpr.data
    offset = 1
    textures = []
    for _ in range(data[0]):
        end, name_hash, parameter_type, value_offset = parameter_end(data, offset)
        if parameter_type == 6:
            length, path_offset = decode_7bit(data, value_offset)
            path = data[path_offset : path_offset + length].decode("utf-8")
            stored_hash = struct.unpack_from("<I", data, end - 4)[0]
            textures.append(
                {
                    "parameter_hash": f"{name_hash:08x}",
                    "path": path,
                    "stored_path_hash": f"{stored_hash:08x}",
                }
            )
        offset = end
    return textures


def scan_modelbin(
    data: bytes,
    label: str,
    tokens: tuple[str, ...],
    generated_prefix: str | None,
) -> dict[str, Any]:
    outer = parse_bundle(data)
    materials = []
    findings = []
    for blob in outer.blobs:
        if blob.tag != "MatI":
            continue
        shader = shader_name(blob.data)
        textures = texture_parameters(blob.data)
        material = {
            "id": material_id(blob),
            "name": material_name(blob),
            "shader": shader,
            "textures": textures,
        }
        materials.append(material)
        for texture in textures:
            lowered = texture["path"].lower()
            matched = [token for token in tokens if token in lowered]
            if matched:
                findings.append(
                    {
                        "kind": "forbidden_token",
                        "material_id": material["id"],
                        "material_name": material["name"],
                        "shader": shader,
                        "parameter_hash": texture["parameter_hash"],
                        "path": texture["path"],
                        "tokens": matched,
                    }
                )
            if generated_prefix is not None:
                basename = PurePosixPath(lowered.replace("\\", "/")).name
                if not basename.startswith(generated_prefix.lower()):
                    findings.append(
                        {
                            "kind": "not_generated_texture",
                            "material_id": material["id"],
                            "material_name": material["name"],
                            "shader": shader,
                            "parameter_hash": texture["parameter_hash"],
                            "path": texture["path"],
                            "required_prefix": generated_prefix,
                        }
                    )
    return {
        "label": label,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "materials": materials,
        "texture_count": sum(len(item["textures"]) for item in materials),
        "findings": findings,
    }


def scan_input(
    path: Path,
    tokens: tuple[str, ...],
    generated_prefix: str | None,
) -> list[dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() == ".modelbin":
        return [scan_modelbin(resolved.read_bytes(), str(resolved), tokens, generated_prefix)]
    if resolved.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported input type: {resolved}")
    results = []
    with zipfile.ZipFile(resolved) as zipped:
        entries = [entry for entry in zipped.infolist() if entry.filename.lower().endswith(".modelbin")]
        if not entries:
            raise ValueError(f"ZIP has no modelbin entries: {resolved}")
        for entry in entries:
            results.append(
                scan_modelbin(
                    zipped.read(entry),
                    f"{resolved}!/{entry.filename}",
                    tokens,
                    generated_prefix,
                )
            )
    return results


def main() -> int:
    args = arguments()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = tuple(dict.fromkeys(token.lower() for token in (*DEFAULT_TOKENS, *args.forbidden_token)))
    inputs = []
    for path in args.inputs:
        inputs.extend(scan_input(path, tokens, args.require_generated_prefix))
    findings = [finding for item in inputs for finding in item["findings"]]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "policy": {
            "forbidden_tokens": list(tokens),
            "require_generated_prefix": args.require_generated_prefix,
            "scope": "parsed MatI/MTPR Texture2D paths only",
        },
        "inputs": inputs,
        "summary": {
            "modelbins": len(inputs),
            "materials": sum(len(item["materials"]) for item in inputs),
            "textures": sum(item["texture_count"] for item in inputs),
            "finding_count": len(findings),
            "passed": not findings,
            "game_validated": False,
        },
        "findings": findings,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_DONOR_LEAKAGE=" + json.dumps(report["summary"], separators=(",", ":")))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
