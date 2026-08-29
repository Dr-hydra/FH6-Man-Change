#!/usr/bin/env python3
"""Transplant complete FH6 MatI payloads by material ID without changing other blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle, rebuild_with_blob_data


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Modelbin supplying complete MatI payloads.")
    parser.add_argument("target", type=Path, help="Modelbin retaining all non-MatI data.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def material_id(blob) -> int:
    values = [entry.value for entry in blob.metadata if entry.tag == "Id  "]
    if len(values) != 1 or len(values[0]) != 4:
        raise ValueError(f"MatI blob {blob.index} has no unique four-byte Id metadata")
    return struct.unpack("<i", values[0])[0]


def materials_by_id(bundle) -> dict[int, object]:
    result: dict[int, object] = {}
    for blob in bundle.blobs:
        if blob.tag != "MatI":
            continue
        identifier = material_id(blob)
        if identifier in result:
            raise ValueError(f"Duplicate MatI ID {identifier}")
        result[identifier] = blob
    return result


def main() -> int:
    args = arguments()
    source_path = args.source.resolve(strict=True)
    target_path = args.target.resolve(strict=True)
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    source_data = source_path.read_bytes()
    target_data = target_path.read_bytes()
    source_bundle = parse_bundle(source_data)
    target_bundle = parse_bundle(target_data)
    source_materials = materials_by_id(source_bundle)
    target_materials = materials_by_id(target_bundle)
    if set(source_materials) != set(target_materials):
        raise ValueError(
            "Source and target MatI IDs differ: "
            f"source={sorted(source_materials)}, target={sorted(target_materials)}"
        )
    replacements = {
        target_blob.index: source_materials[identifier].data
        for identifier, target_blob in target_materials.items()
    }
    output_data = rebuild_with_blob_data(target_bundle, replacements)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_data)
    output_report = inspector.inspect(output_path)
    if output_report["parsed"]["errors"]:
        raise ValueError(f"Output parser errors: {output_report['parsed']['errors']}")
    output_bundle = parse_bundle(output_data)
    output_materials = materials_by_id(output_bundle)
    non_material_blobs_unchanged = all(
        source_blob.data == output_blob.data
        for source_blob, output_blob in zip(target_bundle.blobs, output_bundle.blobs)
        if source_blob.tag != "MatI"
    )
    if not non_material_blobs_unchanged:
        raise ValueError("A non-MatI payload changed during the transplant")
    if any(
        output_materials[identifier].data != source_materials[identifier].data
        for identifier in source_materials
    ):
        raise ValueError("One or more MatI payloads differ from the source")

    report = {
        "schema_version": 1,
        "created_local": datetime.now(timezone.utc).astimezone().isoformat(),
        "state": "offline-candidate",
        "purpose": "MatI-only payload transplant by numeric material ID.",
        "source": {"path": str(source_path), "sha256": sha256_path(source_path)},
        "target": {"path": str(target_path), "sha256": sha256_path(target_path)},
        "output": {
            "path": str(output_path),
            "sha256": sha256_path(output_path),
            "bytes": len(output_data),
        },
        "materials": [
            {
                "id": identifier,
                "source_payload_sha256": sha256_bytes(source_materials[identifier].data),
                "output_payload_sha256": sha256_bytes(output_materials[identifier].data),
                "target_metadata_name": next(
                    (
                        entry.value.decode("utf-8", errors="replace")
                        for entry in target_materials[identifier].metadata
                        if entry.tag == "Name"
                    ),
                    None,
                ),
            }
            for identifier in sorted(source_materials)
        ],
        "validation": {
            "parse_errors": output_report["parsed"]["errors"],
            "non_material_payloads_unchanged": non_material_blobs_unchanged,
            "all_source_payloads_applied": True,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_MATI_TRANSPLANT="
        + json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "materials": len(source_materials),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
