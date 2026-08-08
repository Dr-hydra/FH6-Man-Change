#!/usr/bin/env python3
"""Build an invisible six-draw intermediate for the Alice display slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--vertices", required=True, type=Path)
    parser.add_argument("--bone-indices", required=True, type=Path)
    parser.add_argument("--indices", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    args = arguments()
    source_path = args.source_manifest.resolve(strict=True)
    outputs = {
        "manifest": args.output_manifest.resolve(),
        "vertices": args.vertices.resolve(),
        "bone_indices": args.bone_indices.resolve(),
        "indices": args.indices.resolve(),
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    bone_order = list(source["skinning"]["bone_order"])
    if "Head" not in bone_order:
        raise ValueError("Alice skeleton is missing Head")
    head_index = bone_order.index("Head")

    vertices = bytearray()
    bones = bytearray()
    indices: list[int] = []
    draws = []
    position = (0.0, 0.0, 1.62)
    normal = (0.0, 0.0, 1.0)
    tangent = (1.0, 0.0, 0.0, 1.0)
    uv = (0.0, 0.0)
    weights = (1.0, 0.0, 0.0, 0.0)
    record = struct.pack("<16f", *(position + normal + tangent + uv + weights))
    bone_record = struct.pack("<4H", head_index, 0, 0, 0)
    for material_id in range(6):
        vertex_start = material_id * 3
        index_start = len(indices)
        for _ in range(3):
            vertices.extend(record)
            bones.extend(bone_record)
        indices.extend((vertex_start, vertex_start + 1, vertex_start + 2))
        draws.append(
            {
                "material_id": material_id,
                "start_index": index_start,
                "index_count": 3,
                "triangles": 1,
                "vertex_start": vertex_start,
                "vertex_count": 3,
                "unique_vertex_count": 3,
                "source_material_histogram": {"invisible_placeholder": 1},
                "source_material_names": ["invisible_placeholder"],
            }
        )
    index_payload = struct.pack(f"<{len(indices)}H", *indices)
    outputs["vertices"].write_bytes(vertices)
    outputs["bone_indices"].write_bytes(bones)
    outputs["indices"].write_bytes(index_payload)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Invisible Alice display placeholder while face geometry is hosted by the Sakura helmet.",
        "source": {"manifest": str(source_path)},
        "geometry": {
            "source_vertices": 18,
            "export_vertices": 18,
            "corner_split_vertices_added": 0,
            "triangles": 6,
            "indices": 18,
            "maximum_index": 17,
            "bounds_min": list(position),
            "bounds_max": list(position),
            "draw_policy": "driver_body6_placeholder",
            "draws": draws,
        },
        "skinning": {
            "bone_count": len(bone_order),
            "bone_order": bone_order,
            "max_influences": 1,
            "zero_weight_vertices": 0,
            "weight_sum_min": 1.0,
            "weight_sum_max": 1.0,
            "policy": "All zero-area placeholder vertices bind rigidly to Head.",
        },
        "files": {
            "vertices": {"path": str(outputs["vertices"]), "bytes": len(vertices), "sha256": sha256_bytes(vertices)},
            "bone_indices": {"path": str(outputs["bone_indices"]), "bytes": len(bones), "sha256": sha256_bytes(bones)},
            "indices": {"path": str(outputs["indices"]), "bytes": len(index_payload), "sha256": sha256_bytes(index_payload)},
        },
        "license_guard": "Local technical validation only; do not redistribute.",
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_ALICE_PLACEHOLDER_INTERMEDIATE="
        + json.dumps({"manifest": str(outputs["manifest"]), "vertices": 18, "indices": 18}, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
