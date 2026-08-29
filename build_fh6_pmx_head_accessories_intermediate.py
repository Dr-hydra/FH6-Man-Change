#!/usr/bin/env python3
"""Build the complete PMX-derived Si Display head intermediate.

The output keeps the working six head roles and restores the PMX Cloth1/Cloth2
head attachments as two additional draws.  Every draw is deliberately rigid to
Helmet-local ``Head``; this is a Display candidate, not a facial-animation or
Driver export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

from patch_fh6_garment_modelbin import read_intermediate


HEAD_DISPLAY_Z_OFFSET = 0.020
PMX_HELMET_Z_OFFSET = -0.012


DRAW_PLAN = (
    (0, "helmet", (0,), "hair"),
    (1, "face", (1,), "sclera"),
    (2, "face", (2,), "face"),
    (3, "face", (3,), "iris"),
    (4, "face", (4,), "eyelashes"),
    (5, "helmet", (5,), "hair_shadow"),
    (6, "helmet", (1, 2, 3), "cloth1_head_attachments"),
    (7, "helmet", (4,), "cloth2_horns_attachments"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("helmet_manifest", type=Path)
    parser.add_argument("face_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--vertices", required=True, type=Path)
    parser.add_argument("--bone-indices", required=True, type=Path)
    parser.add_argument("--indices", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draws_by_id(manifest: dict) -> dict[int, dict]:
    draws = {int(draw["material_id"]): draw for draw in manifest["geometry"]["draws"]}
    if len(draws) != len(manifest["geometry"]["draws"]):
        raise ValueError("Intermediate contains duplicate material draw IDs")
    return draws


def main() -> int:
    args = arguments()
    helmet_path = args.helmet_manifest.resolve(strict=True)
    face_path = args.face_manifest.resolve(strict=True)
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

    helmet_manifest = json.loads(helmet_path.read_text(encoding="utf-8"))
    face_manifest = json.loads(face_path.read_text(encoding="utf-8"))
    helmet_vertices, helmet_indices = read_intermediate(helmet_path, helmet_manifest)
    face_vertices, face_indices = read_intermediate(face_path, face_manifest)
    sources = {
        "helmet": (
            helmet_manifest,
            helmet_vertices,
            helmet_indices,
            draws_by_id(helmet_manifest),
        ),
        "face": (face_manifest, face_vertices, face_indices, draws_by_id(face_manifest)),
    }

    bone_order = list(helmet_manifest["skinning"]["bone_order"])
    if "Head" not in bone_order:
        raise ValueError("Helmet skeleton has no Head bone")
    head_index = bone_order.index("Head")

    output_vertices: list[dict] = []
    output_indices: list[int] = []
    output_draws: list[dict] = []
    for target_id, source_name, source_ids, role in DRAW_PLAN:
        _manifest, vertices, indices, draws = sources[source_name]
        target_vertex_start = len(output_vertices)
        target_index_start = len(output_indices)
        source_records = []
        role_triangles = 0

        for source_id in source_ids:
            if source_id not in draws:
                raise ValueError(f"{source_name} intermediate is missing draw {source_id}")
            source_draw = draws[source_id]
            source_vertex_start = int(source_draw["vertex_start"])
            source_vertex_count = int(source_draw["vertex_count"])
            source_index_start = int(source_draw["start_index"])
            source_index_count = int(source_draw["index_count"])
            if source_vertex_count <= 0 or source_index_count <= 0 or source_index_count % 3:
                raise ValueError(f"Invalid source range for {source_name} draw {source_id}")

            part_vertex_start = len(output_vertices)
            source_values = vertices[
                source_vertex_start : source_vertex_start + source_vertex_count
            ]
            if len(source_values) != source_vertex_count:
                raise ValueError(f"Source vertex range exceeds {source_name} intermediate")
            offset = HEAD_DISPLAY_Z_OFFSET if source_name == "face" else PMX_HELMET_Z_OFFSET
            for vertex in source_values:
                position = tuple(float(value) for value in vertex["position"])
                output_vertices.append(
                    {
                        **vertex,
                        "position": (position[0], position[1], position[2] + offset),
                        "bones": (head_index, 0, 0, 0),
                        "weights": (1.0, 0.0, 0.0, 0.0),
                    }
                )

            source_values_indices = indices[
                source_index_start : source_index_start + source_index_count
            ]
            if len(source_values_indices) != source_index_count:
                raise ValueError(f"Source index range exceeds {source_name} intermediate")
            for source_index in source_values_indices:
                local_index = int(source_index) - source_vertex_start
                if local_index < 0 or local_index >= source_vertex_count:
                    raise ValueError(
                        f"{source_name} draw {source_id} index resolves outside its vertex domain"
                    )
                output_indices.append(part_vertex_start + local_index)

            source_records.append(
                {
                    "source_draw": source_id,
                    "vertices": source_vertex_count,
                    "triangles": source_index_count // 3,
                }
            )
            role_triangles += source_index_count // 3

        target_vertex_count = len(output_vertices) - target_vertex_start
        target_index_count = len(output_indices) - target_index_start
        output_draws.append(
            {
                "material_id": target_id,
                "start_index": target_index_start,
                "index_count": target_index_count,
                "triangles": role_triangles,
                "vertex_start": target_vertex_start,
                "vertex_count": target_vertex_count,
                "unique_vertex_count": target_vertex_count,
                "source_material_histogram": {role: role_triangles},
                "source_material_names": [role],
                "source_intermediate": source_name,
                "source_draws": list(source_ids),
                "source_parts": source_records,
                "rigid_binding": "Head",
            }
        )

    if len(output_vertices) > 65_535:
        raise ValueError("Complete PMX head exceeds the R16_UINT vertex domain")
    if max(output_indices, default=0) >= len(output_vertices):
        raise ValueError("Complete PMX head index exceeds the vertex domain")

    vertex_payload = bytearray()
    bone_payload = bytearray()
    for vertex in output_vertices:
        values = (
            *vertex["position"],
            *vertex["normal"],
            *vertex["tangent"],
            *vertex["uv"],
            *vertex["weights"],
        )
        if len(values) != 16:
            raise ValueError("Intermediate vertex does not contain 16 floats")
        vertex_payload.extend(struct.pack("<16f", *values))
        bone_payload.extend(struct.pack("<4H", *vertex["bones"]))
    index_payload = struct.pack(f"<{len(output_indices)}H", *output_indices)
    outputs["vertices"].write_bytes(vertex_payload)
    outputs["bone_indices"].write_bytes(bone_payload)
    outputs["indices"].write_bytes(index_payload)

    bounds_min = [
        min(float(vertex["position"][axis]) for vertex in output_vertices)
        for axis in range(3)
    ]
    bounds_max = [
        max(float(vertex["position"][axis]) for vertex in output_vertices)
        for axis in range(3)
    ]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Complete PMX-derived Si Display head, eyes, hair, horns, and ornaments for the dedicated Sakura Helmet slot.",
        "source": {
            "helmet_manifest": str(helmet_path),
            "helmet_manifest_sha256": sha256_path(helmet_path),
            "face_manifest": str(face_path),
            "face_manifest_sha256": sha256_path(face_path),
            "face_z_offset": HEAD_DISPLAY_Z_OFFSET,
            "pmx_helmet_z_offset": PMX_HELMET_Z_OFFSET,
        },
        "geometry": {
            "source_vertices": len(output_vertices),
            "export_vertices": len(output_vertices),
            "corner_split_vertices_added": 0,
            "triangles": len(output_indices) // 3,
            "indices": len(output_indices),
            "maximum_index": max(output_indices, default=0),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "draw_policy": "pmx_head_accessories8",
            "draws": output_draws,
        },
        "skinning": {
            "bone_count": len(bone_order),
            "bone_order": bone_order,
            "max_influences": 1,
            "zero_weight_vertices": 0,
            "weight_sum_min": 1.0,
            "weight_sum_max": 1.0,
            "policy": "All eight dedicated Sakura Helmet draws bind rigidly to component-local Head.",
        },
        "files": {
            "vertices": {
                "path": str(outputs["vertices"]),
                "bytes": len(vertex_payload),
                "sha256": sha256_bytes(vertex_payload),
                "stride": 64,
            },
            "bone_indices": {
                "path": str(outputs["bone_indices"]),
                "bytes": len(bone_payload),
                "sha256": sha256_bytes(bone_payload),
                "stride": 8,
            },
            "indices": {
                "path": str(outputs["indices"]),
                "bytes": len(index_payload),
                "sha256": sha256_bytes(index_payload),
                "stride": 2,
            },
        },
        "material_policy": {
            "0": "hair",
            "1": "sclera",
            "2": "face",
            "3": "iris",
            "4": "eyelashes",
            "5": "hair_shadow",
            "6": "cloth1_head_attachments",
            "7": "cloth2_horns_attachments",
        },
        "license_guard": "Local technical validation only; do not redistribute.",
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "FH6_PMX_HEAD_ACCESSORIES_INTERMEDIATE="
        + json.dumps(
            {
                "manifest": str(outputs["manifest"]),
                "vertices": len(output_vertices),
                "indices": len(output_indices),
                "draws": len(output_draws),
                "restored_attachment_vertices": sum(
                    int(draw["vertex_count"]) for draw in output_draws if int(draw["material_id"]) in (6, 7)
                ),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
