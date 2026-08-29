#!/usr/bin/env python3
"""Merge head accessories and a tapered neck into the race-suit intermediate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from patch_fh6_garment_modelbin import read_intermediate


HEAD_Z_OFFSET = 0.020
HEAD_ACCESSORY_Z_OFFSET = -0.012
NECK_CENTER_Y = -0.013493


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("helmet_manifest", type=Path)
    parser.add_argument("head_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--vertices", required=True, type=Path)
    parser.add_argument("--bone-indices", required=True, type=Path)
    parser.add_argument("--indices", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoothstep(low: float, high: float, value: float) -> float:
    if high <= low:
        raise ValueError("Invalid smoothstep range")
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    return t * t * (3.0 - 2.0 * t)


def draw_map(manifest: dict) -> dict[int, dict]:
    draws = {int(draw["material_id"]): draw for draw in manifest["geometry"]["draws"]}
    if len(draws) != len(manifest["geometry"]["draws"]):
        raise ValueError("Intermediate contains duplicate draw IDs")
    return draws


def normalized_assignments(assignments: dict[int, float]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ordered = sorted(
        ((index, weight) for index, weight in assignments.items() if weight > 1e-8),
        key=lambda item: (-item[1], item[0]),
    )[:4]
    total = sum(weight for _, weight in ordered)
    if total <= 0.0:
        raise ValueError("Weight edit produced a zero-weight vertex")
    ordered = [(index, weight / total) for index, weight in ordered]
    ordered.extend([(0, 0.0)] * (4 - len(ordered)))
    return tuple(index for index, _ in ordered), tuple(weight for _, weight in ordered)


def rigid_vertex(vertex: dict, target_index: int) -> dict:
    return {**vertex, "bones": (target_index, 0, 0, 0), "weights": (1.0, 0.0, 0.0, 0.0)}


def neck_vertex(vertex: dict, bone_order: list[str], stats: Counter[str]) -> dict:
    x, y, z = (float(value) for value in vertex["position"])
    z += HEAD_Z_OFFSET
    fade = 1.0 - smoothstep(1.535, 1.570, z)
    radial_scale = 1.0 + 0.12 * fade
    position = (x * radial_scale, NECK_CENTER_Y + (y - NECK_CENTER_Y) * radial_scale, z)

    nx, ny, nz = (float(value) for value in vertex["normal"])
    normal_length = math.sqrt((nx / radial_scale) ** 2 + (ny / radial_scale) ** 2 + nz * nz)
    normal = (nx / radial_scale / normal_length, ny / radial_scale / normal_length, nz / normal_length)

    head_factor = smoothstep(1.535, 1.600, z)
    head_index = bone_order.index("Head")
    neck_index = bone_order.index("Neck1")
    bones, weights = normalized_assignments({head_index: head_factor, neck_index: 1.0 - head_factor})
    stats["neck vertices tapered"] += 1
    return {**vertex, "position": position, "normal": normal, "bones": bones, "weights": weights}


def source_draw_values(
    vertices: list[dict],
    indices: list[int],
    draw: dict,
) -> tuple[list[dict], list[int]]:
    vertex_start = int(draw["vertex_start"])
    vertex_count = int(draw["vertex_count"])
    index_start = int(draw["start_index"])
    index_count = int(draw["index_count"])
    selected_vertices = vertices[vertex_start : vertex_start + vertex_count]
    selected_indices = indices[index_start : index_start + index_count]
    if len(selected_vertices) != vertex_count or len(selected_indices) != index_count:
        raise ValueError("Source draw exceeds its intermediate buffers")
    local_indices = [int(value) - vertex_start for value in selected_indices]
    if min(local_indices, default=0) < 0 or max(local_indices, default=0) >= vertex_count:
        raise ValueError("Source draw index resolves outside its vertex domain")
    return selected_vertices, local_indices


def main() -> int:
    args = arguments()
    paths = {
        "base": args.base_manifest.resolve(strict=True),
        "helmet": args.helmet_manifest.resolve(strict=True),
        "head": args.head_manifest.resolve(strict=True),
    }
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

    manifests = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    data = {
        name: (*read_intermediate(paths[name], manifest), draw_map(manifest))
        for name, manifest in manifests.items()
    }
    bone_order = list(manifests["base"]["skinning"]["bone_order"])
    for required in ("Head", "Neck1"):
        if required not in bone_order:
            raise ValueError(f"Base skeleton is missing {required!r}")

    source_plan = {
        0: [("base", 0, "base"), ("helmet", 1, "head_accessory"), ("helmet", 2, "head_accessory"), ("helmet", 3, "head_accessory")],
        1: [("base", 1, "base"), ("head", 5, "neck")],
        2: [("base", 2, "base"), ("helmet", 4, "head_accessory")],
        3: [("base", 3, "base")],
        4: [("base", 4, "base")],
        5: [("base", 5, "base")],
        6: [("base", 6, "base")],
        7: [("base", 7, "base")],
    }

    stats: Counter[str] = Counter()
    output_vertices: list[dict] = []
    output_indices: list[int] = []
    output_draws: list[dict] = []
    for material_id in range(8):
        draw_vertex_start = len(output_vertices)
        draw_index_start = len(output_indices)
        materials: Counter[str] = Counter()
        provenance = []
        for source_name, source_id, policy in source_plan[material_id]:
            source_vertices, source_indices, source_draws = data[source_name]
            if source_id not in source_draws:
                raise ValueError(f"{source_name} intermediate is missing draw {source_id}")
            source_draw = source_draws[source_id]
            selected_vertices, local_indices = source_draw_values(source_vertices, source_indices, source_draw)
            source_target_start = len(output_vertices)
            for vertex in selected_vertices:
                if policy == "base":
                    output_vertices.append(dict(vertex))
                elif policy == "head_accessory":
                    shifted = dict(vertex)
                    x, y, z = (float(value) for value in shifted["position"])
                    shifted["position"] = (x, y, z + HEAD_ACCESSORY_Z_OFFSET)
                    output_vertices.append(rigid_vertex(shifted, bone_order.index("Head")))
                    stats["head accessory vertices shifted"] += 1
                elif policy == "neck":
                    output_vertices.append(neck_vertex(dict(vertex), bone_order, stats))
                else:
                    raise ValueError(f"Unknown source policy {policy!r}")
            output_indices.extend(source_target_start + local_index for local_index in local_indices)
            materials.update(source_draw.get("source_material_histogram", {}))
            provenance.append({"source": source_name, "draw": source_id, "vertices": len(selected_vertices), "indices": len(local_indices)})

        vertex_count = len(output_vertices) - draw_vertex_start
        index_count = len(output_indices) - draw_index_start
        if vertex_count <= 0 or index_count <= 0 or index_count % 3:
            raise ValueError(f"Output draw {material_id} is empty or not triangle-aligned")
        output_draws.append(
            {
                "material_id": material_id,
                "start_index": draw_index_start,
                "index_count": index_count,
                "triangles": index_count // 3,
                "vertex_start": draw_vertex_start,
                "vertex_count": vertex_count,
                "unique_vertex_count": vertex_count,
                "source_material_histogram": dict(sorted(materials.items())),
                "source_material_names": sorted(materials),
                "provenance": provenance,
            }
        )

    if len(output_vertices) > 65_535 or max(output_indices, default=0) >= len(output_vertices):
        raise ValueError("Combined outfit exceeds the R16_UINT vertex domain")

    vertex_payload = bytearray()
    bone_payload = bytearray()
    influence_histogram: Counter[int] = Counter()
    for vertex in output_vertices:
        values = (*vertex["position"], *vertex["normal"], *vertex["tangent"], *vertex["uv"], *vertex["weights"])
        vertex_payload.extend(struct.pack("<16f", *values))
        bone_payload.extend(struct.pack("<4H", *vertex["bones"]))
        influence_histogram[sum(weight > 0.0 for weight in vertex["weights"])] += 1
    index_payload = struct.pack(f"<{len(output_indices)}H", *output_indices)
    outputs["vertices"].write_bytes(vertex_payload)
    outputs["bone_indices"].write_bytes(bone_payload)
    outputs["indices"].write_bytes(index_payload)

    bounds_min = [min(float(vertex["position"][axis]) for vertex in output_vertices) for axis in range(3)]
    bounds_max = [max(float(vertex["position"][axis]) for vertex in output_vertices) for axis in range(3)]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Race-suit display intermediate with lowered head accessories, a tapered neck seam, and preserved limb skinning.",
        "source": {
            name: {"manifest": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "geometry": {
            "source_vertices": len(output_vertices),
            "export_vertices": len(output_vertices),
            "triangles": len(output_indices) // 3,
            "indices": len(output_indices),
            "maximum_index": max(output_indices, default=0),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "draw_policy": "racesuit8_lowered_head_accessories_neck_source_skin",
            "draws": output_draws,
        },
        "skinning": {
            "skeleton_bones": len(bone_order),
            "bone_order": bone_order,
            "max_influences": 4,
            "influence_histogram": dict(sorted(influence_histogram.items())),
            "geometry_adjustments": dict(sorted(stats.items())),
            "head_accessory_z_offset": HEAD_ACCESSORY_Z_OFFSET,
            "neck_policy": {
                "z_offset": HEAD_Z_OFFSET,
                "lower_radial_scale": 1.12,
                "fade_range_z": [1.535, 1.570],
                "binding": "smooth Neck1-to-Head",
            },
        },
        "files": {
            "vertices": {"path": str(outputs["vertices"]), "bytes": len(vertex_payload), "sha256": sha256_bytes(vertex_payload)},
            "bone_indices": {"path": str(outputs["bone_indices"]), "bytes": len(bone_payload), "sha256": sha256_bytes(bone_payload)},
            "indices": {"path": str(outputs["indices"]), "bytes": len(index_payload), "sha256": sha256_bytes(index_payload)},
        },
        "material_policy": {
            "0": "Cloth1 plus restored Cloth1 head accessories",
            "1": "body skin plus tapered head neck",
            "2": "Cloth2 plus restored Cloth2 head accessories",
            "3": "cuffs and outer sleeves with source skinning",
            "4": "upper sleeves",
            "5": "hands and forearms with source skinning",
            "6": "alpha cloth",
            "7": "shoes with source skinning",
        },
        "license_guard": "Local technical validation only; do not redistribute.",
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_OUTFIT_ATTACHMENTS_INTERMEDIATE="
        + json.dumps(
            {
                "manifest": str(outputs["manifest"]),
                "vertices": len(output_vertices),
                "indices": len(output_indices),
                "triangles": len(output_indices) // 3,
                "geometry_adjustments": dict(sorted(stats.items())),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
