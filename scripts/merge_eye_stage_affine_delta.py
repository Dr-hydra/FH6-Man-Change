#!/usr/bin/env python3
"""Merge the v007 eye stage with the audited v006 affine REST displacement.

The v007 eye stage is derived from the v005 Head/Hair mesh and preserves the
original vertex order for the first 23,381 vertices.  v006 contains the same
mesh after the segment-affine REST warp.  This tool transfers that measured
displacement field to v007, including shape keys, and interpolates the field
for the 158 v007 sclera vertices that were appended by the eye stage.

All inputs are opened read-only and a new blend/report pair is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree


MESH_NAME = "Si_Display_HeadHair_LOD0"


def args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye-blend", required=True, type=Path)
    parser.add_argument("--base-blend", required=True, type=Path)
    parser.add_argument("--affine-blend", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--nearest", type=int, default=8)
    return parser.parse_args(argv)


def load_coords(path: Path) -> list[Vector]:
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve(strict=True)))
    obj = bpy.data.objects.get(MESH_NAME)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"{path} does not contain {MESH_NAME}")
    return [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]


def make_delta_field(base: list[Vector], affine: list[Vector]) -> tuple[list[Vector], KDTree]:
    if len(base) != len(affine):
        raise RuntimeError(f"Base/affine vertex counts differ: {len(base)} != {len(affine)}")
    deltas = [after - before for before, after in zip(base, affine)]
    tree = KDTree(len(base))
    for index, point in enumerate(base):
        tree.insert(point, index)
    tree.balance()
    return deltas, tree


def interpolated_delta(point: Vector, base: list[Vector], deltas: list[Vector], tree: KDTree, nearest: int) -> Vector:
    matches = tree.find_n(point, max(1, nearest))
    if not matches:
        return Vector()
    weighted = Vector()
    total = 0.0
    for candidate, index, distance in matches:
        if distance <= 1e-10:
            return deltas[index].copy()
        weight = 1.0 / (distance * distance)
        weighted += deltas[index] * weight
        total += weight
    return weighted / total if total else Vector()


def main() -> int:
    options = args()
    output = options.output_blend.resolve()
    report_path = options.report.resolve()
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    base = load_coords(options.base_blend)
    affine = load_coords(options.affine_blend)
    deltas, tree = make_delta_field(base, affine)

    bpy.ops.wm.open_mainfile(filepath=str(options.eye_blend.resolve(strict=True)))
    obj = bpy.data.objects.get(MESH_NAME)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"{options.eye_blend} does not contain {MESH_NAME}")
    if len(obj.data.vertices) < len(base):
        raise RuntimeError("Eye-stage mesh has fewer vertices than the affine baseline")

    inverse = obj.matrix_world.inverted()
    local_deltas: list[Vector] = []
    for index, vertex in enumerate(obj.data.vertices):
        world_point = obj.matrix_world @ vertex.co
        delta = deltas[index] if index < len(deltas) else interpolated_delta(world_point, base, deltas, tree, options.nearest)
        local_delta = inverse.to_3x3() @ delta
        vertex.co += local_delta
        local_deltas.append(local_delta)

    shape_key_count = 0
    if obj.data.shape_keys is not None:
        for block in obj.data.shape_keys.key_blocks:
            for index, point in enumerate(block.data):
                # The same vertex correspondence applies to all absolute shape
                # key coordinates; appended sclera vertices have no shape keys
                # and therefore receive the interpolated field.
                world_point = obj.matrix_world @ point.co
                delta = deltas[index] if index < len(deltas) else interpolated_delta(world_point, base, deltas, tree, options.nearest)
                point.co += inverse.to_3x3() @ delta
            shape_key_count += 1
    obj.data.update()

    lengths = sorted(delta.length * 1000.0 for delta in deltas)
    appended = max(0, len(obj.data.vertices) - len(base))
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Merge v007 continuous sclera/iris eye stage with v006 segment-affine REST field",
        "inputs": {
            "eye_blend": str(options.eye_blend.resolve()),
            "base_blend": str(options.base_blend.resolve()),
            "affine_blend": str(options.affine_blend.resolve()),
        },
        "output": {"blend": str(output), "object": MESH_NAME},
        "correspondence": {
            "base_vertices": len(base),
            "affine_vertices": len(affine),
            "eye_stage_vertices": len(obj.data.vertices),
            "appended_eye_stage_vertices": appended,
            "interpolation_neighbors": options.nearest,
            "vertex_order_preserved": True,
        },
        "affine_delta_mm": {
            "min": lengths[0] if lengths else 0.0,
            "p50": lengths[len(lengths) // 2] if lengths else 0.0,
            "p95": lengths[min(len(lengths) - 1, math.floor((len(lengths) - 1) * 0.95))] if lengths else 0.0,
            "max": lengths[-1] if lengths else 0.0,
        },
        "shape_keys_warped": shape_key_count,
        "constraints": {
            "source_geometry_authority": "FBX-derived v005/v006/v007 milestones",
            "game_directory_modified": False,
            "modelbin_written": False,
        },
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_EYE_AFFINE_MERGE=" + json.dumps(report["output"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
