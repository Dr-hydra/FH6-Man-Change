#!/usr/bin/env python3
"""Emit per-column neck top-ring pose distances for a saved validation blend."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_si_fbx_display_seams as base


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def coords(obj: bpy.types.Object) -> list:
    return base.evaluated_coordinates(obj)


def main() -> None:
    args = arguments()
    head = bpy.data.objects[base.HEAD_MESH_NAME]
    body = bpy.data.objects[base.BODY_MESH_NAME]
    head_armature = bpy.data.objects[base.HEAD_ARMATURE_NAME]
    body_armature = bpy.data.objects[base.BODY_ARMATURE_NAME]
    row_attr = body.data.attributes["fh6_neck_bridge_row"]
    col_attr = body.data.attributes["fh6_neck_bridge_column"]
    top_ids = [index for index in range(len(row_attr.data)) if row_attr.data[index].value == 3]
    columns = {index: int(col_attr.data[index].value) for index in top_ids}
    face_polygons = base.material_polygons(head, "面")
    head_armature.data.pose_position = "REST"
    body_armature.data.pose_position = "REST"
    for armature in (head_armature, body_armature):
        for bone in armature.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
            bone.location = (0.0, 0.0, 0.0)
            bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    rest_head = coords(head)
    rest_body = coords(body)
    base.apply_diagnostic_pose((head_armature, body_armature))
    pose_head = coords(head)
    pose_body = coords(body)
    rest_tree = BVHTree.FromPolygons(rest_head, face_polygons, all_triangles=False)
    pose_tree = BVHTree.FromPolygons(pose_head, face_polygons, all_triangles=False)
    group_names = {group.index: group.name for group in body.vertex_groups}
    rows = []
    for index in sorted(top_ids, key=lambda item: columns[item]):
        rest_nearest = rest_tree.find_nearest(rest_body[index])
        pose_nearest = pose_tree.find_nearest(pose_body[index])
        weights = {
            group_names[item.group]: round(float(item.weight), 8)
            for item in body.data.vertices[index].groups
            if item.weight > 1.0e-8
        }
        rows.append(
            {
                "column": columns[index],
                "vertex": index,
                "weights": dict(sorted(weights.items(), key=lambda item: (-item[1], item[0]))),
                "rest_position_m": [round(float(value), 9) for value in rest_body[index]],
                "pose_position_m": [round(float(value), 9) for value in pose_body[index]],
                "rest_distance_mm": round(float(rest_nearest[3]) * 1000.0, 6) if rest_nearest else None,
                "pose_distance_mm": round(float(pose_nearest[3]) * 1000.0, 6) if pose_nearest else None,
            }
        )
    output = {
        "blend": str(Path(bpy.data.filepath).resolve()),
        "top_vertices": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SI_NECK_POSE_AUDIT=" + str(args.output.resolve()))


if __name__ == "__main__":
    main()
