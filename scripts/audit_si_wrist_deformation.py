#!/usr/bin/env python3
"""Print compact wrist weight and extreme-pose deformation diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="Si_Display_BodyGarment_LOD0")
    parser.add_argument("--armature", default="FH6_Outfit_Race_Suit_Modern_F_Skeleton")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def material_ids(obj: bpy.types.Object, material: str) -> set[int]:
    slots = {i for i, slot in enumerate(obj.material_slots) if slot.material and slot.material.name == material}
    return {v for poly in obj.data.polygons if poly.material_index in slots for v in poly.vertices}


def pose_coordinates(obj: bpy.types.Object, armature: bpy.types.Object, angle: float) -> list[Vector]:
    bone = armature.pose.bones.get("LeftHand")
    if bone is None:
        raise RuntimeError("Missing LeftHand pose bone")
    for item in armature.pose.bones:
        item.rotation_mode = "XYZ"
        item.rotation_euler = (0.0, 0.0, 0.0)
        item.location = (0.0, 0.0, 0.0)
        item.scale = (1.0, 1.0, 1.0)
    bone.rotation_euler[2] = math.radians(angle)
    bpy.context.view_layer.update()
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def main() -> None:
    args = arguments()
    obj = bpy.data.objects.get(args.mesh)
    armature = bpy.data.objects.get(args.armature)
    if obj is None or armature is None:
        raise RuntimeError("Missing requested mesh or armature")
    ids = material_ids(obj, "肌")
    rest = [obj.matrix_world @ v.co for v in obj.data.vertices]
    pose = pose_coordinates(obj, armature, 55.0)
    group_names = {group.index: group.name for group in obj.vertex_groups}
    hand_x = (armature.matrix_world @ armature.data.bones["LeftHand"].head_local).x
    bins: dict[int, list[dict[str, float]]] = defaultdict(list)
    for index in ids:
        point = rest[index]
        if point.x > -0.35 or point.z < 0.84 or point.z > 1.30:
            continue
        distance = abs(point.x - hand_x)
        bin_index = int(distance / 0.02)
        weights = {group_names[item.group]: float(item.weight) for item in obj.data.vertices[index].groups if item.weight > 1.0e-6}
        forearm = sum(value for name, value in weights.items() if name.startswith("LeftForeArm"))
        hand = sum(value for name, value in weights.items() if name.startswith(("LeftHand", "LeftIndex", "LeftMiddle", "LeftRing", "LeftPinky", "LeftThumb")))
        bins[bin_index].append({"forearm": forearm, "hand": hand, "x": point.x, "index": index})
    edges: list[dict[str, object]] = []
    for edge in obj.data.edges:
        a, b = edge.vertices
        if a not in ids or b not in ids:
            continue
        rest_length = (rest[a] - rest[b]).length
        pose_length = (pose[a] - pose[b]).length
        if rest_length <= 1.0e-8:
            continue
        ratio = max(pose_length / rest_length, rest_length / pose_length) if pose_length > 1.0e-8 else math.inf
        if ratio >= 1.4:
            edges.append({"vertices": [a, b], "ratio": ratio, "rest": [list(rest[a]), list(rest[b])], "pose": [list(pose[a]), list(pose[b])]})
    output = {
        "blend": str(Path(bpy.data.filepath).resolve()),
        "bins": {
            str(index): {
                "count": len(items),
                "mean_forearm": sum(item["forearm"] for item in items) / len(items),
                "mean_hand": sum(item["hand"] for item in items) / len(items),
                "min_x": min(item["x"] for item in items),
                "max_x": max(item["x"] for item in items),
            }
            for index, items in sorted(bins.items())
            if items
        },
        "high_stretch_edges": sorted(edges, key=lambda item: float(item["ratio"]), reverse=True)[:40],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"SI_WRIST_DEFORMATION_AUDIT={args.output.resolve()}")


if __name__ == "__main__":
    main()
