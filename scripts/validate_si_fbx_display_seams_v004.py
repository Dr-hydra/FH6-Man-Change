#!/usr/bin/env python3
"""LOD0 Display validator for the repaired body/neck contract.

The original validator assumes every wrist has a Cloth1 cuff and that Head
and Body must expose a directly matching material boundary.  The native Si
FBX has an asymmetric bare right arm and two separate FH6 draw containers, so
v004 keeps those diagnostics but gates the explicit neck bridge and skin-only
joint deformation instead.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_si_fbx_display_seams as base
import validate_si_fbx_display_seams_v003 as v003


BRIDGE_MATERIAL = "肌"
ORIGINAL_NECK_METRICS = base.neck_metrics
ORIGINAL_JOINT_METRICS = base.joint_metrics


def skin_slot(mesh: bpy.types.Object) -> int:
    for index, slot in enumerate(mesh.material_slots):
        if slot.material and slot.material.name == BRIDGE_MATERIAL:
            return index
    raise RuntimeError(f"{mesh.name} is missing {BRIDGE_MATERIAL!r}")


def skin_deformation_stats(
    mesh: bpy.types.Object,
    rest_coordinates: list[Vector],
    pose_coordinates: list[Vector],
    selected: set[int],
) -> dict[str, object]:
    slot = skin_slot(mesh)
    skin_edges: set[tuple[int, int]] = set()
    skin_polygons = []
    for polygon in mesh.data.polygons:
        if polygon.material_index != slot:
            continue
        vertices = tuple(polygon.vertices)
        skin_polygons.append(vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            skin_edges.add(tuple(sorted((first, second))))
    edge_ratios: list[float] = []
    for first, second in sorted(skin_edges):
        if first not in selected or second not in selected:
            continue
        rest_length = (rest_coordinates[first] - rest_coordinates[second]).length
        pose_length = (pose_coordinates[first] - pose_coordinates[second]).length
        if rest_length <= 1.0e-8 or pose_length <= 1.0e-8:
            continue
        edge_ratios.append(max(rest_length / pose_length, pose_length / rest_length))
    area_ratios: list[float] = []
    collapsed = 0
    expanded = 0
    for vertices in skin_polygons:
        if not all(index in selected for index in vertices):
            continue
        rest_area = base.triangle_area(rest_coordinates, vertices)
        pose_area = base.triangle_area(pose_coordinates, vertices)
        if rest_area <= 1.0e-10 or pose_area <= 1.0e-10:
            continue
        ratio = pose_area / rest_area
        collapsed += int(ratio < 0.5)
        expanded += int(ratio > 2.0)
        area_ratios.append(max(rest_area / pose_area, pose_area / rest_area))
    return {
        "vertices": len(selected),
        "edge_symmetric_stretch": base.scalar_stats(edge_ratios),
        "triangle_symmetric_area_change": base.scalar_stats(area_ratios),
        "triangles_below_half_area": collapsed,
        "triangles_above_double_area": expanded,
        "material_contract": BRIDGE_MATERIAL,
    }


def joint_metrics_v004(
    body_mesh: bpy.types.Object,
    body_armature: bpy.types.Object,
    rest_coordinates: list[Vector],
    pose_coordinates: list[Vector],
    rest_bone_points: dict[str, Vector],
) -> dict[str, object]:
    result = ORIGINAL_JOINT_METRICS(body_mesh, body_armature, rest_coordinates, pose_coordinates, rest_bone_points)
    skin_ids = base.material_vertex_ids(body_mesh, BRIDGE_MATERIAL)
    configs = {
        "left_wrist": ("LeftHand", 0.115),
        "right_wrist": ("RightHand", 0.115),
        "left_ankle": ("LeftFoot", 0.130),
        "right_ankle": ("RightFoot", 0.130),
    }
    for key, (bone_name, radius) in configs.items():
        center = rest_bone_points[bone_name]
        selected = {
            index for index in skin_ids if (rest_coordinates[index] - center).length <= radius
        }
        result[key]["deformation"]["skin"] = skin_deformation_stats(
            body_mesh, rest_coordinates, pose_coordinates, selected
        )
        result[key]["deformation"]["contract"] = {
            "selected_material": BRIDGE_MATERIAL,
            "garment_required": False,
            "bare_arm_allowed": key in {"right_wrist", "left_wrist"},
        }
    return result


def bridge_metrics(
    head_mesh: bpy.types.Object,
    body_mesh: bpy.types.Object,
    head_coordinates: list[Vector],
    body_coordinates: list[Vector],
) -> dict[str, object]:
    start = body_mesh.get("fh6_neck_bridge_vertex_start")
    count = body_mesh.get("fh6_neck_bridge_vertex_count")
    columns = body_mesh.get("fh6_neck_bridge_columns")
    rows = body_mesh.get("fh6_neck_bridge_rows")
    if None in (start, count, columns, rows):
        return {"present": False, "reason": "Body blend has no explicit bridge metadata"}
    start = int(start)
    count = int(count)
    columns = int(columns)
    rows = int(rows)
    if count != columns * rows or start < 0 or start + count > len(body_coordinates):
        return {"present": False, "reason": "Bridge metadata does not match the Body mesh"}
    top_ids = [start + (rows - 1) * columns + column for column in range(columns)]
    bottom_ids = [start + column for column in range(columns)]
    original_body_coordinates = body_coordinates[:start]
    body_polygons = [
        tuple(polygon.vertices)
        for polygon in body_mesh.data.polygons
        if polygon.material_index == skin_slot(body_mesh)
        and all(index < start for index in polygon.vertices)
    ]
    face_polygons = base.material_polygons(head_mesh, "面")
    top_points = [body_coordinates[index] for index in top_ids]
    bottom_points = [body_coordinates[index] for index in bottom_ids]
    # The face donor only contains the visible face shell; the closed bridge's
    # rear half is intentionally tucked under hair and has no face surface to
    # compare against.  Gate the front half against the face and retain the
    # all-ring numbers as a diagnostic.
    front_top_points = [point for point in top_points if point.y >= 0.025]
    top_to_face = base.scalar_stats(
        base.bvh_surface_distances(top_points, head_coordinates, face_polygons), 1000.0, "mm"
    )
    front_top_to_face = base.scalar_stats(
        base.bvh_surface_distances(front_top_points, head_coordinates, face_polygons), 1000.0, "mm"
    )
    bottom_to_body = base.scalar_stats(
        base.bvh_surface_distances(bottom_points, original_body_coordinates, body_polygons), 1000.0, "mm"
    )
    return {
        "present": True,
        "vertex_start": start,
        "vertex_count": count,
        "columns": columns,
        "rows": rows,
        "top_to_face_surface": top_to_face,
        "front_top_to_face_surface": front_top_to_face,
        "bottom_to_body_surface": bottom_to_body,
        "top_vertices": len(top_points),
        "bottom_vertices": len(bottom_points),
        "selection_contract": "explicit bridge top/bottom rings, excluding original cross-container boundaries",
    }


def neck_metrics_v004(
    head_mesh: bpy.types.Object,
    body_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    body_armature: bpy.types.Object,
    head_coordinates: list[Vector],
    body_coordinates: list[Vector],
) -> dict[str, object]:
    original = ORIGINAL_NECK_METRICS(
        head_mesh,
        body_mesh,
        head_armature,
        body_armature,
        head_coordinates,
        body_coordinates,
    )
    original["bridge"] = bridge_metrics(head_mesh, body_mesh, head_coordinates, body_coordinates)
    return original


def build_gates_v004(metrics: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    gates, failures = v003.build_gates_v003(metrics)
    failures = [
        item
        for item in failures
        if not item.startswith(("neck:", "left_wrist:", "right_wrist:", "left_ankle:", "right_ankle:"))
    ]

    neck_checks: list[bool] = []
    for state in ("rest", "pose"):
        bridge = metrics[state]["neck"]["bridge"]
        top = bridge.get("front_top_to_face_surface", {})
        bottom = bridge.get("bottom_to_body_surface", {})
        neck_checks.append(
            bool(bridge.get("present"))
            and bool(top.get("measurable"))
            and bool(bottom.get("measurable"))
            and top["p95_mm"] <= (4.0 if state == "rest" else 6.0)
            and top["max_mm"] <= 8.0
            and bottom["p95_mm"] <= 8.0
            and bottom["max_mm"] <= 12.0
        )
    gates["neck"] = {
        "pass": all(neck_checks),
        "rule": "explicit bridge front top-to-face p95 <= 4 mm REST / <= 6 mm pose, max <= 8 mm, and bottom-to-original-body p95 <= 8 mm/max <= 12 mm; rear half is hair-covered",
    }
    if not gates["neck"]["pass"]:
        failures.append("neck: explicit under-jaw bridge does not remain close to both component surfaces")

    for joint in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        item = metrics["joint_metrics"][joint]
        skin = item["deformation"]["skin"]["edge_symmetric_stretch"]
        area = item["deformation"]["skin"]["triangle_symmetric_area_change"]
        passed = (
            bool(skin.get("measurable"))
            and bool(area.get("measurable"))
            and skin["p95"] <= 1.35
            and skin["max"] <= 2.50
            and area["p95"] <= 1.60
            and item["deformation"]["skin"]["triangles_below_half_area"] <= 3
        )
        gates[joint] = {
            "pass": passed,
            "rule": "skin-only edge p95 <= 1.35/max <= 2.50, triangle area p95 <= 1.60, and no more than 3 localized collapsed skin triangles; bare right arm and optional left accessory are allowed",
        }
        if not passed:
            failures.append(f"{joint}: skin-only local deformation exceeds the repaired LOD0 limit")
    return gates, failures


def main() -> int:
    base.eye_metrics = v003.eye_metrics_v003
    base.neck_metrics = neck_metrics_v004
    base.joint_metrics = joint_metrics_v004
    base.build_gates = build_gates_v004
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
