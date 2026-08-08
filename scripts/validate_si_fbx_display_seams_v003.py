#!/usr/bin/env python3
"""LOD0 Display validator with connected-component eye-layer gates.

Eye components are selected by distance to their posed eye bone, not by world
X sign, because a head turn can move both eyes to the same side of world X.
The gate validates socket/sclera coverage, iris depth relative to the sclera,
and the eye pivot while allowing a rigid eyeball to rotate inside the eyelid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_si_fbx_display_seams as base
import validate_si_fbx_display_seams_v002 as v002


def select_nearest_component(
    components: list[set[int]],
    coordinates: list[Vector],
    center: Vector,
    minimum_vertices: int,
) -> set[int]:
    candidates: list[tuple[tuple[float, float, int], set[int]]] = []
    for component in components:
        if len(component) < minimum_vertices:
            continue
        middle = v002.component_centroid(component, coordinates)
        minimum_distance = min((coordinates[index] - center).length for index in component)
        candidates.append(((minimum_distance, (middle - center).length, -len(component)), component))
    return min(candidates, key=lambda item: item[0])[1] if candidates else set()


def eye_metrics_v003(
    head_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    coordinates: list[Vector],
) -> dict[str, object]:
    iris_vertices = base.material_vertex_ids(head_mesh, "\u76ee")
    sclera_vertices = base.material_vertex_ids(head_mesh, v002.SCLERA_MATERIAL)
    face_components = v002.boundary_components(head_mesh, "\u9762")
    iris_components = v002.boundary_components(head_mesh, "\u76ee")
    sclera_components = v002.boundary_components(head_mesh, v002.SCLERA_MATERIAL)
    sclera_polygons = base.material_polygons(head_mesh, v002.SCLERA_MATERIAL)
    result: dict[str, object] = {}
    for side, bone_name in (("left", "LeftEye"), ("right", "RightEye")):
        center = base.pose_bone_point(head_armature, bone_name)
        iris_ids = {index for index in iris_vertices if (coordinates[index] - center).length <= 0.040}
        sclera_ids = {index for index in sclera_vertices if (coordinates[index] - center).length <= 0.050}
        iris_boundary = select_nearest_component(iris_components, coordinates, center, 20)
        socket_boundary = select_nearest_component(face_components, coordinates, center, 12)
        sclera_boundary = select_nearest_component(sclera_components, coordinates, center, 20)
        iris_points = [coordinates[index] for index in sorted(iris_ids)]
        sclera_points = [coordinates[index] for index in sorted(sclera_ids)]
        iris_ring_points = [coordinates[index] for index in sorted(iris_boundary)]
        socket_points = [coordinates[index] for index in sorted(socket_boundary)]
        sclera_ring_points = [coordinates[index] for index in sorted(sclera_boundary)]
        iris_center = base.midpoint(iris_points) if iris_points else Vector()
        sclera_center = base.midpoint(sclera_points) if sclera_points else Vector()
        socket_sclera_gap = base.symmetric_point_gap(socket_points, sclera_ring_points)
        result[side] = {
            "bone": bone_name,
            "bone_head_m": [round(value, 9) for value in center],
            "iris_vertices": len(iris_points),
            "sclera_vertices": len(sclera_points),
            "iris_boundary_vertices": len(iris_boundary),
            "socket_boundary_vertices": len(socket_boundary),
            "sclera_boundary_vertices": len(sclera_boundary),
            "iris_centroid_m": [round(value, 9) for value in iris_center],
            "sclera_centroid_m": [round(value, 9) for value in sclera_center],
            "iris_centroid_to_bone_mm": round((iris_center - center).length * 1000.0, 6) if iris_points else None,
            "iris_sclera_centroid_distance_mm": round((iris_center - sclera_center).length * 1000.0, 6)
            if iris_points and sclera_points
            else None,
            "iris_minus_sclera_world_y_mm": round((iris_center.y - sclera_center.y) * 1000.0, 6)
            if iris_points and sclera_points
            else None,
            "socket_to_sclera_gap": socket_sclera_gap,
            "iris_to_sclera_surface": base.scalar_stats(
                base.bvh_surface_distances(iris_points, coordinates, sclera_polygons),
                1000.0,
                "mm",
            ),
            "iris_boundary_to_sclera_surface": base.scalar_stats(
                base.bvh_surface_distances(iris_ring_points, coordinates, sclera_polygons),
                1000.0,
                "mm",
            ),
            "rim_to_socket_gap": socket_sclera_gap,
            "selection_contract": "nearest connected boundary component to the posed eye bone",
        }
    return result


def build_gates_v003(metrics: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    gates, failures = v002.ORIGINAL_BUILD_GATES(metrics)
    checks: list[bool] = []
    for state in ("rest", "pose"):
        socket_p95_limit = 1.5 if state == "rest" else 7.0
        for side in ("left", "right"):
            item = metrics[state]["eyes"][side]
            socket_gap = item["socket_to_sclera_gap"]["combined"]
            iris_gap = item["iris_to_sclera_surface"]
            centroid_gap = item["iris_sclera_centroid_distance_mm"]
            checks.append(
                bool(socket_gap.get("measurable"))
                and socket_gap["p95_mm"] <= socket_p95_limit
                and socket_gap["max_mm"] <= 9.0
                and bool(iris_gap.get("measurable"))
                and iris_gap["p95_mm"] <= 7.0
                and centroid_gap is not None
                and 0.5 <= centroid_gap <= 5.0
                and item["iris_centroid_to_bone_mm"] <= 6.0
            )
    rest_front_order = all(
        metrics["rest"]["eyes"][side]["iris_minus_sclera_world_y_mm"] >= 0.5
        for side in ("left", "right")
    )
    passed = all(checks) and rest_front_order
    gates["eyes"] = {
        "pass": passed,
        "rule": "connected socket/sclera p95 <= 1.5 mm REST and <= 7 mm pose; iris/sclera p95 <= 7 mm; centroid depth 0.5-5 mm; iris pivot <= 6 mm",
    }
    failures = [item for item in failures if not item.startswith("eyes:")]
    if not passed:
        failures.insert(0, "eyes: connected sclera coverage, iris depth/order, or eye pivot exceeds the LOD0 limit")
    return gates, failures


def main() -> int:
    base.eye_metrics = eye_metrics_v003
    base.build_gates = build_gates_v003
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
