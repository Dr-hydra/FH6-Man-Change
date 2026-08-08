#!/usr/bin/env python3
"""Run the LOD0 Display validator with topology-specific eye measurements.

The v001 validator used every face boundary vertex inside a radius and treated
the iris rim as the eye/socket seam. Once the required opaque sclera draw
exists, the meaningful seams are face socket to sclera and iris to sclera.
This wrapper leaves v001 untouched and replaces only its eye measurement and
eye gate implementations.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_si_fbx_display_seams as base


SCLERA_MATERIAL = "\u5de9\u819c"
ORIGINAL_BUILD_GATES = base.build_gates


def graph_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    remaining = set(adjacency)
    result: list[set[int]] = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            queue.extend(sorted(adjacency.get(current, set()) - component))
        remaining.difference_update(component)
        result.append(component)
    return sorted(result, key=lambda item: (-len(item), min(item)))


def boundary_components(mesh: bpy.types.Object, material: str) -> list[set[int]]:
    _boundary, adjacency = base.material_boundary_graph(mesh, material)
    return graph_components(adjacency)


def component_centroid(component: set[int], coordinates: list[Vector]) -> Vector:
    return base.midpoint(coordinates[index] for index in component)


def select_eye_component(
    components: list[set[int]],
    coordinates: list[Vector],
    center: Vector,
    side_sign: float,
    minimum_vertices: int,
) -> set[int]:
    candidates: list[tuple[tuple[float, float, int], set[int]]] = []
    for component in components:
        if len(component) < minimum_vertices:
            continue
        middle = component_centroid(component, coordinates)
        if middle.x * side_sign <= 0.0:
            continue
        minimum_distance = min((coordinates[index] - center).length for index in component)
        candidates.append(((minimum_distance, (middle - center).length, -len(component)), component))
    if not candidates:
        return set()
    return min(candidates, key=lambda item: item[0])[1]


def eye_metrics_v002(
    head_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    coordinates: list[Vector],
) -> dict[str, object]:
    iris_vertices = base.material_vertex_ids(head_mesh, "\u76ee")
    face_components = boundary_components(head_mesh, "\u9762")
    iris_components = boundary_components(head_mesh, "\u76ee")
    sclera_components = boundary_components(head_mesh, SCLERA_MATERIAL)
    sclera_polygons = base.material_polygons(head_mesh, SCLERA_MATERIAL)
    result: dict[str, object] = {}
    for side, bone_name, side_sign in (
        ("left", "LeftEye", -1.0),
        ("right", "RightEye", 1.0),
    ):
        center = base.pose_bone_point(head_armature, bone_name)
        iris_ids = {index for index in iris_vertices if (coordinates[index] - center).length <= 0.040}
        iris_boundary = select_eye_component(iris_components, coordinates, center, side_sign, 20)
        socket_boundary = select_eye_component(face_components, coordinates, center, side_sign, 12)
        sclera_boundary = select_eye_component(sclera_components, coordinates, center, side_sign, 20)
        iris_points = [coordinates[index] for index in sorted(iris_ids)]
        iris_ring_points = [coordinates[index] for index in sorted(iris_boundary)]
        socket_points = [coordinates[index] for index in sorted(socket_boundary)]
        sclera_ring_points = [coordinates[index] for index in sorted(sclera_boundary)]
        iris_center = base.midpoint(iris_points) if iris_points else Vector()
        socket_sclera_gap = base.symmetric_point_gap(socket_points, sclera_ring_points)
        result[side] = {
            "bone": bone_name,
            "bone_head_m": [round(value, 9) for value in center],
            "iris_vertices": len(iris_points),
            "iris_boundary_vertices": len(iris_boundary),
            "socket_boundary_vertices": len(socket_boundary),
            "sclera_boundary_vertices": len(sclera_boundary),
            "iris_centroid_m": [round(value, 9) for value in iris_center],
            "iris_centroid_to_bone_mm": round((iris_center - center).length * 1000.0, 6) if iris_points else None,
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
            # Compatibility field consumed by the original gate before this
            # wrapper replaces the final rule text and adds iris/sclera depth.
            "rim_to_socket_gap": socket_sclera_gap,
            "selection_contract": "nearest connected boundary component on the matching eye side",
        }
    return result


def build_gates_v002(metrics: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    gates, failures = ORIGINAL_BUILD_GATES(metrics)
    checks: list[bool] = []
    for state in ("rest", "pose"):
        for side in ("left", "right"):
            item = metrics[state]["eyes"][side]
            socket_gap = item["socket_to_sclera_gap"]["combined"]
            iris_gap = item["iris_to_sclera_surface"]
            checks.append(
                bool(socket_gap.get("measurable"))
                and socket_gap["p95_mm"] <= 6.0
                and socket_gap["max_mm"] <= 9.0
                and bool(iris_gap.get("measurable"))
                and iris_gap["p95_mm"] <= 2.5
                and item["iris_centroid_to_bone_mm"] <= 6.0
            )
    passed = all(checks)
    gates["eyes"] = {
        "pass": passed,
        "rule": "socket/sclera p95 <= 6 mm and max <= 9 mm; iris/sclera p95 <= 2.5 mm; iris centroid/eye bone <= 6 mm in REST and pose",
    }
    failures = [item for item in failures if not item.startswith("eyes:")]
    if not passed:
        failures.insert(0, "eyes: sclera/socket coverage, iris depth, or eye pivot exceeds the LOD0 limit")
    return gates, failures


def main() -> int:
    base.eye_metrics = eye_metrics_v002
    base.build_gates = build_gates_v002
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
