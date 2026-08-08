#!/usr/bin/env python3
"""LOD0 validator for the native-boundary four-ring neck repair."""

from __future__ import annotations

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
import validate_si_fbx_display_seams_v004 as v004


BRIDGE_MATERIAL = "肌"
ORIGINAL_NECK_METRICS = v004.ORIGINAL_NECK_METRICS


def point_attribute(obj: bpy.types.Object, name: str):
    attribute = obj.data.attributes.get(name)
    if attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT":
        return None
    return attribute


def bridge_ids(obj: bpy.types.Object) -> dict[str, object]:
    row_attribute = point_attribute(obj, "fh6_neck_bridge_row")
    column_attribute = point_attribute(obj, "fh6_neck_bridge_column")
    bottom_attribute = point_attribute(obj, "fh6_neck_bridge_bottom")
    if row_attribute is None or column_attribute is None or bottom_attribute is None:
        return {"present": False, "reason": "native bridge point attributes are missing"}
    rows: dict[int, list[int]] = defaultdict(list)
    columns: dict[int, list[int]] = defaultdict(list)
    bottom = []
    for index in range(len(row_attribute.data)):
        row = int(row_attribute.data[index].value)
        column = int(column_attribute.data[index].value)
        if row >= 1:
            rows[row].append(index)
            columns[column].append(index)
        if int(bottom_attribute.data[index].value) == 1:
            bottom.append(index)
    return {
        "present": True,
        "rows": {row: sorted(indices) for row, indices in sorted(rows.items())},
        "columns": {column: sorted(indices) for column, indices in sorted(columns.items())},
        "bottom": sorted(bottom),
    }


def mesh_edge_counts(obj: bpy.types.Object, material_index: int | None = None) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for polygon in obj.data.polygons:
        if material_index is not None and polygon.material_index != material_index:
            continue
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            counts[tuple(sorted((first, second)))] += 1
    return counts


def evaluated_vertex_normals(obj: bpy.types.Object) -> dict[int, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    sums: dict[int, Vector] = defaultdict(Vector)
    counts: Counter[int] = Counter()
    try:
        for loop in mesh.loops:
            sums[loop.vertex_index] += Vector(mesh.corner_normals[loop.index].vector)
            counts[loop.vertex_index] += 1
        normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
        return {
            index: (normal_matrix @ (value / max(counts[index], 1))).normalized()
            for index, value in sums.items()
            if value.length > 1.0e-8
        }
    finally:
        evaluated.to_mesh_clear()


def interpolated_triangle_normal(
    point: Vector,
    triangle: list[int] | tuple[int, int, int],
    coordinates: list[Vector],
    normals: dict[int, Vector],
) -> Vector | None:
    if len(triangle) != 3 or any(index not in normals for index in triangle):
        return None
    first, second, third = (coordinates[index] for index in triangle)
    edge_a = second - first
    edge_b = third - first
    offset = point - first
    aa = edge_a.dot(edge_a)
    ab = edge_a.dot(edge_b)
    bb = edge_b.dot(edge_b)
    oa = offset.dot(edge_a)
    ob = offset.dot(edge_b)
    denominator = aa * bb - ab * ab
    if abs(denominator) <= 1.0e-16:
        return normals[triangle[0]].copy()
    second_weight = (bb * oa - ab * ob) / denominator
    third_weight = (aa * ob - ab * oa) / denominator
    first_weight = 1.0 - second_weight - third_weight
    weights = [max(0.0, min(1.0, value)) for value in (first_weight, second_weight, third_weight)]
    weight_sum = sum(weights)
    if weight_sum <= 1.0e-12:
        return normals[triangle[0]].copy()
    result = sum(
        (normals[index] * (weight / weight_sum) for index, weight in zip(triangle, weights)),
        Vector(),
    )
    return result.normalized() if result.length > 1.0e-8 else None


def scalar_summary(values: list[float], scale: float = 1.0, suffix: str = "") -> dict[str, object]:
    result = base.scalar_stats(values, scale, suffix)
    if values:
        scaled = sorted(value * scale for value in values)
        position = (len(scaled) - 1) * 0.05
        low = int(position)
        high = min(low + 1, len(scaled) - 1)
        factor = position - low
        result["p05"] = round(scaled[low] * (1.0 - factor) + scaled[high] * factor, 6)
    return result


def bridge_metrics_v005(
    head_mesh: bpy.types.Object,
    body_mesh: bpy.types.Object,
    head_coordinates: list[Vector],
    body_coordinates: list[Vector],
) -> dict[str, object]:
    original = ORIGINAL_NECK_METRICS(
        head_mesh,
        body_mesh,
        bpy.data.objects.get("FH6_Helmet_Race_Modern_Skeleton"),
        bpy.data.objects.get("FH6_Outfit_Race_Suit_Modern_F_Skeleton"),
        head_coordinates,
        body_coordinates,
    )
    ids = bridge_ids(body_mesh)
    if not ids.get("present"):
        original["bridge"] = ids
        return original
    rows: dict[int, list[int]] = ids["rows"]
    bottom_ids: list[int] = ids["bottom"]
    row_counts = {str(row): len(indices) for row, indices in sorted(rows.items())}
    topology: dict[str, object] = {
        "row_counts": row_counts,
        "bottom_vertices": len(bottom_ids),
        "total_new_vertices": sum(len(indices) for indices in rows.values()),
        "expected_rows": 3,
        "expected_columns": 48,
    }
    skin_slot = next(
        index
        for index, slot in enumerate(body_mesh.material_slots)
        if slot.material is not None and slot.material.name == BRIDGE_MATERIAL
    )
    all_edges = mesh_edge_counts(body_mesh)
    skin_edges = mesh_edge_counts(body_mesh, skin_slot)
    bottom_edge_ids = {
        edge
        for edge in skin_edges
        if edge[0] in bottom_ids and edge[1] in bottom_ids
    }
    topology["bottom_edges"] = len(bottom_edge_ids)
    topology["bottom_edges_shared_with_native_body"] = sum(all_edges[edge] >= 2 for edge in bottom_edge_ids)
    topology["bottom_boundary_edges_remaining"] = sum(all_edges[edge] == 1 for edge in bottom_edge_ids)
    bridge_faces = [
        polygon
        for polygon in body_mesh.data.polygons
        if polygon.material_index == skin_slot
        and any(int(body_mesh.data.attributes["fh6_neck_bridge_row"].data[index].value) >= 1 for index in polygon.vertices)
    ]
    topology["bridge_faces"] = len(bridge_faces)
    topology["bridge_faces_all_triangles"] = all(len(polygon.vertices) == 3 for polygon in bridge_faces)

    top_by_column = {
        int(body_mesh.data.attributes["fh6_neck_bridge_column"].data[index].value): index
        for index in rows.get(max(rows), [])
    }
    top_ids = [top_by_column[column] for column in sorted(top_by_column)]
    front_top_ids = [index for column, index in sorted(top_by_column.items()) if column < 32]
    top_points = [body_coordinates[index] for index in top_ids]
    front_top_points = [body_coordinates[index] for index in front_top_ids]
    face_polygons = base.material_polygons(head_mesh, "面")
    top_to_face = base.scalar_stats(
        base.bvh_surface_distances(top_points, head_coordinates, face_polygons), 1000.0, "mm"
    )
    front_top_to_face = base.scalar_stats(
        base.bvh_surface_distances(front_top_points, head_coordinates, face_polygons), 1000.0, "mm"
    )
    bottom_to_body = base.scalar_stats([0.0 for _ in bottom_ids], 1000.0, "mm")

    body_normals = evaluated_vertex_normals(body_mesh)
    head_normals = evaluated_vertex_normals(head_mesh)
    face_tree = BVHTree.FromPolygons(head_coordinates, face_polygons, all_triangles=False)
    normal_dots: list[float] = []
    for index in front_top_ids:
        point = body_coordinates[index]
        nearest = face_tree.find_nearest(point)
        if nearest is None or index not in body_normals:
            continue
        face_normal = interpolated_triangle_normal(
            nearest[0], face_polygons[nearest[2]], head_coordinates, head_normals
        )
        if face_normal is not None:
            normal_dots.append(float(body_normals[index].dot(face_normal)))
    normal_stats = scalar_summary(normal_dots)
    top_weight_sums: list[float] = []
    top_head_weights: list[float] = []
    group_names = {group.index: group.name for group in body_mesh.vertex_groups}
    for index in top_ids:
        weights = {group_names[item.group]: float(item.weight) for item in body_mesh.data.vertices[index].groups}
        top_weight_sums.append(sum(weights.values()))
        top_head_weights.append(sum(value for name, value in weights.items() if name in {"Head", "Jaw"} or name.endswith("_Jaw")))
    weight_stats = {
        "sum": scalar_summary(top_weight_sums),
        "head_jaw_fraction": scalar_summary(top_head_weights),
    }
    original["bridge"] = {
        "present": True,
        "vertex_start": body_mesh.get("fh6_neck_bridge_vertex_start"),
        "vertex_count": body_mesh.get("fh6_neck_bridge_vertex_count"),
        "columns": body_mesh.get("fh6_neck_bridge_columns"),
        "rows": body_mesh.get("fh6_neck_bridge_rows"),
        "top_vertices": len(top_ids),
        "bottom_vertices": len(bottom_ids),
        "top_to_face_surface": top_to_face,
        "front_top_to_face_surface": front_top_to_face,
        "bottom_to_body_surface": bottom_to_body,
        "top_normal_dot_face": normal_stats,
        "top_weight_contract": weight_stats,
        "topology": topology,
        "selection_contract": "native 36-point Body opening + 3 new 48-point rings + measured jaw arc",
    }
    return original


def neck_metrics_v005(
    head_mesh: bpy.types.Object,
    body_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    body_armature: bpy.types.Object,
    head_coordinates: list[Vector],
    body_coordinates: list[Vector],
) -> dict[str, object]:
    result = ORIGINAL_NECK_METRICS(
        head_mesh, body_mesh, head_armature, body_armature, head_coordinates, body_coordinates
    )
    bridge = bridge_metrics_v005(head_mesh, body_mesh, head_coordinates, body_coordinates)
    result["bridge"] = bridge["bridge"]
    return result


def build_gates_v005(metrics: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    gates, failures = v004.build_gates_v004(metrics)
    failures = [item for item in failures if not item.startswith("neck:")]
    checks: list[bool] = []
    topology_checks: list[bool] = []
    for state in ("rest", "pose"):
        bridge = metrics[state]["neck"]["bridge"]
        top = bridge.get("front_top_to_face_surface", {})
        normals = bridge.get("top_normal_dot_face", {})
        topology = bridge.get("topology", {})
        top_rows = bridge.get("rows") == 4
        top_count = bridge.get("top_vertices") == 48
        bottom_count = bridge.get("bottom_vertices", 0) >= 24
        surface_pass = (
            bool(top.get("measurable"))
            and top["p95_mm"] <= 2.0
            and top["max_mm"] <= 5.0
            and bool(normals.get("measurable"))
            and normals["p05"] >= 0.80
        )
        topology_pass = (
            top_rows
            and top_count
            and bottom_count
            and topology.get("row_counts") == {"1": 48, "2": 48, "3": 48}
            and topology.get("bottom_edges_shared_with_native_body") == topology.get("bottom_edges")
            and topology.get("bottom_boundary_edges_remaining") == 0
            and topology.get("bridge_faces_all_triangles") is True
        )
        checks.append(surface_pass)
        topology_checks.append(topology_pass)
    gates["neck"] = {
        "pass": all(checks) and all(topology_checks),
        "rule": "native bottom ring fully shared, exactly 4 total rings, 48 vertices per new ring, front top p95 <= 2 mm/max <= 5 mm, evaluated top-to-face shading-normal p05 >= 0.80 in REST and pose",
    }
    if not gates["neck"]["pass"]:
        failures.append("neck: native four-ring bridge topology, face distance, or normal continuity failed")
    return gates, failures


def main() -> int:
    base.eye_metrics = v003.eye_metrics_v003
    base.neck_metrics = neck_metrics_v005
    base.joint_metrics = v004.joint_metrics_v004
    base.build_gates = build_gates_v005
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
