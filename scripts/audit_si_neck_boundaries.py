#!/usr/bin/env python3
"""Audit ordered neck boundary loops without mutating the opened blend."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--z-min", type=float, default=1.35)
    parser.add_argument("--z-max", type=float, default=1.75)
    parser.add_argument("--max-abs-x", type=float, default=0.22)
    parser.add_argument("--merge-center-seams", action="store_true")
    return parser.parse_args(argv)


def material_slot(obj: bpy.types.Object, name: str) -> int:
    for index, slot in enumerate(obj.material_slots):
        if slot.material is not None and slot.material.name == name:
            return index
    raise RuntimeError(f"{obj.name} is missing material {name!r}")


def polygon_edges(polygons: list[bpy.types.MeshPolygon]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for polygon in polygons:
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            counts[tuple(sorted((first, second)))] += 1
    return counts


def connected_edge_components(edges: set[tuple[int, int]]) -> list[tuple[set[int], set[tuple[int, int]]]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    remaining = set(adjacency)
    components: list[tuple[set[int], set[tuple[int, int]]]] = []
    while remaining:
        root = min(remaining)
        vertices: set[int] = set()
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if current in vertices:
                continue
            vertices.add(current)
            queue.extend(adjacency[current] - vertices)
        component_edges = {edge for edge in edges if edge[0] in vertices and edge[1] in vertices}
        components.append((vertices, component_edges))
        remaining -= vertices
    return sorted(components, key=lambda item: (-len(item[0]), min(item[0])))


def ordered_vertices(vertices: set[int], edges: set[tuple[int, int]]) -> tuple[list[int], bool]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    if not vertices or any(len(adjacency[index]) > 2 for index in vertices):
        return sorted(vertices), False
    endpoints = sorted(index for index in vertices if len(adjacency[index]) == 1)
    closed = not endpoints and all(len(adjacency[index]) == 2 for index in vertices)
    start = endpoints[0] if endpoints else min(vertices)
    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = sorted(adjacency[current] - ({previous} if previous is not None else set()))
        if not candidates:
            break
        next_index = candidates[0]
        if closed and next_index == start:
            break
        if next_index in ordered:
            return sorted(vertices), False
        ordered.append(next_index)
        previous, current = current, next_index
        if len(ordered) == len(vertices):
            break
    return ordered if len(ordered) == len(vertices) else sorted(vertices), closed


def bounds(points: list[Vector]) -> dict[str, list[float]]:
    return {
        "min_m": [round(min(point[axis] for point in points), 9) for axis in range(3)],
        "max_m": [round(max(point[axis] for point in points), 9) for axis in range(3)],
        "centroid_m": [round(sum(point[axis] for point in points) / len(points), 9) for axis in range(3)],
    }


def vertex_weights(obj: bpy.types.Object, vertex_index: int, names: dict[int, str]) -> dict[str, float]:
    weights = {
        names[item.group]: float(item.weight)
        for item in obj.data.vertices[vertex_index].groups
        if item.group in names and item.weight > 1.0e-8
    }
    return dict(sorted(weights.items(), key=lambda item: (-item[1], item[0])))


def component_report(
    obj: bpy.types.Object,
    vertices: set[int],
    edges: set[tuple[int, int]],
    world_coordinates: list[Vector],
    world_normals: list[Vector],
    mesh_boundary_edges: set[tuple[int, int]],
    group_names: dict[int, str],
    args: argparse.Namespace,
) -> dict[str, object]:
    ordered, closed = ordered_vertices(vertices, edges)
    points = [world_coordinates[index] for index in ordered]
    component_bounds = bounds(points)
    intersects_neck_region = (
        component_bounds["min_m"][2] <= args.z_max
        and component_bounds["max_m"][2] >= args.z_min
        and min(abs(component_bounds["min_m"][0]), abs(component_bounds["max_m"][0])) <= args.max_abs_x
    )
    degree_counts: Counter[int] = Counter()
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    for index in vertices:
        degree_counts[len(adjacency[index])] += 1
    weight_totals: Counter[str] = Counter()
    for index in vertices:
        weight_totals.update(vertex_weights(obj, index, group_names))
    result: dict[str, object] = {
        "vertices": len(vertices),
        "edges": len(edges),
        "closed_cycle": closed,
        "degree_histogram": dict(sorted(degree_counts.items())),
        "mesh_open_edges": sum(edge in mesh_boundary_edges for edge in edges),
        "material_seam_edges": sum(edge not in mesh_boundary_edges for edge in edges),
        "bounds": component_bounds,
        "intersects_neck_region": intersects_neck_region,
        "mean_weights": {
            name: round(total / len(vertices), 8)
            for name, total in weight_totals.most_common(12)
        },
    }
    if intersects_neck_region:
        result["ordered_vertices"] = [
            {
                "index": index,
                "position_m": [round(float(value), 9) for value in world_coordinates[index]],
                "normal": [round(float(value), 9) for value in world_normals[index]],
                "weights": vertex_weights(obj, index, group_names),
            }
            for index in ordered
        ]
    return result


def merge_center_seams(obj: bpy.types.Object, slot: int, args: argparse.Namespace) -> dict[str, int]:
    material_polygons = [polygon for polygon in obj.data.polygons if polygon.material_index == slot]
    boundary_edges = {edge for edge, count in polygon_edges(material_polygons).items() if count == 1}
    coordinates = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    candidates: list[set[int]] = []
    for vertices, _edges in connected_edge_components(boundary_edges):
        points = [coordinates[index] for index in vertices]
        minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
        maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
        if (
            30 <= len(vertices) <= 50
            and 1.49 <= minimum.z <= 1.55
            and 1.62 <= maximum.z <= 1.68
            and maximum.x - minimum.x >= 0.045
            and maximum.x - minimum.x <= 0.065
            and maximum.y - minimum.y >= 0.070
        ):
            candidates.append(vertices)
    if len(candidates) != 2:
        raise RuntimeError(f"Expected two neck half-shell boundaries, found {len(candidates)}")
    boundary_vertices = set().union(*candidates)
    selected = {
        index
        for index in boundary_vertices
        if abs(coordinates[index].x) <= 1.0e-5
    }
    before = len(obj.data.vertices)
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.verts.ensure_lookup_table()
    bmesh.ops.remove_doubles(mesh, verts=[mesh.verts[index] for index in sorted(selected)], dist=2.0e-5)
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    return {
        "candidate_components": len(candidates),
        "candidate_vertices": sum(len(component) for component in candidates),
        "selected_vertices": len(selected),
        "merged_vertices": before - len(obj.data.vertices),
        "result_vertices": len(obj.data.vertices),
    }


def main() -> None:
    args = arguments()
    obj = bpy.data.objects.get(args.mesh)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"Opened blend is missing mesh {args.mesh!r}")
    slot = material_slot(obj, args.material)
    merge_report = merge_center_seams(obj, slot, args) if args.merge_center_seams else None
    all_polygons = list(obj.data.polygons)
    material_polygons = [polygon for polygon in all_polygons if polygon.material_index == slot]
    all_counts = polygon_edges(all_polygons)
    material_counts = polygon_edges(material_polygons)
    mesh_boundary_edges = {edge for edge, count in all_counts.items() if count == 1}
    material_boundary_edges = {edge for edge, count in material_counts.items() if count == 1}
    coordinates = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    normals = [(normal_matrix @ vertex.normal).normalized() for vertex in obj.data.vertices]
    group_names = {group.index: group.name for group in obj.vertex_groups}
    components = connected_edge_components(material_boundary_edges)
    report = {
        "blend": str(Path(bpy.data.filepath).resolve()),
        "mesh": obj.name,
        "material": args.material,
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "material_polygons": len(material_polygons),
        "mesh_boundary_edges": len(mesh_boundary_edges),
        "material_boundary_edges": len(material_boundary_edges),
        "center_seam_merge": merge_report,
        "region": {
            "z_min": args.z_min,
            "z_max": args.z_max,
            "max_abs_x": args.max_abs_x,
        },
        "components": [
            component_report(
                obj,
                vertices,
                edges,
                coordinates,
                normals,
                mesh_boundary_edges,
                group_names,
                args,
            )
            for vertices, edges in components
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "SI_NECK_BOUNDARY_AUDIT="
        + json.dumps(
            {
                "output": str(args.output.resolve()),
                "components": len(components),
                "neck_components": sum(item["intersects_neck_region"] for item in report["components"]),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
