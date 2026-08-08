#!/usr/bin/env python3
"""Build a reproducible LOD0 Display seam and pose validation scene.

The script opens the Head/Hair retarget as its immutable input scene, appends
only the Body/Garment export mesh and donor armature, and writes a separate
validation milestone.  It does not mutate either source blend or any game
file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


HEAD_MESH_NAME = "Si_Display_HeadHair_LOD0"
HEAD_ARMATURE_NAME = "FH6_Helmet_Race_Modern_Skeleton"
BODY_MESH_NAME = "Si_Display_BodyGarment_LOD0"
BODY_ARMATURE_NAME = "FH6_Outfit_Race_Suit_Modern_F_Skeleton"

POSE_DEGREES = {
    "Neck": (0.0, 0.0, 12.0),
    "Head": (0.0, 0.0, 28.0),
    "LeftEye": (8.0, 0.0, 10.0),
    "RightEye": (8.0, 0.0, 10.0),
    "LeftHand": (0.0, 0.0, 55.0),
    "RightHand": (0.0, 0.0, -55.0),
    "LeftFoot": (28.0, 0.0, 0.0),
    "RightFoot": (28.0, 0.0, 0.0),
}

REGION_FILES = (
    "face-front",
    "face-side",
    "eyes",
    "neck",
    "left-wrist",
    "right-wrist",
    "left-ankle",
    "right-ankle",
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-blend", required=True, type=Path)
    parser.add_argument("--body-blend", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--save-blend", required=True, type=Path)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_outputs_absent(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def append_body_package(body_blend: Path) -> tuple[bpy.types.Object, bpy.types.Object]:
    requested = {BODY_MESH_NAME, BODY_ARMATURE_NAME}
    with bpy.data.libraries.load(str(body_blend), link=False) as (source, target):
        missing = requested - set(source.objects)
        if missing:
            raise RuntimeError(f"Body blend is missing objects: {sorted(missing)}")
        target.objects = [name for name in source.objects if name in requested]
    appended = {obj.name: obj for obj in target.objects if obj is not None}
    body_mesh = appended.get(BODY_MESH_NAME)
    body_armature = appended.get(BODY_ARMATURE_NAME)
    if body_mesh is None or body_armature is None:
        raise RuntimeError("Failed to append the Body/Garment validation package")
    scene = bpy.context.scene
    for obj in (body_armature, body_mesh):
        if not obj.users_collection:
            scene.collection.objects.link(obj)
    modifiers = [modifier for modifier in body_mesh.modifiers if modifier.type == "ARMATURE"]
    if len(modifiers) != 1:
        raise RuntimeError(f"Expected one body armature modifier, found {len(modifiers)}")
    modifiers[0].object = body_armature
    return body_mesh, body_armature


def reset_pose(armature: bpy.types.Object) -> None:
    armature.data.pose_position = "POSE"
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.location = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)


def apply_diagnostic_pose(armatures: Iterable[bpy.types.Object]) -> None:
    for armature in armatures:
        reset_pose(armature)
        for name, degrees in POSE_DEGREES.items():
            bone = armature.pose.bones.get(name)
            if bone is None:
                raise RuntimeError(f"{armature.name} is missing diagnostic pose bone {name!r}")
            bone.rotation_euler = tuple(math.radians(value) for value in degrees)
    bpy.context.view_layer.update()


def pose_bone_point(armature: bpy.types.Object, name: str, point: str = "head") -> Vector:
    bone = armature.pose.bones.get(name)
    if bone is None:
        raise RuntimeError(f"Missing pose bone {name!r} on {armature.name}")
    return armature.matrix_world @ getattr(bone, point)


def midpoint(points: Iterable[Vector]) -> Vector:
    points = list(points)
    if not points:
        raise ValueError("Cannot find the midpoint of an empty point list")
    return sum(points, Vector()) / len(points)


def configure_render(scene: bpy.types.Scene, meshes: tuple[bpy.types.Object, bpy.types.Object]) -> bpy.types.Object:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    palette = {
        "发": (0.055, 0.070, 0.085, 1.0),
        "发影": (0.18, 0.22, 0.27, 1.0),
        "睫眉": (0.07, 0.045, 0.035, 1.0),
        "目影": (0.90, 0.92, 0.94, 1.0),
        "面": (0.64, 0.36, 0.27, 1.0),
        "目": (0.10, 0.52, 0.72, 1.0),
        "肌": (0.78, 0.50, 0.37, 1.0),
        "Cloth1": (0.42, 0.085, 0.075, 1.0),
        "Cloth1Alpha": (0.16, 0.42, 0.30, 1.0),
    }
    for mesh in meshes:
        for slot in mesh.material_slots:
            if slot.material is not None and slot.material.name in palette:
                slot.material.diffuse_color = palette[slot.material.name]

    camera_data = bpy.data.cameras.new("Si Display Seam Validation Camera")
    camera_data.lens = 70.0
    camera = bpy.data.objects.new("Si Display Seam Validation Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    allowed = {*meshes, camera}
    for obj in scene.objects:
        obj.hide_render = obj not in allowed
    for mesh in meshes:
        mesh.hide_render = False
        mesh.hide_viewport = False
    return camera


def aim(camera: bpy.types.Object, target: Vector, direction: Vector, distance: float) -> None:
    camera.location = target + direction.normalized() * distance
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render_region(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    target: Vector,
    direction: Vector,
    distance: float,
    output: Path,
) -> None:
    aim(camera, target, direction, distance)
    scene.render.filepath = str(output)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def render_suite(
    state: str,
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    head_armature: bpy.types.Object,
    body_armature: bpy.types.Object,
    outputs: dict[str, Path],
) -> None:
    eye_target = midpoint(
        (
            pose_bone_point(head_armature, "LeftEye"),
            pose_bone_point(head_armature, "RightEye"),
        )
    )
    head_target = pose_bone_point(head_armature, "Head") + Vector((0.0, 0.035, 0.055))
    neck_target = midpoint(
        (
            pose_bone_point(head_armature, "Neck", "tail"),
            pose_bone_point(body_armature, "Neck", "tail"),
            pose_bone_point(head_armature, "Head"),
            pose_bone_point(body_armature, "Head"),
        )
    )
    views = {
        "face-front": (head_target, Vector((0.0, 1.0, 0.0)), 0.55),
        "face-side": (head_target, Vector((1.0, 0.0, 0.0)), 0.55),
        "eyes": (eye_target, Vector((0.0, 1.0, 0.0)), 0.22),
        "neck": (neck_target, Vector((0.0, 1.0, 0.0)), 0.30),
        "left-wrist": (pose_bone_point(body_armature, "LeftHand"), Vector((0.0, 1.0, 0.0)), 0.43),
        "right-wrist": (pose_bone_point(body_armature, "RightHand"), Vector((0.0, 1.0, 0.0)), 0.43),
        "left-ankle": (pose_bone_point(body_armature, "LeftFoot"), Vector((-1.0, 0.0, 0.0)), 0.42),
        "right-ankle": (pose_bone_point(body_armature, "RightFoot"), Vector((1.0, 0.0, 0.0)), 0.42),
    }
    for region, (target, direction, distance) in views.items():
        render_region(scene, camera, target, direction, distance, outputs[f"{state}-{region}"])


def evaluated_coordinates(obj: bpy.types.Object) -> list[Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world
        return [matrix @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def material_index(obj: bpy.types.Object, name: str) -> int:
    for index, slot in enumerate(obj.material_slots):
        if slot.material is not None and slot.material.name == name:
            return index
    raise RuntimeError(f"{obj.name} is missing material {name!r}")


def material_vertex_ids(obj: bpy.types.Object, name: str) -> set[int]:
    index = material_index(obj, name)
    return {vertex for polygon in obj.data.polygons if polygon.material_index == index for vertex in polygon.vertices}


def material_polygons(obj: bpy.types.Object, name: str) -> list[tuple[int, ...]]:
    index = material_index(obj, name)
    return [tuple(polygon.vertices) for polygon in obj.data.polygons if polygon.material_index == index]


def material_boundary_graph(obj: bpy.types.Object, name: str) -> tuple[set[int], dict[int, set[int]]]:
    index = material_index(obj, name)
    counts: Counter[tuple[int, int]] = Counter()
    for polygon in obj.data.polygons:
        if polygon.material_index != index:
            continue
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            counts[tuple(sorted((first, second)))] += 1
    adjacency: dict[int, set[int]] = defaultdict(set)
    boundary: set[int] = set()
    for (first, second), count in counts.items():
        if count != 1:
            continue
        boundary.update((first, second))
        adjacency[first].add(second)
        adjacency[second].add(first)
    return boundary, adjacency


def component_ids(adjacency: dict[int, set[int]]) -> list[set[int]]:
    seen: set[int] = set()
    result: list[set[int]] = []
    for root in adjacency:
        if root in seen:
            continue
        current = {root}
        stack = [root]
        seen.add(root)
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    current.add(neighbor)
                    stack.append(neighbor)
        result.append(current)
    return sorted(result, key=lambda item: (-len(item), min(item)))


def point_summary(points: list[Vector]) -> dict[str, object]:
    if not points:
        return {"count": 0, "measurable": False}
    center = midpoint(points)
    return {
        "count": len(points),
        "measurable": True,
        "centroid_m": [round(value, 9) for value in center],
        "bounds_min_m": [round(min(point[axis] for point in points), 9) for axis in range(3)],
        "bounds_max_m": [round(max(point[axis] for point in points), 9) for axis in range(3)],
    }


def boundary_component_report(obj: bpy.types.Object, material: str, coordinates: list[Vector]) -> list[dict[str, object]]:
    _boundary, adjacency = material_boundary_graph(obj, material)
    return [point_summary([coordinates[index] for index in component]) for component in component_ids(adjacency)[:16]]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def scalar_stats(values: list[float], scale: float = 1.0, suffix: str = "") -> dict[str, object]:
    if not values:
        return {"count": 0, "measurable": False}
    key = lambda name: f"{name}_{suffix}" if suffix else name
    scaled = [value * scale for value in values]
    return {
        "count": len(scaled),
        "measurable": True,
        key("min"): round(min(scaled), 6),
        key("mean"): round(sum(scaled) / len(scaled), 6),
        key("p50"): round(percentile(scaled, 0.50), 6),
        key("p95"): round(percentile(scaled, 0.95), 6),
        key("max"): round(max(scaled), 6),
    }


def kd_nearest_distances(source: list[Vector], target: list[Vector]) -> list[float]:
    if not source or not target:
        return []
    tree = KDTree(len(target))
    for index, point in enumerate(target):
        tree.insert(point, index)
    tree.balance()
    return [tree.find(point)[2] for point in source]


def symmetric_point_gap(first: list[Vector], second: list[Vector]) -> dict[str, object]:
    first_to_second = kd_nearest_distances(first, second)
    second_to_first = kd_nearest_distances(second, first)
    coverage = lambda values: {
        f"within_{limit:g}mm": round(sum(distance <= limit / 1000.0 for distance in values) / len(values), 6)
        if values
        else None
        for limit in (1.0, 2.0, 5.0, 10.0)
    }
    return {
        "first_points": len(first),
        "second_points": len(second),
        "first_to_second": scalar_stats(first_to_second, 1000.0, "mm"),
        "second_to_first": scalar_stats(second_to_first, 1000.0, "mm"),
        "combined": scalar_stats(first_to_second + second_to_first, 1000.0, "mm"),
        "first_to_second_coverage": coverage(first_to_second),
        "second_to_first_coverage": coverage(second_to_first),
    }


def bvh_surface_distances(points: list[Vector], coordinates: list[Vector], polygons: list[tuple[int, ...]]) -> list[float]:
    if not points or not polygons:
        return []
    tree = BVHTree.FromPolygons(coordinates, polygons, all_triangles=False)
    distances: list[float] = []
    for point in points:
        nearest = tree.find_nearest(point)
        if nearest is not None:
            distances.append(float(nearest[3]))
    return distances


def eye_metrics(
    head_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    coordinates: list[Vector],
) -> dict[str, object]:
    iris_vertices = material_vertex_ids(head_mesh, "目")
    face_boundary, face_adjacency = material_boundary_graph(head_mesh, "面")
    iris_boundary, iris_adjacency = material_boundary_graph(head_mesh, "目")
    face_polygons = material_polygons(head_mesh, "面")
    sclera_name = next(
        (
            name
            for name in ("巩膜", "Sclera", "sclera", "眼白")
            if any(slot.material is not None and slot.material.name == name for slot in head_mesh.material_slots)
        ),
    )
    sclera_vertices = material_vertex_ids(head_mesh, sclera_name) if sclera_name else set()
    sclera_boundary, sclera_adjacency = (
        material_boundary_graph(head_mesh, sclera_name) if sclera_name else (set(), {})
    )
    sclera_polygons = material_polygons(head_mesh, sclera_name) if sclera_name else []

    def nearest_component(
        adjacency: dict[int, set[int]],
        points: list[Vector],
        center: Vector,
    ) -> set[int]:
        components = component_ids(adjacency)
        if not components:
            return set()
        return min(
            components,
            key=lambda component: sum((points[index] - center).length for index in component) / len(component),
        )

    result: dict[str, object] = {}
    for side, bone_name in (("left", "LeftEye"), ("right", "RightEye")):
        center = pose_bone_point(head_armature, bone_name)
        iris_ids = {index for index in iris_vertices if (coordinates[index] - center).length <= 0.040}
        face_component = nearest_component(face_adjacency, coordinates, center)
        iris_component = nearest_component(iris_adjacency, coordinates, center)
        sclera_component = nearest_component(sclera_adjacency, coordinates, center) if sclera_name else set()
        iris_ring_ids = iris_component & iris_ids
        socket_ids = face_component
        sclera_ids = {index for index in sclera_vertices if (coordinates[index] - center).length <= 0.040}
        sclera_ring_ids = sclera_component & sclera_ids
        iris_points = [coordinates[index] for index in sorted(iris_ids)]
        ring_points = [coordinates[index] for index in sorted(iris_ring_ids or iris_ids)]
        socket_points = [coordinates[index] for index in sorted(socket_ids)]
        sclera_points = [coordinates[index] for index in sorted(sclera_ring_ids or sclera_ids)]
        iris_center = midpoint(iris_points) if iris_points else Vector()
        socket_gap = symmetric_point_gap(socket_points, sclera_points) if sclera_name else symmetric_point_gap(ring_points, socket_points)
        result[side] = {
            "bone": bone_name,
            "bone_head_m": [round(value, 9) for value in center],
            "iris_vertices": len(iris_points),
            "iris_boundary_vertices": len(iris_ring_ids),
            "socket_boundary_vertices": len(socket_ids),
            "sclera_vertices": len(sclera_points),
            "sclera_boundary_vertices": len(sclera_ring_ids),
            "iris_centroid_m": [round(value, 9) for value in iris_center],
            "iris_centroid_to_bone_mm": round((iris_center - center).length * 1000.0, 6) if iris_points else None,
            "rim_to_socket_gap": socket_gap,
            "socket_to_sclera_gap": socket_gap if sclera_name else None,
            "iris_to_sclera_surface": scalar_stats(
                bvh_surface_distances(ring_points, coordinates, sclera_polygons), 1000.0, "mm"
            )
            if sclera_name
            else None,
            "iris_to_face_surface": scalar_stats(
                bvh_surface_distances(ring_points, coordinates, face_polygons), 1000.0, "mm"
            ),
            "selection_contract": "nearest connected boundary component to the posed eye bone",
        }
    return result


def neck_metrics(
    head_mesh: bpy.types.Object,
    body_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    body_armature: bpy.types.Object,
    head_coordinates: list[Vector],
    body_coordinates: list[Vector],
) -> dict[str, object]:
    head_boundary, _adjacency = material_boundary_graph(head_mesh, "面")
    body_boundary, _adjacency = material_boundary_graph(body_mesh, "肌")
    centers = (
        pose_bone_point(head_armature, "Neck", "tail"),
        pose_bone_point(body_armature, "Neck", "tail"),
        pose_bone_point(head_armature, "Head"),
        pose_bone_point(body_armature, "Head"),
    )
    center = midpoint(centers)

    def in_volume(point: Vector) -> bool:
        return abs(point.x - center.x) <= 0.14 and abs(point.y - center.y) <= 0.15 and abs(point.z - center.z) <= 0.075

    head_ids = {index for index in head_boundary if in_volume(head_coordinates[index])}
    body_ids = {index for index in body_boundary if in_volume(body_coordinates[index])}
    head_points = [head_coordinates[index] for index in sorted(head_ids)]
    body_points = [body_coordinates[index] for index in sorted(body_ids)]
    head_surface = material_polygons(head_mesh, "面")
    body_surface = material_polygons(body_mesh, "肌")
    return {
        "sample_center_m": [round(value, 9) for value in center],
        "sample_half_extent_m": [0.14, 0.15, 0.075],
        "head_boundary": point_summary(head_points),
        "body_boundary": point_summary(body_points),
        "boundary_gap": symmetric_point_gap(head_points, body_points),
        "head_boundary_to_body_surface": scalar_stats(
            bvh_surface_distances(head_points, body_coordinates, body_surface), 1000.0, "mm"
        ),
        "body_boundary_to_head_surface": scalar_stats(
            bvh_surface_distances(body_points, head_coordinates, head_surface), 1000.0, "mm"
        ),
    }


def local_bounds(points: list[Vector]) -> dict[str, object]:
    summary = point_summary(points)
    if not points:
        return summary
    minimum = Vector(summary["bounds_min_m"])
    maximum = Vector(summary["bounds_max_m"])
    summary["extent_mm"] = [round((maximum[axis] - minimum[axis]) * 1000.0, 6) for axis in range(3)]
    return summary


def triangle_area(coordinates: list[Vector], vertices: tuple[int, ...]) -> float:
    if len(vertices) < 3:
        return 0.0
    origin = coordinates[vertices[0]]
    area = 0.0
    for index in range(1, len(vertices) - 1):
        area += ((coordinates[vertices[index]] - origin).cross(coordinates[vertices[index + 1]] - origin)).length * 0.5
    return area


def deformation_stats(
    obj: bpy.types.Object,
    rest_coordinates: list[Vector],
    pose_coordinates: list[Vector],
    selected: set[int],
) -> dict[str, object]:
    edge_ratios: list[float] = []
    for edge in obj.data.edges:
        first, second = edge.vertices
        if first not in selected or second not in selected:
            continue
        rest_length = (rest_coordinates[first] - rest_coordinates[second]).length
        pose_length = (pose_coordinates[first] - pose_coordinates[second]).length
        if rest_length <= 1e-7 or pose_length <= 1e-7:
            continue
        ratio = pose_length / rest_length
        edge_ratios.append(max(ratio, 1.0 / ratio))
    area_ratios: list[float] = []
    collapsed = 0
    expanded = 0
    for polygon in obj.data.polygons:
        vertices = tuple(polygon.vertices)
        if not all(index in selected for index in vertices):
            continue
        rest_area = triangle_area(rest_coordinates, vertices)
        pose_area = triangle_area(pose_coordinates, vertices)
        if rest_area <= 1e-10 or pose_area <= 1e-10:
            continue
        ratio = pose_area / rest_area
        collapsed += int(ratio < 0.5)
        expanded += int(ratio > 2.0)
        area_ratios.append(max(ratio, 1.0 / ratio))
    return {
        "vertices": len(selected),
        "edge_symmetric_stretch": scalar_stats(edge_ratios),
        "triangle_symmetric_area_change": scalar_stats(area_ratios),
        "triangles_below_half_area": collapsed,
        "triangles_above_double_area": expanded,
    }


def joint_metrics(
    body_mesh: bpy.types.Object,
    body_armature: bpy.types.Object,
    rest_coordinates: list[Vector],
    pose_coordinates: list[Vector],
    rest_bone_points: dict[str, Vector],
) -> dict[str, object]:
    skin_ids = material_vertex_ids(body_mesh, "肌")
    cloth_ids = material_vertex_ids(body_mesh, "Cloth1")
    skin_boundary, _adjacency = material_boundary_graph(body_mesh, "肌")
    cloth_boundary, _adjacency = material_boundary_graph(body_mesh, "Cloth1")
    skin_polygons = material_polygons(body_mesh, "肌")
    cloth_polygons = material_polygons(body_mesh, "Cloth1")
    result: dict[str, object] = {}
    configs = {
        "left_wrist": ("LeftHand", 0.115, 0.080),
        "right_wrist": ("RightHand", 0.115, 0.080),
        "left_ankle": ("LeftFoot", 0.130, 0.065),
        "right_ankle": ("RightFoot", 0.130, 0.065),
    }
    for key, (bone_name, deformation_radius, seam_radius) in configs.items():
        rest_center = rest_bone_points[bone_name]
        pose_center = pose_bone_point(body_armature, bone_name)
        local = {index for index, point in enumerate(rest_coordinates) if (point - rest_center).length <= deformation_radius}
        skin_local = local & skin_ids
        cloth_local = local & cloth_ids
        skin_seam = {index for index in skin_boundary if (rest_coordinates[index] - rest_center).length <= seam_radius}
        cloth_seam = {index for index in cloth_boundary if (rest_coordinates[index] - rest_center).length <= seam_radius}
        result[key] = {
            "bone": bone_name,
            "rest_bone_head_m": [round(value, 9) for value in rest_center],
            "pose_bone_head_m": [round(value, 9) for value in pose_center],
            "selection_radius_m": deformation_radius,
            "seam_radius_m": seam_radius,
            "rest": {
                "skin_bounds": local_bounds([rest_coordinates[index] for index in sorted(skin_local)]),
                "garment_bounds": local_bounds([rest_coordinates[index] for index in sorted(cloth_local)]),
                "skin_garment_boundary_gap": symmetric_point_gap(
                    [rest_coordinates[index] for index in sorted(skin_seam)],
                    [rest_coordinates[index] for index in sorted(cloth_seam)],
                ),
                "skin_boundary_to_garment_surface": scalar_stats(
                    bvh_surface_distances(
                        [rest_coordinates[index] for index in sorted(skin_seam)],
                        rest_coordinates,
                        cloth_polygons,
                    ),
                    1000.0,
                    "mm",
                ),
                "garment_boundary_to_skin_surface": scalar_stats(
                    bvh_surface_distances(
                        [rest_coordinates[index] for index in sorted(cloth_seam)],
                        rest_coordinates,
                        skin_polygons,
                    ),
                    1000.0,
                    "mm",
                ),
            },
            "pose": {
                "skin_bounds": local_bounds([pose_coordinates[index] for index in sorted(skin_local)]),
                "garment_bounds": local_bounds([pose_coordinates[index] for index in sorted(cloth_local)]),
                "skin_garment_boundary_gap": symmetric_point_gap(
                    [pose_coordinates[index] for index in sorted(skin_seam)],
                    [pose_coordinates[index] for index in sorted(cloth_seam)],
                ),
                "skin_boundary_to_garment_surface": scalar_stats(
                    bvh_surface_distances(
                        [pose_coordinates[index] for index in sorted(skin_seam)],
                        pose_coordinates,
                        cloth_polygons,
                    ),
                    1000.0,
                    "mm",
                ),
                "garment_boundary_to_skin_surface": scalar_stats(
                    bvh_surface_distances(
                        [pose_coordinates[index] for index in sorted(cloth_seam)],
                        pose_coordinates,
                        skin_polygons,
                    ),
                    1000.0,
                    "mm",
                ),
            },
            "deformation": {
                "skin": deformation_stats(body_mesh, rest_coordinates, pose_coordinates, skin_local),
                "garment": deformation_stats(body_mesh, rest_coordinates, pose_coordinates, cloth_local),
            },
        }
    return result


def forehead_metrics(
    head_mesh: bpy.types.Object,
    head_armature: bpy.types.Object,
    coordinates: list[Vector],
    forehead_ids: set[int] | None = None,
) -> tuple[dict[str, object], set[int]]:
    left_eye = pose_bone_point(head_armature, "LeftEye")
    right_eye = pose_bone_point(head_armature, "RightEye")
    eye_center = midpoint((left_eye, right_eye))
    face_ids = material_vertex_ids(head_mesh, "面")
    if forehead_ids is None:
        forehead_ids = {
            index
            for index in face_ids
            if abs(coordinates[index].x - eye_center.x) <= 0.060
            and eye_center.z + 0.014 <= coordinates[index].z <= eye_center.z + 0.095
            and coordinates[index].y >= eye_center.y - 0.015
        }
    forehead_points = [coordinates[index] for index in sorted(forehead_ids)]
    hair_polygons = material_polygons(head_mesh, "发")
    distances = bvh_surface_distances(forehead_points, coordinates, hair_polygons)
    front_hair_ids = {
        index
        for index in material_vertex_ids(head_mesh, "发")
        if abs(coordinates[index].x - eye_center.x) <= 0.060
        and coordinates[index].y >= eye_center.y - 0.020
        and coordinates[index].z >= eye_center.z
    }
    front_hair_z = [coordinates[index].z for index in front_hair_ids]
    brow_z = midpoint(
        (
            pose_bone_point(head_armature, "L_Brow_01"),
            pose_bone_point(head_armature, "R_Brow_01"),
        )
    ).z
    result = {
        "forehead_vertices": len(forehead_points),
        "forehead_to_hair_surface": scalar_stats(distances, 1000.0, "mm"),
        "coverage_within_5mm": round(sum(value <= 0.005 for value in distances) / len(distances), 6) if distances else None,
        "coverage_within_10mm": round(sum(value <= 0.010 for value in distances) / len(distances), 6) if distances else None,
        "front_hair_vertices": len(front_hair_ids),
        "front_hair_low_z_m": round(percentile(front_hair_z, 0.05), 9) if front_hair_z else None,
        "front_hair_p25_z_m": round(percentile(front_hair_z, 0.25), 9) if front_hair_z else None,
        "front_hair_p50_z_m": round(percentile(front_hair_z, 0.50), 9) if front_hair_z else None,
        "brow_to_front_hair_low_mm": round((percentile(front_hair_z, 0.05) - brow_z) * 1000.0, 6) if front_hair_z else None,
    }
    return result, forehead_ids


def skeleton_delta(
    head_armature: bpy.types.Object,
    body_armature: bpy.types.Object,
    pose: bool,
) -> dict[str, object]:
    names = ("Neck", "Head", "LeftEye", "RightEye", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
    result: dict[str, object] = {}
    for name in names:
        if pose:
            head_first = pose_bone_point(head_armature, name)
            body_first = pose_bone_point(body_armature, name)
            head_second = pose_bone_point(head_armature, name, "tail")
            body_second = pose_bone_point(body_armature, name, "tail")
        else:
            head_bone = head_armature.data.bones[name]
            body_bone = body_armature.data.bones[name]
            head_first = head_armature.matrix_world @ head_bone.head_local
            body_first = body_armature.matrix_world @ body_bone.head_local
            head_second = head_armature.matrix_world @ head_bone.tail_local
            body_second = body_armature.matrix_world @ body_bone.tail_local
        result[name] = {
            "head_delta_mm": round((head_first - body_first).length * 1000.0, 6),
            "tail_delta_mm": round((head_second - body_second).length * 1000.0, 6),
        }
    return result


def build_gates(metrics: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    gates: dict[str, object] = {}

    eye_checks: list[bool] = []
    for state in ("rest", "pose"):
        for side in ("left", "right"):
            item = metrics[state]["eyes"][side]
            gap = item["rim_to_socket_gap"]["combined"]
            passed = bool(gap.get("measurable")) and gap["p95_mm"] <= 6.0 and item["iris_centroid_to_bone_mm"] <= 6.0
            eye_checks.append(passed)
    gates["eyes"] = {"pass": all(eye_checks), "rule": "rim/socket p95 <= 6 mm and iris centroid/bone <= 6 mm in REST and pose"}
    if not gates["eyes"]["pass"]:
        failures.append("eyes: iris/socket spacing or eye pivot exceeds the provisional LOD0 limit")

    neck_checks: list[bool] = []
    for state in ("rest", "pose"):
        gap = metrics[state]["neck"]["boundary_gap"]["combined"]
        neck_checks.append(bool(gap.get("measurable")) and gap["p95_mm"] <= 5.0 and gap["max_mm"] <= 12.0)
    gates["neck"] = {"pass": all(neck_checks), "rule": "cross-component neck boundary p95 <= 5 mm and max <= 12 mm in REST and pose"}
    if not gates["neck"]["pass"]:
        failures.append("neck: Head and Body boundary rings do not form a close REST/pose seam")

    for joint in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        item = metrics["joint_metrics"][joint]
        skin = item["deformation"]["skin"]["edge_symmetric_stretch"]
        garment = item["deformation"]["garment"]["edge_symmetric_stretch"]
        gap = item["pose"]["skin_boundary_to_garment_surface"]
        passed = (
            bool(skin.get("measurable"))
            and bool(garment.get("measurable"))
            and bool(gap.get("measurable"))
            and skin["p95"] <= 1.35
            and garment["p95"] <= 1.35
            and gap["p95_mm"] <= 8.0
        )
        gates[joint] = {"pass": passed, "rule": "skin/garment edge p95 stretch <= 1.35 and posed skin-boundary/surface p95 <= 8 mm"}
        if not passed:
            failures.append(f"{joint}: local deformation or skin/garment seam exceeds the provisional LOD0 limit")

    forehead = metrics["rest"]["forehead"]
    hair_pass = (
        forehead["coverage_within_10mm"] is not None
        and forehead["coverage_within_10mm"] >= 0.80
        and forehead["forehead_to_hair_surface"].get("p95_mm", float("inf")) <= 15.0
    )
    gates["forehead_hair"] = {"pass": hair_pass, "rule": ">= 80% forehead samples within 10 mm of hair and forehead/hair p95 <= 15 mm"}
    if not hair_pass:
        failures.append("forehead_hair: front hair coverage remains too high or too far from the forehead")

    return gates, failures


def main() -> int:
    args = arguments()
    head_blend = args.head_blend.resolve(strict=True)
    body_blend = args.body_blend.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    save_blend = args.save_blend.resolve()
    if Path(bpy.data.filepath).resolve() != head_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {head_blend}")

    outputs = {
        f"{state}-{region}": output_dir / f"{state}-{region}.png"
        for state in ("rest", "pose")
        for region in REGION_FILES
    }
    ensure_outputs_absent([*outputs.values(), report_path, save_blend])

    scene = bpy.context.scene
    head_mesh = bpy.data.objects.get(HEAD_MESH_NAME)
    head_armature = bpy.data.objects.get(HEAD_ARMATURE_NAME)
    if head_mesh is None or head_armature is None:
        raise RuntimeError("Head/Hair blend is missing its export mesh or donor armature")
    body_mesh, body_armature = append_body_package(body_blend)
    if head_mesh.type != "MESH" or body_mesh.type != "MESH":
        raise RuntimeError("Validation packages must contain mesh export objects")
    if head_armature.type != "ARMATURE" or body_armature.type != "ARMATURE":
        raise RuntimeError("Validation packages must contain armatures")
    if head_armature is body_armature:
        raise RuntimeError("Head/Hair and Body/Garment must retain separate donor armatures")
    if len(head_armature.data.bones) != 241 or len(body_armature.data.bones) != 241:
        raise RuntimeError(
            f"Expected two 241-bone donors, found {len(head_armature.data.bones)} and {len(body_armature.data.bones)}"
        )

    camera = configure_render(scene, (head_mesh, body_mesh))
    armatures = (head_armature, body_armature)
    for armature in armatures:
        reset_pose(armature)
    bpy.context.view_layer.update()

    rest_head_coordinates = evaluated_coordinates(head_mesh)
    rest_body_coordinates = evaluated_coordinates(body_mesh)
    rest_bone_points = {
        name: pose_bone_point(body_armature, name)
        for name in ("LeftHand", "RightHand", "LeftFoot", "RightFoot")
    }
    render_suite("rest", scene, camera, head_armature, body_armature, outputs)
    rest_forehead, forehead_ids = forehead_metrics(head_mesh, head_armature, rest_head_coordinates)
    metrics: dict[str, object] = {
        "rest": {
            "skeleton_delta": skeleton_delta(head_armature, body_armature, pose=False),
            "eyes": eye_metrics(head_mesh, head_armature, rest_head_coordinates),
            "neck": neck_metrics(
                head_mesh,
                body_mesh,
                head_armature,
                body_armature,
                rest_head_coordinates,
                rest_body_coordinates,
            ),
            "forehead": rest_forehead,
        }
    }

    apply_diagnostic_pose(armatures)
    pose_head_coordinates = evaluated_coordinates(head_mesh)
    pose_body_coordinates = evaluated_coordinates(body_mesh)
    render_suite("pose", scene, camera, head_armature, body_armature, outputs)
    pose_forehead, _forehead_ids = forehead_metrics(head_mesh, head_armature, pose_head_coordinates, forehead_ids)
    metrics["pose"] = {
        "skeleton_delta": skeleton_delta(head_armature, body_armature, pose=True),
        "eyes": eye_metrics(head_mesh, head_armature, pose_head_coordinates),
        "neck": neck_metrics(
            head_mesh,
            body_mesh,
            head_armature,
            body_armature,
            pose_head_coordinates,
            pose_body_coordinates,
        ),
        "forehead": pose_forehead,
    }
    metrics["joint_metrics"] = joint_metrics(
        body_mesh,
        body_armature,
        rest_body_coordinates,
        pose_body_coordinates,
        rest_bone_points,
    )
    metrics["boundary_components"] = {
        "head_face_rest": boundary_component_report(head_mesh, "面", rest_head_coordinates),
        "body_skin_rest": boundary_component_report(body_mesh, "肌", rest_body_coordinates),
    }
    gates, diagnostic_failures = build_gates(metrics)

    for armature in armatures:
        reset_pose(armature)
    bpy.context.view_layer.update()
    scene["fh6_validation_purpose"] = "Si FBX Display LOD0 seam and pose inspection"
    scene["fh6_validation_game_tested"] = False
    bpy.ops.wm.save_as_mainfile(filepath=str(save_blend), compress=False, check_existing=False)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "FBX-first FH6 Display LOD0 combined geometry, seam, and pose validation.",
        "inputs": {
            "head_hair_blend": str(head_blend),
            "head_hair_sha256": sha256(head_blend),
            "body_garment_blend": str(body_blend),
            "body_garment_sha256": sha256(body_blend),
        },
        "assembly": {
            "head_mesh": head_mesh.name,
            "head_vertices": len(head_mesh.data.vertices),
            "head_armature": head_armature.name,
            "head_bones": len(head_armature.data.bones),
            "body_mesh": body_mesh.name,
            "body_vertices": len(body_mesh.data.vertices),
            "body_armature": body_armature.name,
            "body_bones": len(body_armature.data.bones),
            "armatures_are_distinct": head_armature is not body_armature,
        },
        "pose_degrees_xyz": {name: list(values) for name, values in POSE_DEGREES.items()},
        "camera_convention": {
            "front": "+Y camera looking toward -Y",
            "side": "+X camera looking toward -X",
        },
        "metrics": metrics,
        "diagnostic_gates": gates,
        "diagnostic_failures": diagnostic_failures,
        "renders": {name: str(path) for name, path in outputs.items()},
        "validation_blend": str(save_blend),
        "validation_blend_sha256": sha256(save_blend),
        "validation_level": {
            "structural_inputs": True,
            "blender_rest_and_pose": True,
            "modelbin": False,
            "offline_game": False,
        },
        "threshold_note": "Diagnostic limits are conservative LOD0 inspection gates, not claims about undocumented engine tolerances.",
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_SI_DISPLAY_SEAM_VALIDATION="
        + json.dumps(
            {
                "report": str(report_path),
                "blend": str(save_blend),
                "renders": len(outputs),
                "diagnostic_failures": diagnostic_failures,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
