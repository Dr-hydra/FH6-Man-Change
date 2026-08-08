#!/usr/bin/env python3
"""Repair Si Display LOD0 skin weights and the under-jaw neck opening.

The script treats the existing FBX retarget as immutable input.  It uses the
retail Female body/arms donor, whose REST skeleton is identical to the Outfit
container skeleton, for local surface-projected weights.  A small open loft
is then built from the body neck surface to the lower face boundary and joined
to the Body/Garment export mesh as the existing skin material.
"""

from __future__ import annotations

import argparse
import bmesh
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


BODY_MESH = "Si_Display_BodyGarment_LOD0"
BODY_ARMATURE = "FH6_Outfit_Race_Suit_Modern_F_Skeleton"
HEAD_MESH = "Si_Display_HeadHair_LOD0"
FEMALE_BODY = "Female_Body"
FEMALE_ARMS = "Female_Body.001"
SKIN_MATERIAL = "肌"
GARMENT_MATERIALS = ("Cloth1", "Cloth1Alpha")
GARMENT_ARM_POSES = {
    "shoulders": {
        "LeftShoulder": (8.0, 0.0, 0.0),
        "RightShoulder": (8.0, 0.0, 0.0),
        "LeftArm": (65.0, 0.0, 0.0),
        "RightArm": (65.0, 0.0, 0.0),
    },
    "elbows": {
        "LeftArm": (25.0, 0.0, 0.0),
        "RightArm": (25.0, 0.0, 0.0),
        "LeftForeArm": (75.0, 0.0, 0.0),
        "RightForeArm": (-75.0, 0.0, 0.0),
    },
}
MAX_INFLUENCES = 4
MIN_WEIGHT = 0.001
NECK_COLUMNS = 48
NECK_FRONT_COLUMNS = 32
NECK_ROWS = 4
CENTER_SEAM_EPSILON = 2.0e-5


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--head-blend", required=True, type=Path)
    parser.add_argument("--female-donor-blend", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_absent(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def material_slot(obj: bpy.types.Object, name: str) -> int:
    for index, slot in enumerate(obj.material_slots):
        if slot.material is not None and slot.material.name == name:
            return index
    raise RuntimeError(f"{obj.name} is missing material {name!r}")


def material_vertices(obj: bpy.types.Object, name: str) -> set[int]:
    slot = material_slot(obj, name)
    return {
        vertex
        for polygon in obj.data.polygons
        if polygon.material_index == slot
        for vertex in polygon.vertices
    }


def weight_dict(obj: bpy.types.Object, vertex_index: int, allowed: set[str] | None = None) -> dict[str, float]:
    names = {group.index: group.name for group in obj.vertex_groups}
    result: dict[str, float] = {}
    for item in obj.data.vertices[vertex_index].groups:
        name = names.get(item.group)
        if name is None or item.weight <= 1.0e-8 or (allowed is not None and name not in allowed):
            continue
        result[name] = result.get(name, 0.0) + float(item.weight)
    return result


def clean_weights(weights: dict[str, float], limit: int = MAX_INFLUENCES) -> dict[str, float]:
    kept = [(name, weight) for name, weight in weights.items() if weight >= MIN_WEIGHT]
    kept.sort(key=lambda item: (-item[1], item[0]))
    kept = kept[:limit]
    total = sum(weight for _name, weight in kept)
    if total <= 0.0:
        return {}
    return {name: weight / total for name, weight in kept}


def blend_weights(first: dict[str, float], second: dict[str, float], factor: float) -> dict[str, float]:
    combined: dict[str, float] = defaultdict(float)
    for name, weight in first.items():
        combined[name] += weight * (1.0 - factor)
    for name, weight in second.items():
        combined[name] += weight * factor
    return clean_weights(dict(combined))


def replace_vertex_weights(obj: bpy.types.Object, vertex_index: int, weights: dict[str, float]) -> None:
    for group in obj.vertex_groups:
        try:
            group.remove([vertex_index])
        except RuntimeError:
            pass
    for name, weight in weights.items():
        group = obj.vertex_groups.get(name)
        if group is None:
            group = obj.vertex_groups.new(name=name)
        group.add([vertex_index], weight, "REPLACE")


def append_objects(blend: Path, names: set[str]) -> list[bpy.types.Object]:
    with bpy.data.libraries.load(str(blend), link=False) as (source, target):
        missing = names - set(source.objects)
        if missing:
            raise RuntimeError(f"{blend} is missing objects {sorted(missing)}")
        target.objects = [name for name in source.objects if name in names]
    objects = [obj for obj in target.objects if obj is not None]
    for obj in objects:
        if not obj.users_collection:
            bpy.context.scene.collection.objects.link(obj)
    return objects


class DonorSurface:
    def __init__(self, obj: bpy.types.Object, allowed_bones: set[str]) -> None:
        self.obj = obj
        obj.data.calc_loop_triangles()
        self.coordinates = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        self.triangles = [tuple(item.vertices) for item in obj.data.loop_triangles]
        self.bvh = BVHTree.FromPolygons(self.coordinates, self.triangles, all_triangles=True)
        self.weights = [weight_dict(obj, index, allowed_bones) for index in range(len(obj.data.vertices))]

    @staticmethod
    def barycentric(point: Vector, a: Vector, b: Vector, c: Vector) -> tuple[float, float, float]:
        edge0 = b - a
        edge1 = c - a
        offset = point - a
        d00 = edge0.dot(edge0)
        d01 = edge0.dot(edge1)
        d11 = edge1.dot(edge1)
        d20 = offset.dot(edge0)
        d21 = offset.dot(edge1)
        denominator = d00 * d11 - d01 * d01
        if abs(denominator) <= 1.0e-14:
            return (1.0, 0.0, 0.0)
        second = (d11 * d20 - d01 * d21) / denominator
        third = (d00 * d21 - d01 * d20) / denominator
        first = 1.0 - second - third
        values = [max(0.0, min(1.0, value)) for value in (first, second, third)]
        total = sum(values)
        return tuple(value / total for value in values) if total > 0.0 else (1.0, 0.0, 0.0)

    def sample(self, point: Vector) -> tuple[dict[str, float], float]:
        nearest, _normal, triangle_index, distance = self.bvh.find_nearest(point)
        if nearest is None or triangle_index is None or distance is None:
            return {}, math.inf
        triangle = self.triangles[triangle_index]
        factors = self.barycentric(nearest, *(self.coordinates[index] for index in triangle))
        result: dict[str, float] = defaultdict(float)
        for vertex_index, factor in zip(triangle, factors):
            for name, weight in self.weights[vertex_index].items():
                result[name] += weight * factor
        return clean_weights(dict(result)), float(distance)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def smooth_wrist_transition(body: bpy.types.Object, armature: bpy.types.Object) -> dict[str, object]:
    """Widen the Hand/ForeArm blend over a stable REST-space wrist band.

    The source hand topology has a long, low-resolution transition ring.  A
    nearest-surface transfer alone can put the hand fraction on one side of
    that ring.  Rebuilding only the family fraction preserves the donor's
    finger/twist distribution while making the hinge continuous.
    """
    skin_ids = material_vertices(body, SKIN_MATERIAL)
    coordinates = [body.matrix_world @ vertex.co for vertex in body.data.vertices]
    group_names = {group.index: group.name for group in body.vertex_groups}
    changed = 0
    report: dict[str, object] = {}
    for side, sign in (("Left", -1.0), ("Right", 1.0)):
        hand_bone = armature.data.bones.get(f"{side}Hand")
        if hand_bone is None:
            continue
        hand_head = armature.matrix_world @ hand_bone.head_local
        side_ids = {
            index
            for index in skin_ids
            if abs(coordinates[index].x) >= 0.43
            and 0.84 <= coordinates[index].z <= 1.30
            and coordinates[index].x * sign > 0.0
        }
        affected = 0
        fractions: list[float] = []
        for index in sorted(side_ids):
            point = coordinates[index]
            outward = (point.x - hand_head.x) * sign
            # 140 mm total transition band centred on the Hand head.
            fraction = smoothstep((outward + 0.070) / 0.140)
            current = clean_weights(weight_dict(body, index))
            hand_part = {
                name: weight
                for name, weight in current.items()
                if name.startswith((f"{side}Hand", f"{side}Index", f"{side}Middle", f"{side}Ring", f"{side}Pinky", f"{side}Thumb"))
            }
            fore_part = {name: weight for name, weight in current.items() if name not in hand_part}
            hand_part = clean_weights(hand_part)
            fore_part = clean_weights(fore_part)
            if not hand_part or not fore_part:
                continue
            combined: dict[str, float] = defaultdict(float)
            for name, weight in fore_part.items():
                combined[name] += weight * (1.0 - fraction)
            for name, weight in hand_part.items():
                combined[name] += weight * fraction
            replacement = clean_weights(dict(combined))
            if replacement:
                replace_vertex_weights(body, index, replacement)
                changed += 1
                affected += 1
                fractions.append(fraction)
        report[side.casefold()] = {
            "selected_vertices": len(side_ids),
            "changed_vertices": affected,
            "hand_fraction_min": min(fractions) if fractions else None,
            "hand_fraction_max": max(fractions) if fractions else None,
        }
    body.data.update()
    report["changed_vertices"] = changed
    report["method"] = "smoothstep family blend across 140 mm Hand-head centred band"
    return report


def transfer_skin_weights(
    body: bpy.types.Object,
    armature: bpy.types.Object,
    female_body: bpy.types.Object,
    female_arms: bpy.types.Object,
) -> dict[str, object]:
    allowed_bones = {bone.name for bone in armature.data.bones}
    body_surface = DonorSurface(female_body, allowed_bones)
    arms_surface = DonorSurface(female_arms, allowed_bones)
    skin_ids = material_vertices(body, SKIN_MATERIAL)
    coordinates = [body.matrix_world @ vertex.co for vertex in body.data.vertices]
    inverse_world = body.matrix_world.inverted()
    landmarks = {
        name: armature.matrix_world @ armature.data.bones[name].head_local
        for name in ("Neck", "Neck1", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
    }
    region_counts: Counter[str] = Counter()
    skipped_distance: Counter[str] = Counter()
    distances: dict[str, list[float]] = defaultdict(list)
    changed = 0
    for index in sorted(skin_ids):
        point = coordinates[index]
        surface: DonorSurface | None = None
        region = ""
        blend = 0.0
        max_distance = 0.0
        # The semantic Female donor splits arms/hands from the torso/legs.
        # These coordinate windows are anchored to the identical donor REST
        # skeleton and intentionally avoid the source ears/head fragments.
        if abs(point.x) >= 0.20 and 0.84 <= point.z <= 1.32:
            surface = arms_surface
            region = "arms_hands"
            blend = 1.0 if abs(point.x) >= 0.45 else 0.82
            max_distance = 0.090
        else:
            ankle_distance = min((point - landmarks["LeftFoot"]).length, (point - landmarks["RightFoot"]).length)
            if ankle_distance <= 0.20:
                surface = body_surface
                region = "feet_ankles"
                blend = 1.0 if ankle_distance <= 0.14 else max(0.35, (0.20 - ankle_distance) / 0.06)
                max_distance = 0.085
            elif abs(point.x) <= 0.18 and 1.40 <= point.z <= 1.60:
                surface = body_surface
                region = "neck_shoulders"
                blend = 0.85
                max_distance = 0.100
        if surface is None:
            continue
        projected, distance = surface.sample(point)
        distances[region].append(distance)
        region_counts[region] += 1
        if not projected or distance > max_distance:
            skipped_distance[region] += 1
            continue
        original = clean_weights(weight_dict(body, index, allowed_bones))
        replacement = blend_weights(original, projected, blend)
        if not replacement:
            continue
        replace_vertex_weights(body, index, replacement)
        changed += 1
    body.data.update()
    # The barycentric donor already carries the production Hand/ForeArm
    # transition.  An additional family blend created visible folds on this
    # source topology, so keep it as an audit-only helper rather than applying
    # it to the export.
    wrist_smoothing = {"method": "barycentric donor weights retained", "changed_vertices": 0}
    return {
        "method": "nearest-triangle barycentric transfer from identical-REST Female body/arms donor",
        "changed_vertices": changed,
        "selected_by_region": dict(region_counts),
        "skipped_over_distance": dict(skipped_distance),
        "distance_mm": {
            region: {
                "count": len(items),
                "mean": round(sum(items) / len(items) * 1000.0, 6),
                "p95": round(float(percentile(items, 0.95)) * 1000.0, 6),
                "max": round(max(items) * 1000.0, 6),
            }
            for region, items in sorted(distances.items())
            if items
        },
        "wrist_smoothing": wrist_smoothing,
    }


def reset_armature_pose(armature: bpy.types.Object, pose_position: str = "POSE") -> None:
    armature.data.pose_position = pose_position
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.location = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)


def apply_arm_pose(armature: bpy.types.Object, rotations: dict[str, tuple[float, float, float]]) -> None:
    reset_armature_pose(armature)
    for name, degrees in rotations.items():
        bone = armature.pose.bones.get(name)
        if bone is None:
            raise RuntimeError(f"Missing garment arm diagnostic bone {name!r}")
        bone.rotation_euler = tuple(math.radians(value) for value in degrees)
    bpy.context.view_layer.update()


def evaluated_coordinates(obj: bpy.types.Object) -> list[Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world
        return [matrix @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def garment_edge_context(body: bpy.types.Object) -> dict[tuple[int, int], set[int]]:
    slots = {material_slot(body, name) for name in GARMENT_MATERIALS}
    context: dict[tuple[int, int], set[int]] = defaultdict(set)
    for polygon in body.data.polygons:
        if polygon.material_index not in slots:
            continue
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            context[tuple(sorted((first, second)))].update(vertices)
    return context


def severe_garment_edges(
    rest_coordinates: list[Vector],
    pose_coordinates: list[Vector],
    edges: Iterable[tuple[int, int]],
) -> list[dict[str, object]]:
    severe = []
    for first, second in edges:
        rest_length = (rest_coordinates[first] - rest_coordinates[second]).length
        pose_length = (pose_coordinates[first] - pose_coordinates[second]).length
        if rest_length <= 1.0e-8 or pose_length <= 1.0e-8:
            continue
        ratio = max(rest_length / pose_length, pose_length / rest_length)
        absolute_delta = abs(rest_length - pose_length)
        if ratio <= 8.0 or absolute_delta <= 0.015:
            continue
        severe.append(
            {
                "edge": [first, second],
                "ratio": round(ratio, 6),
                "rest_length_mm": round(rest_length * 1000.0, 6),
                "pose_length_mm": round(pose_length * 1000.0, 6),
                "absolute_delta_mm": round(absolute_delta * 1000.0, 6),
            }
        )
    return sorted(severe, key=lambda item: float(item["ratio"]), reverse=True)


def repair_garment_arm_discontinuities(
    body: bpy.types.Object,
    armature: bpy.types.Object,
    female_arms: bpy.types.Object,
) -> dict[str, object]:
    """Repair complete disconnected garment components that split across arm/core bones."""
    allowed_bones = {bone.name for bone in armature.data.bones}
    donor_surface = DonorSurface(female_arms, allowed_bones)
    rest_coordinates = [body.matrix_world @ vertex.co for vertex in body.data.vertices]
    edge_context = garment_edge_context(body)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edge_context:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[set[int]] = []
    vertex_component: dict[int, int] = {}
    for start in sorted(adjacency):
        if start in vertex_component:
            continue
        component_index = len(components)
        component = {start}
        vertex_component[start] = component_index
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in vertex_component:
                    continue
                vertex_component[neighbor] = component_index
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)

    initial_severe: dict[str, list[dict[str, object]]] = {}
    severe_edges: set[tuple[int, int]] = set()
    for pose_name, rotations in GARMENT_ARM_POSES.items():
        apply_arm_pose(armature, rotations)
        items = severe_garment_edges(
            rest_coordinates, evaluated_coordinates(body), edge_context
        )
        initial_severe[pose_name] = items
        severe_edges.update(tuple(item["edge"]) for item in items)
    affected_components = sorted({vertex_component[edge[0]] for edge in severe_edges})

    changed: set[int] = set()
    component_reports: list[dict[str, object]] = []
    for component_index in affected_components:
        component = components[component_index]
        points = [rest_coordinates[index] for index in component]
        bounds_min = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
        bounds_max = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
        extent = bounds_max - bounds_min
        samples = {
            index: donor_surface.sample(rest_coordinates[index]) for index in sorted(component)
        }
        valid_samples = {
            index: sample for index, sample in samples.items() if sample[0] and math.isfinite(sample[1])
        }
        if not valid_samples:
            component_reports.append(
                {
                    "component": component_index,
                    "vertices": len(component),
                    "mode": "unresolved",
                    "reason": "no Female Arms surface sample",
                }
            )
            continue

        # The only large affected components are the actual left/right sleeve
        # shells. Smaller pieces and long strips are disconnected accessories,
        # so a uniform attachment transform preserves their authored shape.
        surface_mode = len(component) >= 120
        if surface_mode:
            replacements = {index: weights for index, (weights, _distance) in valid_samples.items()}
            attachment_vertex = None
            attachment_distance = None
            mode = "per_vertex_surface"
        else:
            attachment_vertex, (attachment_weights, attachment_distance) = min(
                valid_samples.items(), key=lambda item: item[1][1]
            )
            replacements = {index: attachment_weights for index in component}
            mode = "rigid_nearest_attachment"
        for index, weights in replacements.items():
            replace_vertex_weights(body, index, weights)
        changed.update(replacements)
        distances = [distance for _weights, distance in valid_samples.values()]
        component_reports.append(
            {
                "component": component_index,
                "vertices": len(component),
                "mode": mode,
                "bounds_min_m": [round(value, 9) for value in bounds_min],
                "bounds_max_m": [round(value, 9) for value in bounds_max],
                "extent_m": [round(value, 9) for value in extent],
                "attachment_vertex": attachment_vertex,
                "attachment_distance_mm": round(attachment_distance * 1000.0, 6)
                if attachment_distance is not None
                else None,
                "surface_distance_mm": {
                    "min": round(min(distances) * 1000.0, 6),
                    "mean": round(sum(distances) / len(distances) * 1000.0, 6),
                    "max": round(max(distances) * 1000.0, 6),
                },
            }
        )
    body.data.update()
    bpy.context.view_layer.update()

    final_severe: dict[str, list[dict[str, object]]] = {}
    for pose_name, rotations in GARMENT_ARM_POSES.items():
        apply_arm_pose(armature, rotations)
        final_severe[pose_name] = severe_garment_edges(
            rest_coordinates, evaluated_coordinates(body), edge_context
        )
    reset_armature_pose(armature, "REST")
    bpy.context.view_layer.update()
    return {
        "method": "component-level Female Arms repair for garment edges with ratio > 8 and absolute change > 15 mm; sleeve shells use per-vertex projection and disconnected accessories use a rigid nearest attachment",
        "pose_degrees_xyz": {
            name: {bone: list(values) for bone, values in rotations.items()}
            for name, rotations in GARMENT_ARM_POSES.items()
        },
        "initial_severe_edges": {name: len(items) for name, items in initial_severe.items()},
        "initial_worst_edges": {name: items[:10] for name, items in initial_severe.items()},
        "affected_components": len(affected_components),
        "components": component_reports,
        "changed_vertices": len(changed),
        "final_severe_edges": {name: len(items) for name, items in final_severe.items()},
        "final_worst_edges": {name: items[:10] for name, items in final_severe.items()},
    }


def material_boundary_components_with_edges(
    obj: bpy.types.Object,
    material_name: str,
) -> list[tuple[set[int], set[tuple[int, int]]]]:
    slot = material_slot(obj, material_name)
    edge_counts: Counter[tuple[int, int]] = Counter()
    for polygon in obj.data.polygons:
        if polygon.material_index != slot:
            continue
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            edge_counts[tuple(sorted((first, second)))] += 1
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (first, second), count in edge_counts.items():
        if count != 1:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
    components: list[tuple[set[int], set[tuple[int, int]]]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        component: set[int] = set()
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            queue.extend(adjacency[current] - component)
        remaining -= component
        component_edges = {edge for edge in boundary_edges if edge[0] in component and edge[1] in component}
        components.append((component, component_edges))
    return sorted(components, key=lambda item: (-len(item[0]), min(item[0])))


def material_boundary_components(obj: bpy.types.Object, material_name: str) -> list[set[int]]:
    return [vertices for vertices, _edges in material_boundary_components_with_edges(obj, material_name)]


def ordered_boundary_loop(vertices: set[int], edges: set[tuple[int, int]]) -> tuple[list[int], bool]:
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
        choices = sorted(adjacency[current] - ({previous} if previous is not None else set()))
        if not choices:
            break
        next_index = choices[0]
        if closed and next_index == start:
            break
        if next_index in ordered:
            return sorted(vertices), False
        ordered.append(next_index)
        previous, current = current, next_index
        if len(ordered) == len(vertices):
            break
    return (ordered if len(ordered) == len(vertices) else sorted(vertices)), closed


def boundary_edge_counts(obj: bpy.types.Object) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            counts[tuple(sorted((first, second)))] += 1
    return counts


def weld_neck_centerline(body: bpy.types.Object) -> tuple[list[int], dict[str, object]]:
    """Weld only the coincident centerline vertices of the two native neck halves."""
    material_components = material_boundary_components_with_edges(body, SKIN_MATERIAL)
    coordinates = [body.matrix_world @ vertex.co for vertex in body.data.vertices]
    candidates: list[set[int]] = []
    for vertices, _edges in material_components:
        points = [coordinates[index] for index in vertices]
        minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
        maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
        if (
            30 <= len(vertices) <= 50
            and 1.49 <= minimum.z <= 1.55
            and 1.62 <= maximum.z <= 1.68
            and 0.045 <= maximum.x - minimum.x <= 0.065
            and maximum.y - minimum.y >= 0.070
        ):
            candidates.append(vertices)
    if len(candidates) != 2:
        raise RuntimeError(f"Expected two native neck half-shell boundaries, found {len(candidates)}")
    selected = {
        index
        for component in candidates
        for index in component
        if abs(coordinates[index].x) <= CENTER_SEAM_EPSILON
    }
    before_vertices = len(body.data.vertices)
    mesh = bmesh.new()
    mesh.from_mesh(body.data)
    mesh.verts.ensure_lookup_table()
    bmesh.ops.remove_doubles(
        mesh,
        verts=[mesh.verts[index] for index in sorted(selected)],
        dist=CENTER_SEAM_EPSILON,
    )
    mesh.to_mesh(body.data)
    mesh.free()
    body.data.update()

    all_edges = boundary_edge_counts(body)
    material_components = material_boundary_components_with_edges(body, SKIN_MATERIAL)
    ring_candidates: list[tuple[set[int], set[tuple[int, int]]]] = []
    for vertices, edges in material_components:
        points = [body.matrix_world @ body.data.vertices[index].co for index in vertices]
        minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
        maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
        centroid_x = sum(point.x for point in points) / len(points)
        if (
            30 <= len(vertices) <= 50
            and 1.50 <= minimum.z <= 1.58
            and 1.62 <= maximum.z <= 1.68
            and 0.09 <= maximum.x - minimum.x <= 0.12
            and 0.07 <= maximum.y - minimum.y <= 0.11
            and abs(centroid_x) <= 0.01
            and all(all_edges[edge] == 1 for edge in edges)
        ):
            ring_candidates.append((vertices, edges))
    if len(ring_candidates) != 1:
        raise RuntimeError(f"Expected one welded neck opening ring, found {len(ring_candidates)}")
    ring_vertices, ring_edges = ring_candidates[0]
    ordered, closed = ordered_boundary_loop(ring_vertices, ring_edges)
    if not closed or len(ordered) < 24:
        raise RuntimeError("Welded neck opening is not a single closed boundary loop")
    report = {
        "half_shell_components": len(candidates),
        "selected_center_vertices": len(selected),
        "merged_vertices": before_vertices - len(body.data.vertices),
        "ring_vertices": len(ordered),
        "ring_edges": len(ring_edges),
        "ring_closed": closed,
        "ring_bounds_min_m": [
            round(float(min((body.matrix_world @ body.data.vertices[index].co)[axis] for index in ordered)), 9)
            for axis in range(3)
        ],
        "ring_bounds_max_m": [
            round(float(max((body.matrix_world @ body.data.vertices[index].co)[axis] for index in ordered)), 9)
            for axis in range(3)
        ],
    }
    return ordered, report


def blend_weight_maps(first: dict[str, float], second: dict[str, float], factor: float) -> dict[str, float]:
    combined: dict[str, float] = defaultdict(float)
    for name, weight in first.items():
        combined[name] += weight * (1.0 - factor)
    for name, weight in second.items():
        combined[name] += weight * factor
    return clean_weights(dict(combined))


def canonical_face_bridge_weights(weights: dict[str, float], allowed: set[str]) -> dict[str, float]:
    """Collapse facial side/detail bones to the driven neck/head chain.

    The two donor armatures contain ear and dense cheek bones at slightly
    different rest pivots.  Carrying those groups onto a cross-component neck
    bridge creates a visible split when Head/Neck are posed independently.
    Jaw remains a real donor bone; all ear/detail groups are folded into Head.
    """
    combined: dict[str, float] = defaultdict(float)
    for name, weight in weights.items():
        if name in {"L_Ear", "R_Ear"} or "Ear" in name:
            destination = "Head"
        elif "Cheek" in name or "Jaw" in name:
            destination = "Jaw" if "Jaw" in allowed else "Head"
        elif name in {"Head", "Neck", "Neck1", "Spine", "Spine1", "Spine2"}:
            destination = name
        else:
            destination = "Head"
        if destination in allowed:
            combined[destination] += weight
    return clean_weights(dict(combined))


def resample_samples(samples: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if len(samples) < 2:
        raise RuntimeError("A bridge curve requires at least two samples")
    deduplicated = [samples[0]]
    for sample in samples[1:]:
        if (sample["point"] - deduplicated[-1]["point"]).length > 1.0e-6:
            deduplicated.append(sample)
    lengths = [0.0]
    for first, second in zip(deduplicated, deduplicated[1:]):
        lengths.append(lengths[-1] + (second["point"] - first["point"]).length)
    if lengths[-1] <= 1.0e-6:
        raise RuntimeError("Bridge curve has zero arc length")
    output: list[dict[str, object]] = []
    segment = 0
    for sample_index in range(count):
        target = lengths[-1] * sample_index / (count - 1)
        while segment + 1 < len(lengths) - 1 and lengths[segment + 1] < target:
            segment += 1
        start = lengths[segment]
        end = lengths[segment + 1]
        factor = (target - start) / max(end - start, 1.0e-9)
        first = deduplicated[segment]
        second = deduplicated[segment + 1]
        normal = first["normal"].lerp(second["normal"], factor)
        output.append(
            {
                "point": first["point"].lerp(second["point"], factor),
                "normal": normal.normalized() if normal.length > 1.0e-8 else first["normal"].copy(),
                "weights": blend_weight_maps(first["weights"], second["weights"], factor),
            }
        )
    return output


def longest_true_run(flags: list[bool]) -> list[int]:
    if not any(flags):
        return []
    if all(flags):
        return list(range(len(flags)))
    runs: list[list[int]] = []
    for start, flag in enumerate(flags):
        if not flag or flags[(start - 1) % len(flags)]:
            continue
        current: list[int] = []
        index = start
        while flags[index] and len(current) < len(flags):
            current.append(index)
            index = (index + 1) % len(flags)
        runs.append(current)
    return max(runs, key=len)


def lower_jaw_curve(head: bpy.types.Object) -> tuple[list[dict[str, object]], dict[str, object]]:
    coordinates = [head.matrix_world @ vertex.co for vertex in head.data.vertices]
    normal_matrix = head.matrix_world.to_3x3().inverted().transposed()
    candidates: list[tuple[set[int], set[tuple[int, int]]]] = []
    for component, edges in material_boundary_components_with_edges(head, "面"):
        points = [coordinates[index] for index in component]
        xs = [point.x for point in points]
        zs = [point.z for point in points]
        if len(points) >= 50 and max(xs) - min(xs) >= 0.09 and min(zs) <= 1.61 and max(zs) >= 1.68:
            candidates.append((component, edges))
    if not candidates:
        raise RuntimeError("Could not identify the outer face boundary for the neck bridge")
    component, edges = min(candidates, key=lambda item: min(coordinates[index].z for index in item[0]))
    ordered, closed = ordered_boundary_loop(component, edges)
    if not closed:
        raise RuntimeError("The outer face boundary is not a closed loop")
    points = [coordinates[index] for index in ordered]
    minimum_z = min(point.z for point in points)
    flags = [point.z <= minimum_z + 0.052 and point.y >= 0.030 for point in points]
    run = longest_true_run(flags)
    if len(run) < 8:
        raise RuntimeError("Could not isolate a contiguous lower-jaw boundary arc")
    jaw_indices = [ordered[index] for index in run]
    jaw = [
        {
            "point": coordinates[index].copy(),
            "normal": (normal_matrix @ head.data.vertices[index].normal).normalized(),
            "weights": clean_weights(weight_dict(head, index)),
        }
        for index in jaw_indices
    ]
    if jaw[0]["point"].x > jaw[-1]["point"].x:
        jaw.reverse()
    return jaw, {
        "component_vertices": len(component),
        "jaw_vertices": len(jaw),
        "closed_source_boundary": closed,
        "bounds_min_m": [round(min(sample["point"][axis] for sample in jaw), 9) for axis in range(3)],
        "bounds_max_m": [round(max(sample["point"][axis] for sample in jaw), 9) for axis in range(3)],
    }


def body_skin_surface(body: bpy.types.Object) -> tuple[BVHTree, list[Vector], list[tuple[int, int, int]]]:
    required_groups = {
        name
        for row in row_samples[1:]
        for sample in row
        for name in sample["weights"]
        if name in allowed_bones
    }
    for name in sorted(required_groups):
        if body.vertex_groups.get(name) is None:
            body.vertex_groups.new(name=name)

    slot = material_slot(body, SKIN_MATERIAL)
    coordinates = [body.matrix_world @ vertex.co for vertex in body.data.vertices]
    triangles: list[tuple[int, int, int]] = []
    for polygon in body.data.polygons:
        if polygon.material_index != slot:
            continue
        vertices = list(polygon.vertices)
        for index in range(1, len(vertices) - 1):
            triangles.append((vertices[0], vertices[index], vertices[index + 1]))
    return BVHTree.FromPolygons(coordinates, triangles, all_triangles=True), coordinates, triangles


def resample_closed_samples(samples: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if len(samples) < 3:
        raise RuntimeError("A closed bridge ring requires at least three samples")
    lengths = [0.0]
    for index, first in enumerate(samples):
        second = samples[(index + 1) % len(samples)]
        lengths.append(lengths[-1] + (second["point"] - first["point"]).length)
    total = lengths[-1]
    if total <= 1.0e-6:
        raise RuntimeError("Closed neck ring has zero arc length")
    output: list[dict[str, object]] = []
    segment = 0
    for sample_index in range(count):
        target = total * sample_index / count
        while segment + 1 < len(lengths) and lengths[segment + 1] < target:
            segment += 1
        first_index = segment % len(samples)
        second_index = (first_index + 1) % len(samples)
        start = lengths[segment]
        end = lengths[segment + 1]
        factor = (target - start) / max(end - start, 1.0e-9)
        first = samples[first_index]
        second = samples[second_index]
        normal = first["normal"].lerp(second["normal"], factor)
        output.append(
            {
                "point": first["point"].lerp(second["point"], factor),
                "normal": normal.normalized() if normal.length > 1.0e-8 else first["normal"].copy(),
                "weights": blend_weight_maps(first["weights"], second["weights"], factor),
            }
        )
    return output


def align_closed_samples(
    samples: list[dict[str, object]],
    target: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    best: tuple[float, bool, int, list[dict[str, object]], list[dict[str, object]]] | None = None
    for reversed_order in (False, True):
        oriented = list(reversed(samples)) if reversed_order else list(samples)
        for shift in range(len(oriented)):
            ordered = oriented[shift:] + oriented[:shift]
            resampled = resample_closed_samples(ordered, len(target))
            distances = [
                (sample["point"] - target_sample["point"]).length
                for sample, target_sample in zip(resampled, target)
            ]
            score = sum(distance * distance for distance in distances) / len(distances)
            candidate = (score, reversed_order, shift, ordered, resampled)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError("Could not align the native neck boundary to the face ring")
    score, reversed_order, shift, ordered, resampled = best
    distances = [
        (sample["point"] - target_sample["point"]).length
        for sample, target_sample in zip(resampled, target)
    ]
    return ordered, resampled, {
        "reversed": reversed_order,
        "cyclic_shift": shift,
        "rms_distance_mm": round(math.sqrt(score) * 1000.0, 6),
        "min_distance_mm": round(min(distances) * 1000.0, 6),
        "p95_distance_mm": round(float(percentile(distances, 0.95)) * 1000.0, 6),
        "max_distance_mm": round(max(distances) * 1000.0, 6),
    }


def zipper_faces(first: list[bmesh.types.BMVert], second: list[bmesh.types.BMVert]) -> list[tuple[bmesh.types.BMVert, bmesh.types.BMVert, bmesh.types.BMVert]]:
    """Triangulate a closed ring pair with different vertex counts."""
    faces: list[tuple[bmesh.types.BMVert, bmesh.types.BMVert, bmesh.types.BMVert]] = []
    first_index = 0
    second_index = 0
    while first_index < len(first) or second_index < len(second):
        first_next = (first_index + 1) / len(first) if first_index < len(first) else math.inf
        second_next = (second_index + 1) / len(second) if second_index < len(second) else math.inf
        first_current = first[first_index % len(first)]
        second_current = second[second_index % len(second)]
        if second_next <= first_next:
            second_after = second[(second_index + 1) % len(second)]
            faces.append((first_current, second_current, second_after))
            second_index += 1
        else:
            first_after = first[(first_index + 1) % len(first)]
            faces.append((first_current, second_current, first_after))
            first_index += 1
    return faces


def add_neck_bridge(body: bpy.types.Object, armature: bpy.types.Object, head: bpy.types.Object) -> dict[str, object]:
    bottom_indices, weld_report = weld_neck_centerline(body)
    jaw, jaw_report = lower_jaw_curve(head)
    allowed_bones = {bone.name for bone in armature.data.bones}
    body_normal_matrix = body.matrix_world.to_3x3().inverted().transposed()
    inverse_world = body.matrix_world.inverted()

    bottom_samples = []
    for index in bottom_indices:
        world_point = body.matrix_world @ body.data.vertices[index].co
        normal = (body_normal_matrix @ body.data.vertices[index].normal).normalized()
        bottom_samples.append(
            {
                "point": world_point,
                "normal": normal,
                "weights": clean_weights(weight_dict(body, index, allowed_bones)),
                "source_index": index,
            }
        )
    jaw = [
        {
            **sample,
            "weights": canonical_face_bridge_weights(sample["weights"], allowed_bones),
        }
        for sample in jaw
    ]
    front = resample_samples(jaw, NECK_FRONT_COLUMNS)
    if front[0]["point"].x > front[-1]["point"].x:
        front.reverse()
    left = front[0]
    right = front[-1]
    back_count = NECK_COLUMNS - NECK_FRONT_COLUMNS
    side_y = (left["point"].y + right["point"].y) * 0.5
    back_depth = max(0.075, abs(left["point"].y - right["point"].y) + 0.055)
    top_ring = list(front)
    for index in range(1, back_count + 1):
        factor = index / (back_count + 1)
        theta = math.pi * factor
        x = (left["point"].x + right["point"].x) * 0.5 + (right["point"].x - left["point"].x) * 0.5 * math.cos(theta)
        y = side_y - back_depth * math.sin(theta)
        z = right["point"].z * (1.0 - factor) + left["point"].z * factor
        normal = Vector((x, y - side_y, 0.0))
        if normal.length <= 1.0e-8:
            normal = Vector((0.0, -1.0, 0.0))
        top_ring.append(
            {
                "point": Vector((x, y, z)),
                "normal": normal.normalized(),
                "weights": blend_weight_maps(right["weights"], left["weights"], factor),
            }
        )
    top_ring = [
        {
            **sample,
            "weights": clean_weights(sample["weights"], limit=MAX_INFLUENCES),
        }
        for sample in top_ring
    ]
    bottom_samples, bottom_resampled, alignment_report = align_closed_samples(bottom_samples, top_ring)
    bottom_indices = [int(sample["source_index"]) for sample in bottom_samples]

    row_factors = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    row_samples: list[list[dict[str, object]]] = [bottom_resampled]
    for factor in row_factors:
        smooth = factor * factor * (3.0 - 2.0 * factor)
        row_samples.append(
            [
                {
                    "point": bottom["point"].lerp(top["point"], factor),
                    "normal": bottom["normal"].lerp(top["normal"], smooth).normalized(),
                    "weights": blend_weight_maps(bottom["weights"], top["weights"], smooth),
                }
                for bottom, top in zip(bottom_resampled, top_ring)
            ]
        )

    slot = material_slot(body, SKIN_MATERIAL)
    before_vertices = len(body.data.vertices)
    before_polygons = len(body.data.polygons)
    mesh = bmesh.new()
    mesh.from_mesh(body.data)
    mesh.verts.ensure_lookup_table()
    deform = mesh.verts.layers.deform.verify()
    row_layer = mesh.verts.layers.int.get("fh6_neck_bridge_row") or mesh.verts.layers.int.new("fh6_neck_bridge_row")
    column_layer = mesh.verts.layers.int.get("fh6_neck_bridge_column") or mesh.verts.layers.int.new("fh6_neck_bridge_column")
    bottom_layer = mesh.verts.layers.int.get("fh6_neck_bridge_bottom") or mesh.verts.layers.int.new("fh6_neck_bridge_bottom")
    for vertex in mesh.verts:
        vertex[row_layer] = -1
        vertex[column_layer] = -1
        vertex[bottom_layer] = 0
    bottom_vertices = [mesh.verts[index] for index in bottom_indices]
    for vertex in bottom_vertices:
        vertex[bottom_layer] = 1

    group_indices = {group.name: group.index for group in body.vertex_groups}
    created_rows: list[list[bmesh.types.BMVert]] = [bottom_vertices]
    desired_normals: dict[tuple[int, int], Vector] = {}
    for row in range(1, NECK_ROWS):
        current_row: list[bmesh.types.BMVert] = []
        for column, sample in enumerate(row_samples[row]):
            vertex = mesh.verts.new(tuple(inverse_world @ sample["point"]))
            vertex[row_layer] = row
            vertex[column_layer] = column
            for name, weight in clean_weights(sample["weights"], MAX_INFLUENCES).items():
                group_index = group_indices.get(name)
                if group_index is not None:
                    vertex[deform][group_index] = weight
            current_row.append(vertex)
            desired_normals[(row, column)] = (body_normal_matrix @ sample["normal"]).normalized()
        created_rows.append(current_row)

    uv_layer = mesh.loops.layers.uv.verify()
    uv_by_vertex: dict[bmesh.types.BMVert, tuple[float, float]] = {}
    for column, vertex in enumerate(bottom_vertices):
        uv_by_vertex[vertex] = (column / len(bottom_vertices), 0.0)
    for row in range(1, NECK_ROWS):
        for column, vertex in enumerate(created_rows[row]):
            uv_by_vertex[vertex] = (column / NECK_COLUMNS, row / (NECK_ROWS - 1))

    created_faces: list[bmesh.types.BMFace] = []
    def create_face(vertices: tuple[bmesh.types.BMVert, bmesh.types.BMVert, bmesh.types.BMVert]) -> None:
        face = mesh.faces.new(vertices)
        face.material_index = slot
        face.smooth = True
        for loop in face.loops:
            loop[uv_layer].uv = uv_by_vertex[loop.vert]
        created_faces.append(face)

    for vertices in zipper_faces(created_rows[0], created_rows[1]):
        create_face(vertices)
    for row in range(1, NECK_ROWS - 1):
        first = created_rows[row]
        second = created_rows[row + 1]
        for column in range(NECK_COLUMNS):
            next_column = (column + 1) % NECK_COLUMNS
            create_face((first[column], first[next_column], second[next_column]))
            create_face((first[column], second[next_column], second[column]))
    mesh.normal_update()
    radial_score = 0.0
    for face in created_faces:
        center = body.matrix_world @ face.calc_center_median()
        radial = Vector((center.x, center.y - side_y, 0.0))
        if radial.length > 1.0e-8:
            radial.normalize()
            radial_score += (body.matrix_world.to_3x3() @ face.normal).dot(radial)
    winding_flipped = radial_score < 0.0
    if winding_flipped:
        for face in created_faces:
            face.normal_flip()
        radial_score = -radial_score
    mesh.to_mesh(body.data)
    mesh.free()
    body.data.update()

    # Keep donor corner normals everywhere else, while making the new bridge
    # follow the measured skin normals across the two component boundary.
    body.data.update()
    corner_normals = [Vector(item.vector) for item in body.data.corner_normals]
    row_attribute = body.data.attributes.get("fh6_neck_bridge_row")
    column_attribute = body.data.attributes.get("fh6_neck_bridge_column")
    bottom_attribute = body.data.attributes.get("fh6_neck_bridge_bottom")
    if row_attribute is None or column_attribute is None or bottom_attribute is None:
        raise RuntimeError("Neck bridge vertex metadata was not preserved by BMesh")
    desired_bottom = {
        index: (body_normal_matrix @ sample["normal"]).normalized()
        for index, sample in zip(bottom_indices, bottom_samples)
    }
    for polygon in body.data.polygons:
        if not any(row_attribute.data[index].value >= 1 for index in polygon.vertices):
            continue
        for loop_index in polygon.loop_indices:
            vertex_index = body.data.loops[loop_index].vertex_index
            row = int(row_attribute.data[vertex_index].value)
            column = int(column_attribute.data[vertex_index].value)
            if row >= 1:
                corner_normals[loop_index] = desired_normals[(row, column)]
            elif vertex_index in desired_bottom:
                corner_normals[loop_index] = desired_bottom[vertex_index]
    body.data.normals_split_custom_set(corner_normals)

    added_vertices = len(body.data.vertices) - before_vertices
    added_polygons = len(body.data.polygons) - before_polygons
    body["fh6_neck_bridge"] = "native-boundary-4-ring-v001"
    body["fh6_neck_bridge_vertex_start"] = before_vertices
    body["fh6_neck_bridge_vertex_count"] = added_vertices
    body["fh6_neck_bridge_columns"] = NECK_COLUMNS
    body["fh6_neck_bridge_rows"] = NECK_ROWS
    body["fh6_neck_bridge_bottom_count"] = len(bottom_indices)
    return {
        "method": "native welded body opening to measured jaw boundary, four total rings, canonical face bridge weights",
        "jaw": jaw_report,
        "weld": weld_report,
        "ring_alignment": alignment_report,
        "columns": NECK_COLUMNS,
        "front_columns": NECK_FRONT_COLUMNS,
        "rows": NECK_ROWS,
        "bottom_ring_vertices": len(bottom_indices),
        "added_vertices": added_vertices,
        "added_polygons": added_polygons,
        "radial_winding_score": round(radial_score, 8),
        "winding_flipped": winding_flipped,
        "weight_rows": ["native body boundary", "Body/Head distance 1/3", "Body/Head distance 2/3", "measured face boundary"],
    }


def normalize_weights(obj: bpy.types.Object, armature: bpy.types.Object) -> dict[str, int]:
    allowed = {bone.name for bone in armature.data.bones}
    changed = 0
    for vertex in obj.data.vertices:
        raw = weight_dict(obj, vertex.index, allowed)
        cleaned = clean_weights(raw)
        if not cleaned:
            continue
        raw_total = sum(raw.values())
        needs_update = (
            len(raw) > MAX_INFLUENCES
            or abs(raw_total - 1.0) > 1.0e-6
            or set(raw) != set(cleaned)
            or any(abs(raw[name] - cleaned[name]) > 1.0e-6 for name in cleaned)
        )
        if needs_update:
            replace_vertex_weights(obj, vertex.index, cleaned)
            changed += 1
    obj.data.update()
    return {"changed_vertices": changed}


def validate_weights(obj: bpy.types.Object, armature: bpy.types.Object) -> dict[str, object]:
    allowed = {bone.name for bone in armature.data.bones}
    histogram: Counter[int] = Counter()
    sums: list[float] = []
    unresolved: set[str] = set()
    zero = 0
    over = 0
    not_normalized = 0
    for vertex in obj.data.vertices:
        weights = []
        for item in vertex.groups:
            group = obj.vertex_groups[item.group]
            if item.weight <= 1.0e-8:
                continue
            weights.append(float(item.weight))
            if group.name not in allowed:
                unresolved.add(group.name)
        histogram[len(weights)] += 1
        zero += int(not weights)
        over += int(len(weights) > MAX_INFLUENCES)
        if weights:
            total = sum(weights)
            sums.append(total)
            not_normalized += int(abs(total - 1.0) > 1.0e-4)
    return {
        "influence_histogram": dict(sorted(histogram.items())),
        "zero_weight_vertices": zero,
        "vertices_over_limit": over,
        "vertices_not_normalized": not_normalized,
        "unresolved_groups": sorted(unresolved),
        "weight_sum_min": min(sums) if sums else 0.0,
        "weight_sum_max": max(sums) if sums else 0.0,
        "max_weight_sum_error": max((abs(total - 1.0) for total in sums), default=0.0),
    }


def main() -> None:
    args = arguments()
    source = args.source_blend.resolve()
    head_blend = args.head_blend.resolve()
    donor_blend = args.female_donor_blend.resolve()
    output = args.output_blend.resolve()
    report_path = args.report.resolve()
    ensure_absent((output, report_path))
    if Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source}")
    body = bpy.data.objects.get(BODY_MESH)
    armature = bpy.data.objects.get(BODY_ARMATURE)
    if body is None or body.type != "MESH" or armature is None or armature.type != "ARMATURE":
        raise RuntimeError("Body repair input is missing the expected export mesh or donor armature")
    if armature.data.pose_position != "REST":
        armature.data.pose_position = "REST"
    # Copy the mutable data block even though the output is saved to a new
    # file; this keeps linked/source mesh users isolated in the in-memory scene.
    body.data = body.data.copy()

    donors = append_objects(donor_blend, {FEMALE_BODY, FEMALE_ARMS})
    donor_by_name = {obj.name: obj for obj in donors}
    transfer_report = transfer_skin_weights(body, armature, donor_by_name[FEMALE_BODY], donor_by_name[FEMALE_ARMS])
    garment_arm_report = repair_garment_arm_discontinuities(
        body, armature, donor_by_name[FEMALE_ARMS]
    )
    head_objects = append_objects(head_blend, {HEAD_MESH})
    head = head_objects[0]
    neck_report = add_neck_bridge(body, armature, head)
    normalization_report = normalize_weights(body, armature)

    for obj in [*donors, *head_objects]:
        bpy.data.objects.remove(obj, do_unlink=True)
    body["fh6_joint_repair"] = "female-donor-barycentric-garment-pose-v002"
    body["fh6_neck_bridge"] = "native-boundary-4-ring-v001"
    validation = validate_weights(body, armature)
    if (
        validation["zero_weight_vertices"]
        or validation["vertices_over_limit"]
        or validation["vertices_not_normalized"]
        or validation["unresolved_groups"]
    ):
        raise RuntimeError(f"Body repair weight validation failed: {validation}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=False, check_existing=False)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Si FBX Display LOD0 body/garment weight and under-jaw neck repair.",
        "inputs": {
            "body_blend": str(source),
            "body_sha256": sha256(source),
            "head_blend": str(head_blend),
            "head_sha256": sha256(head_blend),
            "female_donor_blend": str(donor_blend),
            "female_donor_sha256": sha256(donor_blend),
        },
        "result": {
            "blend": str(output),
            "mesh": body.name,
            "vertices": len(body.data.vertices),
            "polygons": len(body.data.polygons),
            "materials": [slot.material.name if slot.material else None for slot in body.material_slots],
            "weights": validation,
        },
        "weight_transfer": transfer_report,
        "garment_arm_discontinuity_repair": garment_arm_report,
        "post_bridge_normalization": normalization_report,
        "neck_bridge": neck_report,
        "validation_level": {"data": True, "blender_visual": False, "modelbin": False, "offline_game": False},
        "license_guard": "Local technical validation only; do not redistribute the source character.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_SI_BODY_REPAIR=" + json.dumps({"blend": str(output), "vertices": len(body.data.vertices), "changed_weights": transfer_report["changed_vertices"] + garment_arm_report["changed_vertices"], "garment_arm_vertices": garment_arm_report["changed_vertices"], "neck_vertices": neck_report["added_vertices"]}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
