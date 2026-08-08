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
MAX_INFLUENCES = 4
MIN_WEIGHT = 0.001
NECK_COLUMNS = 48
NECK_ROWS = 8


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


def material_boundary_components(obj: bpy.types.Object, material_name: str) -> list[set[int]]:
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
    components: list[set[int]] = []
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
        components.append(component)
    return sorted(components, key=lambda item: (-len(item), min(item)))


def resample_curve(points: list[Vector], count: int) -> list[Vector]:
    if len(points) < 2:
        raise RuntimeError("A bridge curve requires at least two points")
    deduplicated = [points[0]]
    for point in points[1:]:
        if (point - deduplicated[-1]).length > 1.0e-6:
            deduplicated.append(point)
    lengths = [0.0]
    for first, second in zip(deduplicated, deduplicated[1:]):
        lengths.append(lengths[-1] + (second - first).length)
    if lengths[-1] <= 1.0e-6:
        raise RuntimeError("Bridge curve has zero arc length")
    output: list[Vector] = []
    segment = 0
    for sample in range(count):
        target = lengths[-1] * sample / (count - 1)
        while segment + 1 < len(lengths) - 1 and lengths[segment + 1] < target:
            segment += 1
        start = lengths[segment]
        end = lengths[segment + 1]
        factor = (target - start) / max(end - start, 1.0e-9)
        output.append(deduplicated[segment].lerp(deduplicated[segment + 1], factor))
    return output


def lower_jaw_curve(head: bpy.types.Object) -> tuple[list[Vector], dict[str, object]]:
    coordinates = [head.matrix_world @ vertex.co for vertex in head.data.vertices]
    candidates: list[tuple[set[int], list[Vector]]] = []
    for component in material_boundary_components(head, "面"):
        points = [coordinates[index] for index in component]
        xs = [point.x for point in points]
        zs = [point.z for point in points]
        if len(points) >= 50 and max(xs) - min(xs) >= 0.09 and min(zs) <= 1.61 and max(zs) >= 1.68:
            candidates.append((component, points))
    if not candidates:
        raise RuntimeError("Could not identify the outer face boundary for the neck bridge")
    component, points = min(candidates, key=lambda item: min(point.z for point in item[1]))
    minimum_z = min(point.z for point in points)
    jaw = [point for point in points if point.z <= minimum_z + 0.052 and point.y >= 0.030]
    jaw.sort(key=lambda point: (point.x, point.z, point.y))
    sampled = resample_curve(jaw, NECK_COLUMNS)
    return sampled, {
        "component_vertices": len(component),
        "jaw_vertices": len(jaw),
        "bounds_min_m": [round(min(point[axis] for point in jaw), 9) for axis in range(3)],
        "bounds_max_m": [round(max(point[axis] for point in jaw), 9) for axis in range(3)],
    }


def body_skin_surface(body: bpy.types.Object) -> tuple[BVHTree, list[Vector], list[tuple[int, int, int]]]:
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


def add_neck_bridge(body: bpy.types.Object, armature: bpy.types.Object, head: bpy.types.Object) -> dict[str, object]:
    jaw, jaw_report = lower_jaw_curve(head)
    jaw_by_x = sorted(jaw, key=lambda point: point.x)

    def jaw_at_x(value: float) -> Vector:
        if value <= jaw_by_x[0].x:
            return jaw_by_x[0].copy()
        if value >= jaw_by_x[-1].x:
            return jaw_by_x[-1].copy()
        for first, second in zip(jaw_by_x, jaw_by_x[1:]):
            if first.x <= value <= second.x:
                factor = (value - first.x) / max(second.x - first.x, 1.0e-9)
                return first.lerp(second, factor)
        return jaw_by_x[-1].copy()

    # A closed upper ring keeps the side/back neck under the hair filled as
    # well as the visible throat.  The front half follows the measured jaw;
    # the back half is deliberately tucked under the hair volume.
    jaw_radius_x = max(abs(point.x) for point in jaw) * 1.10
    center_y = 0.034
    back_y = -0.020
    top_ring: list[Vector] = []
    for column in range(NECK_COLUMNS):
        theta = 2.0 * math.pi * column / NECK_COLUMNS
        x = jaw_radius_x * math.cos(theta)
        if math.sin(theta) >= -0.02:
            point = jaw_at_x(x)
            # Gently extend the jaw endpoints to the wider side ring.
            point.x = x
            top = point
        else:
            top = Vector((x, back_y + 0.040 * math.sin(theta), 1.638))
        top_ring.append(top)

    skin_bvh, body_coordinates, _triangles = body_skin_surface(body)
    skin_ids = material_vertices(body, SKIN_MATERIAL)
    skin_tree = KDTree(len(skin_ids))
    for index in skin_ids:
        skin_tree.insert(body_coordinates[index], index)
    skin_tree.balance()
    lower: list[Vector] = []
    lower_indices: list[int] = []
    gaps: list[float] = []
    for point in top_ring:
        nearest, normal, _polygon, distance = skin_bvh.find_nearest(point)
        if nearest is None or distance is None:
            raise RuntimeError("Could not project the neck ring to the body skin")
        direction = (point - nearest).normalized() if distance > 1.0e-8 else Vector((0.0, 1.0, 0.0))
        # Deliberately overlap the existing body surface so the separate FH6
        # component draw cannot expose a dark slit at the lower edge.
        lower_point = nearest - direction * 0.0050
        lower.append(lower_point)
        _co, nearest_index, _nearest_distance = skin_tree.find(nearest)
        lower_indices.append(nearest_index)
        gaps.append(float(distance))
    # Smooth the body projection while preserving the ring's overall shape.
    for _iteration in range(3):
        lower = [
            lower[(column - 1) % NECK_COLUMNS].lerp(lower[(column + 1) % NECK_COLUMNS], 0.5)
            for column in range(NECK_COLUMNS)
        ]

    inverse_world = body.matrix_world.inverted()
    vertices: list[tuple[float, float, float]] = []
    vertex_weights: list[dict[str, float]] = []
    allowed_bones = {bone.name for bone in armature.data.bones}
    for row in range(NECK_ROWS):
        factor = row / (NECK_ROWS - 1)
        smooth = factor * factor * (3.0 - 2.0 * factor)
        for column in range(NECK_COLUMNS):
            point = lower[column].lerp(top_ring[column], smooth)
            if row == NECK_ROWS - 1:
                center = Vector((0.0, 0.040, 1.625))
                inward = center - point
                if inward.length > 1.0e-8:
                    point += inward.normalized() * 0.0008
            vertices.append(tuple(inverse_world @ point))
            base = clean_weights(weight_dict(body, lower_indices[column], allowed_bones))
            if row == 0:
                weights = base
            elif row == 1:
                weights = blend_weights(base, {"Neck1": 1.0}, 0.35)
            elif row == 2:
                weighted_base = {name: weight * 0.25 for name, weight in base.items()}
                weighted_base["Neck1"] = weighted_base.get("Neck1", 0.0) + 0.50
                weighted_base["Head"] = weighted_base.get("Head", 0.0) + 0.25
                weights = clean_weights(weighted_base)
            else:
                weights = {"Neck1": 0.20, "Head": 0.80}
            vertex_weights.append(clean_weights(weights))

    faces: list[tuple[int, int, int]] = []
    for row in range(NECK_ROWS - 1):
        for column in range(NECK_COLUMNS):
            next_column = (column + 1) % NECK_COLUMNS
            lower_left = row * NECK_COLUMNS + column
            lower_right = row * NECK_COLUMNS + next_column
            upper_left = lower_left + NECK_COLUMNS
            upper_right = (row + 1) * NECK_COLUMNS + next_column
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    mesh_data = bpy.data.meshes.new("Si_Neck_Bridge_LOD0_Mesh")
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.materials.append(bpy.data.materials[SKIN_MATERIAL])
    uv_layer = mesh_data.uv_layers.new(name="UVMap")
    for polygon in mesh_data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True
        for loop_index in polygon.loop_indices:
            vertex_index = mesh_data.loops[loop_index].vertex_index
            row, column = divmod(vertex_index, NECK_COLUMNS)
            uv_layer.data[loop_index].uv = (column / NECK_COLUMNS, row / (NECK_ROWS - 1))
    patch = bpy.data.objects.new("Si_Neck_Bridge_LOD0", mesh_data)
    bpy.context.scene.collection.objects.link(patch)
    patch.matrix_world = body.matrix_world.copy()
    for vertex_index, weights in enumerate(vertex_weights):
        for name, weight in weights.items():
            group = patch.vertex_groups.get(name) or patch.vertex_groups.new(name=name)
            group.add([vertex_index], weight, "REPLACE")
    modifier = patch.modifiers.new(name="FH6 Outfit Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = False
    patch["fh6_neck_bridge"] = True
    patch["fh6_probe_exclude"] = False

    before_vertices = len(body.data.vertices)
    before_polygons = len(body.data.polygons)
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    patch.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    body.data.update()
    body["fh6_neck_bridge_vertex_start"] = before_vertices
    body["fh6_neck_bridge_vertex_count"] = len(vertex_weights)
    body["fh6_neck_bridge_columns"] = NECK_COLUMNS
    body["fh6_neck_bridge_rows"] = NECK_ROWS
    return {
        "method": "closed four-row loft from projected body skin to measured lower-face boundary",
        "jaw": jaw_report,
        "columns": NECK_COLUMNS,
        "rows": NECK_ROWS,
        "added_vertices": len(body.data.vertices) - before_vertices,
        "added_polygons": len(body.data.polygons) - before_polygons,
        "source_gap_mm": {
            "min": round(min(gaps) * 1000.0, 6),
            "mean": round(sum(gaps) / len(gaps) * 1000.0, 6),
            "p95": round(float(percentile(gaps, 0.95)) * 1000.0, 6),
            "max": round(max(gaps) * 1000.0, 6),
        },
        "weight_rows": ["projected body", "body/Neck1", "body/Neck1/Head", "Neck1/Head"],
    }


def validate_weights(obj: bpy.types.Object, armature: bpy.types.Object) -> dict[str, object]:
    allowed = {bone.name for bone in armature.data.bones}
    histogram: Counter[int] = Counter()
    sums: list[float] = []
    unresolved: set[str] = set()
    zero = 0
    over = 0
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
            sums.append(sum(weights))
    return {
        "influence_histogram": dict(sorted(histogram.items())),
        "zero_weight_vertices": zero,
        "vertices_over_limit": over,
        "unresolved_groups": sorted(unresolved),
        "weight_sum_min": min(sums) if sums else 0.0,
        "weight_sum_max": max(sums) if sums else 0.0,
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
    head_objects = append_objects(head_blend, {HEAD_MESH})
    head = head_objects[0]
    neck_report = add_neck_bridge(body, armature, head)

    for obj in [*donors, *head_objects]:
        bpy.data.objects.remove(obj, do_unlink=True)
    body["fh6_joint_repair"] = "female-donor-barycentric-v001"
    body["fh6_neck_bridge"] = "under-jaw-open-loft-v001"
    validation = validate_weights(body, armature)
    if validation["zero_weight_vertices"] or validation["vertices_over_limit"] or validation["unresolved_groups"]:
        raise RuntimeError(f"Body repair weight validation failed: {validation}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=False, check_existing=False)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Si FBX Display LOD0 body skin weight and under-jaw neck repair.",
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
        "neck_bridge": neck_report,
        "validation_level": {"data": True, "blender_visual": False, "modelbin": False, "offline_game": False},
        "license_guard": "Local technical validation only; do not redistribute the source character.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_SI_BODY_REPAIR=" + json.dumps({"blend": str(output), "vertices": len(body.data.vertices), "changed_weights": transfer_report["changed_vertices"], "neck_vertices": neck_report["added_vertices"]}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
