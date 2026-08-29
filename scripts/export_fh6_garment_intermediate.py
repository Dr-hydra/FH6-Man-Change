#!/usr/bin/env python3
"""Export an auditable, engine-neutral garment geometry/skin intermediate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


VERTEX_FLOATS = 16
VERTEX_STRIDE = VERTEX_FLOATS * 4
BONE_INDEX_STRIDE = 8


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--object", required=True, help="Mesh object to export")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--vertices", required=True, type=Path)
    parser.add_argument("--bone-indices", required=True, type=Path)
    parser.add_argument("--indices", required=True, type=Path)
    parser.add_argument(
        "--draw-policy",
        choices=(
            "single",
            "body2",
            "racesuit8",
            "racesuit8_component",
            "helmet6",
            "head4",
            "head5_display",
            "head6_display",
            "head7_display",
            "head8_f04_skin",
            "driver_body6",
        ),
        default="single",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_absent(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)


def armature_for(mesh: bpy.types.Object) -> bpy.types.Object:
    candidates = [
        modifier.object
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object is not None
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one armature modifier on {mesh.name!r}, found {len(candidates)}")
    return candidates[0]


def bone_order(armature: bpy.types.Object) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for bone in armature.data.bones:
        if "fh6_index" not in bone:
            raise RuntimeError(f"Donor bone {bone.name!r} lacks fh6_index")
        indexed.append((int(bone["fh6_index"]), bone.name))
    indexed.sort()
    expected = list(range(len(indexed)))
    actual = [index for index, _ in indexed]
    if actual != expected:
        raise RuntimeError("Donor fh6_index values are not contiguous in skeleton order")
    return [name for _, name in indexed]


def vertex_weights(
    mesh: bpy.types.Object,
    names_to_indices: dict[str, int],
) -> list[tuple[tuple[float, float, float, float], tuple[int, int, int, int]]]:
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    result = []
    for vertex in mesh.data.vertices:
        assignments: list[tuple[int, float]] = []
        for item in vertex.groups:
            name = group_names.get(item.group)
            if name is None or item.weight <= 0.0:
                continue
            if name not in names_to_indices:
                raise RuntimeError(f"Vertex group {name!r} is absent from donor skeleton")
            assignments.append((names_to_indices[name], float(item.weight)))
        assignments.sort(key=lambda item: (-item[1], item[0]))
        if not assignments:
            raise RuntimeError(f"Vertex {vertex.index} has no donor weight")
        if len(assignments) > 4:
            raise RuntimeError(f"Vertex {vertex.index} has {len(assignments)} influences")
        total = sum(weight for _, weight in assignments)
        if not 0.999 <= total <= 1.001:
            raise RuntimeError(f"Vertex {vertex.index} weights sum to {total}")
        normalized = [(index, weight / total) for index, weight in assignments]
        normalized.extend([(0, 0.0)] * (4 - len(normalized)))
        result.append(
            (
                tuple(weight for _, weight in normalized),
                tuple(index for index, _ in normalized),
            )
        )
    return result


def quantized_key(original_vertex: int, uv: Vector, normal: Vector) -> tuple[int, ...]:
    factor = 10_000_000.0
    return (
        original_vertex,
        int(round(uv.x * factor)),
        int(round(uv.y * factor)),
        int(round(normal.x * factor)),
        int(round(normal.y * factor)),
        int(round(normal.z * factor)),
    )


def racesuit_draw_id(material_name: str, centroid: Vector) -> int:
    """Assign one source material to each retained race-suit draw.

    FH6 binds one MatI material to each Mesh draw. Source materials therefore
    must never be mixed in a draw, even when extra spatial subdivisions are
    needed to keep all eight donor draws populated.
    """

    if material_name == "Cloth2":
        return 2
    if material_name == "Cloth1Alpha":
        return 6
    if material_name == "肌":
        if abs(centroid.x) > 0.35 and centroid.z > 0.75:
            return 5  # Hands
        return 1  # Remaining exposed body surface
    if material_name != "Cloth1":
        raise RuntimeError(f"Unhandled source material in racesuit8 policy: {material_name!r}")
    if centroid.z < 0.22:
        return 7  # Shoes and low ornaments
    if abs(centroid.x) > 0.18 and centroid.z > 1.15:
        return 4  # Shoulders and upper sleeves
    if abs(centroid.x) > 0.32 and 0.75 < centroid.z <= 1.15:
        return 3  # Cuffs and outer sleeves
    return 0  # Main suit, skirt, center ornaments, and tail


def component_draw_id(policy: str, material_name: str, centroid: Vector) -> int:
    if policy == "body2":
        # Female donor draw 1 is the arm/hand pass; legs and torso use draw 0.
        return 1 if abs(centroid.x) > 0.18 and centroid.z > 0.75 else 0
    if policy == "racesuit8_component":
        if material_name == "肌":
            return 5 if abs(centroid.x) > 0.35 and centroid.z > 0.75 else 1
        return racesuit_draw_id(material_name, centroid)
    if policy == "helmet6":
        # Helmet donor material IDs: 0 helmet, 1 rubber, 2 vent, 3 padding,
        # 4 bits, 5 visor. Source hair and head-accessory islands are assigned
        # to stable passes; runtime materials can then provide the textures.
        if material_name == "发影":
            return 5
        if material_name == "发":
            return 0
        if material_name == "Cloth2":
            return 4
        if material_name == "Cloth1":
            if centroid.z >= 1.58:
                return 1
            if centroid.z >= 1.50:
                return 2
            return 3
        raise RuntimeError(f"Unhandled source material in helmet6 policy: {material_name!r}")
    if policy == "head4":
        # DRV_BA face donor IDs: 0 eyelashes, 1 head, 2 teeth/mouth,
        # 3 eyes. Keep all source eye passes together in the rigid eye draw.
        if material_name == "睫眉":
            return 0
        if material_name == "口内":
            return 2
        if material_name in {"目", "目HL", "目白", "目影"}:
            return 3
        if material_name in {"面", "肌", "表情"}:
            return 1
        raise RuntimeError(f"Unhandled source material in head4 policy: {material_name!r}")
    if policy == "head5_display":
        # The Helmet display model has only four available face-related draws.
        # Keep stable, texture-compatible layers separate and park animation-
        # dependent overlays in draw 0, which the combiner deliberately omits.
        if material_name in {"表情", "口内", "目白", "目影"}:
            return 0
        if material_name in {"目", "目HL"}:
            return 1
        if material_name == "面":
            return 2
        if material_name == "肌":
            return 3
        if material_name == "睫眉":
            return 4
        raise RuntimeError(f"Unhandled source material in head5_display policy: {material_name!r}")
    if policy == "head6_display":
        # FBX-first Helmet package: keep hair, face/eye layers, eyelashes and
        # hair shadow on six independent donor draws.  The source has no
        # dedicated sclera mesh, so its eye-shadow plane occupies draw 1 until
        # the material stage supplies the opaque sclera contract.
        if material_name == "发":
            return 0
        if material_name in {"表情", "口内", "目白", "目影"}:
            return 1
        if material_name == "面":
            return 2
        if material_name in {"目", "目HL"}:
            return 3
        if material_name == "睫眉":
            return 4
        if material_name in {"肌", "发影"}:
            return 5
        raise RuntimeError(f"Unhandled source material in head6_display policy: {material_name!r}")
    if policy == "head7_display":
        # Final FBX-first Helmet package. The v007/v008 eye stage adds a
        # dedicated generated sclera material, so it must not be folded into
        # the iris or eye-shadow draw.
        if material_name == "发":
            return 0
        if material_name in {"表情", "口内", "目白", "目影"}:
            return 1
        if material_name == "面":
            return 2
        if material_name in {"目", "目HL"}:
            return 3
        if material_name == "睫眉":
            return 4
        if material_name in {"肌", "发影"}:
            return 5
        if material_name == "巩膜":
            return 6
        raise RuntimeError(f"Unhandled source material in head7_display policy: {material_name!r}")
    if policy == "head8_f04_skin":
        # F04 carries explicit per-face fh6_draw_id values. This fallback only
        # exists to make a missing override fail with a useful material role.
        if material_name == "TR_Hair":
            return 0
        if material_name == "TR_EyeShadow":
            return 1
        if material_name == "TR_Face_Legacy_Details_F02":
            return 2
        if material_name == "TR_EyeSpecular":
            return 3
        if material_name == "TR_Face_FH6_Skin_F04":
            return 6
        raise RuntimeError(f"Unhandled source material in head8_f04_skin policy: {material_name!r}")
    if policy == "driver_body6":
        # Driver_Alice_F MatI IDs: 0 head, 1 eyelashes, 2 eyes,
        # 3 body, 4 arms/hands, and 5 teeth. The writer expands draw 1 into
        # the donor's separate front/back eyelash Mesh descriptors.
        if material_name == "睫眉":
            return 1
        if material_name == "口内":
            return 5
        if material_name in {"目", "目HL", "目白", "目影"}:
            return 2
        if material_name in {"面", "肌", "表情"}:
            return 0
        if material_name.startswith("DriverBody_Skin"):
            return 4 if abs(centroid.x) > 0.18 and centroid.z > 0.75 else 3
        raise RuntimeError(f"Unhandled source material in driver_body6 policy: {material_name!r}")
    raise RuntimeError(f"Unhandled component draw policy: {policy!r}")


def fill_empty_draws(
    triangle_draws: list[tuple[int, int, str]],
    draw_count: int,
    preferred_material: dict[int, str | None],
) -> list[tuple[int, list[int], str]]:
    """Populate donor passes with the smallest possible same-material triangles."""
    by_draw: dict[int, list[int]] = {draw: [] for draw in range(draw_count)}
    for index, (draw_id, _, material_name) in enumerate(triangle_draws):
        by_draw[draw_id].append(index)
    for empty_draw in [draw for draw, indices in by_draw.items() if not indices]:
        required_material = preferred_material.get(empty_draw)
        candidates = [
            index
            for index, (draw_id, _, material_name) in enumerate(triangle_draws)
            if draw_id != empty_draw
            and len(by_draw[draw_id]) > 1
            and (required_material is None or material_name == required_material)
        ]
        if not candidates:
            raise RuntimeError(f"Unable to populate draw {empty_draw} with a same-material triangle")
        source_index = candidates[0]
        source_draw, source_triangle, material_name = triangle_draws[source_index]
        triangle_draws[source_index] = (empty_draw, source_triangle, material_name)
        by_draw[empty_draw].append(source_index)
        by_draw[source_draw].remove(source_index)
    return triangle_draws


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    outputs = {
        "manifest": args.manifest.resolve(),
        "vertices": args.vertices.resolve(),
        "bone_indices": args.bone_indices.resolve(),
        "indices": args.indices.resolve(),
    }
    ensure_absent(list(outputs.values()))
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match --source-blend {source_blend}")

    obj = bpy.data.objects.get(args.object)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"Mesh object {args.object!r} not found")
    if any(scale <= 0.0 for scale in obj.scale):
        raise RuntimeError(f"Mesh has non-positive scale {tuple(obj.scale)}")
    if not obj.data.uv_layers:
        raise RuntimeError("Mesh has no UV layer")

    armature = armature_for(obj)
    bones = bone_order(armature)
    bone_indices = {name: index for index, name in enumerate(bones)}
    source_weights = vertex_weights(obj, bone_indices)

    mesh = obj.data
    mesh.calc_loop_triangles()
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("Mesh has no active UV layer")
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()

    material_names = [material.name if material else f"Material_{index}" for index, material in enumerate(mesh.materials)]
    draw_override = mesh.attributes.get("fh6_draw_id")
    if draw_override is not None and (
        draw_override.domain != "FACE" or draw_override.data_type != "INT"
    ):
        raise RuntimeError("fh6_draw_id must be a FACE/INT attribute")
    # Blender can retain an empty FACE attribute after manual face deletion.
    # Treat that stale record as absent so the documented spatial/material
    # fallback policy remains authoritative for the surviving polygons.
    if draw_override is not None and len(draw_override.data) != len(mesh.polygons):
        if len(draw_override.data) == 0:
            draw_override = None
        else:
            raise RuntimeError(
                "fh6_draw_id length does not match surviving polygon count: "
                f"{len(draw_override.data)} != {len(mesh.polygons)}"
            )

    draw_counts = {
        "single": 1,
        "body2": 2,
        "racesuit8": 8,
        "racesuit8_component": 8,
        "helmet6": 6,
        "head4": 4,
        "head5_display": 5,
        "head6_display": 6,
        "head7_display": 7,
        "head8_f04_skin": 8,
        "driver_body6": 6,
    }
    draw_count = draw_counts[args.draw_policy]
    preferred_material = {
        "body2": {0: "肌", 1: "肌"},
        "racesuit8": {5: "肌"},
        "racesuit8_component": {1: "Cloth1", 5: "Cloth1"},
        "helmet6": {0: "发", 4: "Cloth2", 5: "发影"},
        "head4": {0: "睫眉", 1: "面", 2: "口内", 3: "目"},
        "head5_display": {0: "表情", 1: "目", 2: "面", 3: "肌", 4: "睫眉"},
        "head6_display": {0: "发", 1: "目影", 2: "面", 3: "目", 4: "睫眉", 5: "发影"},
        "head7_display": {0: "发", 1: "目影", 2: "面", 3: "目", 4: "睫眉", 5: "发影", 6: "巩膜"},
        "head8_f04_skin": {0: "TR_Hair", 5: "TR_Hair", 7: "TR_Hair"},
        "driver_body6": {
            0: "面",
            1: "睫眉",
            2: "目",
            3: "DriverBody_Skin",
            4: "DriverBody_Skin",
            5: "口内",
        },
    }.get(args.draw_policy, {})
    triangle_draws: list[tuple[int, int, str]] = []
    for triangle in mesh.loop_triangles:
        polygon = mesh.polygons[triangle.polygon_index]
        material_index = int(polygon.material_index)
        if material_index < 0 or material_index >= len(material_names):
            raise RuntimeError(f"Triangle {triangle.index} has invalid material slot {material_index}")
        centroid = sum(
            (obj.matrix_world @ mesh.vertices[mesh.loops[loop_index].vertex_index].co for loop_index in triangle.loops),
            Vector((0.0, 0.0, 0.0)),
        ) / 3.0
        material_name = material_names[material_index]
        override = int(draw_override.data[polygon.index].value) if draw_override is not None else -1
        if override >= 0:
            if args.draw_policy == "single" or override >= draw_count:
                raise RuntimeError(
                    f"Triangle {triangle.index} has invalid fh6_draw_id={override} for {args.draw_policy}"
                )
            draw_id = override
        elif args.draw_policy == "single":
            draw_id = 0
        else:
            draw_id = component_draw_id(args.draw_policy, material_name, centroid)
        triangle_draws.append((draw_id, triangle.index, material_name))
    if args.draw_policy != "single":
        triangle_draws = fill_empty_draws(triangle_draws, draw_count, preferred_material)

    output_positions: list[Vector] = []
    output_normals: list[Vector] = []
    output_uvs: list[Vector] = []
    output_source_vertices: list[int] = []
    draw_material_histograms: list[Counter[str]] = [Counter() for _ in range(draw_count)]
    key_to_output: dict[tuple[int, ...], int] = {}

    corner_normals = mesh.corner_normals
    for draw_id, triangle_index, material_name in triangle_draws:
        triangle = mesh.loop_triangles[triangle_index]
        polygon = mesh.polygons[triangle.polygon_index]
        draw_material_histograms[draw_id][material_name] += 1
        triangle_indices: list[int] = []
        for loop_index in triangle.loops:
            loop = mesh.loops[loop_index]
            source_vertex = int(loop.vertex_index)
            uv = Vector(uv_layer.data[loop_index].uv)
            local_normal = Vector(corner_normals[loop_index].vector)
            if local_normal.length < 1e-4:
                local_normal = Vector(polygon.normal) if Vector(polygon.normal).length > 1e-4 else Vector((0.0, 0.0, 1.0))
            world_normal = (normal_matrix @ local_normal).normalized()
            # A source corner may be used by multiple material draws. Each draw
            # needs its own dense vertex domain in FH6 Mesh descriptors.
            base_key = quantized_key(source_vertex, uv, world_normal)
            key = (draw_id,) + base_key
            output_index = key_to_output.get(key)
            if output_index is None:
                output_index = len(output_positions)
                key_to_output[key] = output_index
                output_positions.append(obj.matrix_world @ mesh.vertices[source_vertex].co)
                output_normals.append(world_normal)
                output_uvs.append(uv.copy())
                output_source_vertices.append(source_vertex)
            triangle_indices.append(output_index)
        if len(set(triangle_indices)) != 3:
            raise RuntimeError(f"Degenerate triangle {triangle.index} after corner deduplication")
        triangle_draws[triangle_index] = (draw_id, triangle_indices, material_name)

    indices_by_draw: list[list[int]] = [[] for _ in range(draw_count)]
    for draw_id, triangle_indices, _ in triangle_draws:
        indices_by_draw[draw_id].extend(triangle_indices)
    empty_draws = [index for index, values in enumerate(indices_by_draw) if not values]
    if empty_draws:
        raise RuntimeError(f"Draw policy {args.draw_policy!r} produced empty target draws: {empty_draws}")

    # Reorder the globally deduplicated staging vertices into independent,
    # contiguous draw domains. The per-draw key above guarantees that a vertex
    # cannot be shared between material passes; this remap makes that contract
    # explicit in the emitted buffers.
    old_to_new: dict[int, int] = {}
    reordered_positions: list[Vector] = []
    reordered_normals: list[Vector] = []
    reordered_uvs: list[Vector] = []
    reordered_sources: list[int] = []
    draw_vertex_ranges: list[tuple[int, int]] = []
    for draw_id, draw_indices in enumerate(indices_by_draw):
        old_vertices = sorted(set(draw_indices))
        vertex_start = len(reordered_positions)
        for old_index in old_vertices:
            old_to_new[old_index] = len(reordered_positions)
            reordered_positions.append(output_positions[old_index])
            reordered_normals.append(output_normals[old_index])
            reordered_uvs.append(output_uvs[old_index])
            reordered_sources.append(output_source_vertices[old_index])
        draw_vertex_ranges.append((vertex_start, len(old_vertices)))
    if len(old_to_new) != len(output_positions):
        raise RuntimeError("Some staging vertices are not referenced by a draw")
    output_positions = reordered_positions
    output_normals = reordered_normals
    output_uvs = reordered_uvs
    output_source_vertices = reordered_sources
    for draw_id, draw_indices in enumerate(indices_by_draw):
        indices_by_draw[draw_id] = [old_to_new[index] for index in draw_indices]

    draw_ranges: list[dict[str, object]] = []
    indices: list[int] = []
    for material_id, draw_indices in enumerate(indices_by_draw):
        start_index = len(indices)
        indices.extend(draw_indices)
        vertex_start, vertex_count = draw_vertex_ranges[material_id]
        expected_vertices = list(range(vertex_start, vertex_start + vertex_count))
        if sorted(set(draw_indices)) != expected_vertices:
            raise RuntimeError(
                f"Draw {material_id} does not occupy a dense contiguous vertex range: "
                f"min={min(draw_indices, default=-1)} max={max(draw_indices, default=-1)} "
                f"count={len(set(draw_indices))} expected_start={vertex_start}"
            )
        draw_ranges.append({
            "material_id": material_id,
            "start_index": start_index,
            "index_count": len(draw_indices),
            "triangles": len(draw_indices) // 3,
            "vertex_start": vertex_start,
            "vertex_count": vertex_count,
            "unique_vertex_count": vertex_count,
            "source_material_histogram": dict(sorted(draw_material_histograms[material_id].items())),
            "source_material_names": sorted(draw_material_histograms[material_id]),
        })

    if args.draw_policy in {"racesuit8", "racesuit8_component", "helmet6", "body2"}:
        mixed_draws = [
            draw["material_id"]
            for draw in draw_ranges
            if len(draw["source_material_names"]) != 1
        ]
        if mixed_draws:
            raise RuntimeError(f"racesuit8 produced mixed source materials in draws: {mixed_draws}")

    if sum(int(draw["vertex_count"]) for draw in draw_ranges) != len(output_positions):
        raise RuntimeError("Draw vertex ranges do not cover the complete vertex buffer")
    for draw in draw_ranges:
        vertex_start = int(draw["vertex_start"])
        vertex_end = vertex_start + int(draw["vertex_count"])
        start_index = int(draw["start_index"])
        index_end = start_index + int(draw["index_count"])
        if any(index < vertex_start or index >= vertex_end for index in indices[start_index:index_end]):
            raise RuntimeError(f"Draw {draw['material_id']} index resolves outside its vertex range")

    if len(output_positions) > 65_535:
        raise RuntimeError(f"Export domain has {len(output_positions)} vertices, exceeding R16_UINT")

    tangent_accum = [Vector((0.0, 0.0, 0.0)) for _ in output_positions]
    bitangent_accum = [Vector((0.0, 0.0, 0.0)) for _ in output_positions]
    degenerate_uv_triangles = 0
    for offset in range(0, len(indices), 3):
        ia, ib, ic = indices[offset : offset + 3]
        p0, p1, p2 = output_positions[ia], output_positions[ib], output_positions[ic]
        uv0, uv1, uv2 = output_uvs[ia], output_uvs[ib], output_uvs[ic]
        edge1 = p1 - p0
        edge2 = p2 - p0
        delta1 = uv1 - uv0
        delta2 = uv2 - uv0
        determinant = delta1.x * delta2.y - delta1.y * delta2.x
        if abs(determinant) <= 1e-12:
            degenerate_uv_triangles += 1
            continue
        reciprocal = 1.0 / determinant
        tangent = (edge1 * delta2.y - edge2 * delta1.y) * reciprocal
        bitangent = (edge2 * delta1.x - edge1 * delta2.x) * reciprocal
        for index in (ia, ib, ic):
            tangent_accum[index] += tangent
            bitangent_accum[index] += bitangent

    tangents: list[tuple[float, float, float, float]] = []
    tangent_handedness: Counter[int] = Counter()
    tangent_fallbacks = 0
    for index, normal in enumerate(output_normals):
        tangent = tangent_accum[index] - normal * normal.dot(tangent_accum[index])
        if tangent.length_squared <= 1e-16:
            tangent_fallbacks += 1
            reference = Vector((0.0, 0.0, 1.0)) if abs(normal.z) < 0.9 else Vector((0.0, 1.0, 0.0))
            tangent = normal.cross(reference)
        tangent.normalize()
        handedness = -1.0 if normal.cross(tangent).dot(bitangent_accum[index]) < 0.0 else 1.0
        tangent_handedness[int(handedness)] += 1
        tangents.append((tangent.x, tangent.y, tangent.z, handedness))

    vertex_payload = bytearray()
    bone_payload = bytearray()
    used_bone_names: set[str] = set()
    influence_histogram: Counter[int] = Counter()
    weight_sum_min = math.inf
    weight_sum_max = -math.inf
    for position, normal, tangent, uv, source_vertex in zip(
        output_positions,
        output_normals,
        tangents,
        output_uvs,
        output_source_vertices,
    ):
        weights, vertex_bones = source_weights[source_vertex]
        nonzero = sum(1 for weight in weights if weight > 0.0)
        influence_histogram[nonzero] += 1
        total = sum(weights)
        weight_sum_min = min(weight_sum_min, total)
        weight_sum_max = max(weight_sum_max, total)
        for weight, bone_index in zip(weights, vertex_bones):
            if weight > 0.0:
                used_bone_names.add(bones[bone_index])
        vertex_payload.extend(
            struct.pack(
                "<16f",
                position.x,
                position.y,
                position.z,
                normal.x,
                normal.y,
                normal.z,
                tangent[0],
                tangent[1],
                tangent[2],
                tangent[3],
                uv.x,
                uv.y,
                weights[0],
                weights[1],
                weights[2],
                weights[3],
            )
        )
        bone_payload.extend(struct.pack("<4H", *vertex_bones))

    index_payload = struct.pack(f"<{len(indices)}H", *indices)
    outputs["vertices"].write_bytes(vertex_payload)
    outputs["bone_indices"].write_bytes(bone_payload)
    outputs["indices"].write_bytes(index_payload)

    minimum = Vector(tuple(min(position[axis] for position in output_positions) for axis in range(3)))
    maximum = Vector(tuple(max(position[axis] for position in output_positions) for axis in range(3)))
    uv_minimum = Vector((min(uv.x for uv in output_uvs), min(uv.y for uv in output_uvs)))
    uv_maximum = Vector((max(uv.x for uv in output_uvs), max(uv.y for uv in output_uvs)))
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Engine-neutral debug intermediate; no modelbin encoding is embedded here.",
        "source": {
            "blend": str(source_blend),
            "blend_sha256": sha256(source_blend),
            "object": obj.name,
            "mesh": mesh.name,
            "armature": armature.name,
        },
        "geometry": {
            "source_vertices": len(mesh.vertices),
            "export_vertices": len(output_positions),
            "corner_split_vertices_added": len(output_positions) - len(mesh.vertices),
            "triangles": len(indices) // 3,
            "indices": len(indices),
            "maximum_index": max(indices),
            "bounds_min": list(minimum),
            "bounds_max": list(maximum),
            "uv_min": list(uv_minimum),
            "uv_max": list(uv_maximum),
            "degenerate_uv_triangles": degenerate_uv_triangles,
            "tangent_fallback_vertices": tangent_fallbacks,
            "tangent_handedness": dict(sorted(tangent_handedness.items())),
            "draw_policy": args.draw_policy,
            "draws": draw_ranges if args.draw_policy != "single" else None,
        },
        "skinning": {
            "skeleton_bones": len(bones),
            "bone_order": bones,
            "used_bones": sorted(used_bone_names),
            "used_bone_count": len(used_bone_names),
            "influence_histogram": dict(sorted(influence_histogram.items())),
            "weight_sum_min": weight_sum_min,
            "weight_sum_max": weight_sum_max,
            "max_influences": 4,
        },
        "files": {
            "vertices": {
                "path": str(outputs["vertices"]),
                "sha256": sha256(outputs["vertices"]),
                "bytes": len(vertex_payload),
                "stride": VERTEX_STRIDE,
                "encoding": "little-endian float32",
                "fields": [
                    {"name": "position", "offset": 0, "components": 3},
                    {"name": "normal", "offset": 12, "components": 3},
                    {"name": "tangent", "offset": 24, "components": 4},
                    {"name": "uv0", "offset": 40, "components": 2},
                    {"name": "weights", "offset": 48, "components": 4},
                ],
            },
            "bone_indices": {
                "path": str(outputs["bone_indices"]),
                "sha256": sha256(outputs["bone_indices"]),
                "bytes": len(bone_payload),
                "stride": BONE_INDEX_STRIDE,
                "encoding": "little-endian uint16[4]",
            },
            "indices": {
                "path": str(outputs["indices"]),
                "sha256": sha256(outputs["indices"]),
                "bytes": len(index_payload),
                "stride": 2,
                "encoding": "little-endian uint16",
                "winding": "Blender; the modelbin encoder must swap triangle indices 1 and 2 for the Forza->Blender handedness conversion",
            },
        },
        "material_policy": {
            "source_material_slots": len(mesh.materials),
            "source_material_names": material_names,
            "target_draws": draw_count,
            "first_writer_target": "retain donor MatI and populate every target Mesh with one contiguous, non-overlapping draw range",
        },
        "lod_policy": "LODS and LOD0 initially share this geometry/index domain; no decimation in the first structural candidate.",
        "license_guard": "Local technical validation only; do not redistribute these derived buffers.",
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_GARMENT_INTERMEDIATE="
        + json.dumps(
            {
                "manifest": str(outputs["manifest"]),
                "source_vertices": len(mesh.vertices),
                "export_vertices": len(output_positions),
                "triangles": len(indices) // 3,
                "indices": len(indices),
                "maximum_index": max(indices),
                "used_bones": len(used_bone_names),
                "degenerate_uv_triangles": degenerate_uv_triangles,
                "tangent_fallback_vertices": tangent_fallbacks,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
