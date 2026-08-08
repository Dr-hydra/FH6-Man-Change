#!/usr/bin/env python3
"""Build a continuous socket-derived sclera patch for Si Display LOD0.

This supersedes the v006 experiment, whose sclera duplicated disconnected iris
highlight islands.  The v007 geometry uses one native FBX socket boundary per
eye, creates three concentric rings plus a center cap, and translates every
iris Shape Key by the same REST correction so facial morph deltas are kept.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_si_fbx_display_eye_stage_v006 as common
from retarget_fbx_display_components import verify_weights


OUTER_RECESS_M = 0.00035
RING_SCALES = (1.0, 0.62, 0.28)


def translated_shape_keys(mesh: bpy.types.Object, indices: set[int], local_delta: Vector) -> int:
    shape_keys = mesh.data.shape_keys
    if shape_keys is None:
        return 0
    for key_block in shape_keys.key_blocks:
        for index in indices:
            key_block.data[index].co += local_delta
    return len(shape_keys.key_blocks)


def sorted_socket_points(indices: set[int], coordinates: list[Vector]) -> list[Vector]:
    center = common.centroid(coordinates[index] for index in indices)
    return sorted(
        (coordinates[index] for index in indices),
        key=lambda point: math.atan2(point.z - center.z, point.x - center.x),
    )


def ring_patch(
    socket_points: list[Vector],
    iris_points: list[Vector],
    inverse_world,
    depth_m: float,
) -> tuple[list[Vector], list[list[int]], list[tuple[float, float]], dict[str, object]]:
    if len(socket_points) < 12:
        raise RuntimeError(f"Socket ring is too small: {len(socket_points)}")
    socket_center = common.centroid(socket_points)
    minimum, maximum = common.bounds(socket_points)
    half_width = max((maximum.x - minimum.x) * 0.5, 1e-6)
    half_height = max((maximum.z - minimum.z) * 0.5, 1e-6)
    back_y = min(point.y for point in iris_points) - depth_m
    vertices_world: list[Vector] = []
    rings: list[list[int]] = []
    uvs: list[tuple[float, float]] = []

    for ring_index, scale in enumerate(RING_SCALES):
        ring: list[int] = []
        for socket_point in socket_points:
            if ring_index == 0:
                y = socket_point.y - OUTER_RECESS_M
            elif ring_index == 1:
                y = socket_point.y * 0.35 + back_y * 0.65
            else:
                y = back_y
            point = Vector((
                socket_center.x + (socket_point.x - socket_center.x) * scale,
                y,
                socket_center.z + (socket_point.z - socket_center.z) * scale,
            ))
            ring.append(len(vertices_world))
            vertices_world.append(point)
            uvs.append((
                max(0.0, min(1.0, 0.5 + (point.x - socket_center.x) / (2.0 * half_width))),
                max(0.0, min(1.0, 0.5 + (point.z - socket_center.z) / (2.0 * half_height))),
            ))
        rings.append(ring)

    center_index = len(vertices_world)
    vertices_world.append(Vector((socket_center.x, back_y, socket_center.z)))
    uvs.append((0.5, 0.5))
    faces: list[list[int]] = []
    count = len(socket_points)
    for first_ring, second_ring in zip(rings, rings[1:]):
        for index in range(count):
            following = (index + 1) % count
            # +Y-facing winding for rings laid out in the X/Z plane.
            faces.append([
                first_ring[index],
                second_ring[index],
                second_ring[following],
                first_ring[following],
            ])
    inner = rings[-1]
    for index in range(count):
        following = (index + 1) % count
        faces.append([inner[index], center_index, inner[following]])
    return (
        [inverse_world @ point for point in vertices_world],
        faces,
        uvs,
        {
            "socket_vertices": count,
            "rings": len(RING_SCALES),
            "vertices": len(vertices_world),
            "polygons": len(faces),
            "socket_centroid_m": common.vector_record(socket_center),
            "back_plane_y_m": round(back_y, 9),
            "outer_recess_mm": round(OUTER_RECESS_M * 1000.0, 6),
            "ring_scales": list(RING_SCALES),
        },
    )


def main() -> None:
    args = common.arguments()
    input_blend = args.input_blend.resolve()
    output_blend = args.output_blend.resolve()
    report_path = args.report.resolve()
    common.ensure_absent((output_blend, report_path))
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(bpy.data.filepath).resolve() != input_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match input {input_blend}")

    meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and str(obj.get("fh6_component", "")) == "HeadHair"
        and not bool(obj.get("fh6_probe_exclude", False))
    ]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one active HeadHair mesh, found {[obj.name for obj in meshes]}")
    output_mesh = meshes[0]
    armature = common.donor_armature(output_mesh)
    face_index = common.material_index(output_mesh, common.FACE_MATERIAL)
    iris_index = common.material_index(output_mesh, common.IRIS_MATERIAL)
    coordinates = common.world_coordinates(output_mesh)
    face_boundaries = common.material_boundary_components(output_mesh, face_index)
    iris_vertices = {
        index
        for polygon in common.material_polygons(output_mesh, iris_index)
        for index in polygon.vertices
    }
    inverse_world = output_mesh.matrix_world.inverted()
    local_rotation = inverse_world.to_3x3()
    eye_records: dict[str, dict[str, object]] = {}
    sockets: dict[str, list[Vector]] = {}
    irises: dict[str, list[Vector]] = {}

    for side, definition in common.SIDES.items():
        eye_bone = common.bone_head(armature, str(definition["bone"]))
        sign = float(definition["sign"])
        socket_ids = common.choose_side_component(face_boundaries, coordinates, eye_bone, sign, 12)
        iris_ids = {
            index
            for index in iris_vertices
            if coordinates[index].x * sign > 0.0
            and (coordinates[index] - eye_bone).length <= 0.040
        }
        if len(iris_ids) != 151:
            raise RuntimeError(f"Expected 151 {side} iris vertices, found {len(iris_ids)}")
        socket_center = common.centroid(coordinates[index] for index in socket_ids)
        iris_center_before = common.centroid(coordinates[index] for index in iris_ids)
        world_delta = Vector((socket_center.x - iris_center_before.x, 0.0, socket_center.z - iris_center_before.z))
        local_delta = local_rotation @ world_delta
        for index in iris_ids:
            moved = coordinates[index] + world_delta
            output_mesh.data.vertices[index].co = inverse_world @ moved
            coordinates[index] = moved
        shape_key_count = translated_shape_keys(output_mesh, iris_ids, local_delta)
        output_mesh.data.update()
        iris_center_after = iris_center_before + world_delta
        sockets[side] = sorted_socket_points(socket_ids, coordinates)
        irises[side] = [coordinates[index] for index in sorted(iris_ids)]
        eye_records[side] = {
            "bone": definition["bone"],
            "socket_vertices": len(socket_ids),
            "iris_vertices": len(iris_ids),
            "socket_centroid_m": common.vector_record(socket_center),
            "iris_centroid_before_m": common.vector_record(iris_center_before),
            "iris_centroid_after_m": common.vector_record(iris_center_after),
            "iris_recenter_mm": common.vector_record(world_delta * 1000.0),
            "iris_centroid_to_bone_after_mm": round((iris_center_after - eye_bone).length * 1000.0, 6),
            "shape_keys_translated": shape_key_count,
        }

    sclera_vertices: list[Vector] = []
    sclera_faces: list[list[int]] = []
    sclera_uvs: list[tuple[float, float]] = []
    sclera_groups: dict[str, list[int]] = defaultdict(list)
    depth_m = args.sclera_depth_mm / 1000.0
    for side, definition in common.SIDES.items():
        vertices, faces, uvs, patch_report = ring_patch(
            sockets[side],
            irises[side],
            inverse_world,
            depth_m,
        )
        vertex_offset = len(sclera_vertices)
        sclera_vertices.extend(vertices)
        sclera_uvs.extend(uvs)
        sclera_faces.extend([[index + vertex_offset for index in face] for face in faces])
        sclera_groups[str(definition["bone"])].extend(range(vertex_offset, vertex_offset + len(vertices)))
        eye_records[side]["sclera_patch"] = patch_report
    if len(sclera_vertices) != 158 or len(sclera_faces) != 156:
        raise RuntimeError(f"Unexpected ring-patch topology: {len(sclera_vertices)} vertices, {len(sclera_faces)} faces")

    sclera_data = bpy.data.meshes.new("Si_Display_Sclera_LOD0_RingPatch_Mesh")
    sclera_data.from_pydata(sclera_vertices, [], sclera_faces)
    sclera_data.update(calc_edges=True)
    sclera_data.materials.append(common.create_sclera_material())
    uv_layer = sclera_data.uv_layers.new(name="UV0")
    for loop in sclera_data.loops:
        uv_layer.data[loop.index].uv = sclera_uvs[loop.vertex_index]
    for polygon in sclera_data.polygons:
        polygon.use_smooth = True

    sclera_object = bpy.data.objects.new("Si_Display_Sclera_LOD0_RingPatch", sclera_data)
    sclera_object.matrix_world = output_mesh.matrix_world.copy()
    bpy.context.scene.collection.objects.link(sclera_object)
    for bone, indices in sorted(sclera_groups.items()):
        group = sclera_object.vertex_groups.new(name=bone)
        group.add(indices, 1.0, "REPLACE")
    modifier = sclera_object.modifiers.new(name="FH6 Armature", type="ARMATURE")
    modifier.object = armature
    sclera_object["source_format"] = "fbx-derived"
    sclera_object["source_role"] = "head"
    sclera_object["source_lod"] = "lod0"
    sclera_object["fh6_component"] = "HeadHair"
    sclera_object["fh6_donor"] = output_mesh.get("fh6_donor")
    sclera_object["fh6_weights_retargeted"] = True
    sclera_object["fh6_probe_exclude"] = False
    sclera_object["fh6_geometry_authority"] = "native FBX face socket boundary"

    bpy.ops.object.select_all(action="DESELECT")
    output_mesh.select_set(True)
    sclera_object.select_set(True)
    bpy.context.view_layer.objects.active = output_mesh
    bpy.ops.object.join()
    output_mesh = bpy.context.view_layer.objects.active
    output_mesh.name = "Si_Display_HeadHair_LOD0"
    output_mesh.data.name = "Si_Display_HeadHair_LOD0_EyeStageV007_Mesh"
    output_mesh["fh6_eye_stage"] = "v007"
    output_mesh["fh6_sclera_source"] = "three-ring native FBX socket patch"

    weight_validation = verify_weights(output_mesh, args.max_influences)
    if weight_validation["vertices_over_limit"] or weight_validation["zero_weight_vertices"]:
        raise RuntimeError(f"Eye-stage weight validation failed: {weight_validation}")
    if len(output_mesh.data.vertices) > 65535:
        raise RuntimeError(f"Eye-stage vertex domain exceeds R16 index limit: {len(output_mesh.data.vertices)}")
    if not output_mesh.data.uv_layers:
        raise RuntimeError("Eye-stage output lost UV coordinates")

    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), compress=False, check_existing=False)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "FBX-first Display eye Stage 02 v007: complete iris Shape Key translation and continuous socket-derived sclera.",
        "input": {"blend": str(input_blend), "sha256": common.sha256(input_blend)},
        "output": {
            "blend": str(output_blend),
            "sha256": common.sha256(output_blend),
            "object": output_mesh.name,
            "vertices": len(output_mesh.data.vertices),
            "polygons": len(output_mesh.data.polygons),
            "materials": [material.name if material else None for material in output_mesh.data.materials],
            "uv_layers": [layer.name for layer in output_mesh.data.uv_layers],
            "weights": weight_validation,
        },
        "eyes": eye_records,
        "derived_geometry": {
            "material": common.SCLERA_MATERIAL,
            "vertices": len(sclera_vertices),
            "polygons": len(sclera_faces),
            "rings_per_eye": len(RING_SCALES),
            "topology_source": "native FBX face socket connected boundary",
            "forward_axis": "+Y",
            "depth_mm": args.sclera_depth_mm,
        },
        "constraints": {
            "source_geometry_authority": "FBX",
            "pose_position": "REST",
            "maximum_influences": args.max_influences,
            "maximum_vertex_domain": 65535,
            "modelbin_written": False,
            "game_directory_written": False,
        },
        "supersedes": "v006 disconnected-iris-island sclera experiment",
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_SI_EYE_STAGE_V007=" + json.dumps({
        "output": str(output_blend),
        "vertices": len(output_mesh.data.vertices),
        "polygons": len(output_mesh.data.polygons),
        "sclera_vertices": len(sclera_vertices),
        "weight_validation": weight_validation,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
