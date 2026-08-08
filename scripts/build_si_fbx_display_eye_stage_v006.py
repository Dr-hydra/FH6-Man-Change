#!/usr/bin/env python3
"""Build the FBX-first Si Display eye correction milestone.

The native FBX has an iris shell and an eye-occlusion shell, but no opaque
sclera draw. This stage keeps the FBX face and iris topology authoritative,
recenters each iris on its source socket, and derives a slightly recessed
sclera shell from the corresponding iris component and socket boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retarget_fbx_display_components import verify_weights


FACE_MATERIAL = "\u9762"
IRIS_MATERIAL = "\u76ee"
SCLERA_MATERIAL = "\u5de9\u819c"
SIDES = {
    "left": {"bone": "LeftEye", "sign": -1.0},
    "right": {"bone": "RightEye", "sign": 1.0},
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-blend", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sclera-depth-mm", type=float, default=1.25)
    parser.add_argument("--socket-fill", type=float, default=0.96)
    parser.add_argument("--max-influences", type=int, default=4)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")


def material_index(mesh: bpy.types.Object, name: str) -> int:
    for index, material in enumerate(mesh.data.materials):
        if material is not None and material.name == name:
            return index
    raise RuntimeError(f"{mesh.name} is missing material {name!r}")


def material_polygons(mesh: bpy.types.Object, index: int) -> list[bpy.types.MeshPolygon]:
    return [polygon for polygon in mesh.data.polygons if polygon.material_index == index]


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


def material_vertex_components(mesh: bpy.types.Object, index: int) -> list[set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for polygon in material_polygons(mesh, index):
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            adjacency[first].add(second)
            adjacency[second].add(first)
    return graph_components(adjacency)


def material_boundary_components(mesh: bpy.types.Object, index: int) -> list[set[int]]:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for polygon in material_polygons(mesh, index):
        vertices = list(polygon.vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            edge_counts[tuple(sorted((first, second)))] += 1
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (first, second), count in edge_counts.items():
        if count != 1:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    return graph_components(adjacency)


def world_coordinates(mesh: bpy.types.Object) -> list[Vector]:
    return [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]


def centroid(points: Iterable[Vector]) -> Vector:
    values = list(points)
    if not values:
        raise RuntimeError("Cannot calculate an empty centroid")
    return sum(values, Vector()) / len(values)


def bounds(points: Iterable[Vector]) -> tuple[Vector, Vector]:
    values = list(points)
    if not values:
        raise RuntimeError("Cannot calculate empty bounds")
    return (
        Vector(tuple(min(point[axis] for point in values) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in values) for axis in range(3))),
    )


def donor_armature(mesh: bpy.types.Object) -> bpy.types.Object:
    armatures = {
        modifier.object
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object is not None
    }
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one donor armature modifier on {mesh.name}, found {len(armatures)}")
    armature = next(iter(armatures))
    # Stage 01 saves after clearing pose transforms but leaves Blender's
    # display toggle at POSE. Geometry authority is the REST bind state, so
    # force that state before any landmark or surface measurement.
    armature.data.pose_position = "REST"
    return armature


def bone_head(armature: bpy.types.Object, name: str) -> Vector:
    bone = armature.data.bones.get(name)
    if bone is None:
        raise RuntimeError(f"{armature.name} is missing eye bone {name!r}")
    return armature.matrix_world @ bone.head_local


def choose_side_component(
    components: list[set[int]],
    coordinates: list[Vector],
    center: Vector,
    sign: float,
    minimum_vertices: int,
) -> set[int]:
    candidates: list[tuple[tuple[float, float, int], set[int]]] = []
    for component in components:
        if len(component) < minimum_vertices:
            continue
        points = [coordinates[index] for index in component]
        middle = centroid(points)
        if middle.x * sign <= 0.0:
            continue
        minimum_distance = min((point - center).length for point in points)
        candidates.append(((minimum_distance, (middle - center).length, -len(component)), component))
    if not candidates:
        raise RuntimeError(f"No topology component found for eye side sign {sign:+.0f}")
    return min(candidates, key=lambda item: item[0])[1]


def create_sclera_material() -> bpy.types.Material:
    if bpy.data.materials.get(SCLERA_MATERIAL) is not None:
        raise RuntimeError(f"Material {SCLERA_MATERIAL!r} already exists; input is not a clean Stage 01 blend")
    material = bpy.data.materials.new(SCLERA_MATERIAL)
    material.use_nodes = True
    material.diffuse_color = (0.92, 0.95, 0.97, 1.0)
    material["fh6_material_family"] = "sclera"
    material["fh6_source_material_name"] = "__synthetic_sclera_from_fbx_socket__"
    material["fh6_render_surface"] = "opaque"
    material["fh6_alpha_mode"] = "opaque_white"
    principled = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.92, 0.95, 0.97, 1.0)
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Roughness"].default_value = 0.42
    return material


def vector_record(value: Vector) -> list[float]:
    return [round(float(item), 9) for item in value]


def main() -> None:
    args = arguments()
    input_blend = args.input_blend.resolve()
    output_blend = args.output_blend.resolve()
    report_path = args.report.resolve()
    ensure_absent((output_blend, report_path))
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
    armature = donor_armature(output_mesh)
    face_index = material_index(output_mesh, FACE_MATERIAL)
    iris_index = material_index(output_mesh, IRIS_MATERIAL)
    coordinates = world_coordinates(output_mesh)
    face_boundaries = material_boundary_components(output_mesh, face_index)
    iris_vertices = {
        index
        for polygon in material_polygons(output_mesh, iris_index)
        for index in polygon.vertices
    }

    socket_components: dict[str, set[int]] = {}
    selected_irises: dict[str, set[int]] = {}
    eye_records: dict[str, dict[str, object]] = {}
    inverse_world = output_mesh.matrix_world.inverted()

    for side, definition in SIDES.items():
        eye_bone = bone_head(armature, str(definition["bone"]))
        sign = float(definition["sign"])
        socket = choose_side_component(face_boundaries, coordinates, eye_bone, sign, 12)
        iris = {
            index
            for index in iris_vertices
            if coordinates[index].x * sign > 0.0
            and (coordinates[index] - eye_bone).length <= 0.040
        }
        if len(iris) < 100:
            raise RuntimeError(f"Incomplete {side} iris material selection: {len(iris)} vertices")
        socket_components[side] = socket
        selected_irises[side] = iris
        socket_center = centroid(coordinates[index] for index in socket)
        iris_center_before = centroid(coordinates[index] for index in iris)
        offset = Vector((socket_center.x - iris_center_before.x, 0.0, socket_center.z - iris_center_before.z))
        for index in iris:
            moved = coordinates[index] + offset
            output_mesh.data.vertices[index].co = inverse_world @ moved
            coordinates[index] = moved
        iris_center_after = iris_center_before + offset
        eye_records[side] = {
            "bone": definition["bone"],
            "socket_vertices": len(socket),
            "iris_vertices": len(iris),
            "socket_centroid_m": vector_record(socket_center),
            "iris_centroid_before_m": vector_record(iris_center_before),
            "iris_centroid_after_m": vector_record(iris_center_after),
            "iris_recenter_mm": vector_record(offset * 1000.0),
            "iris_centroid_to_bone_after_mm": round((iris_center_after - eye_bone).length * 1000.0, 6),
        }
    output_mesh.data.update()

    sclera_vertices: list[Vector] = []
    sclera_faces: list[list[int]] = []
    sclera_uv: list[tuple[float, float]] = []
    sclera_groups: dict[str, list[int]] = defaultdict(list)
    sclera_depth = args.sclera_depth_mm / 1000.0

    for side, definition in SIDES.items():
        iris = selected_irises[side]
        socket = socket_components[side]
        socket_points = [coordinates[index] for index in socket]
        iris_points = [coordinates[index] for index in iris]
        socket_minimum, socket_maximum = bounds(socket_points)
        iris_minimum, iris_maximum = bounds(iris_points)
        socket_center = centroid(socket_points)
        iris_width = iris_maximum.x - iris_minimum.x
        iris_height = iris_maximum.z - iris_minimum.z
        if iris_width <= 1e-6 or iris_height <= 1e-6:
            raise RuntimeError(f"Degenerate iris bounds on {side}")
        scale_x = max(0.65, min(1.75, (socket_maximum.x - socket_minimum.x) * args.socket_fill / iris_width))
        scale_z = max(0.65, min(1.75, (socket_maximum.z - socket_minimum.z) * args.socket_fill / iris_height))
        source_to_new: dict[int, int] = {}
        base_index = len(sclera_vertices)
        half_width = max((socket_maximum.x - socket_minimum.x) * 0.5, 1e-6)
        half_height = max((socket_maximum.z - socket_minimum.z) * 0.5, 1e-6)
        for source_index in sorted(iris):
            point = coordinates[source_index]
            derived = Vector((
                socket_center.x + (point.x - socket_center.x) * scale_x,
                point.y - sclera_depth,
                socket_center.z + (point.z - socket_center.z) * scale_z,
            ))
            new_index = len(sclera_vertices)
            source_to_new[source_index] = new_index
            sclera_vertices.append(inverse_world @ derived)
            sclera_uv.append((
                max(0.0, min(1.0, 0.5 + (derived.x - socket_center.x) / (2.0 * half_width))),
                max(0.0, min(1.0, 0.5 + (derived.z - socket_center.z) / (2.0 * half_height))),
            ))
            sclera_groups[str(definition["bone"])].append(new_index)
        face_start = len(sclera_faces)
        for polygon in material_polygons(output_mesh, iris_index):
            if all(index in source_to_new for index in polygon.vertices):
                sclera_faces.append([source_to_new[index] for index in polygon.vertices])
        eye_records[side].update({
            "sclera_vertices": len(iris),
            "sclera_faces": len(sclera_faces) - face_start,
            "sclera_scale_x": round(scale_x, 9),
            "sclera_scale_z": round(scale_z, 9),
            "sclera_depth_mm": args.sclera_depth_mm,
            "sclera_vertex_start": base_index,
        })

    if len(sclera_vertices) != 302 or len(sclera_faces) != 488:
        raise RuntimeError(f"Unexpected derived sclera topology: {len(sclera_vertices)} vertices, {len(sclera_faces)} faces")

    sclera_data = bpy.data.meshes.new("Si_Display_Sclera_LOD0_Mesh")
    sclera_data.from_pydata(sclera_vertices, [], sclera_faces)
    sclera_data.update(calc_edges=True)
    sclera_material = create_sclera_material()
    sclera_data.materials.append(sclera_material)
    uv_layer = sclera_data.uv_layers.new(name="UV0")
    for loop in sclera_data.loops:
        uv_layer.data[loop.index].uv = sclera_uv[loop.vertex_index]
    for polygon in sclera_data.polygons:
        polygon.use_smooth = True

    sclera_object = bpy.data.objects.new("Si_Display_Sclera_LOD0", sclera_data)
    sclera_object.matrix_world = output_mesh.matrix_world.copy()
    bpy.context.scene.collection.objects.link(sclera_object)
    for bone, indices in sorted(sclera_groups.items()):
        group = sclera_object.vertex_groups.new(name=bone)
        group.add(indices, 1.0, "REPLACE")
    modifier = sclera_object.modifiers.new(name="FH6 Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = False
    sclera_object["source_format"] = "fbx-derived"
    sclera_object["source_role"] = "head"
    sclera_object["source_lod"] = "lod0"
    sclera_object["fh6_component"] = "HeadHair"
    sclera_object["fh6_donor"] = output_mesh.get("fh6_donor")
    sclera_object["fh6_weights_retargeted"] = True
    sclera_object["fh6_probe_exclude"] = False
    sclera_object["fh6_geometry_authority"] = "native FBX iris topology and face socket boundary"

    bpy.ops.object.select_all(action="DESELECT")
    output_mesh.select_set(True)
    sclera_object.select_set(True)
    bpy.context.view_layer.objects.active = output_mesh
    bpy.ops.object.join()
    output_mesh = bpy.context.view_layer.objects.active
    output_mesh.name = "Si_Display_HeadHair_LOD0"
    output_mesh.data.name = "Si_Display_HeadHair_LOD0_EyeStage_Mesh"
    output_mesh["fh6_eye_stage"] = "v006"
    output_mesh["fh6_sclera_source"] = "derived from native FBX iris and socket topology"

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
        "purpose": "FBX-first Display eye Stage 02: socket-centered iris plus generated opaque sclera draw.",
        "input": {"blend": str(input_blend), "sha256": sha256(input_blend)},
        "output": {
            "blend": str(output_blend),
            "sha256": sha256(output_blend),
            "object": output_mesh.name,
            "vertices": len(output_mesh.data.vertices),
            "polygons": len(output_mesh.data.polygons),
            "materials": [material.name if material else None for material in output_mesh.data.materials],
            "uv_layers": [layer.name for layer in output_mesh.data.uv_layers],
            "weights": weight_validation,
        },
        "eyes": eye_records,
        "derived_geometry": {
            "material": SCLERA_MATERIAL,
            "vertices": len(sclera_vertices),
            "polygons": len(sclera_faces),
            "topology_source": "native FBX iris connected components",
            "fit_source": "native FBX face socket boundary components",
            "forward_axis": "+Y",
            "depth_mm": args.sclera_depth_mm,
            "socket_fill": args.socket_fill,
        },
        "constraints": {
            "source_geometry_authority": "FBX",
            "pose_position": "REST",
            "maximum_influences": args.max_influences,
            "maximum_vertex_domain": 65535,
            "modelbin_written": False,
            "game_directory_written": False,
        },
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_SI_EYE_STAGE=" + json.dumps({
        "output": str(output_blend),
        "vertices": len(output_mesh.data.vertices),
        "polygons": len(output_mesh.data.polygons),
        "sclera_vertices": len(sclera_vertices),
        "weight_validation": weight_validation,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
