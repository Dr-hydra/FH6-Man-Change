#!/usr/bin/env python3
"""Create a non-destructive Cloth1 upper-garment prototype from the Si PMX baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-split-blend", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview-front", required=True, type=Path)
    parser.add_argument("--preview-back", required=True, type=Path)
    parser.add_argument("--preview-side", required=True, type=Path)
    parser.add_argument("--material", default="Cloth1")
    parser.add_argument("--weld-epsilon", type=float, default=1e-5)
    parser.add_argument("--center-z-min", type=float, default=0.65)
    parser.add_argument("--center-z-max", type=float, default=1.5)
    parser.add_argument("--max-y-extent", type=float, default=0.6)
    parser.add_argument("--max-abs-center-x", type=float, default=0.7)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def select_faces(obj: bpy.types.Object, args: argparse.Namespace) -> tuple[set[int], list[dict]]:
    mesh = obj.data
    material_index = next(
        (
            index
            for index, slot in enumerate(obj.material_slots)
            if slot.material and slot.material.name == args.material
        ),
        None,
    )
    if material_index is None:
        raise ValueError(f"material not found: {args.material}")
    polygons = [polygon for polygon in mesh.polygons if polygon.material_index == material_index]
    material_vertices = {vertex for polygon in polygons for vertex in polygon.vertices}
    union_find = UnionFind(len(mesh.vertices))
    buckets: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    for vertex_index in material_vertices:
        coordinate = mesh.vertices[vertex_index].co
        key = tuple(round(coordinate[axis] / args.weld_epsilon) for axis in range(3))
        buckets[key].append(vertex_index)
    for bucket in buckets.values():
        for index in range(1, len(bucket)):
            union_find.union(bucket[0], bucket[index])
    for polygon in polygons:
        vertices = list(polygon.vertices)
        for index in range(1, len(vertices)):
            union_find.union(vertices[0], vertices[index])

    vertices_by_root: defaultdict[int, set[int]] = defaultdict(set)
    polygons_by_root: defaultdict[int, list[int]] = defaultdict(list)
    for polygon in polygons:
        root = union_find.find(polygon.vertices[0])
        vertices_by_root[root].update(polygon.vertices)
        polygons_by_root[root].append(polygon.index)

    selected_faces: set[int] = set()
    selected_islands = []
    for root, vertices in vertices_by_root.items():
        coordinates = [mesh.vertices[index].co for index in vertices]
        minimum = [min(co[axis] for co in coordinates) for axis in range(3)]
        maximum = [max(co[axis] for co in coordinates) for axis in range(3)]
        center = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
        extent = [maximum[axis] - minimum[axis] for axis in range(3)]
        selected = (
            abs(center[0]) <= args.max_abs_center_x
            and args.center_z_min <= center[2] <= args.center_z_max
            and extent[1] <= args.max_y_extent
        )
        if selected:
            selected_faces.update(polygons_by_root[root])
            selected_islands.append(
                {
                    "seed_vertex": min(vertices),
                    "vertices_before_seam_preservation": len(vertices),
                    "polygons": len(polygons_by_root[root]),
                    "center": center,
                    "extent": extent,
                }
            )
    if not selected_faces:
        raise ValueError("selection produced no garment faces")
    selected_islands.sort(key=lambda item: (-item["vertices_before_seam_preservation"], item["seed_vertex"]))
    return selected_faces, selected_islands


def selected_vertex_indices(mesh: bpy.types.Mesh, face_indices: set[int]) -> set[int]:
    return {
        vertex
        for face_index in face_indices
        for vertex in mesh.polygons[face_index].vertices
    }


def affected_shape_keys(mesh: bpy.types.Mesh, vertex_indices: set[int]) -> list[dict]:
    if not mesh.shape_keys:
        return []
    basis = mesh.shape_keys.key_blocks[0]
    result = []
    for key in mesh.shape_keys.key_blocks[1:]:
        affected = 0
        maximum_delta = 0.0
        for vertex_index in vertex_indices:
            delta = (key.data[vertex_index].co - basis.data[vertex_index].co).length
            if delta > 1e-6:
                affected += 1
                maximum_delta = max(maximum_delta, delta)
        if affected:
            result.append({"name": key.name, "affected_vertices": affected, "max_delta": maximum_delta})
    return result


def prune_to_faces(working: bpy.types.Object, selected_faces: set[int]) -> None:
    while working.data.shape_keys:
        working.shape_key_remove(working.data.shape_keys.key_blocks[-1])
    bm = bmesh.new()
    bm.from_mesh(working.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    source_index = bm.verts.layers.int.get("pmx_source_vertex_index")
    if source_index is None:
        source_index = bm.verts.layers.int.new("pmx_source_vertex_index")
    for vertex in bm.verts:
        vertex[source_index] = vertex.index
    delete_faces = [face for face in bm.faces if face.index not in selected_faces]
    bmesh.ops.delete(bm, geom=delete_faces, context="FACES")
    loose_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
    if loose_vertices:
        bmesh.ops.delete(bm, geom=loose_vertices, context="VERTS")
    bm.to_mesh(working.data)
    bm.free()
    working.data.update()


def prune_vertex_groups(working: bpy.types.Object, armature: bpy.types.Object) -> list[str]:
    deform_names = {bone.name for bone in armature.data.bones if bone.use_deform}
    used_names = {
        working.vertex_groups[element.group].name
        for vertex in working.data.vertices
        for element in vertex.groups
        if element.weight > 1e-8
    }
    keep_names = used_names & deform_names
    for group in list(working.vertex_groups)[::-1]:
        if group.name not in keep_names:
            working.vertex_groups.remove(group)
    return sorted(keep_names)


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def setup_render(scene: bpy.types.Scene, working: bpy.types.Object) -> bpy.types.Object:
    minimum, maximum = bounds([working])
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    camera_data = bpy.data.cameras.new("Garment Prototype Camera")
    camera = bpy.data.objects.new("Garment Prototype Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 58
    camera["target"] = list(center)
    camera["distance"] = max(max(extent) * 2.1, 0.8)
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    return camera


def render_view(scene: bpy.types.Scene, camera: bpy.types.Object, direction: Vector, output: Path) -> None:
    target = Vector(camera["target"])
    distance = float(camera["distance"])
    camera.location = target + direction.normalized() * distance
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = arguments()
    outputs = [
        args.pre_split_blend.resolve(),
        args.output_blend.resolve(),
        args.report.resolve(),
        args.preview_front.resolve(),
        args.preview_back.resolve(),
        args.preview_side.resolve(),
    ]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite milestone: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    source_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(source_meshes) != 1 or len(armatures) != 1:
        raise ValueError("expected one PMX mesh and one armature")
    source = source_meshes[0]
    armature = armatures[0]
    source_shape_key_count = len(source.data.shape_keys.key_blocks) if source.data.shape_keys else 0
    selected_faces, selected_islands = select_faces(source, args)
    selected_vertices = selected_vertex_indices(source.data, selected_faces)
    affected_morphs = affected_shape_keys(source.data, selected_vertices)
    if affected_morphs:
        raise ValueError(f"selected garment unexpectedly uses shape keys: {affected_morphs[:5]}")

    # The skill requires a fresh milestone before topology mutation.
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs[0]), compress=False, check_existing=False)

    source["fh6_probe_exclude"] = True
    source["fh6_source_locked"] = True
    source.hide_render = True
    source.hide_set(True)
    for obj in bpy.context.scene.objects:
        if obj.type == "EMPTY":
            obj.hide_render = True
            obj.hide_set(True)

    work_collection = bpy.data.collections.new("WORK_GARMENT_CLOTH1_UPPER")
    bpy.context.scene.collection.children.link(work_collection)
    working = source.copy()
    working.data = source.data.copy()
    working.name = "Si_Garment_Cloth1_Upper_Prototype"
    working.data.name = "Si_Garment_Cloth1_Upper_Prototype"
    working.hide_render = False
    working.hide_set(False)
    working["fh6_probe_exclude"] = False
    working["fh6_component"] = "Garment"
    working["fh6_donor"] = "Upper_Shirt_Tucked_N_Driver"
    working["fh6_geometry_edited"] = True
    working["fh6_weights_retargeted"] = False
    work_collection.objects.link(working)
    prune_to_faces(working, selected_faces)
    current_source_shape_keys = len(source.data.shape_keys.key_blocks) if source.data.shape_keys else 0
    if current_source_shape_keys != source_shape_key_count:
        raise RuntimeError("working-copy shape-key removal changed the locked source mesh")

    cloth_material = next(
        slot.material
        for slot in source.material_slots
        if slot.material and slot.material.name == args.material
    )
    working.data.materials.clear()
    working.data.materials.append(cloth_material)
    for polygon in working.data.polygons:
        polygon.material_index = 0
    used_bones = prune_vertex_groups(working, armature)

    scene = bpy.context.scene
    scene["working_milestone"] = "si_cloth1_upper_source_weights_v001"
    scene["source_pmx_preserved"] = True
    scene["retarget_started"] = False
    camera = setup_render(scene, working)
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs[3])
    render_view(scene, camera, Vector((0.0, 1.0, 0.0)), outputs[4])
    render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs[5])
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs[3])
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs[1]), compress=False, check_existing=False)

    influence_counts = defaultdict(int)
    non_normalized = 0
    for vertex in working.data.vertices:
        weights = [element.weight for element in vertex.groups if element.weight > 1e-8]
        influence_counts[len(weights)] += 1
        if weights and abs(sum(weights) - 1.0) > 1e-4:
            non_normalized += 1
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(outputs[0]),
        "output_blend": str(outputs[1]),
        "output_blend_sha256": sha256(outputs[1]),
        "material": args.material,
        "selection": {
            "weld_epsilon": args.weld_epsilon,
            "center_z_min": args.center_z_min,
            "center_z_max": args.center_z_max,
            "max_y_extent": args.max_y_extent,
            "max_abs_center_x": args.max_abs_center_x,
            "selected_islands": len(selected_islands),
            "selected_faces_before_split": len(selected_faces),
            "selected_vertices_before_split": len(selected_vertices),
            "islands": selected_islands,
        },
        "result": {
            "object": working.name,
            "vertices": len(working.data.vertices),
            "polygons": len(working.data.polygons),
            "materials": len(working.material_slots),
            "uv_layers": [layer.name for layer in working.data.uv_layers],
            "shape_keys": len(working.data.shape_keys.key_blocks) if working.data.shape_keys else 0,
            "used_source_bones": len(used_bones),
            "influence_histogram": dict(sorted(influence_counts.items())),
            "non_normalized_vertices": non_normalized,
            "source_morphs_affecting_selection": affected_morphs,
            "weights_retargeted": False,
        },
        "previews": {
            "front": str(outputs[3]),
            "back": str(outputs[4]),
            "side": str(outputs[5]),
        },
        "license_guard": "Local technical validation only; do not redistribute this split component.",
    }
    outputs[2].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FH6_GARMENT_PROTOTYPE=" + json.dumps({
        "blend": str(outputs[1]),
        "report": str(outputs[2]),
        "vertices": len(working.data.vertices),
        "polygons": len(working.data.polygons),
        "islands": len(selected_islands),
        "used_source_bones": len(used_bones),
        "non_normalized_vertices": non_normalized,
        "previews": report["previews"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
