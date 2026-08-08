#!/usr/bin/env python3
"""Split the immutable Si PMX baseline into FH6 character components."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_pmx_garment_prototype import (
    UnionFind,
    bounds,
    prune_to_faces,
    prune_vertex_groups,
    selected_vertex_indices,
)


HEAD_MATERIALS = {"面", "目", "目HL", "目白", "目影", "睫眉", "口内", "表情"}
BODY_MATERIAL = "肌"
GARMENT_MATERIALS = {"Cloth1", "Cloth2", "Cloth1Alpha"}
HAIR_MATERIALS = {"发", "发影"}
COMPONENT_ORDER = ("Head", "Body", "Outfit", "Helmet")
COMPONENT_COLORS = {
    "Head": (0.72, 0.30, 0.38, 1.0),
    "Body": (0.20, 0.45, 0.72, 1.0),
    "Outfit": (0.58, 0.62, 0.66, 1.0),
    "Helmet": (0.82, 0.56, 0.16, 1.0),
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--pre-split-blend", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview-front", required=True, type=Path)
    parser.add_argument("--preview-back", required=True, type=Path)
    parser.add_argument("--preview-side", required=True, type=Path)
    parser.add_argument("--head-island-min-z", type=float, default=1.55)
    parser.add_argument("--head-cut-z", type=float, default=1.44)
    parser.add_argument("--head-bone-ratio", type=float, default=0.35)
    parser.add_argument("--weld-epsilon", type=float, default=1e-5)
    return parser.parse_args(argv)


def ensure_outputs_absent(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)


def source_materials(obj: bpy.types.Object) -> dict[int, str]:
    result = {
        index: slot.material.name
        for index, slot in enumerate(obj.material_slots)
        if slot.material is not None
    }
    expected = HEAD_MATERIALS | {BODY_MATERIAL} | GARMENT_MATERIALS | HAIR_MATERIALS
    missing = expected - set(result.values())
    if missing:
        raise RuntimeError(f"Missing source materials: {sorted(missing)}")
    unknown = set(result.values()) - expected
    if unknown:
        raise RuntimeError(f"Unassigned source materials: {sorted(unknown)}")
    return result


def is_head_group(name: str) -> bool:
    lowered = name.lower()
    return (
        name == "頭"
        or lowered.startswith("face")
        or "head" in lowered
        or "cheek" in lowered
        or "jaw" in lowered
        or "ear" in lowered
        or "bowa" in lowered
        or "bowb" in lowered
    )


def head_weight_ratio(
    obj: bpy.types.Object,
    vertex_indices: set[int],
    group_names: dict[int, str],
) -> tuple[float, float]:
    head_weight = 0.0
    total_weight = 0.0
    for vertex_index in vertex_indices:
        for assignment in obj.data.vertices[vertex_index].groups:
            if assignment.weight <= 0.0:
                continue
            total_weight += float(assignment.weight)
            name = group_names.get(assignment.group)
            if name is not None and is_head_group(name):
                head_weight += float(assignment.weight)
    return head_weight, head_weight / total_weight if total_weight else 0.0


def garment_islands(
    obj: bpy.types.Object,
    material_index: int,
    material_name: str,
    weld_epsilon: float,
) -> list[tuple[list[int], set[int]]]:
    mesh = obj.data
    polygons = [polygon for polygon in mesh.polygons if polygon.material_index == material_index]
    vertices = {vertex for polygon in polygons for vertex in polygon.vertices}
    union_find = UnionFind(len(mesh.vertices))
    buckets: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    for vertex_index in vertices:
        coordinate = mesh.vertices[vertex_index].co
        key = tuple(round(coordinate[axis] / weld_epsilon) for axis in range(3))
        buckets[key].append(vertex_index)
    for bucket in buckets.values():
        for index in range(1, len(bucket)):
            union_find.union(bucket[0], bucket[index])
    for polygon in polygons:
        for index in range(1, len(polygon.vertices)):
            union_find.union(polygon.vertices[0], polygon.vertices[index])

    faces_by_root: defaultdict[int, list[int]] = defaultdict(list)
    vertices_by_root: defaultdict[int, set[int]] = defaultdict(set)
    for polygon in polygons:
        root = union_find.find(polygon.vertices[0])
        faces_by_root[root].append(polygon.index)
        vertices_by_root[root].update(polygon.vertices)
    return [
        (faces_by_root[root], vertices_by_root[root])
        for root in sorted(faces_by_root, key=lambda item: min(vertices_by_root[item]))
    ]


def classify_faces(
    obj: bpy.types.Object,
    material_names: dict[int, str],
    head_island_min_z: float,
    head_cut_z: float,
    head_bone_ratio: float,
    weld_epsilon: float,
) -> tuple[dict[str, set[int]], list[dict[str, object]]]:
    mesh = obj.data
    result = {name: set() for name in COMPONENT_ORDER}
    group_names = {group.index: group.name for group in obj.vertex_groups}
    island_records: list[dict[str, object]] = []

    for polygon in mesh.polygons:
        material_name = material_names[polygon.material_index]
        if material_name in HEAD_MATERIALS:
            result["Head"].add(polygon.index)
        elif material_name in HAIR_MATERIALS:
            result["Helmet"].add(polygon.index)
        elif material_name == BODY_MATERIAL:
            centroid_z = sum(mesh.vertices[index].co.z for index in polygon.vertices) / len(polygon.vertices)
            result["Body" if centroid_z < head_cut_z else "Head"].add(polygon.index)

    for material_index, material_name in sorted(material_names.items()):
        if material_name not in GARMENT_MATERIALS:
            continue
        for rank, (face_indices, vertex_indices) in enumerate(
            garment_islands(obj, material_index, material_name, weld_epsilon),
            start=1,
        ):
            coordinates = [mesh.vertices[index].co for index in vertex_indices]
            minimum = [min(co[axis] for co in coordinates) for axis in range(3)]
            maximum = [max(co[axis] for co in coordinates) for axis in range(3)]
            head_weight, head_ratio = head_weight_ratio(obj, vertex_indices, group_names)
            reasons = []
            if minimum[2] >= head_island_min_z:
                reasons.append("island_above_head_boundary")
            if head_ratio >= head_bone_ratio:
                reasons.append("head_driven_weights")
            component = "Helmet" if reasons else "Outfit"
            result[component].update(face_indices)
            island_records.append(
                {
                    "material": material_name,
                    "rank": rank,
                    "component": component,
                    "seed_vertex": min(vertex_indices),
                    "vertices": len(vertex_indices),
                    "faces": len(face_indices),
                    "bounds_min": minimum,
                    "bounds_max": maximum,
                    "head_group_weight": head_weight,
                    "head_group_ratio": head_ratio,
                    "reasons": reasons,
                }
            )

    all_faces = set().union(*result.values())
    duplicates = sum(len(faces) for faces in result.values()) - len(all_faces)
    if len(all_faces) != len(mesh.polygons) or duplicates:
        raise RuntimeError(
            f"Component partition invalid: covered={len(all_faces)}/{len(mesh.polygons)}, duplicates={duplicates}"
        )
    island_records.sort(key=lambda item: (item["component"], item["material"], -int(item["faces"])))
    return result, island_records


def create_component(
    source: bpy.types.Object,
    armature: bpy.types.Object,
    component: str,
    face_indices: set[int],
    material_names: dict[int, str],
) -> tuple[bpy.types.Object, dict[str, object]]:
    collection = bpy.data.collections.new(f"WORK_{component.upper()}")
    bpy.context.scene.collection.children.link(collection)
    working = source.copy()
    working.data = source.data.copy()
    working.name = f"Si_{component}_SourceSplit_v001"
    working.data.name = f"Si_{component}_SourceSplit_v001_Mesh"
    working.hide_render = False
    working.hide_set(False)
    working.color = COMPONENT_COLORS[component]
    working["fh6_probe_exclude"] = False
    working["fh6_component"] = component
    working["fh6_geometry_edited"] = True
    working["fh6_weights_retargeted"] = False
    collection.objects.link(working)
    prune_to_faces(working, face_indices)

    polygon_names = [material_names[polygon.material_index] for polygon in working.data.polygons]
    used_material_names = [
        material_names[index]
        for index in sorted(material_names)
        if material_names[index] in set(polygon_names)
    ]
    source_materials_by_name = {
        slot.material.name: slot.material
        for slot in source.material_slots
        if slot.material is not None
    }
    working.data.materials.clear()
    for name in used_material_names:
        working.data.materials.append(source_materials_by_name[name])
    slots = {name: index for index, name in enumerate(used_material_names)}
    for polygon, name in zip(working.data.polygons, polygon_names):
        polygon.material_index = slots[name]

    used_bones = prune_vertex_groups(working, armature)
    histogram: Counter[int] = Counter()
    zero_weight = 0
    non_normalized = 0
    for vertex in working.data.vertices:
        weights = [item.weight for item in vertex.groups if item.weight > 1e-8]
        histogram[len(weights)] += 1
        zero_weight += int(not weights)
        non_normalized += int(bool(weights) and abs(sum(weights) - 1.0) > 1e-4)
    return working, {
        "object": working.name,
        "vertices": len(working.data.vertices),
        "polygons": len(working.data.polygons),
        "materials": used_material_names,
        "used_source_bones": used_bones,
        "influence_histogram": dict(sorted(histogram.items())),
        "zero_weight_vertices": zero_weight,
        "non_normalized_vertices": non_normalized,
    }


def setup_render(scene: bpy.types.Scene, objects: list[bpy.types.Object]) -> bpy.types.Object:
    minimum, maximum = bounds(objects)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    camera_data = bpy.data.cameras.new("FH6 Component Split Camera")
    camera = bpy.data.objects.new("FH6 Component Split Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 58
    camera["target"] = list(center)
    camera["distance"] = max(max(extent) * 2.15, 0.8)
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    for item in scene.objects:
        item.hide_render = item not in objects and item is not camera
    return camera


def render_view(scene: bpy.types.Scene, camera: bpy.types.Object, direction: Vector, output: Path) -> None:
    target = Vector(camera["target"])
    camera.location = target + direction.normalized() * float(camera["distance"])
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source_blend}")
    outputs = {
        "pre": args.pre_split_blend.resolve(),
        "blend": args.output_blend.resolve(),
        "report": args.report.resolve(),
        "front": args.preview_front.resolve(),
        "back": args.preview_back.resolve(),
        "side": args.preview_side.resolve(),
    }
    ensure_outputs_absent(list(outputs.values()))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(meshes) != 1 or len(armatures) != 1:
        raise RuntimeError(f"Expected one source mesh/armature, found {len(meshes)}/{len(armatures)}")
    source = meshes[0]
    armature = armatures[0]
    names = source_materials(source)
    component_faces, island_records = classify_faces(
        source,
        names,
        args.head_island_min_z,
        args.head_cut_z,
        args.head_bone_ratio,
        args.weld_epsilon,
    )

    # Exact checkpoint before any topology or shape-key mutation.
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["pre"]), compress=False, check_existing=False)
    source["fh6_probe_exclude"] = True
    source["fh6_source_locked"] = True
    source.hide_render = True
    source.hide_set(True)
    armature["fh6_probe_exclude"] = True
    for item in bpy.context.scene.objects:
        if item.type == "EMPTY":
            item.hide_render = True
            item.hide_set(True)

    working_objects = []
    component_reports = {}
    source_vertex_sets = {}
    for component in COMPONENT_ORDER:
        faces = component_faces[component]
        source_vertex_sets[component] = selected_vertex_indices(source.data, faces)
        working, report = create_component(source, armature, component, faces, names)
        working_objects.append(working)
        component_reports[component] = report

    scene = bpy.context.scene
    scene["working_milestone"] = "si_fh6_component_split_v001"
    scene["source_pmx_preserved"] = True
    scene["retarget_started"] = False
    scene["license_guard"] = "Local technical validation only; do not redistribute."
    camera = setup_render(scene, working_objects)
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["front"])
    render_view(scene, camera, Vector((0.0, 1.0, 0.0)), outputs["back"])
    render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs["side"])
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["blend"]), compress=False, check_existing=False)

    split_vertex_total = sum(len(vertices) for vertices in source_vertex_sets.values())
    union_vertices = set().union(*source_vertex_sets.values())
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Deterministic Si split for FH6 Head, Body, Outfit, and Helmet slots.",
        "source_blend": str(source_blend),
        "milestones": {"pre_split": str(outputs["pre"]), "split": str(outputs["blend"])},
        "policy": {
            "head_materials": sorted(HEAD_MATERIALS),
            "body_material": BODY_MATERIAL,
            "garment_materials": sorted(GARMENT_MATERIALS),
            "hair_materials": sorted(HAIR_MATERIALS),
            "head_cut_z": args.head_cut_z,
            "head_island_min_z": args.head_island_min_z,
            "head_bone_ratio": args.head_bone_ratio,
            "helmet_role": "hair, hair shadow, and head-driven garment islands",
        },
        "components": component_reports,
        "garment_islands": island_records,
        "validation": {
            "source_faces": len(source.data.polygons),
            "partition_faces": sum(len(faces) for faces in component_faces.values()),
            "source_vertices_referenced": len(union_vertices),
            "split_vertices": split_vertex_total,
            "boundary_vertex_duplication": split_vertex_total - len(union_vertices),
            "all_components_below_65535": all(
                item["vertices"] <= 65535 for item in component_reports.values()
            ),
            "zero_weight_vertices": sum(
                item["zero_weight_vertices"] for item in component_reports.values()
            ),
            "non_normalized_vertices": sum(
                item["non_normalized_vertices"] for item in component_reports.values()
            ),
        },
        "previews": {"front": str(outputs["front"]), "back": str(outputs["back"]), "side": str(outputs["side"])},
        "license_guard": "Local technical validation only; do not redistribute the split Si components.",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_COMPONENT_SPLIT="
        + json.dumps(
            {
                "blend": str(outputs["blend"]),
                "components": {
                    name: {
                        "vertices": item["vertices"],
                        "polygons": item["polygons"],
                        "materials": item["materials"],
                    }
                    for name, item in component_reports.items()
                },
                "validation": report["validation"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
