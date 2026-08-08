#!/usr/bin/env python3
"""Create a non-head full-body outfit prototype from the immutable Si PMX baseline."""

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
    affected_shape_keys,
    prune_to_faces,
    prune_vertex_groups,
    render_view,
    selected_vertex_indices,
    setup_render,
    sha256,
)


DEFAULT_MATERIALS = ("肌", "Cloth1", "Cloth2", "Cloth1Alpha")


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


def material_indices(obj: bpy.types.Object) -> dict[int, str]:
    wanted = set(DEFAULT_MATERIALS)
    result = {
        index: slot.material.name
        for index, slot in enumerate(obj.material_slots)
        if slot.material and slot.material.name in wanted
    }
    missing = wanted - set(result.values())
    if missing:
        raise RuntimeError(f"Missing source materials: {sorted(missing)}")
    return result


def selected_faces(
    obj: bpy.types.Object,
    selected_materials: dict[int, str],
    head_island_min_z: float,
    head_cut_z: float,
    head_bone_ratio: float,
    weld_epsilon: float,
) -> tuple[set[int], list[dict[str, object]], list[dict[str, object]]]:
    mesh = obj.data
    selected: set[int] = set()
    kept_islands: list[dict[str, object]] = []
    removed_islands: list[dict[str, object]] = []

    group_names = {group.index: group.name for group in obj.vertex_groups}

    def is_head_group(name: str) -> bool:
        lowered = name.lower()
        return (
            name == "頭"
            or lowered.startswith("face")
            or "cheek" in lowered
            or "jaw" in lowered
            or "_ear" in lowered
            or "earring" in lowered
            or "bowb" in lowered
        )

    def island_head_weight(vertex_indices: set[int]) -> tuple[float, float]:
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
        ratio = head_weight / total_weight if total_weight else 0.0
        return head_weight, ratio

    for material_index, material_name in sorted(selected_materials.items()):
        polygons = [polygon for polygon in mesh.polygons if polygon.material_index == material_index]
        if material_name == "肌":
            kept = [
                polygon
                for polygon in polygons
                if sum(mesh.vertices[index].co.z for index in polygon.vertices) / len(polygon.vertices) < head_cut_z
            ]
            removed = len(polygons) - len(kept)
            selected.update(polygon.index for polygon in kept)
            kept_islands.append({
                "material": material_name,
                "policy": "keep body skin below the neck cut",
                "faces": len(kept),
                "head_cut_z": head_cut_z,
            })
            removed_islands.append({
                "material": material_name,
                "policy": "remove connected head skin above the neck cut",
                "faces": removed,
                "head_cut_z": head_cut_z,
                "reason": "face centroid lies at or above the neck cut",
            })
            continue

        material_vertices = {vertex for polygon in polygons for vertex in polygon.vertices}
        union_find = UnionFind(len(mesh.vertices))
        buckets: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
        for vertex_index in material_vertices:
            coordinate = mesh.vertices[vertex_index].co
            key = tuple(round(coordinate[axis] / weld_epsilon) for axis in range(3))
            buckets[key].append(vertex_index)
        for bucket in buckets.values():
            for index in range(1, len(bucket)):
                union_find.union(bucket[0], bucket[index])
        for polygon in polygons:
            vertices = list(polygon.vertices)
            for index in range(1, len(vertices)):
                union_find.union(vertices[0], vertices[index])

        faces_by_root: defaultdict[int, list[int]] = defaultdict(list)
        vertices_by_root: defaultdict[int, set[int]] = defaultdict(set)
        for polygon in polygons:
            root = union_find.find(polygon.vertices[0])
            faces_by_root[root].append(polygon.index)
            vertices_by_root[root].update(polygon.vertices)

        for root, face_indices in faces_by_root.items():
            vertex_indices = vertices_by_root[root]
            coordinates = [mesh.vertices[index].co for index in vertex_indices]
            minimum = [min(co[axis] for co in coordinates) for axis in range(3)]
            maximum = [max(co[axis] for co in coordinates) for axis in range(3)]
            record = {
                "material": material_name,
                "seed_vertex": min(vertex_indices),
                "vertices": len(vertex_indices),
                "faces": len(face_indices),
                "bounds_min": minimum,
                "bounds_max": maximum,
            }
            head_weight, head_ratio = island_head_weight(vertex_indices)
            record["head_group_weight"] = head_weight
            record["head_group_ratio"] = head_ratio
            if minimum[2] >= head_island_min_z:
                record["reason"] = "island lies completely above the outfit head boundary"
                removed_islands.append(record)
            elif head_ratio >= head_bone_ratio:
                record["reason"] = "island is predominantly driven by head, face, ear, earring, or head-bow bones"
                removed_islands.append(record)
            else:
                selected.update(face_indices)
                kept_islands.append(record)

    kept_islands.sort(key=lambda item: (-int(item["faces"]), str(item["material"])))
    removed_islands.sort(key=lambda item: (-int(item["faces"]), str(item["material"])))
    return selected, kept_islands, removed_islands


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
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(meshes) != 1 or len(armatures) != 1:
        raise RuntimeError(f"Expected one source mesh/armature, found {len(meshes)}/{len(armatures)}")
    source = meshes[0]
    armature = armatures[0]
    selected_materials = material_indices(source)
    face_indices, kept_islands, removed_islands = selected_faces(
        source,
        selected_materials,
        args.head_island_min_z,
        args.head_cut_z,
        args.head_bone_ratio,
        args.weld_epsilon,
    )
    vertex_indices = selected_vertex_indices(source.data, face_indices)
    affected_morphs = affected_shape_keys(source.data, vertex_indices)

    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["pre"]), compress=False, check_existing=False)
    source["fh6_probe_exclude"] = True
    source["fh6_source_locked"] = True
    source.hide_render = True
    source.hide_set(True)
    for obj in bpy.context.scene.objects:
        if obj.type == "EMPTY":
            obj.hide_render = True
            obj.hide_set(True)

    collection = bpy.data.collections.new("WORK_FULLBODY_OUTFIT")
    bpy.context.scene.collection.children.link(collection)
    working = source.copy()
    working.data = source.data.copy()
    working.name = "Si_FullBody_Outfit_Prototype_v001"
    working.data.name = "Si_FullBody_Outfit_Prototype_v001_Mesh"
    working.hide_render = False
    working.hide_set(False)
    working["fh6_probe_exclude"] = False
    working["fh6_component"] = "Outfit"
    working["fh6_donor"] = "Outfit_Race_Suit_Modern_F_Driver"
    working["fh6_geometry_edited"] = True
    working["fh6_weights_retargeted"] = False
    collection.objects.link(working)
    prune_to_faces(working, face_indices)

    selected_material_names = [selected_materials[index] for index in sorted(selected_materials)]
    selected_material_objects = {
        slot.material.name: slot.material
        for slot in source.material_slots
        if slot.material and slot.material.name in selected_material_names
    }
    polygon_material_names = [
        selected_materials.get(polygon.material_index, "Cloth1")
        for polygon in working.data.polygons
    ]
    working.data.materials.clear()
    for material_name in selected_material_names:
        working.data.materials.append(selected_material_objects[material_name])
    material_slot_by_name = {name: index for index, name in enumerate(selected_material_names)}
    for polygon, material_name in zip(working.data.polygons, polygon_material_names):
        polygon.material_index = material_slot_by_name[material_name]
    used_bones = prune_vertex_groups(working, armature)

    influence_histogram: Counter[int] = Counter()
    non_normalized = 0
    zero_weight = 0
    for vertex in working.data.vertices:
        weights = [item.weight for item in vertex.groups if item.weight > 1e-8]
        influence_histogram[len(weights)] += 1
        zero_weight += int(not weights)
        non_normalized += int(bool(weights) and abs(sum(weights) - 1.0) > 1e-4)

    scene = bpy.context.scene
    scene["working_milestone"] = "si_fullbody_outfit_source_weights_v001"
    scene["source_pmx_preserved"] = True
    scene["retarget_started"] = False
    camera = setup_render(scene, working)
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["front"])
    render_view(scene, camera, Vector((0.0, 1.0, 0.0)), outputs["back"])
    render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs["side"])
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["blend"]), compress=False, check_existing=False)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Non-head Si full-body outfit prototype for the FH6 modern race-suit slot.",
        "source_blend": str(source_blend),
        "milestones": {"pre_split": str(outputs["pre"]), "prototype": str(outputs["blend"])},
        "selection": {
            "materials": list(DEFAULT_MATERIALS),
            "head_island_min_z": args.head_island_min_z,
            "head_cut_z": args.head_cut_z,
            "head_bone_ratio": args.head_bone_ratio,
            "selected_faces": len(face_indices),
            "selected_source_vertices": len(vertex_indices),
            "kept_islands": kept_islands,
            "removed_head_islands": removed_islands,
            "affected_source_morphs_discarded_on_working_copy": affected_morphs,
        },
        "result": {
            "object": working.name,
            "vertices": len(working.data.vertices),
            "polygons": len(working.data.polygons),
            "materials": len(working.data.materials),
            "material_slots": selected_material_names,
            "uv_layers": [layer.name for layer in working.data.uv_layers],
            "used_source_bones": len(used_bones),
            "influence_histogram": dict(sorted(influence_histogram.items())),
            "zero_weight_vertices": zero_weight,
            "non_normalized_vertices": non_normalized,
        },
        "previews": {"front": str(outputs["front"]), "back": str(outputs["back"]), "side": str(outputs["side"])},
        "license_guard": "Local technical validation only; do not redistribute this split component.",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_FULLBODY_PROTOTYPE=" + json.dumps({
        "blend": str(outputs["blend"]),
        "vertices": report["result"]["vertices"],
        "polygons": report["result"]["polygons"],
        "used_source_bones": len(used_bones),
        "removed_head_islands": len(removed_islands),
        "zero_weight_vertices": zero_weight,
        "non_normalized_vertices": non_normalized,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
