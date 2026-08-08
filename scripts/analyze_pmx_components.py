#!/usr/bin/env python3
"""Analyze PMX material/component partitions in an existing Blender baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy


CATEGORY_BY_MATERIAL = {
    "面": "Head",
    "目": "Head",
    "目HL": "Head",
    "目白": "Head",
    "目影": "Head",
    "睫眉": "Head",
    "口内": "Head",
    "表情": "Head",
    "肌": "Body",
    "Cloth1": "Garment",
    "Cloth2": "Garment",
    "Cloth1Alpha": "Garment",
    "发": "Hair",
    "发影": "Hair",
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def bounds(mesh: bpy.types.Mesh, vertex_indices: set[int]) -> dict:
    if not vertex_indices:
        return {"min": None, "max": None, "extent": None}
    coordinates = [mesh.vertices[index].co for index in vertex_indices]
    minimum = [min(co[axis] for co in coordinates) for axis in range(3)]
    maximum = [max(co[axis] for co in coordinates) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "extent": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def bone_usage(obj: bpy.types.Object, vertex_indices: set[int], deform_groups: set[int]) -> dict:
    usage: defaultdict[int, dict] = defaultdict(lambda: {"vertices": 0, "total_weight": 0.0})
    max_influences = 0
    over_four = 0
    non_normalized = 0
    for vertex_index in vertex_indices:
        weights = [
            (element.group, element.weight)
            for element in obj.data.vertices[vertex_index].groups
            if element.group in deform_groups and element.weight > 1e-8
        ]
        max_influences = max(max_influences, len(weights))
        if len(weights) > 4:
            over_four += 1
        if weights and abs(sum(weight for _, weight in weights) - 1.0) > 1e-4:
            non_normalized += 1
        for group_index, weight in weights:
            usage[group_index]["vertices"] += 1
            usage[group_index]["total_weight"] += weight
    ranked = sorted(
        (
            {
                "group": obj.vertex_groups[group_index].name,
                "vertices": values["vertices"],
                "total_weight": values["total_weight"],
            }
            for group_index, values in usage.items()
        ),
        key=lambda item: (-item["total_weight"], -item["vertices"], item["group"]),
    )
    return {
        "bone_group_count": len(ranked),
        "max_influences": max_influences,
        "vertices_over_four": over_four,
        "non_normalized_vertices": non_normalized,
        "top_bones": ranked[:30],
    }


def morph_usage(mesh: bpy.types.Mesh, vertex_indices: set[int]) -> list[dict]:
    if not mesh.shape_keys or len(mesh.shape_keys.key_blocks) <= 1:
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


def material_images(material: bpy.types.Material | None) -> list[str]:
    if not material or not material.use_nodes or not material.node_tree:
        return []
    return sorted(
        {
            node.image.name
            for node in material.node_tree.nodes
            if node.type == "TEX_IMAGE" and node.image
        }
    )


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(mesh_objects) != 1 or len(armatures) != 1:
        raise ValueError(f"expected one PMX mesh and armature, got {len(mesh_objects)}/{len(armatures)}")
    obj = mesh_objects[0]
    mesh = obj.data
    armature = armatures[0]
    deform_bone_names = {bone.name for bone in armature.data.bones if bone.use_deform}
    deform_groups = {group.index for group in obj.vertex_groups if group.name in deform_bone_names}

    slot_vertices: defaultdict[int, set[int]] = defaultdict(set)
    slot_polygons: defaultdict[int, int] = defaultdict(int)
    slot_loops: defaultdict[int, int] = defaultdict(int)
    for polygon in mesh.polygons:
        slot_polygons[polygon.material_index] += 1
        slot_loops[polygon.material_index] += polygon.loop_total
        slot_vertices[polygon.material_index].update(polygon.vertices)

    categories: defaultdict[str, dict] = defaultdict(
        lambda: {"material_indices": [], "material_names": [], "vertices": set(), "polygons": 0, "loops": 0}
    )
    materials = []
    for material_index, slot in enumerate(obj.material_slots):
        material = slot.material
        name = material.name if material else f"<slot:{material_index}>"
        category = CATEGORY_BY_MATERIAL.get(name, "Unassigned")
        vertices = slot_vertices[material_index]
        categories[category]["material_indices"].append(material_index)
        categories[category]["material_names"].append(name)
        categories[category]["vertices"].update(vertices)
        categories[category]["polygons"] += slot_polygons[material_index]
        categories[category]["loops"] += slot_loops[material_index]
        materials.append(
            {
                "index": material_index,
                "name": name,
                "category": category,
                "polygons": slot_polygons[material_index],
                "loops": slot_loops[material_index],
                "unique_vertices": len(vertices),
                "bounds": bounds(mesh, vertices),
                "images": material_images(material),
                "bone_usage": bone_usage(obj, vertices, deform_groups),
            }
        )

    category_records = []
    for category in sorted(categories):
        item = categories[category]
        vertices = item.pop("vertices")
        category_records.append(
            {
                "category": category,
                **item,
                "unique_vertices": len(vertices),
                "bounds": bounds(mesh, vertices),
                "bone_usage": bone_usage(obj, vertices, deform_groups),
                "morphs": morph_usage(mesh, vertices),
            }
        )

    all_partition_vertices = set().union(*(slot_vertices[index] for index in slot_vertices))
    split_vertex_total = sum(item["unique_vertices"] for item in category_records)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blend_file": bpy.data.filepath,
        "scene": bpy.context.scene.name,
        "source": {
            "mesh": obj.name,
            "armature": armature.name,
            "vertices": len(mesh.vertices),
            "polygons": len(mesh.polygons),
            "loops": len(mesh.loops),
            "materials": len(obj.material_slots),
            "shape_keys_including_basis": len(mesh.shape_keys.key_blocks) if mesh.shape_keys else 0,
            "bones": len(armature.data.bones),
        },
        "materials": materials,
        "categories": category_records,
        "split_budget": {
            "source_vertices_referenced_by_faces": len(all_partition_vertices),
            "sum_of_category_unique_vertices": split_vertex_total,
            "boundary_duplication_if_split": split_vertex_total - len(all_partition_vertices),
            "fh6_vertex_domain_limit": 65535,
            "source_mesh_headroom": 65535 - len(mesh.vertices),
        },
        "license_guard": "Local technical validation only; do not redistribute source-derived components.",
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FH6_PMX_PARTITIONS=" + json.dumps({
        "output": str(output),
        "categories": {
            item["category"]: {
                "vertices": item["unique_vertices"],
                "polygons": item["polygons"],
                "materials": item["material_names"],
                "bones": item["bone_usage"]["bone_group_count"],
                "morphs": len(item["morphs"]),
            }
            for item in category_records
        },
        "boundary_duplication_if_split": report["split_budget"]["boundary_duplication_if_split"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
