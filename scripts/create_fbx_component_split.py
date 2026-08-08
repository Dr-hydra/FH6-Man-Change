#!/usr/bin/env python3
"""Create one native-LOD FH6 component source scene from an immutable FBX baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import bpy


COMPONENT_COLLECTIONS = {
    "body": "FBX_BODY_SOURCE",
    "garment": "FBX_GARMENT_SOURCE",
    "hair": "FBX_HAIR_SOURCE",
    "head": "FBX_HEAD_SOURCE",
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lod", default="lod0")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_armature(source: bpy.types.Object, collection: bpy.types.Collection) -> bpy.types.Object:
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = "Si_FBX_SourceRig_REST"
    duplicate.data.name = "Si_FBX_SourceRig_REST"
    duplicate.data.pose_position = "REST"
    duplicate.hide_select = False
    duplicate.hide_viewport = False
    duplicate.hide_render = True
    duplicate["source_format"] = "fbx"
    duplicate["source_pose_position"] = "REST"
    duplicate["fh6_probe_exclude"] = False
    collection.objects.link(duplicate)
    return duplicate


def duplicate_mesh(
    source: bpy.types.Object,
    target_armature: bpy.types.Object,
    collection: bpy.types.Collection,
    role: str,
) -> bpy.types.Object:
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = f"Si_FBX_{role.title()}_{source.name}"
    duplicate.data.name = duplicate.name
    duplicate.parent = None
    duplicate.matrix_world = source.matrix_world.copy()
    duplicate.hide_select = False
    duplicate.hide_viewport = False
    duplicate.hide_render = False
    duplicate["source_format"] = "fbx"
    duplicate["source_object_name"] = source.name
    duplicate["source_role"] = role
    duplicate["source_lod"] = source.get("source_lod", "lod0")
    duplicate["fh6_weights_retargeted"] = False
    duplicate["fh6_probe_exclude"] = False
    collection.objects.link(duplicate)
    for modifier in duplicate.modifiers:
        if modifier.type == "ARMATURE":
            modifier.object = target_armature
    return duplicate


def mesh_record(obj: bpy.types.Object) -> dict:
    armature_modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE" and modifier.object]
    deform_names = {
        bone.name
        for modifier in armature_modifiers
        for bone in modifier.object.data.bones
        if bone.use_deform
    }
    deform_indices = {group.index for group in obj.vertex_groups if group.name in deform_names}
    histogram: Counter[int] = Counter()
    over_four = 0
    zero_weight = 0
    non_normalized = 0
    for vertex in obj.data.vertices:
        weights = [element.weight for element in vertex.groups if element.group in deform_indices and element.weight > 1e-8]
        histogram[len(weights)] += 1
        over_four += int(len(weights) > 4)
        zero_weight += int(not weights)
        non_normalized += int(bool(weights) and abs(sum(weights) - 1.0) > 1e-4)
    return {
        "object": obj.name,
        "source_object": obj.get("source_object_name"),
        "role": obj["source_role"],
        "source_lod": obj["source_lod"],
        "vertices": len(obj.data.vertices),
        "triangles": sum(max(0, len(polygon.vertices) - 2) for polygon in obj.data.polygons),
        "uv_layers": [layer.name for layer in obj.data.uv_layers],
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "shape_keys_excluding_basis": max(0, len(obj.data.shape_keys.key_blocks) - 1) if obj.data.shape_keys else 0,
        "deform_vertex_groups": len(deform_indices),
        "influence_histogram": dict(sorted(histogram.items())),
        "vertices_over_four_influences": over_four,
        "zero_weight_vertices": zero_weight,
        "weighted_vertices_not_normalized": non_normalized,
        "armature": armature_modifiers[0].object.name if armature_modifiers else None,
    }


def main() -> None:
    args = parse_args()
    source_blend = args.source_blend.resolve()
    output_blend = args.blend.resolve()
    report_path = args.report.resolve()
    if not source_blend.is_file():
        raise FileNotFoundError(source_blend)
    if output_blend.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing FBX component milestone")

    bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    if bpy.context.scene.get("baseline_kind") != "immutable_fbx_source":
        raise ValueError("Source blend is not an immutable FBX baseline")
    source_armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE" and obj.get("source_immutable", False)]
    if len(source_armatures) != 1:
        raise ValueError(f"Expected one immutable FBX armature, found {len(source_armatures)}")

    for obj in bpy.context.scene.objects:
        obj["fh6_probe_exclude"] = True
        if obj.get("source_immutable", False):
            obj.hide_select = True
            obj.hide_render = True

    rig_collection = bpy.data.collections.new("FBX_SOURCE_RIG")
    bpy.context.scene.collection.children.link(rig_collection)
    working_armature = duplicate_armature(source_armatures[0], rig_collection)
    component_collections = {}
    for role, collection_name in COMPONENT_COLLECTIONS.items():
        collection = bpy.data.collections.new(collection_name)
        collection["source_format"] = "fbx"
        collection["source_role"] = role
        bpy.context.scene.collection.children.link(collection)
        component_collections[role] = collection

    source_meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and obj.get("source_immutable", False)
        and obj.get("source_lod") == args.lod
        and obj.get("source_role") in COMPONENT_COLLECTIONS
    ]
    if not source_meshes:
        raise ValueError(f"No FBX component meshes found for {args.lod}")
    working_meshes = [
        duplicate_mesh(source, working_armature, component_collections[source["source_role"]], source["source_role"])
        for source in sorted(source_meshes, key=lambda item: item.name.casefold())
    ]
    records = [mesh_record(obj) for obj in working_meshes]
    component_totals = {}
    for role in COMPONENT_COLLECTIONS:
        role_records = [record for record in records if record["role"] == role]
        component_totals[role] = {
            "objects": len(role_records),
            "vertices": sum(record["vertices"] for record in role_records),
            "triangles": sum(record["triangles"] for record in role_records),
        }

    hard_errors = []
    for record in records:
        if not record["uv_layers"]:
            hard_errors.append({"code": "missing_uv", "object": record["object"]})
        if record["vertices_over_four_influences"]:
            hard_errors.append({"code": "more_than_four_influences", "object": record["object"]})
        if record["zero_weight_vertices"]:
            hard_errors.append({"code": "zero_weight_vertices", "object": record["object"]})
        if record["weighted_vertices_not_normalized"]:
            hard_errors.append({"code": "weights_not_normalized", "object": record["object"]})
    for role, totals in component_totals.items():
        if totals["vertices"] > 65_535:
            hard_errors.append(
                {
                    "code": "r16_component_vertex_domain_exceeded",
                    "role": role,
                    "vertices": totals["vertices"],
                    "limit": 65_535,
                }
            )

    scene = bpy.context.scene
    scene["baseline_kind"] = "fbx_component_source"
    scene["source_format"] = "fbx"
    scene["source_asset_preserved"] = True
    scene["source_blend"] = str(source_blend)
    scene["source_lod"] = args.lod
    scene["source_pose_position"] = "REST"
    scene["fh6_export_ready"] = False
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), compress=False, check_existing=False)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": f"FBX-first {args.lod.upper()} source components prior to FH6 donor retargeting.",
        "source": {"blend": str(source_blend), "blend_sha256": sha256(source_blend), "lod": args.lod},
        "output": {"blend": str(output_blend), "blend_sha256": sha256(output_blend)},
        "armature": {"object": working_armature.name, "bones": len(working_armature.data.bones), "pose_position": working_armature.data.pose_position},
        "totals": {
            "objects": len(records),
            "vertices": sum(record["vertices"] for record in records),
            "triangles": sum(record["triangles"] for record in records),
            "components": component_totals,
        },
        "meshes": records,
        "validation": {"hard_error_count": len(hard_errors), "hard_errors": hard_errors},
        "policies": {
            "primary_source_format": "fbx",
            "pose_position": "REST",
            "effects_included": False,
            "shadow_proxies_included": False,
            "pmx_role": "reference_only",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_FBX_COMPONENT_SUMMARY=" + json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))
    print(f"FH6_FBX_COMPONENT_BLEND={output_blend}")
    print(f"FH6_FBX_COMPONENT_REPORT={report_path}")
    if hard_errors:
        raise RuntimeError(f"FBX component validation failed: {hard_errors}")


if __name__ == "__main__":
    main()
