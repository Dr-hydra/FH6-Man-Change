#!/usr/bin/env python3
"""Retarget all FBX meshes for one Display component package.

This is the FBX-first batch counterpart to ``retarget_fh6_component.py``.  It
keeps the native LOD mesh boundaries until all objects have been warped and
weighted, then joins the package into one exportable mesh sharing one physical
donor skeleton.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retarget_fh6_component import (
    apply_chain_rest_warp,
    alignment_matrix,
    install_donor_modifier,
    render_pose_gates,
)
from retarget_garment_to_fh6_female import (
    bounds,
    capture_source_weights,
    ensure_outputs_absent,
    find_source_armature,
    load_donor_armature,
    render_view,
    replace_weights,
    reset_pose,
    setup_render,
    sha256,
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--donor-blend", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--roles", required=True, help="Comma-separated FBX roles, e.g. head,hair")
    parser.add_argument("--component", required=True, choices=("HeadHair", "BodyGarment"))
    parser.add_argument("--donor-name", required=True)
    parser.add_argument("--output-object", required=True)
    parser.add_argument("--source-prefix", default="FBX_", help="Generated source mesh name prefix")
    parser.add_argument("--pre-milestone", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview-dir", required=True, type=Path)
    parser.add_argument("--prune-threshold", type=float, default=0.001)
    parser.add_argument("--max-influences", type=int, default=4)
    return parser.parse_args(argv)


def semantic_material_alias(obj: bpy.types.Object) -> dict[str, str]:
    role = str(obj.get("source_role", "")).casefold()
    source_name = str(obj.get("source_object_name", obj.name)).casefold()
    if role == "body":
        alias = "肌"
    elif role == "garment":
        if "cloth_02" in source_name:
            alias = "Cloth1Alpha"
        elif "cloth_03" in source_name:
            alias = "Cloth1"
        else:
            alias = "Cloth1"
    elif role == "head":
        if "brow" in source_name:
            alias = "睫眉"
        elif "eyeshadow" in source_name:
            alias = "目影"
        elif "iris" in source_name:
            alias = "目"
        else:
            alias = "面"
    elif role == "hair":
        alias = "发影" if "shadow" in source_name else "发"
    else:
        alias = obj.name
    result: dict[str, str] = {}
    for slot in obj.material_slots:
        if slot.material is None:
            continue
        old = slot.material.name
        # Each FBX source material is copied before aliasing so an unrelated
        # source/reference object cannot observe the export-only name.
        existing = bpy.data.materials.get(alias)
        if existing is not None and existing is not slot.material:
            material = existing
        else:
            material = slot.material.copy()
            material.name = alias
        material["fh6_source_material_name"] = old
        material["fh6_source_role"] = role
        slot.material = material
        result[old] = alias
    return result


def build_target_map(mapping: dict) -> tuple[dict[str, dict], dict[str, list[tuple[str, float]]]]:
    entries = {str(item["source"]): item for item in mapping.get("mappings", [])}
    target_map: dict[str, list[tuple[str, float]]] = {}
    for source, entry in entries.items():
        raw_targets = entry.get("targets") or [{"bone": entry["target"], "weight": 1.0}]
        targets = [(str(item["bone"]), float(item["weight"])) for item in raw_targets if float(item["weight"]) > 1e-8]
        total = sum(weight for _name, weight in targets)
        if total <= 0.0:
            raise ValueError(f"Mapping entry {source!r} has no positive target weight")
        target_map[source] = [(name, weight / total) for name, weight in targets]
    return entries, target_map


def verify_weights(mesh: bpy.types.Object, max_influences: int) -> dict[str, object]:
    histogram: Counter[int] = Counter()
    sums: list[float] = []
    over_limit = 0
    zero_weight = 0
    for vertex in mesh.data.vertices:
        weights = [item.weight for item in vertex.groups if item.weight > 1e-8]
        histogram[len(weights)] += 1
        over_limit += int(len(weights) > max_influences)
        zero_weight += int(not weights)
        if weights:
            sums.append(sum(weights))
    return {
        "influence_histogram": dict(sorted(histogram.items())),
        "vertices_over_limit": over_limit,
        "zero_weight_vertices": zero_weight,
        "weight_sum_min": min(sums) if sums else 0.0,
        "weight_sum_max": max(sums) if sums else 0.0,
    }


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    donor_blend = args.donor_blend.resolve()
    mapping_path = args.mapping.resolve()
    preview_dir = args.preview_dir.resolve()
    outputs = [args.pre_milestone.resolve(), args.output_blend.resolve(), args.report.resolve()]
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_paths = [preview_dir / name for name in ("rest-front.png", "rest-back.png", "rest-side.png", "pose-wrists.png", "pose-ankles.png", "pose-head-turn.png")]
    ensure_outputs_absent(outputs + preview_paths)

    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match source blend {source_blend}")
    scene = bpy.context.scene
    if str(scene.get("source_format", "")).casefold() != "fbx":
        raise RuntimeError("FBX Display retarget requires scene source_format='fbx'")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if str(mapping.get("source_format", "")).casefold() != "fbx":
        raise RuntimeError("Mapping must declare source_format='fbx'")
    mapping_entries, target_map = build_target_map(mapping)
    roles = {item.strip().casefold() for item in args.roles.split(",") if item.strip()}
    if not roles:
        raise ValueError("At least one source role is required")
    meshes = sorted(
        [
            obj
            for obj in bpy.data.objects
            if obj.type == "MESH"
            and obj.name.startswith(args.source_prefix)
            and str(obj.get("source_role", "")).casefold() in roles
        ],
        key=lambda obj: obj.name.casefold(),
    )
    if not meshes:
        raise RuntimeError(f"No FBX meshes found for roles {sorted(roles)}")
    source_armatures = {find_source_armature(mesh) for mesh in meshes}
    if len(source_armatures) != 1:
        raise RuntimeError(f"Expected one shared FBX source armature, found {len(source_armatures)}")
    source_armature = next(iter(source_armatures))
    if source_armature.data.pose_position != "REST":
        raise RuntimeError("FBX source armature must be in REST pose")

    # Immutable checkpoint before donor insertion or mesh/weight mutation.
    bpy.ops.wm.save_as_mainfile(filepath=str(args.pre_milestone.resolve()), compress=False, check_existing=False)

    donor_armature = load_donor_armature(donor_blend)
    donor_armature.name = f"FH6_{args.donor_name}_Skeleton"
    donor_armature.data.name = f"FH6_{args.donor_name}_SkeletonData"
    scene.collection.objects.link(donor_armature)
    donor_armature.show_in_front = True
    donor_armature.display_type = "WIRE"
    donor_armature["fh6_component"] = args.component
    donor_armature["fh6_donor"] = args.donor_name

    transform = alignment_matrix(mapping)
    per_object: list[dict[str, object]] = []
    for mesh in meshes:
        mesh.data = mesh.data.copy()
        source_weights = capture_source_weights(mesh)
        conform = apply_chain_rest_warp(
            mesh,
            source_armature,
            donor_armature,
            source_weights,
            target_map,
            mapping_entries,
            transform,
            args.prune_threshold,
        )
        weight_report = replace_weights(mesh, source_weights, target_map, args.prune_threshold, args.max_influences)
        unresolved = sorted(set(weight_report["target_group_names"]) - {bone.name for bone in donor_armature.data.bones})
        if unresolved:
            raise RuntimeError(f"{mesh.name} has donor groups absent from skeleton: {unresolved}")
        install_donor_modifier(mesh, source_armature, donor_armature, args.component)
        aliases = semantic_material_alias(mesh)
        mesh["fh6_component"] = args.component
        mesh["fh6_donor"] = args.donor_name
        mesh["fh6_conform"] = "chain-warp"
        mesh["fh6_weights_retargeted"] = True
        mesh["fh6_probe_exclude"] = False
        per_object.append({
            "object": mesh.name,
            "source_role": mesh.get("source_role"),
            "source_object": mesh.get("source_object_name"),
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "conform": conform,
            "weights": weight_report,
            "material_aliases": aliases,
        })

    # Hide all non-package source meshes from renders/probes, then combine the
    # package so the engine-neutral exporter sees one dense vertex domain.
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj not in meshes:
            obj["fh6_probe_exclude"] = True
            obj.hide_viewport = True
            obj.hide_render = True
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    output_mesh = bpy.context.view_layer.objects.active
    output_mesh.name = args.output_object
    output_mesh.data.name = f"{args.output_object}_Mesh"
    output_mesh["fh6_component"] = args.component
    output_mesh["fh6_donor"] = args.donor_name
    output_mesh["fh6_conform"] = "chain-warp"
    output_mesh["fh6_source_format"] = "fbx"
    output_mesh["fh6_weights_retargeted"] = True
    output_mesh["fh6_probe_exclude"] = False
    source_armature["fh6_probe_exclude"] = True
    source_armature.hide_viewport = True
    source_armature.hide_render = True

    camera = setup_render(scene, output_mesh)
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), preview_paths[0])
    render_view(scene, camera, Vector((0.0, 1.0, 0.0)), preview_paths[1])
    render_view(scene, camera, Vector((1.0, 0.0, 0.0)), preview_paths[2])
    pose_outputs = {"wrists": preview_paths[3], "ankles": preview_paths[4], "head_turn": preview_paths[5]}
    pose_gates = render_pose_gates(scene, camera, donor_armature, pose_outputs)
    reset_pose(donor_armature)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()), compress=False, check_existing=False)

    minimum, maximum = bounds(output_mesh)
    weight_validation = verify_weights(output_mesh, args.max_influences)
    if weight_validation["vertices_over_limit"] or weight_validation["zero_weight_vertices"]:
        raise RuntimeError(f"Final package weight validation failed: {weight_validation}")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "FBX-first FH6 Display component retarget with donor REST-frame chain warp.",
        "source": {"format": "fbx", "blend": str(source_blend), "sha256": sha256(source_blend), "roles": sorted(roles), "mapping": str(mapping_path), "mapping_sha256": sha256(mapping_path)},
        "donor": {"blend": str(donor_blend), "name": args.donor_name, "armature": donor_armature.name, "bones": len(donor_armature.data.bones)},
        "milestones": {"pre_mutation": str(args.pre_milestone.resolve()), "retargeted": str(args.output_blend.resolve()), "retargeted_sha256": sha256(args.output_blend.resolve())},
        "result": {"component": args.component, "object": output_mesh.name, "vertices": len(output_mesh.data.vertices), "polygons": len(output_mesh.data.polygons), "bounds_min": list(minimum), "bounds_max": list(maximum), "weights": weight_validation, "materials": [slot.material.name if slot.material else None for slot in output_mesh.material_slots]},
        "objects": per_object,
        "alignment": mapping["alignment"],
        "chain_warp": {"required": True, "rule": "per-weight source/donor REST-frame reconstruction", "seam_constraints": mapping.get("chain_warp", {}).get("seam_constraints", [])},
        "pose_gates": pose_gates,
        "previews": {"rest_front": str(preview_paths[0]), "rest_back": str(preview_paths[1]), "rest_side": str(preview_paths[2]), "wrists": str(preview_paths[3]), "ankles": str(preview_paths[4]), "head_turn": str(preview_paths[5])},
        "validation_level": {"data": True, "blender_visual": True, "modelbin": False, "offline_game": False},
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    args.report.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_FBX_DISPLAY_RETARGET=" + json.dumps({"output": str(args.output_blend.resolve()), "component": args.component, "objects": len(meshes), "vertices": len(output_mesh.data.vertices), "weight_validation": weight_validation}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
