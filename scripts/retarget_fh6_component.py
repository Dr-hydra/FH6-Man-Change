#!/usr/bin/env python3
"""Retarget one split Si component to an FH6 component-local donor skeleton."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retarget_garment_to_fh6_female import (
    alignment_matrix,
    bounds,
    capture_source_weights,
    ensure_outputs_absent,
    find_source_armature,
    load_donor_armature,
    render_view,
    replace_weights,
    reset_pose,
    set_local_rotation,
    setup_render,
    sha256,
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--donor-blend", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output-object", required=True)
    parser.add_argument("--component", required=True, choices=("Body", "Outfit", "Helmet", "Head", "Hair"))
    parser.add_argument("--donor-name", required=True)
    parser.add_argument("--pre-milestone", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview-rest-front", required=True, type=Path)
    parser.add_argument("--preview-rest-back", required=True, type=Path)
    parser.add_argument("--preview-rest-side", required=True, type=Path)
    parser.add_argument("--preview-shoulders", type=Path)
    parser.add_argument("--preview-elbows", type=Path)
    parser.add_argument("--preview-wrists", type=Path)
    parser.add_argument("--preview-hips", type=Path)
    parser.add_argument("--preview-knees", type=Path)
    parser.add_argument("--preview-ankles", type=Path)
    parser.add_argument("--preview-head-turn", type=Path)
    parser.add_argument(
        "--conform",
        choices=("global", "bone-translation", "chain-warp", "rigid-head"),
        default="chain-warp",
    )
    parser.add_argument("--rigid-source-bone", default="頭")
    parser.add_argument("--rigid-target-bone", default="Head")
    parser.add_argument("--prune-threshold", type=float, default=0.001)
    parser.add_argument("--max-influences", type=int, default=4)
    return parser.parse_args(argv)


def clean_assignments(
    assignments: list[tuple[str, float]],
    target_map: dict[str, list[tuple[str, float]]],
    threshold: float,
) -> list[tuple[str, str, float]]:
    missing = [source for source, weight in assignments if weight >= threshold and source not in target_map]
    if missing:
        raise RuntimeError(f"No donor mapping for weighted source groups: {sorted(set(missing))}")
    expanded: list[tuple[str, str, float]] = []
    for source, weight in assignments:
        if weight < threshold:
            continue
        for target, target_weight in target_map[source]:
            expanded.append((source, target, weight * target_weight))
    total = sum(weight for _, _, weight in expanded)
    if total <= 0.0:
        raise RuntimeError("Conform cleanup produced a zero-weight vertex")
    return [(source, target, weight / total) for source, target, weight in expanded]


def aligned_world_position(mesh: bpy.types.Object, transform: Matrix, coordinate: Vector) -> Vector:
    return transform @ (mesh.matrix_world @ coordinate)


def apply_global_alignment(mesh: bpy.types.Object, transform: Matrix) -> dict[str, object]:
    local_transform = mesh.matrix_world.inverted() @ transform @ mesh.matrix_world
    mesh.data.transform(local_transform)
    mesh.data.update()
    return {
        "mode": "global",
        "mean_displacement_from_global": 0.0,
        "max_displacement_from_global": 0.0,
    }


def apply_bone_translation_conform(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    source_weights: list[list[tuple[str, float]]],
    target_map: dict[str, list[tuple[str, float]]],
    transform: Matrix,
    threshold: float,
) -> dict[str, object]:
    source_bones = source_armature.data.bones
    target_bones = donor_armature.data.bones
    source_heads = {
        bone.name: transform @ (source_armature.matrix_world @ bone.head_local)
        for bone in source_bones
    }
    target_heads = {
        bone.name: donor_armature.matrix_world @ bone.head_local
        for bone in target_bones
    }
    inverse_mesh_world = mesh.matrix_world.inverted()
    displacements = []
    largest_bone_deltas: dict[tuple[str, str], float] = {}

    for vertex, assignments in zip(mesh.data.vertices, source_weights):
        cleaned = clean_assignments(assignments, target_map, threshold)
        global_position = aligned_world_position(mesh, transform, vertex.co)
        shift = Vector((0.0, 0.0, 0.0))
        for source_name, target_name, weight in cleaned:
            if source_name not in source_heads:
                raise RuntimeError(f"Source bone {source_name!r} is absent from source armature")
            if target_name not in target_heads:
                raise RuntimeError(f"Target bone {target_name!r} is absent from donor armature")
            delta = target_heads[target_name] - source_heads[source_name]
            shift += delta * weight
            key = (source_name, target_name)
            largest_bone_deltas[key] = max(largest_bone_deltas.get(key, 0.0), delta.length)
        vertex.co = inverse_mesh_world @ (global_position + shift)
        displacements.append(shift.length)
    mesh.data.update()
    ranked = sorted(largest_bone_deltas.items(), key=lambda item: (-item[1], item[0]))
    ordered = sorted(displacements)
    percentile_95 = ordered[min(len(ordered) - 1, math.floor(len(ordered) * 0.95))]
    return {
        "mode": "bone-translation",
        "mean_displacement_from_global": sum(displacements) / len(displacements),
        "p95_displacement_from_global": percentile_95,
        "max_displacement_from_global": max(displacements),
        "largest_bone_head_deltas": [
            {"source": source, "target": target, "distance": distance}
            for (source, target), distance in ranked[:30]
        ],
    }


def _rest_frame(armature: bpy.types.Object, bone: bpy.types.Bone) -> Matrix:
    """Return a bone's rest frame in scene/world space."""
    return armature.matrix_world @ bone.matrix_local


def apply_chain_rest_warp(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    source_weights: list[list[tuple[str, float]]],
    target_map: dict[str, list[tuple[str, float]]],
    mapping_entries: dict[str, dict[str, object]],
    transform: Matrix,
    threshold: float,
) -> dict[str, object]:
    """Warp source vertices through source/donor rest frames before skin transfer.

    The global Rz(180)+translation in the mapping is only an initial alignment.
    Each weighted source bone contributes its mapped donor head displacement.
    This corrects limb/seam rest placement while preserving the source surface;
    donor pose deformation then supplies the runtime orientation.
    """
    # The FBX and FH6 donors use different bone-roll conventions.  Until a
    # per-chain axis calibration is available, mapped-head translation is the
    # reliable REST-space operation: it corrects joint/seam placement without
    # rotating or scaling the source surface unexpectedly.
    rest_warp_mode = "mapped_head_translation"
    source_bones = source_armature.data.bones
    donor_bones = donor_armature.data.bones
    source_frames: dict[str, Matrix] = {}
    donor_frames: dict[str, Matrix] = {}
    for name, bone in source_bones.items():
        source_frames[name] = transform @ _rest_frame(source_armature, bone)
    for name, bone in donor_bones.items():
        donor_frames[name] = _rest_frame(donor_armature, bone)

    inverse_mesh_world = mesh.matrix_world.inverted()
    displacements: list[float] = []
    chain_stats: defaultdict[str, list[float]] = defaultdict(list)
    missing_frames: set[str] = set()
    for vertex, assignments in zip(mesh.data.vertices, source_weights):
        cleaned = clean_assignments(assignments, target_map, threshold)
        source_position = transform @ (mesh.matrix_world @ vertex.co)
        warped = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for source_name, target_name, weight in cleaned:
            source_frame = source_frames.get(source_name)
            target_frame = donor_frames.get(target_name)
            if source_frame is None or target_frame is None:
                missing_frames.update(
                    name for name, frame in ((source_name, source_frame), (target_name, target_frame)) if frame is None
                )
                continue
            entry = mapping_entries.get(source_name)
            # Secondary/physics controls often sit far outside the visible
            # surface and are intentionally collapsed to a driven parent.
            # Reconstructing their local frame against the parent would scale
            # hair strands, bows, skirts, or tails. Preserve those shapes with
            # a mapped-head translation; use frame reconstruction only for
            # actual production-chain controls.
            if rest_warp_mode == "mapped_head_translation":
                source_bone = source_bones.get(source_name)
                target_bone = donor_bones.get(target_name)
                if source_bone is None or target_bone is None:
                    missing_frames.update((source_name, target_name))
                    continue
                source_head = transform @ (source_armature.matrix_world @ source_bone.head_local)
                target_head = donor_armature.matrix_world @ target_bone.head_local
                warped += (source_position + (target_head - source_head)) * weight
            else:
                local = source_frame.inverted() @ source_position
                warped += (target_frame @ local) * weight
            total += weight
        if total <= 0.0:
            raise RuntimeError(f"No rest frame available for vertex {vertex.index}")
        warped /= total
        displacement = (warped - source_position).length
        displacements.append(displacement)
        for source_name, _target_name, _weight in cleaned:
            entry = mapping_entries.get(source_name)
            chain = str(entry.get("chain", "unknown")) if entry else "unknown"
            chain_stats[chain].append(displacement)
        vertex.co = inverse_mesh_world @ warped
    mesh.data.update()
    if missing_frames:
        raise RuntimeError(f"Missing source/donor rest frames: {sorted(missing_frames)}")
    ordered = sorted(displacements)
    p95 = ordered[min(len(ordered) - 1, math.floor(len(ordered) * 0.95))] if ordered else 0.0
    chain_report = {
        chain: {
            "vertices": len(values),
            "mean_displacement_m": sum(values) / len(values),
            "max_displacement_m": max(values),
        }
        for chain, values in sorted(chain_stats.items())
        if values
    }
    return {
        "mode": "chain-warp",
        "vertices": len(displacements),
        "mean_displacement_m": sum(displacements) / len(displacements) if displacements else 0.0,
        "p95_displacement_m": p95,
        "max_displacement_m": max(displacements) if displacements else 0.0,
        "chain_stats": chain_report,
        "seam_constraints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "face_neck"],
        "rest_rule": "per-weight mapped source/donor REST head translation with target blend",
    }


def apply_rigid_head_conform(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    transform: Matrix,
    source_bone_name: str,
    target_bone_name: str,
) -> dict[str, object]:
    source_bone = source_armature.data.bones.get(source_bone_name)
    target_bone = donor_armature.data.bones.get(target_bone_name)
    if source_bone is None or target_bone is None:
        raise RuntimeError(
            f"Rigid reference bones missing: source={source_bone_name!r}, target={target_bone_name!r}"
        )
    source_head = transform @ (source_armature.matrix_world @ source_bone.head_local)
    target_head = donor_armature.matrix_world @ target_bone.head_local
    shift = target_head - source_head
    inverse_mesh_world = mesh.matrix_world.inverted()
    for vertex in mesh.data.vertices:
        vertex.co = inverse_mesh_world @ (aligned_world_position(mesh, transform, vertex.co) + shift)
    mesh.data.update()
    return {
        "mode": "rigid-head",
        "source_reference_bone": source_bone_name,
        "target_reference_bone": target_bone_name,
        "head_shift": list(shift),
        "mean_displacement_from_global": shift.length,
        "max_displacement_from_global": shift.length,
    }


def install_donor_modifier(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    component: str,
) -> None:
    for modifier in list(mesh.modifiers):
        if modifier.type == "ARMATURE":
            mesh.modifiers.remove(modifier)
    modifier = mesh.modifiers.new(f"FH6 {component} donor", "ARMATURE")
    modifier.object = donor_armature
    modifier.use_vertex_groups = True
    modifier.use_deform_preserve_volume = False
    source_armature["fh6_probe_exclude"] = True
    source_armature.hide_viewport = True
    source_armature.hide_render = True


def render_pose_gates(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    outputs: dict[str, Path],
) -> dict[str, object]:
    definitions = {
        "shoulders": (("LeftArm", (0.0, 0.0, 35.0)), ("RightArm", (0.0, 0.0, -35.0))),
        "elbows": (("LeftForeArm", (55.0, 0.0, 0.0)), ("RightForeArm", (-55.0, 0.0, 0.0))),
        "wrists": (("LeftHand", (0.0, 0.0, 45.0)), ("RightHand", (0.0, 0.0, -45.0))),
        "hips": (("LeftUpLeg", (25.0, 0.0, 0.0)), ("RightUpLeg", (-25.0, 0.0, 0.0))),
        "knees": (("LeftLeg", (55.0, 0.0, 0.0)), ("RightLeg", (55.0, 0.0, 0.0))),
        "ankles": (("LeftFoot", (20.0, 0.0, 0.0)), ("RightFoot", (20.0, 0.0, 0.0))),
        "head_turn": (("Head", (0.0, 0.0, 35.0)),),
    }
    directions = {
        "knees": Vector((1.0, 0.0, 0.0)),
        "ankles": Vector((1.0, 0.0, 0.0)),
    }
    gates = {}
    for key, rotations in definitions.items():
        if key not in outputs:
            continue
        reset_pose(armature)
        for bone, rotation in rotations:
            set_local_rotation(armature, bone, rotation)
        render_view(scene, camera, directions.get(key, Vector((0.0, -1.0, 0.0))), outputs[key])
        gates[key] = {bone: list(rotation) for bone, rotation in rotations}
    reset_pose(armature)
    return gates


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    donor_blend = args.donor_blend.resolve()
    mapping_path = args.mapping.resolve()
    outputs = {
        "pre": args.pre_milestone.resolve(),
        "blend": args.output_blend.resolve(),
        "report": args.report.resolve(),
        "rest_front": args.preview_rest_front.resolve(),
        "rest_back": args.preview_rest_back.resolve(),
        "rest_side": args.preview_rest_side.resolve(),
    }
    optional_paths = {
        "shoulders": args.preview_shoulders,
        "elbows": args.preview_elbows,
        "wrists": args.preview_wrists,
        "hips": args.preview_hips,
        "knees": args.preview_knees,
        "ankles": args.preview_ankles,
        "head_turn": args.preview_head_turn,
    }
    outputs.update({key: path.resolve() for key, path in optional_paths.items() if path is not None})
    ensure_outputs_absent(list(outputs.values()))
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source_blend}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mesh = bpy.data.objects.get(args.object)
    if mesh is None or mesh.type != "MESH":
        raise RuntimeError(f"Working mesh {args.object!r} not found")
    source_armature = find_source_armature(mesh)
    source_format = str(bpy.context.scene.get("source_format", "unknown")).casefold()
    if source_format == "fbx" and source_armature.data.pose_position != "REST":
        raise RuntimeError("FBX retargeting requires the source armature in REST pose")
    if source_format == "fbx" and "source_format" not in mapping:
        raise RuntimeError("FBX retarget mappings must declare source_format='fbx'")
    mapping_source_format = str(mapping.get("source_format", source_format)).casefold()
    if source_format != "unknown" and mapping_source_format != source_format:
        raise RuntimeError(
            f"Mapping source format {mapping_source_format!r} does not match scene source format {source_format!r}"
        )
    source_weights = capture_source_weights(mesh)
    mapping_entries = {
        str(item["source"]): item for item in mapping.get("mappings", [])
    }
    target_map = {
        source: [
            (str(target["bone"]), float(target["weight"]))
            for target in entry.get("targets", [{"bone": entry["target"], "weight": 1.0}])
        ]
        for source, entry in mapping_entries.items()
    }
    if args.conform == "rigid-head":
        source_group_names = {name for assignments in source_weights for name, weight in assignments if weight > 0.0}
        target_map = {name: [(args.rigid_target_bone, 1.0)] for name in source_group_names}
    else:
        if not target_map:
            raise RuntimeError("Mapping contains no source-to-donor entries")

    # Exact checkpoint before geometry, armature, or vertex-group mutation.
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["pre"]), compress=False, check_existing=False)
    mesh.data = mesh.data.copy()
    donor_armature = load_donor_armature(donor_blend)
    donor_armature.name = f"FH6_{args.donor_name}_Skeleton"
    donor_armature.data.name = f"FH6_{args.donor_name}_SkeletonData"
    bpy.context.scene.collection.objects.link(donor_armature)
    donor_armature.show_in_front = True
    donor_armature.display_type = "WIRE"
    donor_armature["fh6_component"] = args.component
    donor_armature["fh6_donor"] = args.donor_name

    transform = alignment_matrix(mapping)
    if args.conform == "global":
        conform_report = apply_global_alignment(mesh, transform)
    elif args.conform == "bone-translation":
        conform_report = apply_bone_translation_conform(
            mesh,
            source_armature,
            donor_armature,
            source_weights,
            target_map,
            transform,
            args.prune_threshold,
        )
    elif args.conform == "chain-warp":
        conform_report = apply_chain_rest_warp(
            mesh,
            source_armature,
            donor_armature,
            source_weights,
            target_map,
            mapping_entries,
            transform,
            args.prune_threshold,
        )
    else:
        conform_report = apply_rigid_head_conform(
            mesh,
            source_armature,
            donor_armature,
            transform,
            args.rigid_source_bone,
            args.rigid_target_bone,
        )

    weight_report = replace_weights(
        mesh,
        source_weights,
        target_map,
        args.prune_threshold,
        args.max_influences,
    )
    target_bones = {bone.name for bone in donor_armature.data.bones}
    unresolved = sorted(set(weight_report["target_group_names"]) - target_bones)
    if unresolved:
        raise RuntimeError(f"Retargeted groups absent from donor skeleton: {unresolved}")
    install_donor_modifier(mesh, source_armature, donor_armature, args.component)
    mesh.name = args.output_object
    mesh.data.name = f"{args.output_object}_Mesh"
    mesh["fh6_component"] = args.component
    mesh["fh6_donor"] = args.donor_name
    mesh["fh6_conform"] = args.conform
    mesh["fh6_weight_prune_threshold"] = float(args.prune_threshold)
    mesh["fh6_max_influences"] = int(args.max_influences)
    mesh["fh6_weights_retargeted"] = True
    mesh["fh6_probe_exclude"] = False
    for other in bpy.context.scene.objects:
        if other.type == "MESH" and other is not mesh and other.get("fh6_weights_retargeted") is False:
            other["fh6_probe_exclude"] = True

    scene = bpy.context.scene
    scene["retarget_started"] = True
    scene["source_asset_preserved"] = True
    if source_format == "pmx":
        scene["source_pmx_preserved"] = True
    scene["license_guard"] = "Local technical validation only; do not redistribute."
    camera = setup_render(scene, mesh)
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["rest_front"])
    render_view(scene, camera, Vector((0.0, 1.0, 0.0)), outputs["rest_back"])
    render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs["rest_side"])
    pose_gates = render_pose_gates(scene, camera, donor_armature, outputs)
    reset_pose(donor_armature)
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["blend"]), compress=False, check_existing=False)

    minimum, maximum = bounds(mesh)
    influence_histogram: Counter[int] = Counter()
    for vertex in mesh.data.vertices:
        influence_histogram[len([item for item in vertex.groups if item.weight > 1e-8])] += 1
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": f"Retarget Si {args.component} to FH6 {args.donor_name}.",
        "source": {
            "format": source_format,
            "blend": str(source_blend),
            "object": args.object,
            "mapping": str(mapping_path),
            "mapping_sha256": sha256(mapping_path),
        },
        "donor": {
            "blend": str(donor_blend),
            "name": args.donor_name,
            "armature": donor_armature.name,
            "skeleton_bones": len(donor_armature.data.bones),
        },
        "milestones": {
            "pre_mutation": str(outputs["pre"]),
            "retargeted": str(outputs["blend"]),
            "retargeted_sha256": sha256(outputs["blend"]),
        },
        "result": {
            "component": args.component,
            "object": mesh.name,
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "materials": [material.name for material in mesh.data.materials],
            "uv_layers": [layer.name for layer in mesh.data.uv_layers],
            "bounds_min": list(minimum),
            "bounds_max": list(maximum),
            "alignment": mapping["alignment"],
            "conform": conform_report,
            "weights": weight_report,
            "influence_histogram": dict(sorted(influence_histogram.items())),
        },
        "pose_gates": pose_gates,
        "previews": {key: str(path) for key, path in outputs.items() if key not in {"pre", "blend", "report"}},
        "known_loss": [
            "Source secondary physics bones remain collapsed unless the selected FH6 donor drives an equivalent chain.",
            "This milestone is Blender/offline evidence, not an in-game validation result.",
        ],
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_COMPONENT_RETARGET="
        + json.dumps(
            {
                "blend": str(outputs["blend"]),
                "component": args.component,
                "object": mesh.name,
                "vertices": len(mesh.data.vertices),
                "polygons": len(mesh.data.polygons),
                "conform": conform_report,
                "zero_weight_vertices": weight_report["zero_weight_vertices"],
                "vertices_over_limit": weight_report["vertices_over_limit"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
