#!/usr/bin/env python3
"""Retarget Si head and body to one FH6 selectable-face Body-slot skeleton."""

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
    apply_bone_translation_conform,
    install_donor_modifier,
    render_pose_gates,
)
from retarget_garment_to_fh6_female import (
    alignment_matrix,
    bounds,
    capture_source_weights,
    ensure_outputs_absent,
    find_source_armature,
    load_donor_armature,
    render_view,
    reset_pose,
    replace_weights,
    setup_render,
    sha256,
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--donor-blend", required=True, type=Path)
    parser.add_argument("--body-mapping", required=True, type=Path)
    parser.add_argument("--face-mapping", required=True, type=Path)
    parser.add_argument("--body-object", default="Si_Body_SourceSplit_v001")
    parser.add_argument("--head-object", default="Si_Head_SourceSplit_v001")
    parser.add_argument("--output-object", default="Si_DriverBody_FH6_Alice_v001")
    parser.add_argument("--pre-milestone", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview-rest-front", required=True, type=Path)
    parser.add_argument("--preview-rest-back", required=True, type=Path)
    parser.add_argument("--preview-rest-side", required=True, type=Path)
    parser.add_argument("--preview-face", required=True, type=Path)
    parser.add_argument("--preview-hands", required=True, type=Path)
    parser.add_argument("--preview-shoulders", required=True, type=Path)
    parser.add_argument("--preview-wrists", required=True, type=Path)
    parser.add_argument("--preview-head-turn", required=True, type=Path)
    parser.add_argument("--prune-threshold", type=float, default=0.001)
    parser.add_argument("--max-influences", type=int, default=4)
    return parser.parse_args(argv)


def material_histogram(mesh: bpy.types.Object) -> dict[str, int]:
    names = [material.name if material else f"Material_{index}" for index, material in enumerate(mesh.data.materials)]
    counts: Counter[str] = Counter()
    for polygon in mesh.data.polygons:
        if polygon.material_index < 0 or polygon.material_index >= len(names):
            raise RuntimeError(f"Polygon {polygon.index} has invalid material index {polygon.material_index}")
        counts[names[polygon.material_index]] += 1
    return dict(sorted(counts.items()))


def mark_body_materials(body: bpy.types.Object) -> None:
    if not body.material_slots:
        raise RuntimeError("Body source has no material slots")
    for index, slot in enumerate(body.material_slots):
        if slot.material is None:
            raise RuntimeError(f"Body material slot {index} is empty")
        material = slot.material.copy()
        material.name = "DriverBody_Skin" if index == 0 else f"DriverBody_Skin_{index}"
        slot.material = material


def retarget_mesh(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    mapping: dict[str, object],
    threshold: float,
    max_influences: int,
    component: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source_weights = capture_source_weights(mesh)
    bone_map = {item["source"]: item["target"] for item in mapping["mappings"]}
    mesh.data = mesh.data.copy()
    conform = apply_bone_translation_conform(
        mesh,
        source_armature,
        donor_armature,
        source_weights,
        bone_map,
        alignment_matrix(mapping),
        threshold,
    )
    weights = replace_weights(mesh, source_weights, bone_map, threshold, max_influences)
    target_bones = {bone.name for bone in donor_armature.data.bones}
    unresolved = sorted(set(weights["target_group_names"]) - target_bones)
    if unresolved:
        raise RuntimeError(f"{component} target groups are absent from the Alice skeleton: {unresolved}")
    install_donor_modifier(mesh, source_armature, donor_armature, component)
    mesh["fh6_component"] = component
    mesh["fh6_weights_retargeted"] = True
    mesh["fh6_probe_exclude"] = False
    return conform, weights


def join_components(head: bpy.types.Object, body: bpy.types.Object, output_name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()
    result = bpy.context.view_layer.objects.active
    if result is None or result.type != "MESH":
        raise RuntimeError("Head/body join did not produce a mesh")
    result.name = output_name
    result.data.name = f"{output_name}_Mesh"
    result["fh6_component"] = "DriverBody"
    result["fh6_donor"] = "Driver_Alice_F"
    result["fh6_weights_retargeted"] = True
    result["fh6_probe_exclude"] = False
    return result


def render_region(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    mesh: bpy.types.Object,
    output: Path,
    predicate,
) -> dict[str, list[float] | float]:
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices if predicate(mesh.matrix_world @ vertex.co)]
    if not points:
        raise RuntimeError(f"No vertices selected for close-up {output.name}")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    camera["target"] = list(center)
    camera["distance"] = max(max(extent) * 2.2, 0.25)
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), output)
    return {"bounds_min": list(minimum), "bounds_max": list(maximum), "distance": float(camera["distance"])}


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    donor_blend = args.donor_blend.resolve()
    body_mapping_path = args.body_mapping.resolve()
    face_mapping_path = args.face_mapping.resolve()
    outputs = {
        "pre": args.pre_milestone.resolve(),
        "blend": args.output_blend.resolve(),
        "report": args.report.resolve(),
        "rest_front": args.preview_rest_front.resolve(),
        "rest_back": args.preview_rest_back.resolve(),
        "rest_side": args.preview_rest_side.resolve(),
        "face": args.preview_face.resolve(),
        "hands": args.preview_hands.resolve(),
        "shoulders": args.preview_shoulders.resolve(),
        "wrists": args.preview_wrists.resolve(),
        "head_turn": args.preview_head_turn.resolve(),
    }
    ensure_outputs_absent(list(outputs.values()))
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source_blend}")

    body_mapping = json.loads(body_mapping_path.read_text(encoding="utf-8"))
    face_mapping = json.loads(face_mapping_path.read_text(encoding="utf-8"))
    body = bpy.data.objects.get(args.body_object)
    head = bpy.data.objects.get(args.head_object)
    if body is None or body.type != "MESH" or head is None or head.type != "MESH":
        raise RuntimeError("Source head/body mesh objects were not found")
    body_source_armature = find_source_armature(body)
    head_source_armature = find_source_armature(head)
    if body_source_armature is not head_source_armature:
        raise RuntimeError("Source head and body do not share one armature")

    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["pre"]), compress=False, check_existing=False)
    donor_armature = load_donor_armature(donor_blend)
    donor_armature.name = "FH6_Driver_Alice_F_Skeleton"
    donor_armature.data.name = "FH6_Driver_Alice_F_SkeletonData"
    bpy.context.scene.collection.objects.link(donor_armature)
    donor_armature.show_in_front = True
    donor_armature.display_type = "WIRE"
    donor_armature["fh6_component"] = "DriverBody"
    donor_armature["fh6_donor"] = "Driver_Alice_F"

    body_conform, body_weights = retarget_mesh(
        body,
        body_source_armature,
        donor_armature,
        body_mapping,
        args.prune_threshold,
        args.max_influences,
        "Body",
    )
    face_conform, face_weights = retarget_mesh(
        head,
        head_source_armature,
        donor_armature,
        face_mapping,
        args.prune_threshold,
        args.max_influences,
        "Head",
    )
    mark_body_materials(body)
    combined = join_components(head, body, args.output_object)

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj is not combined:
            obj["fh6_probe_exclude"] = True
            obj.hide_viewport = True
            obj.hide_render = True
    body_source_armature["fh6_probe_exclude"] = True
    body_source_armature.hide_viewport = True
    body_source_armature.hide_render = True

    scene = bpy.context.scene
    scene["retarget_started"] = True
    scene["source_pmx_preserved"] = True
    scene["license_guard"] = "Local technical validation only; do not redistribute."
    camera = setup_render(scene, combined)
    full_target = list(camera["target"])
    full_distance = float(camera["distance"])
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["rest_front"])
    render_view(scene, camera, Vector((0.0, 1.0, 0.0)), outputs["rest_back"])
    render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs["rest_side"])
    face_preview = render_region(scene, camera, combined, outputs["face"], lambda point: point.z >= 1.42)
    hands_preview = render_region(
        scene,
        camera,
        combined,
        outputs["hands"],
        lambda point: abs(point.x) >= 0.42 and 0.75 <= point.z <= 1.45,
    )
    camera["target"] = full_target
    camera["distance"] = full_distance
    pose_gates = render_pose_gates(
        scene,
        camera,
        donor_armature,
        {
            "shoulders": outputs["shoulders"],
            "wrists": outputs["wrists"],
            "head_turn": outputs["head_turn"],
        },
    )
    reset_pose(donor_armature)
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["blend"]), compress=False, check_existing=False)

    minimum, maximum = bounds(combined)
    influences: Counter[int] = Counter()
    for vertex in combined.data.vertices:
        influences[len([item for item in vertex.groups if item.weight > 1e-8])] += 1
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Retarget Si head and body to the selectable Driver_Alice_F Body-slot skeleton.",
        "source": {"blend": str(source_blend), "sha256": sha256(source_blend)},
        "donor": {
            "blend": str(donor_blend),
            "sha256": sha256(donor_blend),
            "skeleton_bones": len(donor_armature.data.bones),
        },
        "mappings": {
            "body": {"path": str(body_mapping_path), "sha256": sha256(body_mapping_path)},
            "face": {"path": str(face_mapping_path), "sha256": sha256(face_mapping_path)},
        },
        "milestones": {
            "pre_mutation": str(outputs["pre"]),
            "retargeted": str(outputs["blend"]),
            "retargeted_sha256": sha256(outputs["blend"]),
        },
        "result": {
            "object": combined.name,
            "vertices": len(combined.data.vertices),
            "polygons": len(combined.data.polygons),
            "materials": [material.name for material in combined.data.materials],
            "material_polygon_histogram": material_histogram(combined),
            "bounds_min": list(minimum),
            "bounds_max": list(maximum),
            "influence_histogram": dict(sorted(influences.items())),
            "body": {"conform": body_conform, "weights": body_weights},
            "face": {"conform": face_conform, "weights": face_weights},
        },
        "pose_gates": pose_gates,
        "closeups": {"face": face_preview, "hands": hands_preview},
        "previews": {key: str(path) for key, path in outputs.items() if key not in {"pre", "blend", "report"}},
        "known_loss": [
            "Source facial controls are spatially collapsed onto the Alice donor's driven facial bones.",
            "This milestone is Blender/offline evidence, not an in-game validation result.",
        ],
        "license_guard": "Local technical validation only; do not redistribute the derived model.",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_DRIVER_BODY_RETARGET="
        + json.dumps(
            {
                "blend": str(outputs["blend"]),
                "object": combined.name,
                "vertices": len(combined.data.vertices),
                "polygons": len(combined.data.polygons),
                "skeleton_bones": len(donor_armature.data.bones),
                "materials": material_histogram(combined),
                "body_zero_weights": body_weights["zero_weight_vertices"],
                "face_zero_weights": face_weights["zero_weight_vertices"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
