#!/usr/bin/env python3
"""Create the first Si garment milestone bound to an FH6 female donor skeleton."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--donor-blend", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--object", default="Si_Garment_Cloth1_Upper_Prototype")
    parser.add_argument("--output-object", default="Si_Garment_FH6_Female_Retarget_v002")
    parser.add_argument("--donor-name", default="Upper_Shirt_Tucked_F_Driver")
    parser.add_argument("--pre-milestone", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview-rest-front", required=True, type=Path)
    parser.add_argument("--preview-rest-back", required=True, type=Path)
    parser.add_argument("--preview-rest-side", required=True, type=Path)
    parser.add_argument("--preview-shoulders", required=True, type=Path)
    parser.add_argument("--preview-elbows", required=True, type=Path)
    parser.add_argument("--preview-wrists", required=True, type=Path)
    parser.add_argument("--preview-hips", type=Path)
    parser.add_argument("--preview-knees", type=Path)
    parser.add_argument("--preview-ankles", type=Path)
    parser.add_argument("--prune-threshold", type=float, default=0.001)
    parser.add_argument("--max-influences", type=int, default=4)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_outputs_absent(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)


def find_source_armature(mesh: bpy.types.Object) -> bpy.types.Object:
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            return modifier.object
    raise RuntimeError(f"{mesh.name!r} has no source armature modifier")


def load_donor_armature(path: Path) -> bpy.types.Object:
    with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    objects = [obj for obj in data_to.objects if obj is not None]
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one donor armature, found {len(armatures)}")
    donor = armatures[0]
    for obj in objects:
        if obj is donor:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    return donor


def alignment_matrix(mapping: dict[str, object]) -> Matrix:
    alignment = mapping["alignment"]
    angle = math.radians(float(alignment["rotation_z_degrees"]))
    translation = Vector(alignment["translation"])
    scale = float(alignment.get("scale", 1.0))
    return Matrix.Translation(translation) @ Matrix.Rotation(angle, 4, "Z") @ Matrix.Scale(scale, 4)


def capture_source_weights(mesh: bpy.types.Object) -> list[list[tuple[str, float]]]:
    names = {group.index: group.name for group in mesh.vertex_groups}
    return [
        [(names[item.group], float(item.weight)) for item in vertex.groups if item.group in names and item.weight > 0.0]
        for vertex in mesh.data.vertices
    ]


def replace_weights(
    mesh: bpy.types.Object,
    source_weights: list[list[tuple[str, float]]],
    target_map: dict[str, str | list[tuple[str, float]]],
    threshold: float,
    max_influences: int,
) -> dict[str, object]:
    before_histogram: Counter[int] = Counter(len(items) for items in source_weights)
    before_groups = len(mesh.vertex_groups)
    for group in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(group)

    cleaned: list[list[tuple[str, float]]] = []
    target_names: set[str] = set()
    dropped_below_threshold = 0
    collapsed_assignments = 0
    pruned_after_collapse = 0
    for assignments in source_weights:
        combined: defaultdict[str, float] = defaultdict(float)
        for source_name, weight in assignments:
            if weight < threshold:
                dropped_below_threshold += 1
                continue
            mapped = target_map.get(source_name)
            if mapped is None:
                raise RuntimeError(f"No donor mapping for weighted source group {source_name!r}")
            if isinstance(mapped, str):
                targets = [(mapped, 1.0)]
            else:
                targets = mapped
            for target_name, target_weight in targets:
                combined[target_name] += weight * target_weight
                if target_name != source_name:
                    collapsed_assignments += 1
        ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) > max_influences:
            pruned_after_collapse += len(ranked) - max_influences
            ranked = ranked[:max_influences]
        total = sum(weight for _, weight in ranked)
        if total <= 0.0:
            raise RuntimeError("Weight cleanup produced a zero-weight vertex")
        normalized = [(name, weight / total) for name, weight in ranked]
        cleaned.append(normalized)
        target_names.update(name for name, _ in normalized)

    groups = {name: mesh.vertex_groups.new(name=name) for name in sorted(target_names)}
    for vertex_index, assignments in enumerate(cleaned):
        for name, weight in assignments:
            groups[name].add([vertex_index], weight, "REPLACE")

    after_histogram: Counter[int] = Counter(len(items) for items in cleaned)
    minimum_sum = min(sum(weight for _, weight in items) for items in cleaned)
    maximum_sum = max(sum(weight for _, weight in items) for items in cleaned)
    return {
        "source_vertex_groups": before_groups,
        "target_vertex_groups": len(target_names),
        "target_group_names": sorted(target_names),
        "before_influence_histogram": dict(sorted(before_histogram.items())),
        "after_influence_histogram": dict(sorted(after_histogram.items())),
        "dropped_assignments_below_threshold": dropped_below_threshold,
        "collapsed_assignments": collapsed_assignments,
        "pruned_assignments_after_collapse": pruned_after_collapse,
        "minimum_weight_sum": minimum_sum,
        "maximum_weight_sum": maximum_sum,
        "zero_weight_vertices": sum(1 for items in cleaned if not items),
        "vertices_over_limit": sum(1 for items in cleaned if len(items) > max_influences),
    }


def install_donor_modifier(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
) -> None:
    for modifier in list(mesh.modifiers):
        if modifier.type == "ARMATURE":
            mesh.modifiers.remove(modifier)
    modifier = mesh.modifiers.new("FH6 female garment donor", "ARMATURE")
    modifier.object = donor_armature
    modifier.use_vertex_groups = True
    modifier.use_deform_preserve_volume = False
    source_armature["fh6_probe_exclude"] = True
    source_armature.hide_viewport = True
    source_armature.hide_render = True


def bounds(mesh: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def setup_render(scene: bpy.types.Scene, mesh: bpy.types.Object) -> bpy.types.Object:
    minimum, maximum = bounds(mesh)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    camera_data = bpy.data.cameras.new("FH6 Retarget Gate Camera")
    camera = bpy.data.objects.new("FH6 Retarget Gate Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 58
    camera["target"] = list(center)
    camera["distance"] = max(max(extent) * 2.15, 0.8)
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
    for obj in scene.objects:
        obj.hide_render = obj is not mesh and obj is not camera
    return camera


def render_view(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    direction: Vector,
    output: Path,
) -> None:
    target = Vector(camera["target"])
    camera.location = target + direction.normalized() * float(camera["distance"])
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output.resolve())
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def reset_pose(armature: bpy.types.Object) -> None:
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.location = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def set_local_rotation(armature: bpy.types.Object, name: str, xyz_degrees: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(name)
    if bone is None:
        raise RuntimeError(f"Pose-gate donor bone {name!r} not found")
    bone.rotation_mode = "XYZ"
    bone.rotation_euler = tuple(math.radians(value) for value in xyz_degrees)


def render_pose_gates(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    outputs: dict[str, Path],
) -> dict[str, object]:
    gates: dict[str, object] = {}

    reset_pose(armature)
    set_local_rotation(armature, "LeftArm", (0.0, 0.0, 35.0))
    set_local_rotation(armature, "RightArm", (0.0, 0.0, -35.0))
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["shoulders"])
    gates["shoulders"] = {"LeftArm": [0.0, 0.0, 35.0], "RightArm": [0.0, 0.0, -35.0]}

    reset_pose(armature)
    set_local_rotation(armature, "LeftForeArm", (55.0, 0.0, 0.0))
    set_local_rotation(armature, "RightForeArm", (-55.0, 0.0, 0.0))
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["elbows"])
    gates["elbows"] = {"LeftForeArm": [55.0, 0.0, 0.0], "RightForeArm": [-55.0, 0.0, 0.0]}

    reset_pose(armature)
    set_local_rotation(armature, "LeftHand", (0.0, 0.0, 45.0))
    set_local_rotation(armature, "RightHand", (0.0, 0.0, -45.0))
    render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["wrists"])
    gates["wrists"] = {"LeftHand": [0.0, 0.0, 45.0], "RightHand": [0.0, 0.0, -45.0]}

    if "hips" in outputs:
        reset_pose(armature)
        set_local_rotation(armature, "LeftUpLeg", (25.0, 0.0, 0.0))
        set_local_rotation(armature, "RightUpLeg", (-25.0, 0.0, 0.0))
        render_view(scene, camera, Vector((0.0, -1.0, 0.0)), outputs["hips"])
        gates["hips"] = {"LeftUpLeg": [25.0, 0.0, 0.0], "RightUpLeg": [-25.0, 0.0, 0.0]}

    if "knees" in outputs:
        reset_pose(armature)
        set_local_rotation(armature, "LeftLeg", (55.0, 0.0, 0.0))
        set_local_rotation(armature, "RightLeg", (55.0, 0.0, 0.0))
        render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs["knees"])
        gates["knees"] = {"LeftLeg": [55.0, 0.0, 0.0], "RightLeg": [55.0, 0.0, 0.0]}

    if "ankles" in outputs:
        reset_pose(armature)
        set_local_rotation(armature, "LeftFoot", (20.0, 0.0, 0.0))
        set_local_rotation(armature, "RightFoot", (20.0, 0.0, 0.0))
        render_view(scene, camera, Vector((1.0, 0.0, 0.0)), outputs["ankles"])
        gates["ankles"] = {"LeftFoot": [20.0, 0.0, 0.0], "RightFoot": [20.0, 0.0, 0.0]}

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
        "shoulders": args.preview_shoulders.resolve(),
        "elbows": args.preview_elbows.resolve(),
        "wrists": args.preview_wrists.resolve(),
    }
    for key, value in (("hips", args.preview_hips), ("knees", args.preview_knees), ("ankles", args.preview_ankles)):
        if value is not None:
            outputs[key] = value.resolve()
    ensure_outputs_absent(list(outputs.values()))
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match --source-blend {source_blend}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    bone_map = {item["source"]: item["target"] for item in mapping["mappings"]}
    mesh = bpy.data.objects.get(args.object)
    if mesh is None or mesh.type != "MESH":
        raise RuntimeError(f"Working mesh {args.object!r} not found")
    source_armature = find_source_armature(mesh)

    # Required mutation checkpoint: an exact copy is saved before adding the
    # donor skeleton, transforming geometry, or replacing vertex groups.
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["pre"]), compress=False, check_existing=False)

    source_weights = capture_source_weights(mesh)
    mesh.data = mesh.data.copy()
    transform = alignment_matrix(mapping)
    local_transform = mesh.matrix_world.inverted() @ transform @ mesh.matrix_world
    mesh.data.transform(local_transform)
    mesh.data.update()

    donor_armature = load_donor_armature(donor_blend)
    donor_armature.name = f"FH6_{args.donor_name}_Skeleton"
    donor_armature.data.name = f"FH6_{args.donor_name}_SkeletonData"
    bpy.context.scene.collection.objects.link(donor_armature)
    donor_armature.show_in_front = True
    donor_armature.display_type = "WIRE"
    donor_armature["fh6_component"] = "Garment"
    donor_armature["fh6_donor"] = args.donor_name

    weight_report = replace_weights(
        mesh,
        source_weights,
        bone_map,
        args.prune_threshold,
        args.max_influences,
    )
    target_bones = {bone.name for bone in donor_armature.data.bones}
    unresolved = sorted(set(weight_report["target_group_names"]) - target_bones)
    if unresolved:
        raise RuntimeError(f"Retargeted groups absent from donor skeleton: {unresolved}")
    install_donor_modifier(mesh, source_armature, donor_armature)
    mesh.name = args.output_object
    mesh.data.name = f"{args.output_object}_Mesh"
    mesh["fh6_component"] = "Garment"
    mesh["fh6_donor"] = args.donor_name
    mesh["fh6_weight_prune_threshold"] = float(args.prune_threshold)
    mesh["fh6_max_influences"] = int(args.max_influences)
    mesh["fh6_alignment_rotation_z_degrees"] = float(mapping["alignment"]["rotation_z_degrees"])
    mesh["fh6_alignment_translation"] = list(mapping["alignment"]["translation"])
    bpy.context.scene["retarget_started"] = True
    bpy.context.scene["retarget_donor"] = args.donor_name
    bpy.context.scene["source_pmx_preserved"] = True
    bpy.context.scene["license_guard"] = "Local technical validation only; do not redistribute."

    camera = setup_render(bpy.context.scene, mesh)
    render_view(bpy.context.scene, camera, Vector((0.0, -1.0, 0.0)), outputs["rest_front"])
    render_view(bpy.context.scene, camera, Vector((0.0, 1.0, 0.0)), outputs["rest_back"])
    render_view(bpy.context.scene, camera, Vector((1.0, 0.0, 0.0)), outputs["rest_side"])
    pose_gates = render_pose_gates(bpy.context.scene, camera, donor_armature, outputs)

    reset_pose(donor_armature)
    bpy.ops.wm.save_as_mainfile(filepath=str(outputs["blend"]), compress=False, check_existing=False)
    minimum, maximum = bounds(mesh)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": f"Local Si garment retarget milestone using the FH6 {args.donor_name} donor skeleton.",
        "source": {
            "blend": str(source_blend),
            "object": args.object,
            "mapping": str(mapping_path),
            "mapping_sha256": sha256(mapping_path),
        },
        "donor": {
            "blend": str(donor_blend),
            "armature": donor_armature.name,
            "skeleton_bones": len(donor_armature.data.bones),
        },
        "milestones": {
            "pre_mutation": str(outputs["pre"]),
            "retargeted": str(outputs["blend"]),
            "retargeted_sha256": sha256(outputs["blend"]),
        },
        "result": {
            "object": mesh.name,
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "materials": len(mesh.data.materials),
            "uv_layers": [layer.name for layer in mesh.data.uv_layers],
            "shape_keys": 0 if mesh.data.shape_keys is None else len(mesh.data.shape_keys.key_blocks),
            "bounds_min": list(minimum),
            "bounds_max": list(maximum),
            "alignment": mapping["alignment"],
            "weights": weight_report,
        },
        "pose_gates": pose_gates,
        "previews": {key: str(path) for key, path in outputs.items() if key not in {"pre", "blend", "report"}},
        "known_loss": [
            "Source-specific garment physics controls are collapsed to donor-driven body roles; long sleeves, ribbons, skirt panels, and tail will not retain their original secondary simulation.",
            "This milestone validates donor binding and four-influence skinning only; it is not yet a modelbin export or in-game result.",
        ],
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_GARMENT_RETARGET="
        + json.dumps(
            {
                "blend": str(outputs["blend"]),
                "report": str(outputs["report"]),
                "vertices": len(mesh.data.vertices),
                "polygons": len(mesh.data.polygons),
                "target_groups": weight_report["target_vertex_groups"],
                "zero_weight_vertices": weight_report["zero_weight_vertices"],
                "vertices_over_limit": weight_report["vertices_over_limit"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
