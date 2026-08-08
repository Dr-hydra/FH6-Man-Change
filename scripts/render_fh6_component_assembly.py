#!/usr/bin/env python3
"""Render combined FH6 component pose gates and hand close-ups."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def ensure_absent(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def setup_scene(scene: bpy.types.Scene, meshes: list[bpy.types.Object]) -> bpy.types.Object:
    minimum, maximum = bounds(meshes)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    camera_data = bpy.data.cameras.new("FH6 Assembly Validation Camera")
    camera = bpy.data.objects.new("FH6 Assembly Validation Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 58
    camera["full_target"] = list(center)
    camera["full_distance"] = max(max(extent) * 2.15, 0.8)
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    for item in scene.objects:
        item.hide_render = item not in meshes and item is not camera
    return camera


def aim(camera: bpy.types.Object, target: Vector, direction: Vector, distance: float) -> None:
    camera.location = target + direction.normalized() * distance
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def reset_pose(armatures: list[bpy.types.Object]) -> None:
    for armature in armatures:
        for bone in armature.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
            bone.location = (0.0, 0.0, 0.0)
            bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def rotate_all(
    armatures: list[bpy.types.Object],
    rotations: tuple[tuple[str, tuple[float, float, float]], ...],
) -> None:
    for armature in armatures:
        for bone_name, degrees in rotations:
            bone = armature.pose.bones.get(bone_name)
            if bone is None:
                continue
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = tuple(math.radians(value) for value in degrees)
    bpy.context.view_layer.update()


def hand_bounds(body: bpy.types.Object, side: str) -> tuple[Vector, Vector, int]:
    prefixes = (
        f"{side}Hand",
        f"{side}Index",
        f"{side}Middle",
        f"{side}Pinky",
        f"{side}Ring",
        f"{side}Thumb",
    )
    group_indices = {
        group.index for group in body.vertex_groups if group.name.startswith(prefixes)
    }
    indices = {
        vertex.index
        for vertex in body.data.vertices
        if any(item.group in group_indices and item.weight > 1e-8 for item in vertex.groups)
    }
    if not indices:
        raise RuntimeError(f"No {side} hand vertices found on {body.name}")
    points = [body.matrix_world @ body.data.vertices[index].co for index in indices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum, len(indices)


def finger_rotations(axis: int, degrees: float) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    result = []
    for side in ("Left", "Right"):
        sign = 1.0 if side == "Left" else -1.0
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            for segment in ("1", "2", "3"):
                rotation = [0.0, 0.0, 0.0]
                rotation[axis] = degrees * sign
                result.append((f"{side}{finger}{segment}", tuple(rotation)))
        for segment in ("1", "2", "3"):
            rotation = [0.0, 0.0, 0.0]
            rotation[axis] = degrees * 0.7 * sign
            result.append((f"{side}Thumb{segment}", tuple(rotation)))
    return tuple(result)


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source_blend}")
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    outputs = {
        "rest_front": output_dir / "assembly-rest-front.png",
        "rest_back": output_dir / "assembly-rest-back.png",
        "rest_side": output_dir / "assembly-rest-side.png",
        "shoulders": output_dir / "assembly-pose-shoulders.png",
        "elbows": output_dir / "assembly-pose-elbows.png",
        "wrists": output_dir / "assembly-pose-wrists.png",
        "knees": output_dir / "assembly-pose-knees.png",
        "head_turn": output_dir / "assembly-pose-head-turn.png",
        "left_hand_rest": output_dir / "left-hand-rest.png",
        "left_fingers_x": output_dir / "left-fingers-curl-x.png",
        "left_fingers_y": output_dir / "left-fingers-curl-y.png",
        "left_fingers_z": output_dir / "left-fingers-curl-z.png",
    }
    ensure_absent([*outputs.values(), report_path])

    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.get("fh6_weights_retargeted") is True and not obj.get("fh6_probe_exclude")
    ]
    expected = {"Body", "Outfit", "Helmet"}
    by_component = {obj.get("fh6_component"): obj for obj in meshes}
    if set(by_component) != expected:
        raise RuntimeError(f"Expected retargeted {sorted(expected)}, found {sorted(by_component)}")
    armatures = sorted(
        {
            modifier.object
            for mesh in meshes
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE" and modifier.object is not None
        },
        key=lambda item: item.name,
    )
    if len(armatures) != 3:
        raise RuntimeError(f"Expected three component armatures, found {len(armatures)}")

    scene = bpy.context.scene
    camera = setup_scene(scene, meshes)
    full_target = Vector(camera["full_target"])
    full_distance = float(camera["full_distance"])

    reset_pose(armatures)
    aim(camera, full_target, Vector((0.0, -1.0, 0.0)), full_distance)
    render(scene, outputs["rest_front"])
    aim(camera, full_target, Vector((0.0, 1.0, 0.0)), full_distance)
    render(scene, outputs["rest_back"])
    aim(camera, full_target, Vector((1.0, 0.0, 0.0)), full_distance)
    render(scene, outputs["rest_side"])

    poses = {
        "shoulders": (("LeftArm", (0.0, 0.0, 35.0)), ("RightArm", (0.0, 0.0, -35.0))),
        "elbows": (("LeftForeArm", (55.0, 0.0, 0.0)), ("RightForeArm", (-55.0, 0.0, 0.0))),
        "wrists": (("LeftHand", (0.0, 0.0, 45.0)), ("RightHand", (0.0, 0.0, -45.0))),
        "knees": (("LeftLeg", (55.0, 0.0, 0.0)), ("RightLeg", (55.0, 0.0, 0.0))),
        "head_turn": (("Head", (0.0, 0.0, 35.0)),),
    }
    for key, rotations in poses.items():
        reset_pose(armatures)
        rotate_all(armatures, rotations)
        direction = Vector((1.0, 0.0, 0.0)) if key == "knees" else Vector((0.0, -1.0, 0.0))
        aim(camera, full_target, direction, full_distance)
        render(scene, outputs[key])

    reset_pose(armatures)
    hand_minimum, hand_maximum, hand_vertices = hand_bounds(by_component["Body"], "Left")
    hand_target = (hand_minimum + hand_maximum) * 0.5
    hand_extent = max(hand_maximum - hand_minimum)
    hand_distance = max(hand_extent * 3.0, 0.22)
    camera.data.lens = 70
    aim(camera, hand_target, Vector((0.0, -1.0, 0.0)), hand_distance)
    render(scene, outputs["left_hand_rest"])
    for axis, key in enumerate(("left_fingers_x", "left_fingers_y", "left_fingers_z")):
        reset_pose(armatures)
        rotate_all(armatures, finger_rotations(axis, 32.0))
        aim(camera, hand_target, Vector((0.0, -1.0, 0.0)), hand_distance)
        render(scene, outputs[key])
    reset_pose(armatures)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_blend": str(source_blend),
        "components": {
            component: {
                "object": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "armature": next(
                    modifier.object.name
                    for modifier in obj.modifiers
                    if modifier.type == "ARMATURE" and modifier.object is not None
                ),
            }
            for component, obj in sorted(by_component.items())
        },
        "armatures": [armature.name for armature in armatures],
        "left_hand": {
            "weighted_vertices": hand_vertices,
            "bounds_min": list(hand_minimum),
            "bounds_max": list(hand_maximum),
        },
        "pose_gates": {key: {bone: list(rotation) for bone, rotation in value} for key, value in poses.items()},
        "finger_axis_probes_degrees": 32.0,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "evidence": "Blender visual validation only; not an in-game result.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_ASSEMBLY_VALIDATION="
        + json.dumps(
            {
                "report": str(report_path),
                "components": {key: value["object"] for key, value in report["components"].items()},
                "left_hand_vertices": hand_vertices,
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
