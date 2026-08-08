#!/usr/bin/env python3
"""Render close-up wrist and ankle pose checks from an imported candidate blend."""

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


def reset_pose(armature: bpy.types.Object) -> None:
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.location = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def set_rotation(armature: bpy.types.Object, name: str, degrees: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(name)
    if bone is None:
        raise RuntimeError(f"Missing pose bone {name!r}")
    bone.rotation_mode = "XYZ"
    bone.rotation_euler = tuple(math.radians(value) for value in degrees)


def bone_target(armature: bpy.types.Object, name: str) -> Vector:
    bone = armature.pose.bones.get(name)
    if bone is None:
        raise RuntimeError(f"Missing pose bone {name!r}")
    return armature.matrix_world @ bone.head


def aim(camera: bpy.types.Object, target: Vector, direction: Vector, distance: float) -> None:
    camera.location = target + direction.normalized() * distance
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = arguments()
    source = args.source_blend.resolve()
    if Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source}")
    output_dir = args.output_dir.resolve()
    report = args.report.resolve()
    outputs = {
        "left_wrist": output_dir / "left-wrist-pose.png",
        "right_wrist": output_dir / "right-wrist-pose.png",
        "left_finger_stress": output_dir / "left-finger-stress.png",
        "right_finger_stress": output_dir / "right-finger-stress.png",
        "left_ankle": output_dir / "left-ankle-pose.png",
        "right_ankle": output_dir / "right-ankle-pose.png",
        "left_toe_stress": output_dir / "left-toe-stress.png",
        "right_toe_stress": output_dir / "right-toe-stress.png",
    }
    for path in [*outputs.values(), report]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise RuntimeError("Expected one armature and at least one mesh")
    armature = armatures[0]
    scene = bpy.context.scene
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

    camera_data = bpy.data.cameras.new("FH6 Joint Check Camera")
    camera_data.lens = 58
    camera = bpy.data.objects.new("FH6 Joint Check Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for obj in scene.objects:
        obj.hide_render = obj not in meshes and obj is not camera

    reset_pose(armature)
    set_rotation(armature, "LeftHand", (0.0, 0.0, 55.0))
    set_rotation(armature, "RightHand", (0.0, 0.0, -55.0))
    aim(camera, bone_target(armature, "LeftHand"), Vector((0.0, -1.0, 0.0)), 0.62)
    render(scene, outputs["left_wrist"])
    aim(camera, bone_target(armature, "RightHand"), Vector((0.0, -1.0, 0.0)), 0.62)
    render(scene, outputs["right_wrist"])

    reset_pose(armature)
    finger_tokens = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    for side in ("Left", "Right"):
        for bone in armature.pose.bones:
            if bone.name.startswith(side) and any(token in bone.name for token in finger_tokens):
                set_rotation(armature, bone.name, (35.0, -30.0, 40.0))
    aim(camera, bone_target(armature, "LeftHand"), Vector((0.0, -1.0, 0.0)), 0.62)
    render(scene, outputs["left_finger_stress"])
    aim(camera, bone_target(armature, "RightHand"), Vector((0.0, -1.0, 0.0)), 0.62)
    render(scene, outputs["right_finger_stress"])

    reset_pose(armature)
    set_rotation(armature, "LeftFoot", (28.0, 0.0, 0.0))
    set_rotation(armature, "RightFoot", (28.0, 0.0, 0.0))
    aim(camera, bone_target(armature, "LeftFoot"), Vector((1.0, 0.0, 0.0)), 0.72)
    render(scene, outputs["left_ankle"])
    aim(camera, bone_target(armature, "RightFoot"), Vector((-1.0, 0.0, 0.0)), 0.72)
    render(scene, outputs["right_ankle"])

    reset_pose(armature)
    set_rotation(armature, "LeftToeBase", (42.0, 0.0, 0.0))
    set_rotation(armature, "RightToeBase", (42.0, 0.0, 0.0))
    aim(camera, bone_target(armature, "LeftFoot"), Vector((1.0, 0.0, 0.0)), 0.72)
    render(scene, outputs["left_toe_stress"])
    aim(camera, bone_target(armature, "RightFoot"), Vector((-1.0, 0.0, 0.0)), 0.72)
    render(scene, outputs["right_toe_stress"])
    reset_pose(armature)

    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
                "source_blend": str(source),
                "poses": {
                    "wrists": {"LeftHand": [0.0, 0.0, 55.0], "RightHand": [0.0, 0.0, -55.0]},
                    "finger_stress": {"all_finger_bones": [35.0, -30.0, 40.0]},
                    "ankles": {"LeftFoot": [28.0, 0.0, 0.0], "RightFoot": [28.0, 0.0, 0.0]},
                    "toe_stress": {"LeftToeBase": [42.0, 0.0, 0.0], "RightToeBase": [42.0, 0.0, 0.0]},
                },
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("FH6_JOINT_CHECKS=" + json.dumps({name: str(path) for name, path in outputs.items()}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
