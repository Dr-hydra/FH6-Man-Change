#!/usr/bin/env python3
"""Render a combined helmet/outfit candidate for head and seam inspection."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--helmet-blend", required=True, type=Path)
    parser.add_argument("--outfit-blend", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--save-blend", required=True, type=Path)
    return parser.parse_args(argv)


def aim(camera: bpy.types.Object, target: Vector, direction: Vector, distance: float) -> None:
    camera.location = target + direction.normalized() * distance
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = arguments()
    helmet_blend = args.helmet_blend.resolve()
    outfit_blend = args.outfit_blend.resolve(strict=True)
    if Path(bpy.data.filepath).resolve() != helmet_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {helmet_blend}")
    output_dir = args.output_dir.resolve()
    report = args.report.resolve()
    save_blend = args.save_blend.resolve()
    outputs = {
        "full_front": output_dir / "assembly-full-front.png",
        "head_front": output_dir / "assembly-head-front.png",
        "head_side": output_dir / "assembly-head-side.png",
    }
    for path in [*outputs.values(), report, save_blend]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    helmet_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(helmet_armatures) != 1:
        raise RuntimeError("Helmet blend must contain exactly one armature")
    armature = helmet_armatures[0]

    with bpy.data.libraries.load(str(outfit_blend), link=False) as (source, target):
        target.objects = [name for name in source.objects if name.startswith("Outfit_Race_Suit_Modern_F")]
    outfit_objects = [obj for obj in target.objects if obj is not None and obj.type == "MESH"]
    if len(outfit_objects) != 8:
        raise RuntimeError(f"Expected eight outfit meshes, found {len(outfit_objects)}")
    for obj in outfit_objects:
        bpy.context.scene.collection.objects.link(obj)
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE":
                modifier.object = armature

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    for obj in meshes:
        obj.color = (0.40, 0.58, 0.72, 1.0) if obj.name.startswith("Helmet") else (0.72, 0.72, 0.72, 1.0)

    camera_data = bpy.data.cameras.new("FH6 Candidate Assembly Camera")
    camera_data.lens = 62
    camera = bpy.data.objects.new("FH6 Candidate Assembly Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for obj in scene.objects:
        obj.hide_render = obj not in meshes and obj is not camera

    points = [obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    aim(camera, center, Vector((0.0, 1.0, 0.0)), max(max(extent) * 2.1, 1.0))
    render(scene, outputs["full_front"])

    head_target = armature.matrix_world @ armature.pose.bones["Head"].head
    camera_data.lens = 72
    aim(camera, head_target + Vector((0.0, 0.0, -0.04)), Vector((0.0, 1.0, 0.0)), 0.43)
    render(scene, outputs["head_front"])
    aim(camera, head_target + Vector((0.0, 0.0, -0.04)), Vector((1.0, 0.0, 0.0)), 0.43)
    render(scene, outputs["head_side"])

    bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
                "helmet_blend": str(helmet_blend),
                "outfit_blend": str(outfit_blend),
                "mesh_count": len(meshes),
                "outputs": {name: str(path) for name, path in outputs.items()},
                "blend": str(save_blend),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("FH6_ASSEMBLY_CHECK=" + json.dumps({name: str(path) for name, path in outputs.items()}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
