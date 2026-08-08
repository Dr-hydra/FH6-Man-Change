#!/usr/bin/env python3
"""Export a deterministic REST-space armature inventory from the open blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import bpy


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--armature")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector(values: object) -> list[float]:
    return [float(value) for value in values]


def matrix(values: object) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def select_armature(name: str | None) -> bpy.types.Object:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if name:
        armature = bpy.data.objects.get(name)
        if armature is None or armature.type != "ARMATURE":
            raise ValueError(f"Armature {name!r} was not found")
        return armature
    if len(armatures) != 1:
        raise ValueError(f"Expected one armature, found {len(armatures)}; pass --armature")
    return armatures[0]


def mesh_usage(armature: bpy.types.Object) -> tuple[list[dict[str, object]], Counter[str]]:
    records: list[dict[str, object]] = []
    used_counts: Counter[str] = Counter()
    bone_names = {bone.name for bone in armature.data.bones}
    for obj in sorted((item for item in bpy.data.objects if item.type == "MESH"), key=lambda item: item.name.casefold()):
        modifiers = [
            modifier
            for modifier in obj.modifiers
            if modifier.type == "ARMATURE" and modifier.object is armature
        ]
        if not modifiers:
            continue
        groups = sorted(group.name for group in obj.vertex_groups if group.name in bone_names)
        used_groups = set()
        group_indices = {group.index: group.name for group in obj.vertex_groups if group.name in bone_names}
        influence_histogram: Counter[int] = Counter()
        for vertex in obj.data.vertices:
            influences = [
                assignment
                for assignment in vertex.groups
                if assignment.group in group_indices and assignment.weight > 1e-8
            ]
            influence_histogram[len(influences)] += 1
            for assignment in influences:
                used_groups.add(group_indices[assignment.group])
        for group in used_groups:
            used_counts[group] += 1
        records.append(
            {
                "object": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "declared_bone_groups": groups,
                "used_bones": sorted(used_groups),
                "used_bone_count": len(used_groups),
                "influence_histogram": dict(sorted(influence_histogram.items())),
            }
        )
    return records, used_counts


def main() -> None:
    args = arguments()
    blend = args.blend.resolve()
    output = args.output.resolve()
    if not blend.is_file():
        raise FileNotFoundError(blend)
    if Path(bpy.data.filepath).resolve() != blend:
        raise ValueError(f"Open blend {bpy.data.filepath!r} does not match --blend {blend}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    armature = select_armature(args.armature)
    original_pose_position = armature.data.pose_position
    armature.data.pose_position = "REST"
    meshes, used_counts = mesh_usage(armature)
    bones = []
    for index, bone in enumerate(armature.data.bones):
        world_matrix = armature.matrix_world @ bone.matrix_local
        bones.append(
            {
                "index": index,
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "use_deform": bool(bone.use_deform),
                "head_local": vector(bone.head_local),
                "tail_local": vector(bone.tail_local),
                "matrix_local": matrix(bone.matrix_local),
                "head_world_rest": vector(armature.matrix_world @ bone.head_local),
                "tail_world_rest": vector(armature.matrix_world @ bone.tail_local),
                "matrix_world_rest": matrix(world_matrix),
                "used_by_mesh_count": int(used_counts[bone.name]),
            }
        )
    armature.data.pose_position = original_pose_position

    topology_payload = [
        {
            "index": item["index"],
            "name": item["name"],
            "parent": item["parent"],
            "matrix_local": item["matrix_local"],
        }
        for item in bones
    ]
    topology_sha256 = hashlib.sha256(
        json.dumps(topology_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Deterministic REST-space skeleton and mesh-usage inventory.",
        "source_format": str(bpy.context.scene.get("source_format", "fh6_donor")),
        "blend": str(blend),
        "blend_sha256": sha256(blend),
        "armature": {
            "object": armature.name,
            "data": armature.data.name,
            "bone_count": len(bones),
            "original_pose_position": original_pose_position,
            "evaluated_pose_position": "REST",
            "matrix_world": matrix(armature.matrix_world),
            "topology_sha256": topology_sha256,
        },
        "bones": bones,
        "meshes": meshes,
        "summary": {
            "bone_count": len(bones),
            "deform_bones": sum(1 for bone in bones if bone["use_deform"]),
            "bones_used_by_meshes": sum(1 for bone in bones if bone["used_by_mesh_count"] > 0),
            "mesh_count": len(meshes),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_SKELETON_INVENTORY="
        + json.dumps(
            {
                "output": str(output),
                "armature": armature.name,
                "bones": len(bones),
                "meshes": len(meshes),
                "topology_sha256": topology_sha256,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
