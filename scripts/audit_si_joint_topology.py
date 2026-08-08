#!/usr/bin/env python3
"""Audit local material islands and joint boundaries in Si FBX milestones.

The audit is read-only: it opens one or more blend files in separate Blender
processes (the caller normally invokes it once per file) and writes compact
JSON with mesh bounds, material islands, boundary loops, and bone-weight
summaries around the neck/wrists/ankles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import bpy
from mathutils import Vector


JOINTS = {
    "neck": ("Neck", 0.18, 0.18),
    "left_wrist": ("LeftHand", 0.20, 0.16),
    "right_wrist": ("RightHand", 0.20, 0.16),
    "left_ankle": ("LeftFoot", 0.18, 0.14),
    "right_ankle": ("RightFoot", 0.18, 0.14),
}


def args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mesh", action="append", default=[])
    parser.add_argument("--armature", action="append", default=[])
    return parser.parse_args(argv)


def world_vertices(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def point_summary(points: list[Vector]) -> dict[str, object]:
    if not points:
        return {"count": 0, "measurable": False}
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    mean = sum(points, Vector()) / len(points)
    return {
        "count": len(points),
        "measurable": True,
        "centroid_m": [round(float(x), 9) for x in mean],
        "bounds_min_m": [round(float(x), 9) for x in lo],
        "bounds_max_m": [round(float(x), 9) for x in hi],
        "extent_mm": [round(float((hi[i] - lo[i]) * 1000.0), 6) for i in range(3)],
    }


def material_data(obj: bpy.types.Object, coords: list[Vector]) -> dict[str, object]:
    polygons_by_slot: dict[int, list[bpy.types.MeshPolygon]] = defaultdict(list)
    for poly in obj.data.polygons:
        polygons_by_slot[poly.material_index].append(poly)
    result: dict[str, object] = {}
    for slot_index, slot in enumerate(obj.material_slots):
        name = slot.material.name if slot.material else f"<slot:{slot_index}>"
        polys = polygons_by_slot.get(slot_index, [])
        ids = {v for poly in polys for v in poly.vertices}
        # Edges shared by two polygons of this material are interior; all other
        # material edges form the observable material boundary.
        edge_counts: Counter[tuple[int, int]] = Counter()
        for poly in polys:
            vertices = list(poly.vertices)
            for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                edge_counts[tuple(sorted((a, b)))] += 1
        boundary_ids = {v for edge, count in edge_counts.items() if count == 1 for v in edge}
        result[name] = {
            "slot": slot_index,
            "polygons": len(polys),
            "vertices": len(ids),
            "boundary_vertices": len(boundary_ids),
            "bounds": point_summary([coords[i] for i in ids]),
            "boundary_bounds": point_summary([coords[i] for i in boundary_ids]),
        }
    return result


def armature_points(armature: bpy.types.Object) -> dict[str, Vector]:
    return {bone.name: armature.matrix_world @ bone.head_local for bone in armature.data.bones}


def local_joint_data(obj: bpy.types.Object, armature: bpy.types.Object) -> dict[str, object]:
    coords = world_vertices(obj)
    bones = armature_points(armature)
    # Bone groups are useful even if the mesh has no explicit semantic split.
    group_by_index = {group.index: group.name for group in obj.vertex_groups}
    vertex_materials: dict[int, set[str]] = defaultdict(set)
    for poly in obj.data.polygons:
        material = obj.material_slots[poly.material_index].material if poly.material_index < len(obj.material_slots) else None
        material_name = material.name if material else f"<slot:{poly.material_index}>"
        for vertex_index in poly.vertices:
            vertex_materials[vertex_index].add(material_name)
    result: dict[str, object] = {}
    for key, (bone_name, radius, seam_radius) in JOINTS.items():
        center = bones.get(bone_name)
        if center is None:
            result[key] = {"missing_bone": bone_name}
            continue
        selected = [i for i, point in enumerate(coords) if (point - center).length <= radius]
        material_counts: Counter[str] = Counter()
        weight_counts: Counter[str] = Counter()
        for poly in obj.data.polygons:
            if any(i in selected for i in poly.vertices):
                mat = obj.material_slots[poly.material_index].material if poly.material_index < len(obj.material_slots) else None
                material_counts[mat.name if mat else f"<slot:{poly.material_index}>"] += 1
        for index in selected:
            vertex = obj.data.vertices[index]
            for influence in vertex.groups:
                if influence.weight > 1.0e-5:
                    weight_counts[group_by_index.get(influence.group, f"<group:{influence.group}>")] += 1
        result[key] = {
            "bone": bone_name,
            "bone_head_m": [round(float(x), 9) for x in center],
            "radius_m": radius,
            "selected_vertices": len(selected),
            "material_polygon_counts": dict(material_counts),
            "weight_group_vertex_counts": dict(weight_counts.most_common(20)),
            "selected_bounds": point_summary([coords[i] for i in selected]),
            "near_seam_by_material": {
                name: sum(
                    1
                    for i, point in enumerate(coords)
                    if (point - center).length <= seam_radius
                    and name in vertex_materials.get(i, set())
                )
                for slot, material in enumerate(obj.material_slots)
                for name in ([material.material.name] if material.material else [f"<slot:{slot}>"])
            },
        }
    return result


def main() -> None:
    parsed = args()
    meshes = [bpy.data.objects.get(name) for name in parsed.mesh]
    meshes = [obj for obj in meshes if obj and obj.type == "MESH"]
    armatures = [bpy.data.objects.get(name) for name in parsed.armature]
    armatures = [obj for obj in armatures if obj and obj.type == "ARMATURE"]
    if not meshes:
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not armatures:
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    report: dict[str, object] = {
        "blend": str(Path(bpy.data.filepath).resolve()),
        "meshes": {},
        "armatures": {},
    }
    for armature in armatures:
        report["armatures"][armature.name] = {
            "bones": len(armature.data.bones),
            "bounds": point_summary([armature.matrix_world @ b.head_local for b in armature.data.bones]),
            "landmarks": {
                name: [round(float(x), 9) for x in point]
                for name, point in armature_points(armature).items()
                if name in {item[0] for item in JOINTS.values()} | {"Head", "Neck1"}
            },
        }
    for mesh in meshes:
        coords = world_vertices(mesh)
        armature_modifiers = [m.object for m in mesh.modifiers if m.type == "ARMATURE" and m.object]
        report["meshes"][mesh.name] = {
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "bounds": point_summary(coords),
            "materials": material_data(mesh, coords),
            "joints": local_joint_data(mesh, armature_modifiers[0]) if armature_modifiers else {},
            "custom_properties": {key: mesh.get(key) for key in mesh.keys()},
        }
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SI_JOINT_AUDIT={parsed.output.resolve()}")


if __name__ == "__main__":
    main()
