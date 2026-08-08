#!/usr/bin/env python3
"""Import an FBX character source into an immutable Blender baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


IMAGE_SUFFIXES = {".bmp", ".dds", ".exr", ".jpeg", ".jpg", ".png", ".tga", ".tif", ".tiff"}
LOD_PATTERN = re.compile(r"_(lod\d+)$", re.IGNORECASE)
CORE_ROLES = {"body", "garment", "hair", "head"}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", required=True, type=Path)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--global-scale", type=float, default=100.0)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_startup_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def classify_mesh(name: str) -> tuple[str, str]:
    lowered = name.casefold()
    if "shadowproxydesktop" in lowered:
        lod = "shadow_proxy"
    else:
        match = LOD_PATTERN.search(lowered)
        lod = match.group(1).casefold() if match else "unclassified"

    if "_vfxpart_" in lowered:
        role = "effects"
    elif "_body_" in lowered:
        role = "body"
    elif any(token in lowered for token in ("_face_", "_iris_", "_brow_", "_eyeshadow_")):
        role = "head"
    elif any(token in lowered for token in ("_hair_", "_hairshadow_")):
        role = "hair"
    elif "_cloth_" in lowered:
        match = re.search(r"_cloth_(\d+)_", lowered)
        role = "garment" if match and int(match.group(1)) <= 3 else "effects"
    else:
        role = "unclassified"
    return role, lod


def source_tree_inventory(root: Path) -> list[dict]:
    records = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def image_inventory() -> tuple[list[dict], list[str]]:
    records = []
    missing = []
    for image in sorted(bpy.data.images, key=lambda item: item.name.casefold()):
        raw_path = image.filepath or image.filepath_raw
        resolved = Path(bpy.path.abspath(raw_path)).resolve() if raw_path else None
        exists = bool(image.packed_file) or (resolved is not None and resolved.is_file())
        record = {
            "name": image.name,
            "source": image.source,
            "filepath": raw_path,
            "resolved_path": str(resolved) if resolved else "",
            "exists": exists,
            "size": list(image.size),
        }
        records.append(record)
        if image.source == "FILE" and not exists:
            missing.append(image.name)
    return records, missing


def matrix_is_identity(matrix: Matrix, tolerance: float = 1e-8) -> bool:
    identity = Matrix.Identity(4)
    return all(abs(matrix[row][column] - identity[row][column]) <= tolerance for row in range(4) for column in range(4))


def armature_inventory() -> list[dict]:
    records = []
    for obj in sorted((item for item in bpy.data.objects if item.type == "ARMATURE"), key=lambda item: item.name.casefold()):
        non_identity = [bone.name for bone in obj.pose.bones if not matrix_is_identity(bone.matrix_basis)]
        records.append(
            {
                "object": obj.name,
                "armature": obj.data.name,
                "bones": len(obj.data.bones),
                "deform_bones": sum(1 for bone in obj.data.bones if bone.use_deform),
                "root_bones": [bone.name for bone in obj.data.bones if bone.parent is None],
                "pose_position": obj.data.pose_position,
                "non_identity_pose_bones": len(non_identity),
                "non_identity_pose_bone_names": non_identity,
                "bone_names": [bone.name for bone in obj.data.bones],
            }
        )
    return records


def mesh_inventory() -> tuple[list[dict], dict]:
    records = []
    lod_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"objects": 0, "vertices": 0, "triangles": 0})
    role_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"objects": 0, "vertices": 0, "triangles": 0})
    influence_histogram: Counter[int] = Counter()
    totals = {
        "mesh_objects": 0,
        "vertices": 0,
        "triangles": 0,
        "core_lod0_vertices": 0,
        "core_lod0_triangles": 0,
        "vertices_over_four_influences": 0,
        "zero_weight_vertices_on_skinned_meshes": 0,
        "weighted_vertices_not_normalized": 0,
        "missing_uv_meshes": 0,
    }

    for obj in sorted((item for item in bpy.data.objects if item.type == "MESH"), key=lambda item: item.name.casefold()):
        mesh = obj.data
        role, lod = classify_mesh(obj.name)
        triangles = sum(max(0, len(polygon.vertices) - 2) for polygon in mesh.polygons)
        armatures = [
            modifier.object
            for modifier in obj.modifiers
            if modifier.type == "ARMATURE" and modifier.object and modifier.object.type == "ARMATURE"
        ]
        deform_names = {bone.name for armature in armatures for bone in armature.data.bones if bone.use_deform}
        deform_indices = {group.index for group in obj.vertex_groups if group.name in deform_names}
        mesh_histogram: Counter[int] = Counter()
        over_four = 0
        zero_weight = 0
        non_normalized = 0
        minimum_sum = math.inf
        maximum_sum = -math.inf
        for vertex in mesh.vertices:
            weights = [element.weight for element in vertex.groups if element.group in deform_indices and element.weight > 1e-8]
            count = len(weights)
            mesh_histogram[count] += 1
            influence_histogram[count] += 1
            if count > 4:
                over_four += 1
            if armatures and count == 0:
                zero_weight += 1
            if weights:
                weight_sum = sum(weights)
                minimum_sum = min(minimum_sum, weight_sum)
                maximum_sum = max(maximum_sum, weight_sum)
                if abs(weight_sum - 1.0) > 1e-4:
                    non_normalized += 1

        world_corners = [obj.matrix_world @ Vector(obj.bound_box[index]) for index in range(8)]
        bounds_min = [min(corner[axis] for corner in world_corners) for axis in range(3)]
        bounds_max = [max(corner[axis] for corner in world_corners) for axis in range(3)]
        record = {
            "object": obj.name,
            "mesh": mesh.name,
            "role": role,
            "lod": lod,
            "selected_for_fh6_core": role in CORE_ROLES and lod == "lod0",
            "vertices": len(mesh.vertices),
            "triangles": triangles,
            "uv_layers": [layer.name for layer in mesh.uv_layers],
            "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "shape_keys_excluding_basis": max(0, len(mesh.shape_keys.key_blocks) - 1) if mesh.shape_keys else 0,
            "vertex_groups": len(obj.vertex_groups),
            "deform_vertex_groups": len(deform_indices),
            "armatures": [armature.name for armature in armatures],
            "influence_histogram": dict(sorted(mesh_histogram.items())),
            "vertices_over_four_influences": over_four,
            "zero_weight_vertices": zero_weight,
            "weighted_vertices_not_normalized": non_normalized,
            "minimum_weight_sum": None if math.isinf(minimum_sum) else minimum_sum,
            "maximum_weight_sum": None if math.isinf(maximum_sum) else maximum_sum,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "location": list(obj.location),
            "rotation_degrees": [math.degrees(value) for value in obj.rotation_euler],
            "scale": list(obj.scale),
        }
        records.append(record)

        for bucket, key in ((lod_totals, lod), (role_totals, role)):
            bucket[key]["objects"] += 1
            bucket[key]["vertices"] += len(mesh.vertices)
            bucket[key]["triangles"] += triangles
        totals["mesh_objects"] += 1
        totals["vertices"] += len(mesh.vertices)
        totals["triangles"] += triangles
        totals["vertices_over_four_influences"] += over_four
        totals["zero_weight_vertices_on_skinned_meshes"] += zero_weight
        totals["weighted_vertices_not_normalized"] += non_normalized
        totals["missing_uv_meshes"] += int(not mesh.uv_layers)
        if record["selected_for_fh6_core"]:
            totals["core_lod0_vertices"] += len(mesh.vertices)
            totals["core_lod0_triangles"] += triangles

    totals["influence_histogram"] = dict(sorted(influence_histogram.items()))
    totals["lods"] = dict(sorted(lod_totals.items()))
    totals["roles"] = dict(sorted(role_totals.items()))
    return records, totals


def rest_bind_error() -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    maximum = 0.0
    checked = 0
    for obj in (item for item in bpy.data.objects if item.type == "MESH"):
        if not any(modifier.type == "ARMATURE" and modifier.object for modifier in obj.modifiers):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            if len(evaluated_mesh.vertices) != len(obj.data.vertices):
                continue
            for source_vertex, evaluated_vertex in zip(obj.data.vertices, evaluated_mesh.vertices, strict=True):
                source_position = obj.matrix_world @ source_vertex.co
                evaluated_position = evaluated.matrix_world @ evaluated_vertex.co
                maximum = max(maximum, (evaluated_position - source_position).length)
                checked += 1
        finally:
            evaluated.to_mesh_clear()
    return {"checked_vertices": checked, "maximum_world_space_error": maximum}


def lock_source_collection(imported_objects: list[bpy.types.Object]) -> bpy.types.Collection:
    source = bpy.data.collections.new("SOURCE_FBX")
    bpy.context.scene.collection.children.link(source)
    for obj in imported_objects:
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        source.objects.link(obj)
        obj["source_immutable"] = True
        obj.hide_select = True
        if obj.type == "MESH":
            role, lod = classify_mesh(obj.name)
            obj["source_role"] = role
            obj["source_lod"] = lod
    source["source_immutable"] = True
    return source


def main() -> None:
    args = parse_args()
    fbx = args.fbx.resolve()
    blend = args.blend.resolve()
    metadata_path = args.metadata.resolve()
    if not fbx.is_file():
        raise FileNotFoundError(fbx)
    if fbx.suffix.casefold() != ".fbx":
        raise ValueError(f"Expected an FBX source, got {fbx.suffix}")
    if args.global_scale <= 0:
        raise ValueError("--global-scale must be positive")
    if blend.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing FBX baseline output")

    blend.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    clear_startup_scene()
    result = bpy.ops.import_scene.fbx(
        filepath=str(fbx),
        global_scale=args.global_scale,
        use_custom_normals=True,
        ignore_leaf_bones=False,
        automatic_bone_orientation=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"FBX import failed: {result}")

    imported_objects = list(bpy.context.scene.objects)
    source_collection = lock_source_collection(imported_objects)
    armatures = [obj for obj in imported_objects if obj.type == "ARMATURE"]
    for armature in armatures:
        armature.data.pose_position = "REST"

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene["baseline_kind"] = "immutable_fbx_source"
    scene["source_format"] = "fbx"
    scene["source_fbx"] = str(fbx)
    scene["source_fbx_sha256"] = sha256(fbx)
    scene["source_global_scale"] = args.global_scale
    scene["source_pose_position"] = "REST"
    scene["fh6_export_ready"] = False

    images, missing_images = image_inventory()
    meshes, totals = mesh_inventory()
    armature_records = armature_inventory()
    bind_check = rest_bind_error()
    hard_errors = []
    if len(armature_records) != 1:
        hard_errors.append({"code": "armature_count", "value": len(armature_records)})
    if totals["vertices_over_four_influences"]:
        hard_errors.append({"code": "more_than_four_influences", "vertices": totals["vertices_over_four_influences"]})
    if totals["zero_weight_vertices_on_skinned_meshes"]:
        hard_errors.append({"code": "zero_weight_vertices", "vertices": totals["zero_weight_vertices_on_skinned_meshes"]})
    if totals["weighted_vertices_not_normalized"]:
        hard_errors.append({"code": "weights_not_normalized", "vertices": totals["weighted_vertices_not_normalized"]})
    if totals["core_lod0_vertices"] > 65535:
        hard_errors.append({"code": "core_lod0_vertex_domain", "vertices": totals["core_lod0_vertices"]})
    required_roles = CORE_ROLES - {record["role"] for record in meshes if record["lod"] == "lod0"}
    if required_roles:
        hard_errors.append({"code": "missing_core_roles", "roles": sorted(required_roles)})
    if bind_check["maximum_world_space_error"] > 1e-6:
        hard_errors.append({"code": "rest_bind_error", **bind_check})

    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=False, check_existing=False)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only FBX source baseline for the FBX-first FH6 character pipeline.",
        "source": {
            "format": "fbx",
            "fbx": str(fbx),
            "fbx_sha256": scene["source_fbx_sha256"],
            "root": str(fbx.parent),
            "files": source_tree_inventory(fbx.parent),
        },
        "output": {"blend": str(blend), "blend_sha256": sha256(blend)},
        "software": {"blender": bpy.app.version_string, "python": sys.version.split()[0]},
        "import_settings": {
            "global_scale": args.global_scale,
            "use_custom_normals": True,
            "ignore_leaf_bones": False,
            "automatic_bone_orientation": False,
            "pose_position": "REST",
            "source_collection": source_collection.name,
        },
        "totals": {
            **totals,
            "armature_objects": len(armature_records),
            "bones": sum(record["bones"] for record in armature_records),
            "images": len(images),
            "missing_images": len(missing_images),
        },
        "meshes": meshes,
        "armatures": armature_records,
        "images": images,
        "missing_image_names": missing_images,
        "rest_bind_check": bind_check,
        "validation": {"hard_error_count": len(hard_errors), "hard_errors": hard_errors},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_FBX_BASELINE_SUMMARY=" + json.dumps(metadata["totals"], ensure_ascii=False, sort_keys=True))
    print(f"FH6_FBX_BASELINE_BLEND={blend}")
    print(f"FH6_FBX_BASELINE_METADATA={metadata_path}")
    if hard_errors:
        raise RuntimeError(f"FBX baseline validation failed: {hard_errors}")


if __name__ == "__main__":
    main()
