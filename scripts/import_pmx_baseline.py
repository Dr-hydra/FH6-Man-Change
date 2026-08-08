#!/usr/bin/env python3
"""Legacy PMX importer retained for compatibility and reference baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmx", required=True, type=Path)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.08)
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


def addon_version() -> str:
    try:
        import mmd_tools

        version = getattr(mmd_tools, "__version__", None)
        if version:
            return str(version)
        bl_info = getattr(mmd_tools, "bl_info", {})
        if bl_info.get("version"):
            return ".".join(str(part) for part in bl_info["version"])
        manifest = Path(mmd_tools.__file__).resolve().parent / "blender_manifest.toml"
        if manifest.is_file():
            import tomllib

            return str(tomllib.loads(manifest.read_text(encoding="utf-8"))["version"])
    except Exception:
        pass
    return "unknown"


def image_inventory(pmx: Path) -> tuple[list[dict], list[str]]:
    inventory: list[dict] = []
    missing: list[str] = []
    for image in sorted(bpy.data.images, key=lambda item: item.name.casefold()):
        raw_path = image.filepath or image.filepath_raw
        if image.packed_file:
            resolved = "<packed>"
            exists = True
        elif raw_path.startswith("//"):
            resolved = str((pmx.parent / raw_path[2:]).resolve())
            exists = Path(resolved).is_file()
        elif raw_path:
            resolved = os.path.abspath(bpy.path.abspath(raw_path))
            exists = Path(resolved).is_file()
        else:
            resolved = ""
            exists = image.source in {"GENERATED", "VIEWER"}
        record = {
            "name": image.name,
            "source": image.source,
            "filepath": raw_path,
            "resolved_path": resolved,
            "exists": exists,
            "size": list(image.size),
        }
        inventory.append(record)
        if not exists:
            missing.append(image.name)
    return inventory, missing


def mesh_inventory() -> tuple[list[dict], dict]:
    records: list[dict] = []
    totals = {
        "mesh_objects": 0,
        "vertices": 0,
        "edges": 0,
        "polygons": 0,
        "materials": 0,
        "shape_keys_excluding_basis": 0,
        "max_influences_per_vertex": 0,
        "vertices_over_four_influences": 0,
        "weighted_vertices_not_normalized": 0,
    }
    assigned_materials: set[str] = set()
    for obj in sorted((item for item in bpy.data.objects if item.type == "MESH"), key=lambda item: item.name.casefold()):
        mesh = obj.data
        armature_objects = {
            modifier.object
            for modifier in obj.modifiers
            if modifier.type == "ARMATURE" and modifier.object and modifier.object.type == "ARMATURE"
        }
        deform_bone_names = {
            bone.name
            for armature in armature_objects
            for bone in armature.data.bones
            if bone.use_deform
        }
        deform_group_indices = {
            group.index for group in obj.vertex_groups if group.name in deform_bone_names
        }
        non_deform_groups = [
            group.name for group in obj.vertex_groups if group.index not in deform_group_indices
        ]
        shape_key_count = max(0, len(mesh.shape_keys.key_blocks) - 1) if mesh.shape_keys else 0
        max_influences = 0
        over_four = 0
        non_normalized = 0
        for vertex in mesh.vertices:
            weights = [
                group.weight
                for group in vertex.groups
                if group.group in deform_group_indices and group.weight > 1e-8
            ]
            max_influences = max(max_influences, len(weights))
            if len(weights) > 4:
                over_four += 1
            if weights and abs(sum(weights) - 1.0) > 1e-4:
                non_normalized += 1
        record = {
            "object": obj.name,
            "mesh": mesh.name,
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "polygons": len(mesh.polygons),
            "uv_layers": [layer.name for layer in mesh.uv_layers],
            "material_slots": len(obj.material_slots),
            "shape_keys_excluding_basis": shape_key_count,
            "deform_vertex_groups": len(deform_group_indices),
            "non_deform_vertex_groups": non_deform_groups,
            "max_influences_per_vertex": max_influences,
            "vertices_over_four_influences": over_four,
            "weighted_vertices_not_normalized": non_normalized,
            "modifiers": [{"name": modifier.name, "type": modifier.type} for modifier in obj.modifiers],
        }
        records.append(record)
        totals["mesh_objects"] += 1
        totals["vertices"] += record["vertices"]
        totals["edges"] += record["edges"]
        totals["polygons"] += record["polygons"]
        assigned_materials.update(slot.material.name for slot in obj.material_slots if slot.material)
        totals["shape_keys_excluding_basis"] += shape_key_count
        totals["max_influences_per_vertex"] = max(totals["max_influences_per_vertex"], max_influences)
        totals["vertices_over_four_influences"] += over_four
        totals["weighted_vertices_not_normalized"] += non_normalized
    totals["materials"] = len(assigned_materials)
    return records, totals


def armature_inventory() -> tuple[list[dict], dict]:
    records: list[dict] = []
    for obj in sorted((item for item in bpy.data.objects if item.type == "ARMATURE"), key=lambda item: item.name.casefold()):
        records.append(
            {
                "object": obj.name,
                "armature": obj.data.name,
                "bones": len(obj.data.bones),
                "pose_bones": len(obj.pose.bones),
            }
        )
    return records, {"armature_objects": len(records), "bones": sum(item["bones"] for item in records)}


def main() -> None:
    args = parse_args()
    print("FH6_SOURCE_WARNING=PMX is a legacy compatibility source; use build_si_fbx_source.ps1 for new Si work")
    pmx = args.pmx.resolve()
    blend = args.blend.resolve()
    metadata_path = args.metadata.resolve()
    if not pmx.is_file():
        raise FileNotFoundError(pmx)
    if blend.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing baseline output")

    blend.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.preferences.addon_enable(module="mmd_tools")
    clear_startup_scene()
    result = bpy.ops.mmd_tools.import_model(
        filepath=str(pmx),
        types={"MESH", "ARMATURE", "DISPLAY", "MORPHS"},
        scale=args.scale,
        clean_model=False,
        remove_doubles=False,
        import_adduv2_as_vertex_colors=False,
        fix_bone_order=False,
        fix_ik_links=False,
        apply_bone_fixed_axis=False,
        rename_bones=False,
        use_mipmap=True,
        log_level="WARNING",
        save_log=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"MMD Tools import failed: {result}")

    images, missing_images = image_inventory(pmx)
    meshes, mesh_totals = mesh_inventory()
    armatures, armature_totals = armature_inventory()

    scene = bpy.context.scene
    scene["baseline_kind"] = "immutable_pmx_source"
    scene["source_pmx"] = str(pmx)
    scene["source_pmx_sha256"] = sha256(pmx)
    scene["mmd_tools_version"] = addon_version()
    scene["mmd_physics_instantiated"] = False
    scene["fh6_export_ready"] = False

    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=False, check_existing=False)

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only PMX source baseline; not an FH6-ready export scene.",
        "source": {
            "pmx": str(pmx),
            "pmx_sha256": scene["source_pmx_sha256"],
            "license_readme": str(pmx.parent / "ReadMe.txt"),
        },
        "output": {"blend": str(blend), "blend_sha256": sha256(blend)},
        "software": {
            "blender": bpy.app.version_string,
            "python": sys.version.split()[0],
            "mmd_tools": scene["mmd_tools_version"],
        },
        "import_settings": {
            "scale": args.scale,
            "types": ["ARMATURE", "DISPLAY", "MESH", "MORPHS"],
            "physics_instantiated": False,
            "clean_model": False,
            "remove_doubles": False,
            "fix_bone_order": False,
            "fix_ik_links": False,
            "apply_bone_fixed_axis": False,
            "rename_bones": False,
        },
        "totals": {**mesh_totals, **armature_totals, "images": len(images), "missing_images": len(missing_images)},
        "meshes": meshes,
        "armatures": armatures,
        "images": images,
        "missing_image_names": missing_images,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FH6_BASELINE_SUMMARY=" + json.dumps(metadata["totals"], ensure_ascii=False, sort_keys=True))
    print(f"FH6_BASELINE_BLEND={blend}")
    print(f"FH6_BASELINE_METADATA={metadata_path}")


if __name__ == "__main__":
    main()
