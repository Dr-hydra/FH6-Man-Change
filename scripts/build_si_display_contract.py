#!/usr/bin/env python3
"""Freeze and validate the FBX-first Si Display source/donor/LOD contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_modelbin import ParseError, inspect as inspect_modelbin


WORKSPACE = Path(__file__).resolve().parents[1]
REQUIRED_ROLES = ("head", "hair", "body", "garment")
REQUIRED_PHYSICAL_BONES = {
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Neck1",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftWrist_Corrective",
    "LeftWristDown_Corrective",
    "LeftWristUp_Corrective",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightWrist_Corrective",
    "RightWristDown_Corrective",
    "RightWristUp_Corrective",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "sources" / "si" / "source.config.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (WORKSPACE / candidate).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def error(errors: list[dict[str, Any]], code: str, **details: Any) -> None:
    errors.append({"code": code, **details})


def vec_distance(left: list[float], right: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5


def skeleton_names(inventory: dict[str, Any]) -> list[str]:
    return [str(bone["name"]) for bone in inventory["bones"]]


def skeleton_parents(inventory: dict[str, Any]) -> list[str | None]:
    return [bone["parent"] for bone in inventory["bones"]]


def validate_source(config: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    primary = config["primary"]
    outputs = config["outputs"]
    source_fbx = resolve(primary["path"])
    baseline_report_path = resolve(outputs["baseline_metadata"])
    for path, label in ((source_fbx, "source_fbx"), (baseline_report_path, "baseline_report")):
        if not path.is_file():
            error(errors, "missing_file", label=label, path=str(path))
    if errors:
        return {}
    baseline = load_json(baseline_report_path)
    if config.get("primary_format") != "fbx":
        error(errors, "primary_format_not_fbx", actual=config.get("primary_format"))
    if float(primary.get("global_scale", 0.0)) != 100.0:
        error(errors, "invalid_global_scale", actual=primary.get("global_scale"))
    if str(primary.get("pose_position", "")).upper() != "REST":
        error(errors, "invalid_pose_position", actual=primary.get("pose_position"))
    actual_hash = sha256(source_fbx)
    expected_hash = str(baseline["source"]["fbx_sha256"]).lower()
    if actual_hash != expected_hash:
        error(errors, "source_hash_mismatch", expected=expected_hash, actual=actual_hash)
    if int(baseline["validation"]["hard_error_count"]) != 0:
        error(errors, "baseline_has_hard_errors", count=baseline["validation"]["hard_error_count"])
    if str(baseline["import_settings"]["pose_position"]).upper() != "REST":
        error(errors, "baseline_not_rest")
    if float(baseline["import_settings"]["global_scale"]) != 100.0:
        error(errors, "baseline_scale_mismatch")
    return {
        "format": "fbx",
        "fbx": file_record(source_fbx),
        "baseline_report": file_record(baseline_report_path),
        "global_scale": 100.0,
        "pose_position": "REST",
        "rest_bind_maximum_world_error_m": float(baseline["rest_bind_check"]["maximum_world_space_error"]),
        "source_files": baseline["source"]["files"],
        "license_guard": "Local technical validation only; do not redistribute source-derived assets.",
    }


def validate_lods(config: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = config["outputs"]["components"]
    lod_names = [str(item) for item in config["primary"]["working_lods"]]
    lods: list[dict[str, Any]] = []
    previous_total: int | None = None
    previous_by_role: dict[str, int] | None = None
    for lod in lod_names:
        if lod not in outputs:
            error(errors, "missing_lod_config", lod=lod)
            continue
        entry = outputs[lod]
        blend_path = resolve(entry["blend"])
        report_path = resolve(entry["report"])
        probe_path = resolve(entry["probe"])
        if not all(path.is_file() for path in (blend_path, report_path, probe_path)):
            for path, label in ((blend_path, "blend"), (report_path, "report"), (probe_path, "probe")):
                if not path.is_file():
                    error(errors, "missing_lod_file", lod=lod, kind=label, path=str(path))
            continue
        report = load_json(report_path)
        probe = load_json(probe_path)
        if int(report["validation"]["hard_error_count"]) != 0:
            error(errors, "lod_report_has_hard_errors", lod=lod)
        if int(probe["summary"]["hard_error_count"]) != 0:
            error(errors, "lod_probe_has_hard_errors", lod=lod)
        if report["policies"].get("effects_included") or report["policies"].get("shadow_proxies_included"):
            error(errors, "deferred_geometry_included", lod=lod)
        components = report["totals"]["components"]
        missing_roles = sorted(set(REQUIRED_ROLES) - set(components))
        if missing_roles:
            error(errors, "missing_component_roles", lod=lod, roles=missing_roles)
        role_counts = {role: int(components[role]["vertices"]) for role in REQUIRED_ROLES if role in components}
        for role, vertices in role_counts.items():
            if vertices > 65_535:
                error(errors, "r16_vertex_domain_exceeded", lod=lod, role=role, vertices=vertices)
        total_vertices = int(report["totals"]["vertices"])
        if previous_total is not None and total_vertices > previous_total:
            error(errors, "lod_total_not_monotonic", lod=lod, previous=previous_total, actual=total_vertices)
        if previous_by_role is not None:
            for role, vertices in role_counts.items():
                if vertices > previous_by_role[role]:
                    error(
                        errors,
                        "lod_role_not_monotonic",
                        lod=lod,
                        role=role,
                        previous=previous_by_role[role],
                        actual=vertices,
                    )
        previous_total = total_vertices
        previous_by_role = role_counts
        lods.append(
            {
                "lod": lod,
                "blend": file_record(blend_path),
                "report": file_record(report_path),
                "probe": file_record(probe_path),
                "objects": int(report["totals"]["objects"]),
                "vertices": total_vertices,
                "triangles": int(report["totals"]["triangles"]),
                "components": components,
                "meshes": [
                    {
                        "object": mesh["object"],
                        "source_object": mesh.get("source_object"),
                        "role": mesh["role"],
                        "vertices": int(mesh["vertices"]),
                        "triangles": int(mesh["triangles"]),
                        "materials": mesh["materials"],
                        "uv_layers": mesh["uv_layers"],
                        "deform_vertex_groups": int(mesh["deform_vertex_groups"]),
                    }
                    for mesh in report["meshes"]
                ],
                "hard_error_count": 0,
            }
        )
    return {
        "lods": lods,
        "policy": {
            "roles": list(REQUIRED_ROLES),
            "effects_included": False,
            "shadow_proxies_included": False,
            "r16_vertex_domain_limit": 65_535,
            "mapping_rule": "all LODs use the same role and donor-local skeleton contract",
        },
    }


def validate_container(
    key: str,
    entry: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    paths = {
        label: resolve(entry[label])
        for label in ("archive", "modelbin", "blend", "skeleton_inventory")
    }
    for label, path in paths.items():
        if not path.is_file():
            error(errors, "missing_container_file", container=key, kind=label, path=str(path))
    if any(not path.is_file() for path in paths.values()):
        return {}, None
    try:
        modelbin = inspect_modelbin(paths["modelbin"])
    except (OSError, ParseError) as exc:
        error(errors, "modelbin_parse_failed", container=key, message=str(exc))
        return {}, None
    inventory = load_json(paths["skeleton_inventory"])
    parsed = modelbin["parsed"]
    actual = {
        "bones": int(parsed["skeleton"][0]["bone_count"]),
        "meshes": len(parsed["meshes"]),
        "materials": int(modelbin["blob_tags"].get("MatI", 0)),
        "vertex_layouts": len(parsed["vertex_layouts"]),
        "vertex_buffers": len(parsed["vertex_buffers"]),
        "skin_buffers": len(parsed["skin_buffers"]),
    }
    for field, expected in entry["expected"].items():
        if int(actual[field]) != int(expected):
            error(
                errors,
                "container_contract_mismatch",
                container=key,
                field=field,
                expected=int(expected),
                actual=int(actual[field]),
            )
    if parsed["errors"]:
        error(errors, "container_modelbin_errors", container=key, details=parsed["errors"])
    inventory_names = skeleton_names(inventory)
    modelbin_names = [str(bone["name"]) for bone in parsed["skeleton"][0]["bones"]]
    if modelbin_names != inventory_names:
        error(errors, "container_skeleton_order_mismatch", container=key)
    missing_bones = sorted(REQUIRED_PHYSICAL_BONES - set(inventory_names))
    if missing_bones:
        error(errors, "container_missing_required_bones", container=key, bones=missing_bones)
    index_by_name = {bone["name"]: int(bone["index"]) for bone in inventory["bones"]}
    return (
        {
            "name": entry["name"],
            "roles": entry["roles"],
            "files": {label: file_record(path) for label, path in paths.items()},
            "bundle": {
                "header": modelbin["header"],
                "blob_tags": modelbin["blob_tags"],
                "actual": actual,
                "expected": entry["expected"],
                "parse_errors": parsed["errors"],
            },
            "skeleton": {
                "bone_count": len(inventory_names),
                "topology_sha256": inventory["armature"]["topology_sha256"],
                "required_bone_indices": {
                    name: index_by_name[name]
                    for name in sorted(REQUIRED_PHYSICAL_BONES)
                    if name in index_by_name
                },
            },
            "writer_policy": {
                "retain": ["Skel", "VLay", "MatI", "blob order"],
                "replace": ["VerB", "IndB", "Skin"],
                "update": ["Mesh ranges", "bounds", "counts", "sizes", "absolute offsets", "Modl"],
            },
        },
        inventory,
    )


def semantic_donors(config: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, entry in config["display_target"]["semantic_donors"].items():
        files = {}
        for label in ("blend", "metadata", "skeleton_inventory", "modelbin", "report"):
            if label not in entry:
                continue
            path = resolve(entry[label])
            if not path.is_file():
                error(errors, "missing_semantic_donor_file", donor=key, kind=label, path=str(path))
                continue
            files[label] = file_record(path)
        summary: dict[str, Any] = {}
        if "metadata" in entry and resolve(entry["metadata"]).is_file():
            metadata = load_json(resolve(entry["metadata"]))
            summary = {
                "bones": int(metadata["bone_count"]),
                "materials": len(metadata["materials"]),
                "meshes": len(metadata["meshes"]),
                "mesh_vertices": sum(int(mesh["vertices"]) for mesh in metadata["meshes"]),
                "maximum_source_influences": max(
                    int(influences)
                    for mesh in metadata["meshes"]
                    for influences in mesh["influence_histogram"]
                ),
            }
        result[key] = {
            "name": entry["name"],
            "purpose": entry["purpose"],
            "files": files,
            "summary": summary,
            "container_blocks_reusable": False,
        }
    return result


def compatibility(
    containers: dict[str, dict[str, Any]],
    inventories: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    head_inventory = inventories["head_hair"]
    body_inventory = inventories["body_garment"]
    head_names = skeleton_names(head_inventory)
    body_names = skeleton_names(body_inventory)
    head_parents = skeleton_parents(head_inventory)
    body_parents = skeleton_parents(body_inventory)
    common_prefix = 0
    for index, (head_name, body_name, head_parent, body_parent) in enumerate(
        zip(head_names, body_names, head_parents, body_parents, strict=False)
    ):
        if head_name != body_name or head_parent != body_parent:
            break
        common_prefix = index + 1

    matched_names = sorted(set(head_names) & set(body_names))
    head_by_name = {bone["name"]: bone for bone in head_inventory["bones"]}
    body_by_name = {bone["name"]: bone for bone in body_inventory["bones"]}
    rest_differences = [
        {
            "bone": name,
            "head_distance_m": vec_distance(
                head_by_name[name]["head_world_rest"], body_by_name[name]["head_world_rest"]
            ),
        }
        for name in matched_names
    ]
    rest_differences.sort(key=lambda item: (-item["head_distance_m"], item["bone"]))

    face_path = resolve(config["display_target"]["semantic_donors"]["face"]["skeleton_inventory"])
    face_inventory = load_json(face_path)
    face_names = set(skeleton_names(face_inventory))
    return {
        "physical_skeletons": {
            "common_name_parent_prefix": common_prefix,
            "shared_bone_names": len(matched_names),
            "binary_rest_compatible": False,
            "maximum_matching_bone_head_difference_m": rest_differences[0]["head_distance_m"],
            "largest_rest_differences": rest_differences[:12],
            "rule": "fit independently in each physical container REST space; never copy a semantic donor Skin buffer",
        },
        "face_semantic_to_head_container": {
            "face_bones": len(face_names),
            "shared_names": len(face_names & set(head_names)),
            "head_container_bones": len(head_names),
            "rule": "transfer facial semantics only onto bones present in the retained Head/Hair container Skel",
        },
        "containers": {key: value["name"] for key, value in containers.items()},
    }


def main() -> int:
    args = arguments()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output = args.output.resolve() if args.output else resolve(config["display_target"]["contract_output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    errors: list[dict[str, Any]] = []
    source = validate_source(config, errors)
    lod_plan = validate_lods(config, errors)
    containers: dict[str, dict[str, Any]] = {}
    inventories: dict[str, dict[str, Any]] = {}
    for key, entry in config["display_target"]["physical_containers"].items():
        record, inventory = validate_container(key, entry, errors)
        if record:
            containers[key] = record
        if inventory is not None:
            inventories[key] = inventory
    semantics = semantic_donors(config, errors)
    compatible = (
        compatibility(containers, inventories, config)
        if set(inventories) == {"head_hair", "body_garment"}
        else {}
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Authoritative FBX-first source, LOD, donor and first-writer contract for Si Display.",
        "scope": config["display_target"]["scope"],
        "driver_assets_included": bool(config["display_target"]["driver_assets_included"]),
        "config": file_record(config_path),
        "source_lock": source,
        "lod_plan": lod_plan,
        "physical_containers": containers,
        "semantic_donors": semantics,
        "compatibility": compatible,
        "validation": {"hard_error_count": len(errors), "hard_errors": errors},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_DISPLAY_CONTRACT="
        + json.dumps(
            {
                "output": str(output),
                "lods": len(lod_plan.get("lods", [])),
                "containers": sorted(containers),
                "semantic_donors": sorted(semantics),
                "hard_error_count": len(errors),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
