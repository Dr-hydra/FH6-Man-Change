#!/usr/bin/env python3
"""Retarget an FH6 intermediate Skin stream to a compatible Driver skeleton.

Geometry, UVs, normals, tangents and indices remain intact.  The tool remaps
bone indices by name, merges influences that collapse onto one Driver bone, and
writes a self-contained intermediate for a target Driver donor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("target_inspection", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--fallback-profile",
        choices=("exact", "driver-female-outfit"),
        default="exact",
        help="Named fallback policy for source bones absent from the target skeleton.",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_file(manifest_path: Path, manifest: dict[str, Any], key: str) -> Path:
    raw_path = Path(str(manifest["files"][key]["path"]))
    if not raw_path.is_absolute():
        raw_path = manifest_path.parent / raw_path
    return raw_path.resolve(strict=True)


def fallback_target(name: str, profile: str) -> str | None:
    if profile != "driver-female-outfit":
        return None
    for side in ("Left", "Right"):
        if not name.startswith(side):
            continue
        if name.startswith(f"{side}Knee_Corrective") or name.startswith(f"{side}Leg_TWIST"):
            return f"{side}Leg"
        if (
            name.startswith(f"{side}UpLeg_TWIST")
            or name.startswith(f"{side}Dress")
            or name.startswith(f"{side}Cloth_Corrective")
        ):
            return f"{side}UpLeg"
    return None


def weight_offset(manifest: dict[str, Any]) -> int:
    fields = manifest["files"]["vertices"].get("fields", [])
    for field in fields:
        if field.get("name") == "weights":
            return int(field["offset"])
    if int(manifest["files"]["vertices"].get("stride", 0)) == 64:
        # Early FH6 head intermediates record the standard 64-byte layout
        # without repeating its field table.
        return 48
    raise ValueError("Intermediate vertex layout has no weights field")


def main() -> int:
    args = arguments()
    source_manifest_path = args.source_manifest.resolve(strict=True)
    target_inspection_path = args.target_inspection.resolve(strict=True)
    output_manifest_path = args.output_manifest.resolve()
    report_path = args.report.resolve()
    for path in (output_manifest_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    target_inspection = json.loads(target_inspection_path.read_text(encoding="utf-8"))
    if target_inspection["parsed"]["errors"]:
        raise ValueError(f"Target inspection has parser errors: {target_inspection['parsed']['errors']}")
    skeletons = target_inspection["parsed"]["skeleton"]
    if len(skeletons) != 1:
        raise ValueError(f"Expected one target skeleton, found {len(skeletons)}")
    target_order = [str(bone["name"]) for bone in skeletons[0]["bones"]]
    if len(target_order) != len(set(target_order)):
        raise ValueError("Target skeleton contains duplicate bone names")
    target_indices = {name: index for index, name in enumerate(target_order)}

    source_order = [str(name) for name in source_manifest["skinning"]["bone_order"]]
    if len(source_order) != len(set(source_order)):
        raise ValueError("Source intermediate has duplicate bone names")
    source_bone_count = len(source_order)

    source_vertices_path = input_file(source_manifest_path, source_manifest, "vertices")
    source_bones_path = input_file(source_manifest_path, source_manifest, "bone_indices")
    source_indices_path = input_file(source_manifest_path, source_manifest, "indices")
    vertex_stride = int(source_manifest["files"]["vertices"]["stride"])
    bone_stride = int(source_manifest["files"]["bone_indices"]["stride"])
    index_stride = int(source_manifest["files"]["indices"]["stride"])
    weights_offset = weight_offset(source_manifest)
    if vertex_stride < weights_offset + 16 or bone_stride != 8 or index_stride != 2:
        raise ValueError("Unsupported intermediate binary layout")

    vertices = bytearray(source_vertices_path.read_bytes())
    bones = bytearray(source_bones_path.read_bytes())
    if len(vertices) % vertex_stride or len(bones) % bone_stride:
        raise ValueError("Source intermediate buffer size is not aligned to its declared stride")
    vertex_count = len(vertices) // vertex_stride
    if len(bones) // bone_stride != vertex_count:
        raise ValueError("Source vertex and bone-index counts differ")

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_manifest_path.name.removesuffix(".manifest.json")
    vertices_output = output_manifest_path.parent / f"{output_stem}.vertices.bin"
    bones_output = output_manifest_path.parent / f"{output_stem}.bone-indices.bin"
    indices_output = output_manifest_path.parent / f"{output_stem}.indices.bin"
    for path in (vertices_output, bones_output, indices_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    source_to_target: dict[int, tuple[int, str]] = {}
    fallback_counts: Counter[str] = Counter()
    fallback_mass: Counter[str] = Counter()
    merged_vertices = 0
    changed_vertices = 0
    active_target_indices: set[int] = set()
    weight_sums: list[float] = []

    for vertex_index in range(vertex_count):
        vertex_offset = vertex_index * vertex_stride
        bone_offset = vertex_index * bone_stride
        weights = struct.unpack_from("<4f", vertices, vertex_offset + weights_offset)
        source_indices = struct.unpack_from("<4H", bones, bone_offset)
        merged: dict[int, float] = {}
        source_targets: list[int] = []
        for weight, source_index in zip(weights, source_indices):
            if source_index >= source_bone_count:
                raise ValueError(
                    f"Vertex {vertex_index} references source bone {source_index}, "
                    f"outside 0..{source_bone_count - 1}"
                )
            if weight <= 1e-8:
                continue
            target = source_to_target.get(source_index)
            if target is None:
                source_name = source_order[source_index]
                target_name = source_name if source_name in target_indices else fallback_target(
                    source_name, args.fallback_profile
                )
                if target_name is None or target_name not in target_indices:
                    raise ValueError(
                        f"Active source bone {source_name!r} has no Driver mapping "
                        f"under profile {args.fallback_profile!r}"
                    )
                method = "exact" if target_name == source_name else "fallback"
                target = (target_indices[target_name], method)
                source_to_target[source_index] = target
            target_index, method = target
            merged[target_index] = merged.get(target_index, 0.0) + float(weight)
            source_targets.append(target_index)
            if method == "fallback":
                key = f"{source_order[source_index]}->{target_order[target_index]}"
                fallback_counts[key] += 1
                fallback_mass[key] += float(weight)

        if not merged:
            raise ValueError(f"Vertex {vertex_index} has no active Skin influence")
        sorted_influences = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:4]
        total = sum(weight for _, weight in sorted_influences)
        if total <= 1e-8:
            raise ValueError(f"Vertex {vertex_index} has zero merged Skin weight")
        target_weights = [weight / total for _, weight in sorted_influences]
        target_bones = [index for index, _ in sorted_influences]
        while len(target_weights) < 4:
            target_weights.append(0.0)
            target_bones.append(0)
        if len(set(source_targets)) != len(source_targets):
            merged_vertices += 1
        if tuple(source_indices) != tuple(target_bones) or any(
            abs(left - right) > 1e-7 for left, right in zip(weights, target_weights)
        ):
            changed_vertices += 1
        struct.pack_into("<4f", vertices, vertex_offset + weights_offset, *target_weights)
        struct.pack_into("<4H", bones, bone_offset, *target_bones)
        active_target_indices.update(target_bones[index] for index, weight in enumerate(target_weights) if weight > 0)
        weight_sums.append(sum(target_weights))

    vertices_output.write_bytes(vertices)
    bones_output.write_bytes(bones)
    shutil.copyfile(source_indices_path, indices_output)

    output_manifest = json.loads(json.dumps(source_manifest))
    output_manifest["created_local"] = datetime.now(timezone.utc).astimezone().isoformat()
    output_manifest["driver_skin_remap"] = {
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_path(source_manifest_path),
        "target_inspection": str(target_inspection_path),
        "target_inspection_sha256": sha256_path(target_inspection_path),
        "fallback_profile": args.fallback_profile,
        "source_bones": source_bone_count,
        "target_bones": len(target_order),
        "changed_vertices": changed_vertices,
        "merged_vertices": merged_vertices,
    }
    skinning = output_manifest["skinning"]
    skinning["bone_order"] = target_order
    skinning["used_bones"] = [target_order[index] for index in sorted(active_target_indices)]
    if "bone_count" in skinning:
        skinning["bone_count"] = len(target_order)
    if "skeleton_bones" in skinning:
        skinning["skeleton_bones"] = len(target_order)
    skinning["zero_weight_vertices"] = 0
    skinning["weight_sum_min"] = min(weight_sums)
    skinning["weight_sum_max"] = max(weight_sums)
    output_manifest["files"]["vertices"].update(
        {
            "path": vertices_output.name,
            "bytes": vertices_output.stat().st_size,
            "sha256": sha256_path(vertices_output),
        }
    )
    output_manifest["files"]["bone_indices"].update(
        {
            "path": bones_output.name,
            "bytes": bones_output.stat().st_size,
            "sha256": sha256_path(bones_output),
        }
    )
    output_manifest["files"]["indices"].update(
        {
            "path": indices_output.name,
            "bytes": indices_output.stat().st_size,
            "sha256": sha256_path(indices_output),
        }
    )
    output_manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fallback_report = [
        {
            "mapping": mapping,
            "weighted_assignments": fallback_counts[mapping],
            "weight_mass": fallback_mass[mapping],
        }
        for mapping in sorted(fallback_counts)
    ]
    report = {
        "schema_version": 1,
        "created_local": datetime.now(timezone.utc).astimezone().isoformat(),
        "state": "offline-candidate",
        "purpose": "FH6 intermediate Driver Skin remap with preserved geometry and indices.",
        "source": {
            "manifest": str(source_manifest_path),
            "manifest_sha256": sha256_path(source_manifest_path),
            "vertices": {"path": str(source_vertices_path), "sha256": sha256_path(source_vertices_path)},
            "bone_indices": {"path": str(source_bones_path), "sha256": sha256_path(source_bones_path)},
            "indices": {"path": str(source_indices_path), "sha256": sha256_path(source_indices_path)},
        },
        "target": {
            "inspection": str(target_inspection_path),
            "inspection_sha256": sha256_path(target_inspection_path),
            "bone_count": len(target_order),
        },
        "output": {
            "manifest": str(output_manifest_path),
            "manifest_sha256": sha256_path(output_manifest_path),
            "vertices": {"path": str(vertices_output), "sha256": sha256_path(vertices_output)},
            "bone_indices": {"path": str(bones_output), "sha256": sha256_path(bones_output)},
            "indices": {"path": str(indices_output), "sha256": sha256_path(indices_output)},
        },
        "remap": {
            "fallback_profile": args.fallback_profile,
            "source_bones": source_bone_count,
            "target_bones": len(target_order),
            "active_target_bones": len(active_target_indices),
            "changed_vertices": changed_vertices,
            "merged_vertices": merged_vertices,
            "fallbacks": fallback_report,
            "weight_sum_min": min(weight_sums),
            "weight_sum_max": max(weight_sums),
        },
        "validation": {
            "vertex_count": vertex_count,
            "max_influences": 4,
            "all_bone_indices_resolve": max(active_target_indices, default=0) < len(target_order),
            "geometry_payload_preserved": sha256_path(source_indices_path) == sha256_path(indices_output),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_DRIVER_INTERMEDIATE="
        + json.dumps(
            {
                "manifest": str(output_manifest_path),
                "report": str(report_path),
                "vertices": vertex_count,
                "changed_vertices": changed_vertices,
                "fallbacks": len(fallback_report),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
