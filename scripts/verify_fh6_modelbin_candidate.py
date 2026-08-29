#!/usr/bin/env python3
"""Verify an FBX-first FH6 Display modelbin candidate against its donor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import BundleBlob, BundleError, first_difference, parse_bundle
from patch_fh6_material_profile import (
    materials_by_id,
    normalize_patches,
    patch_mtpr,
    shader_info,
)


PRESERVED_TAGS = {"Skel", "VLay", "MatI"}
HEAD_DISPLAY_RENDER_PASSES = {
    0: 0x38,
    1: 0x19,
    2: 0x19,
    3: 0x19,
    4: 0x38,
    5: 0x3C,
    6: 0x19,
    7: 0x38,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("donor", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--component", required=True, choices=("helmet", "outfit"))
    parser.add_argument(
        "--material-profile",
        type=Path,
        help=(
            "Allow MatI payload changes and verify every final material against "
            "this explicit profile. The donor must be the pre-material geometry candidate."
        ),
    )
    parser.add_argument(
        "--duplicate-draws-for-lod-groups",
        action="store_true",
        help=(
            "Expect every active donor LOD group to reuse the manifest draw "
            "domain through its own contiguous duplicate index partition."
        ),
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_signature(blob: BundleBlob) -> list[tuple[str, int, str]]:
    return [(item.tag, item.version, sha256_bytes(item.value)) for item in blob.metadata]


def main() -> None:
    args = arguments()
    donor_path = args.donor.resolve()
    candidate_path = args.candidate.resolve()
    manifest_path = args.manifest.resolve()
    material_profile_path = (
        args.material_profile.resolve(strict=True) if args.material_profile else None
    )
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    material_profile = (
        json.loads(material_profile_path.read_text(encoding="utf-8"))
        if material_profile_path
        else None
    )
    if material_profile is not None and material_profile.get("schema_version") != 1:
        raise ValueError("Unsupported material profile schema")
    donor_data = donor_path.read_bytes()
    candidate_data = candidate_path.read_bytes()
    donor_outer = parse_bundle(donor_data)
    candidate_outer = parse_bundle(candidate_data)
    candidate_inspection = inspector.inspect(candidate_path)
    parsed = candidate_inspection["parsed"]

    checks: list[dict] = []
    failures: list[str] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            failures.append(name)

    check("parser.errors", not parsed["errors"], parsed["errors"])
    check(
        "bundle.declared_size",
        candidate_inspection["header"]["declared_size"] == len(candidate_data),
        {"declared": candidate_inspection["header"]["declared_size"], "actual": len(candidate_data)},
    )
    check(
        "bundle.tag_sequence",
        [blob.tag for blob in donor_outer.blobs] == [blob.tag for blob in candidate_outer.blobs],
        {"donor": donor_outer.blob_tags, "candidate": candidate_outer.blob_tags},
    )

    preserved_tags = PRESERVED_TAGS - ({"MatI"} if material_profile else set())
    preserved: list[dict] = []
    for donor_blob, candidate_blob in zip(donor_outer.blobs, candidate_outer.blobs):
        if donor_blob.tag not in preserved_tags:
            continue
        data_equal = donor_blob.data == candidate_blob.data
        metadata_equal = metadata_signature(donor_blob) == metadata_signature(candidate_blob)
        preserved.append(
            {
                "blob_index": donor_blob.index,
                "tag": donor_blob.tag,
                "data_equal": data_equal,
                "metadata_equal": metadata_equal,
                "sha256": sha256_bytes(candidate_blob.data),
            }
        )
    check(
        "preserved." + "_".join(sorted(preserved_tags)),
        bool(preserved) and all(item["data_equal"] and item["metadata_equal"] for item in preserved),
        preserved,
    )

    outer_roundtrip_difference = first_difference(
        candidate_data, candidate_outer.rebuild_lossless()
    )
    check(
        "bundle.lossless_roundtrip",
        outer_roundtrip_difference is None,
        {"first_different_offset": outer_roundtrip_difference},
    )

    material_checks: list[dict] = []
    if material_profile is not None:
        unchanged_non_material = []
        for donor_blob, candidate_blob in zip(donor_outer.blobs, candidate_outer.blobs):
            if donor_blob.tag == "MatI":
                continue
            equal = (
                donor_blob.tag == candidate_blob.tag
                and donor_blob.version == candidate_blob.version
                and donor_blob.trailing_size == candidate_blob.trailing_size
                and metadata_signature(donor_blob) == metadata_signature(candidate_blob)
                and donor_blob.data == candidate_blob.data
            )
            unchanged_non_material.append(
                {
                    "blob_index": candidate_blob.index,
                    "tag": candidate_blob.tag,
                    "equal": equal,
                }
            )
        check(
            "materials.only_MatI_payloads_changed",
            bool(unchanged_non_material)
            and all(item["equal"] for item in unchanged_non_material),
            unchanged_non_material,
        )

        donor_materials = materials_by_id(donor_outer)
        candidate_materials = materials_by_id(candidate_outer)
        profile_specs = {
            int(item["target_material_id"]): item
            for item in material_profile.get("materials", [])
        }
        profile_ids = [
            int(item["target_material_id"])
            for item in material_profile.get("materials", [])
        ]
        unique_profile_ids = len(profile_specs) == len(profile_ids)
        material_domains_match = (
            unique_profile_ids
            and set(donor_materials) == set(candidate_materials) == set(profile_specs)
        )
        check(
            "materials.profile_domains",
            material_domains_match,
            {
                "donor": sorted(donor_materials),
                "candidate": sorted(candidate_materials),
                "profile": sorted(profile_specs),
                "profile_ids_unique": unique_profile_ids,
            },
        )

        materials_valid = material_domains_match
        if material_domains_match:
            for material_id_value, material in sorted(candidate_materials.items()):
                spec = profile_specs[material_id_value]
                donor_material = donor_materials[material_id_value]
                metadata_equal = metadata_signature(donor_material) == metadata_signature(
                    material
                )
                try:
                    shader, atst, nested, mtpr = shader_info(material.data)
                    nested_difference = first_difference(
                        material.data, nested.rebuild_lossless()
                    )
                    texture_patches = normalize_patches(spec, "texture_patches")
                    value_patches = normalize_patches(spec, "value_patches")
                    repatched_mtpr, _changes, texture_hashes = patch_mtpr(
                        mtpr.data,
                        texture_patches,
                        value_patches,
                        require_all_textures=bool(
                            spec.get("require_all_template_textures_patched", True)
                        ),
                    )
                    profile_exact = repatched_mtpr == mtpr.data
                    current_valid = (
                        metadata_equal
                        and nested_difference is None
                        and shader == spec["expected_shader"]
                        and atst == spec["expected_atst"]
                        and profile_exact
                    )
                    evidence = {
                        "material_id": material_id_value,
                        "role": spec.get("role"),
                        "metadata_equal": metadata_equal,
                        "shader": shader,
                        "expected_shader": spec["expected_shader"],
                        "atst": atst,
                        "expected_atst": spec["expected_atst"],
                        "nested_roundtrip": nested_difference is None,
                        "nested_first_different_offset": nested_difference,
                        "profile_exact": profile_exact,
                        "texture_hashes": [
                            f"{item:08x}" for item in sorted(texture_hashes)
                        ],
                    }
                except (BundleError, KeyError, ValueError) as exc:
                    current_valid = False
                    evidence = {
                        "material_id": material_id_value,
                        "role": spec.get("role"),
                        "metadata_equal": metadata_equal,
                        "error": str(exc),
                    }
                materials_valid &= current_valid
                evidence["valid"] = current_valid
                material_checks.append(evidence)
        check("materials.nested_profile_roundtrip", materials_valid, material_checks)

    blob_spans = sorted(
        (int(blob["data_offset"]), int(blob["data_offset"]) + int(blob["data_size"]), blob["tag"], blob["index"])
        for blob in candidate_inspection["blobs"]
    )
    spans_valid = (
        bool(blob_spans)
        and blob_spans[0][0] == candidate_inspection["header"]["data_offset"]
        and all(left[1] <= right[0] for left, right in zip(blob_spans, blob_spans[1:]))
        and blob_spans[-1][1] <= len(candidate_data)
    )
    check(
        "bundle.absolute_blob_offsets",
        spans_valid,
        {
            "data_offset": candidate_inspection["header"]["data_offset"],
            "first_span": blob_spans[0] if blob_spans else None,
            "last_span": blob_spans[-1] if blob_spans else None,
            "file_bytes": len(candidate_data),
        },
    )

    skeleton = parsed["skeleton"][0]
    bone_count = int(skeleton["bone_count"])
    meshes = parsed["meshes"]
    index_buffer = parsed["index_buffers"][0]
    vertex_buffers = {int(item["id"]): item for item in parsed["vertex_buffers"]}
    skin_buffers = {int(item["id"]): item for item in parsed["skin_buffers"]}
    index_blob = candidate_outer.blobs[int(index_buffer["blob_index"])].data
    index_count = int(index_buffer["count"])
    index_payload = index_blob[16 : 16 + int(index_buffer["byte_size"])]
    indices = list(struct.unpack(f"<{index_count}H", index_payload))
    lod_flags = sorted({int(mesh["lod_flags"]) for mesh in meshes})
    expected_index_multiplier = len(lod_flags) if args.duplicate_draws_for_lod_groups else 1
    expected_index_count = int(manifest["geometry"]["indices"]) * expected_index_multiplier
    check(
        "index.manifest_count",
        index_count == expected_index_count,
        {
            "candidate": index_count,
            "manifest": int(manifest["geometry"]["indices"]),
            "lod_groups": lod_flags,
            "expected_multiplier": expected_index_multiplier,
            "expected": expected_index_count,
        },
    )
    check(
        "vertex.manifest_count",
        int(manifest["geometry"]["export_vertices"]) == int(vertex_buffers[0]["count"]),
        {"candidate": int(vertex_buffers[0]["count"]), "manifest": int(manifest["geometry"]["export_vertices"])},
    )
    check(
        "index.r16_domain",
        int(vertex_buffers[0]["count"]) <= 65535 and max(indices, default=0) < 65536,
        {"vertices": int(vertex_buffers[0]["count"]), "maximum_index": max(indices, default=None)},
    )

    ranges = sorted(
        (int(mesh["start_index"]), int(mesh["start_index"]) + int(mesh["index_count"]), int(mesh["material_id"]))
        for mesh in meshes
        if int(mesh["index_count"]) > 0
    )
    ranges_cover = (
        bool(ranges)
        and ranges[0][0] == 0
        and ranges[-1][1] == index_count
        and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        and all((end - start) % 3 == 0 for start, end, _ in ranges)
    )
    check("mesh.index_partition", ranges_cover, ranges)

    draws = {int(item["material_id"]): item for item in manifest["geometry"]["draws"]}
    mesh_materials = {int(mesh["material_id"]) for mesh in meshes}
    check("mesh.material_domains", mesh_materials == set(draws), {"candidate": sorted(mesh_materials), "manifest": sorted(draws)})
    if args.duplicate_draws_for_lod_groups:
        lod_domains = {
            lod_flag: sorted(
                int(mesh["material_id"])
                for mesh in meshes
                if int(mesh["lod_flags"]) == lod_flag
            )
            for lod_flag in lod_flags
        }
        check(
            "lod.duplicate_draw_domains",
            len(lod_flags) >= 2
            and all(materials == sorted(draws) for materials in lod_domains.values()),
            {"lod_groups": lod_domains, "manifest_materials": sorted(draws)},
        )

    mesh_domains: list[dict] = []
    mesh_domains_valid = True
    for mesh in meshes:
        material_id = int(mesh["material_id"])
        start = int(mesh["start_index"])
        count = int(mesh["index_count"])
        raw = indices[start : start + count]
        base_vertex = int(mesh["base_vertex"])
        skin_id = int(mesh["skinning_data_buffer_id"])
        skin = skin_buffers[skin_id]
        effective_skin = [value + base_vertex for value in raw]
        domain_valid = bool(raw) and min(effective_skin) >= 0 and max(effective_skin) < int(skin["count"])
        binding_ranges = []
        for binding in mesh["vertex_buffers"]:
            buffer_id = int(binding["id"])
            buffer = vertex_buffers[buffer_id]
            stride = int(binding["stride"])
            offset = int(binding["offset"])
            divisible = stride > 0 and offset % stride == 0
            effective = [value + base_vertex + (offset // stride if divisible else 0) for value in raw]
            binding_valid = divisible and bool(effective) and min(effective) >= 0 and max(effective) < int(buffer["count"])
            domain_valid &= binding_valid
            binding_ranges.append(
                {
                    "buffer_id": buffer_id,
                    "minimum": min(effective, default=None),
                    "maximum": max(effective, default=None),
                    "buffer_count": int(buffer["count"]),
                    "valid": binding_valid,
                }
            )
        draw = draws.get(material_id)
        draw_valid = (
            draw is not None
            and count == int(draw["index_count"])
            and int(mesh["extended_unknown_u32"][1]) == int(draw["vertex_count"])
            and int(mesh["extended_array_count"]) == int(draw["vertex_count"]) * 2
        )
        influence_valid = int(mesh["skinning_elements"]) == int(skin["influences_per_vertex"])
        domain_valid &= draw_valid and influence_valid
        mesh_domains_valid &= domain_valid
        mesh_domains.append(
            {
                "material_id": material_id,
                "lod_flags": int(mesh["lod_flags"]),
                "start_index": start,
                "index_count": count,
                "raw_index_min": min(raw, default=None),
                "raw_index_max": max(raw, default=None),
                "base_vertex": base_vertex,
                "skin_buffer_id": skin_id,
                "skin_min": min(effective_skin, default=None),
                "skin_max": max(effective_skin, default=None),
                "skin_count": int(skin["count"]),
                "skinning_elements": int(mesh["skinning_elements"]),
                "draw_valid": draw_valid,
                "bindings": binding_ranges,
                "valid": domain_valid,
            }
        )
    check("mesh.vertex_skin_domains", mesh_domains_valid, mesh_domains)

    if args.component == "helmet" and manifest["geometry"].get("draw_policy") in {
        "helmet6_face_combined",
        "head6_display",
        "head7_display",
        "head8_f04_skin",
    }:
        actual_render_passes = {int(mesh["material_id"]): int(mesh["render_pass"]) for mesh in meshes}
        expected_render_passes = {
            material_id: HEAD_DISPLAY_RENDER_PASSES[material_id]
            for material_id in sorted(draws)
        }
        check(
            "helmet.head_display_render_passes",
            actual_render_passes == expected_render_passes,
            {"actual": actual_render_passes, "expected": expected_render_passes},
        )

    skin_summaries: list[dict] = []
    skin_valid = True
    for skin_id, skin in sorted(skin_buffers.items()):
        blob = candidate_outer.blobs[int(skin["blob_index"])].data
        count = int(skin["count"])
        stride = int(skin["stride"])
        influences = int(skin["influences_per_vertex"])
        payload = blob[16 : 16 + int(skin["byte_size"])]
        minimum_sum = math.inf
        maximum_sum = -math.inf
        minimum_active = 5
        maximum_active = 0
        bad_bones = 0
        for vertex_index in range(count):
            weights = []
            active = 0
            for influence in range(influences):
                weight, bone = struct.unpack_from("<ee", payload, vertex_index * stride + influence * 4)
                weights.append(float(weight))
                if weight > 0.0:
                    active += 1
                    if bone != round(bone) or bone < 0 or bone >= bone_count:
                        bad_bones += 1
            weight_sum = sum(weights)
            minimum_sum = min(minimum_sum, weight_sum)
            maximum_sum = max(maximum_sum, weight_sum)
            minimum_active = min(minimum_active, active)
            maximum_active = max(maximum_active, active)
        current_valid = (
            stride == influences * 4
            and 1 <= influences <= 4
            and minimum_active >= 1
            and maximum_active <= 4
            and minimum_sum >= 0.999
            and maximum_sum <= 1.001
            and bad_bones == 0
        )
        skin_valid &= current_valid
        skin_summaries.append(
            {
                "id": skin_id,
                "count": count,
                "stride": stride,
                "influences_per_vertex": influences,
                "active_influences_min": minimum_active,
                "active_influences_max": maximum_active,
                "weight_sum_min": minimum_sum,
                "weight_sum_max": maximum_sum,
                "bad_active_bone_indices": bad_bones,
                "bone_count": bone_count,
                "valid": current_valid,
            }
        )
    check("skin.contract", skin_valid, skin_summaries)

    blob_metadata = {int(item["index"]): item.get("metadata", {}) for item in candidate_inspection["blobs"]}
    mesh_bounds: list[dict] = []
    bounds_valid = True
    for mesh in meshes:
        value = blob_metadata[int(mesh["blob_index"])].get("BBox", {}).get("value")
        current_valid = (
            isinstance(value, list)
            and len(value) == 6
            and all(math.isfinite(float(item)) for item in value)
            and all(float(value[axis]) <= float(value[axis + 3]) for axis in range(3))
        )
        bounds_valid &= current_valid
        mesh_bounds.append({"material_id": int(mesh["material_id"]), "bbox": value, "valid": current_valid})
    modl_blob = next(blob for blob in candidate_inspection["blobs"] if blob["tag"] == "Modl")
    model_bounds = modl_blob.get("metadata", {}).get("BBox", {}).get("value")
    if isinstance(model_bounds, list) and len(model_bounds) == 6 and mesh_bounds:
        for item in mesh_bounds:
            value = item["bbox"]
            if not isinstance(value, list) or len(value) != 6:
                bounds_valid = False
                continue
            for axis in range(3):
                bounds_valid &= float(model_bounds[axis]) <= float(value[axis]) + 1e-6
                bounds_valid &= float(model_bounds[axis + 3]) >= float(value[axis + 3]) - 1e-6
    else:
        bounds_valid = False
    check("bounds.mesh_and_model", bounds_valid, {"model": model_bounds, "meshes": mesh_bounds})

    expected_skin_streams = 2 if args.component == "helmet" else 1
    expected_vertex_streams = 3 if args.component == "helmet" else 2
    check(
        "component.buffer_shape",
        len(skin_buffers) == expected_skin_streams and len(vertex_buffers) == expected_vertex_streams,
        {
            "component": args.component,
            "skin_streams": len(skin_buffers),
            "vertex_streams": len(vertex_buffers),
            "expected_skin_streams": expected_skin_streams,
            "expected_vertex_streams": expected_vertex_streams,
        },
    )

    report = {
        "schema_version": 1,
        "purpose": (
            "Structural and material-profile gate for an FBX-first FH6 Display "
            "modelbin candidate; no game-load claim."
            if material_profile is not None
            else "Structural gate for an FBX-first FH6 Display modelbin candidate; no game-load claim."
        ),
        "component": args.component,
        "donor": {"path": str(donor_path), "bytes": len(donor_data), "sha256": sha256(donor_path)},
        "candidate": {"path": str(candidate_path), "bytes": len(candidate_data), "sha256": sha256(candidate_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "material_profile": (
            {"path": str(material_profile_path), "sha256": sha256(material_profile_path)}
            if material_profile_path
            else None
        ),
        "summary": {"checks": len(checks), "passed": len(checks) - len(failures), "failed": len(failures), "hard_errors": failures},
        "checks": checks,
        "lod_observation": {
            "mesh_lod_flags": sorted({int(mesh["lod_flags"]) for mesh in meshes}),
            "policy": "LOD0 geometry is shared by every donor-enabled LOD flag in this first structural candidate.",
        },
        "validation_level": {"structural": not failures, "blender_visual": False, "offline_game": False},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_MODELBIN_CANDIDATE_VERIFY="
        + json.dumps(
            {
                "candidate": str(candidate_path),
                "report": str(report_path),
                "checks": len(checks),
                "failed": len(failures),
                "hard_errors": failures,
            },
            separators=(",", ":"),
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
