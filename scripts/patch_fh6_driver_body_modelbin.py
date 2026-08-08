#!/usr/bin/env python3
"""Patch Driver_Alice_F with a validated six-draw head/body intermediate."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle as parse_outer_bundle
from modelbin_bundle import rebuild_with_blob_data
from patch_fh6_face_modelbin import attribute_binding, patch_position_binding, position_binding
from patch_fh6_garment_modelbin import (
    build_model_buffer,
    describe_attribute_layout,
    encode_geometry,
    read_intermediate,
    sha256,
    sha256_bytes,
    skeleton_world_matrices,
)


EXPECTED_MATERIAL_MESH_COUNTS = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("donor", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--rigid-head",
        action="store_true",
        help="Bind head, eyelashes, and teeth rigidly to Head; eyes remain on LeftEye/RightEye.",
    )
    return parser.parse_args()


def copy_flipped_vertices(vertices: list[dict]) -> list[dict]:
    result = []
    for vertex in vertices:
        tangent = list(vertex["tangent"])
        result.append(
            {
                **vertex,
                "normal": [-float(value) for value in vertex["normal"]],
                "tangent": [float(tangent[0]), float(tangent[1]), float(tangent[2]), -float(tangent[3])],
            }
        )
    return result


def reduced_skin_payload(
    vertices: list[dict],
    stride: int,
    role: str,
    bone_order: list[str],
    *,
    rigid_head: bool = False,
) -> tuple[bytes, dict[str, object]]:
    if stride <= 0 or stride % 4:
        raise ValueError(f"Unsupported Driver Body Skin stride {stride}")
    influence_limit = stride // 4
    payload = bytearray()
    counts: dict[int, int] = {}
    rigid_eye_counts = {"LeftEye": 0, "RightEye": 0}
    for vertex in vertices:
        if role == "eyes":
            if influence_limit < 1:
                raise ValueError("Eye Skin stream cannot hold one influence")
            target_name = "LeftEye" if float(vertex["position"][0]) < 0.0 else "RightEye"
            assignments = [(bone_order.index(target_name), 1.0)]
            rigid_eye_counts[target_name] += 1
        elif rigid_head and role in {"head", "eyelashes", "teeth"}:
            if influence_limit < 1:
                raise ValueError(f"{role} Skin stream cannot hold one influence")
            assignments = [(bone_order.index("Head"), 1.0)]
        else:
            combined: dict[int, float] = {}
            for bone_index, weight in zip(vertex["bones"], vertex["weights"]):
                if float(weight) > 0.0:
                    combined[int(bone_index)] = combined.get(int(bone_index), 0.0) + float(weight)
            assignments = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:influence_limit]
            total = sum(weight for _, weight in assignments)
            if total <= 0.0:
                raise ValueError(f"{role} Skin reduction produced a zero-weight vertex")
            assignments = [(bone_index, weight / total) for bone_index, weight in assignments]
        counts[len(assignments)] = counts.get(len(assignments), 0) + 1
        padded = assignments + [(0, 0.0)] * (influence_limit - len(assignments))
        for bone_index, weight in padded:
            payload.extend(struct.pack("<ee", float(weight), float(bone_index)))
    return bytes(payload), {
        "stride": stride,
        "influence_limit": influence_limit,
        "influence_histogram": {str(key): value for key, value in sorted(counts.items())},
        "rigid_eye_vertices": rigid_eye_counts if role == "eyes" else None,
        "rigid_head": bool(rigid_head and role in {"head", "eyelashes", "teeth"}),
    }


def mesh_groups(meshes: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for mesh in meshes:
        result.setdefault(int(mesh["material_id"]), []).append(mesh)
    for values in result.values():
        values.sort(key=lambda item: int(item["start_index"]))
    return result


def main() -> None:
    args = arguments()
    donor = args.donor.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vertices, source_indices = read_intermediate(manifest_path, manifest)
    draws = manifest["geometry"].get("draws")
    if not isinstance(draws, list) or len(draws) != 6:
        raise ValueError("Driver Body intermediate must contain six draw ranges")
    draw_by_material = {int(draw["material_id"]): draw for draw in draws}
    if set(draw_by_material) != set(range(6)):
        raise ValueError("Driver Body intermediate draw IDs must be 0..5")

    donor_data = donor.read_bytes()
    outer = parse_outer_bundle(donor_data)
    parsed = inspector.parse_known_blobs(donor_data, inspector.parse_bundle(donor_data)[1])
    if parsed["errors"]:
        raise ValueError(f"Donor parser errors: {parsed['errors']}")
    tags = [blob.tag for blob in outer.blobs]
    tag_indices: dict[str, list[int]] = {}
    for index, tag in enumerate(tags):
        tag_indices.setdefault(tag, []).append(index)
    required_counts = {
        "Skel": 1,
        "MatI": 6,
        "Mesh": 7,
        "IndB": 1,
        "VLay": 2,
        "VerB": 4,
        "Skin": 3,
        "Modl": 1,
    }
    for tag, count in required_counts.items():
        if len(tag_indices.get(tag, [])) != count:
            raise ValueError(f"Expected {count} {tag} blobs, found {len(tag_indices.get(tag, []))}: {tags}")

    meshes = parsed["meshes"]
    grouped = mesh_groups(meshes)
    if {key: len(value) for key, value in grouped.items()} != EXPECTED_MATERIAL_MESH_COUNTS:
        raise ValueError("Driver_Alice_F Mesh/material multiplicity differs from the verified contract")
    if any(mesh["blob_version"] != "1.12" for mesh in meshes):
        raise ValueError("Driver_Alice_F Mesh descriptors must be version 1.12")
    eyelashes = grouped[1]
    if {mesh["name"] for mesh in eyelashes} != {"Eyelashes", "Eyelashes_Flip"}:
        raise ValueError("Driver_Alice_F eyelash Mesh names differ from the verified contract")

    layouts = {int(layout["id"]): layout for layout in parsed["vertex_layouts"]}
    vertex_buffers = {int(item["id"]): item for item in parsed["vertex_buffers"]}
    skin_buffers = {int(item["id"]): item for item in parsed["skin_buffers"]}
    if set(layouts) != {0, 1} or set(vertex_buffers) != {0, 1, 2, 3} or set(skin_buffers) != {0, 1, 2}:
        raise ValueError("Driver_Alice_F VLay/VerB/Skin IDs differ from the verified contract")

    skeleton = parsed["skeleton"][0]
    bone_order = [bone["name"] for bone in skeleton["bones"]]
    if bone_order != manifest["skinning"]["bone_order"]:
        raise ValueError("Intermediate donor bone order differs from Driver_Alice_F Skel order")
    world_matrices = skeleton_world_matrices(skeleton["bones"])
    anchor_indices = sorted({int(mesh["bone_index"]) for mesh in meshes})
    anchor_matrix = world_matrices[anchor_indices[0]]
    for anchor in anchor_indices[1:]:
        if any(abs(left - right) > 1e-6 for left, right in zip(anchor_matrix, world_matrices[anchor])):
            raise ValueError("Driver_Alice_F mesh anchor matrices differ")

    by_name = {mesh["name"]: mesh for mesh in meshes}
    specs = [
        {"key": "eyelashes", "mesh": by_name["Eyelashes"], "draw": 1, "role": "eyelashes", "group": "small", "flip": False},
        {"key": "eyelashes_flip", "mesh": by_name["Eyelashes_Flip"], "draw": 1, "role": "eyelashes", "group": "small", "flip": True},
        {"key": "eyes", "mesh": grouped[2][0], "draw": 2, "role": "eyes", "group": "small", "flip": False},
        {"key": "head", "mesh": grouped[0][0], "draw": 0, "role": "head", "group": "main", "flip": False},
        {"key": "body", "mesh": grouped[3][0], "draw": 3, "role": "body", "group": "main", "flip": False},
        {"key": "arms", "mesh": grouped[4][0], "draw": 4, "role": "arms", "group": "main", "flip": False},
        {"key": "teeth", "mesh": grouped[5][0], "draw": 5, "role": "teeth", "group": "teeth", "flip": False},
    ]
    if [spec["mesh"]["name"] for spec in sorted(specs, key=lambda item: int(item["mesh"]["start_index"]))] != [
        "Eyelashes",
        "Eyelashes_Flip",
        "Eyes",
        "CinematicHead",
        "Female_Body",
        "Female_Body",
        "Teeth",
    ]:
        raise ValueError("Driver_Alice_F index Mesh order differs from the verified contract")

    encoded: dict[str, dict[str, bytes]] = {}
    quantization: dict[str, dict[str, object]] = {}
    policies: dict[str, dict[str, object]] = {}
    local_indices: dict[str, list[int]] = {}
    spec_vertices: dict[str, list[dict]] = {}
    for spec in specs:
        draw = draw_by_material[int(spec["draw"])]
        vertex_start = int(draw["vertex_start"])
        vertex_count = int(draw["vertex_count"])
        values = vertices[vertex_start : vertex_start + vertex_count]
        if len(values) != vertex_count or vertex_count <= 0:
            raise ValueError(f"Driver Body draw {spec['draw']} has an invalid vertex range")
        if spec["flip"]:
            values = copy_flipped_vertices(values)
        spec_vertices[str(spec["key"])] = values

        mesh = spec["mesh"]
        layout = describe_attribute_layout(layouts[int(mesh["vertex_layout_id"])])
        attribute = attribute_binding(mesh)
        if int(attribute["stride"]) != int(layout["stride"]):
            raise ValueError(f"{spec['key']} Mesh/VLay attribute strides differ")
        values_encoded, values_quantization = encode_geometry(
            values,
            anchor_matrix,
            [{"material_id": 0, "vertex_start": 0, "vertex_count": vertex_count}],
            layout,
        )
        encoded[str(spec["key"])] = values_encoded
        draw_quantization = values_quantization["draws"][0]
        draw_quantization["material_id"] = int(spec["draw"])
        quantization[str(spec["key"])] = draw_quantization

        index_start = int(draw["start_index"])
        index_count = int(draw["index_count"])
        draw_indices = source_indices[index_start : index_start + index_count]
        values_indices = [int(index) - vertex_start for index in draw_indices]
        if len(values_indices) != index_count or any(index < 0 or index >= vertex_count for index in values_indices):
            raise ValueError(f"{spec['key']} index resolves outside its vertex range")
        local_indices[str(spec["key"])] = values_indices

        skin = skin_buffers[int(mesh["skinning_data_buffer_id"])]
        skin_data, policy = reduced_skin_payload(
            values,
            int(skin["stride"]),
            str(spec["role"]),
            bone_order,
            rigid_head=args.rigid_head,
        )
        encoded[str(spec["key"])]["skin"] = skin_data
        policies[str(spec["key"])] = policy

    position_starts: dict[str, int] = {}
    position_payload = bytearray()
    group_payloads = {
        "small": {"attribute": bytearray(), "skin": bytearray()},
        "main": {"attribute": bytearray(), "skin": bytearray()},
        "teeth": {"attribute": bytearray(), "skin": bytearray()},
    }
    group_counts = {"small": 0, "main": 0, "teeth": 0}
    for spec in specs:
        key = str(spec["key"])
        group = str(spec["group"])
        position_starts[key] = len(position_payload) // 8
        position_payload.extend(encoded[key]["position"])
        group_payloads[group]["attribute"].extend(encoded[key]["attribute"])
        group_payloads[group]["skin"].extend(encoded[key]["skin"])
        group_counts[group] += len(spec_vertices[key])
    if sum(group_counts.values()) != len(position_payload) // 8:
        raise ValueError("Driver Body position groups do not cover the shared Position buffer")

    group_position_offsets = {
        "small": 0,
        "main": group_counts["small"] * 8,
        "teeth": (group_counts["small"] + group_counts["main"]) * 8,
    }
    group_base_vertices = {
        "small": 0,
        "main": -group_counts["small"],
        "teeth": -(group_counts["small"] + group_counts["main"]),
    }
    index_specs = sorted(specs, key=lambda item: int(item["mesh"]["start_index"]))
    raw_indices: list[int] = []
    index_starts: dict[str, int] = {}
    for spec in index_specs:
        key = str(spec["key"])
        index_starts[key] = len(raw_indices)
        start = position_starts[key]
        values_indices = local_indices[key]
        for offset in range(0, len(values_indices), 3):
            first, second, third = values_indices[offset : offset + 3]
            if spec["flip"]:
                raw_indices.extend((first + start, second + start, third + start))
            else:
                raw_indices.extend((first + start, third + start, second + start))
    total_vertices = len(position_payload) // 8
    if max(raw_indices, default=0) >= total_vertices or total_vertices > 65_535:
        raise ValueError("Driver Body shared Position/index domain exceeds R16_UINT")

    replacements: dict[int, bytes] = {}
    index_buffer = parsed["index_buffers"][0]
    replacements[int(index_buffer["blob_index"])] = build_model_buffer(
        struct.pack(f"<{len(raw_indices)}H", *raw_indices),
        len(raw_indices),
        int(index_buffer["stride"]),
        tuple(index_buffer["flags"]),
        int(index_buffer["format"]),
    )
    position_buffer = vertex_buffers[0]
    replacements[int(position_buffer["blob_index"])] = build_model_buffer(
        bytes(position_payload),
        total_vertices,
        int(position_buffer["stride"]),
        tuple(position_buffer["flags"]),
        int(position_buffer["format"]),
    )

    group_buffers = {
        "small": (vertex_buffers[1], skin_buffers[1]),
        "main": (vertex_buffers[2], skin_buffers[0]),
        "teeth": (vertex_buffers[3], skin_buffers[2]),
    }
    for group, (attribute_buffer, skin_buffer) in group_buffers.items():
        count = group_counts[group]
        attribute_data = bytes(group_payloads[group]["attribute"])
        skin_data = bytes(group_payloads[group]["skin"])
        if len(attribute_data) != count * int(attribute_buffer["stride"]):
            raise ValueError(f"{group} attribute payload size differs from the donor layout")
        if len(skin_data) != count * int(skin_buffer["stride"]):
            raise ValueError(f"{group} Skin payload size differs from the donor layout")
        replacements[int(attribute_buffer["blob_index"])] = build_model_buffer(
            attribute_data,
            count,
            int(attribute_buffer["stride"]),
            tuple(attribute_buffer["flags"]),
            int(attribute_buffer["format"]),
        )
        replacements[int(skin_buffer["blob_index"])] = build_model_buffer(
            skin_data,
            count,
            int(skin_buffer["stride"]),
            tuple(skin_buffer["flags"]),
            int(skin_buffer["format"]),
        )

    for spec in specs:
        key = str(spec["key"])
        mesh = spec["mesh"]
        values_quantization = quantization[key]
        vertex_count = len(spec_vertices[key])
        replacements[int(mesh["blob_index"])] = patch_position_binding(
            outer.blobs[int(mesh["blob_index"])].data,
            position_offset=group_position_offsets[str(spec["group"])],
            start_index=index_starts[key],
            base_vertex=group_base_vertices[str(spec["group"])],
            index_count=len(local_indices[key]),
            radius=float(values_quantization["radius"]),
            vertex_indices=list(range(vertex_count)) * 2,
            uv_transform=list(values_quantization["uv_transform"]),
            scale=list(values_quantization["scale"]),
            translate=list(values_quantization["translate"]),
        )

    candidate_data = rebuild_with_blob_data(outer, replacements)
    candidate_bytes = bytearray(candidate_data)
    all_min = [math.inf, math.inf, math.inf]
    all_max = [-math.inf, -math.inf, -math.inf]
    for spec in specs:
        key = str(spec["key"])
        mesh = spec["mesh"]
        values_quantization = quantization[key]
        bounds_min = [float(value) for value in values_quantization["bounds_min"]]
        bounds_max = [float(value) for value in values_quantization["bounds_max"]]
        for axis in range(3):
            all_min[axis] = min(all_min[axis], bounds_min[axis])
            all_max[axis] = max(all_max[axis], bounds_max[axis])
        mesh_blob_index = int(mesh["blob_index"])
        bbox_entries = [entry for entry in outer.blobs[mesh_blob_index].metadata if entry.tag == "BBox"]
        if len(bbox_entries) != 1 or len(bbox_entries[0].value) != 24:
            raise ValueError(f"{key} Mesh has no unique BBox metadata")
        struct.pack_into("<6f", candidate_bytes, bbox_entries[0].value_offset, *(bounds_min + bounds_max))
    modl_blob_index = tag_indices["Modl"][0]
    modl_bbox_entries = [entry for entry in outer.blobs[modl_blob_index].metadata if entry.tag == "BBox"]
    if len(modl_bbox_entries) != 1 or len(modl_bbox_entries[0].value) != 24:
        raise ValueError("Driver_Alice_F Modl has no unique BBox metadata")
    struct.pack_into("<6f", candidate_bytes, modl_bbox_entries[0].value_offset, *(all_min + all_max))
    candidate_data = bytes(candidate_bytes)
    output.write_bytes(candidate_data)

    candidate_report = inspector.inspect(output)
    if candidate_report["parsed"]["errors"]:
        raise ValueError(f"Candidate parser errors: {candidate_report['parsed']['errors']}")
    candidate_outer = parse_outer_bundle(candidate_data)
    preserved_indices = sorted(tag_indices["Skel"] + tag_indices["MatI"] + tag_indices["VLay"] + tag_indices["Modl"])
    preservation = {
        str(index): {
            "tag": outer.blobs[index].tag,
            "preserved": outer.blobs[index].data == candidate_outer.blobs[index].data,
            "sha256": sha256_bytes(candidate_outer.blobs[index].data),
        }
        for index in preserved_indices
    }
    if not all(item["preserved"] for item in preservation.values()):
        raise ValueError("A required Driver_Alice_F Skel/MatI/VLay/Modl blob changed")

    candidate_mesh_by_blob = {int(mesh["blob_index"]): mesh for mesh in candidate_report["parsed"]["meshes"]}
    for spec in specs:
        key = str(spec["key"])
        mesh = candidate_mesh_by_blob[int(spec["mesh"]["blob_index"])]
        vertex_count = len(spec_vertices[key])
        if int(mesh["extended_unknown_u32"][1]) != vertex_count:
            raise ValueError(f"{key} Mesh vertex count is inconsistent")
        if int(mesh["extended_array_count"]) != vertex_count * 2:
            raise ValueError(f"{key} Mesh extended array is inconsistent")
        if int(mesh["base_vertex"]) != group_base_vertices[str(spec["group"])]:
            raise ValueError(f"{key} Mesh base vertex is inconsistent")
        if int(position_binding(mesh)["offset"]) != group_position_offsets[str(spec["group"])]:
            raise ValueError(f"{key} Position binding offset is inconsistent")
        if int(mesh["start_index"]) != index_starts[key] or int(mesh["index_count"]) != len(local_indices[key]):
            raise ValueError(f"{key} Mesh index range is inconsistent")

    candidate_vertex_buffers = {int(item["id"]): item for item in candidate_report["parsed"]["vertex_buffers"]}
    candidate_skin_buffers = {int(item["id"]): item for item in candidate_report["parsed"]["skin_buffers"]}
    expected_vertex_counts = {0: total_vertices, 1: group_counts["small"], 2: group_counts["main"], 3: group_counts["teeth"]}
    expected_skin_counts = {1: group_counts["small"], 0: group_counts["main"], 2: group_counts["teeth"]}
    if {key: int(value["count"]) for key, value in candidate_vertex_buffers.items()} != expected_vertex_counts:
        raise ValueError("Driver Body candidate vertex-buffer counts are inconsistent")
    if {key: int(value["count"]) for key, value in candidate_skin_buffers.items()} != expected_skin_counts:
        raise ValueError("Driver Body candidate Skin-buffer counts are inconsistent")
    if int(candidate_report["parsed"]["index_buffers"][0]["count"]) != len(raw_indices):
        raise ValueError("Driver Body candidate index count is inconsistent")

    report = {
        "schema_version": 1,
        "purpose": "Structural Driver_Alice_F Body-slot candidate containing the Si head and body.",
        "donor": {"path": str(donor), "sha256": sha256(donor), "bytes": len(donor_data)},
        "intermediate": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "logical_vertices": len(vertices),
            "logical_indices": len(source_indices),
        },
        "output": {"path": str(output), "sha256": sha256(output), "bytes": len(candidate_data)},
        "domains": {
            "position_vertices": total_vertices,
            "indices": len(raw_indices),
            "group_vertex_counts": group_counts,
            "group_position_offsets": group_position_offsets,
            "group_base_vertices": group_base_vertices,
        },
        "meshes": [
            {
                "key": spec["key"],
                "name": spec["mesh"]["name"],
                "material_id": int(spec["mesh"]["material_id"]),
                "logical_draw": int(spec["draw"]),
                "vertex_start": position_starts[str(spec["key"])],
                "vertex_count": len(spec_vertices[str(spec["key"])]),
                "start_index": index_starts[str(spec["key"])],
                "index_count": len(local_indices[str(spec["key"])]),
                "flipped": bool(spec["flip"]),
            }
            for spec in specs
        ],
        "quantization": quantization,
        "skin_policy": policies,
        "preserved_blobs": preservation,
        "policies": {
            "skeleton": "donor 246-bone Skel retained byte-exact",
            "materials": "donor six MatI blobs retained byte-exact",
            "layouts": "donor two VLay blobs and shared buffer-domain topology retained",
            "eyelashes": "source eyelash draw duplicated into front/back domains with reversed normals and winding",
            "eyes": "eye vertices rigidly divided between LeftEye and RightEye",
            "head_skinning": (
                "head, eyelashes, and teeth rigidly bound to Head for stable display animation"
                if args.rigid_head
                else "retargeted source facial weights retained"
            ),
            "lod": "Driver_Alice_F has one all-flags Mesh set; the first candidate retains those flags",
        },
        "validation_level": {
            "structural": True,
            "blender_visual": not args.rigid_head,
            "offline_game": False,
        },
        "license_guard": "Local technical validation only; do not redistribute this candidate.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_DRIVER_BODY_MODELBIN="
        + json.dumps(
            {
                "output": str(output),
                "sha256": report["output"]["sha256"],
                "bytes": len(candidate_data),
                "vertices": total_vertices,
                "indices": len(raw_indices),
                "meshes": len(specs),
                "group_counts": group_counts,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
