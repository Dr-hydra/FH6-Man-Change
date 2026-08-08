#!/usr/bin/env python3
"""Patch the four-draw FH6 face donor with a validated head intermediate."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle as parse_outer_bundle
from modelbin_bundle import rebuild_with_blob_data
from patch_fh6_garment_modelbin import (
    build_model_buffer,
    describe_attribute_layout,
    encode_geometry,
    patch_mesh_blob,
    read_intermediate,
    sha256,
    sha256_bytes,
    skeleton_world_matrices,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("donor", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def position_binding(mesh: dict) -> dict:
    matches = [
        item
        for item in mesh["vertex_buffers"]
        if int(item["id"]) == 0 and int(item["input_slot"]) == 0
    ]
    if len(matches) != 1:
        raise ValueError(f"Mesh material {mesh['material_id']} has no unique Position binding")
    return matches[0]


def attribute_binding(mesh: dict) -> dict:
    matches = [item for item in mesh["vertex_buffers"] if int(item["input_slot"]) == 1]
    if len(matches) != 1:
        raise ValueError(f"Mesh material {mesh['material_id']} has no unique Attribute binding")
    return matches[0]


def patch_position_binding(original: bytes, position_offset: int, **mesh_fields: object) -> bytes:
    patched = bytearray(patch_mesh_blob(original, **mesh_fields))
    extended_count = struct.unpack_from("<I", patched, 58)[0]
    suffix_offset = 62 + extended_count * 4
    if suffix_offset + 8 > len(patched):
        raise ValueError("Face Mesh binding suffix is truncated")
    buffer_count = struct.unpack_from("<I", patched, suffix_offset + 4)[0]
    found = False
    for buffer_index in range(buffer_count):
        buffer_offset = suffix_offset + 8 + buffer_index * 20
        if buffer_offset + 20 > len(patched):
            raise ValueError("Face Mesh buffer binding exceeds descriptor")
        buffer_id, input_slot = struct.unpack_from("<ii", patched, buffer_offset)
        if buffer_id == 0 and input_slot == 0:
            struct.pack_into("<i", patched, buffer_offset + 12, position_offset)
            found = True
            break
    if not found:
        raise ValueError("Face Mesh has no Position input binding")
    return bytes(patched)


def skin_payload(
    vertices: list[dict],
    stride: int,
    material_id: int,
    bone_order: list[str],
) -> tuple[bytes, dict[str, object]]:
    if stride <= 0 or stride % 4:
        raise ValueError(f"Unsupported face Skin stride {stride}")
    influence_limit = stride // 4
    payload = bytearray()
    counts: dict[int, int] = {}
    rigid_eye_counts = {"LeftEye": 0, "RightEye": 0}
    for vertex in vertices:
        if material_id == 3:
            if influence_limit != 1:
                raise ValueError("Face eye draw must use a one-influence Skin stream")
            target_name = "LeftEye" if float(vertex["position"][0]) < 0.0 else "RightEye"
            assignments = [(bone_order.index(target_name), 1.0)]
            rigid_eye_counts[target_name] += 1
        else:
            combined: dict[int, float] = {}
            for bone_index, weight in zip(vertex["bones"], vertex["weights"]):
                if float(weight) > 0.0:
                    combined[int(bone_index)] = combined.get(int(bone_index), 0.0) + float(weight)
            assignments = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:influence_limit]
            total = sum(weight for _, weight in assignments)
            if total <= 0.0:
                raise ValueError("Face Skin reduction produced a zero-weight vertex")
            assignments = [(bone_index, weight / total) for bone_index, weight in assignments]
        counts[len(assignments)] = counts.get(len(assignments), 0) + 1
        padded = assignments + [(0, 0.0)] * (influence_limit - len(assignments))
        for bone_index, weight in padded:
            payload.extend(struct.pack("<ee", float(weight), float(bone_index)))
    return bytes(payload), {
        "stride": stride,
        "influence_limit": influence_limit,
        "influence_histogram": {str(key): value for key, value in sorted(counts.items())},
        "rigid_eye_vertices": rigid_eye_counts if material_id == 3 else None,
    }


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
    if not isinstance(draws, list) or len(draws) != 4:
        raise ValueError("Face intermediate must contain four draw ranges")
    draw_by_material = {int(draw["material_id"]): draw for draw in draws}
    if set(draw_by_material) != {0, 1, 2, 3}:
        raise ValueError("Face intermediate draw IDs must be 0..3")

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
        "IndB": 1,
        "VLay": 3,
        "VerB": 5,
        "Skin": 4,
        "Mesh": 4,
        "MatI": 4,
        "Modl": 1,
    }
    for tag, count in required_counts.items():
        if len(tag_indices.get(tag, [])) != count:
            raise ValueError(f"Expected {count} {tag} blobs, found {len(tag_indices.get(tag, []))}: {tags}")

    meshes = parsed["meshes"]
    mesh_by_material = {int(mesh["material_id"]): mesh for mesh in meshes}
    if set(mesh_by_material) != set(draw_by_material):
        raise ValueError("Face donor material IDs differ from the intermediate")
    if any(mesh["blob_version"] != "1.12" for mesh in meshes):
        raise ValueError("Face donor Mesh descriptors must be version 1.12")
    layouts = {int(layout["id"]): layout for layout in parsed["vertex_layouts"]}
    vertex_buffers = {int(item["id"]): item for item in parsed["vertex_buffers"]}
    skin_buffers = {int(item["id"]): item for item in parsed["skin_buffers"]}
    if set(vertex_buffers) != {0, 1, 2, 3, 4} or set(skin_buffers) != {0, 1, 2, 3}:
        raise ValueError("Face donor buffer IDs differ from the verified contract")

    skeleton = parsed["skeleton"][0]
    bone_order = [bone["name"] for bone in skeleton["bones"]]
    if bone_order != manifest["skinning"]["bone_order"]:
        raise ValueError("Intermediate donor bone order differs from Face Skel order")
    world_matrices = skeleton_world_matrices(skeleton["bones"])
    anchor_indices = sorted({int(mesh["bone_index"]) for mesh in meshes})
    anchor_matrix = world_matrices[anchor_indices[0]]
    for anchor in anchor_indices[1:]:
        if any(abs(left - right) > 1e-6 for left, right in zip(anchor_matrix, world_matrices[anchor])):
            raise ValueError("Face donor mesh anchors differ")

    # Preserve the donor's shared Position-buffer domain order, which is
    # encoded by each Mesh binding offset rather than Mesh blob order.
    position_material_order = [
        int(mesh["material_id"])
        for mesh in sorted(meshes, key=lambda item: int(position_binding(item)["offset"]))
    ]
    expected_position_offset = 0
    for material_id in position_material_order:
        binding_offset = int(position_binding(mesh_by_material[material_id])["offset"])
        if binding_offset != expected_position_offset:
            raise ValueError("Face donor Position domains are not tightly packed")
        donor_attribute = vertex_buffers[int(attribute_binding(mesh_by_material[material_id])["id"])]
        expected_position_offset += int(donor_attribute["count"]) * 8

    encoded_by_material: dict[int, dict[str, bytes]] = {}
    quantization_by_material: dict[int, dict[str, object]] = {}
    skin_policy: dict[int, dict[str, object]] = {}
    local_indices_by_material: dict[int, list[int]] = {}
    for material_id in range(4):
        draw = draw_by_material[material_id]
        vertex_start = int(draw["vertex_start"])
        vertex_count = int(draw["vertex_count"])
        draw_vertices = vertices[vertex_start : vertex_start + vertex_count]
        if len(draw_vertices) != vertex_count or vertex_count <= 0:
            raise ValueError(f"Face draw {material_id} has an invalid vertex range")
        mesh = mesh_by_material[material_id]
        layout = describe_attribute_layout(layouts[int(mesh["vertex_layout_id"])])
        attribute = attribute_binding(mesh)
        if int(attribute["stride"]) != int(layout["stride"]):
            raise ValueError(f"Face material {material_id} Mesh/VLay attribute strides differ")
        encoded, quantization = encode_geometry(
            draw_vertices,
            anchor_matrix,
            [{"material_id": 0, "vertex_start": 0, "vertex_count": vertex_count}],
            layout,
        )
        encoded_by_material[material_id] = encoded
        draw_quantization = quantization["draws"][0]
        draw_quantization["material_id"] = material_id
        quantization_by_material[material_id] = draw_quantization

        index_start = int(draw["start_index"])
        index_count = int(draw["index_count"])
        draw_indices = source_indices[index_start : index_start + index_count]
        local_indices = [int(index) - vertex_start for index in draw_indices]
        if len(local_indices) != index_count or any(index < 0 or index >= vertex_count for index in local_indices):
            raise ValueError(f"Face draw {material_id} index resolves outside its vertex range")
        local_indices_by_material[material_id] = local_indices

        skin = skin_buffers[int(mesh["skinning_data_buffer_id"])]
        payload, policy = skin_payload(draw_vertices, int(skin["stride"]), material_id, bone_order)
        encoded_by_material[material_id]["skin"] = payload
        skin_policy[material_id] = policy

    position_starts: dict[int, int] = {}
    position_payload = bytearray()
    for material_id in position_material_order:
        position_starts[material_id] = len(position_payload) // 8
        position_payload.extend(encoded_by_material[material_id]["position"])

    index_material_order = [
        int(mesh["material_id"])
        for mesh in sorted(meshes, key=lambda item: int(item["start_index"]))
    ]
    raw_indices: list[int] = []
    index_starts: dict[int, int] = {}
    for material_id in index_material_order:
        index_starts[material_id] = len(raw_indices)
        position_start = position_starts[material_id]
        local_indices = local_indices_by_material[material_id]
        for offset in range(0, len(local_indices), 3):
            first, second, third = local_indices[offset : offset + 3]
            raw_indices.extend((first + position_start, third + position_start, second + position_start))
    if max(raw_indices, default=0) >= len(vertices):
        raise ValueError("Face raw index exceeds the shared Position domain")

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
        len(vertices),
        int(position_buffer["stride"]),
        tuple(position_buffer["flags"]),
        int(position_buffer["format"]),
    )

    for material_id, mesh in mesh_by_material.items():
        draw = draw_by_material[material_id]
        vertex_count = int(draw["vertex_count"])
        attribute = vertex_buffers[int(attribute_binding(mesh)["id"])]
        attribute_payload = encoded_by_material[material_id]["attribute"]
        if len(attribute_payload) != vertex_count * int(attribute["stride"]):
            raise ValueError(f"Face material {material_id} attribute payload size differs from donor layout")
        replacements[int(attribute["blob_index"])] = build_model_buffer(
            attribute_payload,
            vertex_count,
            int(attribute["stride"]),
            tuple(attribute["flags"]),
            int(attribute["format"]),
        )
        skin = skin_buffers[int(mesh["skinning_data_buffer_id"])]
        skin_data = encoded_by_material[material_id]["skin"]
        if len(skin_data) != vertex_count * int(skin["stride"]):
            raise ValueError(f"Face material {material_id} Skin payload size differs from donor stream")
        replacements[int(skin["blob_index"])] = build_model_buffer(
            skin_data,
            vertex_count,
            int(skin["stride"]),
            tuple(skin["flags"]),
            int(skin["format"]),
        )

        quantization = quantization_by_material[material_id]
        mesh_blob_index = int(mesh["blob_index"])
        replacements[mesh_blob_index] = patch_position_binding(
            outer.blobs[mesh_blob_index].data,
            position_offset=position_starts[material_id] * 8,
            start_index=index_starts[material_id],
            base_vertex=-position_starts[material_id],
            index_count=int(draw["index_count"]),
            radius=float(quantization["radius"]),
            vertex_indices=list(range(vertex_count)) * 2,
            uv_transform=list(quantization["uv_transform"]),
            scale=list(quantization["scale"]),
            translate=list(quantization["translate"]),
        )

    candidate_data = rebuild_with_blob_data(outer, replacements)
    candidate_bytes = bytearray(candidate_data)
    all_min = [math.inf, math.inf, math.inf]
    all_max = [-math.inf, -math.inf, -math.inf]
    for material_id, mesh in mesh_by_material.items():
        quantization = quantization_by_material[material_id]
        bounds_min = [float(value) for value in quantization["bounds_min"]]
        bounds_max = [float(value) for value in quantization["bounds_max"]]
        for axis in range(3):
            all_min[axis] = min(all_min[axis], bounds_min[axis])
            all_max[axis] = max(all_max[axis], bounds_max[axis])
        mesh_blob_index = int(mesh["blob_index"])
        bbox_entries = [entry for entry in outer.blobs[mesh_blob_index].metadata if entry.tag == "BBox"]
        if len(bbox_entries) != 1 or len(bbox_entries[0].value) != 24:
            raise ValueError(f"Face Mesh material {material_id} has no unique BBox metadata")
        struct.pack_into("<6f", candidate_bytes, bbox_entries[0].value_offset, *(bounds_min + bounds_max))
    modl_blob_index = tag_indices["Modl"][0]
    modl_bbox_entries = [entry for entry in outer.blobs[modl_blob_index].metadata if entry.tag == "BBox"]
    if len(modl_bbox_entries) != 1 or len(modl_bbox_entries[0].value) != 24:
        raise ValueError("Face Modl has no unique BBox metadata")
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
        raise ValueError("A required Face Skel/MatI/VLay/Modl blob changed")

    candidate_meshes = candidate_report["parsed"]["meshes"]
    candidate_index = candidate_report["parsed"]["index_buffers"][0]
    if int(candidate_index["count"]) != len(raw_indices):
        raise ValueError("Face candidate index count changed during rebuild")
    for mesh in candidate_meshes:
        material_id = int(mesh["material_id"])
        draw = draw_by_material[material_id]
        expected_vertex_count = int(draw["vertex_count"])
        if int(mesh["extended_unknown_u32"][1]) != expected_vertex_count:
            raise ValueError(f"Face material {material_id} Mesh vertex count is inconsistent")
        if int(mesh["extended_array_count"]) != expected_vertex_count * 2:
            raise ValueError(f"Face material {material_id} Mesh extended array is inconsistent")
        if int(mesh["base_vertex"]) != -position_starts[material_id]:
            raise ValueError(f"Face material {material_id} base vertex is inconsistent")
        if int(position_binding(mesh)["offset"]) != position_starts[material_id] * 8:
            raise ValueError(f"Face material {material_id} Position offset is inconsistent")

    report = {
        "schema_version": 1,
        "purpose": "Structural FH6 face candidate retaining donor layouts, materials, and facial skeleton.",
        "donor": {"path": str(donor), "sha256": sha256(donor), "bytes": len(donor_data)},
        "intermediate": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "vertices": len(vertices),
            "indices": len(raw_indices),
            "triangles": len(raw_indices) // 3,
        },
        "candidate": {
            "path": str(output),
            "sha256": sha256(output),
            "bytes": len(candidate_data),
            "bundle": candidate_report["header"],
            "parse_errors": candidate_report["parsed"]["errors"],
            "meshes": candidate_meshes,
            "index_buffer": candidate_index,
            "vertex_buffers": candidate_report["parsed"]["vertex_buffers"],
            "skin_buffers": candidate_report["parsed"]["skin_buffers"],
        },
        "position_material_order": position_material_order,
        "index_material_order": index_material_order,
        "quantization": [quantization_by_material[index] for index in range(4)],
        "skin_policy": {str(key): value for key, value in sorted(skin_policy.items())},
        "preserved_blobs": preservation,
        "policies": {
            "skeleton": "donor 330-bone Skel retained byte-exact",
            "materials": "donor four MatI blobs retained byte-exact",
            "layouts": "donor three VLay blobs and per-draw buffer bindings retained",
            "eyes": "eye draw rigidly divided between LeftEye and RightEye to satisfy donor one-influence Skin",
            "face": "retargeted source facial weights retained up to the donor Skin stride",
            "lod": "donor LOD flags retained; first candidate uses the full-resolution exported draw domains",
        },
        "validation_level": {"structural": True, "blender_visual": True, "offline_game": False},
        "license_guard": "Local technical validation only; do not redistribute this candidate.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_FACE_MODELBIN=" + json.dumps({
        "output": str(output),
        "report": str(report_path),
        "bytes": len(candidate_data),
        "vertices": len(vertices),
        "indices": len(raw_indices),
        "parse_errors": len(candidate_report["parsed"]["errors"]),
        "attribute_strides": [item["stride"] for item in candidate_report["parsed"]["vertex_buffers"]],
        "skin_strides": [item["stride"] for item in candidate_report["parsed"]["skin_buffers"]],
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
