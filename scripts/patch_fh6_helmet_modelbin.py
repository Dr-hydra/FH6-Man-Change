#!/usr/bin/env python3
"""Patch a FH6 helmet donor with the component intermediate.

Helmet donors use two vertex layouts and two skin streams.  The retail donor
currently stores one influence per vertex, while the FBX-first intermediate
contains up to four influences.  The first writer keeps the donor ``Skel``,
``VLay`` and ``MatI`` blobs, but expands both Skin streams to the standard
four-influence stride and updates each Mesh descriptor's skinning-element
count.  The donor mesh order is material-interleaved, so the index buffer is
rebuilt in that order while the intermediate's dense vertex domains are
remapped accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle as parse_outer_bundle
from modelbin_bundle import rebuild_with_blob_data
from patch_fh6_garment_modelbin import (
    build_model_buffer,
    encode_geometry,
    read_intermediate,
    sha256,
    sha256_bytes,
    skeleton_world_matrices,
)


FACE_COMBINED_RENDER_PASSES = {
    0: 0x38,  # Hair
    1: 0x19,  # Eyes
    2: 0x19,  # Face
    3: 0x19,  # Neck
    4: 0x38,  # Eyelashes
    5: 0x3C,  # Hair shadow
    6: 0x19,  # Sclera
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("donor", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def patch_mesh_bindings(
    original: bytes,
    *,
    start_index: int,
    base_vertex: int,
    index_count: int,
    radius: float,
    vertex_indices: list[int],
    uv_transform: list[float],
    scale: list[float],
    translate: list[float],
    position_offset: int,
) -> bytes:
    """Patch Mesh 1.12 fields and the position-buffer byte offset.

    The opaque Mesh suffix contains the vertex-layout binding table.  Its
    first buffer is the shared Position stream; layout-0 meshes point at the
    second Position domain via this offset while retaining base-vertex bias.
    """

    from patch_fh6_garment_modelbin import patch_mesh_blob

    patched = bytearray(
        patch_mesh_blob(
            original,
            start_index=start_index,
            base_vertex=base_vertex,
            index_count=index_count,
            radius=radius,
            vertex_indices=vertex_indices,
            uv_transform=uv_transform,
            scale=scale,
            translate=translate,
        )
    )
    extended_count = struct.unpack_from("<I", patched, 58)[0]
    suffix_offset = 62 + extended_count * 4
    if suffix_offset + 8 > len(patched):
        raise ValueError("Mesh binding suffix is truncated")
    buffer_count = struct.unpack_from("<I", patched, suffix_offset + 4)[0]
    if buffer_count < 2:
        raise ValueError("Helmet Mesh must bind Position and Attribute buffers")
    position_buffer = None
    for buffer_index in range(buffer_count):
        buffer_offset = suffix_offset + 8 + buffer_index * 20
        if buffer_offset + 20 > len(patched):
            raise ValueError("Mesh buffer binding exceeds descriptor")
        buffer_id, input_slot = struct.unpack_from("<ii", patched, buffer_offset)
        if buffer_id == 0 and input_slot == 0:
            position_buffer = buffer_offset
            break
    if position_buffer is None:
        raise ValueError("Helmet Mesh has no Position input binding")
    # Binding fields are id/input_slot/stride/offset followed by a 4-byte
    # version-specific field; offset is therefore at +12.
    struct.pack_into("<i", patched, position_buffer + 12, position_offset)
    return bytes(patched)


def patch_mesh_skinning_elements(original: bytes, influences: int) -> bytes:
    """Patch Mesh 1.12's skinning-element count.

    The count is the byte immediately after the render-pass field in the
    Mesh 1.12 descriptor (offset 17 in the descriptor payload).  It must
    agree with the Skin stream stride: four R16G16_FLOAT pairs are 16 bytes.
    Keeping this explicit prevents a runtime from interpreting an expanded
    stream as one-influence data.
    """

    if influences < 1 or influences > 4:
        raise ValueError(f"Unsupported skinning influence count: {influences}")
    if len(original) <= 17:
        raise ValueError("Mesh descriptor is too small for skinning-elements field")
    patched = bytearray(original)
    patched[17] = influences
    return bytes(patched)


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
    source_vertices, source_indices = read_intermediate(manifest_path, manifest)
    donor_data = donor.read_bytes()
    outer = parse_outer_bundle(donor_data)
    parsed = inspector.parse_known_blobs(donor_data, inspector.parse_bundle(donor_data)[1])
    if parsed["errors"]:
        raise ValueError(f"Donor parser errors: {parsed['errors']}")

    tags = [blob.tag for blob in outer.blobs]
    tag_indices: dict[str, list[int]] = {}
    for index, tag in enumerate(tags):
        tag_indices.setdefault(tag, []).append(index)
    required_counts = {"Skel": 1, "IndB": 1, "VLay": 2, "VerB": 3, "Skin": 2, "Modl": 1}
    for tag, count in required_counts.items():
        if len(tag_indices.get(tag, [])) != count:
            raise ValueError(f"Expected {count} {tag} blobs, found {len(tag_indices.get(tag, []))}: {tags}")
    mesh_blob_indices = tag_indices.get("Mesh", [])
    meshes = parsed["meshes"]
    if len(meshes) not in (6, 7) or len(meshes) != len(mesh_blob_indices):
        raise ValueError("Helmet donor must contain six or seven Mesh 1.12 descriptors")
    if any(mesh["blob_version"] != "1.12" for mesh in meshes):
        raise ValueError("Helmet donor Mesh descriptors must be version 1.12")

    skeleton = parsed["skeleton"][0]
    donor_bone_order = [bone["name"] for bone in skeleton["bones"]]
    if donor_bone_order != manifest["skinning"]["bone_order"]:
        raise ValueError("Intermediate donor bone order differs from Helmet Skel order")

    source_draws = manifest["geometry"]["draws"]
    draw_by_material = {int(draw["material_id"]): draw for draw in source_draws}
    if len(draw_by_material) != len(source_draws):
        raise ValueError("Helmet intermediate contains duplicate material draw IDs")
    layout_by_material = {int(mesh["material_id"]): int(mesh["vertex_layout_id"]) for mesh in meshes}
    if len(layout_by_material) != len(meshes):
        raise ValueError("Helmet donor contains duplicate material Mesh IDs")
    if set(layout_by_material) != set(draw_by_material):
        raise ValueError("Helmet donor and intermediate material domains differ")
    for material_id, layout_id in layout_by_material.items():
        if layout_id not in (0, 1):
            raise ValueError(f"Unsupported Helmet vertex layout {layout_id} for material {material_id}")

    # Reorder source draw domains exactly as the donor Mesh blobs are ordered.
    # This keeps each Mesh's start_index contiguous while allowing a shared
    # Position stream with layout-1 vertices before layout-0 vertices.
    permutation: list[int] = []
    new_draws: list[dict] = []
    source_to_new: dict[int, int] = {}
    index_payload_values: list[int] = []
    layout_local_cursor = {0: 0, 1: 0}
    for mesh in meshes:
        material_id = int(mesh["material_id"])
        source_draw = draw_by_material[material_id]
        source_start = int(source_draw["vertex_start"])
        source_count = int(source_draw["vertex_count"])
        layout_id = layout_by_material[material_id]
        target_start = len(permutation)
        local_start = layout_local_cursor[layout_id]
        permutation.extend(range(source_start, source_start + source_count))
        for local_index in range(source_count):
            source_to_new[source_start + local_index] = target_start + local_index
        layout_local_cursor[layout_id] += source_count

        source_index_start = int(source_draw["start_index"])
        source_index_count = int(source_draw["index_count"])
        if source_index_count % 3:
            raise ValueError(f"Material {material_id} has a non-triangle-aligned index range")
        source_slice = source_indices[source_index_start : source_index_start + source_index_count]
        if len(source_slice) != source_index_count:
            raise ValueError(f"Material {material_id} index range exceeds intermediate buffer")
        for offset in range(0, source_index_count, 3):
            first, second, third = source_slice[offset : offset + 3]
            for vertex_index in (first, third, second):
                # Intermediate indices are already global within the dense
                # source vertex domain; the draw range only validates them.
                source_global = int(vertex_index)
                if source_global not in source_to_new:
                    raise ValueError(f"Material {material_id} index resolves outside its vertex domain")
                index_payload_values.append(source_to_new[source_global])
        new_start = len(index_payload_values) - source_index_count
        new_draws.append(
            {
                **source_draw,
                "start_index": new_start,
                "vertex_start": target_start,
                "vertex_count": source_count,
                "layout_id": layout_id,
                "layout_local_start": local_start,
                "layout_local_count": source_count,
            }
        )

    if len(permutation) != len(source_vertices):
        raise ValueError("Helmet draw domains do not cover the complete vertex buffer")
    if len(index_payload_values) != len(source_indices):
        raise ValueError("Helmet draw domains do not cover the complete index buffer")
    reordered_vertices = [source_vertices[index] for index in permutation]
    if max(index_payload_values, default=0) >= len(reordered_vertices):
        raise ValueError("Reordered Helmet index exceeds Position domain")

    world_matrices = skeleton_world_matrices(skeleton["bones"])
    anchor_indices = sorted({int(mesh["bone_index"]) for mesh in meshes})
    anchor_matrix = world_matrices[anchor_indices[0]]
    for anchor in anchor_indices[1:]:
        if any(abs(left - right) > 1e-6 for left, right in zip(anchor_matrix, world_matrices[anchor])):
            raise ValueError("Helmet donor mesh anchors differ")
    # encode_geometry indexes its per-draw quantization table by material ID;
    # keep that table material-ordered even though Mesh/index domains follow
    # the donor's interleaved blob order above.
    encoded, quantization = encode_geometry(
        reordered_vertices,
        anchor_matrix,
        sorted(new_draws, key=lambda item: int(item["material_id"])),
    )

    layout1_count = layout_local_cursor[1]
    layout0_count = layout_local_cursor[0]
    position_payload = encoded["position"]
    attribute1_payload = bytearray()
    attribute0_payload = bytearray()
    skin1_payload = bytearray()
    skin0_payload = bytearray()
    for index, layout_id in enumerate([1] * layout1_count + [0] * layout0_count):
        attribute_record = encoded["attribute"][index * 40 : (index + 1) * 40]
        skin_record = encoded["skin"][index * 16 : (index + 1) * 16]
        if layout_id == 1:
            # Layout 1 contains NORMAL.yz, TEXCOORD0/2 and TANGENT0/2.
            attribute1_payload.extend(attribute_record[0:4])
            attribute1_payload.extend(attribute_record[4:8])
            attribute1_payload.extend(attribute_record[12:16])
            attribute1_payload.extend(attribute_record[24:28])
            attribute1_payload.extend(attribute_record[32:36])
            skin1_payload.extend(skin_record)
        else:
            attribute0_payload.extend(attribute_record)
            skin0_payload.extend(skin_record)

    if len(attribute1_payload) != layout1_count * 20 or len(attribute0_payload) != layout0_count * 40:
        raise ValueError("Helmet attribute layout payload sizes are inconsistent")
    if len(skin1_payload) != layout1_count * 16 or len(skin0_payload) != layout0_count * 16:
        raise ValueError("Helmet skin layout payload sizes are inconsistent")

    replacements: dict[int, bytes] = {}
    replacement_by_id = {int(item["id"]): item for item in parsed["vertex_buffers"]}
    skin_by_id = {int(item["id"]): item for item in parsed["skin_buffers"]}
    replacements[parsed["index_buffers"][0]["blob_index"]] = build_model_buffer(
        struct.pack(f"<{len(index_payload_values)}H", *index_payload_values),
        len(index_payload_values),
        2,
        tuple(parsed["index_buffers"][0]["flags"]),
        int(parsed["index_buffers"][0]["format"]),
    )
    replacements[replacement_by_id[0]["blob_index"]] = build_model_buffer(
        position_payload,
        len(reordered_vertices),
        8,
        tuple(replacement_by_id[0]["flags"]),
        int(replacement_by_id[0]["format"]),
    )
    replacements[replacement_by_id[1]["blob_index"]] = build_model_buffer(
        bytes(attribute1_payload), layout1_count, 20, tuple(replacement_by_id[1]["flags"]), int(replacement_by_id[1]["format"])
    )
    replacements[replacement_by_id[2]["blob_index"]] = build_model_buffer(
        bytes(attribute0_payload), layout0_count, 40, tuple(replacement_by_id[2]["flags"]), int(replacement_by_id[2]["format"])
    )
    # The source intermediate is a normal four-influence export.  The retail
    # helmet donor's streams are one-influence, but retaining only pair 0
    # silently drops facial/eyelid/hair weights and produces broken animation.
    # Expand both streams and advertise the count on every Mesh descriptor.
    skin_influences = 4
    for skin_payload, layout_count, skin_id in (
        (skin1_payload, layout1_count, 1),
        (skin0_payload, layout0_count, 0),
    ):
        expected_bytes = layout_count * skin_influences * 4
        if len(skin_payload) != expected_bytes:
            raise ValueError(
                f"Helmet Skin {skin_id} payload has {len(skin_payload)} bytes; "
                f"expected {expected_bytes} for {layout_count} vertices"
            )
        replacements[skin_by_id[skin_id]["blob_index"]] = build_model_buffer(
            bytes(skin_payload),
            layout_count,
            skin_influences * 4,
            tuple(skin_by_id[skin_id]["flags"]),
            int(skin_by_id[skin_id]["format"]),
        )

    quantization_by_material = {int(item["material_id"]): item for item in quantization["draws"]}
    mesh_replacements: dict[int, bytes] = {}
    for mesh, blob_index, draw in zip(meshes, mesh_blob_indices, new_draws):
        material_id = int(mesh["material_id"])
        draw_quantization = quantization_by_material[material_id]
        layout_id = int(draw["layout_id"])
        local_start = int(draw["layout_local_start"])
        local_count = int(draw["layout_local_count"])
        patched_mesh = patch_mesh_bindings(
            outer.blobs[blob_index].data,
            start_index=int(draw["start_index"]),
            base_vertex=0 if layout_id == 1 else -layout1_count,
            index_count=int(draw["index_count"]),
            radius=float(draw_quantization["radius"]),
            vertex_indices=list(range(local_start, local_start + local_count)) * 2,
            uv_transform=list(draw_quantization["uv_transform"]),
            scale=list(draw_quantization["scale"]),
            translate=list(draw_quantization["translate"]),
            position_offset=0 if layout_id == 1 else layout1_count * 8,
        )
        if manifest["geometry"].get("draw_policy") in {
            "helmet6_face_combined",
            "head6_display",
            "head7_display",
        }:
            patched_mesh = bytearray(patched_mesh)
            struct.pack_into("<H", patched_mesh, 14, FACE_COMBINED_RENDER_PASSES[material_id])
            patched_mesh = bytes(patched_mesh)
        mesh_replacements[blob_index] = patch_mesh_skinning_elements(patched_mesh, skin_influences)
    replacements.update(mesh_replacements)
    candidate_data = rebuild_with_blob_data(outer, replacements)

    # Mesh/Modl BBox metadata is outside opaque Mesh data.
    candidate_bytes = bytearray(candidate_data)
    bounds_by_material = {
        int(item["material_id"]): ([float(v) for v in item["bounds_min"]], [float(v) for v in item["bounds_max"]])
        for item in quantization["draws"]
    }
    all_min = [math.inf, math.inf, math.inf]
    all_max = [-math.inf, -math.inf, -math.inf]
    for mesh, blob_index in zip(meshes, mesh_blob_indices):
        bounds_min, bounds_max = bounds_by_material[int(mesh["material_id"])]
        for axis in range(3):
            all_min[axis] = min(all_min[axis], bounds_min[axis])
            all_max[axis] = max(all_max[axis], bounds_max[axis])
        bbox_entries = [entry for entry in outer.blobs[blob_index].metadata if entry.tag == "BBox"]
        if len(bbox_entries) != 1 or len(bbox_entries[0].value) != 24:
            raise ValueError(f"Helmet Mesh {mesh['material_id']} does not have one BBox metadata entry")
        struct.pack_into("<6f", candidate_bytes, bbox_entries[0].value_offset, *(bounds_min + bounds_max))
    modl_blob_index = tag_indices["Modl"][0]
    modl_bbox_entries = [entry for entry in outer.blobs[modl_blob_index].metadata if entry.tag == "BBox"]
    if len(modl_bbox_entries) != 1 or len(modl_bbox_entries[0].value) != 24:
        raise ValueError("Helmet Modl does not have one BBox metadata entry")
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
        raise ValueError("A required Helmet Skel/MatI/VLay/Modl blob changed")

    report = {
        "schema_version": 1,
        "purpose": "Structural FH6 Helmet head component candidate with donor layouts/material slots retained; not yet game validated.",
        "donor": {"path": str(donor), "sha256": sha256(donor), "bytes": len(donor_data)},
        "intermediate": {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "vertices": len(source_vertices), "indices": len(index_payload_values), "triangles": len(index_payload_values) // 3},
        "candidate": {
            "path": str(output), "sha256": sha256(output), "bytes": len(candidate_data), "bundle": candidate_report["header"], "parse_errors": candidate_report["parsed"]["errors"],
            "meshes": [{"material_id": m["material_id"], "lod_flags": m["lod_flags"], "skinning_elements": m["skinning_elements"], "vertex_layout_id": m["vertex_layout_id"], "start_index": m["start_index"], "index_count": m["index_count"], "base_vertex": m["base_vertex"], "extended_array_count": m["extended_array_count"], "vertex_buffers": m["vertex_buffers"], "skinning_data_buffer_id": m["skinning_data_buffer_id"]} for m in candidate_report["parsed"]["meshes"]],
            "index_buffer": candidate_report["parsed"]["index_buffers"][0], "vertex_buffers": candidate_report["parsed"]["vertex_buffers"], "skin_buffers": candidate_report["parsed"]["skin_buffers"],
        },
        "quantization": quantization,
        "layout_domains": {"layout1": {"materials": [m for m, v in layout_by_material.items() if v == 1], "vertices": layout1_count, "attribute_stride": 20, "skin_stride": 16}, "layout0": {"materials": [m for m, v in layout_by_material.items() if v == 0], "vertices": layout0_count, "attribute_stride": 40, "skin_stride": 16}},
        "preserved_blobs": preservation,
        "policies": {"skeleton": "donor Skel retained byte-exact", "materials": "donor MatI retained byte-exact", "layouts": "donor VLay retained byte-exact; per-mesh layout IDs and buffer bindings retained", "mesh_order": "index and vertex domains rebuilt in donor Mesh blob order", "skinning": "source four-influence weights retained; both donor Skin streams expanded to stride 16 and Mesh skinning_elements=4", "render_pass": "head display face/eyes/sclera use 0x19, hair/eyelashes use 0x38, and hair shadow uses 0x3C", "lod": "all donor LOD flags retained; first candidate shares the full-resolution domain"},
        "validation_level": {"structural": True, "blender_visual": False, "offline_game": False},
        "license_guard": "Local technical validation only; do not redistribute this candidate.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_HELMET_MODELBIN=" + json.dumps({"output": str(output), "report": str(report_path), "bytes": len(candidate_data), "vertices": len(reordered_vertices), "layout1_vertices": layout1_count, "layout0_vertices": layout0_count, "indices": len(index_payload_values), "parse_errors": len(candidate_report['parsed']['errors'])}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
