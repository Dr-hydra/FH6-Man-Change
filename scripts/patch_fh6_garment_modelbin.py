#!/usr/bin/env python3
"""Patch an FH6 female shirt donor with a validated garment intermediate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import parse_bundle as parse_outer_bundle
from modelbin_bundle import rebuild_with_blob_data


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("donor", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--static-bone-name")
    parser.add_argument("--collapse-fingers-to-hands", action="store_true")
    parser.add_argument("--collapse-toes-to-feet", action="store_true")
    parser.add_argument(
        "--duplicate-draws-for-lod-groups",
        action="store_true",
        help=(
            "Reuse one intermediate vertex/Skin domain for every donor LOD group, "
            "while emitting disjoint duplicate index partitions for each group."
        ),
    )
    parser.add_argument(
        "--rigid-extremities",
        action="store_true",
        help="Rigidly bind outer hand and foot regions for stable display animation.",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def vector_length(vector: tuple[float, float, float] | list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def normalize(vector: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    length = vector_length(vector)
    if length <= 1e-20:
        raise ValueError("Cannot normalize zero-length vector")
    return tuple(component / length for component in vector)


def matrix_multiply(left: list[float], right: list[float]) -> list[float]:
    return [
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    ]


def matrix_inverse(matrix: list[float]) -> list[float]:
    rows = [
        [float(matrix[row * 4 + column]) for column in range(4)]
        + [1.0 if row == column else 0.0 for column in range(4)]
        for row in range(4)
    ]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= 1e-15:
            raise ValueError("Singular skeleton matrix")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(4):
            if row == column:
                continue
            factor = rows[row][column]
            rows[row] = [left - factor * right for left, right in zip(rows[row], rows[column])]
    return [rows[row][column + 4] for row in range(4) for column in range(4)]


def transform_row(vector: tuple[float, float, float], matrix: list[float], translate: bool) -> tuple[float, float, float]:
    return tuple(
        sum(vector[input_axis] * matrix[input_axis * 4 + output_axis] for input_axis in range(3))
        + (matrix[12 + output_axis] if translate else 0.0)
        for output_axis in range(3)
    )


def skeleton_world_matrices(bones: list[dict]) -> list[list[float]]:
    matrices: list[list[float]] = []
    for index, bone in enumerate(bones):
        local = [float(value) for value in bone["matrix"]]
        parent = int(bone["parent"])
        if parent >= index:
            raise ValueError(f"bone {index} has non-previous parent {parent}")
        matrices.append(matrix_multiply(local, matrices[parent]) if parent >= 0 else local)
    return matrices


def blender_to_forza(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    # Inverse of import_fh6_modelbin_baseline.axis_convert().
    return (-vector[0], vector[2], -vector[1])


def forza_to_blender(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-vector[0], -vector[2], vector[1])


def snorm16_encode(value: float) -> int:
    return int(round(clamp(value, -1.0, 1.0) * 32767.0))


def snorm16_decode(value: int) -> float:
    return max(-1.0, value / 32767.0)


def unorm16_encode(value: float) -> int:
    return int(round(clamp(value, 0.0, 1.0) * 65535.0))


def pack_tangent(vector: tuple[float, float, float], handedness: float) -> int:
    components = [int(round((clamp(value, -1.0, 1.0) * 0.5 + 0.5) * 1023.0)) for value in vector]
    w = 3 if handedness > 0.0 else 0
    return components[0] | (components[1] << 10) | (components[2] << 20) | (w << 30)


def unpack_tangent(value: int) -> tuple[tuple[float, float, float], float]:
    vector = tuple((((value >> shift) & 0x3FF) / 1023.0) * 2.0 - 1.0 for shift in (0, 10, 20))
    handedness = 1.0 if ((value >> 30) & 0x3) >= 2 else -1.0
    return normalize(vector), handedness


def angle_degrees(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    dot = clamp(sum(a * b for a, b in zip(normalize(left), normalize(right))), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def read_intermediate(manifest_path: Path, manifest: dict) -> tuple[list[dict], list[int]]:
    vertex_info = manifest["files"]["vertices"]
    bone_info = manifest["files"]["bone_indices"]
    index_info = manifest["files"]["indices"]
    def resolve_input(raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else manifest_path.parent / path

    vertex_path = resolve_input(vertex_info["path"])
    bone_path = resolve_input(bone_info["path"])
    index_path = resolve_input(index_info["path"])
    for path, info in ((vertex_path, vertex_info), (bone_path, bone_info), (index_path, index_info)):
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Intermediate hash mismatch: {path}")

    vertex_data = vertex_path.read_bytes()
    bone_data = bone_path.read_bytes()
    index_data = index_path.read_bytes()
    vertex_count = int(manifest["geometry"]["export_vertices"])
    index_count = int(manifest["geometry"]["indices"])
    if len(vertex_data) != vertex_count * 64:
        raise ValueError("Unexpected intermediate vertex payload size")
    if len(bone_data) != vertex_count * 8:
        raise ValueError("Unexpected intermediate bone-index payload size")
    if len(index_data) != index_count * 2:
        raise ValueError("Unexpected intermediate index payload size")

    vertices: list[dict] = []
    for index in range(vertex_count):
        values = struct.unpack_from("<16f", vertex_data, index * 64)
        bones = struct.unpack_from("<4H", bone_data, index * 8)
        vertices.append(
            {
                "position": values[0:3],
                "normal": values[3:6],
                "tangent": values[6:10],
                "uv": values[10:12],
                "weights": values[12:16],
                "bones": bones,
            }
        )
    indices = list(struct.unpack(f"<{index_count}H", index_data))
    if max(indices, default=0) >= vertex_count:
        raise ValueError("Intermediate index resolves beyond vertex domain")
    return vertices, indices


def collapse_influences(vertices: list[dict], remap: dict[int, int]) -> tuple[int, int]:
    changed_vertices = 0
    merged_influences = 0
    for vertex in vertices:
        combined: dict[int, float] = {}
        changed = False
        active_count = 0
        for bone_index, weight in zip(vertex["bones"], vertex["weights"]):
            if weight <= 0.0:
                continue
            active_count += 1
            target = remap.get(int(bone_index), int(bone_index))
            changed |= target != bone_index
            combined[target] = combined.get(target, 0.0) + float(weight)
        if not changed:
            continue
        changed_vertices += 1
        merged_influences += active_count - len(combined)
        assignments = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        if len(assignments) > 4:
            raise ValueError("Bone collapse unexpectedly increased the influence count")
        total = sum(weight for _, weight in assignments)
        if total <= 0.0:
            raise ValueError("Bone collapse produced a zero-weight vertex")
        normalized = [(index, weight / total) for index, weight in assignments]
        normalized.extend([(0, 0.0)] * (4 - len(normalized)))
        vertex["bones"] = tuple(index for index, _ in normalized)
        vertex["weights"] = tuple(weight for _, weight in normalized)

    return changed_vertices, merged_influences


def collapse_fingers_to_hands(vertices: list[dict], bone_order: list[str]) -> dict:
    finger_tokens = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    remap: dict[int, int] = {}
    remapped_names: dict[str, str] = {}
    for side in ("Left", "Right"):
        hand_name = f"{side}Hand"
        if hand_name not in bone_order:
            raise ValueError(f"Donor skeleton is missing {hand_name!r}")
        hand_index = bone_order.index(hand_name)
        for index, name in enumerate(bone_order):
            if name.startswith(side) and any(token in name for token in finger_tokens):
                remap[index] = hand_index
                remapped_names[name] = hand_name
    if not remap:
        raise ValueError("Donor skeleton has no finger bones to collapse")
    changed_vertices, merged_influences = collapse_influences(vertices, remap)
    return {
        "enabled": True,
        "changed_vertices": changed_vertices,
        "merged_influences": merged_influences,
        "bone_remap": remapped_names,
    }


def collapse_toes_to_feet(vertices: list[dict], bone_order: list[str]) -> dict:
    remap: dict[int, int] = {}
    remapped_names: dict[str, str] = {}
    for side in ("Left", "Right"):
        foot_name = f"{side}Foot"
        toe_name = f"{side}ToeBase"
        if foot_name not in bone_order or toe_name not in bone_order:
            raise ValueError(f"Donor skeleton is missing {foot_name!r} or {toe_name!r}")
        remap[bone_order.index(toe_name)] = bone_order.index(foot_name)
        remapped_names[toe_name] = foot_name
    changed_vertices, merged_influences = collapse_influences(vertices, remap)
    return {
        "enabled": True,
        "changed_vertices": changed_vertices,
        "merged_influences": merged_influences,
        "bone_remap": remapped_names,
    }


def rigid_extremities(vertices: list[dict], bone_order: list[str]) -> dict:
    required = ("LeftHand", "RightHand", "LeftFoot", "RightFoot")
    missing = [name for name in required if name not in bone_order]
    if missing:
        raise ValueError(f"Donor skeleton is missing extremity bones: {missing}")
    indices = {name: bone_order.index(name) for name in required}
    counts = {name: 0 for name in required}

    for vertex in vertices:
        x, _y, z = (float(value) for value in vertex["position"])
        target_name = None
        if abs(x) > 0.48 and z > 0.80:
            target_name = "LeftHand" if x < 0.0 else "RightHand"
        elif abs(x) > 0.04 and z < 0.26:
            target_name = "LeftFoot" if x < 0.0 else "RightFoot"
        if target_name is None:
            continue
        vertex["bones"] = (indices[target_name], 0, 0, 0)
        vertex["weights"] = (1.0, 0.0, 0.0, 0.0)
        counts[target_name] += 1

    if not all(counts.values()):
        raise ValueError(f"Rigid extremity thresholds missed a required region: {counts}")
    return {
        "enabled": True,
        "thresholds": {"hand_abs_x_gt": 0.48, "hand_z_gt": 0.80, "foot_abs_x_gt": 0.04, "foot_z_lt": 0.26},
        "changed_vertices": sum(counts.values()),
        "bone_counts": counts,
    }


def build_model_buffer(payload: bytes, count: int, stride: int, flags: tuple[int, int], fmt: int) -> bytes:
    return struct.pack("<IIHBBI", count, len(payload), stride, flags[0], flags[1], fmt) + payload


def describe_attribute_layout(layout: dict) -> dict[str, object]:
    """Return the donor's slot-1 stream contract used by the first writer."""
    elements = layout.get("elements", [])
    if not elements:
        raise ValueError("Donor VLay has no elements")
    stream = [item for item in elements if int(item["input_slot"]) == 1]
    if not stream or stream[0]["semantic"] != "NORMAL" or int(stream[0]["semantic_index"]) != 0:
        raise ValueError("Donor VLay must begin its attribute stream with NORMAL0")
    if int(stream[0]["format"]) != 37:
        raise ValueError("Donor NORMAL0 is not R16G16_SNORM")
    uv_indices: list[int] = []
    tangent_indices: list[int] = []
    expected_order: list[tuple[str, int]] = []
    for item in stream:
        semantic = str(item["semantic"])
        semantic_index = int(item["semantic_index"])
        fmt = int(item["format"])
        if semantic == "NORMAL" and semantic_index == 0 and len(expected_order) == 0:
            expected_order.append((semantic, semantic_index))
        elif semantic == "TEXCOORD" and fmt == 35:
            uv_indices.append(semantic_index)
            expected_order.append((semantic, semantic_index))
        elif semantic == "TANGENT" and fmt == 24:
            tangent_indices.append(semantic_index)
            expected_order.append((semantic, semantic_index))
        else:
            raise ValueError(
                f"Unsupported donor VLay attribute {semantic}{semantic_index} format {fmt}"
            )
    if len(set(uv_indices)) != len(uv_indices) or len(set(tangent_indices)) != len(tangent_indices):
        raise ValueError("Donor VLay repeats a UV or tangent semantic index")
    stride = 4 + len(uv_indices) * 4 + len(tangent_indices) * 4
    return {
        "stride": stride,
        "uv_indices": uv_indices,
        "tangent_indices": tangent_indices,
        "order": expected_order,
    }


def patch_mesh_blob(
    original: bytes,
    *,
    start_index: int,
    base_vertex: int,
    index_count: int,
    radius: float,
    vertex_indices: list[int],
    uv_transform: list[float] | None,
    scale: list[float] | None,
    translate: list[float] | None,
) -> bytes:
    if len(original) < 238:
        raise ValueError("Mesh 1.12 blob is unexpectedly small")
    old_extended_count = struct.unpack_from("<I", original, 58)[0]
    old_suffix_offset = 62 + old_extended_count * 4
    if old_suffix_offset > len(original):
        raise ValueError("Mesh extended vertex array exceeds blob")
    suffix = bytearray(original[old_suffix_offset:])
    if len(suffix) != 176:
        raise ValueError(f"Expected 176-byte Mesh 1.12 suffix, found {len(suffix)}")

    prefix = bytearray(original[:62])
    struct.pack_into("<i", prefix, 34, start_index)
    struct.pack_into("<i", prefix, 38, base_vertex)
    struct.pack_into("<I", prefix, 42, index_count)
    struct.pack_into("<f", prefix, 50, radius)
    unique_vertex_count = len(set(vertex_indices)) if vertex_indices else 0
    if vertex_indices and len(vertex_indices) != unique_vertex_count * 2:
        raise ValueError("Mesh extended array must contain two complete vertex-index passes")
    struct.pack_into("<I", prefix, 54, unique_vertex_count)
    struct.pack_into("<I", prefix, 58, len(vertex_indices))

    if uv_transform is not None:
        if len(uv_transform) != 4:
            raise ValueError("UV transform must contain four floats")
        suffix[-112:-32] = struct.pack("<20f", *(uv_transform * 5))
    if scale is not None:
        suffix[-32:-16] = struct.pack("<4f", *scale)
    if translate is not None:
        suffix[-16:] = struct.pack("<4f", *translate)
    array_payload = struct.pack(f"<{len(vertex_indices)}I", *vertex_indices) if vertex_indices else b""
    return bytes(prefix) + array_payload + bytes(suffix)


def encode_geometry(
    vertices: list[dict],
    world_matrix: list[float],
    draw_ranges: list[dict[str, object]],
    attribute_layout: dict[str, object] | None = None,
) -> tuple[dict[str, bytes], dict]:
    if attribute_layout is None:
        attribute_layout = {
            "stride": 40,
            "uv_indices": [0, 1, 2, 3, 4],
            "tangent_indices": [0, 1, 2, 4],
            "order": [
                ("NORMAL", 0),
                *( ("TEXCOORD", index) for index in range(5) ),
                *( ("TANGENT", index) for index in (0, 1, 2, 4) ),
            ],
        }
    inverse_world = matrix_inverse(world_matrix)
    local_positions: list[tuple[float, float, float]] = []
    local_normals: list[tuple[float, float, float]] = []
    local_tangents: list[tuple[float, float, float]] = []
    for vertex in vertices:
        forza_world_position = blender_to_forza(vertex["position"])
        forza_world_normal = blender_to_forza(vertex["normal"])
        forza_world_tangent = blender_to_forza(vertex["tangent"][:3])
        local_positions.append(transform_row(forza_world_position, inverse_world, True))
        local_normals.append(normalize(transform_row(forza_world_normal, inverse_world, False)))
        local_tangents.append(normalize(transform_row(forza_world_tangent, inverse_world, False)))

    vertex_draws: list[int] = [0] * len(vertices)
    draw_quantization: list[dict[str, object]] = []
    for draw in draw_ranges:
        material_id = int(draw["material_id"])
        vertex_start = int(draw["vertex_start"])
        vertex_count = int(draw["vertex_count"])
        for index in range(vertex_start, vertex_start + vertex_count):
            vertex_draws[index] = material_id
        draw_positions = local_positions[vertex_start : vertex_start + vertex_count]
        minimum = [min(position[axis] for position in draw_positions) for axis in range(3)]
        maximum = [max(position[axis] for position in draw_positions) for axis in range(3)]
        draw_translate = [f32((minimum[axis] + maximum[axis]) * 0.5) for axis in range(3)] + [0.0]
        draw_scale = [f32(max((maximum[axis] - minimum[axis]) * 0.5, 1e-8)) for axis in range(3)] + [0.0]
        draw_radius = f32(max(vector_length(tuple(position[axis] - draw_translate[axis] for axis in range(3))) for position in draw_positions))
        draw_vertices = vertices[vertex_start : vertex_start + vertex_count]
        u_values = [float(vertex["uv"][0]) for vertex in draw_vertices]
        forza_v_values = [1.0 - float(vertex["uv"][1]) for vertex in draw_vertices]
        uv_offset_u = f32(min(u_values))
        uv_scale_u = f32(max(max(u_values) - uv_offset_u, 1e-8))
        uv_offset_v = f32(min(forza_v_values))
        uv_scale_v = f32(max(max(forza_v_values) - uv_offset_v, 1e-8))
        draw_quantization.append({
            "material_id": material_id,
            "scale": draw_scale,
            "translate": draw_translate,
            "radius": draw_radius,
            "uv_transform": [uv_offset_u, uv_scale_u, uv_offset_v, uv_scale_v],
            "bounds_min": minimum,
            "bounds_max": maximum,
        })

    if any(index < 0 or index >= len(draw_quantization) for index in vertex_draws):
        raise ValueError("Every exported vertex must belong to a draw range")

    minimum = [min(position[axis] for position in local_positions) for axis in range(3)]
    maximum = [max(position[axis] for position in local_positions) for axis in range(3)]
    translate = [f32((minimum[axis] + maximum[axis]) * 0.5) for axis in range(3)] + [0.0]
    scale = [f32(max((maximum[axis] - minimum[axis]) * 0.5, 1e-8)) for axis in range(3)] + [0.0]
    radius = f32(max(vector_length(tuple(position[axis] - translate[axis] for axis in range(3))) for position in local_positions))
    u_values = [float(vertex["uv"][0]) for vertex in vertices]
    forza_v_values = [1.0 - float(vertex["uv"][1]) for vertex in vertices]
    uv_offset_u = f32(min(u_values))
    uv_scale_u = f32(max(max(u_values) - uv_offset_u, 1e-8))
    uv_offset_v = f32(min(forza_v_values))
    uv_scale_v = f32(max(max(forza_v_values) - uv_offset_v, 1e-8))
    uv_transform = [uv_offset_u, uv_scale_u, uv_offset_v, uv_scale_v]

    position_payload = bytearray()
    attribute_payload = bytearray()
    skin_payload = bytearray()
    position_errors: list[float] = []
    normal_errors: list[float] = []
    tangent_errors: list[float] = []
    uv_errors: list[float] = []
    weight_errors: list[float] = []
    half_weight_sums: list[float] = []
    half_bone_index_errors: list[float] = []

    for index, (vertex, local_position, local_normal, local_tangent) in enumerate(zip(
        vertices, local_positions, local_normals, local_tangents
    )):
        draw_info = draw_quantization[vertex_draws[index]]
        draw_scale = draw_info["scale"]
        draw_translate = draw_info["translate"]
        draw_uv_transform = draw_info["uv_transform"]
        q_position = [snorm16_encode((local_position[axis] - draw_translate[axis]) / draw_scale[axis]) for axis in range(3)]
        q_normal = [snorm16_encode(component) for component in local_normal]
        position_payload.extend(struct.pack("<4h", q_position[0], q_position[1], q_position[2], q_normal[0]))

        raw_u = unorm16_encode((vertex["uv"][0] - draw_uv_transform[0]) / draw_uv_transform[1])
        raw_v = unorm16_encode(((1.0 - vertex["uv"][1]) - draw_uv_transform[2]) / draw_uv_transform[3])
        # Coordinate conversion reflects handedness, so the tangent-frame sign flips.
        packed_tangent = pack_tangent(local_tangent, -float(vertex["tangent"][3]))
        for semantic, semantic_index in attribute_layout["order"]:
            if semantic == "NORMAL":
                attribute_payload.extend(struct.pack("<2h", q_normal[1], q_normal[2]))
            elif semantic == "TEXCOORD":
                attribute_payload.extend(struct.pack("<2H", raw_u, raw_v))
            elif semantic == "TANGENT":
                attribute_payload.extend(struct.pack("<I", packed_tangent))
            else:
                raise ValueError(f"Unhandled attribute semantic {semantic}{semantic_index}")

        decoded_local_position = tuple(snorm16_decode(q_position[axis]) * draw_scale[axis] + draw_translate[axis] for axis in range(3))
        decoded_forza_position = transform_row(decoded_local_position, world_matrix, True)
        decoded_blender_position = forza_to_blender(decoded_forza_position)
        position_errors.append(vector_length(tuple(decoded_blender_position[axis] - vertex["position"][axis] for axis in range(3))))

        decoded_local_normal = tuple(snorm16_decode(value) for value in q_normal)
        decoded_forza_normal = normalize(transform_row(decoded_local_normal, world_matrix, False))
        decoded_blender_normal = forza_to_blender(decoded_forza_normal)
        normal_errors.append(angle_degrees(decoded_blender_normal, vertex["normal"]))

        decoded_local_tangent, decoded_handedness = unpack_tangent(packed_tangent)
        decoded_forza_tangent = normalize(transform_row(decoded_local_tangent, world_matrix, False))
        decoded_blender_tangent = forza_to_blender(decoded_forza_tangent)
        tangent_errors.append(angle_degrees(decoded_blender_tangent, vertex["tangent"][:3]))
        if decoded_handedness != -float(vertex["tangent"][3]):
            raise ValueError("Packed tangent handedness changed")

        decoded_u = (raw_u / 65535.0) * draw_uv_transform[1] + draw_uv_transform[0]
        decoded_forza_v = (raw_v / 65535.0) * draw_uv_transform[3] + draw_uv_transform[2]
        decoded_uv = (decoded_u, 1.0 - decoded_forza_v)
        uv_errors.append(math.hypot(decoded_uv[0] - vertex["uv"][0], decoded_uv[1] - vertex["uv"][1]))

        half_pairs: list[tuple[float, float]] = []
        for weight, bone_index in zip(vertex["weights"], vertex["bones"]):
            packed_pair = struct.pack("<ee", float(weight), float(bone_index))
            skin_payload.extend(packed_pair)
            half_weight, half_bone = struct.unpack("<ee", packed_pair)
            half_pairs.append((half_weight, half_bone))
            weight_errors.append(abs(half_weight - weight))
            half_bone_index_errors.append(abs(half_bone - bone_index))
        half_weight_sums.append(sum(weight for weight, _ in half_pairs))

    return (
        {
            "position": bytes(position_payload),
            "attribute": bytes(attribute_payload),
            "skin": bytes(skin_payload),
        },
        {
            "scale": scale,
            "translate": translate,
            "radius": radius,
            "uv_transform": uv_transform,
            "position_error_max_m": max(position_errors),
            "position_error_rms_m": math.sqrt(sum(value * value for value in position_errors) / len(position_errors)),
            "normal_error_max_degrees": max(normal_errors),
            "tangent_error_max_degrees": max(tangent_errors),
            "uv_error_max": max(uv_errors),
            "weight_error_max": max(weight_errors),
            "half_weight_sum_min": min(half_weight_sums),
            "half_weight_sum_max": max(half_weight_sums),
            "half_bone_index_error_max": max(half_bone_index_errors),
            "draws": draw_quantization,
            "attribute_layout": {
                "stride": int(attribute_layout["stride"]),
                "uv_indices": list(attribute_layout["uv_indices"]),
                "tangent_indices": list(attribute_layout["tangent_indices"]),
            },
        },
    )


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
    vertices, blender_indices = read_intermediate(manifest_path, manifest)
    donor_data = donor.read_bytes()
    outer = parse_outer_bundle(donor_data)
    header, blobs = inspector.parse_bundle(donor_data)
    parsed = inspector.parse_known_blobs(donor_data, blobs)
    if parsed["errors"]:
        raise ValueError(f"Donor parser errors: {parsed['errors']}")
    actual_tags = [blob.tag for blob in outer.blobs]
    tag_indices: dict[str, list[int]] = {}
    for index, tag in enumerate(actual_tags):
        tag_indices.setdefault(tag, []).append(index)
    required_counts = {"Skel": 1, "IndB": 1, "VLay": 1, "VerB": 2, "Skin": 1, "Modl": 1}
    for tag, count in required_counts.items():
        if len(tag_indices.get(tag, [])) != count:
            raise ValueError(f"Expected {count} {tag} blobs, found {len(tag_indices.get(tag, []))}: {actual_tags}")
    if len(tag_indices.get("MatI", [])) < 1 or len(tag_indices.get("Mesh", [])) < 2:
        raise ValueError(f"Donor must contain MatI and at least two Mesh blobs: {actual_tags}")
    meshes = parsed["meshes"]
    mesh_blob_indices = tag_indices["Mesh"]
    if len(meshes) != len(mesh_blob_indices) or any(mesh["blob_version"] != "1.12" for mesh in meshes):
        raise ValueError("Expected one parsed Mesh 1.12 descriptor per Mesh blob")
    skeleton = parsed["skeleton"][0]
    if len(parsed["vertex_layouts"]) != 1:
        raise ValueError(f"Expected one VLay blob, found {len(parsed['vertex_layouts'])}")
    attribute_layout = describe_attribute_layout(parsed["vertex_layouts"][0])
    donor_bone_order = [bone["name"] for bone in skeleton["bones"]]
    if donor_bone_order != manifest["skinning"]["bone_order"]:
        raise ValueError("Intermediate donor bone order differs from modelbin Skel order")
    for vertex in vertices:
        if max(vertex["bones"]) >= len(donor_bone_order):
            raise ValueError("Intermediate bone index exceeds donor skeleton")
    hand_collapse = {"enabled": False, "changed_vertices": 0, "merged_influences": 0, "bone_remap": {}}
    if args.collapse_fingers_to_hands:
        hand_collapse = collapse_fingers_to_hands(vertices, donor_bone_order)
        if hand_collapse["changed_vertices"] == 0:
            raise ValueError("Finger collapse did not affect any vertices")
    toe_collapse = {"enabled": False, "changed_vertices": 0, "merged_influences": 0, "bone_remap": {}}
    if args.collapse_toes_to_feet:
        toe_collapse = collapse_toes_to_feet(vertices, donor_bone_order)
        if toe_collapse["changed_vertices"] == 0:
            raise ValueError("Toe collapse did not affect any vertices")
    extremity_rigidity = {"enabled": False, "changed_vertices": 0, "bone_counts": {}}
    if args.rigid_extremities:
        extremity_rigidity = rigid_extremities(vertices, donor_bone_order)
    if args.static_bone_name is not None:
        if args.static_bone_name not in donor_bone_order:
            raise ValueError(f"Static bone {args.static_bone_name!r} is absent from donor skeleton")
        static_bone_index = donor_bone_order.index(args.static_bone_name)
        for vertex in vertices:
            vertex["weights"] = (1.0, 0.0, 0.0, 0.0)
            vertex["bones"] = (static_bone_index, 0, 0, 0)

    draws = manifest["geometry"].get("draws")
    draw_by_material: dict[int, dict] | None = None
    lod_group_order: list[int] = []
    lod_index_offsets: dict[int, int] = {}
    if draws is not None:
        draw_by_material = {int(draw["material_id"]): draw for draw in draws}
        if len(draw_by_material) != len(draws):
            raise ValueError("Intermediate draw material IDs are not unique")
        if {int(mesh["material_id"]) for mesh in meshes} != set(draw_by_material):
            raise ValueError("Intermediate draw material IDs do not match donor Mesh material IDs")
        if args.duplicate_draws_for_lod_groups:
            for mesh in meshes:
                lod_flags = int(mesh["lod_flags"])
                if lod_flags not in lod_group_order:
                    lod_group_order.append(lod_flags)
            if len(lod_group_order) < 2:
                raise ValueError(
                    "--duplicate-draws-for-lod-groups requires at least two donor LOD groups"
                )
            for lod_flags in lod_group_order:
                material_ids = [
                    int(mesh["material_id"])
                    for mesh in meshes
                    if int(mesh["lod_flags"]) == lod_flags
                ]
                if len(material_ids) != len(draws) or set(material_ids) != set(draw_by_material):
                    raise ValueError(
                        "Every duplicated LOD group must contain exactly one Mesh per "
                        f"intermediate material: lod={lod_flags}, materials={material_ids}"
                    )
        elif len(meshes) != len(draws):
            raise ValueError(
                f"Strict draw export requires one draw per Mesh: {len(draws)} draws, {len(meshes)} meshes"
            )
        cursor = 0
        vertex_cursor = 0
        for draw in sorted(draws, key=lambda item: int(item["start_index"])):
            start = int(draw["start_index"])
            count = int(draw["index_count"])
            if start != cursor or count <= 0 or count % 3:
                raise ValueError(f"Draw ranges must be nonempty, triangle-aligned, and contiguous: {draw}")
            cursor += count
            vertex_start = int(draw.get("vertex_start", -1))
            vertex_count = int(draw.get("vertex_count", 0))
            if vertex_start != vertex_cursor or vertex_count <= 0:
                raise ValueError(f"Draw vertex ranges must be dense, contiguous, and nonempty: {draw}")
            vertex_cursor += vertex_count
        if cursor != len(blender_indices):
            raise ValueError("Draw ranges do not partition the complete intermediate index buffer")
        if vertex_cursor != len(vertices):
            raise ValueError(
                f"Draw vertex-count sum {vertex_cursor} differs from VerB/Skin count {len(vertices)}"
            )
        for draw in draws:
            start = int(draw["start_index"])
            end = start + int(draw["index_count"])
            vertex_start = int(draw["vertex_start"])
            vertex_end = vertex_start + int(draw["vertex_count"])
            if any(index < vertex_start or index >= vertex_end for index in blender_indices[start:end]):
                raise ValueError(f"Draw {draw['material_id']} index resolves outside its vertex range")

    world_matrices = skeleton_world_matrices(skeleton["bones"])
    active_source_meshes = meshes if draw_by_material is not None else [
        mesh for mesh in meshes if int(mesh["material_id"]) == 0
    ]
    if not active_source_meshes:
        raise ValueError("No donor Mesh descriptor is active for export")
    anchor_indices = sorted({int(mesh["bone_index"]) for mesh in active_source_meshes})
    anchor_matrix = world_matrices[anchor_indices[0]]
    for anchor in anchor_indices[1:]:
        if any(abs(left - right) > 1e-6 for left, right in zip(anchor_matrix, world_matrices[anchor])):
            raise ValueError("Material-0 LOD anchors differ; a shared first-writer domain is unsafe")
    encoded, quantization = encode_geometry(vertices, anchor_matrix, draws or [{
        "material_id": 0,
        "vertex_start": 0,
        "vertex_count": len(vertices),
    }], attribute_layout)

    source_forza_indices: list[int] = []
    for offset in range(0, len(blender_indices), 3):
        first, second, third = blender_indices[offset : offset + 3]
        source_forza_indices.extend((first, third, second))
    if lod_group_order:
        forza_indices = source_forza_indices * len(lod_group_order)
        lod_index_offsets = {
            lod_flags: group_index * len(source_forza_indices)
            for group_index, lod_flags in enumerate(lod_group_order)
        }
    else:
        forza_indices = source_forza_indices
    index_payload = struct.pack(f"<{len(forza_indices)}H", *forza_indices)
    vertex_count = len(vertices)
    uv_transform = quantization["uv_transform"]
    scale = quantization["scale"]
    translate = quantization["translate"]

    replacements: dict[int, bytes] = {}
    active_meshes = 0
    for mesh, blob_index in zip(meshes, mesh_blob_indices):
        material_id = int(mesh["material_id"])
        active = material_id in draw_by_material if draw_by_material is not None else material_id == 0
        active_meshes += int(active)
        draw = draw_by_material[material_id] if draw_by_material is not None else None
        lod_offset = lod_index_offsets.get(int(mesh["lod_flags"]), 0)
        start_index = (
            int(draw["start_index"]) + lod_offset
            if draw is not None
            else (0 if active else len(forza_indices))
        )
        index_count = int(draw["index_count"]) if draw is not None else (len(forza_indices) if active else 0)
        if active:
            vertex_start = int(draw["vertex_start"]) if draw is not None else 0
            mesh_vertex_count = int(draw["vertex_count"]) if draw is not None else vertex_count
            mesh_vertex_array = list(range(vertex_start, vertex_start + mesh_vertex_count)) * 2
            draw_quantization = (
                next(item for item in quantization["draws"] if int(item["material_id"]) == material_id)
                if draw is not None
                else quantization["draws"][0]
            )
        else:
            mesh_vertex_array = []
            draw_quantization = None
        replacements[blob_index] = patch_mesh_blob(
            outer.blobs[blob_index].data,
            start_index=start_index,
            base_vertex=0,
            index_count=index_count,
            radius=float(draw_quantization["radius"]) if active else 0.0,
            vertex_indices=mesh_vertex_array if active else [],
            uv_transform=draw_quantization["uv_transform"] if active else None,
            scale=draw_quantization["scale"] if active else None,
            translate=draw_quantization["translate"] if active else None,
        )
    replacements[tag_indices["IndB"][0]] = build_model_buffer(index_payload, len(forza_indices), 2, (1, 0), 57)
    replacements[tag_indices["VerB"][0]] = build_model_buffer(encoded["position"], vertex_count, 8, (1, 0), 13)
    attribute_stride = int(attribute_layout["stride"])
    if len(encoded["attribute"]) != vertex_count * attribute_stride:
        raise ValueError(
            f"Encoded attribute payload has {len(encoded['attribute'])} bytes; "
            f"expected {vertex_count * attribute_stride} for donor VLay"
        )
    replacements[tag_indices["VerB"][1]] = build_model_buffer(
        encoded["attribute"], vertex_count, attribute_stride, (10, 0), 37
    )
    replacements[tag_indices["Skin"][0]] = build_model_buffer(encoded["skin"], vertex_count, 16, (4, 0), 34)
    candidate_data = rebuild_with_blob_data(outer, replacements)
    # Mesh and Modl BBox metadata lives outside the opaque Mesh payload. Keep
    # it synchronized with the transformed local geometry so runtime culling
    # and animation resource construction see valid bounds.
    candidate_bytes = bytearray(candidate_data)
    bounds_by_material = {
        int(item["material_id"]): (
            [float(value) for value in item["bounds_min"]],
            [float(value) for value in item["bounds_max"]],
        )
        for item in quantization["draws"]
    }
    all_min = [math.inf, math.inf, math.inf]
    all_max = [-math.inf, -math.inf, -math.inf]
    for mesh, blob_index in zip(meshes, mesh_blob_indices):
        material_id = int(mesh["material_id"])
        if material_id not in bounds_by_material:
            continue
        bounds_min, bounds_max = bounds_by_material[material_id]
        for axis in range(3):
            all_min[axis] = min(all_min[axis], bounds_min[axis])
            all_max[axis] = max(all_max[axis], bounds_max[axis])
        bbox_entries = [entry for entry in outer.blobs[blob_index].metadata if entry.tag == "BBox"]
        if len(bbox_entries) != 1 or len(bbox_entries[0].value) != 24:
            raise ValueError(f"Mesh {material_id} does not have one 24-byte BBox metadata entry")
        struct.pack_into("<6f", candidate_bytes, bbox_entries[0].value_offset, *(bounds_min + bounds_max))
    modl_blob_index = tag_indices["Modl"][0]
    modl_bbox_entries = [entry for entry in outer.blobs[modl_blob_index].metadata if entry.tag == "BBox"]
    if len(modl_bbox_entries) != 1 or len(modl_bbox_entries[0].value) != 24:
        raise ValueError("Modl does not have one 24-byte BBox metadata entry")
    struct.pack_into("<6f", candidate_bytes, modl_bbox_entries[0].value_offset, *(all_min + all_max))
    candidate_data = bytes(candidate_bytes)
    output.write_bytes(candidate_data)

    candidate_report = inspector.inspect(output)
    if candidate_report["parsed"]["errors"]:
        raise ValueError(f"Candidate parser errors: {candidate_report['parsed']['errors']}")
    candidate_outer = parse_outer_bundle(candidate_data)
    preserved_indices = sorted(
        tag_indices["Skel"] + tag_indices["MatI"] + tag_indices["VLay"] + tag_indices["Modl"]
    )
    preservation = {
        str(index): {
            "tag": outer.blobs[index].tag,
            "preserved": outer.blobs[index].data == candidate_outer.blobs[index].data,
            "sha256": sha256_bytes(candidate_outer.blobs[index].data),
        }
        for index in preserved_indices
    }
    if not all(item["preserved"] for item in preservation.values()):
        raise ValueError("A required donor Skel/MatI/VLay/Modl blob changed")

    candidate_meshes = candidate_report["parsed"]["meshes"]
    candidate_index = candidate_report["parsed"]["index_buffers"][0]
    candidate_vertices = candidate_report["parsed"]["vertex_buffers"]
    candidate_skin = candidate_report["parsed"]["skin_buffers"][0]
    candidate_ranges = sorted(
        (int(mesh["start_index"]), int(mesh["start_index"]) + int(mesh["index_count"]))
        for mesh in candidate_meshes
        if int(mesh["index_count"]) > 0
    )
    if draw_by_material is not None:
        if not candidate_ranges or candidate_ranges[0][0] != 0 or candidate_ranges[-1][1] != len(forza_indices):
            raise ValueError("Candidate Mesh ranges do not cover the complete index buffer")
        if any(left[1] != right[0] for left, right in zip(candidate_ranges, candidate_ranges[1:])):
            raise ValueError("Candidate Mesh ranges contain a gap or overlap")
        if sum(mesh["index_count"] for mesh in candidate_meshes) != len(forza_indices):
            raise ValueError("Candidate Mesh index-count sum differs from IndB count")
    report = {
        "schema_version": 1,
        "purpose": "Structural FH6 garment modelbin candidate using donor materials and full-resolution geometry; not yet game validated.",
        "donor": {
            "path": str(donor),
            "sha256": sha256(donor),
            "bytes": len(donor_data),
        },
        "intermediate": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "vertices": vertex_count,
            "indices": len(forza_indices),
            "triangles": len(forza_indices) // 3,
        },
        "candidate": {
            "path": str(output),
            "sha256": sha256(output),
            "bytes": len(candidate_data),
            "bundle": candidate_report["header"],
            "parse_errors": candidate_report["parsed"]["errors"],
            "meshes": [
                {
                    "name": mesh["name"],
                    "lod_flags": mesh["lod_flags"],
                    "material_id": mesh["material_id"],
                    "start_index": mesh["start_index"],
                    "index_count": mesh["index_count"],
                    "vertex_count": mesh["extended_unknown_u32"][1],
                    "extended_array_count": mesh["extended_array_count"],
                    "scale": mesh["scale"],
                    "translate": mesh["translate"],
                }
                for mesh in candidate_meshes
            ],
            "index_buffer": {
                "count": candidate_index["count"],
                "bytes": candidate_index["byte_size"],
                "stride": candidate_index["stride"],
                "format": candidate_index["format"],
            },
            "vertex_buffers": [
                {
                    "id": item["id"],
                    "count": item["count"],
                    "bytes": item["byte_size"],
                    "stride": item["stride"],
                    "format": item["format"],
                }
                for item in candidate_vertices
            ],
            "skin_buffer": {
                "count": candidate_skin["count"],
                "bytes": candidate_skin["byte_size"],
                "stride": candidate_skin["stride"],
                "format": candidate_skin["format"],
                "weight_sum_min": candidate_skin["weight_sum_min"],
                "weight_sum_max": candidate_skin["weight_sum_max"],
                "bone_index_min": candidate_skin["bone_index_min"],
                "bone_index_max": candidate_skin["bone_index_max"],
            },
        },
        "quantization": quantization,
        "hand_collapse": hand_collapse,
        "toe_collapse": toe_collapse,
        "extremity_rigidity": extremity_rigidity,
        "preserved_blobs": preservation,
        "policies": {
            "skeleton": "donor Skel retained byte-exact",
            "layout": "donor VLay retained byte-exact",
            "materials": "donor MatI retained byte-exact; source geometry uses material id 0",
            "lod": (
                f"{len(lod_group_order)} donor LOD groups share one vertex/Skin domain "
                "and use duplicate disjoint index partitions"
                if lod_group_order
                else (
                    "strict intermediate draws physically partition one index buffer"
                    if draw_by_material is not None
                    else f"{active_meshes} material-0 LOD descriptors share one full-resolution vertex/index domain"
                )
            ),
            "inactive_materials": "none" if draw_by_material is not None else "all nonzero donor material Mesh descriptors are retained with empty draw ranges",
            "extra_uvs": f"donor VLay UV indices {attribute_layout['uv_indices']} duplicate TEXCOORD0 in the first writer",
            "extra_tangents": f"donor VLay tangent indices {attribute_layout['tangent_indices']} duplicate TANGENT0 in the first writer",
            "skinning_mode": (
                f"all vertices rigidly bound to {args.static_bone_name!r} for diagnostic isolation"
                if args.static_bone_name is not None
                else (
                    "outer hand and foot regions rigidly bound to their terminal bones; remaining intermediate skinning retained"
                    if args.rigid_extremities
                    else "terminal finger/toe influences collapsed; remaining intermediate skinning retained"
                    if args.collapse_fingers_to_hands or args.collapse_toes_to_feet
                    else "intermediate four-influence skinning retained"
                )
            ),
        },
        "validation_level": {
            "structural": True,
            "blender_visual": False,
            "offline_game": False,
        },
        "license_guard": "Local technical validation only; do not redistribute this candidate.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_GARMENT_MODELBIN="
        + json.dumps(
            {
                "output": str(output),
                "report": str(report_path),
                "bytes": len(candidate_data),
                "vertices": vertex_count,
                "indices": len(forza_indices),
                "parse_errors": len(candidate_report["parsed"]["errors"]),
                "position_error_max_m": quantization["position_error_max_m"],
                "normal_error_max_degrees": quantization["normal_error_max_degrees"],
                "half_weight_sum_min": quantization["half_weight_sum_min"],
                "half_weight_sum_max": quantization["half_weight_sum_max"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
