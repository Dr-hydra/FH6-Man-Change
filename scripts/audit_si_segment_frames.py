#!/usr/bin/env python3
"""Audit Si/FH6 REST segment geometry without consuming bone matrices.

The skeleton inventories contain matrix_local for archival purposes, but this
audit intentionally derives every frame from REST heads, REST tails, hierarchy,
and explicit anatomical correspondences.  It does not mutate Blender scenes or
retarget outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "work/si/fbx-source/milestone-02-donor-plan/skeletons/source-fbx.skeleton.json"
DEFAULT_OUTPUT = ROOT / "work/si/fbx-source/milestone-05-validation-v001/segment-frame-audit"
VALIDATION_REPORT = ROOT / "work/si/fbx-source/milestone-05-validation-v001/lod0/si-display-lod0-validation-v002.report.json"

CONTAINERS = {
    "head_hair_helmet": {
        "skeleton": ROOT / "work/si/fbx-source/milestone-02-donor-plan/skeletons/helmet-race-modern.skeleton.json",
        "mapping": ROOT / "work/si/fbx-source/milestone-02-donor-plan/bone-maps/head-hair-to-helmet-v004.json",
        "retarget_report": ROOT / "work/si/fbx-source/milestone-03-retarget-v004/lod0/head-hair-retarget.report.json",
        "required_families": {"spine_neck_head"},
    },
    "body_garment_outfit": {
        "skeleton": ROOT / "work/si/fbx-source/milestone-02-donor-plan/skeletons/outfit-race-suit-modern-f.skeleton.json",
        "mapping": ROOT / "work/si/fbx-source/milestone-02-donor-plan/bone-maps/body-garment-to-outfit-v004.json",
        "retarget_report": ROOT / "work/si/fbx-source/milestone-03-retarget-v004/lod0/body-garment-retarget.report.json",
        "required_families": {"spine_neck_head", "arms", "legs", "fingers"},
    },
}


def finger_source(side: str, digit: int) -> list[str]:
    prefix = f"Bip001_{side}_Finger{digit}"
    return [f"Bip001_{side}_Hand", prefix, f"{prefix}1", f"{prefix}2"]


def finger_target(side: str, digit: str) -> list[str]:
    prefix = "Left" if side == "L" else "Right"
    if digit == "Thumb":
        return [f"{prefix}Hand", f"{prefix}Thumb1", f"{prefix}Thumb2", f"{prefix}Thumb3"]
    return [
        f"{prefix}Hand",
        f"{prefix}{digit}Meta",
        f"{prefix}{digit}1",
        f"{prefix}{digit}2",
        f"{prefix}{digit}3",
    ]


CHAIN_DEFS: list[dict[str, Any]] = [
    {
        "name": "spine_neck_head",
        "family": "spine_neck_head",
        "source_nodes": [
            "Bip001_Pelvis",
            "Bip001_Spine",
            "Bip001_Spine1",
            "Bip001_Spine2",
            "Bip001_Neck",
            "Bip001_Head",
        ],
        "target_nodes": ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head"],
        "semantic_nodes": [
            ("Bip001_Pelvis", "Hips"),
            ("Bip001_Spine", "Spine"),
            ("Bip001_Spine1", "Spine1"),
            ("Bip001_Spine2", "Spine2"),
            ("Bip001_Neck", "Neck"),
            ("Bip001_Head", "Head"),
        ],
        "terminal": False,
        "roll_hint": (1.0, 0.0, 0.0),
        "structural_note": "FH6 inserts Neck1 between source Neck and Head; solve it by arc-length resampling.",
    },
    {
        "name": "left_arm",
        "family": "arms",
        "source_nodes": ["Bip001_L_Clavicle", "Bip001_L_UpperArm", "Bip001_L_Forearm", "Bip001_L_Hand"],
        "target_nodes": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
        "semantic_nodes": [
            ("Bip001_L_Clavicle", "LeftShoulder"),
            ("Bip001_L_UpperArm", "LeftArm"),
            ("Bip001_L_Forearm", "LeftForeArm"),
            ("Bip001_L_Hand", "LeftHand"),
        ],
        "terminal": False,
        "roll_hint": (0.0, 1.0, 0.0),
    },
    {
        "name": "right_arm",
        "family": "arms",
        "source_nodes": ["Bip001_R_Clavicle", "Bip001_R_UpperArm", "Bip001_R_Forearm", "Bip001_R_Hand"],
        "target_nodes": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
        "semantic_nodes": [
            ("Bip001_R_Clavicle", "RightShoulder"),
            ("Bip001_R_UpperArm", "RightArm"),
            ("Bip001_R_Forearm", "RightForeArm"),
            ("Bip001_R_Hand", "RightHand"),
        ],
        "terminal": False,
        "roll_hint": (0.0, 1.0, 0.0),
    },
    {
        "name": "left_leg",
        "family": "legs",
        "source_nodes": ["Bip001_L_Thigh", "Bip001_L_Calf", "Bip001_L_Foot", "Bip001_L_Toe0"],
        "target_nodes": ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"],
        "semantic_nodes": [
            ("Bip001_L_Thigh", "LeftUpLeg"),
            ("Bip001_L_Calf", "LeftLeg"),
            ("Bip001_L_Foot", "LeftFoot"),
            ("Bip001_L_Toe0", "LeftToeBase"),
        ],
        "terminal": False,
        "roll_hint": (0.0, 1.0, 0.0),
    },
    {
        "name": "right_leg",
        "family": "legs",
        "source_nodes": ["Bip001_R_Thigh", "Bip001_R_Calf", "Bip001_R_Foot", "Bip001_R_Toe0"],
        "target_nodes": ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"],
        "semantic_nodes": [
            ("Bip001_R_Thigh", "RightUpLeg"),
            ("Bip001_R_Calf", "RightLeg"),
            ("Bip001_R_Foot", "RightFoot"),
            ("Bip001_R_Toe0", "RightToeBase"),
        ],
        "terminal": False,
        "roll_hint": (0.0, 1.0, 0.0),
    },
]

for side, side_name in (("L", "left"), ("R", "right")):
    for digit_index, digit_name in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky")):
        source_nodes = finger_source(side, digit_index)
        target_nodes = finger_target(side, digit_name)
        semantic_targets = [target_nodes[0], *target_nodes[-3:]]
        CHAIN_DEFS.append(
            {
                "name": f"{side_name}_{digit_name.lower()}",
                "family": "fingers",
                "source_nodes": source_nodes,
                "target_nodes": target_nodes,
                "semantic_nodes": list(zip(source_nodes, semantic_targets)),
                "terminal": False,
                "roll_hint": (0.0, 1.0, 0.0),
                "structural_note": (
                    None
                    if digit_name == "Thumb"
                    else f"FH6 inserts {target_nodes[1]} between Hand and {target_nodes[2]}; use arc-length resampling."
                ),
            }
        )


GATES = {
    "spine_neck_head": {
        "affine_anchor_rms_mm_max": 1.0,
        "affine_anchor_max_mm_max": 2.0,
        "axis_residual_deg_max": 0.5,
        "roll_residual_deg_max": 2.0,
        "joint_continuity_mm_max": 1.0,
        "condition_number_max": 2.25,
        "downstream_neck_surface_p95_mm_max": 5.0,
    },
    "arms": {
        "affine_anchor_rms_mm_max": 1.5,
        "affine_anchor_max_mm_max": 2.5,
        "axis_residual_deg_max": 0.75,
        "roll_residual_deg_max": 3.0,
        "joint_continuity_mm_max": 1.0,
        "condition_number_max": 2.5,
        "downstream_wrist_surface_p95_mm_max": 8.0,
        "downstream_pose_edge_stretch_p95_max": 1.35,
    },
    "legs": {
        "affine_anchor_rms_mm_max": 1.5,
        "affine_anchor_max_mm_max": 2.5,
        "axis_residual_deg_max": 0.75,
        "roll_residual_deg_max": 3.0,
        "joint_continuity_mm_max": 1.0,
        "condition_number_max": 2.5,
        "downstream_ankle_surface_p95_mm_max": 8.0,
        "downstream_pose_edge_stretch_p95_max": 1.35,
    },
    "fingers": {
        "affine_anchor_rms_mm_max": 0.75,
        "affine_anchor_max_mm_max": 1.5,
        "axis_residual_deg_max": 1.0,
        "roll_residual_deg_max": 5.0,
        "joint_continuity_mm_max": 0.5,
        "condition_number_max": 2.5,
    },
}

EPS_LENGTH_M = 5.0e-4
EPS_VECTOR = 1.0e-10
CURVATURE_CONFIDENCE_SIN = math.sin(math.radians(3.0))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Sequence[float], value: float) -> tuple[float, float, float]:
    return (a[0] * value, a[1] * value, a[2] * value)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def normalize(a: Sequence[float]) -> tuple[float, float, float] | None:
    magnitude = length(a)
    if magnitude <= EPS_VECTOR:
        return None
    return mul(a, 1.0 / magnitude)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return length(sub(a, b))


def project_plane(vector: Sequence[float], normal: Sequence[float]) -> tuple[float, float, float]:
    return sub(vector, mul(normal, dot(vector, normal)))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = normalize(a)
    nb = normalize(b)
    if na is None or nb is None:
        return float("nan")
    return math.degrees(math.acos(clamp(dot(na, nb), -1.0, 1.0)))


def rotate_axis_angle(vector: Sequence[float], axis: Sequence[float], angle: float) -> tuple[float, float, float]:
    unit_axis = normalize(axis)
    if unit_axis is None or abs(angle) <= EPS_VECTOR:
        return tuple(vector)  # type: ignore[return-value]
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return add(add(mul(vector, cosine), mul(cross(unit_axis, vector), sine)), mul(unit_axis, dot(unit_axis, vector) * (1.0 - cosine)))


def swing_vector(vector: Sequence[float], source_axis: Sequence[float], target_axis: Sequence[float], source_roll: Sequence[float]) -> tuple[float, float, float]:
    source = normalize(source_axis)
    target = normalize(target_axis)
    if source is None or target is None:
        return tuple(vector)  # type: ignore[return-value]
    cosine = clamp(dot(source, target), -1.0, 1.0)
    rotation_axis = cross(source, target)
    sine = length(rotation_axis)
    if sine > EPS_VECTOR:
        return rotate_axis_angle(vector, rotation_axis, math.atan2(sine, cosine))
    if cosine < 0.0:
        fallback = normalize(project_plane(source_roll, source))
        if fallback is None:
            fallback = choose_least_parallel(source)
        return rotate_axis_angle(vector, fallback, math.pi)
    return tuple(vector)  # type: ignore[return-value]


def signed_angle_deg(a: Sequence[float], b: Sequence[float], axis: Sequence[float]) -> float:
    na = normalize(project_plane(a, axis))
    nb = normalize(project_plane(b, axis))
    unit_axis = normalize(axis)
    if na is None or nb is None or unit_axis is None:
        return float("nan")
    return math.degrees(math.atan2(dot(unit_axis, cross(na, nb)), clamp(dot(na, nb), -1.0, 1.0)))


def choose_least_parallel(axis: Sequence[float]) -> tuple[float, float, float]:
    candidates = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    candidate = min(candidates, key=lambda item: abs(dot(axis, item)))
    projected = normalize(project_plane(candidate, axis))
    if projected is None:
        raise RuntimeError("Unable to construct deterministic perpendicular axis")
    return projected


def transform_point(point: Sequence[float], rotation_z_degrees: float, translation: Sequence[float], scale: float) -> tuple[float, float, float]:
    radians = math.radians(rotation_z_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    x = scale * (cosine * point[0] - sine * point[1])
    y = scale * (sine * point[0] + cosine * point[1])
    z = scale * point[2]
    return (x + translation[0], y + translation[1], z + translation[2])


def bone_index(skeleton: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {bone["name"]: bone for bone in skeleton["bones"]}


def frame_references(axes: Sequence[Sequence[float]], hint: Sequence[float]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    previous_reference: tuple[float, float, float] | None = None
    for index, axis_raw in enumerate(axes):
        axis = normalize(axis_raw)
        if axis is None:
            references.append({"reference": None, "binormal": None, "basis": "degenerate", "confidence": "invalid"})
            continue

        candidates: list[tuple[str, tuple[float, float, float], float]] = []
        projected_hint = project_plane(hint, axis)
        candidates.append(("anatomical world hint", projected_hint, length(projected_hint)))
        if index + 1 < len(axes):
            next_axis = normalize(axes[index + 1])
            if next_axis is not None:
                projected = project_plane(next_axis, axis)
                candidates.append(("next-segment bend projection", projected, length(projected)))
        if index > 0:
            prior_axis = normalize(axes[index - 1])
            if prior_axis is not None:
                projected = project_plane(prior_axis, axis)
                candidates.append(("prior-segment bend projection", projected, length(projected)))
        chosen_name = "least-parallel canonical axis"
        chosen = choose_least_parallel(axis)
        chosen_strength = 0.0
        for name, candidate, strength in candidates:
            unit = normalize(candidate)
            if unit is not None and strength > EPS_VECTOR:
                chosen_name = name
                chosen = unit
                chosen_strength = strength
                if strength >= CURVATURE_CONFIDENCE_SIN:
                    break

        if previous_reference is not None:
            transported = normalize(project_plane(previous_reference, axis))
            if transported is not None:
                if dot(chosen, transported) < 0.0:
                    chosen = mul(chosen, -1.0)
                if chosen_strength < CURVATURE_CONFIDENCE_SIN:
                    chosen = transported
                    chosen_name = "parallel-transported prior reference"

        binormal = normalize(cross(axis, chosen))
        if binormal is None:
            chosen = choose_least_parallel(axis)
            binormal = normalize(cross(axis, chosen))
        previous_reference = chosen
        references.append(
            {
                "reference": chosen,
                "binormal": binormal,
                "basis": chosen_name,
                "confidence": "geometric" if chosen_strength >= CURVATURE_CONFIDENCE_SIN else "fallback",
                "projection_strength": chosen_strength,
            }
        )
    return references


def chain_inventory(
    skeleton: dict[str, Any],
    nodes: Sequence[str],
    hint: Sequence[float],
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    bones = bone_index(skeleton)
    missing = [name for name in nodes if name not in bones]
    if missing:
        return {"missing_bones": missing, "segments": []}

    def point(value: Sequence[float]) -> tuple[float, float, float]:
        if alignment is None:
            return tuple(value)  # type: ignore[return-value]
        return transform_point(value, alignment["rotation_z_degrees"], alignment["translation"], alignment.get("scale", 1.0))

    effective_axes: list[tuple[float, float, float]] = []
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(nodes):
        bone = bones[name]
        original_head = tuple(bone["head_world_rest"])
        original_tail = tuple(bone["tail_world_rest"])
        head = point(original_head)
        tail = point(original_tail)
        if index + 1 < len(nodes):
            effective_tail = point(bones[nodes[index + 1]]["head_world_rest"])
            endpoint_rule = "next anatomical bone head"
        else:
            effective_tail = tail
            endpoint_rule = "terminal raw bone tail"
        raw_vector = sub(tail, head)
        effective_vector = sub(effective_tail, head)
        raw_length = length(raw_vector)
        effective_length = length(effective_vector)
        effective_axes.append(effective_vector)
        rows.append(
            {
                "bone": name,
                "parent": bone.get("parent"),
                "used_by_mesh_count": bone.get("used_by_mesh_count"),
                "head_world_rest": original_head,
                "tail_world_rest": original_tail,
                "head_analysis_space": head,
                "tail_analysis_space": tail,
                "raw_length_m": raw_length,
                "raw_unit_axis": normalize(raw_vector),
                "effective_tail_analysis_space": effective_tail,
                "effective_endpoint_rule": endpoint_rule,
                "effective_length_m": effective_length,
                "effective_unit_axis": normalize(effective_vector),
                "raw_tail_is_anatomical_endpoint": index + 1 == len(nodes),
                "degenerate": effective_length < EPS_LENGTH_M,
            }
        )
    references = frame_references(effective_axes, hint)
    for row, reference in zip(rows, references):
        row["roll_proxy"] = reference
    return {"missing_bones": [], "segments": rows}


def matrix_from_columns(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[columns[column][row] for column in range(3)] for row in range(3)]


def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(len(matrix))] for row in range(len(matrix[0]))]


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(len(right))) for column in range(len(right[0]))]
        for row in range(len(left))
    ]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def affine_from_frames(
    source_head: Sequence[float],
    source_end: Sequence[float],
    source_reference: Sequence[float],
    target_head: Sequence[float],
    target_end: Sequence[float],
    target_reference: Sequence[float],
) -> dict[str, Any]:
    source_vector = sub(source_end, source_head)
    target_vector = sub(target_end, target_head)
    source_length = length(source_vector)
    target_length = length(target_vector)
    source_axis = normalize(source_vector)
    target_axis = normalize(target_vector)
    if source_length < EPS_LENGTH_M or target_length < EPS_LENGTH_M or source_axis is None or target_axis is None:
        return {"valid": False, "reason": "segment shorter than 0.5 mm"}
    source_ref = normalize(project_plane(source_reference, source_axis))
    target_ref = normalize(project_plane(target_reference, target_axis))
    if source_ref is None:
        source_ref = choose_least_parallel(source_axis)
    if target_ref is None:
        target_ref = choose_least_parallel(target_axis)
    source_binormal = normalize(cross(source_axis, source_ref))
    target_binormal = normalize(cross(target_axis, target_ref))
    if source_binormal is None or target_binormal is None:
        return {"valid": False, "reason": "roll basis is degenerate"}
    source_frame = matrix_from_columns((source_axis, source_ref, source_binormal))
    target_frame = matrix_from_columns((target_axis, target_ref, target_binormal))
    axial_scale = target_length / source_length
    scale_matrix = [[axial_scale, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    linear = matmul(matmul(target_frame, scale_matrix), transpose(source_frame))
    translation = sub(target_head, matvec(linear, source_head))
    mapped_head = add(matvec(linear, source_head), translation)
    mapped_end = add(matvec(linear, source_end), translation)
    condition_number = max(axial_scale, 1.0) / min(axial_scale, 1.0)
    swung_reference = swing_vector(source_ref, source_axis, target_axis, source_ref)
    return {
        "valid": True,
        "linear_3x3": linear,
        "translation": translation,
        "axial_scale": axial_scale,
        "radial_scale": 1.0,
        "determinant": axial_scale,
        "condition_number": condition_number,
        "swing_axis_angle_deg": angle_deg(source_axis, target_axis),
        "roll_delta_deg": signed_angle_deg(swung_reference, target_ref, target_axis),
        "verification": {
            "head_residual_m": distance(mapped_head, target_head),
            "end_residual_m": distance(mapped_end, target_end),
        },
    }


def mapping_index(mapping: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in mapping["mappings"]:
        result[entry["source"]].append(entry)
    return result


def mapping_status(entries: Sequence[dict[str, Any]], expected_target: str) -> dict[str, Any]:
    actual = sorted({target["bone"] for entry in entries for target in entry.get("targets", [])})
    if not actual:
        return {"status": "missing", "actual_targets": []}
    if expected_target in actual:
        return {"status": "exact", "actual_targets": actual}
    return {"status": "different", "actual_targets": actual}


def semantic_segments(
    source_skeleton: dict[str, Any],
    donor_skeleton: dict[str, Any],
    chain: dict[str, Any],
    alignment: dict[str, Any],
    map_by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    source_bones = bone_index(source_skeleton)
    donor_bones = bone_index(donor_skeleton)
    nodes: Sequence[tuple[str, str]] = chain["semantic_nodes"]
    rows: list[dict[str, Any]] = []
    source_axes: list[tuple[float, float, float]] = []
    target_axes: list[tuple[float, float, float]] = []
    geometry: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

    for index, (source_name, target_name) in enumerate(nodes):
        source_bone = source_bones[source_name]
        target_bone = donor_bones[target_name]
        source_head = transform_point(
            source_bone["head_world_rest"],
            alignment["rotation_z_degrees"],
            alignment["translation"],
            alignment.get("scale", 1.0),
        )
        target_head = tuple(target_bone["head_world_rest"])
        if index + 1 < len(nodes):
            next_source, next_target = nodes[index + 1]
            source_end = transform_point(
                source_bones[next_source]["head_world_rest"],
                alignment["rotation_z_degrees"],
                alignment["translation"],
                alignment.get("scale", 1.0),
            )
            target_end = tuple(donor_bones[next_target]["head_world_rest"])
            endpoint_rule = "next semantic joint head"
        elif chain.get("terminal"):
            source_end = transform_point(
                source_bone["tail_world_rest"],
                alignment["rotation_z_degrees"],
                alignment["translation"],
                alignment.get("scale", 1.0),
            )
            target_end = tuple(target_bone["tail_world_rest"])
            endpoint_rule = "terminal raw bone tail"
        else:
            break
        geometry.append((source_head, source_end, target_head, target_end))
        source_axes.append(sub(source_end, source_head))
        target_axes.append(sub(target_end, target_head))
        rows.append(
            {
                "segment": f"{source_name} -> {nodes[index + 1][0] if index + 1 < len(nodes) else 'tail'}",
                "source_start": source_name,
                "source_end": nodes[index + 1][0] if index + 1 < len(nodes) else None,
                "target_start": target_name,
                "target_end": nodes[index + 1][1] if index + 1 < len(nodes) else None,
                "endpoint_rule": endpoint_rule,
                "source_used_by_mesh_count": source_bone.get("used_by_mesh_count"),
                "mapping": mapping_status(map_by_source.get(source_name, []), target_name),
            }
        )

    source_references = frame_references(source_axes, chain["roll_hint"])
    target_references = frame_references(target_axes, chain["roll_hint"])
    for row, points, source_reference, target_reference in zip(rows, geometry, source_references, target_references):
        source_head, source_end, target_head, target_end = points
        source_vector = sub(source_end, source_head)
        target_vector = sub(target_end, target_head)
        source_length = length(source_vector)
        target_length = length(target_vector)
        source_axis = normalize(source_vector)
        target_axis = normalize(target_vector)
        affine = affine_from_frames(
            source_head,
            source_end,
            source_reference["reference"] or choose_least_parallel(source_axis or (1.0, 0.0, 0.0)),
            target_head,
            target_end,
            target_reference["reference"] or choose_least_parallel(target_axis or (1.0, 0.0, 0.0)),
        )
        row.update(
            {
                "source_head_aligned": source_head,
                "source_end_aligned": source_end,
                "target_head": target_head,
                "target_end": target_end,
                "source_length_m": source_length,
                "target_length_m": target_length,
                "source_unit_axis": source_axis,
                "target_unit_axis": target_axis,
                "source_roll_proxy": source_reference,
                "target_roll_proxy": target_reference,
                "initial_head_error_m": distance(source_head, target_head),
                "initial_end_error_m": distance(source_end, target_end),
                "initial_axis_error_deg": angle_deg(source_vector, target_vector),
                "length_ratio_target_over_source": target_length / source_length if source_length > EPS_VECTOR else None,
                "segment_affine": affine,
            }
        )
    return rows


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: Iterable[float]) -> dict[str, float] | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return {
        "count": len(finite),
        "min": min(finite),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "rms": math.sqrt(statistics.fmean(value * value for value in finite)),
        "p95": percentile(finite, 0.95),
        "max": max(finite),
    }


def aggregate_segments(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid_affines = [row["segment_affine"] for row in rows if row["segment_affine"].get("valid")]
    joint_continuity: list[float] = []
    for previous, following in zip(rows, rows[1:]):
        previous_affine = previous["segment_affine"]
        following_affine = following["segment_affine"]
        if not previous_affine.get("valid") or not following_affine.get("valid"):
            continue
        previous_end = add(
            matvec(previous_affine["linear_3x3"], previous["source_end_aligned"]),
            previous_affine["translation"],
        )
        following_head = add(
            matvec(following_affine["linear_3x3"], following["source_head_aligned"]),
            following_affine["translation"],
        )
        joint_continuity.append(distance(previous_end, following_head))
    mapping_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        mapping_counts[row["mapping"]["status"]] += 1
    return {
        "initial_head_error_m": stats(row["initial_head_error_m"] for row in rows),
        "initial_end_error_m": stats(row["initial_end_error_m"] for row in rows),
        "initial_axis_error_deg": stats(row["initial_axis_error_deg"] for row in rows),
        "initial_abs_roll_delta_deg": stats(abs(item["roll_delta_deg"]) for item in valid_affines),
        "length_ratio_target_over_source": stats(row["length_ratio_target_over_source"] for row in rows),
        "affine_condition_number": stats(item["condition_number"] for item in valid_affines),
        "ideal_affine_head_residual_m": stats(item["verification"]["head_residual_m"] for item in valid_affines),
        "ideal_affine_end_residual_m": stats(item["verification"]["end_residual_m"] for item in valid_affines),
        "ideal_affine_joint_continuity_m": stats(joint_continuity),
        "mapping_status_counts": dict(sorted(mapping_counts.items())),
        "mapping_gaps": [
            {
                "source": row["source_start"],
                "expected_target": row["target_start"],
                "status": row["mapping"]["status"],
                "actual_targets": row["mapping"]["actual_targets"],
                "source_used_by_mesh_count": row["source_used_by_mesh_count"],
            }
            for row in rows
            if row["mapping"]["status"] != "exact"
        ],
    }


def collect_target_groups(retarget_report: dict[str, Any]) -> list[str]:
    groups: set[str] = set()
    for obj in retarget_report.get("objects", []):
        groups.update(obj.get("weights", {}).get("target_group_names", []))
    return sorted(groups)


def downstream_validation_snapshot(path: Path) -> dict[str, Any]:
    validation = load_json(path)
    snapshot: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256(path),
        "diagnostic_gates": validation.get("diagnostic_gates", {}),
        "neck": {},
        "joints": {},
    }
    for state in ("rest", "pose"):
        neck = validation.get("metrics", {}).get(state, {}).get("neck", {})
        combined = neck.get("boundary_gap", {}).get("combined", {})
        snapshot["neck"][state] = {
            "boundary_p95_mm": combined.get("p95_mm"),
            "boundary_max_mm": combined.get("max_mm"),
        }
    for joint_name, joint in validation.get("metrics", {}).get("joint_metrics", {}).items():
        snapshot["joints"][joint_name] = {
            state: {
                "skin_boundary_to_garment_surface_p95_mm": joint.get(state, {})
                .get("skin_boundary_to_garment_surface", {})
                .get("p95_mm"),
                "skin_edge_stretch_p95": joint.get("deformation", {})
                .get("skin", {})
                .get("edge_symmetric_stretch", {})
                .get("p95"),
                "garment_edge_stretch_p95": joint.get("deformation", {})
                .get("garment", {})
                .get("edge_symmetric_stretch", {})
                .get("p95"),
            }
            for state in ("rest", "pose")
        }
    return snapshot


def auxiliary_controls(
    source_skeleton: dict[str, Any],
    donor_skeleton: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    source_bones = bone_index(source_skeleton)
    donor_bones = bone_index(donor_skeleton)
    alignment = mapping["alignment"]
    rows: list[dict[str, Any]] = []
    by_target: dict[str, list[str]] = defaultdict(list)
    for entry in mapping["mappings"]:
        target_names = [target["bone"] for target in entry.get("targets", [])]
        source_name = entry["source"]
        is_auxiliary = source_name.startswith("Bip") and (
            any("TWIST" in target or "Corrective" in target for target in target_names)
            or any(token in source_name for token in ("Twist", "ty_", "tz_", "bend_jnt"))
        )
        if not is_auxiliary or source_name not in source_bones:
            continue
        source_bone = source_bones[source_name]
        source_head = transform_point(
            source_bone["head_world_rest"],
            alignment["rotation_z_degrees"],
            alignment["translation"],
            alignment.get("scale", 1.0),
        )
        for target in target_names:
            if target not in donor_bones:
                continue
            target_bone = donor_bones[target]
            target_head = tuple(target_bone["head_world_rest"])
            by_target[target].append(source_name)
            rows.append(
                {
                    "source": source_name,
                    "target": target,
                    "chain": entry.get("chain"),
                    "semantic_family": entry.get("semantic_family"),
                    "source_parent": source_bone.get("parent"),
                    "target_parent": target_bone.get("parent"),
                    "source_head_aligned": source_head,
                    "target_head": target_head,
                    "initial_head_error_m": distance(source_head, target_head),
                    "source_used_by_mesh_count": source_bone.get("used_by_mesh_count"),
                }
            )
    duplicate_targets = {
        target: sorted(sources)
        for target, sources in sorted(by_target.items())
        if len(set(sources)) > 1
    }
    return {
        "count": len(rows),
        "head_error_m": stats(row["initial_head_error_m"] for row in rows),
        "duplicate_target_controls": duplicate_targets,
        "controls": sorted(rows, key=lambda row: (row["target"], row["source"])),
        "calibration_rule": (
            "Do not fit auxiliary raw tails. Project each source control head onto its owning core anatomical segment, "
            "transfer normalized arc coordinate and radial offset through that segment affine, then inherit the calibrated "
            "roll frame. Corrective plus/minus controls that merge to one FH6 bone remain weight-routing inputs, not "
            "independent affine anchors."
        ),
    }


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 9)
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    if isinstance(value, tuple):
        return [round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    return value


def build_report(source_path: Path) -> dict[str, Any]:
    source_skeleton = load_json(source_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only REST segment/frame audit for the Si FBX Display retarget.",
        "method_constraints": {
            "matrix_local_consumed": False,
            "geometry_fields": ["head_world_rest", "tail_world_rest", "parent"],
            "source_authority": "native FBX REST skeleton",
            "units": "meters",
            "raw_tail_warning": (
                "Raw bone tails are orientation/display handles and frequently do not coincide with the next anatomical joint. "
                "Core affine lengths and axes therefore use consecutive semantic bone heads; terminal raw tails are used only "
                "when no successor exists."
            ),
        },
        "inputs": {
            "source_skeleton": {"path": str(source_path), "sha256": sha256(source_path)},
            "containers": {},
        },
        "recommended_algorithm": {
            "frame_construction": [
                "Apply only the declared Rz/global-scale/translation to source REST points, then discard global fit as a final warp.",
                "Build anatomical joint polylines from bone heads. Use the next semantic joint head as a segment endpoint; retain raw tail only for terminal diagnostics.",
                "Normalize each segment to obtain its primary axis. Obtain roll proxy from a projected anatomical world hint, then parallel transport it; fall back to bend direction and finally the least-parallel canonical axis.",
                "Resolve 180-degree axis opposition with the roll proxy so the swing is deterministic. Enforce reference sign continuity and left/right mirror consistency.",
                "Construct a right-handed frame [axis, roll-reference, binormal] directly from vectors. No Blender bone matrix is reused.",
            ],
            "segment_affine": (
                "For source frame Fs and donor frame Ft, use A = Ft * diag(Lt/Ls, 1, 1) * transpose(Fs), "
                "b = target_head - A*source_head. This maps both semantic endpoints exactly while preserving radial scale. "
                "Blend adjacent affines with smoothstep/dual-quaternion rotation blending in joint zones; never linearly blend full affine matrices."
            ),
            "inserted_nodes": (
                "Resample by cumulative anatomical arc length where donor topology inserts Neck1 or finger Meta bones. "
                "Treat those nodes as interpolation knots, not reasons to stretch one source raw bone tail."
            ),
            "twist_corrective": (
                "Project source twist/corrective heads into the owning calibrated core segment frame. Transfer normalized axial coordinate "
                "and radial offset, inherit roll, and exclude merged plus/minus controls from the core anchor solve."
            ),
            "seams": (
                "Use a shared calibrated frame and identical blended weights for both sides of neck, wrist, and ankle seam rings. "
                "Solve topology bridge correspondence separately from skeleton fitting."
            ),
            "degenerate_handling": [
                "Reject an anatomical segment shorter than 0.5 mm as a scale anchor; borrow the closest non-degenerate parent/successor axis and mark the fallback.",
                "If the anatomical hint projection is below sin(3 deg), parallel transport the preceding roll reference; use bend projection only when transport is unavailable.",
                "If an affine condition number exceeds the family gate, resample the chain or add a manual landmark. Do not silently clamp the final transform.",
                "Require positive determinant; an axis reflection must be fixed in the source global alignment, never hidden inside a segment affine.",
            ],
        },
        "gates": GATES,
        "containers": {},
        "blocking_findings": [],
    }
    if VALIDATION_REPORT.exists():
        report["downstream_validation_snapshot"] = downstream_validation_snapshot(VALIDATION_REPORT)

    for container_name, config in CONTAINERS.items():
        donor_skeleton = load_json(config["skeleton"])
        mapping = load_json(config["mapping"])
        retarget_report = load_json(config["retarget_report"])
        map_by_source = mapping_index(mapping)
        target_groups = collect_target_groups(retarget_report)
        container: dict[str, Any] = {
            "donor": mapping["donor"]["container"],
            "required_families": sorted(config["required_families"]),
            "alignment": mapping["alignment"],
            "current_retarget_evidence": {
                "report": str(config["retarget_report"]),
                "report_sha256": sha256(config["retarget_report"]),
                "reported_rest_rule": [
                    obj.get("conform", {}).get("rest_rule")
                    for obj in retarget_report.get("objects", [])
                    if obj.get("conform", {}).get("rest_rule")
                ],
                "target_weight_groups": target_groups,
            },
            "chains": {},
            "auxiliary_controls": auxiliary_controls(source_skeleton, donor_skeleton, mapping),
        }
        report["inputs"]["containers"][container_name] = {
            "donor_skeleton": {"path": str(config["skeleton"]), "sha256": sha256(config["skeleton"])},
            "mapping": {"path": str(config["mapping"]), "sha256": sha256(config["mapping"])},
            "retarget_report": {"path": str(config["retarget_report"]), "sha256": sha256(config["retarget_report"])},
        }
        for chain in CHAIN_DEFS:
            source_inventory = chain_inventory(source_skeleton, chain["source_nodes"], chain["roll_hint"], mapping["alignment"])
            donor_inventory = chain_inventory(donor_skeleton, chain["target_nodes"], chain["roll_hint"], None)
            rows = semantic_segments(source_skeleton, donor_skeleton, chain, mapping["alignment"], map_by_source)
            aggregate = aggregate_segments(rows)
            required = chain["family"] in config["required_families"]
            chain_report = {
                "family": chain["family"],
                "required_for_container": required,
                "structural_note": chain.get("structural_note"),
                "source_inventory": source_inventory,
                "donor_inventory": donor_inventory,
                "semantic_segment_pairs": rows,
                "aggregate": aggregate,
                "recommended_gate": GATES[chain["family"]],
            }
            container["chains"][chain["name"]] = chain_report
            if required and aggregate["mapping_gaps"]:
                report["blocking_findings"].append(
                    {
                        "container": container_name,
                        "chain": chain["name"],
                        "type": "core_frame_correspondence_gap",
                        "gaps": aggregate["mapping_gaps"],
                        "note": (
                            "Add anchor-only correspondences even when the source core bone has no direct vertex weights; "
                            "weight routing and frame calibration are separate contracts."
                        ),
                    }
                )
        main_limb_targets = {
            "LeftUpLeg",
            "LeftLeg",
            "RightUpLeg",
            "RightLeg",
            "LeftArm",
            "LeftForeArm",
            "RightArm",
            "RightForeArm",
        }
        container["current_retarget_evidence"]["absent_main_limb_target_groups"] = sorted(main_limb_targets - set(target_groups))
        report["containers"][container_name] = container
    return round_floats(report)


def markdown_stats(item: dict[str, Any] | None, multiplier: float = 1.0) -> str:
    if not item:
        return "n/a"
    return f"rms {item['rms'] * multiplier:.3f}, p95 {item['p95'] * multiplier:.3f}, max {item['max'] * multiplier:.3f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Si Display segment-frame audit v001",
        "",
        "This is a read-only audit. It derives all frames from REST heads/tails and hierarchy; `matrix_local` is not consumed.",
        "",
        "## Executive findings",
        "",
        "- The current v004 retarget reports `mapped source/donor REST head translation with target blend`; it does not calibrate segment axis, length, or roll.",
        "- Raw bone tails are not reliable anatomical endpoints. The audit uses consecutive semantic joint heads for core segment length and direction.",
        "- FH6 has an inserted `Neck1` and non-thumb finger Meta nodes. They require chain arc-length resampling.",
        "- Outfit v004 lacks anchor correspondences for the unweighted source thigh/calf main bones. Add anchor-only pairs for frame solving without inventing vertex weights.",
        "",
        "## Observed pre-fit residuals",
        "",
        "| Container | Chain | Head error mm | End error mm | Axis error deg | Roll delta deg | Length ratio | Mapping gaps |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for container_name, container in report["containers"].items():
        for chain_name, chain in container["chains"].items():
            aggregate = chain["aggregate"]
            lines.append(
                "| {container} | {chain} | {head} | {end} | {axis} | {roll} | {ratio} | {gaps} |".format(
                    container=container_name,
                    chain=chain_name,
                    head=markdown_stats(aggregate["initial_head_error_m"], 1000.0),
                    end=markdown_stats(aggregate["initial_end_error_m"], 1000.0),
                    axis=markdown_stats(aggregate["initial_axis_error_deg"]),
                    roll=markdown_stats(aggregate["initial_abs_roll_delta_deg"]),
                    ratio=markdown_stats(aggregate["length_ratio_target_over_source"]),
                    gaps=len(aggregate["mapping_gaps"]),
                )
            )

    lines.extend(
        [
            "",
            "These are diagnostic pre-fit errors after the global Rz/translation. They are not acceptance failures by themselves; they quantify why head translation cannot be the final warp.",
            "",
            "## Existing LOD0 downstream evidence",
            "",
        ]
    )
    downstream = report.get("downstream_validation_snapshot")
    if downstream:
        rest_neck = downstream["neck"]["rest"]
        pose_neck = downstream["neck"]["pose"]
        lines.extend(
            [
                f"- Neck boundary: REST p95 {rest_neck['boundary_p95_mm']:.3f} mm / max {rest_neck['boundary_max_mm']:.3f} mm; "
                f"pose p95 {pose_neck['boundary_p95_mm']:.3f} mm / max {pose_neck['boundary_max_mm']:.3f} mm.",
            ]
        )
        for joint_name, joint in downstream["joints"].items():
            lines.append(
                f"- `{joint_name}` skin-to-garment surface p95: REST "
                f"{joint['rest']['skin_boundary_to_garment_surface_p95_mm']:.3f} mm, pose "
                f"{joint['pose']['skin_boundary_to_garment_surface_p95_mm']:.3f} mm; "
                f"skin/garment edge-stretch p95 {joint['pose']['skin_edge_stretch_p95']:.3f}/"
                f"{joint['pose']['garment_edge_stretch_p95']:.3f}."
            )
        lines.extend(
            [
                "",
                "The wrist failure is already present in REST while edge stretch remains moderate, which supports a frame/segment placement fault rather than pose-only weight explosion. The current ankle pass covers only the existing Foot pose and does not waive knee/ToeBase pressure tests.",
                "",
            ]
        )
    else:
        lines.extend(["- Current LOD0 validation report was not present.", ""])
    lines.extend(
        [
            "## Blocking correspondence gaps",
            "",
        ]
    )
    if report["blocking_findings"]:
        for finding in report["blocking_findings"]:
            for gap in finding["gaps"]:
                lines.append(
                    f"- `{finding['container']}/{finding['chain']}`: `{gap['source']}` -> `{gap['expected_target']}` is {gap['status']} "
                    f"(actual: {gap['actual_targets']}, source mesh-use count: {gap['source_used_by_mesh_count']})."
                )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "The thigh/calf items with mesh-use count zero are still required as anchor-only frame correspondences. They should not create artificial vertex groups.",
            "",
            "## Explicit calibration contract",
            "",
            "1. Apply the declared source global alignment to REST points only.",
            "2. Build anatomical polylines from consecutive semantic bone heads. Keep raw tails only for terminal diagnostics.",
            "3. Derive axis and roll proxy geometrically. Use an anatomical world hint, parallel transport, bend projection, then a least-parallel canonical fallback.",
            "4. Construct `Fs=[axis, roll, binormal]` and `Ft` without any Blender matrix. Use `A=Ft*diag(Lt/Ls,1,1)*Fs^T`, `b=Ht-A*Hs`.",
            "5. Resample inserted `Neck1` and finger Meta nodes by cumulative arc length. Map twist/corrective controls through the owning core segment frame.",
            "6. Blend rotations with quaternion/dual-quaternion interpolation and axial/radial scale separately near joints; do not linearly blend full affine matrices.",
            "7. Use identical calibrated transforms and weights on both sides of neck, wrist, and ankle seam rings.",
            "",
            "## Quantitative gates",
            "",
            "| Family | Anchor RMS | Anchor max | Axis | Roll | Joint continuity | Condition number | Downstream gate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for family, gate in report["gates"].items():
        downstream = []
        for key, value in gate.items():
            if key.startswith("downstream_"):
                downstream.append(f"{key} <= {value}")
        lines.append(
            f"| {family} | <= {gate['affine_anchor_rms_mm_max']} mm | <= {gate['affine_anchor_max_mm_max']} mm | "
            f"<= {gate['axis_residual_deg_max']} deg | <= {gate['roll_residual_deg_max']} deg | "
            f"<= {gate['joint_continuity_mm_max']} mm | <= {gate['condition_number_max']} | {'; '.join(downstream) or 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "Ideal per-segment endpoint residual is numerical zero by construction. Practical thresholds above apply after chain resampling, joint-zone blending, and seam constraints.",
            "",
            "## Implementation order",
            "",
            "1. Add a frame-only core correspondence table, including thigh/calf anchors and inserted-node metadata.",
            "2. Implement the vector-derived frame and segment-affine solver as a pure testable module.",
            "3. Validate skeleton-only endpoints/axes/roll before touching mesh vertices.",
            "4. Apply to torso/neck, then arms/wrists/fingers, then legs/ankles/toes; route twist/corrective controls last within each chain.",
            "5. Re-run the existing LOD0 neck/wrist/ankle surface and pose gates, then propagate the same calibrated contract to LOD1-LOD3.",
            "",
            f"Full machine-readable segment inventories and explicit affine matrices: `{path.with_suffix('.json')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-skeleton", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.source_skeleton.resolve())
    json_path = output_dir / "si-display-segment-frame-audit-v001.json"
    markdown_path = output_dir / "si-display-segment-frame-audit-v001.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_markdown(report, markdown_path)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "blocking_findings": len(report["blocking_findings"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
