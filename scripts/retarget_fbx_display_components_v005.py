#!/usr/bin/env python3
"""Run the FBX Display batch retarget with calibrated landmark segments.

This wrapper intentionally leaves the v004 retarget scripts untouched.  It
replaces only their geometry-warp callback, using anatomical landmark segments
and explicit forward/up roll references instead of Blender bone roll matrices.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import retarget_fbx_display_components as batch
from retarget_fh6_component import clean_assignments


@dataclass(frozen=True)
class SegmentDefinition:
    source_start: str
    source_end: str
    donor_start: str
    donor_end: str
    roll_reference: str = "forward"


@dataclass
class SegmentTransform:
    name: str
    source_start: Vector
    source_axes: tuple[Vector, Vector, Vector]
    donor_start: Vector
    donor_axes: tuple[Vector, Vector, Vector]
    source_length: float
    longitudinal_scale: float
    transverse_scale: float = 1.0
    depth_scale: float = 1.0

    def apply(self, point: Vector) -> Vector:
        offset = point - self.source_start
        longitudinal = offset.dot(self.source_axes[0]) * self.longitudinal_scale
        transverse = offset.dot(self.source_axes[1]) * self.transverse_scale
        depth = offset.dot(self.source_axes[2]) * self.depth_scale
        return (
            self.donor_start
            + self.donor_axes[0] * longitudinal
            + self.donor_axes[1] * transverse
            + self.donor_axes[2] * depth
        )


SEGMENTS: dict[str, SegmentDefinition] = {
    "pelvis_spine": SegmentDefinition("Bip001_Pelvis", "Bip001_Spine", "Hips", "Spine"),
    "spine_1": SegmentDefinition("Bip001_Spine", "Bip001_Spine1", "Spine", "Spine1"),
    "spine_2": SegmentDefinition("Bip001_Spine1", "Bip001_Spine2", "Spine1", "Spine2"),
    "upper_torso": SegmentDefinition("Bip001_Spine2", "Bip001_Neck", "Spine2", "Neck"),
    "neck_head": SegmentDefinition("Bip001_Neck", "Bip001_Head", "Neck", "Head"),
    "left_clavicle": SegmentDefinition("Bip001_L_Clavicle", "Bip001_L_UpperArm", "LeftShoulder", "LeftArm"),
    "left_upper_arm": SegmentDefinition("Bip001_L_UpperArm", "Bip001_L_Forearm", "LeftArm", "LeftForeArm"),
    "left_forearm": SegmentDefinition("Bip001_L_Forearm", "Bip001_L_Hand", "LeftForeArm", "LeftHand"),
    "left_hand": SegmentDefinition("Bip001_L_Hand", "Bip001_L_Finger2", "LeftHand", "LeftMiddle1"),
    "right_clavicle": SegmentDefinition("Bip001_R_Clavicle", "Bip001_R_UpperArm", "RightShoulder", "RightArm"),
    "right_upper_arm": SegmentDefinition("Bip001_R_UpperArm", "Bip001_R_Forearm", "RightArm", "RightForeArm"),
    "right_forearm": SegmentDefinition("Bip001_R_Forearm", "Bip001_R_Hand", "RightForeArm", "RightHand"),
    "right_hand": SegmentDefinition("Bip001_R_Hand", "Bip001_R_Finger2", "RightHand", "RightMiddle1"),
    "left_thigh": SegmentDefinition("Bip001_L_Thigh", "Bip001_L_Calf", "LeftUpLeg", "LeftLeg"),
    "left_calf": SegmentDefinition("Bip001_L_Calf", "Bip001_L_Foot", "LeftLeg", "LeftFoot"),
    "left_foot": SegmentDefinition("Bip001_L_Foot", "Bip001_L_Toe0", "LeftFoot", "LeftToeBase", "up"),
    "right_thigh": SegmentDefinition("Bip001_R_Thigh", "Bip001_R_Calf", "RightUpLeg", "RightLeg"),
    "right_calf": SegmentDefinition("Bip001_R_Calf", "Bip001_R_Foot", "RightLeg", "RightFoot"),
    "right_foot": SegmentDefinition("Bip001_R_Foot", "Bip001_R_Toe0", "RightFoot", "RightToeBase", "up"),
}


EXACT_SEGMENTS = {
    "Bip001_Pelvis": "pelvis_spine",
    "Bip001_Spine": "spine_1",
    "Bip001_Spine1": "spine_2",
    "Bip001_Spine2": "upper_torso",
    "Bip001_Neck": "neck_head",
    "Bip001_Head": "head_landmarks",
    "face_Head": "head_landmarks",
    "Bip001_L_Clavicle": "left_clavicle",
    "Bip001_L_UpperArm": "left_upper_arm",
    "Bip001_L_Forearm": "left_forearm",
    "Bip001_L_Hand": "left_hand",
    "Bip001_R_Clavicle": "right_clavicle",
    "Bip001_R_UpperArm": "right_upper_arm",
    "Bip001_R_Forearm": "right_forearm",
    "Bip001_R_Hand": "right_hand",
    "Bip001_L_Thigh": "left_thigh",
    "Bip001_L_Calf": "left_calf",
    "Bip001_L_Foot": "left_foot",
    "Bip001_L_Toe0": "left_foot",
    "Bip001_R_Thigh": "right_thigh",
    "Bip001_R_Calf": "right_calf",
    "Bip001_R_Foot": "right_foot",
    "Bip001_R_Toe0": "right_foot",
}


CHAIN_CANDIDATES = {
    "head_face": ("head_landmarks", "neck_head", "upper_torso"),
    "left_eye": ("head_landmarks",),
    "right_eye": ("head_landmarks",),
    "left_arm": ("left_clavicle", "left_upper_arm", "left_forearm", "left_hand"),
    "right_arm": ("right_clavicle", "right_upper_arm", "right_forearm", "right_hand"),
    "left_leg": ("left_thigh", "left_calf", "left_foot"),
    "right_leg": ("right_thigh", "right_calf", "right_foot"),
    "torso": tuple(SEGMENTS),
}


def world_head(armature: bpy.types.Object, name: str) -> Vector:
    bone = armature.data.bones.get(name)
    if bone is None:
        raise RuntimeError(f"{armature.name} is missing landmark bone {name!r}")
    return armature.matrix_world @ bone.head_local


def calibrated_axes(direction: Vector, reference_name: str) -> tuple[Vector, Vector, Vector]:
    longitudinal = direction.normalized()
    references = {
        "forward": (Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), Vector((1.0, 0.0, 0.0))),
        "up": (Vector((0.0, 0.0, 1.0)), Vector((0.0, 1.0, 0.0)), Vector((1.0, 0.0, 0.0))),
    }[reference_name]
    transverse = None
    for reference in references:
        projected = reference - longitudinal * reference.dot(longitudinal)
        if projected.length > 1e-5:
            transverse = projected.normalized()
            break
    if transverse is None:
        raise RuntimeError("Could not construct an anatomical roll reference")
    depth = longitudinal.cross(transverse).normalized()
    transverse = depth.cross(longitudinal).normalized()
    return longitudinal, transverse, depth


def build_head_landmark_transform(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    alignment: Matrix,
) -> SegmentTransform:
    source_left = alignment @ world_head(source_armature, "faceLfIrisJoint")
    source_right = alignment @ world_head(source_armature, "faceRtIrisJoint")
    donor_left = world_head(donor_armature, "LeftEye")
    donor_right = world_head(donor_armature, "RightEye")
    source_origin = (source_left + source_right) * 0.5
    donor_origin = (donor_left + donor_right) * 0.5
    source_width = (source_left - source_right).length
    donor_width = (donor_left - donor_right).length
    if source_width <= 1e-6 or donor_width <= 1e-6:
        raise RuntimeError("Eye landmarks cannot define a stable Head scale")
    # Rz(180) already establishes the common anatomical axes.  Keeping those
    # axes explicit avoids inheriting incompatible FBX/FH6 bone rolls.
    axes = (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))
    return SegmentTransform(
        name="head_landmarks",
        source_start=source_origin,
        source_axes=axes,
        donor_start=donor_origin,
        donor_axes=axes,
        source_length=source_width,
        longitudinal_scale=donor_width / source_width,
        transverse_scale=donor_width / source_width,
        depth_scale=donor_width / source_width,
    )


def build_segment_transforms(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    alignment: Matrix,
) -> dict[str, SegmentTransform]:
    result: dict[str, SegmentTransform] = {}
    for name, definition in SEGMENTS.items():
        source_start = alignment @ world_head(source_armature, definition.source_start)
        source_end = alignment @ world_head(source_armature, definition.source_end)
        donor_start = world_head(donor_armature, definition.donor_start)
        donor_end = world_head(donor_armature, definition.donor_end)
        source_direction = source_end - source_start
        donor_direction = donor_end - donor_start
        if source_direction.length <= 1e-6 or donor_direction.length <= 1e-6:
            raise RuntimeError(f"Degenerate anatomical segment {name}")
        result[name] = SegmentTransform(
            name=name,
            source_start=source_start,
            source_axes=calibrated_axes(source_direction, definition.roll_reference),
            donor_start=donor_start,
            donor_axes=calibrated_axes(donor_direction, definition.roll_reference),
            source_length=source_direction.length,
            longitudinal_scale=donor_direction.length / source_direction.length,
        )
    result["head_landmarks"] = build_head_landmark_transform(source_armature, donor_armature, alignment)
    return result


def point_segment_distance(point: Vector, transform: SegmentTransform) -> float:
    source_length = transform.source_length
    direction = transform.source_axes[0]
    offset = point - transform.source_start
    distance_along = max(0.0, min(source_length, offset.dot(direction)))
    return (point - (transform.source_start + direction * distance_along)).length


def source_side(name: str) -> str | None:
    folded = name.casefold()
    if folded.startswith(("bip001_l", "bip_l", "l_")) or "facelf" in folded or "eyelf" in folded:
        return "left"
    if folded.startswith(("bip001_r", "bip_r", "r_")) or "facert" in folded or "eyert" in folded:
        return "right"
    return None


def token_segment(source: str) -> str | None:
    folded = source.casefold()
    side = source_side(source)
    if any(token in folded for token in ("face", "eye", "iris", "pupil", "brow", "lip", "cheek", "jaw", "tooth", "tongue", "ear", "hair", "head")):
        return "head_landmarks"
    if side is not None:
        prefix = "left" if side == "left" else "right"
        if any(token in folded for token in ("finger", "thumb", "index", "middle", "ring", "pinky")):
            return f"{prefix}_hand"
        if "hand_ty" in folded or "wrist" in folded:
            return f"{prefix}_forearm"
        if "hand" in folded:
            return f"{prefix}_hand"
        if "forearm" in folded:
            return f"{prefix}_forearm"
        if any(token in folded for token in ("upperarm", "sleeve")):
            return f"{prefix}_upper_arm"
        if any(token in folded for token in ("clavicle", "shoulder")):
            return f"{prefix}_clavicle"
        if any(token in folded for token in ("foot", "toe")):
            return f"{prefix}_foot"
        if "calf" in folded:
            return f"{prefix}_calf"
        if "thigh" in folded:
            return f"{prefix}_thigh"
    if "pelvis" in folded or any(token in folded for token in ("skirt", "dress", "tail")):
        return "pelvis_spine"
    if "spine2" in folded or any(token in folded for token in ("breast", "collar", "necklace")):
        return "upper_torso"
    if "spine1" in folded:
        return "spine_2"
    if "spine" in folded:
        return "spine_1"
    return None


def select_segment(
    mesh_role: str,
    source: str,
    entry: dict[str, object] | None,
    source_armature: bpy.types.Object,
    alignment: Matrix,
    transforms: dict[str, SegmentTransform],
) -> str:
    if mesh_role in {"head", "hair"}:
        return "head_landmarks"
    exact = EXACT_SEGMENTS.get(source)
    if exact is not None:
        return exact
    token = token_segment(source)
    if token is not None:
        return token
    source_bone = source_armature.data.bones.get(source)
    if source_bone is None:
        return "pelvis_spine"
    point = alignment @ (source_armature.matrix_world @ source_bone.head_local)
    chain = str(entry.get("chain", "torso")) if entry else "torso"
    candidates = CHAIN_CANDIDATES.get(chain, CHAIN_CANDIDATES["torso"])
    return min(
        candidates,
        key=lambda name: (
            point_segment_distance(point, transforms[name]),
            name,
        ),
    )


def apply_landmark_segment_warp(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    source_weights: list[list[tuple[str, float]]],
    target_map: dict[str, list[tuple[str, float]]],
    mapping_entries: dict[str, dict[str, object]],
    transform: Matrix,
    threshold: float,
) -> dict[str, object]:
    transforms = build_segment_transforms(source_armature, donor_armature, transform)
    role = str(mesh.get("source_role", "")).casefold()
    inverse_mesh_world = mesh.matrix_world.inverted()
    segment_cache: dict[str, str] = {}
    segment_assignments: Counter[str] = Counter()
    displacements: list[float] = []
    for vertex, assignments in zip(mesh.data.vertices, source_weights):
        cleaned = clean_assignments(assignments, target_map, threshold)
        source_position = transform @ (mesh.matrix_world @ vertex.co)
        warped = Vector()
        total = 0.0
        for source_name, _target_name, weight in cleaned:
            segment_name = segment_cache.get(source_name)
            if segment_name is None:
                segment_name = select_segment(
                    role,
                    source_name,
                    mapping_entries.get(source_name),
                    source_armature,
                    transform,
                    transforms,
                )
                segment_cache[source_name] = segment_name
            warped += transforms[segment_name].apply(source_position) * weight
            segment_assignments[segment_name] += 1
            total += weight
        if total <= 0.0:
            raise RuntimeError(f"No calibrated segment available for vertex {vertex.index}")
        warped /= total
        displacements.append((warped - source_position).length)
        vertex.co = inverse_mesh_world @ warped
    mesh.data.update()
    ordered = sorted(displacements)
    p95 = ordered[min(len(ordered) - 1, math.floor((len(ordered) - 1) * 0.95))] if ordered else 0.0
    return {
        "mode": "landmark-segment-affine-v005",
        "vertices": len(displacements),
        "mean_displacement_m": sum(displacements) / len(displacements) if displacements else 0.0,
        "p95_displacement_m": p95,
        "max_displacement_m": max(displacements) if displacements else 0.0,
        "segment_assignments": dict(sorted(segment_assignments.items())),
        "source_group_segments": dict(sorted(segment_cache.items())),
        "segment_scales": {
            name: round(segment.longitudinal_scale, 9)
            for name, segment in sorted(transforms.items())
        },
        "axis_calibration": "landmark direction plus projected global forward/up reference; no matrix_local roll reuse",
        "seam_constraints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "face_neck"],
        "rest_rule": "core-chain landmark segment affine with longitudinal scale and preserved transverse dimensions",
    }


def main() -> None:
    batch.apply_chain_rest_warp = apply_landmark_segment_warp
    batch.main()


if __name__ == "__main__":
    main()
