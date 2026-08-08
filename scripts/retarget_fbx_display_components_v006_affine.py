#!/usr/bin/env python3
"""FBX-first Display retarget using audited vector-derived segment affines.

This is a new wrapper; v004 and v005 remain immutable baselines.  It replaces
the batch geometry callback with the pure-Python ``si_segment_affine`` solver,
loads frame-only anchors without creating weight groups, arc-resamples FH6's
inserted Neck1/finger Meta knots, and preserves Shape Key deltas.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import retarget_fbx_display_components as batch
import retarget_fbx_display_components_v005 as v005
import si_segment_affine as segment
from retarget_fh6_component import clean_assignments


WORKSPACE = SCRIPT_DIR.parent
ANCHOR_METADATA = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-05-validation-v001"
    / "segment-frame-audit"
    / "si-display-frame-anchors-v001.json"
)

CONDITION_GATES = {
    "spine_neck_head": 2.25,
    "arms": 2.50,
    "legs": 2.50,
    "fingers": 2.50,
    "head": 2.25,
}


@dataclass(frozen=True)
class ChainSpec:
    name: str
    family: str
    source_nodes: tuple[str, ...]
    target_nodes: tuple[str, ...]
    roll_hint: segment.Vec3
    inserted_target_indices: tuple[int, ...] = ()


CORE_CHAINS = (
    ChainSpec(
        "spine_core",
        "spine_neck_head",
        ("Bip001_Pelvis", "Bip001_Spine", "Bip001_Spine1", "Bip001_Spine2", "Bip001_Neck"),
        ("Hips", "Spine", "Spine1", "Spine2", "Neck"),
        (1.0, 0.0, 0.0),
    ),
    ChainSpec(
        "neck_head",
        "spine_neck_head",
        ("Bip001_Neck", "Bip001_Head"),
        ("Neck", "Neck1", "Head"),
        (1.0, 0.0, 0.0),
        (1,),
    ),
    ChainSpec(
        "left_arm",
        "arms",
        ("Bip001_L_Clavicle", "Bip001_L_UpperArm", "Bip001_L_Forearm", "Bip001_L_Hand", "Bip001_L_Finger2"),
        ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftMiddle1"),
        (0.0, 1.0, 0.0),
    ),
    ChainSpec(
        "right_arm",
        "arms",
        ("Bip001_R_Clavicle", "Bip001_R_UpperArm", "Bip001_R_Forearm", "Bip001_R_Hand", "Bip001_R_Finger2"),
        ("RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightMiddle1"),
        (0.0, 1.0, 0.0),
    ),
    ChainSpec(
        "left_leg",
        "legs",
        ("Bip001_L_Thigh", "Bip001_L_Calf", "Bip001_L_Foot", "Bip001_L_Toe0"),
        ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"),
        (0.0, 1.0, 0.0),
    ),
    ChainSpec(
        "right_leg",
        "legs",
        ("Bip001_R_Thigh", "Bip001_R_Calf", "Bip001_R_Foot", "Bip001_R_Toe0"),
        ("RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"),
        (0.0, 1.0, 0.0),
    ),
)


def finger_specs() -> tuple[ChainSpec, ...]:
    result: list[ChainSpec] = []
    for source_side, target_side, side_name in (("L", "Left", "left"), ("R", "Right", "right")):
        definitions = (
            (0, "Thumb", False),
            (1, "Index", True),
            (2, "Middle", True),
            (3, "Ring", True),
            (4, "Pinky", True),
        )
        for digit, target_digit, inserted_meta in definitions:
            source_nodes = (
                f"Bip001_{source_side}_Hand",
                f"Bip001_{source_side}_Finger{digit}",
                f"Bip001_{source_side}_Finger{digit}1",
                f"Bip001_{source_side}_Finger{digit}2",
            )
            if inserted_meta:
                target_nodes = (
                    f"{target_side}Hand",
                    f"{target_side}{target_digit}Meta",
                    f"{target_side}{target_digit}1",
                    f"{target_side}{target_digit}2",
                    f"{target_side}{target_digit}3",
                )
                inserted = (1,)
            else:
                target_nodes = (
                    f"{target_side}Hand",
                    f"{target_side}{target_digit}1",
                    f"{target_side}{target_digit}2",
                    f"{target_side}{target_digit}3",
                )
                inserted = ()
            result.append(
                ChainSpec(
                    f"{side_name}_{target_digit.casefold()}",
                    "fingers",
                    source_nodes,
                    target_nodes,
                    (0.0, 1.0, 0.0),
                    inserted,
                )
            )
    return tuple(result)


@dataclass(frozen=True)
class AffineTransform:
    name: str
    chain: str
    family: str
    source_start: Vector
    source_end: Vector
    source_fraction_start: float
    source_fraction_end: float
    affine: segment.SegmentAffine
    source_frame: segment.SegmentFrame
    target_frame: segment.SegmentFrame

    @property
    def source_length(self) -> float:
        return (self.source_end - self.source_start).length

    @property
    def source_axes(self) -> tuple[Vector, Vector, Vector]:
        return tuple(Vector(axis) for axis in (self.source_frame.axis, self.source_frame.roll, self.source_frame.binormal))  # type: ignore[return-value]

    def apply(self, point: Vector) -> Vector:
        return Vector(self.affine.apply(tuple(point)))


@dataclass
class TransformRegistry:
    transforms: dict[str, AffineTransform]
    chain_transforms: dict[str, list[AffineTransform]]
    source_groups: dict[str, AffineTransform]
    aliases: dict[str, AffineTransform]
    head: AffineTransform
    anchor_report: list[dict[str, object]]
    inserted_report: list[dict[str, object]]


def world_head(armature: bpy.types.Object, name: str) -> Vector:
    bone = armature.data.bones.get(name)
    if bone is None:
        raise RuntimeError(f"{armature.name} is missing REST landmark {name!r}")
    return armature.matrix_world @ bone.head_local


def aligned_head(armature: bpy.types.Object, name: str, alignment: Matrix) -> Vector:
    return alignment @ world_head(armature, name)


def solve_affine(
    name: str,
    chain: str,
    family: str,
    source_start: Vector,
    source_end: Vector,
    target_start: Vector,
    target_end: Vector,
    source_frame: segment.SegmentFrame,
    target_frame: segment.SegmentFrame,
    source_fraction_start: float,
    source_fraction_end: float,
    radial_scale: float = 1.0,
) -> AffineTransform:
    affine = segment.solve_segment_affine(
        tuple(source_start),
        tuple(source_end),
        source_frame,
        tuple(target_start),
        tuple(target_end),
        target_frame,
        radial_scale=radial_scale,
        max_condition=CONDITION_GATES[family],
    )
    return AffineTransform(
        name=name,
        chain=chain,
        family=family,
        source_start=source_start,
        source_end=source_end,
        source_fraction_start=source_fraction_start,
        source_fraction_end=source_fraction_end,
        affine=affine,
        source_frame=source_frame,
        target_frame=target_frame,
    )


def build_chain(
    spec: ChainSpec,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    alignment: Matrix,
) -> tuple[list[AffineTransform], dict[str, AffineTransform], dict[str, object]]:
    source_points = [aligned_head(source_armature, name, alignment) for name in spec.source_nodes]
    target_points = [world_head(donor_armature, name) for name in spec.target_nodes]
    if spec.inserted_target_indices:
        correspondences = segment.map_target_knots_to_source(
            [tuple(point) for point in source_points],
            [tuple(point) for point in target_points],
            inserted_target_indices=spec.inserted_target_indices,
        )
        sampled_source = [Vector(item.source_sample.point) for item in correspondences]
        fractions = [item.source_sample.fraction for item in correspondences]
    else:
        if len(source_points) != len(target_points):
            raise RuntimeError(f"{spec.name} has unequal knots without inserted-node metadata")
        sampled_source = source_points
        fractions = list(segment.normalized_arc_fractions([tuple(point) for point in source_points]))

    source_frames = segment.derive_chain_frames(
        [tuple(point) for point in sampled_source],
        anatomical_hint=spec.roll_hint,
    )
    target_frames = segment.derive_chain_frames(
        [tuple(point) for point in target_points],
        anatomical_hint=spec.roll_hint,
    )
    transforms: list[AffineTransform] = []
    for index, (source_frame, target_frame) in enumerate(zip(source_frames, target_frames)):
        transforms.append(
            solve_affine(
                f"{spec.name}:{index:02d}:{spec.target_nodes[index]}->{spec.target_nodes[index + 1]}",
                spec.name,
                spec.family,
                sampled_source[index],
                sampled_source[index + 1],
                target_points[index],
                target_points[index + 1],
                source_frame,
                target_frame,
                fractions[index],
                fractions[index + 1],
            )
        )

    source_fractions = segment.normalized_arc_fractions([tuple(point) for point in source_points])
    source_groups: dict[str, AffineTransform] = {}
    for source_name, fraction in zip(spec.source_nodes, source_fractions):
        candidates = [
            transform
            for transform in transforms
            if transform.source_fraction_start <= fraction + 1e-9
        ]
        source_groups[source_name] = candidates[-1] if candidates else transforms[0]
    report = {
        "chain": spec.name,
        "family": spec.family,
        "source_nodes": list(spec.source_nodes),
        "target_nodes": list(spec.target_nodes),
        "inserted_target_indices": list(spec.inserted_target_indices),
        "source_sample_fractions": [round(value, 9) for value in fractions],
        "segments": [
            {
                "name": transform.name,
                "axial_scale": round(transform.affine.axial_scale, 9),
                "radial_scale": round(transform.affine.radial_scale, 9),
                "determinant": round(transform.affine.determinant, 9),
                "condition_number": round(transform.affine.condition_number, 9),
                "source_frame_basis": transform.source_frame.basis,
                "target_frame_basis": transform.target_frame.basis,
                "head_residual_m": segment.distance(transform.affine.apply(tuple(transform.source_start)), tuple(target_points[len(transforms) - len(transforms) + transforms.index(transform)])),
                "end_residual_m": segment.distance(transform.affine.apply(tuple(transform.source_end)), tuple(target_points[transforms.index(transform) + 1])),
            }
            for transform in transforms
        ],
    }
    return transforms, source_groups, report


def build_head_transform(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    alignment: Matrix,
) -> AffineTransform:
    source_left = aligned_head(source_armature, "faceLfIrisJoint", alignment)
    source_right = aligned_head(source_armature, "faceRtIrisJoint", alignment)
    target_left = world_head(donor_armature, "LeftEye")
    target_right = world_head(donor_armature, "RightEye")
    source_frame = segment.derive_segment_frame(tuple(source_left), tuple(source_right), anatomical_hint=(0.0, 1.0, 0.0))
    target_frame = segment.derive_segment_frame(tuple(target_left), tuple(target_right), anatomical_hint=(0.0, 1.0, 0.0))
    uniform_scale = (target_right - target_left).length / (source_right - source_left).length
    return solve_affine(
        "head_landmarks",
        "head_face",
        "head",
        source_left,
        source_right,
        target_left,
        target_right,
        source_frame,
        target_frame,
        0.0,
        1.0,
        radial_scale=uniform_scale,
    )


def container_name(donor_armature: bpy.types.Object) -> str:
    return "head_hair_helmet" if "helmet" in donor_armature.name.casefold() else "body_garment_outfit"


def validate_frame_only_anchors(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    alignment: Matrix,
    container: str,
) -> list[dict[str, object]]:
    metadata = segment.load_frame_anchor_metadata(ANCHOR_METADATA)
    report: list[dict[str, object]] = []
    previous_source_roll: dict[str, segment.Vec3] = {}
    previous_target_roll: dict[str, segment.Vec3] = {}
    for anchor in segment.iter_frame_anchors(metadata, container):
        if anchor.affects_weight_transfer:
            raise RuntimeError(f"Frame-only anchor {anchor.anchor_id} attempted to affect weights")
        source_start = aligned_head(source_armature, anchor.source_start, alignment)
        source_end = aligned_head(source_armature, anchor.source_end, alignment)
        target_start = world_head(donor_armature, anchor.target_start)
        target_end = world_head(donor_armature, anchor.target_end)
        source_frame = segment.derive_segment_frame(
            tuple(source_start),
            tuple(source_end),
            anatomical_hint=anchor.roll_hint,
            previous_roll=previous_source_roll.get(anchor.family),
        )
        target_frame = segment.derive_segment_frame(
            tuple(target_start),
            tuple(target_end),
            anatomical_hint=anchor.roll_hint,
            previous_roll=previous_target_roll.get(anchor.family),
        )
        affine = segment.solve_segment_affine(
            tuple(source_start),
            tuple(source_end),
            source_frame,
            tuple(target_start),
            tuple(target_end),
            target_frame,
            max_condition=CONDITION_GATES[anchor.family],
        )
        previous_source_roll[anchor.family] = source_frame.roll
        previous_target_roll[anchor.family] = target_frame.roll
        report.append({
            "id": anchor.anchor_id,
            "family": anchor.family,
            "affects_weight_transfer": False,
            "determinant": round(affine.determinant, 9),
            "condition_number": round(affine.condition_number, 9),
            "head_residual_m": segment.distance(affine.apply(tuple(source_start)), tuple(target_start)),
            "end_residual_m": segment.distance(affine.apply(tuple(source_end)), tuple(target_end)),
        })
    return report


def alias_table(source_groups: dict[str, AffineTransform], head: AffineTransform) -> dict[str, AffineTransform]:
    aliases = {"head_landmarks": head}
    mapping = {
        "pelvis_spine": "Bip001_Pelvis",
        "spine_1": "Bip001_Spine",
        "spine_2": "Bip001_Spine1",
        "upper_torso": "Bip001_Spine2",
        "neck_head": "Bip001_Neck",
        "left_clavicle": "Bip001_L_Clavicle",
        "left_upper_arm": "Bip001_L_UpperArm",
        "left_forearm": "Bip001_L_Forearm",
        "left_hand": "Bip001_L_Hand",
        "right_clavicle": "Bip001_R_Clavicle",
        "right_upper_arm": "Bip001_R_UpperArm",
        "right_forearm": "Bip001_R_Forearm",
        "right_hand": "Bip001_R_Hand",
        "left_thigh": "Bip001_L_Thigh",
        "left_calf": "Bip001_L_Calf",
        "left_foot": "Bip001_L_Foot",
        "right_thigh": "Bip001_R_Thigh",
        "right_calf": "Bip001_R_Calf",
        "right_foot": "Bip001_R_Foot",
    }
    for alias, source_name in mapping.items():
        if source_name in source_groups:
            aliases[alias] = source_groups[source_name]
    return aliases


def build_registry(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    alignment: Matrix,
) -> TransformRegistry:
    container = container_name(donor_armature)
    specs = list(CORE_CHAINS)
    if container == "body_garment_outfit":
        specs.extend(finger_specs())
    transforms: dict[str, AffineTransform] = {}
    chains: dict[str, list[AffineTransform]] = {}
    source_groups: dict[str, AffineTransform] = {}
    inserted_report: list[dict[str, object]] = []
    for spec in specs:
        chain_transforms, chain_groups, chain_report = build_chain(spec, source_armature, donor_armature, alignment)
        chains[spec.name] = chain_transforms
        transforms.update({transform.name: transform for transform in chain_transforms})
        for source_name, transform in chain_groups.items():
            if spec.name == "neck_head":
                source_groups[source_name] = transform
            elif spec.family == "fingers" and not source_name.endswith("_Hand"):
                source_groups[source_name] = transform
            else:
                source_groups.setdefault(source_name, transform)
        if spec.inserted_target_indices:
            inserted_report.append(chain_report)
    head = build_head_transform(source_armature, donor_armature, alignment)
    transforms[head.name] = head
    return TransformRegistry(
        transforms=transforms,
        chain_transforms=chains,
        source_groups=source_groups,
        aliases=alias_table(source_groups, head),
        head=head,
        anchor_report=validate_frame_only_anchors(source_armature, donor_armature, alignment, container),
        inserted_report=inserted_report,
    )


def point_segment_distance(point: Vector, transform: AffineTransform) -> float:
    direction = (transform.source_end - transform.source_start).normalized()
    along = max(0.0, min(transform.source_length, (point - transform.source_start).dot(direction)))
    return (point - (transform.source_start + direction * along)).length


def candidates_for_entry(entry: dict[str, object] | None, registry: TransformRegistry) -> list[AffineTransform]:
    chain = str(entry.get("chain", "torso")) if entry else "torso"
    names = {
        "left_arm": ("left_arm",),
        "right_arm": ("right_arm",),
        "left_leg": ("left_leg",),
        "right_leg": ("right_leg",),
        "head_face": ("neck_head",),
        "torso": ("spine_core", "left_arm", "right_arm"),
    }.get(chain, tuple(registry.chain_transforms))
    return [transform for name in names for transform in registry.chain_transforms.get(name, [])]


def select_transform(
    role: str,
    source_name: str,
    entry: dict[str, object] | None,
    source_armature: bpy.types.Object,
    alignment: Matrix,
    registry: TransformRegistry,
) -> AffineTransform:
    if role in {"head", "hair"}:
        return registry.head
    exact = registry.source_groups.get(source_name)
    if exact is not None:
        return exact
    if source_name == "face_Head":
        return registry.head
    token = v005.token_segment(source_name)
    if token is not None and token in registry.aliases:
        return registry.aliases[token]
    bone = source_armature.data.bones.get(source_name)
    if bone is None:
        return registry.aliases["pelvis_spine"]
    point = alignment @ (source_armature.matrix_world @ bone.head_local)
    candidates = candidates_for_entry(entry, registry) or list(registry.transforms.values())
    return min(candidates, key=lambda transform: (point_segment_distance(point, transform), transform.name))


def warp_point(
    point: Vector,
    assignments: list[tuple[str, str, float]],
    cache: dict[str, AffineTransform],
    role: str,
    source_armature: bpy.types.Object,
    alignment: Matrix,
    mapping_entries: dict[str, dict[str, object]],
    registry: TransformRegistry,
    counts: Counter[str] | None = None,
) -> Vector:
    warped = Vector()
    total = 0.0
    for source_name, _target_name, weight in assignments:
        transform = cache.get(source_name)
        if transform is None:
            transform = select_transform(
                role,
                source_name,
                mapping_entries.get(source_name),
                source_armature,
                alignment,
                registry,
            )
            cache[source_name] = transform
        warped += transform.apply(point) * weight
        total += weight
        if counts is not None:
            counts[transform.name] += 1
    if total <= 0.0:
        raise RuntimeError("No calibrated affine available for a weighted vertex")
    return warped / total


def apply_segment_affine_warp(
    mesh: bpy.types.Object,
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    source_weights: list[list[tuple[str, float]]],
    target_map: dict[str, list[tuple[str, float]]],
    mapping_entries: dict[str, dict[str, object]],
    transform: Matrix,
    threshold: float,
) -> dict[str, object]:
    registry = build_registry(source_armature, donor_armature, transform)
    role = str(mesh.get("source_role", "")).casefold()
    inverse_mesh_world = mesh.matrix_world.inverted()
    cleaned_by_vertex = [clean_assignments(assignments, target_map, threshold) for assignments in source_weights]
    transform_cache: dict[str, AffineTransform] = {}
    assignment_counts: Counter[str] = Counter()
    displacements: list[float] = []
    for vertex, assignments in zip(mesh.data.vertices, cleaned_by_vertex):
        source_point = transform @ (mesh.matrix_world @ vertex.co)
        warped = warp_point(
            source_point,
            assignments,
            transform_cache,
            role,
            source_armature,
            transform,
            mapping_entries,
            registry,
            assignment_counts,
        )
        displacements.append((warped - source_point).length)
        vertex.co = inverse_mesh_world @ warped

    shape_key_count = 0
    if mesh.data.shape_keys is not None:
        for key_block in mesh.data.shape_keys.key_blocks:
            for index, assignments in enumerate(cleaned_by_vertex):
                source_point = transform @ (mesh.matrix_world @ key_block.data[index].co)
                warped = warp_point(
                    source_point,
                    assignments,
                    transform_cache,
                    role,
                    source_armature,
                    transform,
                    mapping_entries,
                    registry,
                )
                key_block.data[index].co = inverse_mesh_world @ warped
            shape_key_count += 1
    mesh.data.update()
    ordered = sorted(displacements)
    p95 = ordered[min(len(ordered) - 1, math.floor((len(ordered) - 1) * 0.95))] if ordered else 0.0
    return {
        "mode": "vector-derived-segment-affine-v006",
        "vertices": len(displacements),
        "shape_keys_warped": shape_key_count,
        "mean_displacement_m": sum(displacements) / len(displacements) if displacements else 0.0,
        "p95_displacement_m": p95,
        "max_displacement_m": max(displacements) if displacements else 0.0,
        "segment_assignments": dict(sorted(assignment_counts.items())),
        "source_group_segments": {name: affine.name for name, affine in sorted(transform_cache.items())},
        "segment_affines": {
            name: {
                "family": affine.family,
                "axial_scale": round(affine.affine.axial_scale, 9),
                "radial_scale": round(affine.affine.radial_scale, 9),
                "determinant": round(affine.affine.determinant, 9),
                "condition_number": round(affine.affine.condition_number, 9),
            }
            for name, affine in sorted(registry.transforms.items())
        },
        "frame_only_anchors": registry.anchor_report,
        "inserted_chain_nodes": registry.inserted_report,
        "anchor_metadata": str(ANCHOR_METADATA.resolve()),
        "anchor_only_rule": "affects_weight_transfer=false; no synthetic vertex groups",
        "axis_calibration": "vector-derived REST frames with anatomical hints and parallel-transported roll; matrix_local forbidden",
        "joint_blend": "weighted transformed-point blend; seam-ring constraints remain a subsequent milestone",
        "seam_constraints_pending": ["neck_bridge", "left_wrist_ring", "right_wrist_ring"],
    }


def main() -> None:
    batch.apply_chain_rest_warp = apply_segment_affine_warp
    batch.main()


if __name__ == "__main__":
    main()
