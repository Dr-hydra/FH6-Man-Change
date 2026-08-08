from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import si_segment_affine as segment  # noqa: E402


SOURCE_SKELETON = ROOT / "work/si/fbx-source/milestone-02-donor-plan/skeletons/source-fbx.skeleton.json"
HELMET_SKELETON = ROOT / "work/si/fbx-source/milestone-02-donor-plan/skeletons/helmet-race-modern.skeleton.json"
OUTFIT_SKELETON = ROOT / "work/si/fbx-source/milestone-02-donor-plan/skeletons/outfit-race-suit-modern-f.skeleton.json"
HELMET_MAPPING = ROOT / "work/si/fbx-source/milestone-02-donor-plan/bone-maps/head-hair-to-helmet-v004.json"
OUTFIT_MAPPING = ROOT / "work/si/fbx-source/milestone-02-donor-plan/bone-maps/body-garment-to-outfit-v004.json"
ANCHORS = ROOT / "work/si/fbx-source/milestone-05-validation-v001/segment-frame-audit/si-display-frame-anchors-v001.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bones(path: Path) -> dict[str, dict]:
    return {bone["name"]: bone for bone in load_json(path)["bones"]}


class VectorFrameTests(unittest.TestCase):
    def test_frame_is_right_handed_and_orthonormal(self) -> None:
        frame = segment.derive_segment_frame(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0),
            anatomical_hint=(1.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(segment.norm(frame.axis), 1.0, places=12)
        self.assertAlmostEqual(segment.norm(frame.roll), 1.0, places=12)
        self.assertAlmostEqual(segment.norm(frame.binormal), 1.0, places=12)
        self.assertAlmostEqual(segment.dot(frame.axis, frame.roll), 0.0, places=12)
        self.assertAlmostEqual(segment.dot(frame.axis, frame.binormal), 0.0, places=12)
        self.assertAlmostEqual(segment.dot(frame.roll, frame.binormal), 0.0, places=12)
        self.assertAlmostEqual(frame.determinant, 1.0, places=12)

    def test_parallel_hint_uses_transported_roll(self) -> None:
        frame = segment.derive_segment_frame(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            anatomical_hint=(0.0, 0.0, 1.0),
            previous_roll=(1.0, 0.0, 0.0),
        )
        self.assertEqual(frame.basis, "parallel-transported previous roll")
        self.assertGreater(segment.dot(frame.roll, (1.0, 0.0, 0.0)), 0.999999)

    def test_short_segment_is_rejected(self) -> None:
        with self.assertRaises(segment.DegenerateSegmentError):
            segment.derive_segment_frame(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0001),
                anatomical_hint=(1.0, 0.0, 0.0),
            )


class SegmentAffineTests(unittest.TestCase):
    def test_affine_maps_both_endpoints(self) -> None:
        source_head = (1.0, 2.0, 3.0)
        source_end = (1.0, 2.0, 5.0)
        target_head = (-4.0, 7.0, 1.0)
        target_end = (0.0, 7.0, 1.0)
        source_frame = segment.derive_segment_frame(source_head, source_end, anatomical_hint=(1.0, 0.0, 0.0))
        target_frame = segment.derive_segment_frame(target_head, target_end, anatomical_hint=(0.0, 0.0, 1.0))
        affine = segment.solve_segment_affine(
            source_head,
            source_end,
            source_frame,
            target_head,
            target_end,
            target_frame,
            max_condition=2.1,
        )
        self.assertLess(segment.distance(affine.apply(source_head), target_head), 1.0e-10)
        self.assertLess(segment.distance(affine.apply(source_end), target_end), 1.0e-10)
        self.assertAlmostEqual(affine.axial_scale, 2.0, places=12)
        self.assertAlmostEqual(affine.determinant, 2.0, places=12)
        self.assertAlmostEqual(affine.condition_number, 2.0, places=9)

    def test_antiparallel_axis_does_not_reflect(self) -> None:
        source_head = (0.0, 0.0, 0.0)
        source_end = (1.0, 0.0, 0.0)
        target_head = (2.0, 0.0, 0.0)
        target_end = (1.0, 0.0, 0.0)
        source_frame = segment.derive_segment_frame(source_head, source_end, anatomical_hint=(0.0, 0.0, 1.0))
        target_frame = segment.derive_segment_frame(target_head, target_end, anatomical_hint=(0.0, 0.0, 1.0))
        affine = segment.solve_segment_affine(source_head, source_end, source_frame, target_head, target_end, target_frame)
        self.assertGreater(affine.determinant, 0.0)
        self.assertLess(segment.distance(affine.apply(source_end), target_end), 1.0e-10)

    def test_reflection_is_rejected(self) -> None:
        with self.assertRaises(segment.TransformQualityError):
            segment.validate_linear_transform(
                ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                max_condition=2.5,
            )

    def test_ill_conditioned_transform_is_rejected(self) -> None:
        with self.assertRaises(segment.TransformQualityError):
            segment.validate_linear_transform(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
                max_condition=2.5,
            )

    def test_condition_number_is_rotation_invariant(self) -> None:
        rotated_anisotropic = (
            (math.sqrt(2.0), -math.sqrt(0.125), 0.0),
            (math.sqrt(2.0), math.sqrt(0.125), 0.0),
            (0.0, 0.0, 1.0),
        )
        quality = segment.validate_linear_transform(rotated_anisotropic, max_condition=4.01)
        self.assertAlmostEqual(quality.condition_number, 4.0, places=9)


class ArcLengthTests(unittest.TestCase):
    def test_inserted_knot_uses_target_arc_fraction(self) -> None:
        correspondences = segment.map_target_knots_to_source(
            [(0.0, 0.0, 0.0), (0.0, 4.0, 0.0)],
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)],
            inserted_target_indices=[1],
        )
        self.assertTrue(correspondences[1].inserted)
        self.assertAlmostEqual(correspondences[1].target_fraction, 0.5, places=12)
        self.assertEqual(correspondences[1].source_sample.point, (0.0, 2.0, 0.0))

    def test_invalid_inserted_endpoint_is_rejected(self) -> None:
        with self.assertRaises(segment.ArcLengthError):
            segment.map_target_knots_to_source(
                [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                inserted_target_indices=[0],
            )


class WorkspaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = segment.load_frame_anchor_metadata(ANCHORS)
        cls.source_bones = bones(SOURCE_SKELETON)
        cls.helmet_bones = bones(HELMET_SKELETON)
        cls.outfit_bones = bones(OUTFIT_SKELETON)
        cls.helmet_alignment = segment.RigidZAlignment.from_mapping(load_json(HELMET_MAPPING))
        cls.outfit_alignment = segment.RigidZAlignment.from_mapping(load_json(OUTFIT_MAPPING))

    def test_frame_only_anchor_contract_is_complete(self) -> None:
        helmet = list(segment.iter_frame_anchors(self.metadata, "head_hair_helmet"))
        outfit = list(segment.iter_frame_anchors(self.metadata, "body_garment_outfit"))
        self.assertEqual(len(helmet), 5)
        self.assertEqual(len(outfit), 4)
        self.assertTrue(all(not anchor.affects_weight_transfer for anchor in helmet + outfit))
        self.assertEqual(
            {(anchor.source_start, anchor.target_start) for anchor in outfit},
            {
                ("Bip001_L_Thigh", "LeftUpLeg"),
                ("Bip001_L_Calf", "LeftLeg"),
                ("Bip001_R_Thigh", "RightUpLeg"),
                ("Bip001_R_Calf", "RightLeg"),
            },
        )

    def test_real_anchor_affines_are_positive_and_within_gate(self) -> None:
        cases = (
            ("head_hair_helmet", self.helmet_bones, self.helmet_alignment, 2.25),
            ("body_garment_outfit", self.outfit_bones, self.outfit_alignment, 2.5),
        )
        for container_name, donor_bones, alignment, condition_gate in cases:
            for anchor in segment.iter_frame_anchors(self.metadata, container_name):
                with self.subTest(anchor=anchor.anchor_id):
                    source_head = alignment.apply_point(self.source_bones[anchor.source_start]["head_world_rest"])
                    source_end = alignment.apply_point(self.source_bones[anchor.source_end]["head_world_rest"])
                    target_head = tuple(donor_bones[anchor.target_start]["head_world_rest"])
                    target_end = tuple(donor_bones[anchor.target_end]["head_world_rest"])
                    source_frame = segment.derive_segment_frame(source_head, source_end, anatomical_hint=anchor.roll_hint)
                    target_frame = segment.derive_segment_frame(target_head, target_end, anatomical_hint=anchor.roll_hint)
                    affine = segment.solve_segment_affine(
                        source_head,
                        source_end,
                        source_frame,
                        target_head,
                        target_end,
                        target_frame,
                        max_condition=condition_gate,
                    )
                    self.assertGreater(affine.determinant, 0.0)
                    self.assertLessEqual(affine.condition_number, condition_gate)
                    self.assertLess(segment.distance(affine.apply(source_head), target_head), 1.0e-8)
                    self.assertLess(segment.distance(affine.apply(source_end), target_end), 1.0e-8)

    def test_neck1_and_finger_meta_arc_helpers_use_real_skeletons(self) -> None:
        records = self.metadata["inserted_chain_nodes"]
        self.assertEqual(len(records), 10)
        for record in records:
            with self.subTest(record=record["id"]):
                donor_bones = self.helmet_bones if record["container"] == "head_hair_helmet" else self.outfit_bones
                alignment = self.helmet_alignment if record["container"] == "head_hair_helmet" else self.outfit_alignment
                source_points = [alignment.apply_point(self.source_bones[name]["head_world_rest"]) for name in record["source_nodes"]]
                target_points = [tuple(donor_bones[name]["head_world_rest"]) for name in record["target_nodes"]]
                correspondences = segment.map_target_knots_to_source(
                    source_points,
                    target_points,
                    inserted_target_indices=record["inserted_target_indices"],
                )
                self.assertEqual(len(correspondences), len(record["target_nodes"]))
                fractions = [item.target_fraction for item in correspondences]
                self.assertEqual(fractions, sorted(fractions))
                for inserted_index in record["inserted_target_indices"]:
                    self.assertTrue(correspondences[inserted_index].inserted)
                    self.assertGreater(correspondences[inserted_index].target_fraction, 0.0)
                    self.assertLess(correspondences[inserted_index].target_fraction, 1.0)
                if record["id"].endswith("_meta"):
                    self.assertEqual(correspondences[1].source_sample.segment_index, 0)
                if record["id"].endswith("neck1"):
                    self.assertAlmostEqual(correspondences[1].target_fraction, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
