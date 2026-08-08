#!/usr/bin/env python3
"""Build an explicit Si PMX garment -> FH6 donor bone-collapse report.

This script is analysis-only.  It does not edit the source mesh, armature, or
weights.  The resulting mapping is consumed by the retarget milestone script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--donor-blend", required=True, type=Path)
    parser.add_argument("--object", default="Si_Garment_Cloth1_Upper_Prototype")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def vec(values: Vector) -> list[float]:
    return [float(values.x), float(values.y), float(values.z)]


def load_donor_objects(path: Path) -> list[bpy.types.Object]:
    with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    return [obj for obj in data_to.objects if obj is not None]


def find_source_armature(mesh: bpy.types.Object) -> bpy.types.Object:
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            return modifier.object
    raise RuntimeError(f"{mesh.name!r} has no source armature modifier")


def group_statistics(mesh: bpy.types.Object) -> dict[str, dict[str, object]]:
    accum: dict[str, dict[str, object]] = {
        group.name: {
            "vertices": 0,
            "total_weight": 0.0,
            "weighted_position": Vector((0.0, 0.0, 0.0)),
            "minimum": Vector((math.inf, math.inf, math.inf)),
            "maximum": Vector((-math.inf, -math.inf, -math.inf)),
        }
        for group in mesh.vertex_groups
    }
    index_to_name = {group.index: group.name for group in mesh.vertex_groups}
    for vertex in mesh.data.vertices:
        position = mesh.matrix_world @ vertex.co
        for assignment in vertex.groups:
            if assignment.weight <= 0.0 or assignment.group not in index_to_name:
                continue
            item = accum[index_to_name[assignment.group]]
            item["vertices"] += 1
            item["total_weight"] += float(assignment.weight)
            item["weighted_position"] += position * assignment.weight
            minimum = item["minimum"]
            maximum = item["maximum"]
            for axis in range(3):
                minimum[axis] = min(minimum[axis], position[axis])
                maximum[axis] = max(maximum[axis], position[axis])

    result: dict[str, dict[str, object]] = {}
    for name, item in accum.items():
        total = float(item["total_weight"])
        if total <= 0.0:
            continue
        result[name] = {
            "vertices": int(item["vertices"]),
            "total_weight": total,
            "centroid_source": item["weighted_position"] / total,
            "bounds_min_source": item["minimum"],
            "bounds_max_source": item["maximum"],
        }
    return result


def donor_used_bones(objects: list[bpy.types.Object]) -> tuple[bpy.types.Object, set[str]]:
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one donor armature, found {len(armatures)}")
    used = {
        group.name
        for obj in objects
        if obj.type == "MESH"
        for group in obj.vertex_groups
    }
    return armatures[0], used


def alignment_translation(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    rotation: Matrix,
) -> tuple[Vector, list[dict[str, object]]]:
    pairs = [
        ("左肩", "LeftShoulder"),
        ("右肩", "RightShoulder"),
        ("首", "Neck"),
    ]
    deltas: list[Vector] = []
    landmarks: list[dict[str, object]] = []
    for source_name, donor_name in pairs:
        source_bone = source_armature.data.bones.get(source_name)
        donor_bone = donor_armature.data.bones.get(donor_name)
        if source_bone is None or donor_bone is None:
            continue
        source_head = source_armature.matrix_world @ source_bone.head_local
        rotated_source = rotation @ source_head
        donor_head = donor_armature.matrix_world @ donor_bone.head_local
        delta = donor_head - rotated_source
        deltas.append(delta)
        landmarks.append(
            {
                "source": source_name,
                "target": donor_name,
                "source_head": vec(source_head),
                "source_head_rotated": vec(rotated_source),
                "donor_head": vec(donor_head),
                "delta": vec(delta),
            }
        )
    if not deltas:
        raise RuntimeError("No source/donor alignment landmarks resolved")
    translation = sum(deltas, Vector((0.0, 0.0, 0.0))) / len(deltas)
    return translation, landmarks


def source_side(name: str, aligned_centroid: Vector) -> str | None:
    lowered = name.lower()
    if name.startswith("左") or lowered.startswith("l_") or "left" in lowered or "lf" in lowered:
        return "Left"
    if name.startswith("右") or lowered.startswith("r_") or "right" in lowered or "rt" in lowered:
        return "Right"
    # After the 180-degree Z alignment, donor Left occupies negative X.
    if abs(aligned_centroid.x) >= 0.12:
        return "Left" if aligned_centroid.x < 0.0 else "Right"
    return None


def nearest_target(
    centroid: Vector,
    candidate_names: list[str],
    donor_heads: dict[str, Vector],
) -> tuple[str, float]:
    available = [name for name in candidate_names if name in donor_heads]
    if not available:
        raise RuntimeError(f"None of the candidate donor bones exist: {candidate_names}")
    target = min(available, key=lambda name: (centroid - donor_heads[name]).length)
    return target, float((centroid - donor_heads[target]).length)


def choose_target(
    source_name: str,
    aligned_centroid: Vector,
    donor_heads: dict[str, Vector],
    donor_used: set[str],
) -> tuple[str, str, float, float, str]:
    exact = {
        "センター": "Hips",
        "グルーブ": "Hips",
        "腰": "Hips",
        "下半身": "Hips",
        "上半身": "Spine",
        "上半身1": "Spine1",
        "上半身2": "Spine2",
        "首": "Neck",
        "頭": "Head",
        "左肩": "LeftShoulder",
        "右肩": "RightShoulder",
        "左ひじ": "LeftForeArm_TWIST0",
        "右ひじ": "RightForeArm_TWIST0",
        "左手首": "LeftHand",
        "右手首": "RightHand",
        "左足": "LeftUpLeg",
        "左足D": "LeftUpLeg",
        "左ひざ": "LeftLeg",
        "左ひざD": "LeftLeg",
        "左足首": "LeftFoot",
        "左足首D": "LeftFoot",
        "左足先EX": "LeftToeBase",
        "右足": "RightUpLeg",
        "右足D": "RightUpLeg",
        "右ひざ": "RightLeg",
        "右ひざD": "RightLeg",
        "右足首": "RightFoot",
        "右足首D": "RightFoot",
        "右足先EX": "RightToeBase",
        "左腕": "LeftArm_TWIST1",
        "左腕捩": "LeftArm_TWIST1",
        "左腕捩1": "LeftArm_TWIST2",
        "左ひじ": "LeftForeArm_TWIST0",
        "左手捩": "LeftForeArm_TWIST1",
        "左手捩1": "LeftForeArm_TWIST2",
        "左手首": "LeftHand",
        "右腕": "RightArm_TWIST1",
        "右腕捩": "RightArm_TWIST1",
        "右腕捩1": "RightArm_TWIST2",
        "右ひじ": "RightForeArm_TWIST0",
        "右手捩": "RightForeArm_TWIST1",
        "右手捩1": "RightForeArm_TWIST2",
        "右手首": "RightHand",
    }
    finger_stems = {
        "親指": "Thumb",
        "人指": "Index",
        "中指": "Middle",
        "薬指": "Ring",
        "小指": "Pinky",
    }
    for prefix, side_name in (("左", "Left"), ("右", "Right")):
        for source_stem, target_stem in finger_stems.items():
            for source_number in range(3):
                target_number = source_number + 1
                source_number_value = source_number if source_stem == "親指" else source_number + 1
                for digit in (str(source_number_value), "０１２３４５６７８９"[source_number_value]):
                    exact[f"{prefix}{source_stem}{digit}"] = f"{side_name}{target_stem}{target_number}"
    leg_role_candidates = {
        "左足": ("LeftUpLeg", "LeftUpLeg_TWIST1"),
        "左足D": ("LeftUpLeg", "LeftUpLeg_TWIST1"),
        "左ひざ": ("LeftLeg", "LeftLeg_TWIST0"),
        "左ひざD": ("LeftLeg", "LeftLeg_TWIST0"),
        "右足": ("RightUpLeg", "RightUpLeg_TWIST1"),
        "右足D": ("RightUpLeg", "RightUpLeg_TWIST1"),
        "右ひざ": ("RightLeg", "RightLeg_TWIST0"),
        "右ひざD": ("RightLeg", "RightLeg_TWIST0"),
    }
    for source, candidates in leg_role_candidates.items():
        target = next((name for name in candidates if name in donor_used), None)
        if target is not None:
            exact[source] = target
    if source_name in exact and exact[source_name] in donor_used:
        target = exact[source_name]
        distance = float((aligned_centroid - donor_heads[target]).length)
        return target, "role_exact", 1.0, distance, "source role maps directly to the donor garment role"

    lowered = source_name.lower()
    side = source_side(source_name, aligned_centroid)

    chest_terms = ("胸", "bust", "breast", "pec")
    if side is not None and any(term in lowered or term in source_name for term in chest_terms):
        target = f"{side}Pec_Corrective"
        if target in donor_used:
            distance = float((aligned_centroid - donor_heads[target]).length)
            return target, "role_collapse", 0.82, distance, "source chest control collapses to the donor pec corrective"

    shoulder_terms = ("肩", "shoulder", "scap")
    if side is not None and any(term in lowered or term in source_name for term in shoulder_terms):
        target = f"{side}Shoulder"
        if target in donor_used:
            distance = float((aligned_centroid - donor_heads[target]).length)
            return target, "role_collapse", 0.86, distance, "source shoulder/attachment control collapses to donor shoulder"

    arm_candidates: dict[str, list[str]] = {
        "Left": [
            "LeftShoulder",
            "LeftArm_TWIST1",
            "LeftArm_TWIST2",
            "LeftArm_TWIST3",
            "LeftForeArm_TWIST0",
            "LeftForeArm_TWIST1",
            "LeftForeArm_TWIST2",
            "LeftForeArm_TWIST3",
            "LeftHand",
        ],
        "Right": [
            "RightShoulder",
            "RightArm_TWIST1",
            "RightArm_TWIST2",
            "RightArm_TWIST3",
            "RightForeArm_TWIST0",
            "RightForeArm_TWIST1",
            "RightForeArm_TWIST2",
            "RightForeArm_TWIST3",
            "RightHand",
        ],
    }
    arm_terms = (
        "腕",
        "ひじ",
        "手首",
        "手捩",
        "sleeve",
        "bracelet",
        "bracer",
        "wrist",
        "elbow",
        "arm",
        "cuff",
        "glove",
    )
    if side is not None and any(term in lowered or term in source_name for term in arm_terms):
        target, distance = nearest_target(aligned_centroid, arm_candidates[side], donor_heads)
        return target, "spatial_arm_collapse", 0.74, distance, "secondary arm/sleeve chain collapses to nearest driven donor arm bone"

    leg_terms = ("足", "ひざ", "leg", "knee", "ankle", "foot", "thigh", "toe")
    if side is not None and any(term in lowered or term in source_name for term in leg_terms):
        target, distance = nearest_target(
            aligned_centroid,
            [
                f"{side}UpLeg",
                f"{side}UpLeg_TWIST1",
                f"{side}Leg",
                f"{side}Leg_TWIST0",
                f"{side}Foot",
                f"{side}ToeBase",
            ],
            donor_heads,
        )
        return target, "spatial_leg_collapse", 0.74, distance, "secondary lower-body control collapses to the nearest driven donor leg bone"

    if "裙子" in source_name or "skirt" in lowered:
        target, distance = nearest_target(
            aligned_centroid,
            ["Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"],
            donor_heads,
        )
        return target, "spatial_skirt_collapse", 0.62, distance, "skirt physics control collapses to the nearest pelvis or upper-leg role"

    if "tail" in lowered:
        target = "Hips"
        distance = float((aligned_centroid - donor_heads[target]).length)
        return target, "role_collapse", 0.55, distance, "tail physics chain collapses to the driven pelvis"

    torso = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head"]
    spatial = list(torso)
    if side is not None:
        spatial.extend(arm_candidates[side])
        pec = f"{side}Pec_Corrective"
        if pec in donor_used:
            spatial.append(pec)
    spatial = [name for name in spatial if name in donor_used]
    target, distance = nearest_target(aligned_centroid, spatial, donor_heads)
    return target, "spatial_fallback", 0.48, distance, "unmatched secondary control collapses to nearest driven donor role"


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    donor_blend = args.donor_blend.resolve()
    output = args.output.resolve()
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match --source-blend {source_blend}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh = bpy.data.objects.get(args.object)
    if mesh is None or mesh.type != "MESH":
        raise RuntimeError(f"Source working mesh {args.object!r} not found")
    source_armature = find_source_armature(mesh)
    donor_objects = load_donor_objects(donor_blend)
    donor_armature, used = donor_used_bones(donor_objects)
    donor_heads = {
        bone.name: donor_armature.matrix_world @ bone.head_local
        for bone in donor_armature.data.bones
        if bone.name in used
    }

    rotation = Matrix.Rotation(math.pi, 4, "Z")
    translation, landmarks = alignment_translation(source_armature, donor_armature, rotation)
    stats = group_statistics(mesh)
    mappings: list[dict[str, object]] = []
    method_counts: Counter[str] = Counter()
    target_source_weight: defaultdict[str, float] = defaultdict(float)
    target_group_count: Counter[str] = Counter()
    for name, item in stats.items():
        source_centroid = item["centroid_source"]
        aligned_centroid = rotation @ source_centroid + translation
        target, method, confidence, distance, rationale = choose_target(
            name, aligned_centroid, donor_heads, used
        )
        total_weight = float(item["total_weight"])
        method_counts[method] += 1
        target_source_weight[target] += total_weight
        target_group_count[target] += 1
        mappings.append(
            {
                "source": name,
                "target": target,
                "method": method,
                "confidence": confidence,
                "distance_m": distance,
                "source_vertices": int(item["vertices"]),
                "source_total_weight": total_weight,
                "centroid_source": vec(source_centroid),
                "centroid_aligned": vec(aligned_centroid),
                "bounds_min_source": vec(item["bounds_min_source"]),
                "bounds_max_source": vec(item["bounds_max_source"]),
                "rationale": rationale,
                "loss": "secondary/source-specific motion is collapsed to a donor-driven role"
                if method != "role_exact"
                else None,
            }
        )

    mappings.sort(key=lambda item: (-float(item["source_total_weight"]), str(item["source"])))
    target_distribution = [
        {
            "target": target,
            "source_groups": int(target_group_count[target]),
            "source_total_weight": float(weight),
        }
        for target, weight in sorted(target_source_weight.items(), key=lambda item: (-item[1], item[0]))
    ]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Analysis-only explicit mapping for the first FH6 female garment retarget milestone.",
        "source": {
            "blend": str(source_blend),
            "object": mesh.name,
            "armature": source_armature.name,
            "weighted_groups": len(stats),
        },
        "donor": {
            "blend": str(donor_blend),
            "armature": donor_armature.name,
            "skeleton_bones": len(donor_armature.data.bones),
            "mesh_used_bones": sorted(used),
            "mesh_used_bone_count": len(used),
        },
        "alignment": {
            "rotation_z_degrees": 180.0,
            "translation": vec(translation),
            "scale": 1.0,
            "landmarks": landmarks,
            "reason": "PMX faces -Y with anatomical Left at +X; imported FH6 donor faces +Y with Left at -X.",
        },
        "summary": {
            "mapped_groups": len(mappings),
            "unmapped_groups": 0,
            "unique_targets": len(target_distribution),
            "method_counts": dict(sorted(method_counts.items())),
            "weight_prune_threshold_next_stage": 0.001,
            "max_influences_next_stage": 4,
        },
        "target_distribution": target_distribution,
        "mappings": mappings,
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_GARMENT_BONE_MAP="
        + json.dumps(
            {
                "output": str(output),
                "source_groups": len(stats),
                "mapped_groups": len(mappings),
                "unique_targets": len(target_distribution),
                "translation": vec(translation),
                "method_counts": dict(sorted(method_counts.items())),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
