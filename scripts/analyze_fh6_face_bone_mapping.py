#!/usr/bin/env python3
"""Build an explicit Si face -> FH6 facial-skeleton mapping report."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_garment_bone_mapping import (
    donor_used_bones,
    find_source_armature,
    group_statistics,
    load_donor_objects,
    vec,
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--donor-blend", required=True, type=Path)
    parser.add_argument("--object", required=True, help="Source mesh object to analyze")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def aligned_translation(
    source_armature: bpy.types.Object,
    donor_armature: bpy.types.Object,
    rotation: Matrix,
) -> tuple[Vector, list[dict[str, object]]]:
    pairs = (("左肩", "LeftShoulder"), ("右肩", "RightShoulder"), ("首", "Neck"), ("頭", "Head"))
    landmarks: list[dict[str, object]] = []
    deltas: list[Vector] = []
    for source_name, target_name in pairs:
        source = source_armature.data.bones.get(source_name)
        target = donor_armature.data.bones.get(target_name)
        if source is None or target is None:
            continue
        source_head = source_armature.matrix_world @ source.head_local
        rotated = rotation @ source_head
        target_head = donor_armature.matrix_world @ target.head_local
        delta = target_head - rotated
        deltas.append(delta)
        landmarks.append({
            "source": source_name,
            "target": target_name,
            "source_head": vec(source_head),
            "source_head_rotated": vec(rotated),
            "donor_head": vec(target_head),
            "delta": vec(delta),
        })
    if not deltas:
        raise RuntimeError("No face alignment landmarks resolved")
    return sum(deltas, Vector((0.0, 0.0, 0.0))) / len(deltas), landmarks


def nearest(
    centroid: Vector,
    candidates: list[str],
    donor_heads: dict[str, Vector],
) -> tuple[str, float]:
    available = [name for name in candidates if name in donor_heads]
    if not available:
        raise RuntimeError(f"No donor candidates exist: {candidates}")
    target = min(available, key=lambda name: ((centroid - donor_heads[name]).length, name))
    return target, float((centroid - donor_heads[target]).length)


def family(donor_heads: dict[str, Vector], prefix: str) -> list[str]:
    return sorted(name for name in donor_heads if name.startswith(prefix))


def first_available(donor_heads: dict[str, Vector], names: list[str]) -> str | None:
    return next((name for name in names if name in donor_heads), None)


def facial_candidates(donor_heads: dict[str, Vector]) -> list[str]:
    alice_tokens = ("Brow", "Cheek", "Chin", "Ear", "Eye", "Jaw", "Lip", "Nose", "Tongue")
    return sorted(
        name
        for name in donor_heads
        if name in {"Head", "Jaw", "LeftEye", "RightEye"}
        or name.startswith("Face_")
        or name.startswith("Nose_")
        or name.startswith("Tongue_")
        or (name.startswith(("L_", "M_", "R_")) and any(token in name for token in alice_tokens))
    )


def choose_target(
    source_name: str,
    centroid: Vector,
    donor_heads: dict[str, Vector],
) -> tuple[str, str, float, str]:
    exact_roles = {
        "首": "Neck",
        "頭": "Head",
        "face_Head": "Head",
        "faceLfPupilJoint": "LeftEye",
        "faceLfHighlightJoint": "LeftEye",
        "faceLfHighlightJointA": "LeftEye",
        "faceRtPupilJoint": "RightEye",
        "faceRtHighlightJoint": "RightEye",
        "faceRtHighlightJointA": "RightEye",
        "瞳左": "LeftEye",
        "瞳右": "RightEye",
        "faceMdJawDnJoint": "Jaw",
        "faceMdToothDnJoint": "Jaw",
        "faceMdToothUpJoint": "Head",
        "lineJoint": "Head",
        "line_toothJoint": "Jaw",
    }
    if source_name in exact_roles and exact_roles[source_name] in donor_heads:
        target = exact_roles[source_name]
        return target, "role_exact", float((centroid - donor_heads[target]).length), "explicit facial role"

    ear_targets = {
        "L_ear_01_jnt": ["L_Ear", "Face_L_Ear_001"],
        "R_ear_01_jnt": ["R_Ear", "Face_R_Ear_001"],
    }
    if source_name in ear_targets:
        target = first_available(donor_heads, ear_targets[source_name])
        if target is not None:
            return target, "role_exact", float((centroid - donor_heads[target]).length), "explicit ear role"

    tongue_match = re.fullmatch(r"TongueMd0([1-4])Joint", source_name)
    if tongue_match:
        chain_index = int(tongue_match.group(1)) - 1
        chains = (
            ("TongueA", "TongueB", "TongueC", "TongueC"),
            ("Tongue_1", "Tongue_2", "Tongue_3", "Tongue_3"),
        )
        target = first_available(donor_heads, [chain[chain_index] for chain in chains])
        if target is None:
            target = first_available(donor_heads, ["Jaw", "Head"])
        if target is None:
            raise RuntimeError("Donor has no tongue, jaw, or head target")
        return target, "role_chain", float((centroid - donor_heads[target]).length), "tongue chain order"

    side = None
    if "Lf" in source_name or source_name.startswith("L_"):
        side = "L"
    elif "Rt" in source_name or source_name.startswith("R_"):
        side = "R"

    if source_name.startswith("NoseMd"):
        candidates = (
            family(donor_heads, "Face_M_Nose_")
            + family(donor_heads, "Nose_")
            + [name for name in donor_heads if "Nose" in name and name.startswith(("L_", "M_", "R_"))]
        )
        target, distance = nearest(centroid, candidates, donor_heads)
        return target, "spatial_nose", distance, "nearest central nose control"

    if source_name.startswith("brow") and side:
        candidates = family(donor_heads, f"Face_{side}_Brow_") + family(donor_heads, f"{side}_Brow_")
        target, distance = nearest(centroid, candidates, donor_heads)
        return target, "spatial_brow", distance, "nearest same-side brow control"

    if source_name.startswith("eye") and side:
        if "Irissd" in source_name:
            target = "LeftEye" if side == "L" else "RightEye"
            return target, "role_eye", float((centroid - donor_heads[target]).length), "iris follows the eye joint"
        candidates = (
            family(donor_heads, f"Face_{side}_Eye_Inner_")
            + family(donor_heads, f"Face_{side}_Eye_Outer_")
            + [
                name
                for name in donor_heads
                if name.startswith(f"{side}_Eye") and name not in {"LeftEye", "RightEye"}
            ]
        )
        target, distance = nearest(centroid, candidates, donor_heads)
        return target, "spatial_eye", distance, "nearest same-side eyelid/eye-ring control"

    if source_name.startswith("face") and "Cheek" in source_name and side:
        candidates = family(donor_heads, f"Face_{side}_Cheek_") + family(donor_heads, f"{side}_Cheek_")
        target, distance = nearest(centroid, candidates, donor_heads)
        return target, "spatial_cheek", distance, "nearest same-side cheek control"

    if source_name.startswith("lip"):
        candidates = (
            family(donor_heads, "Face_Mouth_Inner_")
            + family(donor_heads, "Face_Mouth_Outer_")
            + sorted(name for name in donor_heads if "Lip" in name)
        )
        target, distance = nearest(centroid, candidates, donor_heads)
        return target, "spatial_mouth", distance, "nearest mouth-ring control"

    candidates = facial_candidates(donor_heads)
    target, distance = nearest(centroid, candidates, donor_heads)
    return target, "spatial_face_fallback", distance, "nearest driven facial control"


def main() -> None:
    args = arguments()
    source_blend = args.source_blend.resolve()
    donor_blend = args.donor_blend.resolve()
    output = args.output.resolve()
    if Path(bpy.data.filepath).resolve() != source_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {source_blend}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh = bpy.data.objects.get(args.object)
    if mesh is None or mesh.type != "MESH":
        raise RuntimeError(f"Source face mesh {args.object!r} not found")
    source_armature = find_source_armature(mesh)
    donor_objects = load_donor_objects(donor_blend)
    donor_armature, donor_used = donor_used_bones(donor_objects)
    donor_heads = {
        bone.name: donor_armature.matrix_world @ bone.head_local
        for bone in donor_armature.data.bones
        if bone.name in donor_used
    }

    rotation = Matrix.Rotation(math.pi, 4, "Z")
    translation, landmarks = aligned_translation(source_armature, donor_armature, rotation)
    stats = group_statistics(mesh)
    mappings: list[dict[str, object]] = []
    methods: Counter[str] = Counter()
    target_weight: defaultdict[str, float] = defaultdict(float)
    for source_name, item in stats.items():
        source_centroid = item["centroid_source"]
        aligned_centroid = rotation @ source_centroid + translation
        target, method, distance, rationale = choose_target(source_name, aligned_centroid, donor_heads)
        weight = float(item["total_weight"])
        methods[method] += 1
        target_weight[target] += weight
        mappings.append({
            "source": source_name,
            "target": target,
            "method": method,
            "distance_m": distance,
            "source_vertices": int(item["vertices"]),
            "source_total_weight": weight,
            "centroid_source": vec(source_centroid),
            "centroid_aligned": vec(aligned_centroid),
            "rationale": rationale,
        })
    mappings.sort(key=lambda item: (-float(item["source_total_weight"]), str(item["source"])))
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Explicit Si face mapping to the FH6 DRV_BA_F_01 facial skeleton.",
        "source": {"blend": str(source_blend), "object": mesh.name, "weighted_groups": len(stats)},
        "donor": {
            "blend": str(donor_blend),
            "armature": donor_armature.name,
            "skeleton_bones": len(donor_armature.data.bones),
            "mesh_used_bone_count": len(donor_used),
        },
        "alignment": {
            "rotation_z_degrees": 180.0,
            "translation": vec(translation),
            "scale": 1.0,
            "landmarks": landmarks,
        },
        "summary": {
            "mapped_groups": len(mappings),
            "unmapped_groups": 0,
            "unique_targets": len(target_weight),
            "method_counts": dict(sorted(methods.items())),
            "weight_prune_threshold_next_stage": 0.001,
            "max_influences_next_stage": 4,
        },
        "target_distribution": [
            {"target": target, "source_total_weight": weight}
            for target, weight in sorted(target_weight.items(), key=lambda item: (-item[1], item[0]))
        ],
        "mappings": mappings,
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_FACE_BONE_MAP=" + json.dumps({
        "output": str(output),
        "mapped_groups": len(mappings),
        "unique_targets": len(target_weight),
        "method_counts": dict(sorted(methods.items())),
        "translation": vec(translation),
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
