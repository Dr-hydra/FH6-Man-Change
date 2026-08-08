#!/usr/bin/env python3
"""Build explicit FBX-to-physical-Display skeleton mapping contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
SIDE_TARGET = {"L": "Left", "R": "Right"}
FINGER_TARGET = {
    "Finger0": "Thumb",
    "Finger1": "Index",
    "Finger2": "Middle",
    "Finger3": "Ring",
    "Finger4": "Pinky",
}
CORE_SOURCE_TARGET = {
    "Bip001": "Root",
    "Bip001_Pelvis": "Hips",
    "Bip001_Spine": "Spine",
    "Bip001_Spine1": "Spine1",
    "Bip001_Spine2": "Spine2",
    "Bip001_Neck": "Neck",
    "Bip001_Head": "Head",
    "face_Head": "Head",
}
EYE_RING_TARGETS = {
    "L": ("L_Eye_Inner", "L_Eye_Outer", "L_Eyelid_Lower", "L_Eyelid_Upper"),
    "R": ("R_Eye_Inner", "R_Eye_Outer", "R_Eyelid_Lower", "R_Eyelid_Upper"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "sources" / "si" / "source.config.json",
    )
    parser.add_argument("--head-output", type=Path)
    parser.add_argument("--body-output", type=Path)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (WORKSPACE / candidate).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector(values: list[float]) -> list[float]:
    return [float(value) for value in values]


def add(left: list[float], right: list[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right, strict=True)]


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right, strict=True)]


def scale(values: list[float], factor: float) -> list[float]:
    return [float(value) * factor for value in values]


def length(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def rotate_z_180(values: list[float]) -> list[float]:
    return [-float(values[0]), -float(values[1]), float(values[2])]


def distance(left: list[float], right: list[float]) -> float:
    return length(subtract(left, right))


def side_of(name: str) -> str | None:
    if (
        re.search(r"(?:^|_)L(?:_|[A-Z])", name)
        or "Lf" in name
        or name.startswith(("Left", "lipL"))
    ):
        return "L"
    if (
        re.search(r"(?:^|_)R(?:_|[A-Z])", name)
        or "Rt" in name
        or name.startswith(("Right", "lipR"))
    ):
        return "R"
    return None


def target_prefix(name: str, side: str | None) -> str:
    return SIDE_TARGET.get(side or "", "")


def bone_maps(inventory: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bones = {str(bone["name"]): bone for bone in inventory["bones"]}
    parents = {name: bone.get("parent") for name, bone in bones.items()}
    return bones, parents


def source_usage(
    baseline: dict[str, Any],
    source_inventory: dict[str, Any],
    roles: set[str],
    lods: set[str],
) -> dict[str, dict[str, Any]]:
    inventory_meshes = {str(mesh["object"]): mesh for mesh in source_inventory["meshes"]}
    usage: dict[str, dict[str, Any]] = {}
    for mesh in baseline["meshes"]:
        role = str(mesh["role"])
        lod = str(mesh["lod"])
        if role not in roles or lod not in lods:
            continue
        source_object = str(mesh["object"])
        inventory_mesh = inventory_meshes.get(source_object)
        if inventory_mesh is None:
            continue
        for bone_name in inventory_mesh["used_bones"]:
            record = usage.setdefault(
                str(bone_name),
                {"roles": set(), "lods": set(), "meshes": [], "objects": set()},
            )
            record["roles"].add(role)
            record["lods"].add(lod)
            record["meshes"].append(
                {
                    "object": source_object,
                    "role": role,
                    "lod": lod,
                    "vertices": int(mesh["vertices"]),
                }
            )
            record["objects"].add(source_object)
    for record in usage.values():
        record["roles"] = sorted(record["roles"])
        record["lods"] = sorted(record["lods"])
        record["objects"] = sorted(record["objects"])
        record["meshes"].sort(key=lambda item: (item["lod"], item["object"]))
    return usage


def alignment(source_bones: dict[str, dict[str, Any]], target_bones: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = (
        ("Bip001_L_Clavicle", "LeftShoulder"),
        ("Bip001_R_Clavicle", "RightShoulder"),
        ("Bip001_Neck", "Neck"),
    )
    deltas: list[list[float]] = []
    landmarks = []
    for source_name, target_name in pairs:
        source = source_bones.get(source_name)
        target = target_bones.get(target_name)
        if source is None or target is None:
            continue
        source_head = vector(source["head_world_rest"])
        rotated = rotate_z_180(source_head)
        target_head = vector(target["head_world_rest"])
        delta = subtract(target_head, rotated)
        deltas.append(delta)
        landmarks.append(
            {
                "source": source_name,
                "target": target_name,
                "source_head_world_rest": source_head,
                "source_head_rotated": rotated,
                "target_head_world_rest": target_head,
                "delta": delta,
            }
        )
    if len(deltas) < 2:
        raise ValueError("At least two REST alignment landmarks are required")
    translation = scale(
        [sum(delta[axis] for delta in deltas) for axis in range(3)],
        1.0 / len(deltas),
    )
    return {
        "rotation_z_degrees": 180.0,
        "translation": translation,
        "scale": 1.0,
        "landmarks": landmarks,
        "reason": "FBX source world Left (+X) becomes Display donor Left (-X) after Rz(180); translation is landmark mean only.",
        "global_transform_is_initial_only": True,
        "required_next_stage": "chain_local_affine_rest_warp",
    }


def aligned_position(bone: dict[str, Any], transform: dict[str, Any]) -> list[float]:
    return add(rotate_z_180(vector(bone["head_world_rest"])), transform["translation"])


def nearest_target(
    position: list[float],
    candidates: list[str],
    target_bones: dict[str, dict[str, Any]],
) -> str | None:
    available = [name for name in candidates if name in target_bones]
    if not available:
        return None
    return min(available, key=lambda name: (distance(position, target_bones[name]["head_world_rest"]), name))


def blend_targets(
    position: list[float],
    candidates: list[str],
    target_bones: dict[str, dict[str, Any]],
    count: int = 2,
) -> list[dict[str, Any]]:
    available = [name for name in candidates if name in target_bones]
    if not available:
        return []
    ranked = sorted(
        ((distance(position, target_bones[name]["head_world_rest"]), name) for name in available),
        key=lambda item: (item[0], item[1]),
    )[:count]
    if ranked[0][0] < 1e-8 or len(ranked) == 1:
        return [{"bone": ranked[0][1], "weight": 1.0}]
    inverse = [1.0 / max(item[0], 1e-6) for item in ranked]
    total = sum(inverse)
    return [{"bone": item[1], "weight": value / total} for item, value in zip(ranked, inverse, strict=True)]


def angular_blend_targets(
    position: list[float],
    pivot: list[float],
    candidates: list[str],
    target_bones: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project an eyelid point onto the physical eye ring in polar REST space."""
    source_vector = subtract(position, pivot)
    source_angle = math.atan2(source_vector[2], source_vector[0])

    def angular_distance(name: str) -> float:
        target_vector = subtract(target_bones[name]["head_world_rest"], pivot)
        target_angle = math.atan2(target_vector[2], target_vector[0])
        delta = abs(source_angle - target_angle) % (2.0 * math.pi)
        return min(delta, 2.0 * math.pi - delta)

    available = [name for name in candidates if name in target_bones]
    ranked = sorted(((angular_distance(name), name) for name in available), key=lambda item: (item[0], item[1]))[:2]
    if not ranked:
        return []
    if ranked[0][0] < 1e-8 or len(ranked) == 1:
        return [{"bone": ranked[0][1], "weight": 1.0}]
    inverse = [1.0 / max(item[0], 1e-6) for item in ranked]
    total = sum(inverse)
    return [{"bone": item[1], "weight": value / total} for item, value in zip(ranked, inverse, strict=True)]


def face_family_target(
    name: str,
    position: list[float],
    target_bones: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str] | None:
    lowered = name.casefold()
    side = side_of(name)
    side_word = target_prefix(name, side)
    face_prefix = side
    if name in {"eyeLfJoint", "eyeRtJoint"} and side:
        return [{"bone": f"{side_word}Eye", "weight": 1.0}], "explicit", "eye_pivot"
    if side and (
        lowered.startswith(("facelfiris", "facertiris", "facelfpupil", "facertpupil", "facelfhighlight", "facerthighlight"))
        or "irissd" in lowered
    ):
        return [{"bone": f"{side_word}Eye", "weight": 1.0}], "explicit", "eye_or_iris"
    ring_match = re.search(r"eye(?:lf|rt)(0[1-8])", lowered) if side else None
    if side and ring_match:
        # The FBX ring is numbered from the inner/upper arc toward the outer/lower
        # arc.  The retained 241-bone Display skeleton exposes four semantic
        # controls (inner, outer, upper, lower) at one eye pivot, so use the
        # source ring order instead of collapsing by coincident target positions.
        ring_index = int(ring_match.group(1))
        horizontal = "Inner" if ring_index in {1, 2, 8} else "Outer"
        lid = "Upper" if ring_index <= 5 else "Lower"
        targets = [
            {"bone": f"{side}_Eye_{horizontal}", "weight": 0.60},
            {"bone": f"{side}_Eyelid_{lid}", "weight": 0.40},
        ]
        if all(item["bone"] in target_bones for item in targets):
            return targets, "rest_space_projection", "eyelid_arc"
    if "jawdn" in lowered or "toothdn" in lowered:
        return [{"bone": "Jaw", "weight": 1.0}], "explicit", "jaw_or_lower_teeth"
    if "toothup" in lowered or lowered.startswith("line_tooth"):
        return [{"bone": "Head", "weight": 1.0}], "explicit", "upper_teeth"
    if "tongue" in lowered:
        match = re.search(r"(\d+)", name)
        index = int(match.group(1)) if match else 1
        target = f"Tongue_{min(max(index, 1), 3)}"
        if target in target_bones:
            return [{"bone": target, "weight": 1.0}], "explicit", "tongue"
    if "nose" in lowered:
        candidates = ["Nose_Bridge_Mid", "Nose_Bridge_Top", "Nose_Tip"]
        if side:
            candidates += [name for name in target_bones if name.startswith(f"{face_prefix}_Nose_")]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "rest_space_projection", "nose_family"
    if lowered.startswith("lipm"):
        upper = "up" in lowered
        candidates = [
            name
            for name in target_bones
            if name.startswith("M_Lip_Upper" if upper else "M_Lip_Lower")
        ]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "rest_space_projection", "lip_arc"
    if side is None:
        return None
    if lowered.startswith(("brow", "browline")):
        candidates = [
            name
            for name in target_bones
            if name.startswith(f"{face_prefix}_Brow_")
        ]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "rest_space_projection", "brow_arc"
    if lowered.startswith("face") and "cheek" in lowered:
        candidates = [name for name in target_bones if name.startswith(f"{face_prefix}_Cheek_")]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "rest_space_projection", "cheek_family"
    if lowered.startswith("lip"):
        upper = "up" in lowered
        candidates = [
            name
            for name in target_bones
            if name.startswith(f"{face_prefix}_Lip_Upper_" if upper else f"{face_prefix}_Lip_Lower_")
        ]
        if not candidates:
            candidates = [name for name in target_bones if name.startswith(f"{face_prefix}_Lip_")]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "rest_space_projection", "lip_arc"
    return None


def core_target(name: str, target_bones: dict[str, dict[str, Any]]) -> tuple[str, str, str] | None:
    if name in CORE_SOURCE_TARGET and CORE_SOURCE_TARGET[name] in target_bones:
        return CORE_SOURCE_TARGET[name], "explicit", "core"
    match = re.fullmatch(r"Bip001_([LR])_(.+)", name)
    if not match:
        match = re.fullmatch(r"Bip001_([LR])(.+)", name)
    if not match:
        return None
    side, stem = match.groups()
    prefix = SIDE_TARGET[side]
    if stem in {"Clavicle", "UpperArm", "Forearm", "Hand", "Thigh", "Calf", "Foot", "Toe0"}:
        stem_map = {
            "Clavicle": "Shoulder",
            "UpperArm": "Arm",
            "Forearm": "ForeArm",
            "Hand": "Hand",
            "Thigh": "UpLeg",
            "Calf": "Leg",
            "Foot": "Foot",
            "Toe0": "ToeBase",
        }
        target = prefix + stem_map[stem]
        if target in target_bones:
            return target, "explicit", "core"
    finger_match = re.fullmatch(r"(Finger[0-4])(\d*)", stem)
    if finger_match:
        finger, suffix = finger_match.groups()
        target_stem = FINGER_TARGET[finger]
        if finger == "Finger0":
            segment = {"": 1, "1": 2, "2": 3}.get(suffix)
        else:
            segment = {"": 1, "1": 2, "2": 3}.get(suffix)
        if segment:
            target = f"{prefix}{target_stem}{segment}"
            if target in target_bones:
                return target, "explicit", "finger"
    twist_match = re.fullmatch(r"(.+?)(Twist|Twist1)$", stem)
    if twist_match:
        base, suffix = twist_match.groups()
        family = {
            "UpArm": "Arm_TWIST",
            "Fore": "ForeArm_TWIST",
            "Thigh": "UpLeg_TWIST",
            "Calf": "Leg_TWIST",
        }.get(base)
        if family:
            target = f"{prefix}{family}{0 if suffix == 'Twist' else 1}"
            if base in {"UpArm", "Fore", "Thigh"} and suffix == "Twist1":
                target = f"{prefix}{family}2"
            if target in target_bones:
                return target, "rest_space_projection", f"{base.casefold()}_twist"
    return None


def corrective_target(
    name: str,
    position: list[float],
    target_bones: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str] | None:
    side = side_of(name)
    if side is None:
        return None
    prefix = SIDE_TARGET[side]
    lowered = name.casefold()
    candidates: list[str] = []
    family = "corrective_projection"
    if "hand_ty_plus" in lowered:
        candidates = [f"{prefix}WristUp_Corrective", f"{prefix}Wrist_Corrective"]
        family = "wrist_up_corrective"
    elif "hand_ty_minus" in lowered:
        candidates = [f"{prefix}WristDown_Corrective", f"{prefix}Wrist_Corrective"]
        family = "wrist_down_corrective"
    elif "forearm_ty_plus" in lowered:
        candidates = [f"{prefix}ForeArm_TWIST1", f"{prefix}ForeArm_TWIST2", f"{prefix}WristUp_Corrective"]
        family = "forearm_plus_projection"
    elif "forearm_ty_minus" in lowered:
        candidates = [f"{prefix}ForeArm_TWIST2", f"{prefix}ForeArm_TWIST1", f"{prefix}WristDown_Corrective"]
        family = "forearm_minus_projection"
    elif "upperarm_ty" in lowered or "upperarm_tz" in lowered:
        candidates = [f"{prefix}Bicep_Corrective", f"{prefix}Arm_TWIST1", f"{prefix}Arm_TWIST2"]
        family = "upperarm_corrective_projection"
    elif "calf_ty_plus" in lowered or "calf_bend" in lowered or "calftwist_bend" in lowered:
        candidates = [f"{prefix}Knee_CorrectiveFront", f"{prefix}Leg_TWIST0", f"{prefix}Leg_TWIST1"]
        family = "calf_front_projection"
    elif "calf_ty_minus" in lowered:
        candidates = [f"{prefix}Knee_CorrectiveBack", f"{prefix}Leg_TWIST0", f"{prefix}Leg_TWIST1"]
        family = "calf_back_projection"
    elif "thigh_ty" in lowered or "thigh_tz" in lowered or "thigh_bend" in lowered or "thightwist1_bend" in lowered:
        candidates = [f"{prefix}UpLeg_TWIST0", f"{prefix}UpLeg_TWIST1", f"{prefix}Cloth_CorrectiveFront"]
        family = "thigh_corrective_projection"
    elif "foot_ty_plus" in lowered:
        candidates = [f"{prefix}Foot", f"{prefix}ToeBase"]
        family = "foot_plus_projection"
    elif "foot_ty_minus" in lowered:
        candidates = [f"{prefix}Foot", f"{prefix}ToeBase"]
        family = "foot_minus_projection"
    if not candidates:
        return None
    targets = blend_targets(position, candidates, target_bones, count=2)
    return (targets, "rest_space_projection", family) if targets else None


def secondary_target(
    name: str,
    position: list[float],
    target_bones: dict[str, dict[str, Any]],
    role_set: set[str],
) -> tuple[list[dict[str, Any]], str, str] | None:
    side = side_of(name)
    prefix = SIDE_TARGET.get(side or "")
    lowered = name.casefold()
    if "ear" in lowered and prefix:
        target = f"{side}_Ear"
        if target in target_bones:
            return [{"bone": target, "weight": 1.0}], "parent_fallback", "ear_attachment"
    if role_set <= {"head", "hair"}:
        if "hair" in lowered or "strand" in lowered or "bow" in lowered or "earring" in lowered or "wing" in lowered:
            return [{"bone": "Head", "weight": 1.0}], "parent_fallback", "head_accessory"
        return [{"bone": "Head", "weight": 1.0}], "parent_fallback", "head_fallback"
    if "collar" in lowered:
        return [{"bone": "Neck", "weight": 1.0}], "parent_fallback", "collar_attachment"
    if "breast" in lowered and prefix and f"{prefix}Pec_Corrective" in target_bones:
        return [{"bone": f"{prefix}Pec_Corrective", "weight": 1.0}], "parent_fallback", "pec_attachment"
    if any(token in lowered for token in ("sleeve", "bracelet", "wheat", "ribbon", "wing", "cloud")) and prefix:
        candidates = [
            f"{prefix}Shoulder",
            f"{prefix}Arm",
            f"{prefix}Arm_TWIST1",
            f"{prefix}Arm_TWIST2",
            f"{prefix}ForeArm",
            f"{prefix}ForeArm_TWIST1",
            f"{prefix}ForeArm_TWIST2",
            f"{prefix}Hand",
        ]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "rest_space_projection", "arm_attachment"
    if any(token in lowered for token in ("skirt", "dress", "tail", "tassel")):
        candidates = [
            name
            for name in target_bones
            if name.startswith(("LeftDress", "RightDress"))
            and (side is None or name.startswith(prefix))
        ]
        candidates += ["Spine2", "Spine1", "Spine", "Hips"]
        targets = blend_targets(position, candidates, target_bones, count=2)
        if targets:
            return targets, "parent_fallback", "lower_garment_attachment"
    if any(token in lowered for token in ("cloth", "tail", "tassel", "bow", "ribbon")):
        targets = blend_targets(position, ["Spine2", "Spine1", "Spine", "Hips"], target_bones, count=2)
        if targets:
            return targets, "parent_fallback", "torso_attachment"
    return None


def parent_fallback(
    name: str,
    source_parents: dict[str, str | None],
    mapped: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    current = source_parents.get(name)
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        entry = mapped.get(current)
        if entry:
            return entry["targets"]
        current = source_parents.get(current)
    return None


def classify_chain(targets: list[dict[str, Any]], name: str) -> str:
    target = str(targets[0]["bone"]) if targets else ""
    lowered = name.casefold()
    if "eye" in lowered or "iris" in lowered or "pupil" in lowered or "highlight" in lowered:
        return "left_eye" if target.startswith(("Left", "L_")) else "right_eye"
    if target in {"Head", "Neck", "Neck1"} or lowered.startswith("face"):
        return "head_face"
    if "wrist" in target.casefold() or "forearm" in target.casefold() or "hand" in target.casefold():
        return "left_arm" if target.startswith(("Left", "L_")) else "right_arm"
    if "foot" in target.casefold() or "leg" in target.casefold() or "toe" in target.casefold():
        return "left_leg" if target.startswith(("Left", "L_")) else "right_leg"
    return "torso"


def build_map(
    package: str,
    roles: set[str],
    container: dict[str, Any],
    baseline: dict[str, Any],
    source_inventory: dict[str, Any],
    target_inventory: dict[str, Any],
    lods: set[str],
) -> dict[str, Any]:
    source_bones, source_parents = bone_maps(source_inventory)
    target_bones, target_parents = bone_maps(target_inventory)
    usage = source_usage(baseline, source_inventory, roles, lods)
    transform = alignment(source_bones, target_bones)
    mappings: dict[str, dict[str, Any]] = {}
    method_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []

    ordered_source_names = [
        str(bone["name"])
        for bone in source_inventory["bones"]
        if str(bone["name"]) in usage
    ]
    for source_name in ordered_source_names:
        source_bone = source_bones[source_name]
        position = aligned_position(source_bone, transform)
        chosen: tuple[list[dict[str, Any]], str, str] | None = None
        direct = core_target(source_name, target_bones)
        if direct:
            chosen = ([{"bone": direct[0], "weight": 1.0}], direct[1], direct[2])
        if chosen is None and roles <= {"head", "hair"}:
            chosen = face_family_target(source_name, position, target_bones)
        if chosen is None:
            chosen = corrective_target(source_name, position, target_bones)
        if chosen is None:
            chosen = secondary_target(source_name, position, target_bones, roles)
        if chosen is None:
            fallback = parent_fallback(source_name, source_parents, mappings)
            if fallback:
                chosen = (fallback, "parent_fallback", "nearest_mapped_parent")
        if chosen is None:
            candidates = [
                name
                for name in target_bones
                if name in {"Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"}
                or name.endswith(("Arm", "ForeArm", "Hand", "Foot", "ToeBase", "UpLeg", "Leg"))
            ]
            nearest = nearest_target(position, candidates, target_bones)
            if nearest:
                chosen = ([{"bone": nearest, "weight": 1.0}], "rest_space_projection", "nearest_core_fallback")
        if chosen is None or not chosen[0]:
            errors.append({"source": source_name, "code": "unmapped_source_bone"})
            continue
        targets, mode, semantic_family = chosen
        targets = [
            {"bone": str(item["bone"]), "weight": float(item["weight"])}
            for item in targets
            if float(item["weight"]) > 1e-8
        ]
        total = sum(float(item["weight"]) for item in targets)
        targets = [{"bone": item["bone"], "weight": item["weight"] / total} for item in targets]
        missing_targets = [item["bone"] for item in targets if item["bone"] not in target_bones]
        if missing_targets:
            errors.append({"source": source_name, "code": "target_bone_missing", "targets": missing_targets})
            continue
        primary_target = max(targets, key=lambda item: (item["weight"], item["bone"]))["bone"]
        target_position = [
            sum(float(item["weight"]) * float(target_bones[item["bone"]]["head_world_rest"][axis]) for item in targets)
            for axis in range(3)
        ]
        usage_record = usage[source_name]
        entry = {
            "source": source_name,
            "target": primary_target,
            "targets": targets,
            "mode": mode,
            "source_format": "fbx",
            "component_roles": usage_record["roles"],
            "lods": usage_record["lods"],
            "source_objects": usage_record["objects"],
            "chain": classify_chain(targets, source_name),
            "semantic_family": semantic_family,
            "source_parent": source_parents.get(source_name),
            "target_parent": target_parents.get(primary_target),
            "source_head_world_rest": vector(source_bone["head_world_rest"]),
            "source_head_aligned": position,
            "target_head_world_rest": target_position,
            "rest_head_error_m": distance(position, target_position),
            "rest_rule": "chain_local_affine_rest_warp" if mode != "parent_fallback" else "inherit_mapped_parent_rest_warp",
            "projection_basis": (
                "source_eye_ring_index_01_to_05_upper_06_to_08_lower_with_inner_outer_arc"
                if semantic_family == "eyelid_arc"
                else None
            ),
            "weight_rule": "merge-targets-prune-threshold-0.001-normalize-max4",
            "seam_group": (
                "left_wrist" if primary_target.startswith("Left") and "Wrist" in primary_target else
                "right_wrist" if primary_target.startswith("Right") and "Wrist" in primary_target else
                "left_ankle" if primary_target.startswith("Left") and primary_target in {"LeftFoot", "LeftToeBase"} else
                "right_ankle" if primary_target.startswith("Right") and primary_target in {"RightFoot", "RightToeBase"} else
                None
            ),
            "rationale": {
                "direct_core": "native FBX production chain maps to the matching physical Display role"
                if mode == "explicit" and semantic_family in {"core", "finger"}
                else None,
                "secondary": "source-specific helper or accessory retains semantic family through target corrective/twist or closest driven parent",
                "face": "facial source control is assigned to the retained physical container face family; dense Face donor is semantic-only",
            },
        }
        mappings[source_name] = entry
        method_counts[mode] += 1
        family_counts[semantic_family] += 1

    ordered_entries = [mappings[name] for name in ordered_source_names if name in mappings]
    for entry in ordered_entries:
        if len(entry["targets"]) > 4:
            errors.append({"source": entry["source"], "code": "mapping_target_influence_limit_exceeded"})
    unmapped = sorted(set(usage) - set(mappings))
    errors.extend({"source": name, "code": "unmapped_used_source_bone"} for name in unmapped)
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": f"FBX-first REST mapping for Si {package} Display components.",
        "source_format": "fbx",
        "component_package": package,
        "source": {
            "blend": source_inventory["blend"],
            "armature": source_inventory["armature"]["object"],
            "bone_count": int(source_inventory["armature"]["bone_count"]),
            "topology_sha256": source_inventory["armature"]["topology_sha256"],
            "roles": sorted(roles),
            "lods": sorted(lods),
            "used_source_bone_count": len(usage),
        },
        "donor": {
            "container": container["name"],
            "blend": container["files"]["blend"]["path"],
            "modelbin": container["files"]["modelbin"]["path"],
            "armature": target_inventory["armature"]["object"],
            "bone_count": int(target_inventory["armature"]["bone_count"]),
            "topology_sha256": target_inventory["armature"]["topology_sha256"],
            "required_local_skeleton": True,
        },
        "alignment": transform,
        "chain_warp": {
            "required": True,
            "global_transform_is_initial_only": True,
            "chains": [
                "torso",
                "head_neck",
                "left_arm",
                "right_arm",
                "left_leg",
                "right_leg",
                "left_eye",
                "right_eye",
                "hair_accessory_fallback",
            ],
            "anchors": "source/donor REST bone heads and tails, solved per chain with endpoint-preserving affine warp",
            "seam_constraints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "face_neck"],
        },
        "summary": {
            "mapped_groups": len(ordered_entries),
            "unmapped_groups": len(unmapped) + len(errors),
            "method_counts": dict(sorted(method_counts.items())),
            "semantic_family_counts": dict(sorted(family_counts.items())),
            "max_targets_per_source": max((len(entry["targets"]) for entry in ordered_entries), default=0),
            "weight_prune_threshold_next_stage": 0.001,
            "max_influences_next_stage": 4,
        },
        "mappings": ordered_entries,
        "validation": {"hard_error_count": len(errors), "hard_errors": errors},
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }


def main() -> int:
    args = arguments()
    config_path = args.config.resolve()
    config = load_json(config_path)
    contract_path = resolve(config["display_target"]["contract_output"])
    contract = load_json(contract_path)
    head_output = args.head_output.resolve() if args.head_output else WORKSPACE / "work" / "si" / "fbx-source" / "milestone-02-donor-plan" / "bone-maps" / "head-hair-to-helmet-v004.json"
    body_output = args.body_output.resolve() if args.body_output else WORKSPACE / "work" / "si" / "fbx-source" / "milestone-02-donor-plan" / "bone-maps" / "body-garment-to-outfit-v004.json"
    for path in (head_output, body_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    baseline = load_json(resolve(config["outputs"]["baseline_metadata"]))
    source_inventory = load_json(WORKSPACE / "work" / "si" / "fbx-source" / "milestone-02-donor-plan" / "skeletons" / "source-fbx.skeleton.json")
    lods = {str(lod) for lod in config["primary"]["working_lods"]}
    outputs = config["outputs"]["components"]
    source_components = {}
    for lod in lods:
        report = load_json(resolve(outputs[lod]["report"]))
        source_components[lod] = report

    records: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for key, output, role_set, container_key in (
        ("head_hair", head_output, {"head", "hair"}, "head_hair"),
        ("body_garment", body_output, {"body", "garment"}, "body_garment"),
    ):
        container = contract["physical_containers"][container_key]
        target_inventory_path = resolve(config["display_target"]["physical_containers"][container_key]["skeleton_inventory"])
        target_inventory = load_json(target_inventory_path)
        report = build_map(
            key,
            role_set,
            container,
            baseline,
            source_inventory,
            target_inventory,
            lods,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records.append((output, report, target_inventory))
    hard_errors = sum(int(report["validation"]["hard_error_count"]) for _, report, _ in records)
    print(
        "FH6_FBX_BONE_MAPS="
        + json.dumps(
            {
                "outputs": [str(output) for output, _, _ in records],
                "mapped_groups": {output.stem: report["summary"]["mapped_groups"] for output, report, _ in records},
                "hard_error_count": hard_errors,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
