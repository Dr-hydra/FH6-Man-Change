#!/usr/bin/env python3
"""Validate Si Display LOD0 across isolated extreme limb-chain poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_si_fbx_display_seams as base


BODY_MESH_NAME = "Si_Display_BodyGarment_LOD0"
BODY_ARMATURE_NAME = "FH6_Outfit_Race_Suit_Modern_F_Skeleton"
HEAD_MESH_NAME = "Si_Display_HeadHair_LOD0"
HEAD_ARMATURE_NAME = "FH6_Helmet_Race_Modern_Skeleton"
MATERIALS = ("肌", "Cloth1", "Cloth1Alpha")

POSE_CASES: dict[str, dict[str, object]] = {
    "shoulders": {
        "rotations": {
            "LeftShoulder": (8.0, 0.0, 0.0),
            "RightShoulder": (8.0, 0.0, 0.0),
            "LeftArm": (65.0, 0.0, 0.0),
            "RightArm": (65.0, 0.0, 0.0),
        },
        "regions": ("left_shoulder", "right_shoulder"),
        "views": (
            ("front", ("LeftArm", "RightArm"), (0.0, 1.0, 0.0), 1.55),
            ("rear", ("LeftArm", "RightArm"), (0.0, -1.0, 0.0), 1.55),
        ),
    },
    "elbows": {
        "rotations": {
            "LeftArm": (25.0, 0.0, 0.0),
            "RightArm": (25.0, 0.0, 0.0),
            "LeftForeArm": (75.0, 0.0, 0.0),
            "RightForeArm": (-75.0, 0.0, 0.0),
        },
        "regions": ("left_elbow", "right_elbow"),
        "views": (
            ("front", ("LeftForeArm", "RightForeArm"), (0.0, 1.0, 0.0), 1.45),
            ("rear", ("LeftForeArm", "RightForeArm"), (0.0, -1.0, 0.0), 1.45),
            ("left", ("LeftForeArm",), (0.0, 1.0, 0.0), 0.42),
            ("right", ("RightForeArm",), (0.0, 1.0, 0.0), 0.42),
        ),
    },
    "fingers": {
        "rotations": {
            **{
                f"{side}{finger}{segment}": (angle, 0.0, 0.0)
                for side in ("Left", "Right")
                for finger in ("Index", "Middle", "Ring", "Pinky")
                for segment, angle in ((1, 48.0), (2, 62.0), (3, 52.0))
            },
            **{
                f"{side}Thumb{segment}": (angle, 0.0, 0.0)
                for side in ("Left", "Right")
                for segment, angle in ((1, 30.0), (2, 42.0), (3, 34.0))
            },
        },
        "regions": ("left_hand", "right_hand"),
        "views": (
            ("left", ("LeftMiddle2",), (0.0, 1.0, 0.0), 0.31),
            ("right", ("RightMiddle2",), (0.0, 1.0, 0.0), 0.31),
        ),
    },
    "hips": {
        "rotations": {
            "LeftUpLeg": (48.0, 0.0, 18.0),
            "RightUpLeg": (-32.0, 0.0, -18.0),
        },
        "regions": ("left_hip", "right_hip"),
        "views": (
            ("front", ("LeftUpLeg", "RightUpLeg"), (0.0, 1.0, 0.0), 0.90),
            ("side", ("LeftUpLeg", "RightUpLeg"), (1.0, 0.0, 0.0), 0.90),
        ),
    },
    "knees": {
        "rotations": {
            "LeftUpLeg": (18.0, 0.0, 0.0),
            "RightUpLeg": (18.0, 0.0, 0.0),
            "LeftLeg": (82.0, 0.0, 0.0),
            "RightLeg": (82.0, 0.0, 0.0),
        },
        "regions": ("left_knee", "right_knee"),
        "views": (
            ("front", ("LeftLeg", "RightLeg"), (0.0, 1.0, 0.0), 1.00),
            ("side", ("LeftLeg", "RightLeg"), (1.0, 0.0, 0.0), 1.00),
        ),
    },
    "toes": {
        "rotations": {
            "LeftFoot": (22.0, 0.0, 0.0),
            "RightFoot": (22.0, 0.0, 0.0),
            "LeftToeBase": (38.0, 0.0, 0.0),
            "RightToeBase": (38.0, 0.0, 0.0),
        },
        "regions": ("left_toes", "right_toes"),
        "views": (
            ("left", ("LeftToeBase",), (-1.0, 0.0, 0.0), 0.33),
            ("right", ("RightToeBase",), (1.0, 0.0, 0.0), 0.33),
        ),
    },
}

REGIONS: dict[str, tuple[str, float]] = {
    "left_shoulder": ("LeftArm", 0.235),
    "right_shoulder": ("RightArm", 0.235),
    "left_elbow": ("LeftForeArm", 0.205),
    "right_elbow": ("RightForeArm", 0.205),
    "left_hand": ("LeftHand", 0.190),
    "right_hand": ("RightHand", 0.190),
    "left_hip": ("LeftUpLeg", 0.245),
    "right_hip": ("RightUpLeg", 0.245),
    "left_knee": ("LeftLeg", 0.220),
    "right_knee": ("RightLeg", 0.220),
    "left_toes": ("LeftToeBase", 0.175),
    "right_toes": ("RightToeBase", 0.175),
}

LIMITS = {
    "edge_p95": 1.60,
    "edge_p99": 2.50,
    "area_p95": 2.50,
    "area_p99": 4.00,
    "catastrophic_edge_ratio": 8.00,
    "catastrophic_edge_delta_mm": 15.00,
    "collapsed_quarter_fraction": 0.01,
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-blend", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    factor = position - low
    return ordered[low] * (1.0 - factor) + ordered[high] * factor


def scalar_summary(values: list[float], scale: float = 1.0, suffix: str = "") -> dict[str, object]:
    result = base.scalar_stats(values, scale, suffix)
    if values:
        result[f"p99{suffix}"] = round(percentile(values, 0.99) * scale, 6)
    return result


def material_vertices(obj: bpy.types.Object) -> dict[str, set[int]]:
    return {name: base.material_vertex_ids(obj, name) for name in MATERIALS}


def vertex_weights(obj: bpy.types.Object, index: int) -> dict[str, float]:
    group_names = {group.index: group.name for group in obj.vertex_groups}
    return {
        group_names[item.group]: round(float(item.weight), 6)
        for item in obj.data.vertices[index].groups
        if item.weight > 1.0e-6 and item.group in group_names
    }


def deformation_stats(
    obj: bpy.types.Object,
    rest_coordinates: list[Vector],
    pose_coordinates: list[Vector],
    selected: set[int],
    material_name: str | None,
) -> dict[str, object]:
    material_index = base.material_index(obj, material_name) if material_name else None
    polygons = [
        (polygon.index, tuple(polygon.vertices))
        for polygon in obj.data.polygons
        if (material_index is None or polygon.material_index == material_index)
        and all(index in selected for index in polygon.vertices)
    ]
    edges: set[tuple[int, int]] = set()
    for _polygon_index, vertices in polygons:
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            edges.add(tuple(sorted((first, second))))

    edge_ratios: list[float] = []
    edge_outliers: list[dict[str, object]] = []
    for first, second in sorted(edges):
        rest_length = (rest_coordinates[first] - rest_coordinates[second]).length
        pose_length = (pose_coordinates[first] - pose_coordinates[second]).length
        if rest_length <= 1.0e-8 or pose_length <= 1.0e-8:
            continue
        ratio = max(rest_length / pose_length, pose_length / rest_length)
        edge_ratios.append(ratio)
        edge_outliers.append(
            {
                "vertices": [first, second],
                "ratio": round(ratio, 6),
                "rest_length_mm": round(rest_length * 1000.0, 6),
                "pose_length_mm": round(pose_length * 1000.0, 6),
                "absolute_delta_mm": round(abs(rest_length - pose_length) * 1000.0, 6),
            }
        )

    area_ratios: list[float] = []
    triangle_outliers: list[dict[str, object]] = []
    collapsed_quarter = 0
    collapsed_half = 0
    expanded_double = 0
    expanded_quadruple = 0
    for polygon_index, vertices in polygons:
        rest_area = base.triangle_area(rest_coordinates, vertices)
        pose_area = base.triangle_area(pose_coordinates, vertices)
        if rest_area <= 1.0e-10 or pose_area <= 1.0e-10:
            continue
        ratio = pose_area / rest_area
        collapsed_quarter += int(ratio < 0.25)
        collapsed_half += int(ratio < 0.50)
        expanded_double += int(ratio > 2.0)
        expanded_quadruple += int(ratio > 4.0)
        symmetric_ratio = max(rest_area / pose_area, pose_area / rest_area)
        area_ratios.append(symmetric_ratio)
        triangle_outliers.append(
            {
                "polygon": polygon_index,
                "vertices": list(vertices),
                "ratio": round(symmetric_ratio, 6),
                "signed_pose_to_rest_ratio": round(ratio, 6),
                "rest_area_mm2": round(rest_area * 1_000_000.0, 6),
                "pose_area_mm2": round(pose_area * 1_000_000.0, 6),
            }
        )

    non_finite = sum(
        any(not math.isfinite(value) for value in pose_coordinates[index]) for index in selected
    )
    catastrophic_edges = sum(
        item["ratio"] > LIMITS["catastrophic_edge_ratio"]
        and item["absolute_delta_mm"] > LIMITS["catastrophic_edge_delta_mm"]
        for item in edge_outliers
    )
    displacements = [(pose_coordinates[index] - rest_coordinates[index]).length for index in selected]
    return {
        "material": material_name or "all",
        "vertices": len(selected),
        "polygons": len(polygons),
        "edges": len(edges),
        "non_finite_vertices": non_finite,
        "catastrophic_edges": catastrophic_edges,
        "displacement": scalar_summary(displacements, 1000.0, "_mm"),
        "edge_symmetric_stretch": scalar_summary(edge_ratios),
        "triangle_symmetric_area_change": scalar_summary(area_ratios),
        "triangles_below_quarter_area": collapsed_quarter,
        "triangles_below_half_area": collapsed_half,
        "triangles_above_double_area": expanded_double,
        "triangles_above_quadruple_area": expanded_quadruple,
        "worst_edges": [
            {
                **item,
                "weights": {
                    str(index): vertex_weights(obj, index) for index in item["vertices"]
                },
            }
            for item in sorted(edge_outliers, key=lambda value: value["ratio"], reverse=True)[:5]
        ],
        "worst_triangles": [
            {
                **item,
                "weights": {
                    str(index): vertex_weights(obj, index) for index in item["vertices"]
                },
            }
            for item in sorted(triangle_outliers, key=lambda value: value["ratio"], reverse=True)[:5]
        ],
    }


def metric_pass(stats: dict[str, object]) -> tuple[bool | None, list[str]]:
    edge = stats["edge_symmetric_stretch"]
    area = stats["triangle_symmetric_area_change"]
    if stats["vertices"] < 12 or stats["edges"] < 12 or stats["polygons"] < 6:
        return None, []
    reasons = []
    if stats["non_finite_vertices"]:
        reasons.append("non-finite posed coordinates")
    if not edge.get("measurable") or not area.get("measurable"):
        reasons.append("deformation is not measurable")
    else:
        if edge["p95"] > LIMITS["edge_p95"]:
            reasons.append(f"edge p95 {edge['p95']:.6f} > {LIMITS['edge_p95']:.2f}")
        if edge["p99"] > LIMITS["edge_p99"]:
            reasons.append(f"edge p99 {edge['p99']:.6f} > {LIMITS['edge_p99']:.2f}")
        if area["p95"] > LIMITS["area_p95"]:
            reasons.append(f"area p95 {area['p95']:.6f} > {LIMITS['area_p95']:.2f}")
        if area["p99"] > LIMITS["area_p99"]:
            reasons.append(f"area p99 {area['p99']:.6f} > {LIMITS['area_p99']:.2f}")
    if stats["catastrophic_edges"]:
        reasons.append(
            f"catastrophic edges {stats['catastrophic_edges']} > 0 "
            f"(ratio > {LIMITS['catastrophic_edge_ratio']:.2f} and delta > "
            f"{LIMITS['catastrophic_edge_delta_mm']:.2f} mm)"
        )
    allowed_collapsed = max(
        2, math.ceil(stats["polygons"] * LIMITS["collapsed_quarter_fraction"])
    )
    if stats["triangles_below_quarter_area"] > allowed_collapsed:
        reasons.append(
            f"quarter-area triangles {stats['triangles_below_quarter_area']} > {allowed_collapsed}"
        )
    return not reasons, reasons


def apply_pose(armature: bpy.types.Object, rotations: dict[str, tuple[float, float, float]]) -> None:
    base.reset_pose(armature)
    for name, degrees in rotations.items():
        bone = armature.pose.bones.get(name)
        if bone is None:
            raise RuntimeError(f"Missing extreme-pose bone {name!r}")
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = tuple(math.radians(value) for value in degrees)
    bpy.context.view_layer.update()


def render_case(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    case_name: str,
    views: tuple[tuple[str, tuple[str, ...], tuple[float, float, float], float], ...],
    outputs: dict[str, Path],
) -> None:
    for view_name, bones, direction, distance in views:
        target = base.midpoint(base.pose_bone_point(armature, name) for name in bones)
        base.render_region(
            scene,
            camera,
            target,
            Vector(direction),
            distance,
            outputs[f"{case_name}-{view_name}"],
        )


def main() -> int:
    args = arguments()
    input_blend = args.input_blend.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    if Path(bpy.data.filepath).resolve() != input_blend:
        raise RuntimeError(f"Open blend {bpy.data.filepath!r} does not match {input_blend}")

    outputs = {
        f"{case_name}-{view[0]}": output_dir / f"{case_name}-{view[0]}.png"
        for case_name, case in POSE_CASES.items()
        for view in case["views"]
    }
    base.ensure_outputs_absent([*outputs.values(), report_path])

    body_mesh = bpy.data.objects.get(BODY_MESH_NAME)
    body_armature = bpy.data.objects.get(BODY_ARMATURE_NAME)
    head_mesh = bpy.data.objects.get(HEAD_MESH_NAME)
    head_armature = bpy.data.objects.get(HEAD_ARMATURE_NAME)
    if None in (body_mesh, body_armature, head_mesh, head_armature):
        raise RuntimeError("Validation blend is missing a Display mesh or armature")

    base.reset_pose(body_armature)
    base.reset_pose(head_armature)
    bpy.context.view_layer.update()
    rest_coordinates = base.evaluated_coordinates(body_mesh)
    rest_centers = {
        region: base.pose_bone_point(body_armature, bone_name)
        for region, (bone_name, _radius) in REGIONS.items()
    }
    region_vertices = {
        region: {
            index
            for index, point in enumerate(rest_coordinates)
            if (point - rest_centers[region]).length <= radius
        }
        for region, (_bone_name, radius) in REGIONS.items()
    }
    by_material = material_vertices(body_mesh)

    scene = bpy.context.scene
    camera = base.configure_render(scene, (head_mesh, body_mesh))
    head_mesh.hide_render = True
    pose_metrics: dict[str, object] = {}
    failures: list[str] = []
    gates: dict[str, object] = {}

    for case_name, case in POSE_CASES.items():
        apply_pose(body_armature, case["rotations"])
        pose_coordinates = base.evaluated_coordinates(body_mesh)
        case_metrics: dict[str, object] = {}
        case_pass = True
        case_reasons: list[str] = []
        for region in case["regions"]:
            selected = region_vertices[region]
            subsets = {"all": selected}
            subsets.update({name: selected & by_material[name] for name in MATERIALS})
            region_metrics = {
                name: deformation_stats(
                    body_mesh,
                    rest_coordinates,
                    pose_coordinates,
                    subset,
                    None if name == "all" else name,
                )
                for name, subset in subsets.items()
            }
            region_results = {}
            measurable = 0
            for name, stats in region_metrics.items():
                passed, reasons = metric_pass(stats)
                region_results[name] = {"pass": passed, "reasons": reasons}
                if passed is not None:
                    measurable += 1
                    if not passed:
                        case_pass = False
                        for reason in reasons:
                            case_reasons.append(f"{region}/{name}: {reason}")
            if measurable == 0:
                case_pass = False
                case_reasons.append(f"{region}: no measurable topology")
            case_metrics[region] = {
                "center_bone": REGIONS[region][0],
                "selection_radius_m": REGIONS[region][1],
                "rest_center_m": [round(value, 9) for value in rest_centers[region]],
                "metrics": region_metrics,
                "gates": region_results,
            }
        render_case(scene, camera, body_armature, case_name, case["views"], outputs)
        pose_metrics[case_name] = case_metrics
        gates[case_name] = {
            "pass": case_pass,
            "rule": "per measurable material: edge p95/p99 <= 1.60/2.50, triangle area p95/p99 <= 2.50/4.00, no edge with ratio > 8 and absolute delta > 15 mm, <= 1% severe quarter-area collapse, and finite posed coordinates; raw maxima remain diagnostic",
            "failures": case_reasons,
        }
        failures.extend(f"{case_name}: {reason}" for reason in case_reasons)

    base.reset_pose(body_armature)
    base.reset_pose(head_armature)
    bpy.context.view_layer.update()

    report = {
        "schema_version": 1,
        "created_local": datetime.now().astimezone().isoformat(),
        "purpose": "Si FBX Display LOD0 isolated extreme-pose validation for full limb chains.",
        "input": {"blend": str(input_blend), "sha256": sha256(input_blend)},
        "pose_degrees_xyz": {
            name: {bone: list(values) for bone, values in case["rotations"].items()}
            for name, case in POSE_CASES.items()
        },
        "region_contract": {
            name: {"center_bone": bone, "radius_m": radius}
            for name, (bone, radius) in REGIONS.items()
        },
        "limits": LIMITS,
        "metrics": pose_metrics,
        "diagnostic_gates": gates,
        "diagnostic_failures": failures,
        "renders": {name: str(path) for name, path in outputs.items()},
        "validation_level": {
            "structural_inputs": True,
            "blender_extreme_poses": True,
            "modelbin": False,
            "offline_game": False,
        },
        "threshold_note": "Conservative LOD0 diagnostics for local deformation under isolated extreme poses; visual renders remain required evidence.",
        "license_guard": "Local technical validation only; do not redistribute the split Si component.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_SI_EXTREME_POSE_VALIDATION="
        + json.dumps(
            {"report": str(report_path), "renders": len(outputs), "failures": len(failures)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
