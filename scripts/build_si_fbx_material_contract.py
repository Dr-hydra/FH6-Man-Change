#!/usr/bin/env python3
"""Build the FBX-first Si Display material and texture channel contract.

This is an audit/manifest generator.  It does not write a modelbin, swatchbin,
ZIP, or anything under the game installation.  Source PNGs are measured as
read-only inputs and donor MatI blobs are inspected only for shader/layout
evidence.  Final donor texture bindings are deliberately prohibited here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from modelbin_bundle import parse_bundle  # noqa: E402
from patch_fh6_racesuit_materials import (  # noqa: E402
    decode_7bit,
    material_id,
    material_name,
    parameter_end,
)


CONTRACT_VERSION = 1
DEFAULT_CONFIG = WORKSPACE / "sources" / "si" / "source.config.json"
DEFAULT_OUTPUT_DIR = WORKSPACE / "work" / "si" / "fbx-source" / "material-contract-v001"

FORBIDDEN_DONOR_TOKENS = (
    "alice",
    "average_kim",
    "race_suit",
    "race_helmet",
    "branding",
    "logo",
    "stitch",
    "detailnormal",
    "micronormal",
)

UNBOUND_CORE_ALLOWLIST = {
    "T_actor_common_matcap_08_D.png": "Source preview matcap only; exclude from FH6 PBR/runtime response to avoid the previous plastic look."
}

NEUTRAL_RESPONSE_BASELINE = {
    "main_tint_rgba": [1.0, 1.0, 1.0, 1.0],
    "roughness": 0.6,
    "metallic": 0.0,
    "specular_level": 0.25,
    "emission_strength": 0.0,
    "detail_normal": False,
    "logo": False,
    "stitch": False,
    "pattern": False,
    "donor_material_response": False,
}

CHANNEL_RESTORE_ORDER = [
    "source base color and role-specific raw Alpha",
    "source N or HN normal convention",
    "source P/RS/RD packed response",
    "source ST/M/SDF auxiliary response",
    "source E emission last",
]

NORMAL_XY_B_ZERO = {
    "encoding": "tangent_xy_unorm8_with_blue_zero",
    "channels": "R/G encode signed tangent X/Y; source B is a constant zero pad",
    "decode": "ignore B; convert R/G from [0,1] to [-1,1] and reconstruct normalized Z",
    "warning": "Sampling B as a signed normal component produces an inverted/black normal response",
}

SUFFIX_SEMANTICS: dict[str, dict[str, Any]] = {
    "D": {
        "meaning": "base color, diffuse, or color atlas",
        "intended_color_space": "sRGB",
        "alpha_policy": "preserve_raw_8bit; usage is role-specific and never premultiply",
    },
    "N": {
        "meaning": "tangent normal, two-channel XY packing for Si role textures",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; normally opaque padding",
        "normal_encoding": NORMAL_XY_B_ZERO,
    },
    "P": {
        "meaning": "packed material/property response map",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; packed data, not surface opacity by default",
    },
    "E": {
        "meaning": "emission/self-illumination response",
        "intended_color_space": "sRGB",
        "alpha_policy": "preserve_raw_8bit; shader decides contribution, never discard",
    },
    "ST": {
        "meaning": "style, tint, stitch, or detail response map",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; packed data",
    },
    "HN": {
        "meaning": "hair normal plus auxiliary packed channels",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; alpha is an auxiliary hair response, not discarded",
        "normal_encoding": {
            "encoding": "hair_rgba_packed; source B is not a zero pad",
            "channels": "retain all four source channels until the FH6 hair shader contract is written",
            "decode": "do not apply the N-map B=0 reconstruction to HN",
        },
    },
    "RS": {
        "meaning": "roughness/specular or response swatch",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; packed response data",
    },
    "M": {
        "meaning": "mask, material selector, or overlay response",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; mask channels remain available to the shader",
    },
    "RD": {
        "meaning": "ramp/lookup response data",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; ramp alpha is data, not implicit opacity",
    },
    "SDF": {
        "meaning": "signed-distance face response",
        "intended_color_space": "Non-Color",
        "alpha_policy": "preserve_raw_8bit; alpha is part of the source field",
    },
}


# Every source FBX material is covered exactly once.  The Sclera family is a
# deliberate generated draw: the FBX contains an eye-shadow shell but no
# separate opaque-white sclera material, so the geometry stage must provide it.
MATERIAL_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "skin_face",
        "component": "Head",
        "source_materials": ["M_actor_jsspsi_face_01"],
        "shader": "characterskin",
        "shader_template": "DRV_BA_F_01_Face:Mat_Head",
        "physical_container": "Helmet_Race_Modern",
        "render": {"surface": "opaque", "alpha_mode": "opaque_data_alpha", "atst": "0000"},
        "channels": [
            ("base_color", "T_actor_jsspsi_face_01_D.png", "sRGB", "auxiliary_data_alpha"),
            ("normal_xy", "T_actor_jsspsi_face_01_N.png", "Non-Color", "opaque_padding"),
            ("style", "T_actor_jsspsi_face_01_ST.png", "Non-Color", "packed_data"),
            ("face_ramp", "T_actor_common_face_01_RD.png", "Non-Color", "packed_data"),
            ("face_mask", "T_actor_common_female_face_01_cm_M.png", "Non-Color", "packed_data"),
            ("face_sdf", "T_actor_common_female_face_01_SDF.png", "Non-Color", "packed_data"),
            ("emotion_atlas", "T_actor_common_female_emotion_atlas_01_D.png", "sRGB", "atlas_alpha"),
            ("skin_lut", "T_actor_common_femaleskincolor01_lut_D.png", "sRGB", "opaque"),
        ],
        "notes": "Face D alpha is retained as source data; it must not turn the face translucent.",
    },
    {
        "id": "skin_body",
        "component": "Body",
        "source_materials": ["M_actor_jsspsi_body_01"],
        "shader": "characterskin",
        "shader_template": "Female:Mat_Body",
        "physical_container": "Outfit_Race_Suit_Modern_F",
        "render": {"surface": "opaque", "alpha_mode": "opaque_data_alpha", "atst": "0000"},
        "channels": [
            ("base_color", "T_actor_jsspsi_body_01_D.png", "sRGB", "auxiliary_data_alpha"),
            ("normal_xy", "T_actor_jsspsi_body_01_N.png", "Non-Color", "opaque_padding"),
            ("body_ramp", "T_actor_common_body_01_RD.png", "Non-Color", "packed_data"),
            ("skin_lut", "T_actor_common_femaleskincolor03_lut_D.png", "sRGB", "opaque"),
        ],
        "notes": "Body D alpha is preserved but is not a transparency switch.",
    },
    {
        "id": "hair_brow",
        "component": "Hair",
        "source_materials": ["M_actor_jsspsi_hair_01", "M_actor_jsspsi_brow_01"],
        "shader": "hairalphatestnotangentgradmap",
        "shader_template": "Driver_Alice_F:Mat_Eyelashes",
        "physical_container": "Helmet_Race_Modern",
        "render": {"surface": "alpha_cutout", "alpha_mode": "texture_alpha_test", "atst": "0100"},
        "channels": [
            ("base_color", "T_actor_jsspsi_hair_01_D.png", "sRGB", "alpha_cutout"),
            ("hair_normal", "T_actor_jsspsi_hair_01_HN.png", "Non-Color", "auxiliary_data_alpha"),
            ("property", "T_actor_jsspsi_hair_01_P.png", "Non-Color", "packed_data_alpha"),
            ("response", "T_actor_jsspsi_hair_01_RS.png", "Non-Color", "packed_data"),
            ("style", "T_actor_jsspsi_hair_01_ST.png", "Non-Color", "packed_data"),
            ("hairline_mask", "T_actor_common_hairline_03_M.png", "Non-Color", "packed_data"),
            ("hair_ramp", "T_actor_common_hair_03_RD.png", "Non-Color", "packed_data_alpha"),
            ("hairstyle_response", "T_actor_common_hairst_01_ST.png", "Non-Color", "packed_data"),
        ],
        "notes": "Brow shares the alpha-hair shader family to fit the six-slot Helmet container; its UV/tint remap remains a geometry/material validation gate.",
    },
    {
        "id": "hair_shadow",
        "component": "Hair",
        "source_materials": ["M_hairshadow_common_01"],
        "shader": "eyesao",
        "shader_template": "DRV_BA_F_01_Face:Mat_Eyelashes",
        "physical_container": "Helmet_Race_Modern",
        "render": {"surface": "overlay", "alpha_mode": "mask_overlay", "atst": "0001"},
        "channels": [
            ("mask", "T_actor_common_hairshadow_01_M.png", "Non-Color", "opaque_mask"),
            ("hair_ramp", "T_actor_common_hair_03_RD.png", "Non-Color", "packed_data_alpha"),
        ],
        "notes": "Hair shadow remains a separate overlay draw; no helmet visor/rubber response is allowed.",
    },
    {
        "id": "eye_shadow",
        "component": "Head",
        "source_materials": ["M_eyewhiteshadow_common_01"],
        "shader": "eyesao",
        "shader_template": "DRV_BA_F_01_Face:Mat_Eyelashes",
        "physical_container": "Helmet_Race_Modern",
        "render": {"surface": "overlay", "alpha_mode": "mask_overlay", "atst": "0001"},
        "channels": [
            ("mask", "T_actor_common_eyeshadow_01_M.png", "Non-Color", "opaque_mask"),
            ("face_ramp", "T_actor_common_face_01_RD.png", "Non-Color", "packed_data"),
        ],
        "notes": "This 38-vertex source shell is an eye-white shadow/occlusion layer, not the required opaque sclera.",
    },
    {
        "id": "iris",
        "component": "Head",
        "source_materials": ["M_actor_jsspsi_iris_01"],
        "shader": "charactereye",
        "shader_template": "DRV_BA_F_01_Face:Mat_Eye",
        "physical_container": "Helmet_Race_Modern",
        "render": {"surface": "eye_iris", "alpha_mode": "texture_alpha", "atst": "0000"},
        "channels": [
            ("iris_color", "T_actor_jsspsi_iris_01_D.png", "sRGB", "independent_texture_alpha"),
            ("sclera_white_fallback", "common_deco_white.png", "sRGB", "opaque_only_for_generated_sclera"),
        ],
        "notes": "Iris alpha must remain an independent sampler path; no flat-color replacement is permitted.",
    },
    {
        "id": "sclera",
        "component": "Head",
        "source_materials": [],
        "shader": "charactereyeball",
        "shader_template": "Driver_Alice_F:Mat_Eyes",
        "physical_container": "Helmet_Race_Modern",
        "render": {"surface": "opaque", "alpha_mode": "opaque_white", "atst": "0000"},
        "channels": [
            ("opaque_white", "common_deco_white.png", "sRGB", "forced_opaque")
        ],
        "notes": "Synthetic material required because the FBX has no separate opaque-white sclera material; geometry must provide a sclera shell.",
    },
    {
        "id": "cloth_opaque",
        "component": "Garment",
        "source_materials": ["M_actor_jsspsi_cloth_01", "M_actor_jsspsi_cloth_03"],
        "shader": "uber_clothing",
        "shader_template": "Outfit_Race_Suit_Modern_F:Mat_Suit",
        "physical_container": "Outfit_Race_Suit_Modern_F",
        "render": {"surface": "opaque", "alpha_mode": "opaque_data_alpha", "atst": "0100"},
        "channels": [
            ("base_color", "T_actor_jsspsi_cloth_01_D.png", "sRGB", "opaque_data_alpha"),
            ("emission", "T_actor_jsspsi_cloth_01_E.png", "sRGB", "packed_data"),
            ("normal_xy", "T_actor_jsspsi_cloth_01_N.png", "Non-Color", "opaque_padding"),
            ("property", "T_actor_jsspsi_cloth_01_P.png", "Non-Color", "packed_data_alpha"),
            ("style", "T_actor_common_cloth_02_RS.png", "Non-Color", "packed_data"),
            ("mask", "T_actor_jsspsi_cloth_03_M.png", "Non-Color", "opaque_selector"),
            ("response_ramp", "T_actor_common_cloth_04_RD.png", "Non-Color", "packed_data_alpha"),
            ("response_swatch", "T_actor_common_cloth_04_RS.png", "Non-Color", "packed_data"),
        ],
        "notes": "Cloth 03 is a masked decoration layer sharing the opaque-cloth shader family; all donor logo/stitch/detail bindings must be replaced.",
    },
    {
        "id": "cloth_alpha",
        "component": "Garment",
        "source_materials": ["M_actor_jsspsi_cloth_02"],
        "shader": "fh6_alpha_cloth_blend",
        "shader_template": None,
        "physical_container": "Outfit_Race_Suit_Modern_F",
        "render": {"surface": "alpha_cloth", "alpha_mode": "source_texture_alpha", "atst": "template_required"},
        "channels": [
            ("base_color", "T_actor_jsspsi_cloth_02_D.png", "sRGB", "source_alpha_blend_or_cutout"),
            ("emission", "T_actor_jsspsi_cloth_02_E.png", "sRGB", "packed_data"),
            ("normal_xy", "T_actor_jsspsi_cloth_02_N.png", "Non-Color", "opaque_padding"),
            ("property", "T_actor_jsspsi_cloth_02_P.png", "Non-Color", "packed_data_alpha"),
            ("response_swatch", "T_actor_common_cloth_02_RS.png", "Non-Color", "packed_data"),
        ],
        "notes": "Current donors expose uber_clothing/glass/hair-alpha but no verified alpha-cloth blend template; keep this as an explicit blocker instead of silently making the layer opaque.",
    },
]


SEMANTIC_MODELBINS = {
    "DRV_BA_F_01_Face": "work/si/components/baselines/head-face/extracted/DRV_BA_F_01_Face.modelbin",
    "Driver_Alice_F": "work/si/components/baselines/head-alice-f/extracted/Driver_Alice_F.modelbin",
    "Female": "work/si/components/baselines/body/extracted/Female.modelbin",
    "Hair_Bald": "samples/modelbin/Hair_Bald.modelbin",
}

LOD_MATERIAL_ALIASES = {
    "M_actor_lod_jsspsi_body_01": "M_actor_jsspsi_body_01",
    "M_actor_lod_jsspsi_cloth_01": "M_actor_jsspsi_cloth_01",
    "M_actor_lod_jsspsi_cloth_02": "M_actor_jsspsi_cloth_02",
    "M_actor_lod_jsspsi_cloth_03": "M_actor_jsspsi_cloth_03",
    "M_actor_lod_jsspsi_face_01": "M_actor_jsspsi_face_01",
    "M_actor_lod_jsspsi_hair_01": "M_actor_jsspsi_hair_01",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def suffix_for(filename: str) -> str | None:
    match = re.search(r"_([A-Za-z]+)\.png$", filename)
    return match.group(1).upper() if match else None


def alpha_audit(image: Image.Image) -> dict[str, Any]:
    if "A" not in image.getbands():
        return {"present": False, "preserved": True}
    alpha = image.getchannel("A")
    histogram = alpha.histogram()
    pixels = image.width * image.height
    nonopaque = sum(histogram[:255])
    nonzero = sum(histogram[1:])
    return {
        "present": True,
        "preserved": True,
        "min": int(alpha.getextrema()[0]),
        "max": int(alpha.getextrema()[1]),
        "unique_values": sum(value > 0 for value in histogram),
        "nonopaque_fraction": round(nonopaque / pixels, 9),
        "nonzero_fraction": round(nonzero / pixels, 9),
        "premultiply": False,
    }


def image_audit(path: Path, source_root: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        bands = image.getbands()
        extrema = image.getextrema()
        stat = ImageStat.Stat(image)
        channel_stats = {
            band: {
                "min": int(extrema[index][0]),
                "max": int(extrema[index][1]),
                "mean": round(float(stat.mean[index]), 6),
            }
            for index, band in enumerate(bands)
        }
        suffix = suffix_for(path.name)
        normal_audit: dict[str, Any] | None = None
        if suffix == "N" and "B" in bands:
            blue = image.getchannel("B")
            normal_audit = {
                **NORMAL_XY_B_ZERO,
                "blue_min": int(blue.getextrema()[0]),
                "blue_max": int(blue.getextrema()[1]),
                "blue_is_zero_pad": blue.getextrema() == (0, 0),
            }
        elif suffix == "HN" and "B" in bands:
            blue = image.getchannel("B")
            normal_audit = {
                "encoding": "hair_rgba_packed",
                "blue_min": int(blue.getextrema()[0]),
                "blue_max": int(blue.getextrema()[1]),
                "blue_is_zero_pad": blue.getextrema() == (0, 0),
                "must_not_use_xy_b_zero_decode": True,
            }
        info = {
            "path": path.relative_to(source_root).as_posix(),
            "absolute_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
            "format": image.format,
            "mode": image.mode,
            "size": [image.width, image.height],
            "bands": list(bands),
            "source_png_color_profile": sorted(str(key) for key in image.info if key in {"srgb", "gamma", "icc_profile"}),
            "suffix": suffix,
            "suffix_semantics": SUFFIX_SEMANTICS.get(suffix, {"meaning": "unclassified source support"}),
            "channels": channel_stats,
            "alpha": alpha_audit(image),
            "normal": normal_audit,
        }
        return info


def parse_matti(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    outer = parse_bundle(data)
    materials = []
    for blob in outer.blobs:
        if blob.tag != "MatI":
            continue
        nested = parse_bundle(blob.data)
        mati = next((item for item in nested.blobs if item.tag == "MATI"), None)
        mtpr = next((item for item in nested.blobs if item.tag == "MTPR"), None)
        if mati is None or mtpr is None:
            raise ValueError(f"MatI {blob.index} in {path} lacks MATI/MTPR")
        shader = next((meta.value.decode("utf-8") for meta in mati.metadata if meta.tag == "Name"), None)
        atst = next((meta.value.hex() for meta in mati.metadata if meta.tag == "ATST"), None)
        params: list[dict[str, Any]] = []
        payload = mtpr.data
        offset = 1
        for _ in range(payload[0]):
            end, name_hash, parameter_type, value_offset = parameter_end(payload, offset)
            record: dict[str, Any] = {"hash": f"{name_hash:08x}", "type": parameter_type}
            if parameter_type == 6:
                length, string_offset = decode_7bit(payload, value_offset)
                record["texture_path"] = payload[string_offset : string_offset + length].decode("utf-8")
                record["forbidden_tokens"] = [
                    token for token in FORBIDDEN_DONOR_TOKENS if token in record["texture_path"].lower()
                ]
            params.append(record)
            offset = end
        materials.append(
            {
                "id": material_id(blob),
                "name": material_name(blob),
                "shader": shader,
                "atst": atst,
                "parameter_count": int(payload[0]),
                "parameters": params,
                "texture_parameters": [item for item in params if "texture_path" in item],
                "payload_sha256": sha256_bytes(blob.data),
            }
        )
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "bundle": {"tag": "Grub", "version": list(outer.version), "blob_count": len(outer.blobs)},
        "materials": sorted(materials, key=lambda item: int(item["id"])),
    }


def source_mesh_materials(config: dict[str, Any], workspace: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    components = config.get("outputs", {}).get("components", {})
    for lod, spec in sorted(components.items()):
        report_value = spec.get("report")
        if not report_value:
            continue
        report_path = resolve_path(workspace, report_value)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for mesh in report.get("meshes", []):
            for material in mesh.get("materials", []):
                entry = result.setdefault(
                    str(material),
                    {"material": str(material), "roles": set(), "lods": set(), "objects": [], "vertices": 0},
                )
                entry["roles"].add(str(mesh.get("role", "unknown")))
                entry["lods"].add(str(lod))
                entry["vertices"] += int(mesh.get("vertices", 0))
                entry["objects"].append(
                    {
                        "lod": str(lod),
                        "object": mesh.get("object"),
                        "role": mesh.get("role"),
                        "vertices": int(mesh.get("vertices", 0)),
                    }
                )
    for entry in result.values():
        entry["roles"] = sorted(entry["roles"])
        entry["lods"] = sorted(entry["lods"])
    return result


def build_contract(config_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    workspace = WORKSPACE.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    primary = config.get("primary", {})
    fbx_path = resolve_path(workspace, primary.get("path", ""))
    if not fbx_path.is_file():
        raise FileNotFoundError(fbx_path)
    source_root = fbx_path.parent
    baseline_report_path = resolve_path(
        workspace,
        config.get("outputs", {}).get("baseline_metadata", ""),
    )
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    source_meshes = source_mesh_materials(config, workspace)

    images: dict[str, dict[str, Any]] = {}
    for path in sorted(source_root.glob("*.png")):
        images[path.name] = image_audit(path, source_root)
    core_images = {name: info for name, info in images.items() if not name.startswith("T_fx_")}
    missing_images = sorted(
        {
            filename
            for family in MATERIAL_FAMILIES
            for _slot, filename, _space, _alpha in family["channels"]
            if filename not in core_images
        }
    )

    family_entries: list[dict[str, Any]] = []
    source_material_coverage: dict[str, list[str]] = Counter()
    for family in MATERIAL_FAMILIES:
        channels = []
        for slot, filename, color_space, alpha_usage in family["channels"]:
            image = core_images.get(filename)
            channel = {
                "slot": slot,
                "source_file": filename,
                "intended_color_space": color_space,
                "alpha_usage": alpha_usage,
                "raw_alpha_required": True,
                "premultiply": False,
                "write_status": "source_verified_contract_only",
                "fh6_output_written": False,
            }
            if image:
                channel.update(
                    {
                        "source_sha256": image["sha256"],
                        "source_size": image["size"],
                        "source_mode": image["mode"],
                        "alpha_audit": image["alpha"],
                        "normal_audit": image["normal"],
                    }
                )
            channels.append(channel)
        for material in family["source_materials"]:
            source_material_coverage.setdefault(material, []).append(family["id"])
        family_entries.append(
            {
                "id": family["id"],
                "component": family["component"],
                "source_materials": family["source_materials"],
                "shader": {
                    "required_class": family["shader"],
                    "semantic_template": family["shader_template"],
                    "template_status": "verified_by_donor_inventory" if family["shader_template"] else "missing_template",
                    "physical_container": family["physical_container"],
                    "texture_bindings": "fresh_generated_si_swatch_only",
                },
                "render": family["render"],
                "channels": channels,
                "neutral_response_baseline": NEUTRAL_RESPONSE_BASELINE,
                "restore_order": CHANNEL_RESTORE_ORDER,
                "notes": family["notes"],
                "conversion_status": "contract_only_swatch_and_matti_writer_pending",
                "write_readiness": {
                    "source_channels_audited": True,
                    "fh6_swatchbin_written": False,
                    "fresh_matti_written": False,
                    "ready_for_game": False,
                },
            }
        )

    physical_paths = {
        "Helmet_Race_Modern": resolve_path(workspace, config["display_target"]["physical_containers"]["head_hair"]["modelbin"]),
        "Outfit_Race_Suit_Modern_F": resolve_path(workspace, config["display_target"]["physical_containers"]["body_garment"]["modelbin"]),
    }
    donor_inventory = {name: parse_matti(path) for name, path in physical_paths.items()}
    semantic_inventory = {}
    for name, relative in SEMANTIC_MODELBINS.items():
        path = resolve_path(workspace, relative)
        if path.is_file():
            semantic_inventory[name] = parse_matti(path)

    template_index = {}
    for donor_name, donor in {**donor_inventory, **semantic_inventory}.items():
        for material in donor["materials"]:
            template_index[f"{donor_name}:{material['name']}"] = material["shader"]
    template_mismatches = []
    for family in MATERIAL_FAMILIES:
        template = family["shader_template"]
        if template is None:
            continue
        actual_shader = template_index.get(template)
        if actual_shader != family["shader"]:
            template_mismatches.append(
                {"family": family["id"], "template": template, "expected": family["shader"], "actual": actual_shader}
            )

    donor_refs = []
    for donor_name, donor in {**donor_inventory, **semantic_inventory}.items():
        for material in donor["materials"]:
            for parameter in material["texture_parameters"]:
                tokens = parameter.get("forbidden_tokens", [])
                if tokens:
                    donor_refs.append(
                        {
                            "donor": donor_name,
                            "material_id": material["id"],
                            "material": material["name"],
                            "shader": material["shader"],
                            "texture_path": parameter["texture_path"],
                            "forbidden_tokens": tokens,
                        }
                    )

    expected_materials = {
        material
        for family in MATERIAL_FAMILIES
        for material in family["source_materials"]
    }
    actual_materials = set(source_meshes)
    actual_canonical_materials = {LOD_MATERIAL_ALIASES.get(material, material) for material in actual_materials}
    missing_materials = sorted(expected_materials - actual_canonical_materials)
    unexpected_materials = sorted(actual_canonical_materials - expected_materials)
    duplicate_coverage = sorted(
        material for material, families in source_material_coverage.items() if len(families) != 1
    )
    normal_n_images = [info for info in core_images.values() if info.get("suffix") == "N"]
    normal_n_bad = [info["path"] for info in normal_n_images if not info.get("normal", {}).get("blue_is_zero_pad", False)]
    normal_hn_images = [info for info in core_images.values() if info.get("suffix") == "HN"]
    alpha_images = [info["path"] for info in core_images.values() if info.get("alpha", {}).get("present")]
    all_source_paths_lower = [info["path"].lower() for info in core_images.values()]
    source_forbidden_hits = [
        path for path in all_source_paths_lower if any(token in path for token in FORBIDDEN_DONOR_TOKENS)
    ]
    bound_core_files = {
        filename for family in MATERIAL_FAMILIES for _slot, filename, _space, _alpha in family["channels"]
    }
    unbound_core_files = sorted(set(core_images) - bound_core_files)
    unexpected_unbound_files = sorted(set(unbound_core_files) - set(UNBOUND_CORE_ALLOWLIST))

    lock_contract_value = config.get("display_target", {}).get("contract_output")
    lock_contract_path = resolve_path(workspace, lock_contract_value) if lock_contract_value else None
    locked_hashes: dict[str, str] = {}
    if lock_contract_path and lock_contract_path.is_file():
        lock_contract = json.loads(lock_contract_path.read_text(encoding="utf-8"))
        locked_hashes = {
            str(item["path"]): str(item["sha256"])
            for item in lock_contract.get("source_lock", {}).get("source_files", [])
            if str(item.get("path", "")).lower().endswith(".png")
        }
    lock_mismatches = []
    lock_compared = []
    lock_missing = []
    for filename, info in core_images.items():
        locked = locked_hashes.get(filename)
        info["source_lock_sha256"] = locked
        info["source_lock_match"] = locked == info["sha256"] if locked else None
        if locked:
            lock_compared.append(filename)
        else:
            lock_missing.append(filename)
        if locked and locked != info["sha256"]:
            lock_mismatches.append(filename)

    hard_errors: list[str] = []
    if config.get("primary_format") != "fbx":
        hard_errors.append("source.config primary_format is not fbx")
    if float(primary.get("global_scale", 0.0)) != 100.0:
        hard_errors.append("source.config global_scale is not 100")
    if str(primary.get("pose_position", "")).upper() != "REST":
        hard_errors.append("source.config pose_position is not REST")
    if missing_images:
        hard_errors.append(f"required source textures missing: {missing_images}")
    if missing_materials:
        hard_errors.append(f"required FBX materials missing from component reports: {missing_materials}")
    if unexpected_materials:
        hard_errors.append(f"unexpected FBX materials not covered by contract: {unexpected_materials}")
    if duplicate_coverage:
        hard_errors.append(f"source material has multiple family assignments: {duplicate_coverage}")
    if normal_n_bad:
        hard_errors.append(f"N texture B channel is not zero: {normal_n_bad}")
    if source_forbidden_hits:
        hard_errors.append(f"forbidden donor token appears in source texture name: {source_forbidden_hits}")
    if unexpected_unbound_files:
        hard_errors.append(f"core source textures lack a material binding or explicit exclusion: {unexpected_unbound_files}")
    if template_mismatches:
        hard_errors.append(f"semantic shader template mismatch: {template_mismatches}")
    if lock_mismatches:
        hard_errors.append(f"source texture hashes differ from the locked source contract: {lock_mismatches}")
    if lock_contract_path and lock_contract_path.is_file() and lock_missing:
        hard_errors.append(f"core source textures are missing from the locked source contract: {lock_missing}")

    gaps = [
        {
            "id": "swatchbin_encoder",
            "severity": "blocking_for_runtime",
            "status": "missing",
            "detail": "No verified local writer converts the source PNG channels to FH6 burG swatchbin payloads with mips/encoding and raw Alpha preserved.",
            "required_for": "all channel bindings",
        },
        {
            "id": "matti_parameter_name_hash_contract",
            "severity": "blocking_for_runtime",
            "status": "partial",
            "detail": "Donor MTPR parameter hashes/types are parsed, but the final shader-specific name/hash map for characterskin, charactereye, charactereyeball, eyesao, hair alpha, and alpha cloth is not yet written.",
            "required_for": "fresh MatI generation",
        },
        {
            "id": "alpha_cloth_template",
            "severity": "blocking_for_runtime",
            "status": "missing",
            "detail": "Current physical donors expose uber_clothing/glass/hair-alpha but no verified FH6 alpha-cloth blend template; cloth_02 must not be silently forced opaque.",
            "required_for": "cloth_alpha",
        },
        {
            "id": "fresh_matti_writer",
            "severity": "blocking_for_runtime",
            "status": "missing",
            "detail": "First-writer policy still retains donor MatI; a later writer must clone only shader structure and replace every texture/variant/detail/logo binding with generated Si swatches.",
            "required_for": "modelbin material patch",
        },
        {
            "id": "runtime_material_roundtrip",
            "severity": "blocking_for_runtime",
            "status": "missing",
            "detail": "No final material payload scan, swatch round-trip, or offline Display loading evidence exists yet.",
            "required_for": "game gate",
        },
    ]

    contract: dict[str, Any] = {
        "schema_version": CONTRACT_VERSION,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "FBX-first Si Display material, texture-channel, Alpha, shader and donor-leakage contract",
        "scope": {"screen": "Display", "driver_assets_included": False, "pmx_role": "reference_only"},
        "authority": {
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size, "sha256": sha256_path(config_path)},
            "fbx": {"path": str(fbx_path.resolve()), "bytes": fbx_path.stat().st_size, "sha256": sha256_path(fbx_path)},
            "source_lock_contract": {
                "path": str(lock_contract_path.resolve()) if lock_contract_path and lock_contract_path.is_file() else None,
                "sha256": sha256_path(lock_contract_path) if lock_contract_path and lock_contract_path.is_file() else None,
                "texture_hashes_in_catalog": len(locked_hashes),
                "core_texture_hashes_compared": len(lock_compared),
                "core_textures_missing_from_lock": lock_missing,
                "texture_hash_mismatches": lock_mismatches,
            },
            "global_scale": float(primary.get("global_scale", 0.0)),
            "pose": str(primary.get("pose_position", "")).upper(),
            "geometry_bind_weights_lod_authority": "native FBX only",
            "pmx": "Morph, legacy material labels, and historical comparison only",
        },
        "source_materials": {
            "from_component_reports": source_meshes,
            "expected_to_family": {material: families[0] for material, families in sorted(source_material_coverage.items())},
            "lod_aliases": LOD_MATERIAL_ALIASES,
            "synthetic_materials": ["sclera"],
        },
        "suffix_semantics": SUFFIX_SEMANTICS,
        "texture_inventory": {
            "core": core_images,
            "unbound_reference_only": {
                filename: {"reason": UNBOUND_CORE_ALLOWLIST[filename], "source_sha256": core_images[filename]["sha256"]}
                for filename in unbound_core_files
                if filename in UNBOUND_CORE_ALLOWLIST
            },
            "excluded_vfx_and_fx": {name: info for name, info in images.items() if name.startswith("T_fx_")},
        },
        "material_families": family_entries,
        "container_slot_plan": {
            "Helmet_Race_Modern": {
                "slot_count": 6,
                "slot_roles": ["skin_face", "iris", "sclera", "hair_brow", "hair_shadow", "eye_shadow"],
                "source_materials_merged": {"M_actor_jsspsi_hair_01": "hair_brow", "M_actor_jsspsi_brow_01": "hair_brow"},
                "constraint": "Keep Head/Hair draws separate even when Brow shares the alpha-hair MatI family; remap Brow UV/tint before runtime.",
            },
            "Outfit_Race_Suit_Modern_F": {
                "slot_count": 8,
                "slot_roles": ["skin_body", "cloth_opaque", "cloth_alpha"],
                "source_materials_merged": {"M_actor_jsspsi_cloth_01": "cloth_opaque", "M_actor_jsspsi_cloth_03": "cloth_opaque"},
                "constraint": "Retain body/garment draw boundaries and component-local Skel; do not reuse race-suit texture responses.",
            },
        },
        "donor_evidence": {
            "physical_containers": donor_inventory,
            "semantic_shader_templates": semantic_inventory,
            "forbidden_donor_texture_references_seen_in_templates": donor_refs,
            "final_policy": {
                "allowed_final_texture_origin": "generated Si swatchbin only",
                "forbidden_path_tokens": list(FORBIDDEN_DONOR_TOKENS),
                "forbidden_response_types": ["Alice/Kim/driver skin normal", "race-suit normal/mask/spec", "helmet normal", "logo", "stitch", "detail normal", "donor roughness/metal response"],
                "template_rule": "Donor MatI may provide shader class/parameter shape only; every texture and variant response must be explicitly replaced or disabled.",
            },
        },
        "swatch_plan": {
            "status": "contract_only_not_written",
            "source_channels_ready_for_encoder": True,
            "fh6_swatchbin_bytes_ready": False,
            "fresh_matti_bytes_ready": False,
            "format": "FH6 burG swatchbin",
            "one_output_per": "material family/channel binding",
            "requirements": [
                "preserve source PNG bytes and exact Alpha semantics before compression",
                "choose FH6 encoding/mips per shader template, never infer from Blender Principled nodes",
                "keep N maps in two-channel XY/B=0 convention and HN maps in their own packed RGBA convention",
                "record source SHA-256, output SHA-256, width, height, mips and encoding",
            ],
            "neutral_response_baseline": NEUTRAL_RESPONSE_BASELINE,
            "channel_restore_order": CHANNEL_RESTORE_ORDER,
        },
        "validation": {
            "hard_error_count": len(hard_errors),
            "hard_errors": hard_errors,
            "checks": {
                "fbx_authority": not hard_errors or "source.config primary_format is not fbx" not in hard_errors,
                "required_texture_files_present": not missing_images,
                "source_alpha_audited": all("alpha" in info for info in core_images.values()),
                "n_maps_b_zero_verified": not normal_n_bad,
                "hn_maps_not_misclassified_as_n": all(info.get("normal", {}).get("must_not_use_xy_b_zero_decode") for info in normal_hn_images),
                "source_material_coverage_exact": not (missing_materials or unexpected_materials or duplicate_coverage),
                "source_texture_names_clean": not source_forbidden_hits,
                "source_lock_hashes_match": bool(lock_compared) and not (lock_mismatches or lock_missing),
                "core_texture_binding_complete": not unexpected_unbound_files,
                "shader_templates_match_required_classes": not template_mismatches,
                "donor_texture_leakage_policy_declared": True,
                "final_donor_texture_leakage_scan": False,
                "swatchbin_written": False,
                "matti_written": False,
                "runtime_roundtrip": False,
                "offline_display_game_gate": False,
            },
            "gaps": gaps,
            "status": "contract_valid_conversion_pending" if not hard_errors else "contract_invalid",
        },
        "license_guard": "Local technical validation only; do not redistribute source-derived assets.",
    }

    selfcheck = {
        "schema_version": 1,
        "created_utc": contract["created_utc"],
        "contract_schema_version": CONTRACT_VERSION,
        "status": contract["validation"]["status"],
        "hard_error_count": len(hard_errors),
        "hard_errors": hard_errors,
        "checks": contract["validation"]["checks"],
        "source_counts": {
            "core_textures": len(core_images),
            "excluded_fx_textures": len(images) - len(core_images),
            "source_materials": len(actual_materials),
            "canonical_source_materials": len(actual_canonical_materials),
            "material_families": len(MATERIAL_FAMILIES),
            "alpha_audited_textures": len(core_images),
            "alpha_present_textures": len(alpha_images),
            "n_maps": len(normal_n_images),
            "hn_maps": len(normal_hn_images),
            "forbidden_donor_template_refs": len(donor_refs),
            "source_lock_texture_hashes_in_catalog": len(locked_hashes),
            "source_lock_core_texture_hashes_compared": len(lock_compared),
            "unbound_reference_only_textures": len(unbound_core_files),
        },
        "conversion_gaps": gaps,
    }
    return contract, selfcheck


def main() -> int:
    args = arguments()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty material contract directory: {output_dir}")
    contract, selfcheck = build_contract(args.config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "si-display-material-contract-v001.json"
    selfcheck_path = output_dir / "si-display-material-selfcheck-v001.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    selfcheck_path.write_text(json.dumps(selfcheck, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if selfcheck["hard_error_count"]:
        print(json.dumps({"contract": str(contract_path), "selfcheck": str(selfcheck_path), "status": selfcheck["status"]}))
        return 2
    print(
        "SI_FBX_MATERIAL_CONTRACT="
        + json.dumps(
            {"contract": str(contract_path), "selfcheck": str(selfcheck_path), "status": selfcheck["status"]},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
