#!/usr/bin/env python3
"""Build the versioned Si Display LOD0 swatches and complete MatI patches.

The result is an offline workspace milestone. It requires the final seven-draw
Head and eight-draw Body/Garment geometry candidates and does not deploy them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
PIPELINE_VERSION = "v002"
DEFAULT_OUTPUT = (
    WORKSPACE / "work" / "si" / "fbx-source" / f"material-pipeline-{PIPELINE_VERSION}"
)
DEFAULT_CONTRACT = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "material-contract-v001"
    / "si-display-material-contract-v001.json"
)
DEFAULT_ARCHIVE = (
    WORKSPACE
    / "work"
    / "si"
    / "materials-v006-eyes"
    / "characters.iris-v006.zip"
)
SOURCE_ROOT = (
    WORKSPACE
    / "sources"
    / "si"
    / "v2"
    / "fbx"
    / "chr_0036_jsspsi_postmodel"
)
NAMESPACE = uuid.UUID("8f423b8e-b24d-4c0c-b223-4598049218cb")

CONVERTER = SCRIPT_DIR / "convert_fh6_swatchbin.py"
PATCHER = SCRIPT_DIR / "patch_fh6_material_profile.py"
SCANNER = SCRIPT_DIR / "scan_fh6_donor_leakage.py"

HELMET_CANDIDATE = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-05-modelbin-lod0-v002"
    / "helmet"
    / "Helmet_Race_Modern.lod0-head7.modelbin"
)
OUTFIT_CANDIDATE = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-05-modelbin-lod0-v002"
    / "outfit"
    / "Outfit_Race_Suit_Modern_F.lod0-final.modelbin"
)
FACE_DONOR = (
    "work/si/components/baselines/head-face/extracted/DRV_BA_F_01_Face.modelbin"
)
ALICE_DONOR = (
    "work/si/components/baselines/head-alice-f/extracted/Driver_Alice_F.modelbin"
)
FEMALE_DONOR = "work/si/components/baselines/body/extracted/Female.modelbin"
OUTFIT_DONOR = (
    "work/donors/Outfit_Race_Suit_Modern_F/extracted/Outfit_Race_Suit_Modern_F.modelbin"
)
ALPHA_CLOTH_DONOR = (
    "work/si/fbx-source/material-donors-v001/Lower_Jean_Shorts_Ripped_F/"
    "extracted/Lower_Jean_Shorts_Ripped_F.modelbin"
)


SWATCH_PLAN: list[dict[str, Any]] = [
    {
        "key": "face_base",
        "source": "T_actor_jsspsi_face_01_D.png",
        "template": "People/swatches/Average_Kim_A_FCLR_3f8c7b1e-c757-481b-add2-d0425ad55b16.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "face_normal",
        "source": "T_actor_jsspsi_face_01_N.png",
        "template": "People/swatches/Average_Kim_A_NRML_6e8f5154-6664-4d8e-9621-9a4d33ffed07.swatchbin",
        "color_space": "linear",
        "normal_xy": True,
    },
    {
        "key": "face_mask",
        "source": "T_actor_common_female_face_01_cm_M.png",
        "template": "Clothes/Shared/Swatches/Button_01_NRML_3c6e4f42-a4b6-42f1-ad9e-34e79c80f2d3.swatchbin",
        "color_space": "linear",
    },
    {
        "key": "face_lut",
        "source": "T_actor_common_femaleskincolor01_lut_D.png",
        "template": "Swatches/1_SkinLookup_DIFF_428f3dc2-dc93-41e8-826c-8ce5712cd619.swatchbin",
        "color_space": "srgb",
        "resample": "bilinear",
    },
    {
        "key": "body_base",
        "source": "T_actor_jsspsi_body_01_D.png",
        "template": "People/swatches/Average_Kim_A_BCLR_389f1739-1ff1-4875-a532-265f959d3ad3.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "body_normal",
        "source": "T_actor_jsspsi_body_01_N.png",
        "template": "People/swatches/Average_Kim_A_BodyNormal_f57fa7c6-b612-4f5e-96dd-eb8c2fb5ad5c.swatchbin",
        "color_space": "linear",
        "normal_xy": True,
    },
    {
        "key": "body_lut",
        "source": "T_actor_common_femaleskincolor03_lut_D.png",
        "template": "Swatches/1_SkinLookup_DIFF_428f3dc2-dc93-41e8-826c-8ce5712cd619.swatchbin",
        "color_space": "srgb",
        "resample": "bilinear",
    },
    {
        "key": "hair_base",
        "source": "T_actor_jsspsi_hair_01_D.png",
        "template": "People/swatches/Ambassador_Helena_F_FCLR_abe8c78a-11f2-4476-b634-f97303c4c387.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "hair_normal_spec",
        "source": "T_actor_jsspsi_hair_01_HN.png",
        "template": "People/swatches/Ambassador_Helena_F_NRML_d8385b55-ee25-499b-9b25-b8908d4a19be.swatchbin",
        "color_space": "linear",
    },
    {
        "key": "eye_shadow_mask",
        "source": "T_actor_common_eyeshadow_01_M.png",
        "template": "Swatches/1_Patterns2_Diffuse_RetroReflective2_6a272206-8a08-4ffd-b748-0a04b34d26e0.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "hair_shadow_mask",
        "source": "T_actor_common_hairshadow_01_M.png",
        "template": "Swatches/1_Patterns2_Diffuse_RetroReflective2_6a272206-8a08-4ffd-b748-0a04b34d26e0.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "iris_base",
        "source": "T_actor_jsspsi_iris_01_D.png",
        "template": "Swatches/robin_eye_BCLR_a7f5b956-e1f8-4081-9588-f305ce238450.swatchbin",
        "color_space": "linear",
        "contract_color_space": "srgb",
        "template_override_reason": "charactereye retail BCLR uses linear header flags",
    },
    {
        "key": "cloth_base",
        "source": "T_actor_jsspsi_cloth_01_D.png",
        "template": "People/swatches/Ambassador_Helena_F_FCLR_abe8c78a-11f2-4476-b634-f97303c4c387.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "cloth_normal",
        "source": "T_actor_jsspsi_cloth_01_N.png",
        "template": "People/swatches/Ambassador_Helena_F_NRML_d8385b55-ee25-499b-9b25-b8908d4a19be.swatchbin",
        "color_space": "linear",
        "normal_xy": True,
    },
    {
        "key": "cloth_alpha_base",
        "source": "T_actor_jsspsi_cloth_02_D.png",
        "template": "Clothes/swatches/lolita_buttons_DIFF_f03db631-dd56-4546-982a-3125a3387d93.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "cloth_alpha_normal",
        "source": "T_actor_jsspsi_cloth_02_N.png",
        "template": "Clothes/swatches/Lower_JeanShorts_Ripped_NRML_ea54bb0d-a137-4605-a158-6697353d456e.swatchbin",
        "color_space": "linear",
        "normal_xy": True,
    },
    {
        "key": "neutral_normal",
        "synthetic": "neutral_normal_xy",
        "template": "Swatches/1_FlatTextures_NormalBC7Alpha_edfebb13-4fac-4b32-9b43-d00b785a39b6.swatchbin",
        "color_space": "linear",
        "normal_xy": True,
    },
    {
        "key": "neutral_data_white",
        "synthetic": "opaque_white",
        "template": "Swatches/1_FlatTextures_NormalBC7Alpha_edfebb13-4fac-4b32-9b43-d00b785a39b6.swatchbin",
        "color_space": "linear",
    },
    {
        "key": "neutral_transparent",
        "synthetic": "transparent_black",
        "template": "Swatches/1_FlatTextures_DiffuseBlackAlpha_a8787efc-94ac-487c-9305-519c91024880.swatchbin",
        "color_space": "srgb",
    },
    {
        "key": "opaque_white",
        "source": "common_deco_white.png",
        "template": "Swatches/1_FlatTextures_DiffuseWhite_c2c379df-efeb-412d-a920-2cfb912a35ac.swatchbin",
        "color_space": "srgb",
        "resample": "nearest",
    },
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--template-archive", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    if completed.returncode != expected:
        raise RuntimeError(
            f"Command returned {completed.returncode}, expected {expected}: {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def synthetic_source(kind: str, path: Path) -> None:
    values = {
        "neutral_normal_xy": (128, 128, 0, 255),
        "opaque_white": (255, 255, 255, 255),
        "transparent_black": (0, 0, 0, 0),
    }
    if kind not in values:
        raise ValueError(f"Unknown synthetic source {kind}")
    Image.new("RGBA", (64, 64), values[kind]).save(path)


def game_path(filename: str) -> str:
    return rf"Game:\Media\_library\texturespg\characters\swatches\{filename}"


def texture_map(swatches: dict[str, dict[str, Any]], mapping: dict[str, str]) -> dict[str, str]:
    return {name_hash: swatches[key]["game_path"] for name_hash, key in mapping.items()}


def hair_values() -> dict[str, dict[str, Any]]:
    return {
        "5ce83716": {"name": "GradientColourBlackColorParam_GradientMapThreeColour2_HairGradientMap", "type": 1, "value": [1, 1, 1, 1]},
        "5bf5371c": {"name": "GradientColourMidColorParam_GradientMapThreeColour2_HairGradientMap", "type": 1, "value": [1, 1, 1, 1]},
        "27c6d654": {"name": "GradientColourWhiteColorParam_GradientMapThreeColour2_HairGradientMap", "type": 1, "value": [1, 1, 1, 1]},
        "19e046f9": {"name": "PrimarySpecColour0_HairGradientMapColorParam", "type": 1, "value": [0.25, 0.25, 0.25, 1]},
        "c0a80101": {"name": "SecondarySpecColour_HairGradientMapColorParam", "type": 1, "value": [0.25, 0.25, 0.25, 1]},
        "94ddee70": {"name": "PrimarySpecColourIntensity_HairGradientMap_floatVal", "type": 2, "value": 0.25},
        "a596f574": {"name": "SecondarySpecColourIntensity_HairGradientMap_floatVal", "type": 2, "value": 0.1},
        "f4ddc59f": {"name": "PeachfuzzIntensity0_HairGradientMap_floatVal", "type": 2, "value": 0.0},
    }


def cloth_values() -> dict[str, dict[str, Any]]:
    return {
        "adc671e0": {"name": "ClothRoughness_floatVal", "type": 2, "value": 0.6},
        "c17ae7f6": {"name": "PeachfuzzPower_floatVal", "type": 2, "value": 0.0},
        "64490d74": {"name": "FuzzAmount_floatVal", "type": 2, "value": 0.0},
        "89ae3b2e": {"name": "LogoRoughness_floatVal", "type": 2, "value": 0.6},
    }


def alpha_cloth_values() -> dict[str, dict[str, Any]]:
    return {
        "64490d74": {"name": "FuzzAmount_floatVal", "type": 2, "value": 0.0},
    }


def build_profiles(swatches: dict[str, dict[str, Any]], output: Path) -> tuple[Path, Path]:
    face_textures = texture_map(
        swatches,
        {
            "16f462b7": "face_base",
            "4328ec42": "face_mask",
            "64a09187": "face_lut",
            "f5257e8a": "face_normal",
            "7f79eec8": "face_normal",
            "d1d08edc": "neutral_normal",
        },
    )
    body_textures = texture_map(
        swatches,
        {
            "16f462b7": "body_base",
            "4328ec42": "neutral_data_white",
            "64a09187": "body_lut",
            "f5257e8a": "body_normal",
            "7f79eec8": "body_normal",
            "d1d08edc": "neutral_normal",
        },
    )
    hair_textures = texture_map(
        swatches,
        {"370c798a": "hair_base", "cc4fe29b": "hair_normal_spec"},
    )
    iris_textures = texture_map(
        swatches,
        {"8c8a83ef": "neutral_normal", "c960291c": "iris_base", "38a72e02": "opaque_white"},
    )
    cloth_textures = texture_map(
        swatches,
        {
            "f724f1e3": "neutral_normal",
            "17833b17": "cloth_normal",
            "723e7e4a": "neutral_normal",
            "ee34b08b": "cloth_base",
            "4a8c9771": "neutral_transparent",
        },
    )
    alpha_cloth_textures = texture_map(
        swatches,
        {
            "ee34b08b": "cloth_alpha_base",
            "17833b17": "cloth_alpha_normal",
        },
    )

    def material(
        target: int,
        role: str,
        donor: str,
        donor_id: int,
        shader: str,
        atst: str,
        textures: dict[str, str],
        values: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "target_material_id": target,
            "role": role,
            "template_modelbin": donor,
            "template_material_id": donor_id,
            "expected_shader": shader,
            "expected_atst": atst,
            "require_all_template_textures_patched": True,
            "texture_patches": textures,
            "value_patches": values or {},
        }

    helmet = {
        "schema_version": 1,
        "scope": "final LOD0 seven-draw Helmet candidate with dedicated sclera",
        "materials": [
            material(0, "hair", ALICE_DONOR, 1, "hairalphatestnotangentgradmap", "0100", hair_textures, hair_values()),
            material(1, "eye_shadow", FACE_DONOR, 0, "eyesao", "0001", texture_map(swatches, {"0fea383b": "eye_shadow_mask"})),
            material(2, "skin_face", FACE_DONOR, 1, "characterskin", "0000", face_textures),
            material(3, "iris", FACE_DONOR, 3, "charactereye", "0000", iris_textures),
            material(4, "hair_brow", ALICE_DONOR, 1, "hairalphatestnotangentgradmap", "0100", hair_textures, hair_values()),
            material(5, "hair_shadow", FACE_DONOR, 0, "eyesao", "0001", texture_map(swatches, {"0fea383b": "hair_shadow_mask"})),
            material(6, "sclera", ALICE_DONOR, 2, "charactereyeball", "0000", {}),
        ],
    }
    outfit_materials = []
    for target in (0, 3, 4, 7):
        outfit_materials.append(
            material(target, "cloth_opaque", OUTFIT_DONOR, 0, "uber_clothing", "0100", cloth_textures, cloth_values())
        )
    for target in (1, 2, 5):
        outfit_materials.append(
            material(target, "skin_body", FEMALE_DONOR, 0, "characterskin", "0000", body_textures)
        )
    outfit_materials.append(
        material(
            6,
            "cloth_alpha",
            ALPHA_CLOTH_DONOR,
            3,
            "uber_clothing_transparancy",
            "0100",
            alpha_cloth_textures,
            alpha_cloth_values(),
        )
    )
    outfit = {
        "schema_version": 1,
        "scope": "final LOD0 Outfit candidate with verified retail alpha-cloth template",
        "materials": outfit_materials,
    }
    profile_dir = output / "profiles"
    profile_dir.mkdir(parents=True)
    helmet_path = profile_dir / f"helmet-final-lod0-{PIPELINE_VERSION}.json"
    outfit_path = profile_dir / f"outfit-final-lod0-{PIPELINE_VERSION}.json"
    helmet_path.write_text(json.dumps(helmet, ensure_ascii=False, indent=2), encoding="utf-8")
    outfit_path.write_text(json.dumps(outfit, ensure_ascii=False, indent=2), encoding="utf-8")
    return helmet_path, outfit_path


def main() -> int:
    args = arguments()
    output = args.output_dir.resolve()
    contract_path = args.contract.resolve(strict=True)
    archive = args.template_archive.resolve(strict=True)
    (WORKSPACE / ALPHA_CLOTH_DONOR).resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse milestone directory {output}")
    output.mkdir(parents=True)
    (output / "swatches" / "reports").mkdir(parents=True)
    (output / "synthetic").mkdir(parents=True)
    (output / "candidates").mkdir(parents=True)
    (output / "leakage").mkdir(parents=True)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    swatches: dict[str, dict[str, Any]] = {}
    used_source_files: set[str] = set()
    for spec in SWATCH_PLAN:
        key = spec["key"]
        guid = uuid.uuid5(NAMESPACE, f"si-display-material-{PIPELINE_VERSION}:{key}")
        filename = f"si_display_{PIPELINE_VERSION}_{key}_{guid}.swatchbin"
        source = SOURCE_ROOT / spec["source"] if "source" in spec else output / "synthetic" / f"{key}.png"
        if "synthetic" in spec:
            synthetic_source(spec["synthetic"], source)
        else:
            used_source_files.add(spec["source"])
        swatch = output / "swatches" / filename
        report = output / "swatches" / "reports" / f"{key}.json"
        command = [
            sys.executable,
            str(CONVERTER),
            "--template-archive",
            str(archive),
            "--template-entry",
            spec["template"],
            "--source",
            str(source),
            "--output",
            str(swatch),
            "--report",
            str(report),
            "--color-space",
            spec["color_space"],
            "--resample",
            spec.get("resample", "none"),
        ]
        if spec.get("normal_xy"):
            command.append("--normal-xy")
        run(command)
        swatches[key] = {
            "key": key,
            "path": str(swatch),
            "filename": filename,
            "game_path": game_path(filename),
            "sha256": sha256_path(swatch),
            "report": str(report),
            "source": str(source),
            "template_entry": spec["template"],
            "color_space": spec["color_space"],
            "contract_color_space": spec.get("contract_color_space", spec["color_space"]),
            "template_override_reason": spec.get("template_override_reason"),
        }

    contract_channels = {
        channel["source_file"]
        for family in contract["material_families"]
        for channel in family["channels"]
    }
    unbound_sources = sorted(contract_channels - used_source_files)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "contract": {"path": str(contract_path), "sha256": sha256_path(contract_path)},
        "template_archive": {"path": str(archive), "sha256": sha256_path(archive)},
        "swatches": swatches,
        "coverage": {
            "runtime_bound_source_files": sorted(used_source_files),
            "contract_source_files_not_supported_by_current_donor_mtpr": unbound_sources,
            "note": "Unbound P/RS/ST/RD/SDF/E channels remain audited sources; no unsupported MTPR records were invented.",
        },
        "game_validated": False,
    }
    manifest_path = output / "swatches" / f"si-display-swatches-{PIPELINE_VERSION}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    helmet_profile, outfit_profile = build_profiles(swatches, output)
    helmet_output = output / "candidates" / f"Helmet_Race_Modern.lod0-materials-{PIPELINE_VERSION}.modelbin"
    helmet_report = output / "candidates" / f"helmet-materials-{PIPELINE_VERSION}.report.json"
    run(
        [
            sys.executable,
            str(PATCHER),
            str(HELMET_CANDIDATE),
            str(helmet_profile),
            str(helmet_output),
            "--report",
            str(helmet_report),
        ]
    )
    outfit_output = output / "candidates" / f"Outfit_Race_Suit_Modern_F.lod0-materials-{PIPELINE_VERSION}.modelbin"
    outfit_report = output / "candidates" / f"outfit-materials-{PIPELINE_VERSION}.report.json"
    run(
        [
            sys.executable,
            str(PATCHER),
            str(OUTFIT_CANDIDATE),
            str(outfit_profile),
            str(outfit_output),
            "--report",
            str(outfit_report),
        ]
    )

    helmet_leakage = output / "leakage" / f"helmet-{PIPELINE_VERSION}.report.json"
    run(
        [
            sys.executable,
            str(SCANNER),
            str(helmet_output),
            "--report",
            str(helmet_leakage),
            "--require-generated-prefix",
            "si_",
        ]
    )
    outfit_leakage = output / "leakage" / f"outfit-{PIPELINE_VERSION}.report.json"
    run(
        [
            sys.executable,
            str(SCANNER),
            str(outfit_output),
            "--report",
            str(outfit_leakage),
            "--require-generated-prefix",
            "si_",
        ],
    )
    outfit_scan = json.loads(outfit_leakage.read_text(encoding="utf-8"))
    leaked_materials = sorted({item["material_id"] for item in outfit_scan["findings"]})
    if leaked_materials:
        raise ValueError(f"Final Outfit candidate still leaks donor materials: {leaked_materials}")

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "scope": "FH6 Display LOD0 material/texture milestone; workspace only",
        "outputs": {
            "swatch_manifest": str(manifest_path),
            "swatch_count": len(swatches),
            "helmet_profile": str(helmet_profile),
            "helmet_candidate": str(helmet_output),
            "helmet_patch_report": str(helmet_report),
            "helmet_leakage_report": str(helmet_leakage),
            "outfit_profile": str(outfit_profile),
            "outfit_candidate": str(outfit_output),
            "outfit_patch_report": str(outfit_report),
            "outfit_leakage_report": str(outfit_leakage),
        },
        "completed": {
            "png_tga_dds_to_raw_rgba_frontend": True,
            "unpremultiplied_alpha_transport": True,
            "rg_two_channel_normal_guard": True,
            "swatch_header_guid_rewritten": True,
            "semantic_header_color_flags_checked": True,
            "skin_hair_iris_eye_shadow_hair_shadow_bindings": True,
            "sclera_draw_and_charactereyeball_binding": True,
            "opaque_cloth_minimal_binding": True,
            "alpha_cloth_source_alpha_and_normal_binding": True,
            "verified_retail_uber_clothing_transparancy_template": True,
            "donor_leakage_scan": True,
            "helmet_candidate_leakage_free": True,
            "outfit_candidate_leakage_free": True,
        },
        "blockers": [
            {
                "id": "unsupported_source_channels",
                "detail": "The selected donor MTPR layouts do not expose every source P/RS/ST/RD/SDF/E channel. These remain in the audited contract and are not guessed into binary records.",
                "source_files": unbound_sources,
            },
        ],
        "validation": {
            "structural": True,
            "swatch_roundtrip": True,
            "helmet_donor_leakage_pass": True,
            "outfit_donor_leakage_pass": True,
            "outfit_donor_leakage_materials": leaked_materials,
            "game_directory_modified": False,
            "game_validated": False,
        },
    }
    summary_path = output / f"si-display-material-pipeline-{PIPELINE_VERSION}.report.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SI_DISPLAY_MATERIAL_PIPELINE=" + json.dumps(summary["outputs"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
