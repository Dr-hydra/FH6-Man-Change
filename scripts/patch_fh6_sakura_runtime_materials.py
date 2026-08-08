#!/usr/bin/env python3
"""Patch the Sakura race-suit runtime material overrides without XML churn."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


OPTION_TEXTURES = {
    "Outfit_Race_Suit_Modern_Mat_Suit_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_diff_a346c884-e0a3-4ffd-af3f-5e32ac65d8f2.swatchbin",
        "Cloth1",
    ),
    "Outfit_Race_Suit_Modern_Mat_Black_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_gloves_diff_ec74c04d-10a7-40d4-be29-16ddc8c0e638.swatchbin",
        "肌",
    ),
    "Outfit_Race_Suit_Modern_Mat_InnerArm_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_diffuse_blue_25f544a6-e713-4a3d-9e65-a4447f541daf.swatchbin",
        "Cloth2",
    ),
    "Outfit_Race_Suit_Modern_Mat_Cuffs_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_diff_a346c884-e0a3-4ffd-af3f-5e32ac65d8f2.swatchbin",
        "Cloth1",
    ),
    "Outfit_Race_Suit_Modern_Mat_Shoulders_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_diff_a346c884-e0a3-4ffd-af3f-5e32ac65d8f2.swatchbin",
        "Cloth1",
    ),
    "Outfit_Race_Suit_Modern_Mat_Gloves_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_gloves_diff_ec74c04d-10a7-40d4-be29-16ddc8c0e638.swatchbin",
        "肌",
    ),
    "Outfit_Race_Suit_Modern_Mat_GlovePalm_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_gloves_diffuse_hwhite_ccaeb9fb-15dd-41f0-8320-7ad7a78521e8.swatchbin",
        "Cloth1Alpha",
    ),
    "Outfit_Race_Suit_Modern_Mat_Shoes_VIP_Sakura_Racesuit": (
        r"_library\texturespg\characters\swatches\outfit_race_suit_modern_shoes_diff_35038698-cd09-4a0d-b340-ea966105ae09.swatchbin",
        "Cloth1",
    ),
}

HAIR_DIFFUSE_TEXTURE = (
    r"_library\texturespg\characters\swatches\si_jsspsi_hair_diff_"
    r"8d7e3f10-5a1b-5c6d-9e7f-1029384756ab.swatchbin"
)
HAIR_SHADOW_TEXTURE = (
    r"_library\texturespg\characters\swatches\si_jsspsi_hairshadow_"
    r"4c2d1e0f-9a8b-7c6d-5e4f-3210fedcba98.swatchbin"
)

# The Helmet donor's MatI order is helmet, vent, bits, padding, rubber, visor.
# The replacement mesh keeps that material domain, so Sakura runtime options must
# point each source PMX material at a texture that matches its original role.
HELMET_OPTION_PARAMETERS = {
    "Helmet_Race_Modern_Mat_helmet_VIP_Sakura_Helmet": {
        "source_material": "发",
        "parameters": {
            "DiffuseTexture": HAIR_DIFFUSE_TEXTURE,
            "MainTintColorParam": "1,1,1,1",
            "VariantConstant_DetailNorm": "0",
            "VariantConstant_Pattern": "0",
            "VariantConstant_Logo": "0",
            "VariantConstant_Stitch": "0",
            "VariantConstant_MaterialDiff": "1",
            "VariantConstant_MaterialTint": "1",
        },
    },
    "Helmet_Race_Modern_Mat_vent_VIP_Sakura_Helmet": {
        "source_material": "Cloth1",
        "parameters": {
            "DiffuseTexture": OPTION_TEXTURES[
                "Outfit_Race_Suit_Modern_Mat_Suit_VIP_Sakura_Racesuit"
            ][0],
            "MainTintColorParam": "1,1,1,1",
            "VariantConstant_DetailNorm": "0",
            "VariantConstant_Pattern": "0",
            "VariantConstant_Logo": "0",
            "VariantConstant_Stitch": "0",
            "VariantConstant_MaterialDiff": "1",
            "VariantConstant_MaterialTint": "1",
        },
    },
    "Helmet_Race_Modern_Mat_bits_VIP_Sakura_Helmet": {
        "source_material": "Cloth1",
        "parameters": {
            "DiffuseATexture": OPTION_TEXTURES[
                "Outfit_Race_Suit_Modern_Mat_Suit_VIP_Sakura_Racesuit"
            ][0],
        },
    },
    "Helmet_Race_Modern_Mat_padding_VIP_Sakura_Helmet": {
        "source_material": "Cloth1",
        "parameters": {
            "DiffuseTexture": OPTION_TEXTURES[
                "Outfit_Race_Suit_Modern_Mat_Suit_VIP_Sakura_Racesuit"
            ][0],
            "MainTintColorParam": "1,1,1,1",
            "VariantConstant_DetailNorm": "0",
            "VariantConstant_Pattern": "0",
            "VariantConstant_Logo": "0",
            "VariantConstant_Stitch": "0",
            "VariantConstant_MaterialDiff": "1",
            "VariantConstant_MaterialTint": "1",
        },
    },
    "Helmet_Race_Modern_Mat_rubber_VIP_Sakura_Helmet": {
        "source_material": "Cloth2",
        "parameters": {"DiffuseColorParam": "1,1,1,1"},
    },
    "Helmet_Race_Modern_Mat_visor_VIP_Sakura_Helmet": {
        "source_material": "发影",
        "parameters": {
            "TextureTexture": HAIR_SHADOW_TEXTURE,
            "CloudyColorColorParam": "1,1,1,1",
        },
    },
}

COMMON_PARAMETERS = {
    "MainTintColorParam": "1,1,1,1",
    "VariantConstant_DetailNorm": "0",
    "VariantConstant_Pattern": "0",
    "VariantConstant_Logo": "0",
    "VariantConstant_Stitch": "0",
    "VariantConstant_MaterialDiff": "1",
    "VariantConstant_MaterialTint": "1",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_parameter(block: str, name: str, value: str) -> tuple[str, str]:
    pattern = re.compile(
        rf'(<Parameter\s+name="{re.escape(name)}"\s+value=")[^"]*("\s*/>)'
    )
    matches = list(pattern.finditer(block))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name!r} parameter, found {len(matches)}")
    old_value = matches[0].group(0).split('value="', 1)[1].rsplit('"', 1)[0]
    return pattern.sub(lambda match: match.group(1) + value + match.group(2), block), old_value


def main() -> None:
    args = arguments()
    source_path = args.input.resolve(strict=True)
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source = source_path.read_bytes()
    if source.startswith(b"\xef\xbb\xbf"):
        raise ValueError("Unexpected UTF-8 BOM")
    if b"\r\n" not in source:
        raise ValueError("Expected CRLF line endings")
    text = source.decode("utf-8")
    ET.fromstring(text)

    changes = []
    output_text = text
    for option_name, (texture_path, source_material) in OPTION_TEXTURES.items():
        option_pattern = re.compile(
            rf'(<Option\s+name="{re.escape(option_name)}">)(.*?)(\s*</Option>)',
            re.DOTALL,
        )
        matches = list(option_pattern.finditer(output_text))
        if len(matches) != 1:
            raise ValueError(f"Expected one option {option_name!r}, found {len(matches)}")
        match = matches[0]
        block = match.group(0)
        option_changes = []
        values = {"DiffuseTexture": texture_path, **COMMON_PARAMETERS}
        for parameter_name, new_value in values.items():
            block, old_value = replace_parameter(block, parameter_name, new_value)
            option_changes.append(
                {"parameter": parameter_name, "old": old_value, "new": new_value}
            )
        output_text = output_text[: match.start()] + block + output_text[match.end() :]
        changes.append(
            {
                "option": option_name,
                "source_material": source_material,
                "parameters": option_changes,
            }
        )

    for option_name, spec in HELMET_OPTION_PARAMETERS.items():
        option_pattern = re.compile(
            rf'(<Option\s+name="{re.escape(option_name)}">)(.*?)(\s*</Option>)',
            re.DOTALL,
        )
        matches = list(option_pattern.finditer(output_text))
        if len(matches) != 1:
            raise ValueError(f"Expected one option {option_name!r}, found {len(matches)}")
        match = matches[0]
        block = match.group(0)
        option_changes = []
        for parameter_name, new_value in spec["parameters"].items():
            block, old_value = replace_parameter(block, parameter_name, new_value)
            option_changes.append(
                {"parameter": parameter_name, "old": old_value, "new": new_value}
            )
        output_text = output_text[: match.start()] + block + output_text[match.end() :]
        changes.append(
            {
                "option": option_name,
                "source_material": spec["source_material"],
                "parameters": option_changes,
            }
        )

    output_data = output_text.encode("utf-8")
    ET.fromstring(output_data)
    if output_data.count(b"\r\n") != source.count(b"\r\n"):
        raise ValueError("Patch changed the CRLF line count")
    output_path.write_bytes(output_data)

    root = ET.fromstring(output_data)
    options = {option.get("name"): option for option in root.iter("Option")}
    for option_name, (texture_path, _source_material) in OPTION_TEXTURES.items():
        parameters = {
            parameter.get("name"): parameter.get("value")
            for parameter in options[option_name].findall("Parameter")
        }
        expected = {"DiffuseTexture": texture_path, **COMMON_PARAMETERS}
        if any(parameters.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Post-write verification failed for {option_name}")
    for option_name, spec in HELMET_OPTION_PARAMETERS.items():
        parameters = {
            parameter.get("name"): parameter.get("value")
            for parameter in options[option_name].findall("Parameter")
        }
        if any(parameters.get(name) != value for name, value in spec["parameters"].items()):
            raise ValueError(f"Post-write verification failed for {option_name}")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "input": {
            "path": str(source_path),
            "bytes": len(source),
            "sha256": sha256_bytes(source),
        },
        "output": {
            "path": str(output_path),
            "bytes": len(output_data),
            "sha256": sha256_bytes(output_data),
        },
        "options_changed": len(changes),
        "changes": changes,
        "validation": {
            "xml_parse": True,
            "crlf_count_preserved": True,
            "unmodified_bytes": sum(left == right for left, right in zip(source, output_data)),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FH6_SAKURA_RUNTIME_MATERIALS="
        + json.dumps(report["output"], ensure_ascii=False, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
