from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
sys.path.insert(0, str(SCRIPTS))

from convert_fh6_swatchbin import parse_swatchbin  # noqa: E402
from modelbin_bundle import (  # noqa: E402
    blob_spec,
    first_difference,
    parse_bundle,
    rebuild_with_blob_sequence,
)
from patch_fh6_material_profile import (  # noqa: E402
    materials_by_id,
    parameter_name_hash,
    patch_mtpr,
    shader_info,
)
from scan_fh6_donor_leakage import scan_modelbin  # noqa: E402


ARCHIVE = WORKSPACE / "work" / "si" / "materials-v006-eyes" / "characters.iris-v006.zip"
CONVERTER = SCRIPTS / "convert_fh6_swatchbin.py"
HEAD7_BUILDER = SCRIPTS / "build_fh6_head7_donor.py"
VERIFY_CANDIDATE = SCRIPTS / "verify_fh6_modelbin_candidate.py"
CLI = (
    WORKSPACE
    / "tools"
    / "SwatchBinCli"
    / "bin"
    / "Release"
    / "net10.0"
    / "SwatchBinCli.exe"
)
WHITE_ALPHA_64 = (
    "Swatches/1_FlatTextures_DiffuseWhiteAlpha_"
    "9a50b2ee-1d34-4e09-bc2e-a39d4e7b5201.swatchbin"
)


class MaterialPipelineTests(unittest.TestCase):
    def test_final_material_candidates_pass_nested_roundtrip_gate(self) -> None:
        root = WORKSPACE / "work" / "si" / "fbx-source"
        cases = (
            (
                "helmet",
                root
                / "milestone-05-modelbin-lod0-v002"
                / "helmet"
                / "Helmet_Race_Modern.lod0-head7.modelbin",
                root
                / "material-pipeline-v002"
                / "candidates"
                / "Helmet_Race_Modern.lod0-materials-v002.modelbin",
                root
                / "milestone-04-intermediate-v002"
                / "lod0"
                / "head-hair.intermediate.manifest.json",
                root
                / "material-pipeline-v002"
                / "profiles"
                / "helmet-final-lod0-v002.json",
            ),
            (
                "outfit",
                root
                / "milestone-05-modelbin-lod0-v002"
                / "outfit"
                / "Outfit_Race_Suit_Modern_F.lod0-final.modelbin",
                root
                / "material-pipeline-v002"
                / "candidates"
                / "Outfit_Race_Suit_Modern_F.lod0-materials-v002.modelbin",
                root
                / "milestone-04-intermediate-v002"
                / "lod0"
                / "body-garment.intermediate.manifest.json",
                root
                / "material-pipeline-v002"
                / "profiles"
                / "outfit-final-lod0-v002.json",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="fh6-material-verify-test-") as temporary:
            for component, geometry, candidate, manifest, profile in cases:
                with self.subTest(component=component):
                    report = Path(temporary) / f"{component}.json"
                    subprocess.run(
                        [
                            sys.executable,
                            str(VERIFY_CANDIDATE),
                            str(geometry),
                            str(candidate),
                            str(manifest),
                            "--component",
                            component,
                            "--material-profile",
                            str(profile),
                            "--report",
                            str(report),
                        ],
                        cwd=WORKSPACE,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    result = json.loads(report.read_text(encoding="utf-8"))
                    checks = {item["name"]: item for item in result["checks"]}
                    self.assertEqual(result["summary"]["failed"], 0)
                    self.assertTrue(checks["bundle.lossless_roundtrip"]["passed"])
                    self.assertTrue(
                        checks["materials.nested_profile_roundtrip"]["passed"]
                    )

    def test_blob_sequence_writer_is_exact_without_sequence_changes(self) -> None:
        donor = (
            WORKSPACE
            / "work"
            / "si"
            / "components"
            / "baselines"
            / "helmet"
            / "extracted"
            / "Helmet_Race_Modern.modelbin"
        )
        original = donor.read_bytes()
        bundle = parse_bundle(original)
        rebuilt = rebuild_with_blob_sequence(
            bundle, [blob_spec(blob) for blob in bundle.blobs]
        )
        self.assertIsNone(first_difference(original, rebuilt))

    def test_head7_structural_donor_adds_sclera_mesh_and_material(self) -> None:
        helmet = (
            WORKSPACE
            / "work"
            / "si"
            / "components"
            / "baselines"
            / "helmet"
            / "extracted"
            / "Helmet_Race_Modern.modelbin"
        )
        alice = (
            WORKSPACE
            / "work"
            / "si"
            / "components"
            / "baselines"
            / "head-alice-f"
            / "extracted"
            / "Driver_Alice_F.modelbin"
        )
        with tempfile.TemporaryDirectory(prefix="fh6-head7-test-") as temporary:
            root = Path(temporary)
            output = root / "Helmet_Race_Modern.head7.modelbin"
            report = root / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(HEAD7_BUILDER),
                    str(helmet),
                    str(alice),
                    str(output),
                    "--report",
                    str(report),
                ],
                cwd=WORKSPACE,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(report.read_text(encoding="utf-8"))
            bundle = parse_bundle(output.read_bytes())
            materials = materials_by_id(bundle)
            shader, atst, _nested, _mtpr = shader_info(materials[6].data)
            self.assertEqual(result["validation"]["mesh_count"], 7)
            self.assertEqual(result["validation"]["material_count"], 7)
            self.assertEqual((shader, atst), ("charactereyeball", "0000"))

    def test_parameter_name_crc32_contract(self) -> None:
        self.assertEqual(parameter_name_hash("DiffuseTexture"), 0xEE34B08B)
        self.assertEqual(parameter_name_hash("MainNormalsTexture"), 0x17833B17)
        self.assertEqual(parameter_name_hash("TextureTexture"), 0x16F462B7)
        self.assertEqual(parameter_name_hash("MainTintColorParam"), 0x404E4B67)

    def test_swatch_header_parser_matches_retail_template(self) -> None:
        with zipfile.ZipFile(ARCHIVE) as zipped:
            data = zipped.read(WHITE_ALPHA_64)
        info = parse_swatchbin(data)
        self.assertEqual((info["width"], info["height"], info["mips"]), (64, 64, 7))
        self.assertEqual(info["encoding_name"], "BC7")
        self.assertEqual(info["srgb_flags"], [1, 1])
        self.assertEqual(info["guid"], "9a50b2ee-1d34-4e09-bc2e-a39d4e7b5201")

    def test_raw_rgba_encoder_updates_guid_and_audits_alpha(self) -> None:
        self.assertTrue(CLI.exists())
        with tempfile.TemporaryDirectory(prefix="fh6-material-test-") as temporary:
            root = Path(temporary)
            source = root / "transparent.png"
            output_guid = uuid.UUID("10101010-2020-3030-4040-505050505050")
            output = root / f"si_test_alpha_{output_guid}.swatchbin"
            report = root / "report.json"
            image = Image.new("RGBA", (64, 64), (19, 37, 73, 0))
            for x in range(32, 64):
                for y in range(64):
                    image.putpixel((x, y), (211, 101, 47, 255))
            image.save(source)
            subprocess.run(
                [
                    sys.executable,
                    str(CONVERTER),
                    "--template-archive",
                    str(ARCHIVE),
                    "--template-entry",
                    WHITE_ALPHA_64,
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--color-space",
                    "srgb",
                    "--resample",
                    "none",
                ],
                cwd=WORKSPACE,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(result["source"]["premultiplied"])
            self.assertEqual(result["source"]["alpha"]["zero_pixels"], 2048)
            self.assertEqual(result["output"]["header"]["guid"], str(output_guid))
            self.assertEqual(result["validation"]["hard_error_count"], 0)

    def test_tga_and_dds_frontend_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fh6-material-formats-") as temporary:
            root = Path(temporary)
            image = Image.new("RGBA", (64, 64), (31, 79, 127, 193))
            for extension, output_guid in (
                ("tga", uuid.UUID("11111111-2222-3333-4444-555555555555")),
                ("dds", uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
            ):
                source = root / f"source.{extension}"
                output = root / f"si_test_{extension}_{output_guid}.swatchbin"
                report = root / f"{extension}.json"
                image.save(source)
                subprocess.run(
                    [
                        sys.executable,
                        str(CONVERTER),
                        "--template-archive",
                        str(ARCHIVE),
                        "--template-entry",
                        WHITE_ALPHA_64,
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--color-space",
                        "srgb",
                        "--resample",
                        "none",
                    ],
                    cwd=WORKSPACE,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(result["source"]["format"], extension.upper())
                self.assertEqual(result["output"]["header"]["guid"], str(output_guid))

    def test_profile_writer_rejects_unpatched_template_texture_slots(self) -> None:
        donor = (
            WORKSPACE
            / "work"
            / "donors"
            / "Outfit_Race_Suit_Modern_F"
            / "extracted"
            / "Outfit_Race_Suit_Modern_F.modelbin"
        )
        outer = parse_bundle(donor.read_bytes())
        material = materials_by_id(outer)[0]
        _shader, _atst, _nested, mtpr = shader_info(material.data)
        with self.assertRaisesRegex(ValueError, "Every template texture must be patched"):
            patch_mtpr(
                mtpr.data,
                {0xEE34B08B: r"Game:\Media\_library\texturespg\characters\swatches\si_test.swatchbin"},
                {},
                require_all_textures=True,
            )

    def test_retail_alpha_cloth_template_has_only_explicit_source_slots(self) -> None:
        donor = (
            WORKSPACE
            / "work"
            / "si"
            / "fbx-source"
            / "material-donors-v001"
            / "Lower_Jean_Shorts_Ripped_F"
            / "extracted"
            / "Lower_Jean_Shorts_Ripped_F.modelbin"
        )
        material = materials_by_id(parse_bundle(donor.read_bytes()))[3]
        shader, atst, _nested, mtpr = shader_info(material.data)
        _patched, _changes, texture_hashes = patch_mtpr(
            mtpr.data,
            {
                0xEE34B08B: r"Game:\Media\_library\texturespg\characters\swatches\si_alpha_diff.swatchbin",
                0x17833B17: r"Game:\Media\_library\texturespg\characters\swatches\si_alpha_nrml.swatchbin",
            },
            {
                0x64490D74: {
                    "name": "FuzzAmount_floatVal",
                    "type": 2,
                    "value": 0.0,
                }
            },
            require_all_textures=True,
        )
        self.assertEqual((shader, atst), ("uber_clothing_transparancy", "0100"))
        self.assertEqual(texture_hashes, {0xEE34B08B, 0x17833B17})

    def test_leakage_scan_detects_current_racesuit_donor_paths(self) -> None:
        candidate = (
            WORKSPACE
            / "work"
            / "si"
            / "fbx-source"
            / "milestone-05-modelbin-lod0-v001"
            / "outfit"
            / "Outfit_Race_Suit_Modern_F.lod0-candidate.modelbin"
        )
        result = scan_modelbin(
            candidate.read_bytes(),
            str(candidate),
            ("race_suit", "branding", "stitch", "detailnormal"),
            "si_",
        )
        self.assertTrue(result["findings"])
        self.assertTrue(any(finding["kind"] == "forbidden_token" for finding in result["findings"]))


if __name__ == "__main__":
    unittest.main()
