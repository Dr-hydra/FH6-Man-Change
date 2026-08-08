#!/usr/bin/env python3
"""Decode and visualize the final Si Display LOD0 material swatches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat

from convert_fh6_swatchbin import image_alpha, parse_swatchbin
from modelbin_bundle import parse_bundle
from patch_fh6_material_profile import materials_by_id, patch_mtpr, shader_info


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
PIPELINE_ROOT = WORKSPACE / "work" / "si" / "fbx-source" / "material-pipeline-v002"
DEFAULT_MANIFEST = PIPELINE_ROOT / "swatches" / "si-display-swatches-v002.json"
DEFAULT_HELMET_PROFILE = PIPELINE_ROOT / "profiles" / "helmet-final-lod0-v002.json"
DEFAULT_HELMET = (
    PIPELINE_ROOT
    / "candidates"
    / "Helmet_Race_Modern.lod0-materials-v002.modelbin"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-06-display-package-lod0-v001"
    / "previews"
)
DEFAULT_CLI = (
    WORKSPACE
    / "tools"
    / "SwatchBinCli"
    / "bin"
    / "Release"
    / "net10.0"
    / "SwatchBinCli.exe"
)
PREVIEW_KEYS = (
    "face_base",
    "body_base",
    "hair_base",
    "hair_normal_spec",
    "iris_base",
    "cloth_base",
    "cloth_normal",
    "cloth_alpha_base",
    "cloth_alpha_normal",
    "opaque_white",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--helmet-profile", type=Path, default=DEFAULT_HELMET_PROFILE)
    parser.add_argument("--helmet", type=Path, default=DEFAULT_HELMET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        return opened.convert("RGBA")


def image_summary(image: Image.Image) -> dict[str, Any]:
    extrema = image.getextrema()
    return {
        "size": list(image.size),
        "mean_rgba": [round(float(value), 6) for value in ImageStat.Stat(image).mean],
        "extrema_rgba": [[int(low), int(high)] for low, high in extrema],
        "alpha": image_alpha(image),
    }


def checker_composite(image: Image.Image, cell: int = 16) -> Image.Image:
    checker = Image.new("RGBA", image.size, (196, 196, 196, 255))
    draw = ImageDraw.Draw(checker)
    for y in range(0, image.height, cell):
        for x in range(0, image.width, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle(
                    (x, y, min(x + cell - 1, image.width), min(y + cell - 1, image.height)),
                    fill=(236, 236, 236, 255),
                )
    return Image.alpha_composite(checker, image)


def alpha_preview(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    return Image.merge("RGBA", (alpha, alpha, alpha, Image.new("L", image.size, 255)))


def normal_preview(image: Image.Image, maximum: int = 512) -> Image.Image:
    source = image.copy()
    source.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    result = Image.new("RGBA", source.size)
    output = []
    pixels = source.load()
    for row in range(source.height):
        for column in range(source.width):
            red, green, _blue, _alpha = pixels[column, row]
            x = red / 127.5 - 1.0
            y = green / 127.5 - 1.0
            z = math.sqrt(max(0.0, 1.0 - x * x - y * y))
            output.append((red, green, round((z * 0.5 + 0.5) * 255.0), 255))
    result.putdata(output)
    return result


def fitted(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGBA", size, (34, 36, 40, 255))
    working = image.copy()
    working.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - working.width) // 2
    y = (size[1] - working.height) // 2
    result.alpha_composite(working, (x, y))
    return result


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def sheet(
    title: str,
    panels: list[tuple[str, Image.Image]],
    output: Path,
    *,
    columns: int = 4,
) -> None:
    panel_width = 360
    image_height = 320
    label_height = 48
    title_height = 64
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGBA",
        (columns * panel_width, title_height + rows * (image_height + label_height)),
        (24, 26, 30, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), title, fill=(242, 244, 248, 255), font=font(24))
    for index, (label, image) in enumerate(panels):
        column = index % columns
        row = index // columns
        x = column * panel_width
        y = title_height + row * (image_height + label_height)
        canvas.alpha_composite(fitted(image, (panel_width, image_height)), (x, y))
        draw.text(
            (x + 12, y + image_height + 12),
            label,
            fill=(224, 228, 234, 255),
            font=font(18),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output, quality=95)


def main() -> int:
    args = arguments()
    manifest_path = args.manifest.resolve(strict=True)
    profile_path = args.helmet_profile.resolve(strict=True)
    helmet_path = args.helmet.resolve(strict=True)
    cli = args.cli.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview directory {output}")
    decoded_dir = output / "decoded"
    decoded_dir.mkdir(parents=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    swatches = manifest["swatches"]
    missing = sorted(set(PREVIEW_KEYS) - set(swatches))
    if missing:
        raise ValueError(f"Swatch manifest is missing preview keys: {missing}")

    decoded: dict[str, Image.Image] = {}
    sources: dict[str, Image.Image] = {}
    swatch_report: dict[str, Any] = {}
    hard_errors: list[str] = []
    for key in PREVIEW_KEYS:
        record = swatches[key]
        swatch_path = Path(record["path"]).resolve(strict=True)
        source_path = Path(record["source"]).resolve(strict=True)
        decoded_path = decoded_dir / f"{key}.png"
        completed = subprocess.run(
            [str(cli), "decode", str(swatch_path), str(decoded_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        source_image = load_rgba(source_path)
        decoded_image = load_rgba(decoded_path)
        header = parse_swatchbin(swatch_path.read_bytes())
        conversion_report = json.loads(Path(record["report"]).read_text(encoding="utf-8"))
        if conversion_report["validation"]["hard_error_count"]:
            hard_errors.append(f"{key}: conversion report has hard errors")
        if decoded_image.size != (header["width"], header["height"]):
            hard_errors.append(f"{key}: decoded dimensions differ from swatch header")
        decoded[key] = decoded_image
        sources[key] = source_image
        swatch_report[key] = {
            "source": {"path": str(source_path), **image_summary(source_image)},
            "swatch": {
                "path": str(swatch_path),
                "bytes": swatch_path.stat().st_size,
                "sha256": sha256_path(swatch_path),
                "header": header,
            },
            "decoded": {
                "path": str(decoded_path),
                "decoder_stdout": completed.stdout.strip().splitlines(),
                **image_summary(decoded_image),
            },
            "conversion_error": conversion_report["output"]["decoded_error"],
        }

    helmet = parse_bundle(helmet_path.read_bytes())
    materials = materials_by_id(helmet)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    sclera_spec = next(
        item for item in profile["materials"] if item.get("role") == "sclera"
    )
    sclera_id = int(sclera_spec["target_material_id"])
    shader, atst, _nested, mtpr = shader_info(materials[sclera_id].data)
    _unchanged, _changes, sclera_textures = patch_mtpr(
        mtpr.data, {}, {}, require_all_textures=False
    )
    sclera_valid = (
        shader == "charactereyeball"
        and atst == "0000"
        and not sclera_textures
        and not sclera_spec.get("texture_patches")
    )
    if not sclera_valid:
        hard_errors.append("sclera: expected opaque no-texture charactereyeball")

    iris_alpha = swatch_report["iris_base"]["decoded"]["alpha"]
    if iris_alpha["unique_values"] < 16 or iris_alpha["nonopaque_fraction"] <= 0.0:
        hard_errors.append("iris_base: decoded alpha distribution was flattened")
    cloth_alpha = swatch_report["cloth_alpha_base"]["decoded"]["alpha"]
    if cloth_alpha["unique_values"] < 16 or cloth_alpha["nonopaque_fraction"] <= 0.0:
        hard_errors.append("cloth_alpha_base: decoded alpha distribution was flattened")
    hair_alpha = swatch_report["hair_base"]["decoded"]["alpha"]
    if hair_alpha["unique_values"] < 16 or hair_alpha["nonopaque_fraction"] <= 0.0:
        hard_errors.append("hair_base: decoded alpha distribution was flattened")
    white_extrema = swatch_report["opaque_white"]["decoded"]["extrema_rgba"]
    if white_extrema != [[255, 255], [255, 255], [255, 255], [255, 255]]:
        hard_errors.append("opaque_white: decoded reference is not solid opaque white")

    eye_path = output / "eyes-sclera-iris-preview.png"
    iris_on_white = Image.alpha_composite(
        Image.new("RGBA", decoded["iris_base"].size, (255, 255, 255, 255)),
        decoded["iris_base"],
    )
    sheet(
        "Eyes: source, decoded BC7, alpha, and white-sclera composite",
        [
            ("Iris source", checker_composite(sources["iris_base"])),
            ("Iris decoded", checker_composite(decoded["iris_base"])),
            ("Iris alpha", alpha_preview(decoded["iris_base"])),
            ("Iris over white sclera", iris_on_white),
            ("Opaque-white reference", decoded["opaque_white"]),
        ],
        eye_path,
    )

    alpha_cloth_path = output / "alpha-cloth-preview.png"
    sheet(
        "Alpha cloth: source and decoded color/alpha/normal",
        [
            ("Alpha cloth source", checker_composite(sources["cloth_alpha_base"])),
            ("Alpha cloth decoded", checker_composite(decoded["cloth_alpha_base"])),
            ("Alpha cloth alpha", alpha_preview(decoded["cloth_alpha_base"])),
            ("Normal XY reconstructed", normal_preview(decoded["cloth_alpha_normal"])),
        ],
        alpha_cloth_path,
    )

    overview_path = output / "material-overview.png"
    sheet(
        "Si Display LOD0 decoded material overview",
        [
            ("Face base", decoded["face_base"]),
            ("Body base", decoded["body_base"]),
            ("Opaque cloth base", decoded["cloth_base"]),
            ("Opaque cloth normal", normal_preview(decoded["cloth_normal"])),
            ("Hair base", checker_composite(decoded["hair_base"])),
            ("Hair alpha", alpha_preview(decoded["hair_base"])),
            ("Hair normal/spec", normal_preview(decoded["hair_normal_spec"])),
            ("Alpha cloth", checker_composite(decoded["cloth_alpha_base"])),
        ],
        overview_path,
    )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Offline decoded-pixel audit of final Si Display LOD0 swatches; not a game-render claim.",
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
            "helmet_profile": {"path": str(profile_path), "sha256": sha256_path(profile_path)},
            "helmet": {"path": str(helmet_path), "sha256": sha256_path(helmet_path)},
            "decoder": {"path": str(cli), "sha256": sha256_path(cli)},
        },
        "sclera": {
            "material_id": sclera_id,
            "shader": shader,
            "atst": atst,
            "texture_parameter_count": len(sclera_textures),
            "parameter_count": int(mtpr.data[0]),
            "opaque_white_reference": swatch_report["opaque_white"]["decoded"],
            "valid": sclera_valid,
            "note": "The retail charactereyeball template is textureless; the white tile is a visual reference, not a bound sclera texture.",
        },
        "swatches": swatch_report,
        "previews": {
            "eyes": str(eye_path),
            "alpha_cloth": str(alpha_cloth_path),
            "overview": str(overview_path),
        },
        "validation": {
            "decoded_top_mips": True,
            "sclera_retail_shader_identity": sclera_valid,
            "iris_alpha_preserved": "iris_base:" not in " ".join(hard_errors),
            "alpha_cloth_alpha_preserved": "cloth_alpha_base:" not in " ".join(hard_errors),
            "hair_alpha_preserved": "hair_base:" not in " ".join(hard_errors),
            "hard_error_count": len(hard_errors),
            "hard_errors": hard_errors,
            "game_validated": False,
        },
    }
    report_path = output / "material-preview.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "SI_DISPLAY_MATERIAL_PREVIEW="
        + json.dumps(
            {
                "output": str(output),
                "report": str(report_path),
                "previews": report["previews"],
                "hard_errors": len(hard_errors),
            },
            separators=(",", ":"),
        )
    )
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
