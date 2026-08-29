#!/usr/bin/env python3
"""Convert PNG/TGA/DDS source pixels into a donor-compatible FH6 swatchbin.

The donor header remains the format authority.  Pillow supplies exact,
unpremultiplied RGBA bytes to ``SwatchBinCli encode-raw``; the C# encoder only
compresses those bytes and never interprets source colour profiles or alpha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
DEFAULT_CLI = (
    WORKSPACE
    / "tools"
    / "SwatchBinCli"
    / "bin"
    / "Release"
    / "net10.0"
    / "SwatchBinCli.exe"
)
GUID_SUFFIX = re.compile(
    r"(?P<guid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
ENCODINGS = {
    0: "BC1",
    1: "BC2",
    2: "BC3",
    3: "BC4",
    5: "BC5",
    9: "BC7",
    22: "BC7",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    template = parser.add_mutually_exclusive_group(required=True)
    template.add_argument("--template", type=Path)
    template.add_argument("--template-archive", type=Path)
    parser.add_argument("--template-entry")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--guid")
    parser.add_argument("--color-space", required=True, choices=("srgb", "linear"))
    parser.add_argument(
        "--resample",
        default="none",
        choices=("none", "nearest", "bilinear", "bicubic", "lanczos"),
    )
    parser.add_argument(
        "--normal-xy",
        action="store_true",
        help=(
            "Require the Si two-channel-normal zero pad and reconstruct normalized Z "
            "after resampling."
        ),
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_swatchbin(data: bytes) -> dict[str, Any]:
    if len(data) < 0x80 or data[:4] != b"burG":
        raise ValueError("Not an FH6 burG swatchbin")
    header_size = struct.unpack_from("<I", data, 0x08)[0]
    total_size = struct.unpack_from("<I", data, 0x0C)[0]
    width, height = struct.unpack_from("<II", data, 0x4C)
    mip_count = data[0x5A]
    encoding = struct.unpack_from("<I", data, 0x74)[0]
    if header_size != 0x80 + mip_count * 12:
        raise ValueError(
            f"Unexpected header size {header_size}; expected {0x80 + mip_count * 12}"
        )
    if total_size != len(data):
        raise ValueError(f"Declared size {total_size} != actual size {len(data)}")
    if width <= 0 or height <= 0 or mip_count <= 0:
        raise ValueError("Invalid dimensions or mip count")
    if encoding not in ENCODINGS:
        raise ValueError(f"Unsupported swatch encoding {encoding}")
    mip_records = []
    expected_offset = 0
    for index in range(mip_count):
        size, offset, next_offset = struct.unpack_from("<III", data, 0x80 + index * 12)
        if offset != expected_offset:
            raise ValueError(
                f"Mip {index} offset {offset} does not follow {expected_offset}"
            )
        expected_next = 0xFFFFFFFF if index + 1 == mip_count else 92 + index * 12
        if next_offset != expected_next:
            raise ValueError(
                f"Mip {index} next descriptor {next_offset:#x} != {expected_next:#x}"
            )
        mip_records.append({"index": index, "size": size, "offset": offset})
        expected_offset += size
    payload_size = len(data) - header_size
    if expected_offset != payload_size:
        raise ValueError(f"Mip sizes sum to {expected_offset}, payload has {payload_size}")
    if struct.unpack_from("<I", data, 0x24)[0] != payload_size:
        raise ValueError("Primary payload-size field is inconsistent")
    if struct.unpack_from("<I", data, 0x28)[0] != payload_size:
        raise ValueError("Secondary payload-size field is inconsistent")
    return {
        "header_size": header_size,
        "total_size": total_size,
        "payload_size": payload_size,
        "guid": str(uuid.UUID(bytes_le=data[0x3C:0x4C])),
        "width": width,
        "height": height,
        "depth": struct.unpack_from("<I", data, 0x54)[0],
        "mips": mip_count,
        "encoding": encoding,
        "encoding_name": ENCODINGS[encoding],
        "srgb_flags": list(struct.unpack_from("<II", data, 0x60)),
        "mip_records": mip_records,
    }


def image_alpha(image: Image.Image) -> dict[str, Any]:
    alpha = image.getchannel("A")
    histogram = alpha.histogram()
    pixels = image.width * image.height
    return {
        "min": int(alpha.getextrema()[0]),
        "max": int(alpha.getextrema()[1]),
        "unique_values": sum(value > 0 for value in histogram),
        "zero_pixels": int(histogram[0]),
        "nonopaque_pixels": int(sum(histogram[:255])),
        "nonopaque_fraction": round(sum(histogram[:255]) / pixels, 9),
    }


def image_digest(image: Image.Image) -> str:
    return sha256_bytes(image.tobytes("raw", "RGBA"))


def resize_channels(image: Image.Image, size: tuple[int, int], name: str) -> Image.Image:
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if name not in filters:
        raise ValueError("Template dimensions differ; choose an explicit resampling filter")
    # Resize channels independently.  This intentionally avoids premultiplying
    # RGB by alpha, because transparent texels can still carry packed data.
    return Image.merge(
        "RGBA",
        tuple(channel.resize(size, filters[name]) for channel in image.split()),
    )


def reconstruct_normal_z(image: Image.Image) -> Image.Image:
    """Expand signed UNORM8 tangent XY into a normalized positive Z channel."""

    if image.mode != "RGBA":
        raise ValueError("Normal reconstruction requires an RGBA image")
    blue_by_xy = bytearray(256 * 256)
    for red in range(256):
        x = red / 127.5 - 1.0
        for green in range(256):
            y = green / 127.5 - 1.0
            z = math.sqrt(max(0.0, 1.0 - x * x - y * y))
            blue_by_xy[(red << 8) | green] = round((z + 1.0) * 127.5)

    rgba = bytearray(image.tobytes("raw", "RGBA"))
    for offset in range(0, len(rgba), 4):
        rgba[offset + 2] = blue_by_xy[(rgba[offset] << 8) | rgba[offset + 1]]
    return Image.frombytes("RGBA", image.size, bytes(rgba))


def decoded_error(expected: Image.Image, actual: Image.Image) -> dict[str, Any]:
    if expected.size != actual.size:
        raise ValueError("Decoded swatch dimensions do not match encoded source")
    difference = ImageChops.difference(expected, actual)
    means = ImageStat.Stat(difference).mean
    extrema = difference.getextrema()
    return {
        "mean_absolute_error_rgba": [round(float(value), 6) for value in means],
        "maximum_absolute_error_rgba": [int(value[1]) for value in extrema],
    }


def output_guid(output: Path, explicit: str | None) -> uuid.UUID:
    if explicit:
        result = uuid.UUID(explicit)
    else:
        match = GUID_SUFFIX.search(output.stem)
        if not match:
            raise ValueError("Output filename must end in a GUID or --guid must be supplied")
        result = uuid.UUID(match.group("guid"))
    return result


def load_template(args: argparse.Namespace) -> tuple[bytes, dict[str, Any]]:
    if args.template is not None:
        path = args.template.resolve(strict=True)
        data = path.read_bytes()
        return data, {"path": str(path), "archive": None, "entry": None}
    if not args.template_entry:
        raise ValueError("--template-entry is required with --template-archive")
    archive = args.template_archive.resolve(strict=True)
    with zipfile.ZipFile(archive) as zipped:
        names = {name.lower(): name for name in zipped.namelist()}
        entry = names.get(args.template_entry.lower())
        if entry is None:
            raise ValueError(f"Template entry {args.template_entry!r} is absent from {archive}")
        data = zipped.read(entry)
    return data, {"path": None, "archive": str(archive), "entry": entry}


def main() -> int:
    args = arguments()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    cli = args.cli.resolve(strict=True)
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    template_data, template_origin = load_template(args)
    template_info = parse_swatchbin(template_data)
    expected_flags = [1, 1] if args.color_space == "srgb" else [0, 0]
    if template_info["srgb_flags"] != expected_flags:
        raise ValueError(
            f"Template colour-space flags {template_info['srgb_flags']} do not match "
            f"requested {args.color_space} ({expected_flags})"
        )
    guid = output_guid(output, args.guid)

    with Image.open(source) as opened:
        source_format = opened.format
        source_mode = opened.mode
        opened.load()
        rgba = opened.convert("RGBA")
    source_rgba_sha256 = image_digest(rgba)
    source_alpha = image_alpha(rgba)
    source_blue = rgba.getchannel("B").getextrema()
    if args.normal_xy and source_blue != (0, 0):
        raise ValueError(f"--normal-xy requires a zero B channel, found {source_blue}")

    target_size = (template_info["width"], template_info["height"])
    resampled = rgba.size != target_size
    converted = resize_channels(rgba, target_size, args.resample) if resampled else rgba.copy()
    if args.normal_xy:
        converted = reconstruct_normal_z(converted)
    converted_alpha = image_alpha(converted)
    converted_blue = converted.getchannel("B").getextrema()

    with tempfile.TemporaryDirectory(prefix="fh6-swatch-") as temporary:
        temporary_dir = Path(temporary)
        template_path = temporary_dir / "template.swatchbin"
        raw_path = temporary_dir / "source.rgba"
        decoded_path = temporary_dir / "decoded.png"
        template_path.write_bytes(template_data)
        raw_path.write_bytes(converted.tobytes("raw", "RGBA"))
        encode = subprocess.run(
            [
                str(cli),
                "encode-raw",
                str(template_path),
                str(raw_path),
                str(converted.width),
                str(converted.height),
                str(output),
                str(guid),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        decode = subprocess.run(
            [str(cli), "decode", str(output), str(decoded_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        with Image.open(decoded_path) as decoded_source:
            decoded_source.load()
            decoded = decoded_source.convert("RGBA")
        error = decoded_error(converted, decoded)
        decoded_alpha = image_alpha(decoded)

    output_data = output.read_bytes()
    output_info = parse_swatchbin(output_data)
    hard_errors: list[str] = []
    if output_info["guid"] != str(guid):
        hard_errors.append("output header GUID does not match requested GUID")
    if output_info["srgb_flags"] != expected_flags:
        hard_errors.append("output colour-space flags changed")
    if (output_info["width"], output_info["height"]) != target_size:
        hard_errors.append("output dimensions changed")
    if output_info["encoding"] != template_info["encoding"]:
        hard_errors.append("output encoding changed")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_path(source),
            "format": source_format,
            "mode": source_mode,
            "size": list(rgba.size),
            "rgba_sha256": source_rgba_sha256,
            "alpha": source_alpha,
            "blue_extrema": list(source_blue),
            "normal_xy_zero_blue": bool(args.normal_xy),
            "premultiplied": False,
        },
        "template": {
            **template_origin,
            "bytes": len(template_data),
            "sha256": sha256_bytes(template_data),
            "header": template_info,
        },
        "conversion": {
            "color_space": args.color_space,
            "raw_rgba_transport": True,
            "resampled": resampled,
            "resample_filter": args.resample if resampled else None,
            "target_size": list(target_size),
            "normal_z_reconstructed": bool(args.normal_xy),
            "converted_blue_extrema": list(converted_blue),
            "converted_rgba_sha256": image_digest(converted),
            "converted_alpha": converted_alpha,
        },
        "output": {
            "path": str(output),
            "bytes": len(output_data),
            "sha256": sha256_bytes(output_data),
            "header": output_info,
            "decoded_alpha": decoded_alpha,
            "decoded_error": error,
        },
        "encoder": {
            "path": str(cli),
            "stdout": encode.stdout.strip().splitlines(),
            "decode_stdout": decode.stdout.strip().splitlines(),
        },
        "validation": {
            "hard_error_count": len(hard_errors),
            "hard_errors": hard_errors,
            "source_alpha_loaded_without_premultiply": True,
            "normal_z_reconstructed_after_resampling": bool(args.normal_xy),
            "template_layout_preserved": not hard_errors,
            "header_guid_matches_filename_or_explicit_guid": output_info["guid"] == str(guid),
            "top_mip_decoded": True,
            "game_validated": False,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_SWATCHBIN=" + json.dumps(report["output"], separators=(",", ":")))
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
