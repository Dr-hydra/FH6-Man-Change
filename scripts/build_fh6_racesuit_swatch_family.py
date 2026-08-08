#!/usr/bin/env python3
"""Build every race-suit diffuse variant from validated source swatch payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cloth1", required=True, type=Path)
    parser.add_argument("--cloth2", required=True, type=Path)
    parser.add_argument("--body", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=Path)
    parser.add_argument("--shoes", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def texture_info(data: bytes) -> dict[str, int]:
    if len(data) < 0x84 or data[:4] != b"burG":
        raise ValueError("Not an FH6 swatchbin")
    return {
        "header_size": struct.unpack_from("<I", data, 0x08)[0],
        "total_size": struct.unpack_from("<I", data, 0x0C)[0],
        "width": struct.unpack_from("<I", data, 0x4C)[0],
        "height": struct.unpack_from("<I", data, 0x50)[0],
        "mips": data[0x5A],
        "encoding": struct.unpack_from("<I", data, 0x74)[0],
    }


def family(name: str) -> str | None:
    stem = Path(name).name.lower()
    if not stem.endswith(".swatchbin") or not stem.startswith("outfit_race_suit_modern_"):
        return None
    if "_gloves_diffuse_" in stem or "_gloves_diff_" in stem:
        return "gloves"
    if "_shoes_diffuse_" in stem or "_shoes_diff_" in stem:
        return "shoes"
    if "_diffuse_" in stem or "_diff_" in stem:
        return "main"
    return None


def choose_master(name: str, group: str, masters: dict[str, bytes]) -> tuple[str, bytes]:
    lowered = Path(name).name.lower()
    if group == "main" and "_diffuse_blue_" in lowered:
        return "Cloth2", masters["cloth2"]
    if group == "gloves" and "_diffuse_hwhite_" in lowered:
        return "Cloth1Alpha", masters["alpha"]
    if group == "main":
        return "Cloth1", masters["cloth1"]
    if group == "gloves":
        return "肌", masters["body"]
    return "Cloth1", masters["shoes"]


def transplant(target: bytes, master: bytes) -> bytes:
    target_info = texture_info(target)
    master_info = texture_info(master)
    comparable = ("width", "height", "mips", "encoding")
    if any(target_info[key] != master_info[key] for key in comparable):
        raise ValueError(f"Target/master texture layout mismatch: {target_info} != {master_info}")
    target_data_size = len(target) - target_info["header_size"]
    master_data_size = len(master) - master_info["header_size"]
    if target_data_size != master_data_size:
        raise ValueError("Target/master BCn payload sizes differ")
    return target[: target_info["header_size"]] + master[master_info["header_size"] :]


def main() -> None:
    args = arguments()
    archive = args.archive.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    output_dir.mkdir(parents=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    master_paths = {
        "cloth1": args.cloth1.resolve(strict=True),
        "cloth2": args.cloth2.resolve(strict=True),
        "body": args.body.resolve(strict=True),
        "alpha": args.alpha.resolve(strict=True),
        "shoes": args.shoes.resolve(strict=True),
    }
    masters = {key: path.read_bytes() for key, path in master_paths.items()}
    outputs = []
    with zipfile.ZipFile(archive) as zipped:
        selected = [(entry, family(entry.filename)) for entry in zipped.infolist()]
        selected = [(entry, group) for entry, group in selected if group is not None]
        if len(selected) != 17:
            raise ValueError(f"Expected 17 race-suit diffuse variants, found {len(selected)}")
        for entry, group in selected:
            source_material, master = choose_master(entry.filename, group, masters)
            target = zipped.read(entry)
            output = transplant(target, master)
            output_path = output_dir / entry.filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(output)
            outputs.append(
                {
                    "entry": entry.filename,
                    "family": group,
                    "source_material": source_material,
                    "bytes": len(output),
                    "sha256": sha256(output),
                    "layout": texture_info(output),
                }
            )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "archive": str(archive),
        "masters": {key: {"path": str(path), "sha256": sha256(masters[key])} for key, path in master_paths.items()},
        "outputs": outputs,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_RACESUIT_SWATCH_FAMILY=" + json.dumps({"count": len(outputs), "output_dir": str(output_dir)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
