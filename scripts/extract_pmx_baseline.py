#!/usr/bin/env python3
"""Safely extract a PMX source archive and write a reproducible hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


class BaselineError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts:
        raise BaselineError(f"unsafe absolute or empty ZIP path: {name!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise BaselineError(f"unsafe ZIP path component: {name!r}")
    if path.parts[0].endswith(":"):
        raise BaselineError(f"unsafe drive-qualified ZIP path: {name!r}")
    return path


def is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def extract_baseline(archive: Path, output: Path, pmx_name: str) -> dict:
    if output.exists() and any(output.iterdir()):
        raise BaselineError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    archive_bytes = archive.read_bytes()
    manifest_entries: list[dict] = []
    destinations: set[PurePosixPath] = set()
    pmx_count = 0

    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            if is_symlink(info):
                raise BaselineError(f"symbolic-link ZIP member is not allowed: {info.filename!r}")

            member = safe_member_path(info.filename)
            if member.suffix.lower() == ".pmx":
                pmx_count += 1
                destination = PurePosixPath(pmx_name)
            else:
                destination = member

            if destination in destinations:
                raise BaselineError(f"duplicate extraction destination: {destination}")
            destinations.add(destination)

            data = source.read(info)
            if len(data) != info.file_size:
                raise BaselineError(f"size mismatch while reading {info.filename!r}")

            target = output.joinpath(*destination.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest_entries.append(
                {
                    "archive_name": info.filename,
                    "path": destination.as_posix(),
                    "size": len(data),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(data),
                }
            )

    if pmx_count != 1:
        raise BaselineError(f"expected exactly one PMX entry, found {pmx_count}")

    manifest = {
        "schema_version": 1,
        "archive": {
            "path": str(archive.resolve()),
            "size": len(archive_bytes),
            "sha256": sha256_bytes(archive_bytes),
        },
        "pmx_path": pmx_name,
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
    }
    manifest_path = output / "baseline.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pmx-name", default="si.pmx")
    args = parser.parse_args()

    try:
        manifest = extract_baseline(args.archive, args.output, args.pmx_name)
    except (OSError, zipfile.BadZipFile, BaselineError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Extracted {manifest['entry_count']} files to {args.output.resolve()}")
    print(f"Archive SHA-256: {manifest['archive']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

