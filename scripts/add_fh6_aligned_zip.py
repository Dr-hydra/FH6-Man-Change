#!/usr/bin/env python3
"""Append new aligned entries to an FH6 ZIP without recompressing old payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_fh6_aligned_zip import (  # noqa: E402
    CENTRAL,
    EOCD,
    LOCAL,
    central_alignment_extra,
    deflate,
    local_alignment_extra,
    read_entries,
    read_eocd,
    verify,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_additions(values: list[str]) -> dict[str, Path]:
    additions: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Addition must be ENTRY=FILE: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.replace("\\", "/").lstrip("/")
        path = Path(raw_path).resolve(strict=True)
        if not name or name in additions:
            raise ValueError(f"Invalid or duplicate addition: {name!r}")
        additions[name] = path
    if not additions:
        raise ValueError("At least one --add entry is required")
    return additions


def build(source: Path, output: Path, additions: dict[str, Path]) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_size = source.stat().st_size
    with source.open("rb") as stream:
        eocd, comment, _eocd_offset = read_eocd(stream, source_size)
        entries = read_entries(stream, eocd)
        existing = {entry.name for entry in entries}
        overlap = sorted(existing & set(additions))
        if overlap:
            raise ValueError(f"Entries already exist: {overlap}")
        header_template = next(
            (
                entry
                for entry in entries
                if entry.name.startswith("Swatches/") and entry.method == 8
            ),
            None,
        )
        if header_template is None:
            header_template = next(
                (entry for entry in entries if entry.method == 8),
                None,
            )
        if header_template is None:
            raise ValueError("Source archive has no deflated entry to clone")
        name_encoding = "utf-8" if header_template.flags & 0x800 else "cp437"
        central_offset = eocd[6]
        stream.seek(0)
        local_region = stream.read(central_offset)

    raw_additions = {name: path.read_bytes() for name, path in additions.items()}
    new_entries: list[tuple[bytes, tuple, bytes, bytes, dict]] = []
    with output.open("xb") as dst:
        dst.write(local_region)
        for name, raw in raw_additions.items():
            name_bytes = name.encode(name_encoding)
            compressed = deflate(raw)
            crc = zlib.crc32(raw) & 0xFFFFFFFF
            local_offset = dst.tell()
            prefix = local_offset + LOCAL.size + len(name_bytes)
            local_extra = local_alignment_extra(header_template.extra, prefix)
            payload_offset = prefix + len(local_extra)
            if payload_offset % 4096:
                raise AssertionError("New local payload is not aligned")
            local_fixed = LOCAL.pack(
                b"PK\x03\x04",
                header_template.fixed[2],
                header_template.fixed[3],
                header_template.fixed[4],
                header_template.fixed[5],
                header_template.fixed[6],
                crc,
                len(compressed),
                len(raw),
                len(name_bytes),
                len(local_extra),
            )
            dst.write(local_fixed)
            dst.write(name_bytes)
            dst.write(local_extra)
            dst.write(compressed)
            # The central-directory alignment field stores the absolute payload
            # offset as a fixed four-byte value; only the local header carries
            # the variable-size padding.
            central_extra = central_alignment_extra(
                header_template.extra, payload_offset
            )
            central_fixed = CENTRAL.pack(
                b"PK\x01\x02",
                header_template.fixed[1],
                header_template.fixed[2],
                header_template.fixed[3],
                header_template.fixed[4],
                header_template.fixed[5],
                header_template.fixed[6],
                crc,
                len(compressed),
                len(raw),
                len(name_bytes),
                len(central_extra),
                0,
                header_template.fixed[13],
                header_template.fixed[14],
                header_template.fixed[15],
                local_offset,
            )
            new_entries.append(
                (
                    name_bytes,
                    central_fixed,
                    central_extra,
                    b"",
                    {
                        "entry": name,
                        "source_file": str(additions[name]),
                        "uncompressed_bytes": len(raw),
                        "compressed_bytes": len(compressed),
                        "crc32": f"{crc:08x}",
                        "payload_offset": payload_offset,
                        "payload_sha256": sha256_bytes(raw),
                    },
                )
            )

        new_central_offset = dst.tell()
        for entry in entries:
            dst.write(CENTRAL.pack(*entry.fixed))
            dst.write(entry.name_bytes)
            dst.write(entry.extra)
            dst.write(entry.comment)
        for name_bytes, fixed, extra, entry_comment, _report in new_entries:
            dst.write(fixed)
            dst.write(name_bytes)
            dst.write(extra)
            dst.write(entry_comment)
        central_size = dst.tell() - new_central_offset
        dst.write(
            EOCD.pack(
                b"PK\x05\x06",
                0,
                0,
                len(entries) + len(new_entries),
                len(entries) + len(new_entries),
                central_size,
                new_central_offset,
                len(comment),
            )
        )
        dst.write(comment)

    import zipfile

    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Output ZIP failed CRC validation")
        for name, path in additions.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"Added payload mismatch: {name}")
    alignment_validation = verify(
        output, additions, len(entries) + len(new_entries)
    )
    return {
        "entries_before": len(entries),
        "entries_after": len(entries) + len(new_entries),
        "header_template": {
            "entry": header_template.name,
            "version_made_by": header_template.fixed[1],
            "version_needed": header_template.fixed[2],
            "flags": header_template.fixed[3],
            "compression_method": header_template.fixed[4],
        },
        "additions": [item[-1] for item in new_entries],
        "archive": alignment_validation,
        "output_sha256": sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--add", action="append", default=[], metavar="ENTRY=FILE")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    additions = parse_additions(args.add)
    result = build(source, output, additions)
    backup = None
    if args.apply:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = source.with_name(source.name + f".bak-{timestamp}")
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite {backup}")
        shutil.copy2(source, backup)
        temporary = source.with_name(source.name + f".tmp-{timestamp}")
        try:
            shutil.copy2(output, temporary)
            if sha256_file(temporary) != result["output_sha256"]:
                raise ValueError("Temporary deployment hash mismatch")
            os.replace(temporary, source)
        finally:
            if temporary.exists():
                temporary.unlink()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": {"path": str(source), "sha256": sha256_file(backup or source)},
        "output": {"path": str(output), "sha256": result["output_sha256"]},
        "validation": result,
        "deployment": {"applied": args.apply, "backup": str(backup) if backup else None},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_ALIGNED_ZIP_ADD=" + json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
