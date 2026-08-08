#!/usr/bin/env python3
"""Replace entries in an FH6 ZIP while preserving 4096-byte payload alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import zlib
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LOCAL = struct.Struct("<4s5H3I2H")
CENTRAL = struct.Struct("<4s6H3I5H2I")
EOCD = struct.Struct("<4s4H2IH")
LOCAL_SIGNATURE = b"PK\x03\x04"
CENTRAL_SIGNATURE = b"PK\x01\x02"
EOCD_SIGNATURE = b"PK\x05\x06"
ALIGNMENT_FIELD_ID = 0x1123
PAYLOAD_ALIGNMENT = 4096


@dataclass
class Entry:
    index: int
    fixed: tuple
    name_bytes: bytes
    extra: bytes
    comment: bytes

    @property
    def flags(self) -> int:
        return self.fixed[3]

    @property
    def method(self) -> int:
        return self.fixed[4]

    @property
    def crc(self) -> int:
        return self.fixed[7]

    @property
    def compressed_size(self) -> int:
        return self.fixed[8]

    @property
    def size(self) -> int:
        return self.fixed[9]

    @property
    def local_offset(self) -> int:
        return self.fixed[16]

    @property
    def name(self) -> str:
        encoding = "utf-8" if self.flags & 0x800 else "cp437"
        return self.name_bytes.decode(encoding)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replace", action="append", default=[], metavar="ENTRY=FILE")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Back up source and deploy the rebuilt archive")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_replacements(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Replacement must be ENTRY=FILE: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.replace("\\", "/").lstrip("/")
        path = Path(raw_path).resolve()
        if not name or not path.is_file():
            raise FileNotFoundError(f"Invalid replacement {name!r} -> {path}")
        if name in result:
            raise ValueError(f"Duplicate replacement entry: {name}")
        result[name] = path
    if not result:
        raise ValueError("At least one --replace entry is required")
    return result


def read_eocd(stream, size: int) -> tuple[tuple, bytes, int]:
    tail_size = min(size, 65_557)
    stream.seek(size - tail_size)
    tail = stream.read(tail_size)
    relative = tail.rfind(EOCD_SIGNATURE)
    if relative < 0 or relative + EOCD.size > len(tail):
        raise ValueError("ZIP EOCD was not found")
    values = EOCD.unpack_from(tail, relative)
    comment_length = values[-1]
    if relative + EOCD.size + comment_length != len(tail):
        raise ValueError("ZIP has unsupported trailing bytes after EOCD")
    if values[1] != 0 or values[2] != 0 or values[3] != values[4]:
        raise ValueError("Multi-disk ZIPs are not supported")
    return values, tail[relative + EOCD.size :], size - tail_size + relative


def read_entries(stream, eocd: tuple) -> list[Entry]:
    entry_count, central_size, central_offset = eocd[4], eocd[5], eocd[6]
    stream.seek(central_offset)
    payload = stream.read(central_size)
    entries: list[Entry] = []
    offset = 0
    for index in range(entry_count):
        if offset + CENTRAL.size > len(payload):
            raise ValueError("Central directory is truncated")
        fixed = CENTRAL.unpack_from(payload, offset)
        if fixed[0] != CENTRAL_SIGNATURE:
            raise ValueError(f"Invalid central entry signature at index {index}")
        name_length, extra_length, comment_length = fixed[10:13]
        start = offset + CENTRAL.size
        name = payload[start : start + name_length]
        extra_start = start + name_length
        extra = payload[extra_start : extra_start + extra_length]
        comment_start = extra_start + extra_length
        comment = payload[comment_start : comment_start + comment_length]
        offset = comment_start + comment_length
        entries.append(Entry(index, fixed, name, extra, comment))
    if offset != len(payload):
        raise ValueError("Central directory size does not match parsed entries")
    return entries


def extra_fields(extra: bytes) -> list[tuple[int, bytes]]:
    fields = []
    offset = 0
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise ValueError("Truncated ZIP extra field")
        field_id, length = struct.unpack_from("<HH", extra, offset)
        offset += 4
        value = extra[offset : offset + length]
        if len(value) != length:
            raise ValueError("Truncated ZIP extra value")
        fields.append((field_id, value))
        offset += length
    return fields


def central_alignment_extra(extra: bytes, payload_offset: int) -> bytes:
    fields = extra_fields(extra)
    matches = [index for index, (field_id, value) in enumerate(fields) if field_id == ALIGNMENT_FIELD_ID]
    if len(matches) != 1 or len(fields[matches[0]][1]) != 4:
        raise ValueError("Expected one four-byte 0x1123 central alignment field")
    fields[matches[0]] = (ALIGNMENT_FIELD_ID, struct.pack("<I", payload_offset))
    return b"".join(struct.pack("<HH", field_id, len(value)) + value for field_id, value in fields)


def local_alignment_extra(original: bytes, payload_prefix_offset: int) -> bytes:
    other = [(field_id, value) for field_id, value in extra_fields(original) if field_id != ALIGNMENT_FIELD_ID]
    preserved = b"".join(struct.pack("<HH", field_id, len(value)) + value for field_id, value in other)
    natural_offset = payload_prefix_offset + len(preserved)
    if natural_offset % PAYLOAD_ALIGNMENT == 0:
        return preserved
    padding_length = (-(natural_offset + 4)) % PAYLOAD_ALIGNMENT
    return preserved + struct.pack("<HH", ALIGNMENT_FIELD_ID, padding_length) + bytes(padding_length)


def deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


def rebuild(source: Path, output: Path, replacements: dict[str, Path]) -> tuple[list[dict], int]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_size = source.stat().st_size
    replacement_bytes = {name: path.read_bytes() for name, path in replacements.items()}

    with source.open("rb") as src:
        eocd, eocd_comment, _eocd_offset = read_eocd(src, source_size)
        entries = read_entries(src, eocd)
        by_name = {entry.name: entry for entry in entries}
        missing = sorted(set(replacements) - set(by_name))
        if missing:
            raise KeyError(f"Replacement entries do not exist: {missing}")
        local_order = sorted(entries, key=lambda entry: entry.local_offset)
        if local_order[0].local_offset != 0:
            raise ValueError("ZIP preambles are not supported")
        if any(entry.flags & 0x08 for entry in entries):
            raise ValueError("ZIP data descriptors are not supported")
        if any(entry.method != 8 for entry in entries):
            raise ValueError("This FH6 archive must use deflate for every entry")

        new_state: dict[int, dict[str, int]] = {}
        replacement_report: list[dict] = []
        with output.open("xb") as dst:
            for entry in local_order:
                src.seek(entry.local_offset)
                local_fixed = LOCAL.unpack(src.read(LOCAL.size))
                if local_fixed[0] != LOCAL_SIGNATURE:
                    raise ValueError(f"Invalid local signature for {entry.name}")
                local_name = src.read(local_fixed[9])
                local_extra = src.read(local_fixed[10])
                if local_name != entry.name_bytes:
                    raise ValueError(f"Local/central name mismatch for {entry.name}")
                old_data_offset = src.tell()
                if old_data_offset % PAYLOAD_ALIGNMENT:
                    raise ValueError(f"Source payload is not aligned: {entry.name}")

                new_local_offset = dst.tell()
                new_extra = local_alignment_extra(
                    local_extra,
                    new_local_offset + LOCAL.size + len(local_name),
                )
                new_data_offset = new_local_offset + LOCAL.size + len(local_name) + len(new_extra)
                if new_data_offset % PAYLOAD_ALIGNMENT:
                    raise AssertionError("Rebuilt payload alignment failed")

                if entry.name in replacement_bytes:
                    raw = replacement_bytes[entry.name]
                    compressed = deflate(raw)
                    crc = zlib.crc32(raw) & 0xFFFFFFFF
                    compressed_size = len(compressed)
                    size = len(raw)
                else:
                    src.seek(old_data_offset)
                    compressed = None
                    crc = entry.crc
                    compressed_size = entry.compressed_size
                    size = entry.size

                updated_local = list(local_fixed)
                updated_local[6:9] = [crc, compressed_size, size]
                updated_local[10] = len(new_extra)
                dst.write(LOCAL.pack(*updated_local))
                dst.write(local_name)
                dst.write(new_extra)
                if compressed is not None:
                    dst.write(compressed)
                    replacement_report.append(
                        {
                            "entry": entry.name,
                            "source_file": str(replacements[entry.name]),
                            "uncompressed_bytes": size,
                            "compressed_bytes": compressed_size,
                            "crc32": f"{crc:08x}",
                            "payload_offset": new_data_offset,
                            "payload_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
                else:
                    remaining = compressed_size
                    while remaining:
                        chunk = src.read(min(4 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(f"Compressed data is truncated for {entry.name}")
                        dst.write(chunk)
                        remaining -= len(chunk)

                new_state[entry.index] = {
                    "local_offset": new_local_offset,
                    "payload_offset": new_data_offset,
                    "crc": crc,
                    "compressed_size": compressed_size,
                    "size": size,
                }

            central_offset = dst.tell()
            for entry in entries:
                state = new_state[entry.index]
                updated = list(entry.fixed)
                updated[7:10] = [state["crc"], state["compressed_size"], state["size"]]
                updated[16] = state["local_offset"]
                new_extra = central_alignment_extra(entry.extra, state["payload_offset"])
                if len(new_extra) != len(entry.extra):
                    raise ValueError("Central extra field size unexpectedly changed")
                dst.write(CENTRAL.pack(*updated))
                dst.write(entry.name_bytes)
                dst.write(new_extra)
                dst.write(entry.comment)
            central_size = dst.tell() - central_offset
            dst.write(
                EOCD.pack(
                    EOCD_SIGNATURE,
                    0,
                    0,
                    len(entries),
                    len(entries),
                    central_size,
                    central_offset,
                    len(eocd_comment),
                )
            )
            dst.write(eocd_comment)
    return replacement_report, len(entries)


def verify(archive: Path, replacements: dict[str, Path], expected_entries: int) -> dict:
    with archive.open("rb") as stream:
        eocd, _comment, _offset = read_eocd(stream, archive.stat().st_size)
        entries = read_entries(stream, eocd)
        if len(entries) != expected_entries:
            raise ValueError("Rebuilt entry count changed")
        for entry in entries:
            stream.seek(entry.local_offset)
            local = LOCAL.unpack(stream.read(LOCAL.size))
            stream.seek(local[9] + local[10], os.SEEK_CUR)
            payload_offset = stream.tell()
            if payload_offset % PAYLOAD_ALIGNMENT:
                raise ValueError(f"Rebuilt payload is not aligned: {entry.name}")
            recorded = dict(extra_fields(entry.extra)).get(ALIGNMENT_FIELD_ID)
            if recorded != struct.pack("<I", payload_offset):
                raise ValueError(f"Central alignment field is wrong: {entry.name}")

    checked = []
    with zipfile.ZipFile(archive) as zipped:
        if len(zipped.infolist()) != expected_entries:
            raise ValueError("zipfile entry count changed")
        for name, path in replacements.items():
            actual = zipped.read(name)
            expected = path.read_bytes()
            if actual != expected:
                raise ValueError(f"Replacement payload mismatch: {name}")
            checked.append({"entry": name, "sha256": hashlib.sha256(actual).hexdigest()})
    return {"entries": expected_entries, "aligned_payloads": expected_entries, "checked_replacements": checked}


def deploy(source: Path, rebuilt: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = source.with_name(source.name + f".bak-{timestamp}")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    shutil.copy2(source, backup)
    temporary = source.with_name(source.name + f".tmp-{timestamp}")
    try:
        shutil.copy2(rebuilt, temporary)
        rebuilt_hash = sha256(rebuilt)
        if sha256(temporary) != rebuilt_hash:
            raise ValueError("Deployed temporary archive hash mismatch")
        os.replace(temporary, source)
        return backup, rebuilt_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = arguments()
    source = args.source.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    replacements = parse_replacements(args.replace)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source:
        raise ValueError("Output must differ from source")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    source_hash = sha256(source)
    replacement_report, entry_count = rebuild(source, output, replacements)
    validation = verify(output, replacements, entry_count)
    output_hash = sha256(output)
    backup = None
    deployed_hash = None
    if args.apply:
        backup, deployed_hash = deploy(source, output)
        if sha256(source) != output_hash:
            raise ValueError("Final deployed archive hash mismatch")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": {"path": str(source), "bytes": source.stat().st_size if not args.apply else backup.stat().st_size, "sha256": source_hash},
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": output_hash},
        "replacements": replacement_report,
        "validation": validation,
        "deployment": {
            "applied": args.apply,
            "backup": str(backup) if backup else None,
            "deployed_sha256": deployed_hash,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_ALIGNED_ZIP=" + json.dumps({"output": str(output), "sha256": output_hash, "applied": args.apply, "backup": str(backup) if backup else None}, separators=(",", ":")))


if __name__ == "__main__":
    main()
