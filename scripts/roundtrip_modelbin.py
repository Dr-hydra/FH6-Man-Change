#!/usr/bin/env python3
"""Losslessly rebuild an FH6 .modelbin and require byte-for-byte identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from modelbin_bundle import BundleError, first_difference, parse_bundle


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, help="Optional JSON validation report")
    args = parser.parse_args()

    try:
        source = args.source.resolve(strict=True)
        output = args.output.resolve()
        report_path = args.report.resolve() if args.report else None
        if source == output:
            raise BundleError("source and output must be different files")
        if output.exists():
            raise BundleError(f"refusing to overwrite output: {output}")
        if report_path and report_path.exists():
            raise BundleError(f"refusing to overwrite report: {report_path}")

        original = source.read_bytes()
        bundle = parse_bundle(original)
        rebuilt = bundle.rebuild_lossless()
        difference = first_difference(original, rebuilt)
        if difference is not None:
            raise BundleError(f"round-trip mismatch at byte 0x{difference:X}")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rebuilt)
        output_hash = sha256(output.read_bytes())
        source_hash = sha256(original)
        if output_hash != source_hash:
            raise BundleError("output hash changed after writing")

        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "output": str(output),
            "source_size": len(original),
            "output_size": len(rebuilt),
            "source_sha256": source_hash,
            "output_sha256": output_hash,
            "byte_identical": True,
            "bundle_version": f"{bundle.version[0]}.{bundle.version[1]}",
            "declared_size": bundle.declared_size,
            "data_offset": bundle.data_offset,
            "blob_count": len(bundle.blobs),
            "blob_tags": dict(sorted(Counter(blob.tag for blob in bundle.blobs).items())),
            "blobs": [
                {
                    "index": blob.index,
                    "tag": blob.tag,
                    "version": f"{blob.version[0]}.{blob.version[1]}",
                    "metadata_count": len(blob.metadata),
                    "metadata_offset": blob.metadata_offset,
                    "data_offset": blob.data_offset,
                    "data_size": blob.data_size,
                    "trailing_size": blob.trailing_size,
                }
                for blob in bundle.blobs
            ],
        }
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print("FH6_ROUNDTRIP=" + json.dumps({
            "byte_identical": True,
            "size": len(rebuilt),
            "sha256": output_hash,
            "blob_count": len(bundle.blobs),
            "blob_tags": report["blob_tags"],
            "output": str(output),
            "report": str(report_path) if report_path else None,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, BundleError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
