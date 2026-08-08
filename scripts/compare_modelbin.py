#!/usr/bin/env python3
"""Compare two FH6 modelbin files with both bytes and the structural inspector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from inspect_modelbin import ParseError, inspect
from modelbin_bundle import first_difference


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_inspection(path: Path) -> dict:
    report = inspect(path)
    report.pop("path", None)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        source = args.source.resolve(strict=True)
        candidate = args.candidate.resolve(strict=True)
        if source == candidate:
            raise ValueError("source and candidate must be different files")
        report_path = args.report.resolve() if args.report else None
        if report_path and report_path.exists():
            raise ValueError(f"refusing to overwrite report: {report_path}")

        source_data = source.read_bytes()
        candidate_data = candidate.read_bytes()
        source_inspection = normalized_inspection(source)
        candidate_inspection = normalized_inspection(candidate)
        difference = first_difference(source_data, candidate_data)
        byte_identical = difference is None
        structural_equal = source_inspection == candidate_inspection

        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": str(source),
                "size": len(source_data),
                "sha256": digest(source_data),
                "header": source_inspection["header"],
                "blob_tags": source_inspection["blob_tags"],
                "parse_errors": source_inspection["parsed"]["errors"],
            },
            "candidate": {
                "path": str(candidate),
                "size": len(candidate_data),
                "sha256": digest(candidate_data),
                "header": candidate_inspection["header"],
                "blob_tags": candidate_inspection["blob_tags"],
                "parse_errors": candidate_inspection["parsed"]["errors"],
            },
            "byte_identical": byte_identical,
            "first_different_offset": difference,
            "structural_equal": structural_equal,
        }
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print("FH6_MODELBIN_COMPARE=" + json.dumps({
            "byte_identical": byte_identical,
            "structural_equal": structural_equal,
            "first_different_offset": difference,
            "source_sha256": report["source"]["sha256"],
            "candidate_sha256": report["candidate"]["sha256"],
            "parse_errors": len(report["source"]["parse_errors"]) + len(report["candidate"]["parse_errors"]),
            "report": str(report_path) if report_path else None,
        }, ensure_ascii=False, sort_keys=True))
        return 0 if structural_equal else 2
    except (OSError, ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
