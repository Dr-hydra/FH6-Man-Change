#!/usr/bin/env python3
"""Safely deploy a validated FH6 runtime XML with a timestamped backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    target = args.target.resolve(strict=True)
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    data = candidate.read_bytes()
    if data.startswith(b"\xef\xbb\xbf") or b"\r\n" not in data:
        raise ValueError("Candidate XML must be UTF-8 without BOM and use CRLF")
    ET.fromstring(data)
    candidate_hash = sha256(candidate)
    target_hash = sha256(target)
    if candidate_hash == target_hash:
        raise ValueError("Candidate XML is byte-identical to target")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(target.name + f".bak-{timestamp}") if args.apply else None
    if backup and backup.exists():
        raise FileExistsError(f"Refusing to overwrite {backup}")
    if args.apply:
        shutil.copy2(target, backup)
        temporary = target.with_name(target.name + f".tmp-{timestamp}")
        try:
            shutil.copy2(candidate, temporary)
            if sha256(temporary) != candidate_hash:
                raise ValueError("Temporary XML hash mismatch")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "candidate": {"path": str(candidate), "bytes": len(data), "sha256": candidate_hash},
        "target": {"path": str(target), "bytes": target.stat().st_size, "sha256": target_hash},
        "backup": str(backup) if backup else None,
        "deployed_sha256": sha256(target) if args.apply else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_RUNTIME_MATERIALS=" + json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
