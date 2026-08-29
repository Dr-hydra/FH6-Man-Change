#!/usr/bin/env python3
"""Write a stable SHA256 inventory for a confirmed FH6 artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Relative artifact path to exclude; may be repeated.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("Inventory output must live below the artifact root") from exc
    excluded: set[Path] = {output}
    for raw_path in args.exclude:
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Excluded path must live below the artifact root: {raw_path}") from exc
        excluded.add(candidate)

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    content_digest = hashlib.sha256()
    for record in records:
        content_digest.update(
            f"{record['path']}\0{record['sha256']}\n".encode("utf-8")
        )
    report = {
        "schema_version": 1,
        "created_local": datetime.now(timezone.utc).astimezone().isoformat(),
        "root": str(root),
        "excluded": sorted(path.relative_to(root).as_posix() for path in excluded),
        "file_count": len(records),
        "content_sha256": content_digest.hexdigest(),
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "FH6_ARTIFACT_INVENTORY="
        + json.dumps(
            {
                "output": str(output),
                "files": len(records),
                "content_sha256": report["content_sha256"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
