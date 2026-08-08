#!/usr/bin/env python3
"""Extract .modelbin entries from an FH6 character asset ZIP."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="FH6 character ZIP path")
    parser.add_argument("output", type=Path, help="Output directory")
    args = parser.parse_args()

    try:
        with zipfile.ZipFile(args.archive) as archive:
            entries = [item for item in archive.infolist() if item.filename.lower().endswith(".modelbin")]
            if not entries:
                raise ValueError("archive contains no .modelbin entry")
            args.output.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                name = Path(entry.filename).name
                destination = args.output / name
                with archive.open(entry) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                print(f"{destination} ({destination.stat().st_size} bytes)")
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
