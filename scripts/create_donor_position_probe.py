#!/usr/bin/env python3
"""Create a same-size donor probe by slightly scaling quantized positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from inspect_modelbin import inspect


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=0.98)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite probe output or report")
    if not 0.0 < args.scale < 1.0:
        raise ValueError("--scale must be between zero and one")
    source_report = inspect(source)
    position_buffers = [item for item in source_report["parsed"]["vertex_buffers"] if item["format"] == 13]
    if len(position_buffers) != 1:
        raise ValueError(f"Expected one position buffer, found {len(position_buffers)}")
    position = position_buffers[0]
    if position["stride"] != 8:
        raise ValueError(f"Expected eight-byte position stride, found {position['stride']}")
    data = bytearray(source.read_bytes())
    payload_offset = int(position["payload_offset"])
    count = int(position["count"])
    changed_components = 0
    for index in range(count):
        offset = payload_offset + index * 8
        values = list(struct.unpack_from("<4h", data, offset))
        scaled = [int(round(values[axis] * args.scale)) for axis in range(3)] + [values[3]]
        changed_components += sum(left != right for left, right in zip(values[:3], scaled[:3]))
        struct.pack_into("<4h", data, offset, *scaled)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    candidate_report = inspect(output)
    if candidate_report["parsed"]["errors"]:
        raise ValueError(f"Candidate parser errors: {candidate_report['parsed']['errors']}")
    report = {
        "source": str(source),
        "output": str(output),
        "source_sha256": sha256_bytes(source.read_bytes()),
        "output_sha256": sha256_bytes(data),
        "bytes": len(data),
        "vertices": count,
        "position_scale": args.scale,
        "changed_components": changed_components,
        "parse_errors": candidate_report["parsed"]["errors"],
        "contract": "Same-size donor; only VerB0 quantized XYZ payload values changed.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_DONOR_POSITION_PROBE=" + json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
