#!/usr/bin/env python3
"""Report PMX model statistics relevant to FH6 character conversion."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


class PmxError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.encoding = "utf-16-le"

    def read(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise PmxError(f"read outside PMX at 0x{self.pos:X}, size={size}")
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        return value

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u16(self) -> int:
        return self.unpack("<H")[0]

    def i32(self) -> int:
        return self.unpack("<i")[0]

    def f32(self) -> float:
        return self.unpack("<f")[0]

    def text(self) -> str:
        size = self.i32()
        return self.read(size).decode(self.encoding, errors="replace")

    def index(self, size: int, signed: bool = True) -> int:
        formats = {
            (1, True): "<b",
            (1, False): "<B",
            (2, True): "<h",
            (2, False): "<H",
            (4, True): "<i",
            (4, False): "<I",
        }
        try:
            return self.unpack(formats[(size, signed)])[0]
        except KeyError as exc:
            raise PmxError(f"unsupported PMX index size {size}") from exc


def inspect(path: Path) -> dict[str, Any]:
    reader = Reader(path.read_bytes())
    if reader.read(4) != b"PMX ":
        raise PmxError("invalid PMX signature")
    version = reader.f32()
    globals_size = reader.u8()
    globals_data = list(reader.read(globals_size))
    if len(globals_data) < 8:
        raise PmxError("incomplete PMX globals")

    encoding, extra_uv, vertex_index_size, texture_index_size, material_index_size, bone_index_size, morph_index_size, rigid_index_size = globals_data[:8]
    reader.encoding = "utf-8" if encoding else "utf-16-le"
    names = {
        "local": reader.text(),
        "universal": reader.text(),
        "comment_local": reader.text(),
        "comment_universal": reader.text(),
    }

    vertex_count = reader.i32()
    deform_counts: Counter[str] = Counter()
    deform_names = {0: "BDEF1", 1: "BDEF2", 2: "BDEF4", 3: "SDEF", 4: "QDEF"}
    for _ in range(vertex_count):
        reader.read(12 + 12 + 8 + extra_uv * 16)
        deform = reader.u8()
        deform_counts[deform_names.get(deform, f"UNKNOWN_{deform}")] += 1
        if deform == 0:
            reader.index(bone_index_size)
        elif deform == 1:
            reader.index(bone_index_size)
            reader.index(bone_index_size)
            reader.read(4)
        elif deform in (2, 4):
            for _ in range(4):
                reader.index(bone_index_size)
            reader.read(16)
        elif deform == 3:
            reader.index(bone_index_size)
            reader.index(bone_index_size)
            reader.read(4 + 36)
        else:
            raise PmxError(f"unsupported deform type {deform}")
        reader.read(4)

    surface_index_count = reader.i32()
    reader.read(surface_index_count * vertex_index_size)

    texture_count = reader.i32()
    textures = [reader.text() for _ in range(texture_count)]

    material_count = reader.i32()
    materials = []
    for _ in range(material_count):
        local_name = reader.text()
        universal_name = reader.text()
        reader.read(16 + 12 + 4 + 12 + 1 + 16 + 4)
        texture_index = reader.index(texture_index_size)
        sphere_index = reader.index(texture_index_size)
        sphere_mode = reader.u8()
        shared_toon = reader.u8()
        if shared_toon:
            toon_index = reader.u8()
        else:
            toon_index = reader.index(texture_index_size)
        comment = reader.text()
        material_surface_count = reader.i32()
        materials.append(
            {
                "local": local_name,
                "universal": universal_name,
                "texture_index": texture_index,
                "sphere_index": sphere_index,
                "sphere_mode": sphere_mode,
                "toon_index": toon_index,
                "surface_index_count": material_surface_count,
                "comment": comment,
            }
        )

    bone_count = reader.i32()
    bones = []
    ik_bones = 0
    for index in range(bone_count):
        local_name = reader.text()
        universal_name = reader.text()
        position = list(reader.unpack("<3f"))
        parent = reader.index(bone_index_size)
        layer = reader.i32()
        flags = reader.u16()
        if flags & 0x0001:
            reader.index(bone_index_size)
        else:
            reader.read(12)
        if flags & (0x0100 | 0x0200):
            reader.index(bone_index_size)
            reader.read(4)
        if flags & 0x0400:
            reader.read(12)
        if flags & 0x0800:
            reader.read(24)
        if flags & 0x2000:
            reader.read(4)
        if flags & 0x0020:
            ik_bones += 1
            reader.index(bone_index_size)
            reader.read(4 + 4)
            link_count = reader.i32()
            for _ in range(link_count):
                reader.index(bone_index_size)
                has_limits = reader.u8()
                if has_limits:
                    reader.read(24)
        bones.append(
            {
                "index": index,
                "local": local_name,
                "universal": universal_name,
                "parent": parent,
                "layer": layer,
                "flags": flags,
                "position": position,
            }
        )

    morph_count = reader.i32()
    return {
        "path": str(path.resolve()),
        "version": version,
        "encoding": reader.encoding,
        "name": names,
        "index_sizes": {
            "vertex": vertex_index_size,
            "texture": texture_index_size,
            "material": material_index_size,
            "bone": bone_index_size,
            "morph": morph_index_size,
            "rigid_body": rigid_index_size,
        },
        "extra_uv_channels": extra_uv,
        "vertex_count": vertex_count,
        "triangle_count": surface_index_count // 3,
        "deforms": dict(deform_counts),
        "texture_count": texture_count,
        "textures": textures,
        "material_count": material_count,
        "materials": materials,
        "bone_count": bone_count,
        "ik_bone_count": ik_bones,
        "bones": bones,
        "morph_count": morph_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmx", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = inspect(args.pmx)
    except (OSError, PmxError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Model: {report['name']['local']} / {report['name']['universal']}")
        print(f"PMX: {report['version']:.1f}, encoding={report['encoding']}")
        print(f"Vertices: {report['vertex_count']:,}; triangles: {report['triangle_count']:,}")
        print("Deforms: " + ", ".join(f"{key}={value:,}" for key, value in report["deforms"].items()))
        print(f"Materials: {report['material_count']}; textures: {report['texture_count']}")
        print(f"Bones: {report['bone_count']}; IK bones: {report['ik_bone_count']}; morphs: {report['morph_count']}")
        print("First bones: " + ", ".join(bone["local"] for bone in report["bones"][:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
