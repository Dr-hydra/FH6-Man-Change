#!/usr/bin/env python3
"""Inspect ForzaTech .modelbin bundle metadata without Blender dependencies."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ParseError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes, offset: int = 0, limit: int | None = None):
        self.data = data
        self.pos = offset
        self.limit = len(data) if limit is None else limit

    def require(self, size: int) -> None:
        if size < 0 or self.pos + size > self.limit:
            raise ParseError(
                f"read outside buffer at 0x{self.pos:X}: "
                f"need {size} bytes, limit is 0x{self.limit:X}"
            )

    def read(self, size: int) -> bytes:
        self.require(size)
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        return value

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u16(self) -> int:
        return self.unpack("<H")[0]

    def s16(self) -> int:
        return self.unpack("<h")[0]

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def s32(self) -> int:
        return self.unpack("<i")[0]

    def f32(self) -> float:
        return self.unpack("<f")[0]

    def string_u32(self) -> str:
        size = self.u32()
        return self.read(size).decode("utf-8", errors="replace")


def fourcc(raw: bytes) -> str:
    return raw[::-1].decode("ascii", errors="replace")


def version_at_least(version: tuple[int, int], target: tuple[int, int]) -> bool:
    return version >= target


@dataclass
class Metadata:
    tag: str
    version: int
    offset: int
    size: int
    value: Any


@dataclass
class Blob:
    index: int
    tag: str
    version: tuple[int, int]
    metadata_offset: int
    data_offset: int
    data_size: int
    trailing_size: int
    metadata: dict[str, Metadata]

    @property
    def name(self) -> str | None:
        item = self.metadata.get("Name")
        return item.value if item else None

    @property
    def identifier(self) -> int | None:
        item = self.metadata.get("Id  ")
        return item.value if item else None


def parse_metadata(data: bytes, entry_offset: int) -> Metadata:
    reader = Reader(data, entry_offset, entry_offset + 8)
    tag = fourcc(reader.read(4))
    version_and_size = reader.u16()
    relative_offset = reader.u16()
    version = version_and_size & 0xF
    size = version_and_size >> 4
    value_offset = entry_offset + relative_offset
    value_reader = Reader(data, value_offset, value_offset + size)
    raw = value_reader.read(size)

    if tag == "Name":
        value: Any = raw.decode("utf-8", errors="replace")
    elif tag == "Id  " and len(raw) == 4:
        value = struct.unpack("<i", raw)[0]
    elif tag == "BBox" and len(raw) == 24:
        value = list(struct.unpack("<6f", raw))
    else:
        value = raw.hex()

    return Metadata(tag, version, value_offset, size, value)


def parse_bundle(data: bytes) -> tuple[dict[str, Any], list[Blob]]:
    reader = Reader(data)
    tag = fourcc(reader.read(4))
    if tag != "Grub":
        raise ParseError(f"invalid bundle tag {tag!r}; expected 'Grub'")

    version = (reader.u8(), reader.u8())
    legacy_blob_count = reader.u16()
    data_offset = reader.u32()
    declared_size = reader.u32()
    blob_count = reader.u32() if version_at_least(version, (1, 1)) else legacy_blob_count

    blobs: list[Blob] = []
    for index in range(blob_count):
        tag = fourcc(reader.read(4))
        blob_version = (reader.u8(), reader.u8())
        metadata_count = reader.u16()
        metadata_offset = reader.u32()
        blob_data_offset = reader.u32()
        blob_data_size = reader.u32()
        trailing_size = reader.u32()

        if blob_data_offset + blob_data_size > len(data):
            raise ParseError(
                f"blob {index} {tag} exceeds file: "
                f"0x{blob_data_offset:X} + 0x{blob_data_size:X}"
            )

        metadata: dict[str, Metadata] = {}
        for metadata_index in range(metadata_count):
            entry_offset = metadata_offset + metadata_index * 8
            item = parse_metadata(data, entry_offset)
            metadata[item.tag] = item

        blobs.append(
            Blob(
                index=index,
                tag=tag,
                version=blob_version,
                metadata_offset=metadata_offset,
                data_offset=blob_data_offset,
                data_size=blob_data_size,
                trailing_size=trailing_size,
                metadata=metadata,
            )
        )

    header = {
        "tag": "Grub",
        "version": f"{version[0]}.{version[1]}",
        "data_offset": data_offset,
        "declared_size": declared_size,
        "actual_size": len(data),
        "blob_count": blob_count,
    }
    return header, blobs


def blob_reader(data: bytes, blob: Blob) -> Reader:
    return Reader(data, blob.data_offset, blob.data_offset + blob.data_size)


def parse_model(data: bytes, blob: Blob) -> dict[str, Any]:
    reader = blob_reader(data, blob)
    result = {
        "meshes": reader.s16(),
        "buffers": reader.s16(),
        "vertex_layouts": reader.s16(),
        "materials": reader.s16(),
        "unknown_u32": reader.u32(),
        "lod_flags": reader.u16(),
    }
    if version_at_least(blob.version, (1, 2)) and reader.pos < reader.limit:
        result["decompress_flags"] = reader.u8()
    return result


def parse_skeleton(data: bytes, blob: Blob) -> dict[str, Any]:
    reader = blob_reader(data, blob)
    bone_count = reader.u16()
    bones = []
    for index in range(bone_count):
        name = reader.string_u32()
        parent = reader.s16()
        first_child = reader.s16()
        next_sibling = reader.s16()
        matrix = [reader.f32() for _ in range(16)]
        bones.append(
            {
                "index": index,
                "name": name,
                "parent": parent,
                "first_child": first_child,
                "next_sibling": next_sibling,
                "matrix": matrix,
            }
        )
    return {
        "bone_count": bone_count,
        "bytes_consumed": reader.pos - blob.data_offset,
        "bytes_remaining": reader.limit - reader.pos,
        "bones": bones,
    }


def parse_vertex_layout(data: bytes, blob: Blob) -> dict[str, Any]:
    reader = blob_reader(data, blob)
    name_count = reader.u16()
    names = [reader.string_u32() for _ in range(name_count)]
    element_count = reader.u16()
    elements = []
    for _ in range(element_count):
        name_index = reader.u16()
        semantic_index = reader.u16()
        input_slot = reader.u16()
        unknown_u16 = reader.u16()
        dxgi_format = reader.u32()
        unknown_a = reader.u32()
        unknown_b = reader.u32()
        semantic = names[name_index] if name_index < len(names) else f"<name:{name_index}>"
        elements.append(
            {
                "semantic": semantic,
                "semantic_index": semantic_index,
                "input_slot": input_slot,
                "format": dxgi_format,
                "unknown_u16": unknown_u16,
                "unknown_u32": [unknown_a, unknown_b],
            }
        )
    return {
        "names": names,
        "element_count": element_count,
        "elements": elements,
        "bytes_remaining": reader.limit - reader.pos,
    }


def parse_model_buffer(data: bytes, blob: Blob) -> dict[str, Any]:
    reader = blob_reader(data, blob)
    count = reader.u32()
    byte_size = reader.u32()
    stride = reader.u16()
    flags = list(reader.read(2))
    result = {
        "count": count,
        "byte_size": byte_size,
        "stride": stride,
        "flags": flags,
    }
    if version_at_least(blob.version, (1, 0)):
        result["format"] = reader.u32()
    result["payload_offset"] = reader.pos
    result["payload_fits"] = reader.pos + byte_size <= reader.limit
    result["count_times_stride"] = count * stride
    return result


def parse_skin_buffer(data: bytes, blob: Blob) -> dict[str, Any]:
    result = parse_model_buffer(data, blob)
    stride = result["stride"]
    count = result["count"]
    payload_offset = result["payload_offset"]
    byte_size = result["byte_size"]
    if result.get("format") != 34 or stride == 0 or stride % 4 != 0:
        return result
    if payload_offset + byte_size > blob.data_offset + blob.data_size:
        return result

    influences = stride // 4
    weight_sums = []
    bone_indices = []
    samples = []
    for vertex_index in range(count):
        vertex_offset = payload_offset + vertex_index * stride
        pairs = []
        for influence_index in range(influences):
            weight, bone_index = struct.unpack_from(
                "<ee", data, vertex_offset + influence_index * 4
            )
            pairs.append([weight, bone_index])
            if weight > 0:
                bone_indices.append(bone_index)
        weight_sums.append(sum(pair[0] for pair in pairs))
        if vertex_index < 3:
            samples.append(pairs)

    result["encoding"] = "R16G16_FLOAT(weight,bone_index)"
    result["influences_per_vertex"] = influences
    result["weight_sum_min"] = min(weight_sums, default=None)
    result["weight_sum_max"] = max(weight_sums, default=None)
    result["bone_index_min"] = min(bone_indices, default=None)
    result["bone_index_max"] = max(bone_indices, default=None)
    result["samples"] = samples
    return result


def parse_mesh(data: bytes, blob: Blob) -> dict[str, Any]:
    """Parse the mesh descriptor fields used by the public Forza importer."""
    reader = blob_reader(data, blob)
    if version_at_least(blob.version, (1, 13)):
        reader.read(4)
    material_id = reader.s16()
    if version_at_least(blob.version, (1, 9)):
        material_id = reader.s16()
        reader.read(4)
    bone_index = reader.s16()
    lod_flags = reader.u16()
    reader.read(2)
    render_pass = reader.u16()
    reader.read(1)

    skinning_elements = None
    morph_weights = None
    if version_at_least(blob.version, (1, 2)):
        skinning_elements = reader.u8()
        morph_weights = reader.u32() if version_at_least(blob.version, (1, 10)) else reader.u8()
    if version_at_least(blob.version, (1, 3)):
        reader.read(1)
    reader.read(3)
    index_buffer_id = reader.s32()
    reader.read(4)
    start_index = reader.s32()
    base_vertex = reader.s32()
    index_count = reader.u32()
    reader.read(4)
    extended_unknown_u32 = None
    extended_array_count = None
    extended_array_summary = None
    if version_at_least(blob.version, (1, 6)):
        extended_unknown_u32 = [reader.u32(), reader.u32()]
        if version_at_least(blob.version, (1, 11)):
            extended_array_count = reader.u32()
            extended_values = [reader.u32() for _ in range(extended_array_count)]
            extended_array_summary = {
                "minimum": min(extended_values, default=None),
                "maximum": max(extended_values, default=None),
                "unique": len(set(extended_values)),
                "first": extended_values[:8],
                "middle": extended_values[max(0, extended_array_count // 2 - 4) : extended_array_count // 2 + 4],
                "last": extended_values[-8:] if extended_values else [],
            }

    vertex_layout_id = reader.u32()
    buffer_count = reader.u32()
    buffers = []
    for _ in range(buffer_count):
        buffer_id = reader.s32()
        input_slot = reader.s32()
        stride = reader.s32()
        offset = reader.s32()
        if version_at_least(blob.version, (1, 12)):
            reader.read(4)
        buffers.append(
            {
                "id": buffer_id,
                "input_slot": input_slot,
                "stride": stride,
                "offset": offset,
            }
        )

    morph_buffer_id = None
    skinning_data_buffer_id = None
    if version_at_least(blob.version, (1, 4)):
        morph_buffer_id = reader.s32()
        skinning_data_buffer_id = reader.s32()
    constant_count = reader.u32()
    if constant_count:
        reader.read(constant_count * 4)
    if version_at_least(blob.version, (1, 1)):
        reader.read(4)

    uv_transforms = None
    if version_at_least(blob.version, (1, 5)):
        uv_transforms = [[reader.f32() for _ in range(4)] for _ in range(5)]
    scale = None
    translate = None
    if version_at_least(blob.version, (1, 8)):
        scale = [reader.f32() for _ in range(4)]
        translate = [reader.f32() for _ in range(4)]

    return {
        "name": blob.name,
        "blob_index": blob.index,
        "blob_version": f"{blob.version[0]}.{blob.version[1]}",
        "material_id": material_id,
        "bone_index": bone_index,
        "lod_flags": lod_flags,
        "render_pass": render_pass,
        "skinning_elements": skinning_elements,
        "morph_weights": morph_weights,
        "index_buffer_id": index_buffer_id,
        "start_index": start_index,
        "base_vertex": base_vertex,
        "index_count": index_count,
        "extended_unknown_u32": extended_unknown_u32,
        "extended_array_count": extended_array_count,
        "extended_array_summary": extended_array_summary,
        "vertex_layout_id": vertex_layout_id,
        "vertex_buffers": buffers,
        "morph_buffer_id": morph_buffer_id,
        "skinning_data_buffer_id": skinning_data_buffer_id,
        "uv_transforms": uv_transforms,
        "scale": scale,
        "translate": translate,
        "bytes_remaining": reader.limit - reader.pos,
    }


def parse_known_blobs(data: bytes, blobs: list[Blob]) -> dict[str, Any]:
    known: dict[str, Any] = {
        "model": [],
        "skeleton": [],
        "meshes": [],
        "vertex_layouts": [],
        "index_buffers": [],
        "vertex_buffers": [],
        "morph_buffers": [],
        "skin_buffers": [],
        "errors": [],
    }
    handlers = {
        "Modl": ("model", parse_model),
        "Skel": ("skeleton", parse_skeleton),
        "Mesh": ("meshes", parse_mesh),
        "VLay": ("vertex_layouts", parse_vertex_layout),
        "IndB": ("index_buffers", parse_model_buffer),
        "VerB": ("vertex_buffers", parse_model_buffer),
        "MBuf": ("morph_buffers", parse_model_buffer),
        "Skin": ("skin_buffers", parse_skin_buffer),
    }
    for blob in blobs:
        handler_info = handlers.get(blob.tag)
        if not handler_info:
            continue
        target, handler = handler_info
        try:
            parsed = handler(data, blob)
            parsed["blob_index"] = blob.index
            parsed["blob_version"] = f"{blob.version[0]}.{blob.version[1]}"
            parsed["id"] = blob.identifier
            parsed["name"] = blob.name
            known[target].append(parsed)
        except (ParseError, IndexError, struct.error, UnicodeError) as exc:
            known["errors"].append(
                {"blob_index": blob.index, "tag": blob.tag, "error": str(exc)}
            )
    return known


def inspect(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    header, blobs = parse_bundle(data)
    parsed = parse_known_blobs(data, blobs)
    return {
        "path": str(path.resolve()),
        "header": header,
        "blob_tags": dict(sorted(Counter(blob.tag for blob in blobs).items())),
        "blobs": [
            {
                "index": blob.index,
                "tag": blob.tag,
                "version": f"{blob.version[0]}.{blob.version[1]}",
                "data_offset": blob.data_offset,
                "data_size": blob.data_size,
                "trailing_size": blob.trailing_size,
                "id": blob.identifier,
                "name": blob.name,
                "metadata": {
                    tag: {
                        "version": item.version,
                        "offset": item.offset,
                        "size": item.size,
                        "value": item.value,
                    }
                    for tag, item in blob.metadata.items()
                },
            }
            for blob in blobs
        ],
        "parsed": parsed,
    }


def print_text(report: dict[str, Any], show_bones: bool) -> None:
    header = report["header"]
    print(f"File: {report['path']}")
    print(
        f"Bundle: {header['tag']} {header['version']}, "
        f"{header['blob_count']} blobs, "
        f"declared/actual bytes {header['declared_size']}/{header['actual_size']}"
    )
    print("Blob tags: " + ", ".join(f"{tag}={count}" for tag, count in report["blob_tags"].items()))

    for model in report["parsed"]["model"]:
        print(
            "Model: "
            f"meshes={model['meshes']}, buffers={model['buffers']}, "
            f"layouts={model['vertex_layouts']}, materials={model['materials']}, "
            f"lod_flags=0x{model['lod_flags']:X}, version={model['blob_version']}"
        )

    for skeleton in report["parsed"]["skeleton"]:
        print(
            f"Skeleton: bones={skeleton['bone_count']}, "
            f"remaining={skeleton['bytes_remaining']} bytes, "
            f"version={skeleton['blob_version']}"
        )
        if show_bones:
            for bone in skeleton["bones"]:
                print(
                    f"  [{bone['index']:3}] parent={bone['parent']:3} "
                    f"child={bone['first_child']:3} next={bone['next_sibling']:3} "
                    f"{bone['name']}"
                )

    for layout_index, layout in enumerate(report["parsed"]["vertex_layouts"]):
        semantics = ", ".join(
            f"{item['semantic']}{item['semantic_index']}@slot{item['input_slot']}:fmt{item['format']}"
            for item in layout["elements"]
        )
        print(f"Vertex layout {layout_index} (blob {layout['blob_index']}): {semantics}")

    for mesh in report["parsed"]["meshes"]:
        print(
            f"Mesh {mesh['name']!r}: material={mesh['material_id']}, "
            f"lod=0x{mesh['lod_flags']:X}, pass=0x{mesh['render_pass']:X}, "
            f"indices={mesh['index_count']}@{mesh['start_index']}, "
            f"layout={mesh['vertex_layout_id']}, "
            f"skin={mesh['skinning_data_buffer_id']}, "
            f"remaining={mesh['bytes_remaining']} bytes"
        )

    for key, label in (
        ("index_buffers", "Index buffer"),
        ("vertex_buffers", "Vertex buffer"),
        ("morph_buffers", "Morph buffer"),
        ("skin_buffers", "Skin buffer"),
    ):
        for item in report["parsed"][key]:
            print(
                f"{label} blob {item['blob_index']}: id={item['id']}, "
                f"count={item['count']}, bytes={item['byte_size']}, "
                f"stride={item['stride']}, format={item.get('format')}, "
                f"payload_fits={item['payload_fits']}"
            )
            if key == "skin_buffers" and "encoding" in item:
                print(
                    f"  {item['encoding']}, influences={item['influences_per_vertex']}, "
                    f"weight_sum={item['weight_sum_min']:.6f}..{item['weight_sum_max']:.6f}, "
                    f"bone_index={item['bone_index_min']:.0f}..{item['bone_index_max']:.0f}"
                )

    for error in report["parsed"]["errors"]:
        print(
            f"Warning: failed to parse blob {error['blob_index']} "
            f"{error['tag']}: {error['error']}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modelbin", type=Path, help="Path to an extracted .modelbin")
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON")
    parser.add_argument("--bones", action="store_true", help="Print every skeleton bone")
    parser.add_argument("--report", type=Path, help="Write the complete report as JSON")
    args = parser.parse_args()

    try:
        report_path = args.report.resolve() if args.report else None
        if report_path and report_path.exists():
            raise OSError(f"refusing to overwrite report: {report_path}")
        report = inspect(args.modelbin)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print_text(report, args.bones)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
