"""Shared low-level MatI/MTPR parameter codecs.

This module contains format helpers only. It deliberately has no character,
clothing, Swatch, or material-plan constants; those belong to a Mod project.
"""

from __future__ import annotations

import struct
import zlib


def decode_7bit(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(5):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("Invalid 7-bit string length")


def encode_7bit(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def parameter_end(data: bytes, offset: int) -> tuple[int, int, int, int]:
    start = offset
    major, minor = data[offset], data[offset + 1]
    name_hash = struct.unpack_from("<I", data, offset + 2)[0]
    offset += 6
    if major > 3 or (major == 3 and minor >= 1):
        has_extra = data[offset]
        offset += 1
        if has_extra:
            offset += 4
    parameter_type = data[offset]
    offset += 1
    if major >= 3:
        offset += 16
    value_offset = offset

    if parameter_type in (0, 1, 5, 9):
        offset += 16
    elif parameter_type in (2, 3, 4):
        offset += 4
    elif parameter_type == 6:
        length, offset = decode_7bit(data, offset)
        offset += length
        if major >= 2:
            offset += 4
    elif parameter_type == 7:
        offset += 8
        if major >= 1 and minor >= 1:
            offset += 4
    elif parameter_type == 8:
        count = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + count * 16
    elif parameter_type == 11:
        offset += 8
        if major == 1:
            offset += 8
    else:
        raise ValueError(f"Unsupported MTPR parameter type {parameter_type} at 0x{start:X}")
    if offset > len(data):
        raise ValueError("MTPR parameter extends beyond its payload")
    return offset, name_hash, parameter_type, value_offset


def material_id(blob) -> int:
    metadata = next((entry for entry in blob.metadata if entry.tag == "Id  "), None)
    if metadata is None or len(metadata.value) != 4:
        raise ValueError(f"MatI blob {blob.index} has no four-byte material ID")
    return struct.unpack("<I", metadata.value)[0]


def material_name(blob) -> str:
    metadata = next((entry for entry in blob.metadata if entry.tag == "Name"), None)
    return metadata.value.decode("utf-8") if metadata else f"MatI_{blob.index}"


def diffuse_parameter_hash(path: str) -> int:
    return zlib.crc32(path.lower().encode("utf-8")) & 0xFFFFFFFF
