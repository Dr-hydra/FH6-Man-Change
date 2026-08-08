#!/usr/bin/env python3
"""Lossless parser/writer for the outer ForzaTech ``Grub`` bundle.

This module deliberately treats blob payloads as opaque bytes.  It rebuilds the
bundle header, blob table, metadata entries, metadata values, and blob payloads
from parsed fields while preserving otherwise-unclaimed alignment/padding bytes.
That makes it suitable as the first, no-edit round-trip gate before any FH6
geometry buffer is patched.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


class BundleError(ValueError):
    """Raised when a bundle is malformed or cannot be rebuilt losslessly."""


HEADER = struct.Struct("<4sBBHIII")
BLOB_DESCRIPTOR = struct.Struct("<4sBBHIIII")
METADATA_DESCRIPTOR = struct.Struct("<4sHH")


def decode_fourcc(raw: bytes) -> str:
    if len(raw) != 4:
        raise BundleError(f"FourCC must be four bytes, got {len(raw)}")
    try:
        return raw[::-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise BundleError(f"invalid non-ASCII FourCC {raw!r}") from exc


def encode_fourcc(value: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise BundleError(f"invalid non-ASCII FourCC {value!r}") from exc
    if len(encoded) != 4:
        raise BundleError(f"FourCC must be four ASCII bytes, got {value!r}")
    return encoded[::-1]


def require_range(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise BundleError(
            f"{label} outside file: offset=0x{offset:X}, size=0x{size:X}, "
            f"file=0x{len(data):X}"
        )


@dataclass(frozen=True)
class MetadataEntry:
    index: int
    tag: str
    version: int
    entry_offset: int
    value_offset: int
    value: bytes

    @property
    def size(self) -> int:
        return len(self.value)


@dataclass(frozen=True)
class BundleBlob:
    index: int
    tag: str
    version: tuple[int, int]
    metadata_offset: int
    data_offset: int
    trailing_size: int
    metadata: tuple[MetadataEntry, ...]
    data: bytes

    @property
    def data_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class GrubBundle:
    version: tuple[int, int]
    legacy_blob_count: int
    data_offset: int
    declared_size: int
    blobs: tuple[BundleBlob, ...]
    original: bytes

    @property
    def blob_tags(self) -> dict[str, int]:
        return dict(sorted(Counter(blob.tag for blob in self.blobs).items()))

    def rebuild_lossless(self) -> bytes:
        """Rebuild parsed structures and preserve only unclaimed gap bytes raw."""

        output = bytearray(len(self.original))
        claimed = bytearray(len(self.original))

        def write(offset: int, payload: bytes, label: str) -> None:
            require_range(self.original, offset, len(payload), label)
            for position in range(offset, offset + len(payload)):
                if claimed[position]:
                    raise BundleError(f"overlapping rebuilt regions at 0x{position:X}: {label}")
            output[offset : offset + len(payload)] = payload
            claimed[offset : offset + len(payload)] = b"\x01" * len(payload)

        header_blob_count = len(self.blobs) if self.version >= (1, 1) else 0
        header = HEADER.pack(
            encode_fourcc("Grub"),
            self.version[0],
            self.version[1],
            self.legacy_blob_count,
            self.data_offset,
            self.declared_size,
            header_blob_count,
        )
        write(0, header, "bundle header")

        descriptor_offset = HEADER.size
        for blob in self.blobs:
            descriptor = BLOB_DESCRIPTOR.pack(
                encode_fourcc(blob.tag),
                blob.version[0],
                blob.version[1],
                len(blob.metadata),
                blob.metadata_offset,
                blob.data_offset,
                blob.data_size,
                blob.trailing_size,
            )
            write(descriptor_offset + blob.index * BLOB_DESCRIPTOR.size, descriptor, f"blob {blob.index} descriptor")

            for metadata in blob.metadata:
                if metadata.version < 0 or metadata.version > 0xF:
                    raise BundleError(f"metadata version does not fit four bits: {metadata.version}")
                if metadata.size > 0xFFF:
                    raise BundleError(f"metadata value too large: {metadata.size}")
                relative_offset = metadata.value_offset - metadata.entry_offset
                if relative_offset < 0 or relative_offset > 0xFFFF:
                    raise BundleError(f"metadata relative offset out of range: {relative_offset}")
                version_and_size = metadata.version | (metadata.size << 4)
                entry = METADATA_DESCRIPTOR.pack(
                    encode_fourcc(metadata.tag), version_and_size, relative_offset
                )
                write(metadata.entry_offset, entry, f"blob {blob.index} metadata {metadata.index} entry")
                write(metadata.value_offset, metadata.value, f"blob {blob.index} metadata {metadata.index} value")

            write(blob.data_offset, blob.data, f"blob {blob.index} {blob.tag} data")

        # Alignment bytes and presently unknown container fields remain byte-exact,
        # but every recognized structural/data range above was emitted from parsed fields.
        for index, is_claimed in enumerate(claimed):
            if not is_claimed:
                output[index] = self.original[index]
        return bytes(output)


@dataclass(frozen=True)
class MetadataSpec:
    """Offset-independent metadata used when repacking a bundle prefix."""

    tag: str
    version: int
    value: bytes


@dataclass(frozen=True)
class BlobSpec:
    """Offset-independent blob used when changing a bundle's blob sequence."""

    tag: str
    version: tuple[int, int]
    metadata: tuple[MetadataSpec, ...]
    data: bytes
    trailing_size: int | None = None


def blob_spec(blob: BundleBlob) -> BlobSpec:
    """Copy a parsed blob into an offset-independent representation."""

    return BlobSpec(
        tag=blob.tag,
        version=blob.version,
        metadata=tuple(
            MetadataSpec(tag=item.tag, version=item.version, value=item.value)
            for item in blob.metadata
        ),
        data=blob.data,
        trailing_size=blob.trailing_size,
    )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def rebuild_with_blob_sequence(
    bundle: GrubBundle,
    blobs: Sequence[BlobSpec],
    *,
    data_offset_alignment: int = 16,
    payload_alignment: int = 4,
) -> bytes:
    """Repack a bundle after adding, removing, or reordering blobs.

    Unlike :func:`rebuild_with_blob_data`, this function reconstructs the
    descriptor and metadata prefix because changing the blob count changes the
    descriptor-table size. Blob payloads and metadata values remain opaque.
    """

    for name, alignment in (
        ("data_offset_alignment", data_offset_alignment),
        ("payload_alignment", payload_alignment),
    ):
        if alignment <= 0 or alignment & (alignment - 1):
            raise BundleError(f"{name} must be a positive power of two, got {alignment}")
    if bundle.version < (1, 1) and len(blobs) > 0xFFFF:
        raise BundleError("Legacy bundle blob count exceeds uint16")

    descriptor_end = HEADER.size + len(blobs) * BLOB_DESCRIPTOR.size
    metadata_records: list[
        tuple[int, BlobSpec, list[tuple[int, int, MetadataSpec]]]
    ] = []
    cursor = descriptor_end
    for spec in blobs:
        metadata_offset = cursor
        cursor += len(spec.metadata) * METADATA_DESCRIPTOR.size
        entries: list[tuple[int, int, MetadataSpec]] = []
        for index, metadata in enumerate(spec.metadata):
            if metadata.version < 0 or metadata.version > 0xF:
                raise BundleError(
                    f"metadata version does not fit four bits: {metadata.version}"
                )
            if len(metadata.value) > 0xFFF:
                raise BundleError(f"metadata value too large: {len(metadata.value)}")
            entry_offset = metadata_offset + index * METADATA_DESCRIPTOR.size
            value_offset = cursor
            relative_offset = value_offset - entry_offset
            if relative_offset > 0xFFFF:
                raise BundleError(
                    f"metadata relative offset out of range: {relative_offset}"
                )
            entries.append((entry_offset, value_offset, metadata))
            cursor += len(metadata.value)
        metadata_records.append((metadata_offset, spec, entries))

    data_offset = _align(cursor, data_offset_alignment)
    output = bytearray(data_offset)
    modern_count = len(blobs) if bundle.version >= (1, 1) else 0
    legacy_count = len(blobs) if bundle.version < (1, 1) else bundle.legacy_blob_count
    output[: HEADER.size] = HEADER.pack(
        encode_fourcc("Grub"),
        bundle.version[0],
        bundle.version[1],
        legacy_count,
        data_offset,
        0,
        modern_count,
    )

    payload_ranges: list[tuple[int, int, int]] = []
    for spec in blobs:
        padding = (-len(output)) & (payload_alignment - 1)
        if padding:
            output.extend(b"\x00" * padding)
        payload_offset = len(output)
        output.extend(spec.data)
        trailing_size = len(spec.data) if spec.trailing_size is None else spec.trailing_size
        payload_ranges.append((payload_offset, len(spec.data), trailing_size))
    final_padding = (-len(output)) & (payload_alignment - 1)
    if final_padding:
        output.extend(b"\x00" * final_padding)

    declared_size = len(output)
    output[: HEADER.size] = HEADER.pack(
        encode_fourcc("Grub"),
        bundle.version[0],
        bundle.version[1],
        legacy_count,
        data_offset,
        declared_size,
        modern_count,
    )
    for index, ((metadata_offset, spec, entries), payload_range) in enumerate(
        zip(metadata_records, payload_ranges)
    ):
        payload_offset, payload_size, trailing_size = payload_range
        descriptor_offset = HEADER.size + index * BLOB_DESCRIPTOR.size
        output[
            descriptor_offset : descriptor_offset + BLOB_DESCRIPTOR.size
        ] = BLOB_DESCRIPTOR.pack(
            encode_fourcc(spec.tag),
            spec.version[0],
            spec.version[1],
            len(spec.metadata),
            metadata_offset,
            payload_offset,
            payload_size,
            trailing_size,
        )
        for entry_offset, value_offset, metadata in entries:
            version_and_size = metadata.version | (len(metadata.value) << 4)
            relative_offset = value_offset - entry_offset
            output[
                entry_offset : entry_offset + METADATA_DESCRIPTOR.size
            ] = METADATA_DESCRIPTOR.pack(
                encode_fourcc(metadata.tag), version_and_size, relative_offset
            )
            output[value_offset : value_offset + len(metadata.value)] = metadata.value

    return bytes(output)


def parse_bundle(data: bytes) -> GrubBundle:
    require_range(data, 0, HEADER.size, "bundle header")
    raw_tag, major, minor, legacy_count, data_offset, declared_size, modern_count = HEADER.unpack_from(data)
    tag = decode_fourcc(raw_tag)
    if tag != "Grub":
        raise BundleError(f"invalid bundle tag {tag!r}; expected 'Grub'")
    version = (major, minor)
    blob_count = modern_count if version >= (1, 1) else legacy_count
    require_range(data, HEADER.size, blob_count * BLOB_DESCRIPTOR.size, "blob descriptor table")
    if declared_size != len(data):
        raise BundleError(f"declared size {declared_size} does not match actual size {len(data)}")

    blobs: list[BundleBlob] = []
    for index in range(blob_count):
        descriptor_offset = HEADER.size + index * BLOB_DESCRIPTOR.size
        (
            raw_blob_tag,
            blob_major,
            blob_minor,
            metadata_count,
            metadata_offset,
            blob_data_offset,
            blob_data_size,
            trailing_size,
        ) = BLOB_DESCRIPTOR.unpack_from(data, descriptor_offset)
        blob_tag = decode_fourcc(raw_blob_tag)
        require_range(data, metadata_offset, metadata_count * METADATA_DESCRIPTOR.size, f"blob {index} metadata table")
        require_range(data, blob_data_offset, blob_data_size, f"blob {index} data")

        metadata_entries: list[MetadataEntry] = []
        for metadata_index in range(metadata_count):
            entry_offset = metadata_offset + metadata_index * METADATA_DESCRIPTOR.size
            raw_metadata_tag, version_and_size, relative_offset = METADATA_DESCRIPTOR.unpack_from(data, entry_offset)
            metadata_tag = decode_fourcc(raw_metadata_tag)
            metadata_version = version_and_size & 0xF
            metadata_size = version_and_size >> 4
            value_offset = entry_offset + relative_offset
            require_range(data, value_offset, metadata_size, f"blob {index} metadata {metadata_index} value")
            metadata_entries.append(
                MetadataEntry(
                    index=metadata_index,
                    tag=metadata_tag,
                    version=metadata_version,
                    entry_offset=entry_offset,
                    value_offset=value_offset,
                    value=data[value_offset : value_offset + metadata_size],
                )
            )

        blobs.append(
            BundleBlob(
                index=index,
                tag=blob_tag,
                version=(blob_major, blob_minor),
                metadata_offset=metadata_offset,
                data_offset=blob_data_offset,
                trailing_size=trailing_size,
                metadata=tuple(metadata_entries),
                data=data[blob_data_offset : blob_data_offset + blob_data_size],
            )
        )

    return GrubBundle(
        version=version,
        legacy_blob_count=legacy_count,
        data_offset=data_offset,
        declared_size=declared_size,
        blobs=tuple(blobs),
        original=data,
    )


def first_difference(left: bytes, right: bytes) -> int | None:
    for index, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def iter_blob_ranges(bundle: GrubBundle) -> Iterable[tuple[int, int, str]]:
    """Yield blob payload ranges for diagnostics and future patch planning."""

    for blob in bundle.blobs:
        yield blob.data_offset, blob.data_offset + blob.data_size, f"{blob.index}:{blob.tag}"


def rebuild_with_blob_data(
    bundle: GrubBundle,
    replacements: dict[int, bytes],
    *,
    alignment: int = 4,
) -> bytes:
    """Repack a bundle after replacing selected blob payloads.

    The header/descriptor/metadata prefix remains byte-exact except for the
    required declared-size, blob-offset, blob-size, and trailing-size fields.
    Blob payloads are repacked in their original order at the observed
    four-byte alignment.  This is deliberately narrower than a generic Grub
    authoring API and is suitable for donor-template patches.
    """

    if alignment <= 0 or alignment & (alignment - 1):
        raise BundleError(f"alignment must be a positive power of two, got {alignment}")
    unknown = sorted(set(replacements) - {blob.index for blob in bundle.blobs})
    if unknown:
        raise BundleError(f"replacement blob indices do not exist: {unknown}")

    output = bytearray(bundle.original[: bundle.data_offset])
    if len(output) != bundle.data_offset:
        raise BundleError("bundle data_offset exceeds original file")

    new_ranges: dict[int, tuple[int, int, int]] = {}
    for blob in bundle.blobs:
        padding = (-len(output)) & (alignment - 1)
        if padding:
            output.extend(b"\x00" * padding)
        payload = replacements.get(blob.index, blob.data)
        offset = len(output)
        output.extend(payload)
        trailing_size = len(payload) if blob.index in replacements else blob.trailing_size
        new_ranges[blob.index] = (offset, len(payload), trailing_size)

    final_padding = (-len(output)) & (alignment - 1)
    if final_padding:
        output.extend(b"\x00" * final_padding)

    declared_size = len(output)
    raw_tag, major, minor, legacy_count, data_offset, _old_size, modern_count = HEADER.unpack_from(output)
    output[: HEADER.size] = HEADER.pack(
        raw_tag,
        major,
        minor,
        legacy_count,
        data_offset,
        declared_size,
        modern_count,
    )
    for blob in bundle.blobs:
        descriptor_offset = HEADER.size + blob.index * BLOB_DESCRIPTOR.size
        (
            raw_blob_tag,
            blob_major,
            blob_minor,
            metadata_count,
            metadata_offset,
            _old_data_offset,
            _old_data_size,
            _old_trailing_size,
        ) = BLOB_DESCRIPTOR.unpack_from(output, descriptor_offset)
        new_data_offset, new_data_size, new_trailing_size = new_ranges[blob.index]
        output[descriptor_offset : descriptor_offset + BLOB_DESCRIPTOR.size] = BLOB_DESCRIPTOR.pack(
            raw_blob_tag,
            blob_major,
            blob_minor,
            metadata_count,
            metadata_offset,
            new_data_offset,
            new_data_size,
            new_trailing_size,
        )

    return bytes(output)
