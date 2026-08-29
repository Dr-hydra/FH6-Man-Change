#!/usr/bin/env python3
"""Apply small manifest-declared XML element patches without reserializing files."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any


ATTRIBUTE_RE = re.compile(
    r"([A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"
)


def _attributes(start_tag: str) -> dict[str, str]:
    return {
        name: double if double is not None else single
        for name, double, single in ATTRIBUTE_RE.findall(start_tag)
    }


def _matches(start_tag: str, selector: dict[str, str]) -> bool:
    attributes = _attributes(start_tag)
    return all(attributes.get(name) == value for name, value in selector.items())


def _element_spans(
    text: str, tag: str, selector: dict[str, str] | None = None
) -> list[tuple[int, int]]:
    selector = selector or {}
    token_re = re.compile(rf"</?{re.escape(tag)}\b[^>]*>", re.IGNORECASE)
    tokens = list(token_re.finditer(text))
    result: list[tuple[int, int]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        value = token.group(0)
        if value.startswith("</"):
            index += 1
            continue
        if not _matches(value, selector):
            index += 1
            continue
        if value.rstrip().endswith("/>"):
            result.append((token.start(), token.end()))
            index += 1
            continue
        depth = 1
        close_index = index + 1
        while close_index < len(tokens):
            nested = tokens[close_index].group(0)
            if nested.startswith("</"):
                depth -= 1
            elif not nested.rstrip().endswith("/>"):
                depth += 1
            if depth == 0:
                result.append((token.start(), tokens[close_index].end()))
                break
            close_index += 1
        else:
            raise ValueError(f"Unclosed <{tag}> element in XML target")
        index = close_index + 1
    return result


def _one_span(text: str, tag: str, selector: dict[str, str]) -> tuple[int, int]:
    spans = _element_spans(text, tag, selector)
    if len(spans) != 1:
        raise ValueError(
            f"Expected one <{tag}> matching {selector}, found {len(spans)}"
        )
    return spans[0]


def _set_attributes(block: str, tag: str, values: dict[str, str]) -> str:
    start_match = re.match(rf"<{re.escape(tag)}\b[^>]*>", block, re.IGNORECASE)
    if start_match is None:
        raise ValueError(f"Replacement block does not start with <{tag}>")
    start_tag = start_match.group(0)
    for name, value in values.items():
        pattern = re.compile(
            rf"({re.escape(name)}\s*=\s*)(?:\"[^\"]*\"|'[^']*')"
        )
        start_tag, count = pattern.subn(rf'\g<1>"{value}"', start_tag, count=1)
        if count != 1:
            raise ValueError(f"Attribute {name!r} was not found on <{tag}>")
    return start_tag + block[start_match.end() :]


def _normalize_fragment(fragment: str, newline: str) -> str:
    return fragment.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)


def _validate_fragment(fragment: str, expected_tag: str) -> None:
    element = ET.fromstring(fragment.encode("utf-8"))
    if element.tag != expected_tag:
        raise ValueError(
            f"Replacement fragment must be <{expected_tag}>, got <{element.tag}>"
        )


def _child_count(block: str, tag: str) -> int:
    element = ET.fromstring(block.encode("utf-8"))
    return sum(1 for child in element.iter(tag))


def _replace_children(
    block: str,
    *,
    tag: str,
    expected_count: int,
    replacement: str,
) -> str:
    spans = _element_spans(block, tag)
    if len(spans) != expected_count:
        raise ValueError(
            f"Expected {expected_count} <{tag}> children, found {len(spans)}"
        )
    _validate_fragment(replacement, tag)
    for start, end in reversed(spans):
        block = block[:start] + replacement + block[end:]
    return block


def _clone_element(text: str, operation: dict[str, Any], newline: str) -> tuple[str, bool]:
    tag = operation["tag"]
    source_selector = operation["selector"]
    target_selector = operation["new_selector"]
    source_start, source_end = _one_span(text, tag, source_selector)
    source_block = text[source_start:source_end]
    clone = _set_attributes(source_block, tag, operation.get("set_attributes", {}))

    for item in operation.get("text_replacements", []):
        source = item["source"]
        replacement = item["replacement"]
        expected = item["expected_count"]
        actual = clone.count(source)
        if actual != expected:
            raise ValueError(
                f"Expected {expected} occurrences of {source!r} in cloned <{tag}>, "
                f"found {actual}"
            )
        clone = clone.replace(source, replacement)

    for item in operation.get("replace_children", []):
        replacement = _normalize_fragment(item["replacement"], newline)
        clone = _replace_children(
            clone,
            tag=item["tag"],
            expected_count=item["expected_count"],
            replacement=replacement,
        )

    _validate_fragment(clone, tag)
    existing = _element_spans(text, tag, target_selector)
    if existing:
        if len(existing) != 1 or text[existing[0][0] : existing[0][1]] != clone:
            raise ValueError(f"Existing <{tag}> {target_selector} conflicts with Mod patch")
        return text, False

    line_start = text.rfind("\n", 0, source_start) + 1
    indentation = text[line_start:source_start]
    insertion = newline + indentation + clone
    return text[:source_end] + insertion + text[source_end:], True


def _replace_element(text: str, operation: dict[str, Any], newline: str) -> tuple[str, bool]:
    tag = operation["tag"]
    start, end = _one_span(text, tag, operation["selector"])
    original = text[start:end]
    for child_tag, expected in operation.get("expected_child_counts", {}).items():
        actual = _child_count(original, child_tag)
        if actual != expected:
            replacement = _normalize_fragment(operation["replacement"], newline)
            if original == replacement:
                return text, False
            raise ValueError(
                f"Expected {expected} <{child_tag}> children in <{tag}>, found {actual}"
            )
    replacement = _normalize_fragment(operation["replacement"], newline)
    _validate_fragment(replacement, tag)
    if original == replacement:
        return text, False
    return text[:start] + replacement + text[end:], True


def validate_xml_patch_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec.get("game_target"), str):
        raise ValueError("XML patch requires a game_target")
    operations = spec.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("XML patch requires at least one operation")
    for operation in operations:
        kind = operation.get("kind")
        if kind not in {"clone_element", "replace_element"}:
            raise ValueError(f"Unsupported XML patch operation: {kind!r}")
        if not isinstance(operation.get("tag"), str):
            raise ValueError("XML patch operation requires a tag")
        selector = operation.get("selector")
        if not isinstance(selector, dict) or not selector:
            raise ValueError("XML patch operation requires a selector")
        if kind == "clone_element":
            if not isinstance(operation.get("new_selector"), dict):
                raise ValueError("clone_element requires new_selector")
        elif not isinstance(operation.get("replacement"), str):
            raise ValueError("replace_element requires a replacement fragment")


def apply_xml_patch(data: bytes, spec: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    """Apply one XML patch spec while preserving untouched bytes and formatting."""
    validate_xml_patch_spec(spec)
    bom = data.startswith(b"\xef\xbb\xbf")
    payload = data[3:] if bom else data
    text = payload.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    ET.fromstring(payload)

    records: list[dict[str, Any]] = []
    for operation in spec["operations"]:
        kind = operation["kind"]
        if kind == "clone_element":
            text, changed = _clone_element(text, operation, newline)
        else:
            text, changed = _replace_element(text, operation, newline)
        records.append(
            {
                "kind": kind,
                "tag": operation["tag"],
                "selector": operation["selector"],
                "changed": changed,
            }
        )

    encoded = text.encode("utf-8")
    ET.fromstring(encoded)
    return (b"\xef\xbb\xbf" if bom else b"") + encoded, records
