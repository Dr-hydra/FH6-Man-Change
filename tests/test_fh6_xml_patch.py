from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fh6_xml_patch import apply_xml_patch


def test_clone_and_replace_preserve_unrelated_text() -> None:
    source = (
        '<?xml version="1.0" encoding="utf-8"?>\r\n'
        '<Root>\r\n'
        '\t<Asset id="Retail"><Variant path="old"><Materials><M/></Materials></Variant></Asset>\r\n'
        '\t<Item assetid="Retail" id="Special"><Materials><M/><M/></Materials></Item>\r\n'
        '\t<Keep value="byte-exact"/>\r\n'
        '</Root>\r\n'
    ).encode()
    spec = {
        "game_target": "media/test.xml",
        "operations": [
            {
                "kind": "clone_element",
                "tag": "Asset",
                "selector": {"id": "Retail"},
                "new_selector": {"id": "Custom"},
                "set_attributes": {"id": "Custom"},
                "text_replacements": [
                    {"source": 'path="old"', "replacement": 'path="new"', "expected_count": 1}
                ],
                "replace_children": [
                    {
                        "tag": "Materials",
                        "expected_count": 1,
                        "replacement": "<Materials><Custom/></Materials>",
                    }
                ],
            },
            {
                "kind": "replace_element",
                "tag": "Item",
                "selector": {"id": "Special"},
                "expected_child_counts": {"M": 2},
                "replacement": '<Item assetid="Custom" id="Special"/>',
            },
        ],
    }

    patched, records = apply_xml_patch(source, spec)
    text = patched.decode()
    assert '<Asset id="Retail">' in text
    assert '<Asset id="Custom"><Variant path="new">' in text
    assert '<Item assetid="Custom" id="Special"/>' in text
    assert '\t<Keep value="byte-exact"/>\r\n' in text
    assert [item["changed"] for item in records] == [True, True]

    repeated, repeated_records = apply_xml_patch(patched, spec)
    assert repeated == patched
    assert [item["changed"] for item in repeated_records] == [False, False]
