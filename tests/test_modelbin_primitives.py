from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from modelbin_bundle import blob_spec, parse_bundle, rebuild_with_blob_sequence  # noqa: E402


class ModelbinPrimitiveTests(unittest.TestCase):
    def test_sample_bundle_roundtrip_preserves_blob_contract(self) -> None:
        sample = ROOT / "samples" / "modelbin" / "Hair_Bald.modelbin"
        original = sample.read_bytes()
        bundle = parse_bundle(original)
        rebuilt = rebuild_with_blob_sequence(bundle, [blob_spec(blob) for blob in bundle.blobs])
        rebuilt_bundle = parse_bundle(rebuilt)
        self.assertEqual(
            [blob.tag for blob in rebuilt_bundle.blobs],
            [blob.tag for blob in bundle.blobs],
        )
        self.assertEqual(rebuilt_bundle.version, bundle.version)

    def test_sample_bundle_has_required_fh6_blocks(self) -> None:
        sample = ROOT / "samples" / "modelbin" / "Hair_Bald.modelbin"
        tags = {blob.tag for blob in parse_bundle(sample.read_bytes()).blobs}
        self.assertTrue({"Skel", "VLay", "MatI", "Mesh", "Skin", "VerB", "IndB"} <= tags)


if __name__ == "__main__":
    unittest.main()
