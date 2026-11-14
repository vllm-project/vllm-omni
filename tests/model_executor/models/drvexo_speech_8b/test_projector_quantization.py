# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit component precision must bind the exact trained and packed files."""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.drvexo_speech_8b.artifacts import resolve_projector_quantization

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class ProjectorQuantizationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir()
        self.source = self.root / "connector.pt"
        self.packed = self.root / "nvfp4.safetensors"
        self.source.write_bytes(b"original selected trained checkpoint")
        self.packed.write_bytes(b"packed component fixture")
        self.spec = {
            "format": "drvexo_input_projector_v1",
            "precision": "nvfp4",
            "path": self.packed.name,
            "sha256": hashlib.sha256(self.packed.read_bytes()).hexdigest(),
            "source_checkpoint_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
        }

    def test_disabled_path_needs_no_packed_artifact(self):
        self.assertIsNone(resolve_projector_quantization("not-a-path", None, None))

    def test_relative_component_survives_bundle_move(self):
        moved = self.root.with_name("relocated")
        shutil.move(self.root, moved)
        result = resolve_projector_quantization(str(moved), self.spec, str(moved / self.source.name))
        self.assertEqual(result, ("nvfp4", str(moved / self.packed.name)))

    def test_wrong_source_checkpoint_rejected(self):
        self.source.write_bytes(b"different trained projector")
        with self.assertRaisesRegex(ValueError, "source projector hash mismatch"):
            resolve_projector_quantization(str(self.root), self.spec, str(self.source))

    def test_corrupt_packed_weight_rejected(self):
        self.packed.write_bytes(b"changed weight")
        with self.assertRaisesRegex(ValueError, "packed projector hash mismatch"):
            resolve_projector_quantization(str(self.root), self.spec, str(self.source))

    def test_unsupported_precision_is_not_silently_ignored(self):
        self.spec["precision"] = "int4"
        with self.assertRaisesRegex(ValueError, "Unsupported input projector precision"):
            resolve_projector_quantization(str(self.root), self.spec, str(self.source))

    def test_missing_hash_and_bad_spec_rejected(self):
        for value in [False, {}, {"format": "different"}, {**self.spec, "sha256": None}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_projector_quantization(str(self.root), value, str(self.source))


if __name__ == "__main__":
    unittest.main()
