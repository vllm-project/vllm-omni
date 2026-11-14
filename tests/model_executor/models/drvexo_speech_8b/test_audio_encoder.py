# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Packed encoders must bind to exact native sources and portable assets."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.model_executor.models.drvexo_speech_8b import audio_encoder
from vllm_omni.model_executor.models.drvexo_speech_8b.artifacts import resolve_audio_encoder_quantization

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class AudioEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        save_file(dict(self.model.named_parameters()), str(self.root / "source.safetensors"))
        save_file({"weight": self.model.weight.to(torch.float8_e4m3fn)}, str(self.root / "packed.safetensors"))

        def sha(f):
            return hashlib.sha256((self.root / f).read_bytes()).hexdigest()

        self.spec = {
            "format": "drvexo_native_audio_encoder_v1",
            "precision": "fp8",
            "path": "packed.safetensors",
            "sha256": sha("packed.safetensors"),
            "source_path": "source.safetensors",
            "source_sha256": sha("source.safetensors"),
        }

    def test_optional_and_relocated_assets(self):
        self.assertIsNone(resolve_audio_encoder_quantization("missing", None))
        moved = self.root / "moved"
        moved.mkdir()
        for name in ("source.safetensors", "packed.safetensors"):
            (self.root / name).rename(moved / name)
        self.assertEqual(
            resolve_audio_encoder_quantization(str(moved), self.spec),
            ("fp8", str(moved / "packed.safetensors"), str(moved / "source.safetensors")),
        )

    def test_corrupt_source_or_wrong_format_rejected(self):
        with self.assertRaises(ValueError):
            resolve_audio_encoder_quantization(str(self.root), {**self.spec, "format": "other"})
        (self.root / "source.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            resolve_audio_encoder_quantization(str(self.root), self.spec)

    def test_native_source_values_checked_before_module_replacement(self):
        with torch.no_grad():
            self.model.weight.add_(1)
        with (
            patch.object(audio_encoder, "get_tensor_model_parallel_world_size", return_value=1),
            self.assertRaisesRegex(ValueError, "differs from calibration source"),
        ):
            audio_encoder.install_packed_audio_encoder(
                self.model, "fp8", self.root / "packed.safetensors", self.root / "source.safetensors"
            )

    def test_non_tp1_fails_before_reading_assets(self):
        with (
            patch.object(audio_encoder, "get_tensor_model_parallel_world_size", return_value=2),
            self.assertRaisesRegex(ValueError, "TP1"),
        ):
            audio_encoder.install_packed_audio_encoder(self.model, "fp8", "missing", "missing")


if __name__ == "__main__":
    unittest.main()
