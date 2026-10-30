# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Codec loading order, precision validation and artifact integrity."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.model_executor.models.drvexo_speech_8b import code2wav
from vllm_omni.model_executor.models.drvexo_speech_8b.artifacts import resolve_codec_quantization

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CodecLoadingTests(unittest.TestCase):
    def test_bf16_conversion_after_native_load_and_cache_recompute(self):
        events = []

        class Decoder(torch.nn.Linear):
            def precompute_snake_caches(self):
                events.append(("cache", self.weight.dtype))

        def initialize(instance, **kwargs):
            torch.nn.Module.__init__(instance)
            instance.decoder = Decoder(4, 4, dtype=torch.float32)

        def load(instance, weights):
            events.append(("load", instance.decoder.weight.dtype))
            return {"decoder.weight", "decoder.bias"}

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="unused", hf_config=SimpleNamespace(drvexo_codec_compute_dtype="bfloat16")
            )
        )
        with (
            patch.object(code2wav.Qwen3TTSCode2Wav, "__init__", initialize),
            patch.object(code2wav.Qwen3TTSCode2Wav, "load_weights", load),
        ):
            model = code2wav.DRVEXOSpeech8BCode2Wav(vllm_config=config)
            self.assertEqual(model.load_weights(iter(())), {"decoder.weight", "decoder.bias"})
            self.assertEqual(events, [("load", torch.float32), ("cache", torch.bfloat16)])
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                model.load_weights(iter(()))

    def test_invalid_dtype_fails_before_parent_constructor(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(drvexo_codec_compute_dtype="fp4"))
        )
        with patch.object(code2wav.Qwen3TTSCode2Wav, "__init__") as parent:
            with self.assertRaises(ValueError):
                code2wav.DRVEXOSpeech8BCode2Wav(vllm_config=config)
            parent.assert_not_called()

    def test_codec_asset_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "packed").write_bytes(b"packed")
            (root / "reference").write_bytes(b"reference")
            spec = dict(
                format="drvexo_codec_linears_v1",
                precision="nvfp4",
                path="packed",
                source_path="reference",
                sha256=hashlib.sha256(b"packed").hexdigest(),
                source_sha256=hashlib.sha256(b"reference").hexdigest(),
            )
            self.assertEqual(resolve_codec_quantization(temp, spec)[0], "nvfp4")
            (root / "packed").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                resolve_codec_quantization(temp, spec)
