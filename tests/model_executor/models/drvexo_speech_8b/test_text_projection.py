# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""TTS quantization must replace the used module and preserve loader ordering."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.model_executor.models.drvexo_speech_8b import talker, text_projection
from vllm_omni.model_executor.models.drvexo_speech_8b.artifacts import resolve_text_projection_quantization

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TextProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {
            f"linear_fc{i}.{kind}": torch.ones((4, 4) if kind == "weight" else (4,), dtype=torch.bfloat16)
            for i in (1, 2)
            for kind in ("weight", "bias")
        }
        save_file(self.source, str(self.root / "bf16.safetensors"))
        (self.root / "fp8.safetensors").write_bytes(b"packed fixture for path validation")

        def sha(name):
            return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

        self.spec = dict(
            format="drvexo_text_projection_v1",
            precision="fp8",
            path="fp8.safetensors",
            sha256=sha("fp8.safetensors"),
            source_path="bf16.safetensors",
            source_sha256=sha("bf16.safetensors"),
        )

    def test_optional_and_moved_paths(self):
        self.assertIsNone(resolve_text_projection_quantization("missing", None))
        folder = self.root / "moved"
        folder.mkdir()
        for name in ("bf16.safetensors", "fp8.safetensors"):
            (self.root / name).rename(folder / name)
        self.assertEqual(
            resolve_text_projection_quantization(str(folder), self.spec),
            ("fp8", str(folder / "fp8.safetensors"), str(folder / "bf16.safetensors")),
        )

    def test_bad_precision_and_corrupt_artifact_fail(self):
        with self.assertRaises(ValueError):
            resolve_text_projection_quantization(str(self.root), {**self.spec, "precision": "int4"})
        (self.root / "fp8.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            resolve_text_projection_quantization(str(self.root), self.spec)

    def test_original_tensors_are_checked_and_only_projection_removed(self):
        weights = [("talker.text_projection." + k, v) for k, v in self.source.items()]
        keep = ("talker.model.weight", torch.ones(1))
        self.assertEqual(list(text_projection.without_source_projection(iter(weights + [keep]), self.source)), [keep])
        wrong = [(n, v + 1) for n, v in weights]
        for rows in (wrong, weights[:-1], weights + [weights[0]]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                list(text_projection.without_source_projection(iter(rows), self.source))

    def test_constructor_replaces_prompt_builder_reference(self):
        config = SimpleNamespace(
            drvexo_output_conditioning="generated_tokens",
            drvexo_text_projection_quantization=self.spec,
            drvexo_connector_checkpoint="connectors.pt",
            drvexo_speaker_embedding="speaker.pt",
            drvexo_bridge_layers=[],
        )
        vc = SimpleNamespace(model_config=SimpleNamespace(model=str(self.root), hf_config=config))

        def base_init(instance, **kwargs):
            nn.Module.__init__(instance)
            instance.talker_config = SimpleNamespace(text_hidden_size=4, hidden_size=4, hidden_act="silu")
            instance.text_projection = nn.Identity()
            instance._prompt_builder = SimpleNamespace(_text_projection=instance.text_projection)
            instance.gpu_resident_buffer_keys = set()

        replacement = nn.Linear(4, 4)
        with (
            patch.object(talker.Qwen3TTSTalkerForConditionalGeneration, "__init__", base_init),
            patch.object(text_projection, "QuantizedTextProjection", return_value=replacement),
            patch.object(talker, "DRVEXOSpeechBridge", return_value=nn.Identity()),
        ):
            model = talker.DRVEXOSpeech8BTalker(vllm_config=vc)
        self.assertIs(model.text_projection, replacement)
        self.assertIs(model._prompt_builder._text_projection, replacement)

    def test_pad_projection_waits_until_native_loader_finishes(self):
        model = talker.DRVEXOSpeech8BTalker.__new__(talker.DRVEXOSpeech8BTalker)
        nn.Module.__init__(model)
        model.text_projection_quantization = ("fp8", "packed", "reference")
        model._quantized_pad_pending = False
        with patch.object(talker.Qwen3TTSTalkerForConditionalGeneration, "_init_runtime_buffers") as initialize:
            model._init_runtime_buffers()
            initialize.assert_not_called()
            self.assertTrue(model._quantized_pad_pending)
            model._ensure_quantized_pad_embed()
            model._ensure_quantized_pad_embed()
            initialize.assert_called_once()
            self.assertFalse(model._quantized_pad_pending)
            model.text_projection_quantization = None
            model._init_runtime_buffers()
            self.assertEqual(initialize.call_count, 2)


if __name__ == "__main__":
    unittest.main()
