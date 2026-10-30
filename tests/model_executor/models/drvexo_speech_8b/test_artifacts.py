# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Relocation checks exercise actual DRVEXO path consumers with tiny CPU layers."""

import json
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import WhisperFeatureExtractor

from vllm_omni.model_executor.models.drvexo_speech_8b import talker, thinker
from vllm_omni.model_executor.models.drvexo_speech_8b.artifacts import resolve_artifact_path
from vllm_omni.model_executor.models.drvexo_speech_8b.processing import DRVEXOProcessingInfo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@contextmanager
def chdir(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ArtifactRelocationTest(unittest.TestCase):
    def test_moved_bundle_loads_projector_speaker_and_audio_frontend(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            original = parent / "original"
            original.mkdir()
            (original / "talker").mkdir()
            (original / "assets").mkdir()
            WhisperFeatureExtractor(feature_size=128).save_pretrained(original / "audio_encoder")
            projector = thinker._InputProjector(4, 8)
            bridge = nn.Linear(4, 4)
            state = {f"input_projector.{k}": v for k, v in projector.state_dict().items()}
            state.update({f"speech_bridge.{k}": v for k, v in bridge.state_dict().items()})
            speaker = torch.arange(4, dtype=torch.float32)
            weights = {f"connectors.{k}": v for k, v in state.items()}
            weights["speaker.value"] = speaker
            save_file(weights, original / "model.safetensors")
            fmt = "drvexo_unified_safetensors_v1"
            (original / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"format": fmt},
                        "weight_map": {k: "model.safetensors" for k in weights},
                    }
                )
            )
            (original / "components.json").write_text(
                json.dumps(
                    {
                        "format": fmt,
                        "components": {
                            "connectors": {
                                "prefix": "connectors.",
                                "tensor_count": len(state),
                                "payload_kind": "state_dict",
                            },
                            "speaker": {"prefix": "speaker.", "tensor_count": 1, "payload_kind": "tensor"},
                        },
                    }
                )
            )
            for name in ("connectors", "speaker"):
                (original / f"assets/{name}.json").write_text(
                    json.dumps(
                        {
                            "format": "drvexo_unified_tensor_view_v1",
                            "index": "../model.safetensors.index.json",
                            "component": name,
                        }
                    )
                )
            moved = parent / "moved"
            shutil.move(original, moved)
            self.assertFalse(original.exists())
            unrelated = parent / "unrelated-cwd"
            unrelated.mkdir()
            config = SimpleNamespace(
                audio_feature_dim=4,
                hidden_size=8,
                connector_checkpoint="assets/connectors.json",
                audio_encoder_path="audio_encoder",
                audio_encoder_config={},
            )
            vllm_config = SimpleNamespace(model_config=SimpleNamespace(model=str(moved), hf_config=config))

            def base_init(instance, **kwargs):
                nn.Module.__init__(instance)

            with (
                chdir(unrelated),
                patch.object(thinker.Qwen3ForCausalLM, "__init__", base_init),
                patch.object(
                    thinker.DRVEXOSpeech8BForConditionalGeneration, "_mark_tower_model", return_value=nullcontext()
                ),
                patch.object(thinker, "Qwen3OmniMoeAudioEncoder", return_value=nn.Identity()),
            ):
                model = thinker.DRVEXOSpeech8BForConditionalGeneration(vllm_config=vllm_config)
                self.assertEqual(model.audio_encoder_path, str(moved / "audio_encoder"))
                loaded = model._load_connector_checkpoint(model.connector_checkpoint)
                self.assertEqual(loaded, {k for k in state if k.startswith("input_projector.")})
                for name, value in model.input_projector.state_dict().items():
                    torch.testing.assert_close(value, state["input_projector." + name], rtol=0, atol=0)
                info = DRVEXOProcessingInfo(SimpleNamespace(model_config=vllm_config.model_config))
                self.assertEqual(info.feature_extractor.feature_size, 128)

            tts_config = SimpleNamespace(
                drvexo_output_conditioning="generated_tokens",
                drvexo_source_tokenizer="..",
                drvexo_connector_checkpoint="../assets/connectors.json",
                drvexo_speaker_embedding="../assets/speaker.json",
                drvexo_bridge_layers=[],
            )
            tts_vllm = SimpleNamespace(model_config=SimpleNamespace(model=str(moved / "talker"), hf_config=tts_config))

            def tts_base_init(instance, **kwargs):
                nn.Module.__init__(instance)
                instance.model = nn.Module()
                instance.model.layers = nn.ModuleList()
                instance.talker_config = SimpleNamespace(hidden_size=4)
                instance.gpu_resident_buffer_keys = set()

            with (
                chdir(unrelated),
                patch.object(talker.Qwen3TTSTalkerForConditionalGeneration, "__init__", tts_base_init),
                patch.object(talker.Qwen3TTSTalkerForConditionalGeneration, "load_weights", return_value=set()),
                patch.object(talker, "DRVEXOSpeechBridge", side_effect=lambda: nn.Linear(4, 4)),
            ):
                speech = talker.DRVEXOSpeech8BTalker(vllm_config=tts_vllm)
                self.assertEqual(speech.source_tokenizer_path, str(moved))
                speech.load_weights(iter(()))
                torch.testing.assert_close(speech.drvexo_speaker, speaker, rtol=0, atol=0)
                for name, value in speech.speech_bridge.state_dict().items():
                    torch.testing.assert_close(value, state["speech_bridge." + name], rtol=0, atol=0)

    def test_historical_absolute_asset_still_resolves(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weights.pt"
            artifact.touch()
            self.assertEqual(resolve_artifact_path("unused-model-id", str(artifact)), str(artifact))

    def test_missing_local_model_is_rejected_for_relative_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_artifact_path(str(Path(directory) / "missing"), "assets/checkpoint.pt")

    def test_empty_and_remote_asset_paths_are_rejected(self):
        for value in ("", " ", None, "https://example.com/model", "s3://bucket/model"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_artifact_path(".", value)


if __name__ == "__main__":
    unittest.main()
