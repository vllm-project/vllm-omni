# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Whisper loading lifecycle with tiny CPU weights, not a pretrained model.

The vLLM name mapper and Kimi encoder's load_weights method run unchanged.
Only network construction, the QKV parameter's copy operation, and checkpoint
I/O are replaced. This does not validate distributed loading or GPU kernels.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor import model_loader
from vllm.model_executor.models import kimi_audio

from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioInputEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]


@pytest.fixture
def whisper_runtime(monkeypatch):
    runtime = SimpleNamespace(constructions=[], sources=[], load_configs=[])
    runtime.weights = [
        ("model.decoder.embed_tokens.weight", torch.zeros(1)),
        ("model.encoder.conv1.weight", torch.full((2, 2), 7.0)),
        ("model.encoder.layers.0.fc1.weight", torch.full((2, 2), 4.0)),
        ("model.encoder.layers.0.fc2.weight", torch.full((2, 2), 5.0)),
        ("model.encoder.layers.0.self_attn.q_proj.weight", torch.full((2, 2), 1.0)),
        ("model.encoder.layers.0.self_attn.k_proj.weight", torch.full((2, 2), 2.0)),
        ("model.encoder.layers.0.self_attn.v_proj.weight", torch.full((2, 2), 3.0)),
        ("proj_out.weight", torch.zeros(1)),
    ]

    class TinyWhisper(kimi_audio.KimiAudioWhisperEncoder):
        # Keep the real encoder's load_weights and QKV mapper, but construct
        # tiny CPU parameters without starting the distributed runtime.
        def __init__(self, *, vllm_config, prefix):
            torch.nn.Module.__init__(self)
            runtime.constructions.append((vllm_config, prefix))
            self.conv1 = torch.nn.Linear(2, 2, bias=False)
            layer = torch.nn.Module()
            layer.mlp = torch.nn.ModuleDict(
                {"fc1": torch.nn.Linear(2, 2, bias=False), "fc2": torch.nn.Linear(2, 2, bias=False)}
            )
            layer.self_attn = torch.nn.ModuleDict({"qkv_proj": torch.nn.Linear(2, 6, bias=False)})
            layer.self_attn.qkv_proj.weight.weight_loader = self.copy_qkv
            self.layers = torch.nn.ModuleList([layer])

        @staticmethod
        def copy_qkv(param, weight):
            # vLLM's mapper supplies shard_id on the checkpoint tensor.
            start = {"q": 0, "k": 2, "v": 4}[weight.shard_id]
            with torch.no_grad():
                param[start : start + 2].copy_(weight)

    class CheckpointLoader:
        Source = model_loader.DefaultModelLoader.Source

        def __init__(self, config):
            runtime.load_configs.append(config)

        def _get_weights_iterator(self, source):
            runtime.sources.append(source)
            return iter(runtime.weights)

    monkeypatch.setattr(kimi_audio, "KimiAudioWhisperEncoder", TinyWhisper)
    monkeypatch.setattr(model_loader, "DefaultModelLoader", CheckpointLoader)
    runtime.config = SimpleNamespace(
        model_config=SimpleNamespace(model="kimi-checkpoint", revision="pinned-revision", dtype=torch.float64),
        load_config=SimpleNamespace(device="cpu"),
        # The loader's explicit CPU device must take precedence over this.
        device_config=SimpleNamespace(device="cuda:0"),
    )
    return runtime


def test_lazy_construction_load_mapping_and_reuse(whisper_runtime):
    runtime = whisper_runtime
    encoder = KimiAudioInputEncoder(vllm_config=runtime.config, prefix="ar.input_encoder")
    assert encoder.whisper_encoder is None
    assert list(encoder.parameters()) == []
    assert runtime.constructions == runtime.sources == []

    original_dtype = torch.get_default_dtype()
    loaded = encoder.load_whisper_weights()
    whisper = encoder.whisper_encoder
    assert torch.get_default_dtype() == original_dtype
    assert runtime.constructions == [(runtime.config, "ar.input_encoder.whisper_encoder")]
    assert runtime.load_configs == [runtime.config.load_config]
    assert len(runtime.sources) == 1
    source = runtime.sources[0]
    assert (source.model_or_path, source.revision, source.subfolder) == (
        "kimi-checkpoint",
        "pinned-revision",
        "whisper-large-v3",
    )
    assert loaded == set(dict(encoder.named_parameters()))
    assert all(p.device.type == "cpu" and p.dtype == torch.float64 for p in whisper.parameters())
    assert all(not module.training for module in whisper.modules())
    torch.testing.assert_close(whisper.conv1.weight, torch.full((2, 2), 7.0, dtype=torch.float64))
    torch.testing.assert_close(whisper.layers[0].mlp.fc1.weight, torch.full((2, 2), 4.0, dtype=torch.float64))
    torch.testing.assert_close(whisper.layers[0].mlp.fc2.weight, torch.full((2, 2), 5.0, dtype=torch.float64))
    torch.testing.assert_close(
        whisper.layers[0].self_attn.qkv_proj.weight,
        torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=torch.float64),
    )

    expected_loaded = set(loaded)
    loaded.clear()
    assert encoder.load_whisper_weights() == expected_loaded
    assert encoder.whisper_encoder is whisper
    assert len(runtime.constructions) == len(runtime.sources) == 1


def test_incomplete_checkpoint_is_not_cached_and_can_retry(whisper_runtime):
    runtime = whisper_runtime
    encoder = KimiAudioInputEncoder(vllm_config=runtime.config)
    complete_weights = runtime.weights
    runtime.weights = [(name, value) for name, value in complete_weights if name != "model.encoder.conv1.weight"]
    with pytest.raises(ValueError, match="Missing Kimi-Audio Whisper weights.*conv1.weight"):
        encoder.load_whisper_weights()
    assert encoder.whisper_encoder is None
    assert list(encoder.parameters()) == []

    runtime.weights = complete_weights
    loaded = encoder.load_whisper_weights()
    assert "whisper_encoder.conv1.weight" in loaded
    assert len(runtime.constructions) == len(runtime.sources) == 2
