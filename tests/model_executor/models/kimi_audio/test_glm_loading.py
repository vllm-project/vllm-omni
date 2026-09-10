# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load a real, small CPU WhisperVQ network from an official-oracle fixture.

The checkpoint contains random reduced-width weights, not pretrained weights.
No external Kimi runtime, model download, or CUDA execution is required.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import WhisperConfig, WhisperFeatureExtractor
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioInputEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = Path(__file__).parent / "fixtures/glm_encoder_reference.safetensors"


@pytest.fixture
def glm_checkpoint(tmp_path):
    with safe_open(REFERENCE, framework="pt") as fixture:
        config = WhisperConfig(**json.loads(fixture.metadata()["config"]))
    weights = {
        name.removeprefix("weights."): value
        for name, value in load_file(REFERENCE).items()
        if name.startswith("weights.")
    }
    config.save_pretrained(tmp_path)
    WhisperFeatureExtractor(feature_size=config.num_mel_bins).save_pretrained(tmp_path)
    save_file(weights, str(tmp_path / "model.safetensors"))
    return SimpleNamespace(
        path=tmp_path,
        weights=weights,
        config=SimpleNamespace(
            load_config=SimpleNamespace(device="cpu"),
            device_config=SimpleNamespace(device="cuda:2"),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
    )


def test_lazy_glm_loading_preserves_checkpoint_dtype_and_reuses_network(glm_checkpoint):
    runtime = glm_checkpoint
    encoder = KimiAudioInputEncoder(vllm_config=runtime.config)
    assert encoder.audio_tokenizer is None and encoder.glm_feature_extractor is None
    assert list(encoder.parameters()) == []

    # Worker construction can run under the LLM dtype. GLM keeps its own FP32.
    with set_default_torch_dtype(torch.bfloat16):
        loaded = encoder.load_glm_weights(str(runtime.path))
    tokenizer = encoder.audio_tokenizer
    assert all(not module.training for module in tokenizer.modules())
    assert loaded == set(dict(encoder.named_parameters()))
    for name, param in tokenizer.named_parameters():
        assert param.device.type == "cpu" and param.dtype == torch.float32
        torch.testing.assert_close(param, runtime.weights[name], rtol=0, atol=0)
    assert not any("ema_" in name for name in tokenizer.state_dict())
    expected_loaded = set(loaded)
    loaded.clear()
    assert encoder.load_glm_weights(str(runtime.path / ".")) == expected_loaded

    waveform = np.zeros(1281, dtype=np.float32)
    first = encoder.encode_audio(waveform, sampling_rate=16000)
    second = encoder.encode_audio(waveform, sampling_rate=16000)
    assert first.codes == second.codes and len(first.codes) == 2
    assert all(0 <= code < tokenizer.config.quantize_vocab_size for code in first.codes)
    assert first.continuous_features is None and encoder.whisper_encoder is None
    assert encoder.audio_tokenizer is tokenizer

    other_path = runtime.path / "other-snapshot"
    other_path.mkdir()
    with pytest.raises(ValueError, match="different GLM checkpoint"):
        encoder.load_glm_weights(str(other_path))
    other_worker = KimiAudioInputEncoder(vllm_config=runtime.config)
    other_worker.load_glm_weights(str(runtime.path))
    assert other_worker.audio_tokenizer is not tokenizer


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_incomplete_glm_checkpoint_is_not_cached_and_can_retry(glm_checkpoint, corruption):
    runtime = glm_checkpoint
    encoder = KimiAudioInputEncoder(vllm_config=runtime.config)
    broken = dict(runtime.weights)
    if corruption == "missing":
        broken.pop("codebook.weight")
    else:
        broken["unknown.weight"] = torch.zeros(1)
    save_file(broken, str(runtime.path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="codebook.weight|unknown.weight"):
        encoder.load_glm_weights(str(runtime.path))
    assert encoder.audio_tokenizer is None and encoder.glm_feature_extractor is None
    assert list(encoder.parameters()) == []

    save_file(runtime.weights, str(runtime.path / "model.safetensors"))
    loaded = encoder.load_glm_weights(str(runtime.path))
    assert "audio_tokenizer.codebook.weight" in loaded
