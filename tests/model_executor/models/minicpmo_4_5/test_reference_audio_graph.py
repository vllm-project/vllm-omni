# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from s3tokenizer.model import ModelConfig as S3ModelConfig
from s3tokenizer.model_v2 import S3TokenizerV2
from vllm.config import ModelConfig, VllmConfig

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import CampplusTorch
from vllm_omni.model_executor.models.minicpmo_4_5.code2wav.reference_audio_graph import S3CudaGraph, SpeakerCudaGraph

pytestmark = [pytest.mark.core_model]


class _Speaker(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


def _config():
    config = VllmConfig()
    config.model_config = ModelConfig.__new__(ModelConfig)
    config.model_config.multimodal_config = None
    return config


@pytest.mark.cpu
def test_cpu_fallback_does_not_mutate_shared_model_or_config():
    speaker = CampplusTorch(_Speaker(), "cpu")
    config = _config()
    wrapper = SpeakerCudaGraph(speaker, config)
    feat = torch.randn(17, 80, dtype=torch.bfloat16)
    assert torch.equal(wrapper(feat), speaker(feat))
    assert config.model_config.multimodal_config is None
    assert wrapper.graph.vllm_config.model_config.multimodal_config is not None
    assert not wrapper.graph.graphs
    assert wrapper.speaker is speaker


@pytest.mark.cpu
def test_facade_preserves_non_torch_speaker():
    from vllm_omni.model_executor.models.minicpmo_4_5.code2wav.token2wav import MiniCPMO45Token2wav

    facade = MiniCPMO45Token2wav.__new__(MiniCPMO45Token2wav)
    facade._core = SimpleNamespace(_spk_model=object())
    speaker = facade._core._spk_model
    assert facade.enable_speaker_cuda_graph(_config()) is False
    assert facade._core._spk_model is speaker


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_ownership_shape_capacity_and_precision_fallback():
    speaker = CampplusTorch(_Speaker(), "cuda:0")
    wrapper = SpeakerCudaGraph(speaker, _config(), max_graphs=1)
    feat = torch.randn(17, 80, device="cuda")
    wrapper(feat)
    retained = wrapper(feat)
    expected = speaker(feat).clone()
    feat.add_(3)
    assert torch.equal(wrapper(feat), speaker(feat))
    assert torch.equal(retained, expected)
    assert len(wrapper.graph.graphs) == 1
    other = torch.randn(19, 80, device="cuda")
    assert torch.equal(wrapper(other), speaker(other))
    assert len(wrapper.graph.graphs) == 1
    hits = wrapper.graph.get_cumulative_stats()["graph_hits"]
    original = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = not original
        assert torch.equal(wrapper(feat), speaker(feat))
        assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits
    finally:
        torch.backends.cudnn.allow_tf32 = original
    assert torch.equal(wrapper(feat), speaker(feat))
    assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits + 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(wrapper(feat), speaker(feat))
    assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits + 1


def _tokenizer(device):
    torch.manual_seed(17)
    config = S3ModelConfig(n_audio_state=64, n_audio_head=1, n_audio_layer=1, use_sdpa=True)
    tokenizer = S3TokenizerV2("speech_tokenizer_v2_25hz", config).to(device).eval()
    tokenizer.encoder.freqs_cis = tokenizer.encoder.freqs_cis.to(device)
    return tokenizer


def _reference(tokenizer, mels, lengths):
    hidden, lengths = tokenizer.encoder(mels, lengths)
    return tokenizer.quantizer.encode(hidden), lengths


@pytest.mark.cpu
@pytest.mark.parametrize("core_device", ["cpu", "cuda"])
def test_facade_preserves_cpu_tokenizer(core_device, mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5.code2wav.token2wav import MiniCPMO45Token2wav

    tokenizer = torch.nn.Linear(1, 1, device="cpu")
    facade = MiniCPMO45Token2wav.__new__(MiniCPMO45Token2wav)
    facade._core = mocker.Mock(device=torch.device(core_device), audio_tokenizer=tokenizer, _s3_cuda_graph=None)
    assert facade.enable_s3_cuda_graph(_config()) is False
    assert facade._core.audio_tokenizer is tokenizer
    assert facade._core._s3_cuda_graph is None


@pytest.mark.cpu
@pytest.mark.parametrize("frames", [59, 60, 61, 600])
@torch.inference_mode()
def test_cpu_fallback_preserves_tokens_and_lengths(frames):
    tokenizer = _tokenizer("cpu")
    config = _config()
    wrapper = S3CudaGraph(tokenizer, config)
    mels = torch.randn(2, 128, frames)
    lengths = torch.tensor([frames, frames - 11], dtype=torch.int32)
    actual = wrapper(mels, lengths)
    expected = _reference(tokenizer, mels, lengths)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert value.dtype == reference.dtype
    assert config.model_config.multimodal_config is None
    assert not wrapper.graph.graphs


@pytest.mark.cpu
def test_rejects_unknown_encoder():
    tokenizer = _tokenizer("cpu")
    tokenizer.encoder = torch.nn.Identity()
    with pytest.raises(ValueError, match="AudioEncoderV2"):
        S3CudaGraph(tokenizer, _config())


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_keeps_reference_tokens_owned_and_respects_shape_cap():
    tokenizer = _tokenizer("cuda")
    wrapper = S3CudaGraph(tokenizer, _config(), max_graphs=1)
    lengths = torch.tensor([60], dtype=torch.int32, device="cuda")
    mels = torch.randn(1, 128, 60, device="cuda")
    expected = _reference(tokenizer, mels, lengths)
    wrapper(mels, lengths)
    retained = wrapper(mels, lengths)
    mels.add_(1)
    for actual, reference in zip(wrapper(mels, lengths), _reference(tokenizer, mels, lengths), strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual, reference in zip(retained, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert len(wrapper.graph.graphs) == 1
    other = torch.randn(1, 128, 61, device="cuda")
    other_lengths = lengths + 1
    for actual, reference in zip(
        wrapper(other, other_lengths), _reference(tokenizer, other, other_lengths), strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert len(wrapper.graph.graphs) == 1
    hits = wrapper.graph.get_cumulative_stats()["graph_hits"]
    original = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = not original
        for actual, reference in zip(wrapper(mels, lengths), _reference(tokenizer, mels, lengths), strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits
    finally:
        torch.backends.cudnn.allow_tf32 = original
