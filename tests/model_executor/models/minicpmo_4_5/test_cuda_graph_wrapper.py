# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    HiFTGenerator,
)
from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    HiFTGraphWrapper,
)

pytestmark = [pytest.mark.core_model]


class _F0Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv1d(80, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(1).abs()


class _DeterministicSineGen(nn.Module):
    """Remove source RNG so eager and replay compare only execution paths."""

    def __init__(self, num_harmonics: int) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics

    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*f0.shape[:-1], self.num_harmonics)
        sine = f0.new_zeros(shape)
        uv = f0.new_ones((*f0.shape[:-1], 1))
        return sine, uv, sine


def _small_hift() -> HiFTGenerator:
    hift = HiFTGenerator(
        base_channels=32,
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        f0_predictor=_F0Predictor(),
    )
    hift.m_source.l_sin_gen = _DeterministicSineGen(hift.nb_harmonics + 1)
    return hift.eval().cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hift_graph_replay_matches_eager_for_uncached_and_cached_shapes() -> None:
    torch.manual_seed(0)
    hift = _small_hift()
    token2wav = SimpleNamespace(
        hift=hift,
        flow=SimpleNamespace(
            encoder=SimpleNamespace(pre_lookahead_layer=SimpleNamespace(pre_lookahead_len=3)),
            token_mel_ratio=2,
        ),
        mel_cache_len=2,
        source_cache_len=960,
    )
    wrapper = HiFTGraphWrapper(
        token2wav,
        connector_config={"codec_chunk_frames": 2, "codec_left_context_frames": 3},
        capture_batch_sizes=[1],
    )
    wrapper.capture()

    cases = (
        (torch.randn(1, 80, 4, device="cuda"), torch.zeros(1, 1, 0, device="cuda")),
        (torch.randn(1, 80, 6, device="cuda"), torch.randn(1, 1, 960, device="cuda")),
    )
    with torch.inference_mode():
        for speech_feat, cache_source in cases:
            expected_speech, expected_source = hift.inference(speech_feat, cache_source)
            actual_speech, actual_source = wrapper.replay(speech_feat, cache_source)
            torch.testing.assert_close(actual_speech, expected_speech, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(actual_source, expected_source, rtol=1e-4, atol=1e-5)


class _FakeGraph:
    def replay(self) -> None:
        return None


def _fake_wrapper(monkeypatch: pytest.MonkeyPatch) -> HiFTGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = object.__new__(HiFTGraphWrapper)
    wrapper.capture_batch_sizes = [1]
    wrapper.graph = {}
    wrapper.static_speech_inputs = {}
    wrapper.static_cache_source_inputs = {}
    wrapper.static_magnitude_outputs = {}
    wrapper.static_phase_outputs = {}
    wrapper.static_cache_source_outputs = {}
    wrapper.lazy_graph_count = 0
    wrapper.max_lazy_graphs = 1
    wrapper.decode_fn = Mock(return_value=(torch.tensor([[99.0]]), torch.tensor([[[98.0]]])))
    wrapper.finalize_fn = lambda magnitude, phase: magnitude + phase

    def capture(batch_size: int, num_frames: int, cache_len: int) -> None:
        key = (batch_size, num_frames, cache_len)
        wrapper.graph[key] = _FakeGraph()
        wrapper.static_speech_inputs[key] = torch.zeros(batch_size, 80, num_frames)
        wrapper.static_cache_source_inputs[key] = torch.zeros(batch_size, 1, cache_len)
        wrapper.static_magnitude_outputs[key] = torch.ones(batch_size, 1, num_frames)
        wrapper.static_phase_outputs[key] = torch.ones(batch_size, 1, num_frames)
        wrapper.static_cache_source_outputs[key] = torch.ones(batch_size, 1, num_frames)

    wrapper._capture = Mock(side_effect=capture)
    return wrapper


def test_unseen_shape_is_lazily_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _fake_wrapper(monkeypatch)
    speech, source = wrapper.replay(torch.randn(1, 80, 7), torch.zeros(1, 1, 0))

    wrapper._capture.assert_called_once_with(1, 7, 0)
    assert wrapper.lazy_graph_count == 1
    assert speech.shape == (1, 1, 7)
    assert source.shape == (1, 1, 7)
    wrapper.decode_fn.assert_not_called()


def test_lazy_capture_limit_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _fake_wrapper(monkeypatch)
    wrapper.lazy_graph_count = wrapper.max_lazy_graphs
    speech_feat = torch.randn(1, 80, 9)
    cache_source = torch.zeros(1, 1, 0)

    result = wrapper.replay(speech_feat, cache_source)

    wrapper._capture.assert_not_called()
    wrapper.decode_fn.assert_called_once_with(speech_feat, cache_source)
    assert result is wrapper.decode_fn.return_value
