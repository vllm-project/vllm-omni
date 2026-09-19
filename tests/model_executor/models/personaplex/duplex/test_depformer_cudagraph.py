# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA tests for PersonaPlex depformer graph replay"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.personaplex.duplex._depformer_testing import (
    TEMPORAL,
    clone_depformer,
    frame,
    make_depformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_cudagraph import (
    CUDAGraphDepformerWrapper,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required"),
]


@pytest.fixture
def cuda_device() -> torch.device:
    return torch.device("cuda:0")


def test_graphed_depformer_matches_eager(cuda_device: torch.device) -> None:
    eager = make_depformer(cuda_device, seed=20)
    graphed_model = clone_depformer(eager, cuda_device)
    wrapper = CUDAGraphDepformerWrapper(graphed_model, capture_sizes=[1, 2], warmup_iters=3)
    wrapper.warmup(cuda_device)
    assert wrapper.is_ready
    assert wrapper.stats.capture_failure == 0

    for batch, seed in [(1, 21), (2, 22), (1, 23)]:
        text, hidden, tokens, provided = frame(batch, seed, cuda_device)
        want = eager(text, hidden, audio_tokens=tokens, audio_provided=provided)
        got = wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert wrapper.stats.replays == 3
    assert wrapper.stats.eager == 0


def test_padded_batch_matches_unpadded_live_row(cuda_device: torch.device) -> None:
    eager = make_depformer(cuda_device, seed=24)
    wrapper = CUDAGraphDepformerWrapper(clone_depformer(eager, cuda_device), capture_sizes=[2], warmup_iters=3)
    wrapper.warmup(cuda_device)
    text1, hidden1, tokens1, provided1 = frame(1, 25, cuda_device)
    want = eager(text1, hidden1, audio_tokens=tokens1, audio_provided=provided1)
    got = wrapper(text1, hidden1, audio_tokens=tokens1, audio_provided=provided1)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.shape[0] == 1
    assert wrapper.stats.replays == 1


def test_shape_divergence_falls_back_to_eager(cuda_device: torch.device) -> None:
    model = make_depformer(cuda_device, seed=26)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1], warmup_iters=1)
    wrapper.warmup(cuda_device)
    text, _hidden, tokens, provided = frame(1, 27, cuda_device)
    bad_hidden = torch.randn(1, 2, TEMPORAL, device=cuda_device)
    with pytest.raises(ValueError, match="transformer_out must be"):
        wrapper(text, bad_hidden, audio_tokens=tokens, audio_provided=provided)
    assert wrapper.stats.eager_shape_mismatch == 1
    assert wrapper.stats.replays == 0
    assert wrapper.stats.eager == 1


def test_oversized_batch_falls_back_to_eager(cuda_device: torch.device) -> None:
    model = make_depformer(cuda_device, max_graph_batch_size=8, seed=28)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2], warmup_iters=1)
    wrapper.warmup(cuda_device)
    text, hidden, tokens, provided = frame(3, 29, cuda_device)
    want = model(text, hidden, audio_tokens=tokens, audio_provided=provided)
    got = wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert wrapper.stats.eager_shape_mismatch == 1
    assert wrapper.stats.replays == 0


def test_outer_stream_capture_falls_back_to_eager(cuda_device: torch.device) -> None:
    model = make_depformer(cuda_device, seed=30)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1], warmup_iters=1)
    wrapper.warmup(cuda_device)
    text, hidden, tokens, provided = frame(1, 31, cuda_device)
    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer):
        wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    assert wrapper.stats.eager_outer_capture == 1
    assert wrapper.stats.replays == 0
    outer.replay()


def test_capture_failure_stays_eager(cuda_device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_depformer(cuda_device, seed=32)

    def _fail(self, *_args, **_kwargs):
        self.stats.capture_failure += 1
        return None

    monkeypatch.setattr(CUDAGraphDepformerWrapper, "_capture_one", _fail)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1], warmup_iters=1)
    wrapper.warmup(cuda_device)
    assert not wrapper.is_ready
    assert wrapper.stats.capture_failure == 1
    text, hidden, tokens, provided = frame(1, 33, cuda_device)
    want = model(text, hidden, audio_tokens=tokens, audio_provided=provided)
    got = wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert wrapper.stats.replays == 0
    assert wrapper.stats.eager == 1
