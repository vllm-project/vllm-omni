# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiMo's eager and captured local samplers must choose the same speech codes."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_llm import MiMoLocalSamplerTensor, MiMoSampler

pytestmark = [pytest.mark.core_model]

_EMPTY_ID = 1024


def _samplers(batch_size: int, device: str) -> tuple[MiMoSampler, MiMoLocalSamplerTensor]:
    return (
        MiMoSampler(do_sample=False, temperature=0.9, top_p=0.95),
        MiMoLocalSamplerTensor(
            temperature=torch.full((batch_size,), 0.9, dtype=torch.float32, device=device),
            top_k=torch.full((batch_size,), -1, dtype=torch.int64, device=device),
            top_p=torch.full((batch_size,), 0.95, dtype=torch.float32, device=device),
        ),
    )


def _logits(scenario: str, batch_size: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    scores = torch.full((batch_size, _EMPTY_ID + 1), -20.0, dtype=dtype, device=device)
    if scenario == "near_tie":
        # Both values round to 0.1328125 after BF16 division by 0.9.
        scores[:, 0] = 0.119140625
        scores[:, 1] = 0.11962890625
    else:
        scores[:, 0] = 0.0
        scores[:, 1] = 1.0
        # Filtering before excluding this token would remove every valid token.
        scores[:, _EMPTY_ID] = 10.0
    scores[1::2, :2] = scores[1::2, :2].flip(-1)
    return scores


@pytest.mark.cpu
@pytest.mark.parametrize("scenario", ["near_tie", "excluded_maximum"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_local_sampler_parity(scenario: str, dtype: torch.dtype, batch_size: int) -> None:
    scores = _logits(scenario, batch_size, dtype, "cpu")
    original = scores.clone()
    eager, captured = _samplers(batch_size, "cpu")
    expected = 1 - torch.arange(batch_size) % 2

    torch.testing.assert_close(eager.sample(scores, [_EMPTY_ID]), expected)
    torch.testing.assert_close(captured.sample(scores, [_EMPTY_ID]), expected)
    torch.testing.assert_close(scores, original)


@pytest.mark.cpu
def test_stochastic_sampler_filters_after_excluding_tokens() -> None:
    scores = _logits("excluded_maximum", 1, torch.float32, "cpu")
    sampler = MiMoSampler(do_sample=True, temperature=0.9, top_k=1, top_p=0.95)

    # With top-k=1, the only allowed draw is the best nonempty token.
    assert sampler.sample(scores, [_EMPTY_ID]).item() == 1


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_local_sampler_cuda_replay(dtype: torch.dtype, batch_size: int) -> None:
    scores = _logits("near_tie", batch_size, dtype, "cuda")
    eager, captured = _samplers(batch_size, "cuda")
    expected = 1 - torch.arange(batch_size, device="cuda") % 2

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            captured.sample(scores, [_EMPTY_ID])
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = captured.sample(scores, [_EMPTY_ID])

    for scenario in ["near_tie", "excluded_maximum"]:
        scores.copy_(_logits(scenario, batch_size, dtype, "cuda"))
        for flip in [False, True]:
            if flip:
                scores[:, :2].copy_(scores[:, :2].flip(-1))
            graph.replay()
            torch.testing.assert_close(output, 1 - expected if flip else expected)
            torch.testing.assert_close(output, eager.sample(scores, [_EMPTY_ID]))
