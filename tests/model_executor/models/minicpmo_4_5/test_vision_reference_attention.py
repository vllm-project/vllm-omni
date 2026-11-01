# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference vision arithmetic, including CUDA graph replay and masked inputs."""

import pytest
import torch
from vllm.triton_utils import triton

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import SiglipAttention, SiglipVisionConfig


@pytest.mark.core_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("tokens", [1, 7, 31, 32, 33, 63, 257, 576, 1024, 1032, 2048])
def test_scaled_softmax_preserves_bf16_rounding(tokens: int) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.ops.scaled_softmax import scaled_softmax

    generator = torch.Generator(device="cuda").manual_seed(tokens)
    scores = torch.randn((19, tokens), device="cuda", dtype=torch.bfloat16, generator=generator) * 8
    scale = 72**-0.5
    expected = (scores * scale).softmax(-1, dtype=torch.float32).to(scores.dtype)
    actual = scaled_softmax(scores, scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.core_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_scaled_softmax_reuses_kernel_across_row_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.ops.scaled_softmax import (
        _scaled_softmax_kernel,
        scaled_softmax,
    )

    generator = torch.Generator(device="cuda").manual_seed(9517)
    tokens = 1032
    scale = 72**-0.5

    def check_rows(rows: int) -> None:
        scores = torch.randn((rows, tokens), device="cuda", dtype=torch.bfloat16, generator=generator) * 8
        expected = (scores * scale).softmax(-1, dtype=torch.float32).to(scores.dtype)
        actual = scaled_softmax(scores, scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Warm this family before observing misses, independent of existing caches.
    check_rows(19)
    cache_misses = 0
    runtime = triton.knobs.runtime
    previous_hook = runtime.jit_cache_hook

    def record_cache_miss(*, fn, **hook_kwargs) -> bool | None:
        nonlocal cache_misses
        if fn.jit_function is _scaled_softmax_kernel:
            cache_misses += 1
        if previous_hook is not None:
            return previous_hook(fn=fn, **hook_kwargs)
        return None

    # Base allocations preserve pointer alignment. Counts stay int32 while
    # covering tails, scalar-one specialization and observed image batch sizes.
    # A hook miss includes persistent-cache loads, not only backend compilation.
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "jit_cache_hook", record_cache_miss)
        for rows in (1, 2, 19, 32, 33, 16512, 33024, 115584):
            check_rows(rows)

    assert not cache_misses, f"Changing only row count caused {cache_misses} softmax JIT cache misses"


@pytest.mark.core_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("layout", ["transposed", "strided"])
def test_noncontiguous_scores_use_reference_attention(layout: str) -> None:
    attention = SiglipAttention(SiglipVisionConfig(hidden_size=32, num_attention_heads=4)).eval()
    generator = torch.Generator(device="cuda").manual_seed(9517)
    scores = torch.randn((2, 4, 7, 14), device="cuda", dtype=torch.bfloat16, generator=generator)
    scores = scores.transpose(-2, -1) if layout == "transposed" else scores[..., ::2]
    assert not scores.is_contiguous()

    expected = (scores * attention.scale).softmax(-1, dtype=torch.float32).to(scores.dtype)
    actual = attention._normalize_attention_weights(scores)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.core_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_reference_attention_replays_encoder_graph() -> None:
    config = SiglipVisionConfig(hidden_size=144, num_attention_heads=2)
    attention = SiglipAttention(config).cuda().to(torch.bfloat16).eval()
    generator = torch.Generator(device="cuda").manual_seed(9517)
    q, k, v = [
        torch.randn((2, 1032, 2, 72), device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    ]
    with torch.inference_mode():
        query, key, value = (x.transpose(1, 2) for x in (q, k, v))
        weights = (query @ key.transpose(-2, -1)) * attention.scale
        expected = (weights.softmax(-1, dtype=torch.float32).to(q.dtype) @ value).transpose(1, 2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                attention._attend_unpadded(q, k, v)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = attention._attend_unpadded(q, k, v)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("mask_dtype", [torch.bfloat16, torch.float32])
def test_masked_reference_attention_preserves_output_dtype(mask_dtype: torch.dtype) -> None:
    attention = SiglipAttention(SiglipVisionConfig(hidden_size=32, num_attention_heads=4)).eval()
    scores = torch.randn((2, 4, 7, 7), generator=torch.Generator().manual_seed(9517)).to(torch.bfloat16)
    mask = torch.zeros((2, 1, 7, 7), dtype=mask_dtype)
    mask[0, :, :, -2:] = -torch.inf
    expected = (scores * attention.scale + mask).softmax(-1, dtype=torch.float32).to(scores.dtype)
    actual = attention._normalize_attention_weights(scores, mask)
    assert actual.dtype == scores.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.core_model
@pytest.mark.cpu
def test_reference_attention_keeps_autograd() -> None:
    attention = SiglipAttention(SiglipVisionConfig(hidden_size=32, num_attention_heads=4)).eval()
    scores = torch.randn((2, 4, 7, 7), requires_grad=True)
    actual = attention._normalize_attention_weights(scores)
    actual[..., 0].sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0
