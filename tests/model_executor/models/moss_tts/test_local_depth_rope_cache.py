# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU/NPU parity tests for MOSS-TTS Local Depth RoPE caching."""

import pytest
import torch
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
    MossTTSLocalDepthTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.tts, pytest.mark.npu]


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


npu_only = pytest.mark.skipif(not _npu_available(), reason="NPU device or torch_npu not available.")
npu_device = pytest.param("npu", marks=npu_only)


@pytest.fixture(
    params=[pytest.param("cpu", marks=pytest.mark.cpu), npu_device],
    ids=["cpu", "npu"],
)
def device(request):
    return torch.device(request.param)


def _test_config() -> GPT2Config:
    return GPT2Config(
        n_embd=32,
        n_head=4,
        n_inner=64,
        layer_norm_epsilon=1e-6,
        rope_base=10000.0,
    )


def test_rope_cache_matches_reference_and_reuses_storage(device) -> None:
    model = MossTTSLocalDepthTransformer(_test_config()).eval().to(device)
    attn = model.h[0].attn
    attn.prepare_rope_cache(12, device, torch.float32)
    cache_ptr = attn._rope_cos_cache.data_ptr()
    cos, sin = attn._rope_cos_sin(4, device, torch.float32)

    positions = torch.arange(4, dtype=torch.float32, device=device)
    freqs = torch.einsum("s,d->sd", positions, attn.inv_freq.float().to(device))
    expected_cos = freqs.cos().repeat_interleave(2, dim=-1).view(1, 4, 1, attn.head_dim)
    expected_sin = freqs.sin().repeat_interleave(2, dim=-1).view(1, 4, 1, attn.head_dim)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)

    # Smaller slices retain the fixed allocation used by CUDA Graph.
    attn.prepare_rope_cache(6, device, torch.float32)
    assert attn._rope_cos_cache.data_ptr() == cache_ptr


def test_frame_local_kv_matches_full_prefix_and_reuses_slots(device) -> None:
    model = MossTTSLocalDepthTransformer(_test_config()).eval().to(device)
    model.h[0].attn.prepare_rope_cache(12, device, torch.float32)
    cache = (torch.empty(2, 4, 12, 8, device=device), torch.empty(2, 4, 12, 8, device=device))
    with torch.inference_mode():
        for _ in range(2):
            inputs = torch.randn(2, 12, 32, device=device)
            for position in range(12):
                expected = model._forward_prefix(inputs[:, : position + 1])[:, -1:]
                actual = model._forward_prefix(inputs[:, position : position + 1], cache, position)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
