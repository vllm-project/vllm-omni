# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity tests for MOSS-TTS Local Depth RoPE caching."""

import pytest
import torch
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
    MossTTSLocalDepthTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _test_config() -> GPT2Config:
    return GPT2Config(
        n_embd=32,
        n_head=4,
        n_inner=64,
        layer_norm_epsilon=1e-6,
        rope_base=10000.0,
    )


def test_rope_cache_matches_reference_and_reuses_storage() -> None:
    model = MossTTSLocalDepthTransformer(_test_config()).eval()
    attn = model.h[0].attn
    device = torch.device("cpu")

    attn.prepare_rope_cache(12, device, torch.float32)
    cache_ptr = attn._rope_cos_cache.data_ptr()
    cos, sin = attn._rope_cos_sin(4, device, torch.float32)

    positions = torch.arange(4, dtype=torch.float32)
    freqs = torch.einsum("s,d->sd", positions, attn.inv_freq.float())
    expected_cos = freqs.cos().repeat_interleave(2, dim=-1).view(1, 4, 1, attn.head_dim)
    expected_sin = freqs.sin().repeat_interleave(2, dim=-1).view(1, 4, 1, attn.head_dim)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)

    # Smaller slices retain the fixed allocation used by CUDA Graph.
    attn.prepare_rope_cache(6, device, torch.float32)
    assert attn._rope_cos_cache.data_ptr() == cache_ptr
