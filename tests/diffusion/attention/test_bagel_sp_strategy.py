"""Regression test for BAGEL's causal cache attention path."""

import pytest

from vllm_omni.diffusion.attention.layer import Attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bagel_causal_cache_attention_skips_sequence_parallel():
    attention = Attention(
        num_heads=4,
        head_size=8,
        causal=True,
        softmax_scale=8**-0.5,
        num_kv_heads=2,
        skip_sequence_parallel=True,
    )
    assert attention.parallel_strategy.name == "none"
