"""Regression tests for BAGEL's causal cache attention and SP selection."""

from vllm_omni.diffusion.attention.layer import Attention


def test_bagel_causal_cache_attention_skips_sequence_parallel():
    """The manually-managed causal cache path must stay local."""
    attention = Attention(
        num_heads=4,
        head_size=8,
        causal=True,
        softmax_scale=8**-0.5,
        num_kv_heads=2,
        skip_sequence_parallel=True,
    )
    assert attention.parallel_strategy.name == "none"
