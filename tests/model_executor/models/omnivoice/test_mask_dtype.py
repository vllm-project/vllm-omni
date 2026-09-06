# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniVoice attention masks must follow the model dtype, not a hardcoded float32.

SDPA requires the additive attention mask to carry the same dtype as the query.
The generator used to materialize that mask as float32 unconditionally, in three
places (the shared per-forward mask, the CUDA-graph capture buffer, and the
replay-time normalization), so every half-precision deployment died on the first
attention call with::

    RuntimeError: invalid dtype for bias - should match query's dtype

(the exact wording depends on which SDPA backend is selected for the shape; the
math backend words the same rejection as ``attn_mask.dtype``.)

That left OmniVoice servable only in float32, even though the upstream k2-fsa
implementation runs it in float16.

Note the split between the CPU and CUDA tests below: the CPU SDPA backend
silently accepts the mismatched mask, so only the CUDA tests can pin the actual
crash. The CPU tests cover the mask contract itself, which is what the fix
changes and what a future regression would break first.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    OmniVoiceAttention,
    OmniVoiceGenerator,
    _additive_float_mask,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

HALF_DTYPES = [torch.float16, torch.bfloat16]
ALL_DTYPES = HALF_DTYPES + [torch.float32]

cpu_test = pytest.mark.core_model
cuda_test = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tiny_config() -> OmniVoiceConfig:
    """A real OmniVoiceConfig sized so the module fits in a unit test."""
    return OmniVoiceConfig(
        llm_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 64,
            "max_position_embeddings": 128,
        },
        enable_cuda_graph=False,
    )


def _inputs(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(2, 6, 32, device=device, dtype=dtype)
    bool_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool, device=device)
    bool_mask[:, :, :, 4:] = False
    rope_table = torch.zeros(6, 8, device=device, dtype=dtype)
    rope_table[:, :4] = 1.0  # cos = 1, sin = 0: no rotation, so the test is about the mask
    return hidden, bool_mask, rope_table


# --------------------------------------------------------------------------
# The mask contract (CPU)
# --------------------------------------------------------------------------


@cpu_test
@pytest.mark.cpu
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_additive_float_mask_uses_requested_dtype(dtype: torch.dtype) -> None:
    out = _additive_float_mask(torch.tensor([[True, False]]), dtype)
    assert out.dtype == dtype


@cpu_test
@pytest.mark.cpu
def test_additive_float_mask_keeps_mask_semantics() -> None:
    """Attend positions stay at 0.0 and masked ones at -inf, in half precision too."""
    out = _additive_float_mask(torch.tensor([[True, False]]), torch.float16)
    assert out[0, 0].item() == 0.0
    assert out[0, 1].item() == float("-inf")


@cpu_test
@pytest.mark.cpu
def test_additive_float_mask_requires_an_explicit_dtype() -> None:
    """The float32 default is the bug; keeping the parameter required is the fix."""
    with pytest.raises(TypeError):
        _additive_float_mask(torch.tensor([[True]]))  # type: ignore[call-arg]


@cpu_test
@pytest.mark.cpu
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_generator_model_dtype_tracks_its_weights(dtype: torch.dtype) -> None:
    """The single source of truth the three former float32 literals now defer to."""
    assert OmniVoiceGenerator(_tiny_config()).to(dtype).model_dtype == dtype


@cpu_test
@pytest.mark.cpu
@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_attention_accepts_a_model_dtype_mask(dtype: torch.dtype) -> None:
    attn = OmniVoiceAttention(_tiny_config()).to(dtype).eval()
    hidden, bool_mask, rope_table = _inputs(dtype, torch.device("cpu"))

    with torch.inference_mode():
        out = attn(hidden, rope_table, attention_mask=_additive_float_mask(bool_mask, dtype))

    assert out.shape == hidden.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# The crash itself (CUDA only — the CPU backend does not reject the mismatch)
# --------------------------------------------------------------------------


@cuda_test
@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_float32_mask_is_what_broke_half_precision(dtype: torch.dtype) -> None:
    """Pin the original failure so a float32 default cannot come back unnoticed."""
    attn = OmniVoiceAttention(_tiny_config()).to(device="cuda:0", dtype=dtype).eval()
    hidden, bool_mask, rope_table = _inputs(dtype, torch.device("cuda:0"))

    with pytest.raises(RuntimeError, match=r"(invalid dtype for bias|attn_mask)"), torch.inference_mode():
        attn(hidden, rope_table, attention_mask=_additive_float_mask(bool_mask, torch.float32))


@cuda_test
@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_real_path_forward_runs_in_half_precision(dtype: torch.dtype) -> None:
    """The regression, on the production path with the Triton norm kernels live."""
    attn = OmniVoiceAttention(_tiny_config()).to(device="cuda:0", dtype=dtype).eval()
    hidden, bool_mask, rope_table = _inputs(dtype, torch.device("cuda:0"))

    with torch.inference_mode():
        out = attn(hidden, rope_table, attention_mask=_additive_float_mask(bool_mask, dtype))

    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@cuda_test
@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_generator_forward_runs_in_half_precision(dtype: torch.dtype) -> None:
    """End-to-end over the real iterative loop, which is where the float32 mask was built.

    This is the test that fails on an unfixed tree: ``forward`` materialized the
    shared SDPA mask as float32 regardless of the weights' dtype.
    """
    device = torch.device("cuda:0")
    config = _tiny_config()
    generator = OmniVoiceGenerator(config).to(device=device, dtype=dtype).eval()

    seq_len, target_len = 12, 4
    input_ids = torch.zeros(2, config.num_audio_codebook, seq_len, dtype=torch.long, device=device)
    input_ids[:, 1:, :] = config.audio_mask_id
    audio_mask = torch.zeros(2, seq_len, dtype=torch.bool, device=device)
    audio_mask[:, seq_len - target_len :] = True
    attention_mask = torch.ones(2, 1, seq_len, seq_len, dtype=torch.bool, device=device)

    with torch.inference_mode():
        tokens = generator(
            input_ids,
            audio_mask,
            attention_mask,
            target_lens=[target_len],
            seed=0,
            num_step=2,
        )

    assert tokens.shape == (1, config.num_audio_codebook, target_len)
    assert tokens.dtype == torch.long
