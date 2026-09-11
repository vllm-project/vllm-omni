# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice SDPA fallback masks must not depend on the model dtype.

The previous padded attention path used an additive floating-point mask. SDPA
requires such a mask to have the same dtype as the query, and constructing it
as float32 caused half-precision OmniVoice inference to fail on CUDA.

The packed-varlen path no longer needs an additive mask. When SDPA fallback is
required, ``_attention_metadata_from_cu_seqs`` builds a boolean block mask.
Boolean SDPA masks have no dtype coupling with the query, so no mask coercion
is required for float16 or bfloat16 inference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    _attention_metadata_from_cu_seqs,
)

HALF_DTYPES = [torch.float16, torch.bfloat16]

cpu_test = pytest.mark.core_model
cuda_test = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cpu_test
@pytest.mark.cpu
def test_sdpa_fallback_mask_is_dtype_independent() -> None:
    """The fallback mask is boolean rather than an additive model-dtype mask."""
    cu_seqs = torch.tensor([0, 2, 3, 6, 6], dtype=torch.int32)

    metadata = _attention_metadata_from_cu_seqs(
        cu_seqs,
        6,
        needs_sdpa_mask=True,
    )

    assert metadata.attn_mask is not None
    assert metadata.attn_mask.dtype == torch.bool


@cuda_test
@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_sdpa_fallback_mask_needs_no_half_precision_coercion(
    dtype: torch.dtype,
) -> None:
    """Half-precision SDPA accepts the production bool mask without casting."""
    device = torch.device("cuda:0")
    seq_len = 6
    cu_seqs = torch.tensor([0, 2, 3, 6, 6], dtype=torch.int32, device=device)

    metadata = _attention_metadata_from_cu_seqs(
        cu_seqs,
        seq_len,
        needs_sdpa_mask=True,
    )

    assert metadata.attn_mask is not None
    assert metadata.attn_mask.dtype == torch.bool

    query = torch.randn(1, 2, seq_len, 8, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.inference_mode():
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=metadata.attn_mask,
        )

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
