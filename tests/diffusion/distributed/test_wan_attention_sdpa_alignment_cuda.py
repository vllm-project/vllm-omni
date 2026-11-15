# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Wan attention regression for the CUDA memory-efficient SDPA backend.

No checkpoint is needed. The installed patch must run under the memory-efficient
backend and match the math backend. A torch build that still rejects an
unpatched 1x1 query records that strideM error and continues; a build that
accepts the query does not fail this case. This is kernel coverage, not the
Cosmos3-Nano multi-rank tiled-decode acceptance run.
"""

import re

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.distributed.autoencoders.wan_sdpa_alignment import install_wan_attention_sdpa_alignment

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("height", "width"), [(1, 1), (4, 9)])
def test_real_wan_attention_memory_efficient_sdpa(dtype: torch.dtype, height: int, width: int) -> None:
    """Exercise the actual kernel, including an already-aligned tile control."""
    torch.manual_seed(0)
    # Cosmos3-Nano's Wan VAE mid-block has 1024 channels.
    block = WanAttentionBlock(1024).to(device="cuda", dtype=dtype).eval()
    hidden = torch.randn(1, 1024, 1, height, width, device="cuda", dtype=dtype)
    with torch.inference_mode():
        with sdpa_kernel(SDPBackend.MATH):
            expected = block(hidden)
        if (height, width) == (1, 1):
            # Current memory-efficient SDPA rejects this query. A later torch
            # that accepts it should not fail the regression; the patched
            # comparison below is what guards the fix.
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                try:
                    block(hidden)
                except RuntimeError as exc:
                    if re.search(r"not correctly aligned.*strideM", str(exc)) is None:
                        raise
        else:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                unpatched = block(hidden)
        install_wan_attention_sdpa_alignment(block)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            actual = block(hidden)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    if (height, width) != (1, 1):
        torch.testing.assert_close(actual, unpatched, rtol=0, atol=0)
