# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Cosmos3 multiview flex path must compile once, not once per prompt.

``flex_attention`` is compiled with ``dynamic=False``, so every distinct input
shape is a fresh graph.  When UND was padded to the nearest block above each
prompt's real length, the packed key tensor changed shape per prompt-length
bucket ("tensor 'key' size mismatch at index 2") and Dynamo's default limit of
eight recompiles was reached after a handful of prompts, after which the frame
fell back to eager FlexAttention -- a ~2.7 TB score matrix at the released
11-view geometry.  UND is now padded to a fixed capacity instead.

The guard behavior does not depend on sequence length, so this runs on a tiny
layout and needs no checkpoint; it needs CUDA only because the Triton path is
CUDA-only.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

pytestmark.append(
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the multiview Triton flex path is CUDA-only")
)

# Stand-in for the released layout: 8 latent frames over 2 views with 4x4
# patches is 256 GEN tokens instead of 205,920, through the same code path.
_NUM_VIEWS = 2
_LATENT_FRAMES = 8
_PATCH = 4
_HEADS, _KV_HEADS, _HEAD_DIM = 8, 2, 128
_MAX_UND = 512


def _run_generation(module, layout, q, k, v, und_len: int) -> None:
    """One request: reset_cache() cleared the mask cache, then two CFG branches."""
    context = module.MultiviewAttentionContext(layout, {}, {})
    for branch_len in (und_len, und_len + 8):
        k_und = torch.randn(1, branch_len, _KV_HEADS, _HEAD_DIM, device=q.device, dtype=q.dtype)
        module.padded_multiview_flex_attention(q, k, v, k_und, torch.randn_like(k_und), context)


def test_varying_prompt_lengths_do_not_recompile_the_flex_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._dynamo
    from torch._dynamo.testing import CompileCounterWithBackend

    import vllm_omni.diffusion.models.cosmos3.multiview_flex_attention as module

    layout = module.MultiviewLayout(
        num_views=_NUM_VIEWS,
        latent_frames=_LATENT_FRAMES,
        patch_height=_PATCH,
        patch_width=_PATCH,
        condition_frame_indexes=(0, _LATENT_FRAMES // 2),
        max_und_tokens=_MAX_UND,
    )
    device, dtype = torch.device("cuda"), torch.bfloat16
    q = torch.randn(1, layout.gen_tokens, _HEADS, _HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(1, layout.gen_tokens, _KV_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn_like(k)

    # The library assigns this global lazily on first use, so presetting it
    # counts compiles without touching library code.
    counter = CompileCounterWithBackend("inductor")
    monkeypatch.setattr(
        module,
        "_compiled_flex_attention",
        torch.compile(module.torch_flex_attention, dynamic=False, backend=counter),
    )
    torch._dynamo.reset()
    counter.clear()

    # Prompt lengths spanning several 64-token blocks, i.e. what a real serving
    # mix looks like. Before the fix this produced one graph per length.
    for und_len in (100, 200, 300, 400, 500):
        _run_generation(module, layout, q, k, v, und_len)

    assert counter.frame_count == 1, (
        f"prompt length is still a recompile trigger: {counter.frame_count} graphs for 5 prompt lengths"
    )
