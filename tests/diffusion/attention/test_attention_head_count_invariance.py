# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Attention implementations that declare ``head_count_invariant()`` must be bit-identical under head chunking.

The declaration covers full-occupancy shapes. With few heads of a short sequence per call a kernel may split the KV
range to fill the GPU and change its reduction order, so the layer also verifies each shape once at runtime before it
keeps a head-chunked schedule.
"""

import pytest
import torch

from tests.helpers.attention_head_invariance import head_chunk_mismatches
from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.diffusion, *hardware_marks(res={"cuda": "H100"})]

HEAD_SIZE = 128


def _cudnn():
    from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl

    return CuDNNAttentionImpl(num_heads=28, head_size=HEAD_SIZE, softmax_scale=HEAD_SIZE**-0.5)


def _flashinfer(backend):
    def build():
        pytest.importorskip("flashinfer")
        from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionImpl

        return FlashInferAttentionImpl(
            num_heads=28,
            head_size=HEAD_SIZE,
            softmax_scale=HEAD_SIZE**-0.5,
            backend_kwargs={"quant": {"flashinfer_backend": backend}},
        )

    return build


def _sdpa():
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    return SDPAImpl(num_heads=28, head_size=HEAD_SIZE, softmax_scale=HEAD_SIZE**-0.5)


@pytest.mark.parametrize(
    "build",
    [_cudnn, _flashinfer("auto"), _flashinfer("fa2"), _sdpa],
    ids=["cudnn", "flashinfer-auto", "flashinfer-fa2", "sdpa"],
)
def test_declared_head_count_invariance_holds(build):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        impl = build()
    except (ImportError, RuntimeError, ValueError) as error:
        pytest.skip(f"backend unavailable here: {error}")
    if not impl.head_count_invariant():
        pytest.skip(f"{type(impl).__name__} does not declare head-count invariance on this platform")
    # MiniMax-H3's per-rank attention at Ulysses 2: 37,888 tokens, 28 heads of 128.
    mismatches = head_chunk_mismatches(
        lambda q, k, v: impl.forward_cuda(q, k, v), seq_len=37888, num_heads=28, chunk_heads=(1, 2, 4, 7, 14)
    )
    assert not mismatches, f"{type(impl).__name__} differs for {mismatches}-head calls"
