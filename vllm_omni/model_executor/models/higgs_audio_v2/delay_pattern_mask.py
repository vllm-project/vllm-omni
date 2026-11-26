# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Apply the Higgs v2 bos/eos delay-pattern mask in one GPU launch."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mask(
    logits_ptr,
    state_ptr,
    stride_row: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_vocab: tl.constexpr,
    num_codebooks: tl.constexpr,
    vocab_size: tl.constexpr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    block: tl.constexpr,
):
    row, q = tl.program_id(0), tl.program_id(1)
    v = tl.arange(0, block)
    nd = tl.load(state_ptr + row * 2)
    rem = tl.load(state_ptr + row * 2 + 1)
    ramp = rem >= 0
    force_eos = ramp & (q < num_codebooks - rem)
    force_bos = ~ramp & (q > nd)
    content = (v != bos) & ((v != eos) | (~ramp & (q == 0)))
    allowed = tl.where(force_eos, v == eos, tl.where(force_bos, v == bos, content))
    ptr = logits_ptr + row * stride_row + q * stride_cb + v * stride_vocab
    tl.store(ptr, -float("inf"), (v < vocab_size) & ~allowed)


def mask_audio_logits_(logits: torch.Tensor, delay_states: list[tuple[int, int]], bos: int, eos: int) -> None:
    """Mask [requests, codebooks, vocab] in place without touching sampling.

    Each state is (leading delay, remaining delays), with -1 meaning no
    trailing eos ramp. Metadata is copied once, replacing per-codebook
    indexing launches. The caller retains the existing non-CUDA path.
    """
    n, q, v = logits.shape
    if not n:
        return
    state = torch.tensor(delay_states, device=logits.device, dtype=torch.int32)
    _mask[(n, q)](
        logits,
        state,
        *logits.stride(),
        q,
        v,
        bos,
        eos,
        triton.next_power_of_2(v),
        num_warps=4,
    )
