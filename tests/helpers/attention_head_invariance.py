# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check whether an attention callable is head-count invariant.

An attention implementation may return True from ``AttentionImpl.head_count_invariant()`` only if, on every platform
where it does, running it over all heads gives bit-identical output to running it over head chunks and concatenating.
Head-chunked schedules (the copy-engine Ulysses overlap) rely on that. Use this helper from a backend's own tests.
"""

from collections.abc import Callable, Iterable

import torch

Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def head_chunk_mismatches(
    attend: Attend,
    *,
    seq_len: int,
    num_heads: int,
    head_size: int = 128,
    chunk_heads: Iterable[int] = (1, 2),
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> list[int]:
    """Chunk sizes (heads per call) whose concatenated output differs from one call over all heads.

    ``attend`` takes and returns ``[batch, seq, heads, head_size]`` tensors.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    q, k, v = (
        torch.randn(1, seq_len, num_heads, head_size, generator=generator, device=device, dtype=dtype) for _ in range(3)
    )
    full = attend(q, k, v)
    mismatches = []
    for chunk in chunk_heads:
        if num_heads % chunk:
            raise ValueError(f"{chunk} heads per call does not divide {num_heads}")
        parts = [
            attend(*(x[:, :, start : start + chunk].contiguous() for x in (q, k, v)))
            for start in range(0, num_heads, chunk)
        ]
        if not torch.equal(torch.cat(parts, dim=2), full):
            mismatches.append(chunk)
    return mismatches
