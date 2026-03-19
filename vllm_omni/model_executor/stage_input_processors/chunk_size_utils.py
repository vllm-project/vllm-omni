# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def max_ic_for_chunk_size(chunk_size: int) -> int:
    """Largest power of 2 strictly less than chunk_size."""
    if chunk_size <= 2:
        return 1
    return 1 << ((chunk_size - 1).bit_length() - 1)


def compute_dynamic_initial_chunk_size(
    active_requests: int,
    max_batch_size: int,
    max_ic: int,
) -> int:
    """Select IC from power-of-2 steps [2, 4, ..., max_ic] based on load factor.

    - Low load: small IC (faster TTFA).
    - High load: large IC (amortise decode cost).
    """
    steps: list[int] = []
    v = 2
    while v <= max_ic:
        steps.append(v)
        v <<= 1
    if not steps:
        return max(1, max_ic)
    if max_batch_size <= 0:
        return steps[0]
    load_factor = min(active_requests / max_batch_size, 1.0)
    idx = int(round(load_factor * (len(steps) - 1)))
    return steps[idx]
