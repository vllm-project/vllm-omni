# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shape/hardware launch budget for the official SM90 FA3 kernels."""


def split_budget(query_rows, heads, sm_count):
    """Supply at least two CTA waves before FA3's dynamic KV-length clipping.

    The supported SM90 head_dim128 kernel uses a128-row query tile. The
    upper bound depends only on shape/hardware, so graph topology stays fixed
    as history grows. The same budget is used for external ragged KV partitions.
    No resolution, elapsed-seconds or measured per-workload table is used.
    """
    if min(query_rows, heads, sm_count) < 1:
        raise ValueError("Positive query/head/SM geometry required")
    tasks = heads * ((query_rows + 127) // 128)
    needed = max(1, (2 * sm_count + tasks - 1) // tasks)
    return min(128, 1 << (needed - 1).bit_length())


def query_partition(query_rows):
    """Use a compact tail only when it occupies less than a quarter query tile.

    All other query rows stay together in official FA3. This removes the
    workload-specific128+tail boundary; the split follows complete128-row tiles.
    """
    if query_rows < 1:
        raise ValueError("Positive query geometry required")
    tail = query_rows % 128
    if query_rows >= 128 and 0 < tail < 32:
        return query_rows - tail, tail
    return query_rows, 0


def fp8_split_budget(query_rows, heads, sm_count):
    """One wave of 128-row FP8 tiles, with official FA3 clipping short KV."""
    tasks = heads * ((query_rows + 127) // 128)
    ratio = max(1, sm_count // tasks)
    return min(128, 1 << (ratio.bit_length() - 1))
