# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original training-equivalent Realization window accounting (clean + noisy media)."""


def padded_training_rows(shared_rows, block_costs):
    return (shared_rows + sum(block_costs) + 127) // 128 * 128


def budget_drop_count(shared_rows, retained_costs, current_cost, budget):
    if type(budget) is not int or budget < 128:
        raise ValueError("Realization training token budget must be an integer >=128")
    if padded_training_rows(shared_rows, [current_cost]) > budget:
        raise ValueError("Realization training token budget cannot fit one complete block")
    total = shared_rows + sum(retained_costs) + current_cost
    dropped = 0
    for cost in retained_costs:
        if (total + 127) // 128 * 128 <= budget:
            break
        total -= cost
        dropped += 1
    return dropped


def kv_history_blocks(shared_rows, media_rows, budget, *, text_rows=128):
    """Whole clean blocks fitting KV context, reserving the current query.

    Official Intent generation is capped at128 body rows. Reserve its padded
    worst case rather than changing the eviction boundary as text fluctuates.
    Training clean/noisy duplication is not part of this inference budget.
    """
    current_rows = (media_rows + text_rows + 127) // 128 * 128
    if shared_rows + current_rows > budget:
        raise ValueError("Realization KV token budget cannot fit the prefix and current block")
    return (budget - shared_rows - current_rows) // media_rows
