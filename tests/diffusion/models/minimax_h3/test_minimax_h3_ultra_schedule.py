# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import Counter

import pytest

from vllm_omni.diffusion.models.minimax_h3.vae_batching import jobs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_paired_vae_has_no_missing_or_repeated_window_tiles():
    rounds = jobs()
    observed = Counter(job for assignments in rounds for rank in assignments for job in rank)
    assert observed == Counter({(window, tile): 1 for window in range(21) for tile in range(28)})
    assert [sum(len(assignments[rank]) for assignments in rounds) for rank in range(8)] == [74] * 4 + [73] * 4
    # Collective rounds must not expose windows out of temporal order.
    assert [sorted({window for rank in assignments for window, _ in rank}) for assignments in rounds] == [
        list(range(first, min(first + 2, 21))) for first in range(0, 21, 2)
    ]
