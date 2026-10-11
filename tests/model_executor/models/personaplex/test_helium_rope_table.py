# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helium refuses a context longer than its RoPE table."""

import pytest

from vllm_omni.model_executor.models.personaplex.modeling_helium import check_rope_covers_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_context_within_the_rope_table_is_accepted() -> None:
    check_rope_covers_context(46_080, 46_080)
    check_rope_covers_context(4096, 46_080)


def test_context_past_the_rope_table_is_refused() -> None:
    # Raising only Stage 0's max_model_len would let a call read rotations past the table.
    with pytest.raises(ValueError, match="temporal_config.max_position_embeddings"):
        check_rope_covers_context(90_000, 46_080)
