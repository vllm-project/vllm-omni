# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
    finish_graph_prefills,
    mixed_request_is_ready,
    patch_is_valid,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunked_prefill_only_finishes_and_commits_on_last_chunk():
    state = SimpleNamespace(
        is_stopping=False,
        is_last_prefill_chunk=False,
        decode_state_ready=False,
        prefill_completed=False,
        prefill_embeds=object(),
        prefill_masks=object(),
        prompt_cache=object(),
    )
    inputs = (state.prefill_embeds, state.prefill_masks, state.prompt_cache)
    assert mixed_request_is_ready(state, True, True)
    assert not mixed_request_is_ready(state, True, False)
    assert not patch_is_valid(state, True)
    finish_graph_prefills([(state, True, None, 2)])
    assert not state.prefill_completed
    assert inputs == (state.prefill_embeds, state.prefill_masks, state.prompt_cache)
    state.is_last_prefill_chunk = True
    assert patch_is_valid(state, True)
    finish_graph_prefills([(state, True, None, 2)])
    assert state.prefill_completed and state.decode_step_count == 0
    assert (state.prefill_embeds, state.prefill_masks, state.prompt_cache) == (None, None, None)


def test_cpu_stop_latch_masks_decode_patch_commit():
    state = SimpleNamespace(is_stopping=False, decode_state_ready=True)
    assert mixed_request_is_ready(state, False, True)
    assert patch_is_valid(state, False)
    state.is_stopping = True
    assert not patch_is_valid(state, False)
