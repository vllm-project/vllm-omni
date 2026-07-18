# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict

import pytest

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _empty_pipeline() -> DreamZeroPipeline:
    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline._states = OrderedDict()
    pipeline._max_session_states = 2
    return pipeline


def test_dreamzero_pipeline_state_is_session_keyed() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    session_b = pipeline._get_or_create_state("session-b")
    session_a.call_count = 7
    session_b.call_count = 3

    assert pipeline._get_or_create_state("session-a") is session_a
    assert pipeline._get_or_create_state("session-b") is session_b
    assert session_a.call_count == 7
    assert session_b.call_count == 3


def test_dreamzero_pipeline_state_lru_caps_retained_sessions() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    pipeline._get_or_create_state("session-b")
    assert pipeline._get_or_create_state("session-a") is session_a

    pipeline._get_or_create_state("session-c")

    assert list(pipeline._states) == ["session-a", "session-c"]
    assert "session-b" not in pipeline._states


def test_dreamzero_state_owns_no_kv_caches() -> None:
    """KV (self- and cross-attention) is engine-owned since the AR-Diffusion
    paged backend: the model-local cache accessors must be gone so nothing can
    silently bypass the engine pool."""
    state = DreamZeroState()

    for removed in ("get_kv_caches", "create_kv_caches", "update_kv_cache", "get_crossattn_caches"):
        assert not hasattr(state, removed)
