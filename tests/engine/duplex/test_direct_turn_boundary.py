# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.contracts import DuplexOutputAction, DuplexOutputDecision
from vllm_omni.engine.orchestrator import Orchestrator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("ends_turn", [False, True])
def test_direct_control_commits_turn_before_queued_input(ends_turn):
    decision = DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE, ends_model_turn=ends_turn)
    state = SimpleNamespace(streaming=SimpleNamespace(bridge_states={"duplex": {"model_turn_id": 7}}))
    orchestrator = SimpleNamespace(
        duplex_control_plane=SimpleNamespace(decide_output=lambda *_: decision),
        _duplex_output_context=lambda *_args, **_kwargs: None,
    )
    result = Orchestrator._duplex_output_decision(orchestrator, 0, object(), state)
    assert result is decision
    # The next speech unit must be tagged with a new turn even when queued
    # before the serving layer sees the interrupt. Listen keeps the same turn.
    assert state.streaming.bridge_states["duplex"]["model_turn_id"] == 7 + int(ends_turn)
