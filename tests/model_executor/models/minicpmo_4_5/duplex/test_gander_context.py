# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import base64
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.duplex.contracts import DuplexInputMode
from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import build_duplex_data_plane_prompt
from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import make_plan, metadata, select_units, unit_id

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def prompt(seq, *, turn=0):
    p = build_duplex_data_plane_prompt(
        request_id="old",
        fence=DuplexFence("s", turn_id=turn),
        session_config={},
        runtime_config={"gander_enabled": True},
        seq=seq,
        turn_seq=seq,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={"audio": base64.b64encode(bytes(64000)).decode(), "format": "pcm_f32le", "sample_rate_hz": 16000},
        final=False,
    )
    metadata(p)["gander_output_ids"] = [12, 100 + seq, 13]
    return p


def plan(prompts, edits=(), **context):
    return make_plan(
        prompts=prompts,
        runtime_config={
            "gander_enabled": True,
            "gander_context_version": 3,
            "gander_instructions": "new slate",
            "duplex_first_append_context_tokens": 7,
        },
        session_config={},
        request_id="new",
        fence=DuplexFence("s", epoch=1),
        context={"edits": list(edits), **context},
    )


def test_insert_and_reorder_rebuild_exact_order_without_mutating_old_history():
    units = [prompt(1), prompt(2), prompt(3)]
    original = deepcopy(units)
    result = plan(
        units,
        [
            {"op": "move", "unit_id": "u0-3", "before": "u0-1"},
            {
                "op": "insert",
                "unit_id": "external",
                "before": "u0-2",
                "payload": {"gander_control": True, "token_ids": [51, 52]},
            },
        ],
    )
    assert result.retained_unit_ids == ("u0-3", "u0-1", "external", "u0-2")
    assert units == original
    assert [metadata(u.prompt)["seq"] for u in result.units] == [1, 2, 3, 4]
    assert all(metadata(u.prompt)["fence"].epoch == 1 for u in result.units)
    assert metadata(result.units[0].prompt)["payload"]["gander_replay_output_ids"] == [12, 103, 13]
    assert len(result.units[0].prompt["prompt_token_ids"]) == 20  # prefix 7 + input 11 + output prefix 2
    assert len(result.units[2].prompt["prompt_token_ids"]) == 5  # external tokens 2 + unit boundary 3


def test_active_response_output_is_invalidated_but_closed_turn_history_remains():
    result = plan([prompt(1, turn=0), prompt(2, turn=1)], discard_turn_id=1)
    assert metadata(result.units[0].prompt)["payload"]["gander_replay_output_ids"] == [12, 101, 13]
    assert metadata(result.units[1].prompt)["payload"]["gander_replay_output_ids"] == []


def test_pin_protects_against_delete_and_window_eviction():
    units = [prompt(i) for i in range(1, 7)]
    metadata(units[0])["gander_pinned"] = True
    selected = select_units(units, {"gander_history": {"max_units": 4, "retain_units": 3}}, compact=True)
    assert [unit_id(p) for p in selected] == ["u0-1", "u0-5", "u0-6"]
    with pytest.raises(ValueError, match="pinned"):
        plan(units[:3], [{"op": "delete", "unit_id": "u0-1"}])
    result = plan(units[:3], [{"op": "unpin", "unit_id": "u0-1"}, {"op": "delete", "unit_id": "u0-1"}])
    assert "u0-1" in result.dropped_unit_ids


def test_empty_history_retains_prefix_without_fabricated_audio():
    result = plan([prompt(1)], [{"op": "delete", "unit_id": "u0-1"}])
    assert result.retained_unit_ids == ()
    p = result.units[0].prompt
    assert len(p["prompt_token_ids"]) == 8
    assert metadata(p)["payload"]["gander_prefix_seed"] is True
    assert "audio" not in metadata(p)["payload"]


def test_first_control_replay_embeds_prefix_and_preserves_audio_frontend():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    helper = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper.unit_token_id, helper.unit_end_token_id, helper.chunk_eos_token_id = 10, 11, 12
    helper._embed_token = lambda token: torch.tensor([[float(token)]])
    helper._special_token_ids = lambda: {}
    state = _MiniCPMO45Stage0SessionState(
        session_id="s",
        context_token_ids=[1, 2],
        context_embeds=[torch.tensor([[1.0], [2.0]])],
        gander_context_version=3,
    )
    result = helper._stage_control_embeddings(
        state, {"gander_replay": True, "token_ids": [20, 21], "context_version": 3}, epoch=1, seq=1
    )
    assert result["input_token_ids"] == [1, 2, 10, 20, 21]
    assert result["inputs_embeds"].flatten().tolist() == [1, 2, 10, 20, 21]
    assert state.audio_chunk_idx == 0 and state.gander_unit_count == 1
    assert state.audio_buffer.size == 0


def test_old_physical_request_cleanup_cannot_delete_replacement_state():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    new_state = object()
    model = SimpleNamespace(
        _minicpmo45_duplex_request_sessions={"old": ("old", 0), "new": ("new", 0)},
        _minicpmo45_duplex_data_plane_helper=SimpleNamespace(sessions={("old", 0): object(), ("new", 0): new_state}),
        model=SimpleNamespace(),
    )
    MiniCPMO45OmniForConditionalGeneration.on_requests_finished(model, {"old"})
    assert model._minicpmo45_duplex_data_plane_helper.sessions == {("new", 0): new_state}


def test_second_replacement_preserves_outputs_of_already_replayed_closed_turns():
    first = plan([prompt(1, turn=0)])
    # Rebuilt prompts share the active fence for engine ownership, but their
    # teacher-forced history must not become part of a new unfinished reply.
    second = plan([u.prompt for u in first.units], discard_turn_id=0)
    assert metadata(second.units[0].prompt)["gander_output_ids"] == [12, 101, 13]


def test_early_capacity_pressure_leaves_room_for_new_units():
    units = [prompt(i) for i in range(1, 21)]
    metadata(units[0])["gander_pinned"] = True
    retained = select_units(units, {}, compact=True)
    assert len(retained) == 15
    assert unit_id(retained[0]) == "u0-1"
    assert unit_id(retained[-1]) == "u0-20"
