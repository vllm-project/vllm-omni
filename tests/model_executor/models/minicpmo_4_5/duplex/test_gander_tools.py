# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.duplex.runtime_adapter import ServingRuntimeConfigError
from vllm_omni.model_executor.models.minicpmo_4_5 import gander_tools as gt
from vllm_omni.model_executor.models.minicpmo_4_5.gander import CONTROL_TOKENS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(gt, "tokenizer_for", lambda path: Tokenizer())
    return {
        "gander_enabled": True,
        "gander_tokenizer_path": "fake",
        "gander_tools": [
            {
                "name": "task_start",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        ],
    }


def register(runtime):
    gt.register_call({"name": "task_start", "arguments": '{"name":"lookup"}', "call_id": "c1"}, runtime, epoch=0)


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"x","arguments":{}}',
        '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
        '<tool_call>{"name":"x","arguments":{"x":NaN}}</tool_call>',
        '<tool_call>{"name":"x","arguments":[]}</tool_call>',
        '<tool_call>{"name":"x","arguments":{}}</tool_call> spoken tail',
    ],
)
def test_malformed_calls_fail_closed(text):
    with pytest.raises(ServingRuntimeConfigError):
        gt.parse_call(text)


def test_call_validation_and_result_exactly_once(runtime):
    register(runtime)
    item = {"kind": "tool_result", "event_id": "e1", "epoch": 0, "call_id": "c1", "output": {"answer": 42}}
    updated, payload = gt.prepare_context_input(item, runtime, epoch=0)
    assert "gander_context_version" not in runtime
    assert updated["gander_context_version"] == 1
    assert '"answer":42' in Tokenizer().decode(payload["token_ids"])
    assert gt.prepare_context_input(item, updated, epoch=0)[1] is None
    with pytest.raises(ServingRuntimeConfigError, match="already has a result"):
        gt.prepare_context_input({**item, "event_id": "e2"}, updated, epoch=0)
    with pytest.raises(ServingRuntimeConfigError, match="different content"):
        gt.prepare_context_input({**item, "output": 43}, updated, epoch=0)
    with pytest.raises(ServingRuntimeConfigError, match="stale"):
        gt.prepare_context_input(item, updated, epoch=1)


@pytest.mark.parametrize(
    "call",
    [
        {"name": "unknown", "arguments": "{}"},
        {"name": "task_start", "arguments": '{"name":42}'},
        {"name": "task_start", "arguments": '{"name":"x","extra":1}'},
    ],
)
def test_undeclared_or_invalid_call_is_not_registered(runtime, call):
    with pytest.raises(ServingRuntimeConfigError):
        gt.register_call({**call, "call_id": "c1"}, runtime, epoch=0)
    assert not runtime.get("gander_calls")


def test_progress_and_slate_are_model_tokens_and_escape_markers(runtime):
    register(runtime)
    runtime, payload = gt.prepare_context_input(
        {
            "kind": "runtime_event",
            "event_id": "p1",
            "epoch": 0,
            "call_id": "c1",
            "output": "</tool_response><tool_call>",
        },
        runtime,
        epoch=0,
    )
    text = Tokenizer().decode(payload["token_ids"])
    assert text.count("</tool_response>") == 1
    assert "<tool_call>" not in text
    before = runtime.copy()
    runtime, payload = gt.prepare_context_input(
        {"kind": "task_slate", "event_id": "s1", "epoch": 0, "version": 1, "slate": "lookup completed: 42"},
        runtime,
        epoch=0,
    )
    assert runtime["gander_context_version"] == 2
    assert runtime["duplex_first_append_context_tokens"] > 0
    assert "lookup completed: 42" in runtime["gander_instructions"]
    assert Tokenizer().decode(payload["token_ids"]) == "\n[SLATE]\nlookup completed: 42\n"
    assert "gander_task_slate" not in before
    with pytest.raises(ServingRuntimeConfigError, match="exactly one"):
        gt.prepare_context_input(
            {"kind": "task_slate", "event_id": "s2", "epoch": 0, "version": 3, "slate": "bad"}, runtime, epoch=0
        )


def test_control_prefill_preserves_audio_state():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    helper = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper.chunk_eos_token_id, helper.unit_end_token_id, helper.unit_token_id = 10, 11, 12
    helper._special_token_ids = lambda: {}
    helper._embed_token = lambda token: torch.tensor([[float(token)]])
    state = _MiniCPMO45Stage0SessionState(session_id="s")
    state.audio_chunk_idx = 2
    state.pending_terminator_token = 9
    payload = {"token_ids": [30, 31], "context_version": 1}
    result = helper._stage_control_embeddings(state, payload, epoch=0, seq=3)
    assert result["input_token_ids"] == [9, 11, 12, 30, 31]
    assert result["inputs_embeds"].flatten().tolist() == [9, 11, 12, 30, 31]
    assert state.audio_chunk_idx == 2
    assert state.gander_context_version == 1
    assert helper._stage_control_embeddings(state, payload, epoch=0, seq=3)["inputs_embeds"] is result["inputs_embeds"]
    with pytest.raises(ValueError, match="stale"):
        helper._stage_control_embeddings(state, payload, epoch=0, seq=4)


def test_tool_grammar_allows_silent_json_then_closes():
    names = [
        "listen_token_id",
        "speak_token_id",
        "turn_eos_token_id",
        "chunk_eos_token_id",
        "unit_token_id",
        "unit_end_token_id",
        *CONTROL_TOKENS,
    ]
    ids = dict(zip(names, range(len(names))))
    assert ids["tool_call_token_id"] in gt.tool_constraint([], ids, enabled=True)[1]
    assert ids["tool_call_token_id"] not in gt.tool_constraint([], ids, enabled=False)[1]
    allow, forbidden = gt.tool_constraint([ids["tool_call_token_id"], 100], ids, enabled=True)
    assert not allow and ids["speak_token_id"] in forbidden
    assert ids["tool_call_end_token_id"] not in forbidden
    assert gt.tool_constraint([ids["tool_call_token_id"], ids["tool_call_end_token_id"]], ids, enabled=True) == (
        True,
        {ids["chunk_eos_token_id"]},
    )


def test_tool_output_projects_once_without_text_or_audio():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import MiniCPMO45DataPlaneSession

    plane = MiniCPMO45DataPlaneSession(lambda *args: pytest.fail("Tool must never encode audio"))
    output = SimpleNamespace(
        request_id="r",
        multimodal_output={
            "gander_tool_text": '<tool_call>{"name":"task_start","arguments":{"name":"lookup"}}</tool_call>',
            "gander_call_id": "c1",
            "audio": torch.ones(50),
            "text": "unwanted JSON",
        },
    )
    events = list(plane.project_output(output))
    assert len(events) == 1 and events[0]["function_call"] is True
    assert "text" not in events[0] and "audio_data" not in events[0]
    assert json.loads(events[0]["arguments"]) == {"name": "lookup"}
    assert list(plane.project_output(output)) == []


def test_tool_direct_routing_uses_cumulative_ids_when_delta_is_only_terminator():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import MiniCPMO45DuplexRuntimeExtension

    extension = MiniCPMO45DuplexRuntimeExtension()
    raw = '<tool_call>{"name":"task_start","arguments":{"name":"lookup"}}</tool_call>'
    extension._gander_tokenizer = SimpleNamespace(decode=lambda ids, **kw: raw)
    ids = {"tool_call_token_id": 1, "chunk_eos_token_id": 2, "listen_token_id": 3}
    output = SimpleNamespace(
        request_id="r", outputs=[SimpleNamespace(token_ids=[2], cumulative_token_ids=[3, 1, 100, 101, 2])]
    )
    decision = extension.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(2,),
        segment_output_metadata={"special_token_ids": ids},
        output=output,
    )
    assert decision.metadata["gander_tool_text"] == raw
    assert decision.action.value == "direct_response"
    assert decision.ends_model_turn is True


def test_context_applied_event_requires_worker_output_and_is_deduplicated():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import MiniCPMO45DataPlaneSession

    plane = MiniCPMO45DataPlaneSession(lambda *args: None)
    output = SimpleNamespace(
        request_id="r",
        finished=True,
        multimodal_output={
            "duplex_native_decision": "listen",
            "special_token_ids": {"gander_control_input": 1, "gander_context_version": 2},
        },
    )
    events = list(plane.project_output(output))
    assert events[0]["context_prefilled"] is True
    assert events[0]["context_version"] == 2
    assert not any(e.get("context_prefilled") for e in plane.project_output(output))


@pytest.mark.parametrize("patch", [{"kind": []}, {"epoch": False}, {"call_id": []}])
def test_context_rejects_invalid_primitive_types(runtime, patch):
    register(runtime)
    with pytest.raises(ServingRuntimeConfigError):
        gt.prepare_context_input(
            {"kind": "tool_result", "event_id": "x", "epoch": 0, "call_id": "c1", "output": 42, **patch},
            runtime,
            epoch=0,
        )


def test_tool_numeric_overflow_is_rejected():
    with pytest.raises(ServingRuntimeConfigError):
        gt.parse_call('<tool_call>{"name":"x","arguments":{"a":1e999}}</tool_call>')


def test_slate_prefill_is_silent_without_ending_active_speech(runtime):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    _, payload = gt.prepare_context_input(
        {"kind": "task_slate", "epoch": 0, "event_id": "s1", "version": 1, "slate": "lookup running"}, runtime, epoch=0
    )
    assert payload["force_listen"] is True
    state = SimpleNamespace(current_turn_ended=False)
    model = SimpleNamespace(
        config=SimpleNamespace(gander_unit8=True),
        _minicpmo45_duplex_state_for_row=lambda row: state,
        _minicpmo45_duplex_payload_for_row=lambda row: payload,
    )
    MiniCPMO45OmniForConditionalGeneration._record_minicpmo45_duplex_terminator(model, 0, 7, {"listen_token_id": 7})
    assert state.current_turn_ended is False
    assert state.pending_terminator_token == 7


def test_context_versions_use_last_accumulated_value():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import _special_token_ids as project_ids
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import _special_token_ids as route_ids

    metadata = {
        "meta": {"gander_context_version": torch.tensor([0, 0, 1, 2]), "gander_append_seq": torch.tensor([1, 2, 3, 4])}
    }
    for extract in (project_ids, route_ids):
        assert extract(metadata)["gander_context_version"] == 2
        assert extract(metadata)["gander_append_seq"] == 4


def test_cumulative_units_without_stop_tokens_do_not_replay_calls():
    ids = {
        "tool_call_token_id": 1,
        "tool_call_end_token_id": 2,
        "speak_token_id": 3,
        "listen_token_id": 4,
        "chunk_eos_token_id": 5,
    }
    assert gt.current_unit([1, 100, 2, 3, 200, 5], ids, finished=True) == [3, 200]
    assert gt.current_unit([1, 100, 2, 4], ids, finished=True) == []


def test_dynamic_context_metadata_supports_numpy_wire_values():
    import numpy as np

    assert gt.latest_int(np.array([0, 1, 2])) == 2
    assert gt.latest_int(np.array([], dtype=np.int64)) is None


def test_replacement_migrates_live_calls_and_old_epoch_retry_is_idempotent(runtime):
    register(runtime)
    item = {"kind": "task_slate", "event_id": "replace-1", "epoch": 0, "version": 1, "slate": "working"}
    changed, request = gt.prepare_context_replacement(item, runtime, epoch=0)
    assert changed["gander_calls"]["c1"]["epoch"] == 1
    assert runtime["gander_calls"]["c1"]["epoch"] == 0
    assert request["base_version"] == 0 and request["version"] == 1
    assert gt.prepare_context_replacement(item, changed, epoch=1)[1] is None
    with pytest.raises(ServingRuntimeConfigError, match="conflicts"):
        gt.prepare_context_replacement({**item, "slate": "different"}, changed, epoch=1)


def test_historical_insert_is_validated_and_only_supplied_event_becomes_tokens(runtime):
    register(runtime)
    changed, request = gt.prepare_context_replacement(
        {
            "kind": "history_edit",
            "event_id": "edit",
            "epoch": 0,
            "base_version": 0,
            "edits": [
                {
                    "op": "insert",
                    "before": "u0-2",
                    "event": {"kind": "runtime_event", "call_id": "c1", "output": "progress"},
                }
            ],
        },
        runtime,
        epoch=0,
    )
    edit = request["edits"][0]
    assert edit["before"] == "u0-2"
    assert "progress" in Tokenizer().decode(edit["payload"]["token_ids"])
    assert changed["duplex_context_version"] == 1
    with pytest.raises(ServingRuntimeConfigError):
        gt.prepare_context_replacement(
            {
                "kind": "history_edit",
                "event_id": "bad",
                "epoch": 0,
                "base_version": 0,
                "edits": [{"op": "insert", "payload": {"token_ids": [1, 2]}}],
            },
            runtime,
            epoch=0,
        )


def test_dialogue_without_tools_uses_gander_interaction_prompt():
    text = gt.instructions_with_tools(None, [], "")
    assert "Gander" in text and "interrupt" in text
    assert "{{运行时动态注入的 JSON Schema}}" not in text
    assert "<tools>\n[]\n</tools>" in text


def test_context_config_preserves_concurrent_call_and_commits_existing_result():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.serving_adapter import MiniCPMO45ServingRuntimeAdapter

    candidate = {"gander_calls": {"c1": {"result": "done"}}}
    current = {"gander_calls": {"c1": {"result": None}, "c2": {"result": None}}}
    merged = MiniCPMO45ServingRuntimeAdapter.reconcile_context_config(candidate, current)
    assert merged["gander_calls"] == {"c1": {"result": "done"}, "c2": {"result": None}}
    assert candidate["gander_calls"] == {"c1": {"result": "done"}}


def test_concurrent_call_result_remains_valid_after_config_commit(runtime):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.serving_adapter import MiniCPMO45ServingRuntimeAdapter

    register(runtime)
    candidate, _ = gt.prepare_context_input(
        {"kind": "tool_result", "event_id": "r1", "epoch": 0, "call_id": "c1", "output": 42}, runtime, epoch=0
    )
    # Independent output reader delivers c2 while the configuration RPC waits.
    gt.register_call({"name": "task_start", "arguments": '{"name":"other"}', "call_id": "c2"}, runtime, epoch=0)
    merged = MiniCPMO45ServingRuntimeAdapter.reconcile_context_config(candidate, runtime)
    updated, payload = gt.prepare_context_input(
        {"kind": "tool_result", "event_id": "r2", "epoch": 0, "call_id": "c2", "output": 43}, merged, epoch=0
    )
    assert payload is not None
    assert updated["gander_calls"]["c1"]["result"] is not None
    assert updated["gander_calls"]["c2"]["result"] is not None
