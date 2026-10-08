# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import base64
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import build_duplex_data_plane_prompt
from vllm_omni.model_executor.models.minicpmo_4_5.gander import REPLAY_SAMPLED_KEY
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


def test_repeated_replay_preserves_original_sampled_and_forced_listen_distinction():
    units = [prompt(1), prompt(2)]
    metadata(units[1])["payload"]["force_listen"] = True
    first = plan(units)
    second = plan([unit.prompt for unit in first.units])
    for rebuilt in [first, second]:
        assert metadata(rebuilt.units[0].prompt)["payload"][REPLAY_SAMPLED_KEY] is True
        assert metadata(rebuilt.units[1].prompt)["payload"][REPLAY_SAMPLED_KEY] is False
        assert all(metadata(unit.prompt)["payload"]["force_listen"] for unit in rebuilt.units)


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


@pytest.mark.parametrize("sampled", [True, False])
def test_replay_prefill_restores_sampled_history_once_on_retry(sampled):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    helper = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper.unit_token_id, helper.unit_end_token_id, helper.chunk_eos_token_id = 10, 11, 13
    helper.listen_token_id, helper.turn_eos_token_id = 12, 14
    helper._embed_token = lambda token: torch.tensor([[float(token)]])
    helper._special_token_ids = lambda: {}
    retained = [100] * MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
    state = _MiniCPMO45Stage0SessionState(
        session_id="s",
        context_token_ids=[1, 2],
        context_embeds=[torch.tensor([[1.0], [2.0]])],
        generated_tokens=retained.copy(),
    )
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.config = SimpleNamespace(gander_unit8=True)
    model._duplex_data_plane_helper = lambda: helper
    model._minicpmo45_duplex_session_state = lambda *args, **kwargs: state
    model.get_input_embeddings = lambda ids: ids.float().unsqueeze(-1)
    duplex = {
        "data_plane": True,
        "session_id": "s",
        "epoch": 1,
        "seq": 1,
        "payload": {
            "gander_control": True,
            "gander_replay": True,
            "force_listen": True,
            REPLAY_SAMPLED_KEY: sampled,
            "token_ids": [20],
            "context_version": 0,
            "gander_replay_output_ids": [12, 40, 13],
        },
    }
    expected = [*retained[2:], 12, 40] if sampled else retained
    for _ in range(2):
        tokens, embeds, _ = model.preprocess(torch.zeros(6, dtype=torch.long), request_id="new", duplex=duplex)
        assert tokens.tolist() == [1, 2, 10, 20, 12, 40]
        assert embeds.flatten().tolist() == tokens.tolist()
        assert state.generated_tokens == expected
        assert state.gander_unit_count == 1


def test_opening_text_unit_has_exact_slots_and_does_not_consume_audio():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    helper = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper.unit_token_id, helper.unit_end_token_id, helper.chunk_eos_token_id = 10, 11, 12
    helper._embed_token = lambda token: torch.tensor([[float(token)]])
    helper._special_token_ids = lambda: {}
    state = _MiniCPMO45Stage0SessionState(
        session_id="s", context_token_ids=[1, 2], context_embeds=[torch.tensor([[1.0], [2.0]])]
    )
    payload = {"type": "text", "token_ids": [20, 21], "context_version": 0}
    p = build_duplex_data_plane_prompt(
        request_id="r",
        fence=DuplexFence("s"),
        session_config={},
        runtime_config={"gander_enabled": True, "duplex_first_append_context_tokens": 2},
        seq=1,
        turn_seq=1,
        payload=payload,
        final=False,
    )
    result = helper._stage_control_embeddings(state, payload, epoch=0, seq=1)
    assert result["input_token_ids"] == [1, 2, 10, 20, 21]
    assert result["inputs_embeds"].flatten().tolist() == [1, 2, 10, 20, 21]
    assert len(p["prompt_token_ids"]) == result["num_input_tokens"] == 5
    assert state.audio_chunk_idx == 0 and state.audio_buffer.size == 0
    assert state.gander_unit_count == 1 and state.gander_context_version == 0
    retry = helper._stage_control_embeddings(state, payload, epoch=0, seq=1)
    assert retry["input_token_ids"] == result["input_token_ids"]
    assert state.gander_unit_count == 1
    with pytest.raises(ValueError, match="precede audio"):
        helper._stage_control_embeddings(state, payload, epoch=0, seq=2)


def test_seeded_text_is_counted_in_its_own_unit_and_not_in_the_system_prefix():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
        MiniCPMO45DuplexPlugin,
        _apply_first_append_context_tokens,
    )

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

    runtime = {"gander_enabled": True}
    _apply_first_append_context_tokens(
        runtime, tokenizer=Tokenizer(), instructions="system", initial_user_text="原中文", ref_sample_count=3200
    )
    plugin = MiniCPMO45DuplexPlugin(lambda *args: None)
    payload = plugin.initial_input_payload(runtime_config=runtime)
    assert payload is not None
    assert payload["token_ids"] == list("原中文".encode())
    assert (
        runtime["duplex_first_append_context_tokens"]
        == len(b"<|im_start|>system\nsystem\n<|audio_start|><|audio_end|><|im_end|>") + 2
    )
    assert plugin.initial_input_payload(runtime_config={"initial_user_text": "MiniCPM"}) is None


def test_text_replay_keeps_its_tokens_without_reseeding_dropped_opening_text():
    seed = prompt(1)
    metadata(seed)["payload"] = {"type": "text", "token_ids": [51, 52], "context_version": 0}
    result = plan([seed, prompt(2)])
    assert metadata(result.units[0].prompt)["payload"]["token_ids"] == [51, 52]
    assert len(result.units[0].prompt["prompt_token_ids"]) == 12  # prefix 7 + unit 1 + text 2 + history 2
    result = plan([seed, prompt(2)], [{"op": "delete", "unit_id": "u0-1"}])
    assert result.retained_unit_ids == ("u0-2",)
    assert metadata(result.units[0].prompt)["payload"].get("type") != "text"
    assert "token_ids" not in metadata(result.units[0].prompt)["payload"]


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


def test_pins_in_recent_suffix_do_not_reduce_retention():
    units = [prompt(i) for i in range(128)]
    for p in units[-16:]:
        metadata(p)["gander_pinned"] = True
    selected = select_units(units, {}, compact=True)
    assert [unit_id(p) for p in selected] == [unit_id(p) for p in units[-96:]]


def test_pin_only_edit_preserves_history_within_capacity():
    units = [prompt(i) for i in range(4)]
    metadata(units[-1])["gander_pinned"] = True
    selected = select_units(units, {"gander_history": {"max_units": 4, "retain_units": 3}})
    assert [unit_id(p) for p in selected] == [unit_id(p) for p in units]


@pytest.fixture
async def context_budget_history():
    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex.session.context_history import DuplexContextHistory
    from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import GanderContextPolicy

    h = await open_harness()
    try:
        yield DuplexContextHistory(
            h.runner.ctx,
            GanderContextPolicy(),
            out=h.runner.out,
            model=h.runner.model,
            max_tokens=40960,
            wait_for_append_tail=h.runner._wait_for_append_tail,
            close_from_runtime=h.runner._close_from_runtime,
        )
    finally:
        await close_harness(h)


@pytest.mark.parametrize("output_length", [0, 1, 250])
@pytest.mark.parametrize("rollover", [False, True])
def test_replay_budget_counts_output_once_across_repeated_edits(output_length, rollover, context_budget_history):
    from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import GanderContextPolicy

    runtime = {"gander_enabled": True, "duplex_first_append_context_tokens": 100}
    units = []
    for seq in range(1, 129 if rollover else 97):
        p = build_duplex_data_plane_prompt(
            request_id="old",
            fence=DuplexFence("s"),
            session_config={},
            runtime_config=runtime,
            seq=seq,
            turn_seq=seq,
            payload={"audio": base64.b64encode(bytes(64000)).decode(), "format": "pcm_f32le", "sample_rate_hz": 16000},
            final=False,
        )
        metadata(p)["gander_output_ids"] = [100] * output_length
        units.append(p)
    policy = GanderContextPolicy()
    for epoch in (1, 2):
        context: dict[str, object] = (
            {"reason": "context_rollover"}
            if rollover and epoch == 1
            else {"edits": [{"op": "pin", "unit_id": unit_id(units[0])}]}
        )
        rebuilt = make_plan(
            prompts=units,
            runtime_config=runtime,
            session_config={},
            request_id="new",
            fence=DuplexFence("s", epoch=epoch),
            context=context,
        )
        replay = [dict(unit.prompt) for unit in rebuilt.units]
        assert len(replay) == 96
        # Every replayed unit embeds max(0, N-1) outputs in its prompt and
        # samples exactly one terminal token (even when N == 0).
        physical = sum(len(p["prompt_token_ids"]) + 1 for p in replay)
        assert sum(policy.token_count(p) for p in replay) == physical
        history = context_budget_history
        history.max_tokens = physical + 1
        history.check_budget(replay)
        if output_length:
            completed = deepcopy(replay)
            for p in completed:
                metadata(p)["gander_output_ids"] = [7]
            assert sum(policy.token_count(p) for p in completed) == physical
        history.max_tokens = physical
        with pytest.raises(ValueError, match="budget"):
            history.check_budget(replay)
        assert physical < 40960
        units = replay
