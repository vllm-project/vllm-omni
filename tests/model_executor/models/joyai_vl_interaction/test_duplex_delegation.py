# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI delegation: the action becomes one shared tool call.

The model-side rules are tested on their own here. The session lifecycle they
plug into (admission, deadlines, duplicate and late results) is owned by
``vllm_omni.engine.duplex.session.tool_calls`` and covered by the engine tests;
the integration through the real JoyAI plugin is covered by
``test_duplex_delegation_session``.
"""

import json

import pytest

from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action, ParsedAction, parse_action
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex import delegation
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import advance_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DELEGATED = "</response> Checking the pot. </delegation> Is the pot overheated?"


def delegate_action() -> ParsedAction:
    return parse_action(DELEGATED)


def accepted(*, call_id: str = "call_1", output: str = "It is smoking.", runtime=None):
    runtime = runtime if runtime is not None else {}
    arguments = delegation.delegation_arguments(delegate_action())
    return delegation.accept_result(
        runtime,
        {"call_id": call_id, "name": delegation.DELEGATION_TOOL_NAME, "arguments": arguments, "output": output},
        accepted_at=1,
    )


def test_tool_definition_declares_only_delegate():
    tool = delegation.delegation_tool_definition()
    assert tool["type"] == "function"
    assert tool["name"] == "delegate"
    assert tool["parameters"]["required"] == ["question"]
    # A caller may extend their declaration; the shared object must not change.
    tool["parameters"]["required"].append("note")
    assert delegation.delegation_tool_definition()["parameters"]["required"] == ["question"]


def test_system_section_keeps_the_delegation_action():
    section = delegation.delegation_system_section()
    assert "</delegation>" in section
    assert "</response>" in section


def test_arguments_carry_question_and_note():
    arguments = delegation.delegation_arguments(delegate_action())
    assert json.loads(arguments) == {"note": "Checking the pot.", "question": "Is the pot overheated?"}


@pytest.mark.parametrize("raw", ["</delegation>", "<delegation>   "], ids=["close-tag", "spacing"])
def test_delegation_without_question_has_nothing_to_hand_over(raw):
    action = parse_action(raw)
    assert action.action is Action.DELEGATE
    assert delegation.delegation_arguments(action) is None


def test_delegation_without_note_omits_it():
    arguments = delegation.delegation_arguments(parse_action("</delegation> Is the pot overheated?"))
    assert json.loads(arguments) == {"question": "Is the pot overheated?"}


def test_arguments_are_json_safe_for_quotes_and_unicode():
    action = ParsedAction(Action.DELEGATE, text='He said "check".', delegated_question="温度是多少？")
    arguments = delegation.delegation_arguments(action)
    assert json.loads(arguments) == {"note": 'He said "check".', "question": "温度是多少？"}


def test_oversized_delegation_is_refused(monkeypatch):
    monkeypatch.setattr(delegation, "MAX_DELEGATION_ARGUMENT_BYTES", 16)
    assert delegation.delegation_arguments(delegate_action()) is None


def test_accepted_result_keeps_the_session_owned_question():
    runtime = accepted()
    [record] = delegation.result_records(runtime)
    assert record["call_id"] == "call_1"
    assert record["question"] == "Is the pot overheated?"
    assert record["output"] == "It is smoking."


@pytest.mark.parametrize(
    "arguments",
    [None, "", "{", '{"note": "only"}', '{"question": ""}', '["not", "an", "object"]'],
    ids=["missing", "empty", "malformed", "no-question", "blank", "list"],
)
def test_result_without_a_question_is_refused(arguments):
    with pytest.raises(ValueError):
        delegation.accept_result({}, {"call_id": "call_1", "arguments": arguments, "output": "x"}, accepted_at=1)


def test_result_requires_a_string_output():
    with pytest.raises(ValueError):
        delegation.accept_result(
            {},
            {"call_id": "call_1", "arguments": delegation.delegation_arguments(delegate_action()), "output": 3},
            accepted_at=1,
        )


def test_accepting_the_same_call_twice_replaces_the_record():
    runtime = accepted()
    runtime = accepted(output="Still smoking.", runtime=runtime)
    [record] = delegation.result_records(runtime)
    assert record["output"] == "Still smoking."


def test_result_is_cut_to_the_digest_limit():
    runtime = accepted(output="x" * (delegation.DELEGATION_RESULT_LIMIT + 50))
    [record] = delegation.result_records(runtime)
    assert len(record["output"]) == delegation.DELEGATION_RESULT_LIMIT


def test_oldest_result_is_evicted_at_the_capacity():
    runtime = {}
    for index in range(delegation.MAX_DELEGATION_RESULTS + 2):
        runtime = accepted(call_id=f"call_{index}", runtime=runtime)
    kept = [record["call_id"] for record in delegation.result_records(runtime)]
    assert len(kept) == delegation.MAX_DELEGATION_RESULTS
    assert "call_0" not in kept and f"call_{delegation.MAX_DELEGATION_RESULTS + 1}" in kept


def test_rendered_block_pairs_question_and_answer():
    block, rendered = delegation.render_result_blocks(delegation.unanswered_results(accepted()))
    assert block.startswith("[Delegated Results]")
    assert "Question: Is the pot overheated?" in block
    assert "Answer: It is smoking." in block
    assert rendered == ["call_1"]


def test_rendered_block_keeps_the_newest_result_whole():
    runtime = accepted()
    runtime = accepted(call_id="call_2", output="Second answer.", runtime=runtime)
    block, rendered = delegation.render_result_blocks(
        delegation.unanswered_results(runtime),
        window_chars=len("Question: Is the pot overheated?\nAnswer: Second answer."),
    )
    assert "Second answer." in block
    assert "It is smoking." not in block
    # The trimmed result is not in the text, so it must not be booked as shown.
    assert rendered == ["call_2"]


def test_rendered_block_is_empty_without_results():
    assert delegation.render_result_blocks([]) == ("", [])


def test_displayed_result_is_not_shown_again_in_the_same_chunk():
    runtime = accepted()
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="c0") or runtime
    assert delegation.unanswered_results(runtime, chunk="c0") == []
    assert [result.call_id for result in delegation.unanswered_results(runtime, chunk="c1")] == ["call_1"]


def test_silent_reply_leaves_the_result_for_the_next_frame():
    runtime = accepted()
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="c0") or runtime
    settled = delegation.settle_response(runtime, committed=False, chunk="c0")
    updated = {**runtime, **settled}
    assert [result.call_id for result in delegation.unanswered_results(updated, chunk="c0")] == ["call_1"]


def test_stored_reply_answers_the_result_and_allows_one_reuse_in_its_chunk():
    runtime = accepted()
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="c0") or runtime
    runtime = {**runtime, **delegation.settle_response(runtime, committed=True, chunk="c0")}
    [record] = delegation.result_records(runtime)
    assert record["answered"] == 1
    # The chunk keeps speaking for the same goal: the result may be reused once.
    assert [result.call_id for result in delegation.unanswered_results(runtime, chunk="c0")] == ["call_1"]
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="c0") or runtime
    runtime = {**runtime, **delegation.settle_response(runtime, committed=True, chunk="c0")}
    assert [record["answered"] for record in delegation.result_records(runtime)] == [2]
    assert delegation.unanswered_results(runtime, chunk="c0") == []
    assert delegation.unanswered_results(runtime, chunk="c1") == []


def test_expired_result_is_never_shown():
    runtime = accepted()
    runtime = {**runtime, **delegation.settle_response(runtime, committed=False, chunk="c0", expired=["call_1"])}
    assert delegation.expired_call_ids(runtime) == {"call_1"}
    assert delegation.unanswered_results(runtime, chunk="c0") == []


def test_cancelled_turn_expires_unanswered_results_only():
    runtime = accepted()
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="c0") or runtime
    runtime = {**runtime, **delegation.settle_response(runtime, committed=True, chunk="c0")}
    runtime = accepted(call_id="call_2", runtime=runtime)
    updated = delegation.expire_unanswered(runtime)
    assert updated is not None
    expired = {record["call_id"]: record.get("expired") for record in updated["joyai_delegation_results"]}
    assert expired == {"call_1": None, "call_2": True}


def test_reconcile_keeps_a_result_accepted_while_the_prompt_was_planned():
    planned_runtime = accepted()
    live = accepted(call_id="call_2", output="Arrived meanwhile.", runtime=planned_runtime)
    merged = delegation.reconcile_delegation_runtime(live, displayed=["call_1"], chunk="c0")
    assert [record["call_id"] for record in merged] == ["call_1", "call_2"]
    assert merged[0]["displayed_in_chunk"] == "c0"
    assert "displayed_in_chunk" not in merged[1]


def test_reconcile_without_live_results_changes_nothing():
    assert delegation.reconcile_delegation_runtime({}, displayed=["call_1"], chunk="c0") is None


def test_reconcile_does_not_roll_back_a_settle_that_ran_meanwhile():
    """The plan must not overwrite a settle that happened during planning."""
    runtime = accepted()
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="c0") or runtime
    # The response settles while the prompt is still being planned.
    runtime = {**runtime, **delegation.settle_response(runtime, committed=True, chunk="c0")}
    merged = delegation.reconcile_delegation_runtime(runtime, displayed=[], chunk="c0")
    [record] = merged
    assert record["answered"] == 1, "reconciling must not reset the settle"


def test_answered_result_is_not_reused_in_a_later_chunk():
    """The one reuse belongs to the chunk that answered the result."""
    runtime = accepted()
    runtime = delegation.mark_displayed(runtime, call_ids=["call_1"], chunk="chunk-0") or runtime
    runtime = {**runtime, **delegation.settle_response(runtime, committed=True, chunk="chunk-0")}
    [record] = delegation.result_records(runtime)
    assert record["answered_in_chunk"] == "chunk-0"
    # Reusable once inside its own chunk...
    assert [result.call_id for result in delegation.unanswered_results(runtime, chunk="chunk-0")] == ["call_1"]
    # ...but never carried into the next scene.
    assert delegation.unanswered_results(runtime, chunk="chunk-1") == []


def test_chunk_advances_after_a_spoken_reply_and_a_new_frame():
    assert advance_chunk(current=None, spoken=False, has_camera=True) == ("chunk-0", False)
    assert advance_chunk(current="chunk-0", spoken=True, has_camera=False) == ("chunk-0", True)
    assert advance_chunk(current="chunk-0", spoken=True, has_camera=True) == ("chunk-1", False)


def test_chunk_stays_while_the_scene_is_idle():
    for _ in range(3):
        assert advance_chunk(current="chunk-0", spoken=False, has_camera=True) == ("chunk-0", False)


def test_two_sessions_advance_their_chunks_independently():
    """The chunk lives in each session's runtime, not on a shared plugin slot."""
    session_a = {"joyai_chunk_id": "chunk-0", "joyai_chunk_spoken": True}
    session_b = {"joyai_chunk_id": "chunk-0", "joyai_chunk_spoken": False}

    a_chunk, a_spoken = advance_chunk(
        current=session_a["joyai_chunk_id"], spoken=session_a["joyai_chunk_spoken"], has_camera=True
    )
    b_chunk, b_spoken = advance_chunk(
        current=session_b["joyai_chunk_id"], spoken=session_b["joyai_chunk_spoken"], has_camera=True
    )

    assert (a_chunk, a_spoken) == ("chunk-1", False)
    assert (b_chunk, b_spoken) == ("chunk-0", False)
    # Session B is untouched by A's advance.
    assert session_b == {"joyai_chunk_id": "chunk-0", "joyai_chunk_spoken": False}
