# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fake-plugin CPU suite driving the whole action-capture path.

The harness is the real session stack (manager, runner, model channel, MiniCPM-o
plugin with capture declared); only the stage port is fake and stage outputs are
scripted, exactly like the session-stats suite beside it. These tests pin the
Engine-contract criteria: complete, ordered, schema-valid records including
forced kinds; a spoken unit's per-step deliveries fold into one record; capture
off stamps nothing; validation failure closes the session;
retention spills into the terminal cache so a multi-turn episode never overflows;
the terminating funnel seals its status (close/cancel/interruption) and close
mid-response returns the partial set sealed.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from tests.engine.duplex.test_session_runner import (
    LISTEN_TOKEN_ID,
    SESSION_ID,
    Harness,
    append_audio,
    close_harness,
    find,
    listen_output,
    open_harness,
    types,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.protocol.duplex.action_records import (
    DuplexActionRecordSet,
    build_unit_payload,
    decode_unit_payload,
    encode_unit_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CapturePlugin(MiniCPMO45DuplexPlugin):
    """MiniCPM-o plugin with the capture capability declared for testing."""

    def capabilities(self, *, max_sessions: int):
        return replace(super().capabilities(max_sessions=max_sessions), supports_action_capture=True)

    def action_capture_stage_ids(self) -> frozenset[int]:
        return frozenset({0})


def _unit_payload(
    *,
    seq: int,
    session_id: str = SESSION_ID,
    epoch: int = 0,
    unit_index: int | None = None,
    token_ids: tuple[int, ...] = (11, LISTEN_TOKEN_ID),
    logprobs: tuple[float | None, ...] | None = None,
    kinds: tuple[str, ...] | None = None,
    reasons: tuple[str | None, ...] | None = None,
    seed: int | None = 7,
) -> bytes:
    return encode_unit_payload(
        build_unit_payload(
            session_id=session_id,
            stage_id=0,
            unit_index=seq if unit_index is None else unit_index,
            seq=seq,
            epoch=epoch,
            token_ids=list(token_ids),
            logprobs=list(logprobs or (-0.5,) * len(token_ids)),
            action_kinds=list(kinds or ("sampled",) * len(token_ids)),
            reason_codes=list(reasons) if reasons is not None else None,
            streams=[{"name": "content", "stage": 0, "modality": "audio"}],
            sampling_params={"temperature": 0.7},
            effective_seed=seed,
            model={"rep_penalty": 1.05},
        )
    )


def _turn_end_output(request_id: str) -> SimpleNamespace:
    """A finished Stage0 segment that ends the turn with no listen decision: the response ends."""
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(text="", token_ids=[11, 12], stop_reason=None, multimodal_output={})],
        multimodal_output={"turn_end": True},
    )


async def open_capture_harness(**kwargs: Any) -> Harness:
    from tests.engine.duplex.test_session_runner import _fake_encode_audio

    plugin = CapturePlugin(_fake_encode_audio)
    harness = await open_harness(plugin=plugin, action_capture=True, seed=7, **kwargs)  # type: ignore[arg-type]
    assert harness.session.capabilities.supports_action_capture is True
    assert harness.session.actions is not None
    return harness


async def drain_after_close(h: Harness) -> list[DuplexActionRecordSet]:
    """The post-close drain is the suite's only observation window."""
    await h.manager.shutdown()
    drained: list[DuplexActionRecordSet] = []
    while (set_ := h.manager.take_terminal_action_records(SESSION_ID)) is not None:
        drained.append(set_)
    return drained


async def open_response_scope_with_units(h: Harness) -> str:
    """Arrange one listening unit pre-bind plus a bound response scope with one unit."""
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(
        listen_output(request_id),
        stage_id=0,
        segment_finished=True,
        capture_payload=_unit_payload(seq=1),
    )
    response_id = h.session.begin_response(turn_id=h.session.turn_id)
    h.session.actions.ingest(
        _unit_payload(seq=2, token_ids=(12,)),
        epoch=h.session.epoch,
        turn_id=h.session.turn_id,
        response_id=response_id,
        stage_id=0,
        input_modality="audio",
    )
    return response_id


# --------------------------------------------------------------------------- #
# Capture off: nothing is stamped, nothing is ingested                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_capture_off_appends_no_envelope_and_drains_nothing_post_close() -> None:
    h = await open_harness()
    try:
        assert h.session.actions is None
        await h.run(append_audio())
        for submission in h.port.submissions:
            duplex = submission.prompt["model_intermediate_buffer"].get("duplex", {})
            assert "action_capture" not in duplex

        await h.manager.shutdown()
        assert h.manager.take_terminal_action_records(SESSION_ID) is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_capture_on_stamps_the_envelope_with_seed_and_response_index() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        stamped = [
            sub.prompt["model_intermediate_buffer"]["duplex"].get("action_capture") for sub in h.port.submissions
        ]
        assert stamped
        for entry in stamped:
            assert entry == {"enabled": True, "seed": 7, "response_index": 0}
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Records flow end to end and collection is draining                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_records_reach_retrieval_with_forced_kinds_and_origin() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        # Unit 1: a forced listen (silent unit turn ended) + a sampled token.
        await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=(11, LISTEN_TOKEN_ID),
            capture_payload=_unit_payload(
                seq=1,
                token_ids=(11, LISTEN_TOKEN_ID),
                logprobs=(-0.5, None),
                kinds=("sampled", "forced"),
                reasons=(None, "silent_unit_turn_ended"),
            ),
        )
        sets = await drain_after_close(h)
        assert len(sets) == 1
        record_set = sets[0]
        assert record_set.status == "truncated_close"
        assert record_set.epoch == 0
        assert record_set.token_ids == (11, LISTEN_TOKEN_ID)
        assert record_set.action_kinds == ("sampled", "forced")
        assert record_set.reason_codes[-1] == "silent_unit_turn_ended"
        assert record_set.effective_seed == 7
        assert record_set.model == {"rep_penalty": 1.05}
        records = list(record_set.records())
        assert len(records) == 1
        assert records[0].actions[-1].action_kind == "forced"
        # The drain emptied the terminal cache.
        assert h.manager.take_terminal_action_records(SESSION_ID) is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_decisionless_unit_delivery_reaches_the_ledger_and_still_forwards() -> None:
    """A mid-response per-unit delivery is not finished, so the plugin has no decision for it.

    It must still reach the mailbox for ingest while the output forwards to the
    next stage exactly as before; returning early dropped every per-unit
    delivery of a multi-stage pipeline.
    """
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        unit = SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[SimpleNamespace(text="", token_ids=[11, 12], multimodal_output={})],
            multimodal_output={},
        )
        forwards = h.deliver(
            unit, stage_id=0, segment_finished=False, capture_payload=_unit_payload(seq=1, token_ids=(11, 12))
        )
        assert forwards is False
        events = await h.settle()
        assert "error" not in types(events)
        sets = await drain_after_close(h)
        assert len(sets) == 1
        record_set = sets[0]
        assert (record_set.token_ids, record_set.num_units()) == ((11, 12), 1)
    finally:
        await close_harness(h)


def test_unit_payload_fixture_stamps_one_model_unit_per_seq() -> None:
    """Every scripted delivery is its own model unit unless a test passes ``unit_index``."""
    assert decode_unit_payload(_unit_payload(seq=3))["unit_index"] == 3
    assert decode_unit_payload(_unit_payload(seq=3, unit_index=1))["unit_index"] == 1


@pytest.mark.asyncio
async def test_a_spoken_units_decode_steps_fold_into_one_record() -> None:
    """A spoken model unit reaches the ledger as one delivery per decode step.

    Content, content, then the ``chunk_eos`` boundary share the unit's ``unit_index``; the
    next unit's listen opens its own record beside it.
    """
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        events = []
        # seq, sampled token, segment closed: content, content, then the chunk_eos boundary.
        for seq, token, closed in ((1, 20, False), (2, 21, False), (3, 5, True)):
            events += await h.deliver_and_settle(
                SimpleNamespace(
                    request_id=request_id,
                    finished=False,
                    outputs=[SimpleNamespace(text="", token_ids=[token], multimodal_output={})],
                    multimodal_output={},
                ),
                stage_id=0,
                segment_finished=closed,
                capture_payload=_unit_payload(seq=seq, unit_index=1, token_ids=(token,), kinds=("sampled",)),
            )
        events += await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=_unit_payload(seq=4, unit_index=2, token_ids=(LISTEN_TOKEN_ID,)),
        )
        assert "error" not in types(events), events
        [record_set] = await drain_after_close(h)
        assert record_set.token_ids == (20, 21, 5, LISTEN_TOKEN_ID)
        assert record_set.unit_offsets == (0, 3, 4)
        records = list(record_set.records())
        assert [len(record.actions) for record in records] == [3, 1]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_terminal_delivery_still_ingests_its_capture_payload() -> None:
    """The data plane marks a request terminal at turn end or drain; the stage's
    terminal poll can still deliver that response's final unit."""
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        h.session.begin_response(turn_id=h.session.turn_id)
        response_id = h.session.active_response_id
        # What the turn-end and drain paths do once a request's stream is closed.
        h.manager.plugin.data_plane.mark_terminal(request_id)
        assert h.manager.plugin.data_plane.is_terminal(request_id)
        events = await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=_unit_payload(seq=1),
        )
        assert "error" not in types(events)
        h.session.end_response(commit_text=False)
        sets = await drain_after_close(h)
        record_set = next(s for s in sets if s.response_id == response_id)
        assert (record_set.status, record_set.token_ids) == ("completed", (11, LISTEN_TOKEN_ID))
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Validation failure is session-fatal and loud                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_invalid_payload_closes_the_session_with_typed_error() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        bad = _unit_payload(seq=1)
        bad = bad.replace(b'"schema_version":1', b'"schema_version":99')
        events = await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=bad,
        )
        assert "error" in types(events)
        error = next(event for event in events if types([event]) == ["error"])
        assert error.code == "action_capture_invalid"
        assert any(event_type == "session.closed" for event_type in types(events))
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_unit_delivery_without_payload_is_fatal() -> None:
    """A non-finished delivery of a capture stage is a sampled unit: no payload is a transit loss."""
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        events = await h.deliver_and_settle(
            SimpleNamespace(
                request_id=request_id,
                finished=False,
                outputs=[SimpleNamespace(text="", token_ids=[11, 12], multimodal_output={})],
                multimodal_output={},
            ),
            stage_id=0,
            segment_finished=False,
            capture_payload=None,
        )
        assert "error" in types(events)
        error = next(event for event in events if types([event]) == ["error"])
        assert error.code == "action_capture_invalid"
        assert "without payload" in error.message
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_payload_less_non_unit_delivery_is_skipped_and_the_session_keeps_recording() -> None:
    """A poll that sampled nothing is not a transit loss, and later units still record."""
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        parked = SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[SimpleNamespace(text="", token_ids=[11], multimodal_output={})],
            multimodal_output={},
        )
        events = await h.deliver_and_settle(
            parked,
            stage_id=0,
            segment_finished=False,
            capture_payload=None,
            segment_unit_bearing=False,
        )
        assert "error" not in types(events)
        assert h.session.state.name == "OPEN"
        await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=_unit_payload(seq=1),
        )
        sets = await drain_after_close(h)
        assert sets[0].num_units() == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_payload_less_unit_bearing_delivery_stays_fatal() -> None:
    """The producer said this delivery sampled a unit; a missing payload is still fatal."""
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        events = await h.deliver_and_settle(
            SimpleNamespace(
                request_id=request_id,
                finished=False,
                outputs=[SimpleNamespace(text="", token_ids=[11], multimodal_output={})],
                multimodal_output={},
            ),
            stage_id=0,
            segment_finished=False,
            capture_payload=None,
            segment_unit_bearing=True,
        )
        assert "error" in types(events)
        error = next(event for event in events if types([event]) == ["error"])
        assert error.code == "action_capture_invalid"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_closure_without_payload_is_skipped_and_the_response_seals_completed() -> None:
    """A natural turn end samples no unit, so the closure legitimately carries no payload.

    Treating that as a transit loss killed every episode of a thinker-only
    capture session, where a response closes within seconds of its commit.
    """
    h = await open_capture_harness(auto_response=False, stage_count=1)
    try:
        await h.run(append_audio())
        events = await h.run(commands.Commit(create_response=True))
        assert "response.created" in types(events), types(events)
        request_id = h.port.submissions[-1].context.request_id
        response_id = h.session.active_response_id
        events = await h.deliver_and_settle(
            _turn_end_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=None,
        )
        assert "error" not in types(events)
        assert h.session.state.name == "OPEN"
        assert find(events, "response.done").status == "completed"
        sets = await drain_after_close(h)
        record_set = next(s for s in sets if s.response_id == response_id)
        assert record_set.status == "completed"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_non_capture_stage_deliveries_do_not_require_payloads() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        # Stage 1 (TTS) deliveries carry no capture payload and must not fail.
        await h.deliver_and_settle(
            SimpleNamespace(
                request_id=request_id,
                finished=False,
                outputs=[SimpleNamespace(text="hi", token_ids=[], multimodal_output={})],
                multimodal_output={},
            ),
            stage_id=1,
            segment_finished=True,
        )
        assert h.session.state.name == "OPEN"
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Sealing semantics                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_close_mid_response_returns_partial_set_via_terminal_cache() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=_unit_payload(seq=1),
        )
        h.session.begin_response(turn_id=h.session.turn_id)
        await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=_unit_payload(seq=2, token_ids=(12, LISTEN_TOKEN_ID)),
        )
        await h.manager.shutdown()
        sets = h.manager.take_terminal_action_records(SESSION_ID)
        assert sets is not None
        assert sets.status == "truncated_close"
        assert sets.num_units() >= 1
        # Drain the remaining retained sets; the entry then disappears.
        while h.manager.take_terminal_action_records(SESSION_ID) is not None:
            pass
        assert h.manager.take_terminal_action_records(SESSION_ID) is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_barge_in_seals_interruption_and_cancel_seals_truncated_cancel() -> None:
    def cancel_then_barge_in(session: DuplexEngineSession) -> int:
        session.capture_epoch_cause = "truncated_cancel"
        return session.barge_in()

    for seal, expected in (
        (lambda session: session.barge_in(), "truncated_interruption"),
        (cancel_then_barge_in, "truncated_cancel"),
    ):
        h = await open_capture_harness()
        try:
            await h.run(append_audio())
            request_id = h.stage0_request_id()
            await h.deliver_and_settle(
                listen_output(request_id),
                stage_id=0,
                segment_finished=True,
                capture_payload=_unit_payload(seq=1),
            )
            h.session.begin_response(turn_id=h.session.turn_id)
            # Spill the just-sealed listening scope so the interruption seal
            # below fits the one-set retention window.
            assert h.session.actions.pop_oldest_sealed() is not None
            assert h.session.actions is not None
            h.session.actions.ingest(
                _unit_payload(seq=2, token_ids=(12,)),
                epoch=h.session.epoch,
                turn_id=h.session.turn_id,
                response_id=h.session.active_response_id,
                stage_id=0,
                input_modality="audio",
            )
            # barge_in is the epoch-advance funnel every cancel path rides; the
            # delivery path is covered by the retrieval tests above.
            response_id = h.session.active_response_id
            seal(h.session)
            sets = await drain_after_close(h)
            record_set = next(s for s in sets if s.response_id == response_id)
            assert record_set.status == expected
            assert record_set.num_units() == 1
        finally:
            await close_harness(h)


@pytest.mark.asyncio
async def test_multi_turn_scopes_spill_into_the_terminal_cache_and_close_drains_every_set() -> None:
    """A live episode seals the next scope before any client collect can land.

    The engine spills sealed sets past the retention window into the manager's
    terminal cache (no `action_capture_retention_overflow`), and the terminal
    stash merges with the spills, so the post-close drain returns every set.
    """
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        all_events = []
        response_ids = []
        turn = 0
        for seq in (1, 2, 3):
            all_events += await h.deliver_and_settle(
                listen_output(request_id),
                stage_id=0,
                segment_finished=True,
                capture_payload=_unit_payload(seq=seq, token_ids=(10 + seq, LISTEN_TOKEN_ID)),
            )
            response_ids.append(h.session.begin_response(turn_id=turn))
            h.session.complete_model_turn(turn)
            h.session.end_response(commit_text=False)
            turn += 1
        # Three listen→response scope pairs sealed inside one session, no
        # collect in between: the engine spilled instead of erroring.
        assert "error" not in types(all_events)
        assert h.session.state.name == "OPEN"

        await h.manager.shutdown()
        drained: list[DuplexActionRecordSet] = []
        while (set_ := h.manager.take_terminal_action_records(SESSION_ID)) is not None:
            drained.append(set_)
        # Most recent first: per turn the listening scope spills at the
        # response's seal and the response scope spills at the next bind; the
        # terminal stash merged r3 with the mid-session spills instead of
        # replacing them.
        assert [set_.response_id for set_ in drained] == [
            response_ids[2],
            None,
            response_ids[1],
            None,
            response_ids[0],
            None,
        ]
        assert all(set_.status == "completed" for set_ in drained)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_barge_in_through_the_cancel_funnel_seals_truncated_interruption() -> None:
    """The runner's cancel funnel — not ``barge_in`` directly — terminates the

    response mid-speech, so its ``end_response`` seal carries the funnel's
    status; the later ``barge_in`` seal finds no open scope and cannot double-seal.
    """
    h = await open_capture_harness()
    try:
        response_id = await open_response_scope_with_units(h)
        events = await h.run(commands.BargeIn())
        assert "error" not in types(events)
        sets = await drain_after_close(h)
        record_set = next(s for s in sets if s.response_id == response_id)
        assert record_set.status == "truncated_interruption"
        assert record_set.num_units() == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_response_cancel_through_the_cancel_funnel_seals_truncated_cancel() -> None:
    h = await open_capture_harness()
    try:
        response_id = await open_response_scope_with_units(h)
        events = await h.run(commands.CancelResponse(event_id="evt_cancel"))
        assert "error" not in types(events)
        sets = await drain_after_close(h)
        record_set = next(s for s in sets if s.response_id == response_id)
        assert record_set.status == "truncated_cancel"
        assert record_set.num_units() == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_rejected_cancel_leaves_no_cause_for_the_next_interruption() -> None:
    """A cancel with no active response is rejected and seals nothing; the
    next barge-in must still seal ``truncated_interruption``."""
    h = await open_capture_harness()
    try:
        events = await h.run(commands.CancelResponse(event_id="evt_stray"))
        assert types(events) == ["error"]
        assert events[0].code == "response_not_active"
        response_id = await open_response_scope_with_units(h)
        events = await h.run(commands.BargeIn())
        assert "error" not in types(events)
        sets = await drain_after_close(h)
        record_set = next(s for s in sets if s.response_id == response_id)
        assert record_set.status == "truncated_interruption"
        assert record_set.num_units() == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_client_close_mid_response_seals_truncated_close_via_terminal_cache() -> None:
    """The client-close funnel cuts the response (`truncated_close`) and the

    pre-bind listening scope it spilled arrives `completed` after it.
    """
    h = await open_capture_harness()
    try:
        response_id = await open_response_scope_with_units(h)
        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="c-close-capture", session_id=SESSION_ID, reason="client_close")
        )
        await asyncio.wait_for(h.results.get(), timeout=2.0)
        await h.settle()
        drained: list[DuplexActionRecordSet] = []
        while (set_ := h.manager.take_terminal_action_records(SESSION_ID)) is not None:
            drained.append(set_)
        assert [set_.response_id is None for set_ in drained] == [False, True]
        assert (drained[0].response_id, drained[0].status) == (response_id, "truncated_close")
        assert drained[0].num_units() == 1
        assert (drained[1].response_id, drained[1].status) == (None, "completed")
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Admission                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_open_with_capture_is_rejected_when_the_plugin_does_not_support_it() -> None:
    from tests.engine.duplex.test_session_runner import _fake_encode_audio
    from vllm_omni.engine.duplex.config import DuplexSessionConfig
    from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
    from vllm_omni.engine.duplex.messages import OpenDuplexSessionMessage

    class NoCapturePlugin(MiniCPMO45DuplexPlugin):
        """A duplex model whose sampler never gained capture emission."""

        def capabilities(self, *, max_sessions: int):
            return replace(super().capabilities(max_sessions=max_sessions), supports_action_capture=False)

    plugin = NoCapturePlugin(_fake_encode_audio)
    assert plugin.capabilities(max_sessions=1).supports_action_capture is False
    h = await open_harness(plugin=plugin)
    try:
        message = OpenDuplexSessionMessage(
            control_id="c2",
            session_id="duplex-second",
            session_config=DuplexSessionConfig(action_capture=True),
            output_buffer=DuplexOutputBuffer(max_bytes=1024 * 1024, max_events=64),
        )
        await h.manager.handle(message)
        # The typed admission rejection rolled back: no runner, no ledger.
        assert "duplex-second" not in h.manager.runners
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Weight-sync fence #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sessions_with_appended_data_gate_the_weight_sync_fence() -> None:
    from vllm_omni.engine.duplex.messages import DuplexSessionError
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    h = await open_capture_harness()
    try:
        assert h.manager.open_sessions_with_appended_data() == []
        await h.run(append_audio())
        assert h.manager.open_sessions_with_appended_data() == [SESSION_ID]

        omni = DuplexOmni.__new__(DuplexOmni)
        omni.engine = SimpleNamespace(duplex_sessions_with_appended_data=h.manager.open_sessions_with_appended_data)
        with pytest.raises(DuplexSessionError, match="appended data") as exc_info:
            omni._reject_weight_sync_with_open_sessions("start_weight_update")
        assert exc_info.value.code == "weight_sync_sessions_open"

        await h.manager.shutdown()
        # Close drops the session from the fence; a pre-warm open may continue.
        assert h.manager.open_sessions_with_appended_data() == []
        omni._reject_weight_sync_with_open_sessions("start_weight_update")
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Mid-session capture toggles #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_session_update_cannot_change_action_capture_or_seed_after_open() -> None:
    h = await open_capture_harness()
    try:
        events = await h.run(commands.UpdateSession(patch={"action_capture": False}, event_id="evt-capture"))
        assert types(events) == ["error"]
        assert events[0].code == "action_capture_update_unsupported"
        assert events[0].related_event_id == "evt-capture"
        assert h.session.config.action_capture is True
        assert h.session.actions is not None
        events = await h.run(commands.UpdateSession(patch={"seed": 8}, event_id="evt-seed"))
        assert types(events) == ["error"]
        assert events[0].code == "action_capture_update_unsupported"
        assert events[0].related_event_id == "evt-seed"
        assert h.session.config.seed == 7
        events = await h.run(commands.UpdateSession(patch={"seed": "8"}, event_id="evt-seed-type"))
        assert types(events) == ["error"]
        assert events[0].code == "invalid_seed"
        assert h.session.config.seed == 7
        # Realtime clients resend the whole session object, so the current
        # values must not read as a change.
        events = await h.run(commands.UpdateSession(patch={"action_capture": True, "seed": 7}, event_id="evt-echo"))
        assert types(events) == ["session.updated"]
        assert h.session.config.action_capture is True
        assert h.session.config.seed == 7
    finally:
        await close_harness(h)
    # The other polarity: a capture-off session cannot turn capture on either.
    h = await open_harness()
    try:
        events = await h.run(commands.UpdateSession(patch={"action_capture": True}, event_id="evt-on"))
        assert types(events) == ["error"]
        assert events[0].code == "action_capture_update_unsupported"
        assert h.session.config.action_capture is False
        assert h.session.actions is None
    finally:
        await close_harness(h)
