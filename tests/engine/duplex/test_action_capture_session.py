# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fake-plugin CPU suite driving the whole action-capture path.

The harness is the real session stack (manager, runner, model channel, MiniCPM-o
plugin with capture declared); only the stage port is fake and stage outputs are
scripted, exactly like the session-stats suite beside it. These tests pin the
Engine-contract criteria: complete, ordered, schema-valid records including
forced kinds; capture off stamps nothing; validation failure closes the session;
retention overflow errors; close mid-response returns the partial set sealed.
"""

from __future__ import annotations

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
    listen_output,
    open_harness,
    types,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.protocol.duplex.action_records import (
    DuplexActionRecordSet,
    build_unit_payload,
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
            unit_index=0,
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


async def open_capture_harness(**kwargs: Any) -> Harness:
    from tests.engine.duplex.test_session_runner import _fake_encode_audio

    plugin = CapturePlugin(_fake_encode_audio)
    harness = await open_harness(plugin=plugin, action_capture=True, seed=7, **kwargs)  # type: ignore[arg-type]
    assert harness.session.capabilities.supports_action_capture is True
    assert harness.session.actions is not None
    return harness


# --------------------------------------------------------------------------- #
# Capture off: nothing is stamped, nothing is ingested                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_capture_off_appends_no_envelope_and_refuses_collection() -> None:
    h = await open_harness()
    try:
        assert h.session.actions is None
        await h.run(append_audio())
        for submission in h.port.submissions:
            duplex = submission.prompt["model_intermediate_buffer"].get("duplex", {})
            assert "action_capture" not in duplex

        events = await h.run(commands.GetSessionActionRecords(event_id="evt_records"))
        assert types(events) == ["error"]
        assert events[0].code == "action_capture_unsupported"
        assert events[0].related_event_id == "evt_records"
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
        reply_events = await h.run(commands.GetSessionActionRecords(event_id="evt_collect"))
        assert types(reply_events) == ["session.action.records"]
        reply = reply_events[0]
        assert reply.request_event_id == "evt_collect"
        record_set = DuplexActionRecordSet.from_wire(reply.records)
        assert record_set.epoch == 0
        assert record_set.token_ids == (11, LISTEN_TOKEN_ID)
        assert record_set.action_kinds == ("sampled", "forced")
        assert record_set.reason_codes[-1] == "silent_unit_turn_ended"
        assert record_set.effective_seed == 7
        assert record_set.model == {"rep_penalty": 1.05}
        records = list(record_set.records())
        assert len(records) == 1
        assert records[0].actions[-1].action_kind == "forced"

        # Collection cleared the retained set.
        events = await h.run(commands.GetSessionActionRecords(event_id="evt_again"))
        assert types(events) == ["error"]
        assert events[0].code == "action_capture_not_retained"
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
async def test_capture_delivery_without_payload_is_fatal() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        events = await h.deliver_and_settle(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            capture_payload=None,
        )
        assert "error" in types(events)
        error = next(event for event in events if types([event]) == ["error"])
        assert error.code == "action_capture_invalid"
        assert "without payload" in error.message
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
    for seal, expected in (
        (lambda session: session.barge_in(), "truncated_interruption"),
        (
            lambda session: (
                setattr(session, "capture_epoch_cause", "truncated_cancel"),
                session.barge_in(),
            ),
            "truncated_cancel",
        ),
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
            # Drain the just-sealed listening scope so the interruption seal
            # below fits the one-set retention window.
            assert h.session.actions.collect() is not None
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
            seal(h.session)
            reply_events = await h.run(commands.GetSessionActionRecords(event_id="evt_collect"))
            assert types(reply_events) == ["session.action.records"], types(reply_events)
            record_set = DuplexActionRecordSet.from_wire(reply_events[0].records)
            assert record_set.status == expected
            assert record_set.num_units() == 1
        finally:
            await close_harness(h)


@pytest.mark.asyncio
async def test_retention_overflow_errors_the_session() -> None:
    h = await open_capture_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        all_events = []
        for seq in (1, 2, 3):
            all_events += await h.deliver_and_settle(
                listen_output(request_id),
                stage_id=0,
                segment_finished=True,
                capture_payload=_unit_payload(seq=seq, token_ids=(10 + seq, LISTEN_TOKEN_ID)),
            )
            h.session.begin_response(turn_id=h.session.turn_id)
            h.session.end_response(commit_text=False)
        # Two scopes completed uncollected: the overflow is deferred by the
        # synchronous end_response and fails the session on the next delivery.
        codes = [event.code for event in all_events if types([event]) == ["error"]]
        assert "action_capture_retention_overflow" in codes
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
async def test_turning_capture_on_mid_session_creates_the_ledger_lazily() -> None:
    h = await open_harness()
    try:
        assert h.session.actions is None
        events = await h.run(commands.UpdateSession(patch={"action_capture": True}))
        assert "error" not in types(events)
        assert h.session.config.action_capture is True
        # The control path created the ledger; its first scope opens at the
        # first captured delivery.
        assert h.session.actions is not None
    finally:
        await close_harness(h)
