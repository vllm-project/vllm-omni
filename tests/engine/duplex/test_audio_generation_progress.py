# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Accepted input is not generated input; Stage 0 still forwards to the codec.

The stage port and model output are doubles, but completion is processed by
the real manager/runner mailbox. This does not establish acoustic drain.
"""

import asyncio

import pytest
import torch

from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame, open_personaplex_harness
from vllm_omni.engine.duplex import commands
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _prefill(monkeypatch):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)


def _metadata(epoch, sequence):
    return {"meta.duplex_epoch": torch.tensor([epoch]), "meta.duplex_generated_seq": torch.tensor([sequence])}


@pytest.mark.asyncio
async def test_finished_append_reports_its_sequence_without_consuming_stage0(mocker):
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    try:
        for _ in range(3):
            await h.run(frame())
        state = h.session.audio_delivery
        assert state.accepted_seq == 3
        assert state.generated_seq == 0
        output = OmniRequestOutput(request_id=h.stage0_request_id(), finished=False)
        for sequence in (1, 1, 2, 3):
            consumed = h.deliver(
                output,
                stage_id=0,
                segment_finished=True,
                segment_token_ids=(11,),
                segment_output_metadata=_metadata(0, sequence),
            )
            assert consumed is False
            await h.settle()
            assert state.generated_seq == sequence
            assert state.projected_samples == state.completed_samples == 0
        assert not [event for event in h.events if event.type == "response.output_audio.delta"]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "unfinished",
        "missing",
        "wrong_epoch",
        "not_accepted",
        "gap",
        "wrong_request",
        "wrong_stage",
        "no_token",
        "bool_seq",
        "float_seq",
        "vector_seq",
    ],
)
async def test_incomplete_or_unrelated_output_cannot_advance_generation(case, mocker):
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    try:
        await h.run(frame())
        await h.run(frame())
        metadata = _metadata(0, 1)
        if case == "missing":
            metadata = {}
        elif case == "wrong_epoch":
            metadata = _metadata(1, 1)
        elif case == "not_accepted":
            metadata = _metadata(0, 3)
        elif case == "gap":
            metadata = _metadata(0, 2)
        elif case == "bool_seq":
            metadata["meta.duplex_generated_seq"] = torch.tensor([True])
        elif case == "float_seq":
            metadata["meta.duplex_generated_seq"] = torch.tensor([1.5])
        elif case == "vector_seq":
            metadata["meta.duplex_generated_seq"] = torch.tensor([1, 2])
        output = OmniRequestOutput(
            request_id="unrelated" if case == "wrong_request" else h.stage0_request_id(),
            finished=False,
            outputs=[],
        )
        h.deliver(
            output,
            stage_id=1 if case == "wrong_stage" else 0,
            segment_finished=case != "unfinished",
            segment_token_ids=() if case == "no_token" else (11,),
            segment_output_metadata=metadata,
        )
        await h.settle()
        assert h.session.audio_delivery.generated_seq == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_queued_old_generation_receipt_cannot_credit_new_epoch(mocker):
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    try:
        await h.run(frame())
        old_request = h.stage0_request_id()
        await h.deliver_and_settle(code2wav_output(old_request, samples=1920, text="hello"))
        h.submit(commands.CancelResponse())
        h.deliver(
            OmniRequestOutput(request_id=old_request, finished=False),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=(11,),
            segment_output_metadata=_metadata(0, 1),
        )
        await h.settle()
        await h.run(frame())
        assert h.session.audio_delivery.epoch == 1
        assert h.session.audio_delivery.generated_seq == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("submission_fails", [False, True])
async def test_generated_output_before_submission_commit_is_not_lost_or_credited_early(mocker, submission_fails):
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    started, release = asyncio.Event(), asyncio.Event()
    original_submit = h.port.submit

    async def submit(submission):
        h.deliver(
            OmniRequestOutput(request_id=submission.context.request_id, finished=False),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=(11,),
            segment_output_metadata=_metadata(0, 1),
        )
        started.set()
        await release.wait()
        if submission_fails:
            raise RuntimeError("submission failed after output")
        return await original_submit(submission)

    mocker.patch.object(h.port, "submit", new=submit)
    try:
        h.submit(frame())
        await asyncio.wait_for(started.wait(), 3)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert h.session.audio_delivery is None
        h.submit(commands.Heartbeat())

        async def wait_for_heartbeat():
            while True:
                event = await h.output_buffer.get()
                assert event is not None
                if event.type == "session.heartbeat_ack":
                    return

        await asyncio.wait_for(wait_for_heartbeat(), 1)
        assert h.session.audio_delivery is None
        assert not release.is_set()
        release.set()
        await h.settle()
        if submission_fails:
            assert h.session.audio_delivery is None
        else:
            committed_state = h.session.audio_delivery
            assert committed_state is not None
            assert committed_state.accepted_seq == 1
            assert committed_state.generated_seq == 1
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_wire_close_cancels_deferred_generation_receipt_without_waiting_for_submit(mocker):
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    h.port.submit_gate = asyncio.Event()
    try:
        h.submit(frame())
        await asyncio.wait_for(h.port.submit_started.wait(), 1)
        h.deliver(
            OmniRequestOutput(request_id=h.stage0_request_id(), finished=False),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=(11,),
            segment_output_metadata=_metadata(0, 1),
        )

        async def wait_for_deferred_receipt():
            while not any(task.get_name() == "duplex-generation-receipt" for task in h.runner._background_tasks):
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_deferred_receipt(), 1)
        pending_receipts = tuple(h.runner._background_tasks)
        h.submit(commands.CloseSession())
        events = await h.settle(timeout_s=1)
        assert any(event.type == "session.closed" for event in events)
        assert not h.port.submit_gate.is_set()
        assert not h.runner._background_tasks
        assert all(task.done() for task in pending_receipts)
        assert h.session.audio_delivery is None
    finally:
        h.port.submit_gate.set()
        await close_harness(h)
