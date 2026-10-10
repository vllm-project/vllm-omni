# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import base64
import struct
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from tools.lychee_session_lifecycle_probe import run_probe
from vllm_omni.engine.duplex.plugin import load_duplex_plugin
from vllm_omni.model_executor.models.lychee_fd.duplex.capabilities import (
    lychee_native_capabilities,
)
from vllm_omni.model_executor.models.lychee_fd.duplex.data_plane import (
    LycheeDataPlaneContext,
    LycheeDataPlaneSession,
)
from vllm_omni.model_executor.models.lychee_fd.duplex.input import (
    TICKS_PER_WINDOW,
    LycheePcmAppendBuffer,
)
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _wait_for_append_tasks(harness, count):
    # Input conversion uses a thread pool; await observed task creation rather
    # than assuming a fixed number of event-loop turns runs that worker.
    async def queued():
        while len(harness.runner.tasks.append_tasks) < count:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(queued(), timeout=3.0)


def test_plugin_loads_through_pipeline_dotted_path() -> None:
    plugin = load_duplex_plugin(
        "vllm_omni.model_executor.models.lychee_fd.duplex.plugin.LycheeDuplexPlugin",
        lambda *args: None,
    )
    assert isinstance(plugin, LycheeDuplexPlugin)


def test_capabilities_advertise_session_api_and_keep_unproved_paths_disabled() -> None:
    capabilities = lychee_native_capabilities(max_sessions=8)
    assert capabilities.chunk_period_ms == 400
    assert capabilities.supports_input_append is True
    assert capabilities.supports_model_internal_state is True
    assert capabilities.supports_stage_resumption is True
    assert capabilities.supports_core_resumable_request is True
    assert capabilities.requires_model_runner_kv is True
    assert capabilities.requires_native_stage_role is True
    assert capabilities.supports_multi_session is True
    assert capabilities.supports_session_resume is True
    assert capabilities.supports_barge_in is True
    assert capabilities.supports_realtime_endpoint is True
    assert capabilities.supports_playback_ack is True
    assert capabilities.supports_audio_truncate is False
    assert capabilities.supports_chat_completions is False
    assert lychee_native_capabilities(max_sessions=1).supports_multi_session is False


def test_pcm_ledger_parks_short_audio_and_rolls_back_reserved_window() -> None:
    buffer = LycheePcmAppendBuffer()
    half = base64.b64encode(struct.pack("<3200f", *([0.05] * 3200))).decode("ascii")
    payload = {"audio": half, "format": "pcm_f32le", "sample_rate_hz": 16000}
    assert buffer.prepare_append(payload, operation_id="a", chunk_period_ms=400, allow_emit=True) is None
    reservation = buffer.prepare_append(payload, operation_id="b", chunk_period_ms=400, allow_emit=True)
    assert reservation is not None and reservation.payload is not None
    assert buffer.pending_byte_count == 0
    assert reservation.payload["lychee_audio_ledger"]["consumable_tick_end"] == TICKS_PER_WINDOW
    reservation.rollback()
    assert buffer.pending_byte_count == 6400 * 4
    assert buffer.consumable_tick_end == 0


def test_pcm_ledger_rejects_wrong_rate_and_pads_only_the_committed_tail() -> None:
    buffer = LycheePcmAppendBuffer()
    audio = base64.b64encode(struct.pack("<1000f", *([0.05] * 1000))).decode("ascii")
    with pytest.raises(ValueError, match="16000 Hz"):
        buffer.prepare_append(
            {"audio": audio, "format": "pcm_f32le", "sample_rate_hz": 24000},
            operation_id="wrong-rate",
            chunk_period_ms=400,
            allow_emit=True,
        )

    assert (
        buffer.prepare_append(
            {"audio": audio, "format": "pcm_f32le", "sample_rate_hz": 16000},
            operation_id="tail",
            chunk_period_ms=400,
            allow_emit=True,
        )
        is None
    )
    reservation = buffer.prepare_commit(operation_id="commit", chunk_period_ms=400)
    assert reservation.payload is not None
    assert len(base64.b64decode(reservation.payload["audio"])) == 6400 * 4
    ledger = reservation.payload["lychee_audio_ledger"]
    assert ledger["window_sample_end"] == 1000
    assert ledger["padded_sample_end"] == 6400


def test_plugin_rejects_client_owned_runtime_coordinates() -> None:
    plugin = LycheeDuplexPlugin(lambda *args: None)
    with pytest.raises(ValueError, match="server-owned"):
        plugin.validate_client_extra_body({"lychee_audio_window_ms": 200})


def test_plugin_runs_exactly_ten_ticks_per_audio_window() -> None:
    plugin = LycheeDuplexPlugin(lambda *args: None)
    (configured,) = plugin.configure_sampling_params(
        runtime_config={},
        defaults=(SamplingParams(max_tokens=32),),
    )
    assert configured.max_tokens == 10
    assert configured.min_tokens == 10
    assert configured.ignore_eos is True


@pytest.mark.parametrize(
    ("control_token", "expected"),
    [
        (158_354, {"is_listen": True, "preserve_request": True}),
        (158_352, {"model_speak": True, "model_backchannel": False}),
        (158_362, {"model_speak": True, "model_backchannel": True}),
    ],
)
def test_data_plane_projects_typed_control_decisions_once(control_token, expected) -> None:
    plane = LycheeDataPlaneSession()
    request_id = "duplex-s.cHJvYmU.e.0.r.stage0"
    payload = SimpleNamespace(
        tensors={
            "lychee_tick": torch.tensor([9], dtype=torch.int32),
            "lychee_control_token_ids": torch.tensor([control_token], dtype=torch.int32),
            "lychee_audio_window_seq": torch.tensor([1], dtype=torch.int32),
        }
    )
    output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(multimodal_output=payload)],
    )
    context = LycheeDataPlaneContext(
        epoch=0,
        turn_id=3,
        active_response_turn_id=None,
        active_response_id=None,
        auto_responds=True,
        response_format="wav",
        speed=None,
        modalities=("audio",),
    )

    (event,) = tuple(plane.project({"data_plane_outputs": [output]}, context=context))

    assert event["data_plane_request_id"] == request_id
    assert event["model_turn_id"] == (3 if control_token == 158354 else None)
    assert event["model_response_number"] == (0 if control_token == 158354 else 1)
    assert event["lychee_tick"] == 9
    assert event["lychee_audio_window_seq"] == 1
    for key, value in expected.items():
        assert event[key] is value
    assert tuple(plane.project({"data_plane_outputs": [output]}, context=context)) == ()


def test_session_lifecycle_probe_open_append_wait_cancel_close() -> None:
    report = asyncio.run(run_probe())
    assert report["passed"] is True
    assert report["stage0_reserved"] == 1
    assert report["parked_submissions"] == 0
    assert report["whole_window_submissions"] == 1
    assert report["cancel_epoch"] == 1
    assert report["close_events"] == ["session.closed"]


def test_data_plane_cumulative_resumable_windows_emit_each_decision_once() -> None:
    plane = LycheeDataPlaneSession()
    payload = SimpleNamespace(
        tensors={
            "lychee_tick": torch.tensor([8, 9, 10, 19]),
            "lychee_control_token_ids": torch.tensor([158357, 158354, 158357, 158352]),
            "lychee_audio_window_seq": torch.tensor([1, 1, 2, 2]),
        }
    )
    output = SimpleNamespace(
        request_id="continuous", finished=False, outputs=[SimpleNamespace(multimodal_output=payload)]
    )
    events = tuple(plane.project({"data_plane_outputs": [output]}))
    assert [event["lychee_tick"] for event in events] == [9, 19]
    assert [event["lychee_audio_window_seq"] for event in events] == [1, 2]
    assert events[0]["is_listen"] is True
    assert events[1]["model_speak"] is True
    assert tuple(plane.project({"data_plane_outputs": [output]})) == ()
    plane.mark_terminal("continuous")
    assert tuple(plane.project({"data_plane_outputs": [output]})) == ()
    plane.close_stream("continuous")
    plane.begin_request("continuous")
    assert len(tuple(plane.project({"data_plane_outputs": [output]}))) == 2


@pytest.mark.parametrize("abort_fails_once", [False, True])
@pytest.mark.parametrize("existing_binding", [False, True])
@pytest.mark.parametrize("failure", ["commit_before", "commit_after", "result_request_id", "result_stage_id"])
def test_accepted_append_failure_retires_worker_and_consumes_pcm_once(existing_binding, failure, abort_fails_once):
    from tools.lychee_session_lifecycle_probe import SYSTEM_PREFIX, _append, _open

    async def exercise():
        harness = await _open()
        plugin = harness.runner.plugin
        original_submit = harness.port.submit
        original_cleanup = harness.port.cleanup
        original_commit = plugin.commit_append_plan
        accepted_workers = set()
        cleanup_attempts = []
        inject = False

        async def submit(submission):
            owner = submission.context.request_id
            if submission.already_submitted:
                assert owner in accepted_workers, "update targeted an aborted worker"
            else:
                assert owner not in accepted_workers, "new binding reused a live worker"
            accepted_workers.add(owner)
            result = await original_submit(submission)
            if inject and failure == "result_request_id":
                return SimpleNamespace(request_id="unrelated-session-owner", stage_id=0, replica_id=0)
            if inject and failure == "result_stage_id":
                return SimpleNamespace(request_id=owner, stage_id=999, replica_id=0)
            return result

        async def cleanup(request_ids, *, abort=False):
            cleanup_attempts.append((list(request_ids), abort))
            if abort_fails_once and len(cleanup_attempts) == 1:
                assert abort
                assert all(resource.submitted for resource in harness.runner.session.request_resources.values())
                raise RuntimeError("injected first accepted-owner abort failure")
            await original_cleanup(request_ids, abort=abort)
            if abort:
                accepted_workers.difference_update(request_ids)

        def commit(**kwargs):
            if inject and failure == "commit_before":
                raise RuntimeError("injected accepted commit before plugin mutation")
            original_commit(**kwargs)
            if inject and failure == "commit_after":
                raise RuntimeError("injected accepted commit after plugin mutation")

        harness.port.submit = submit
        harness.port.cleanup = cleanup
        plugin.commit_append_plan = commit
        try:
            if existing_binding:
                await harness.run(_append(6400))
            # A separately bound downstream stage is another known owner that
            # compensation must retire. The adapter's returned foreign owner
            # and stage id remain outside this cleanup set.
            session = harness.runner.session
            downstream_owner = harness.manager.stage_request_id(session.fence, stage_id=1)
            session.bind_stage_request(1, downstream_owner, fence=session.fence)
            accepted_workers.add(downstream_owner)
            inject = True
            events = await harness.run(_append(6400))
            session = harness.runner.session
            failed_submission = harness.port.submissions[-1]
            failed_owner = failed_submission.context.request_id
            failed_bridge = failed_submission.prompt["model_intermediate_buffer"]["duplex"]
            assert ("lychee_audio_delta" in failed_bridge) is existing_binding
            assert ("lychee_history" in failed_bridge) is not existing_binding
            if existing_binding:
                assert failed_bridge["lychee_audio_delta"]["audio_window_seq"] == 2
            if abort_fails_once:
                assert len(cleanup_attempts) >= 2
                assert all(abort and set(ids) == {failed_owner, downstream_owner} for ids, abort in cleanup_attempts)
                assert accepted_workers == set()
                assert not session.request_resources
                assert session.session_id not in harness.manager.runners
                assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
                assert any(event.type == "error" for event in events)
                assert all("unrelated-session-owner" not in ids for ids, _ in cleanup_attempts)
                return
            history = plugin.histories[session.session_id]
            assert session.epoch == 1 and session.input_seq == 0
            assert session.active_request_id is None
            assert not session.request_resources
            assert history.request_ids == set()
            assert len(history.audio_windows) == (2 if existing_binding else 1)
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
            assert accepted_workers == set()
            cleanup_ids, aborted = harness.port.cleanups[-1]
            assert aborted and set(cleanup_ids) == {failed_owner, downstream_owner}
            assert plugin.data_plane.is_terminal(failed_owner)
            assert any(event.type == "error" for event in events)
            assert any(event.type == "session.updated" for event in events)
            assert all("unrelated-session-owner" not in ids for ids, _ in harness.port.cleanups)

            # A late old-owner error/output cannot recreate its worker/binding.
            harness.runner.on_stage_request_error(
                0,
                "Lychee transaction aborted; rebuild required; stale",
                request_id=failed_owner,
                expected_epoch=0,
            )
            assert await harness.settle() == []
            inject = False
            await harness.run(_append(3200))
            assert len(history.audio_windows) == (2 if existing_binding else 1)
            await harness.run(_append(3200))
            rebuilt = harness.port.submissions[-1]
            assert rebuilt.context.request_id != failed_owner
            assert rebuilt.already_submitted is False
            assert rebuilt.prompt["prompt_token_ids"] == SYSTEM_PREFIX + [158358]
            bridge = rebuilt.prompt["model_intermediate_buffer"]["duplex"]
            assert "lychee_audio_delta" not in bridge
            windows = bridge["lychee_history"]["audio_windows"]
            assert [window["start_tick"] for window in windows] == ([0, 10, 20] if existing_binding else [0, 10])
            assert rebuilt.context.stage_sampling_params.max_tokens == (29 if existing_binding else 19)
            assert history.request_ids == {rebuilt.context.request_id}
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
            assert accepted_workers == {rebuilt.context.request_id}
            # Only the newly accepted binding can resume with a delta again.
            await harness.run(_append(6400))
            resumed = harness.port.submissions[-1]
            assert resumed.already_submitted is True
            resumed_bridge = resumed.prompt["model_intermediate_buffer"]["duplex"]
            assert "lychee_history" not in resumed_bridge
            delta = resumed_bridge["lychee_audio_delta"]
            assert delta["request_id"] == rebuilt.context.request_id
            assert delta["session_epoch"] == delta["execution_epoch"] == 1
            assert delta["op_seq"] == bridge["seq"] + 1
            assert delta["audio_window_seq"] == (4 if existing_binding else 3)
            assert resumed.context.stage_sampling_params.max_tokens == 10
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_accepted_append_cancel_race_aborts_only_old_epoch_owner():
    from tools.lychee_session_lifecycle_probe import _append, _open

    async def exercise():
        harness = await _open()
        original_submit = harness.port.submit
        next_owner = None

        async def accepted_then_epoch_advanced(submission):
            nonlocal next_owner
            result = await original_submit(submission)
            session = harness.runner.session
            session.barge_in()
            next_owner = harness.manager.stage_request_id(session.fence, stage_id=0)
            session.bind_stage_request(0, next_owner, fence=session.fence)
            session.bind_request(next_owner)
            return result

        harness.port.submit = accepted_then_epoch_advanced
        try:
            await harness.run(_append(6400))
            session = harness.runner.session
            old_owner = harness.port.submissions[-1].context.request_id
            assert session.epoch == 1
            assert session.active_request_id == next_owner
            assert session.stage_request_submitted(0, next_owner)
            assert not session.stage_request_submitted(0, old_owner)
            assert harness.port.cleanups == [([old_owner], True)]
            assert harness.runner.plugin.data_plane.is_terminal(old_owner)
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_accepted_append_abort_failure_preserves_resource_until_close_retries():
    from tools.lychee_session_lifecycle_probe import _append, _open

    async def exercise():
        harness = await _open()
        original_cleanup = harness.port.cleanup
        original_commit = harness.runner.plugin.commit_append_plan
        attempts = []

        def accepted_commit_failed(**kwargs):
            original_commit(**kwargs)
            raise RuntimeError("injected accepted commit failure")

        async def cleanup(request_ids, *, abort=False):
            attempts.append((list(request_ids), abort))
            if len(attempts) == 1:
                assert abort
                assert set(request_ids).issubset(harness.runner.session.resource_request_ids())
                raise RuntimeError("injected first abort failure")
            return await original_cleanup(request_ids, abort=abort)

        harness.runner.plugin.commit_append_plan = accepted_commit_failed
        harness.port.cleanup = cleanup
        try:
            events = await harness.run(_append(6400))
            await harness.settle()
            old_owner = harness.port.submissions[-1].context.request_id
            assert len(attempts) >= 2
            assert all(old_owner in ids and abort for ids, abort in attempts)
            assert not harness.runner.session.request_resources
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
            assert any(event.type == "error" for event in events)
            assert harness.runner.session.session_id not in harness.manager.runners
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_accepted_append_recovery_rolls_back_unsubmitted_queued_pcm():
    from tools.lychee_session_lifecycle_probe import _append, _open

    async def exercise():
        harness = await _open()
        entered = asyncio.Event()
        release = asyncio.Event()
        original_submit = harness.port.submit
        original_commit = harness.runner.plugin.commit_append_plan
        fail = True

        async def gated_submit(submission):
            entered.set()
            await release.wait()
            return await original_submit(submission)

        def commit(**kwargs):
            if fail:
                raise RuntimeError("injected first accepted append commit failure")
            return original_commit(**kwargs)

        harness.port.submit = gated_submit
        harness.runner.plugin.commit_append_plan = commit
        try:
            harness.submit(_append(6400))
            await entered.wait()
            harness.submit(_append(6400))
            await _wait_for_append_tasks(harness, 2)
            release.set()
            await harness.settle()
            history = harness.runner.plugin.histories[harness.runner.session.session_id]
            assert harness.runner.session.epoch == 1
            assert len(harness.port.submissions) == 1
            assert len(history.audio_windows) == 1
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 6400 * 4
            assert not harness.runner.model_state.audio_buffer.has_reserved()

            fail = False
            await harness.run(_append(3200))
            rebuilt = harness.port.submissions[-1]
            assert len(harness.port.submissions) == 2
            assert rebuilt.already_submitted is False
            assert len(history.audio_windows) == 2
            assert rebuilt.context.stage_sampling_params.max_tokens == 19
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 3200 * 4
        finally:
            release.set()
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_multiple_queued_windows_survive_two_recoveries_and_explicit_cancel_clears_them():
    from tools.lychee_session_lifecycle_probe import _append, _open
    from vllm_omni.engine.duplex import commands

    async def exercise():
        harness = await _open()
        original_submit = harness.port.submit
        original_commit = harness.runner.plugin.commit_append_plan
        entered = asyncio.Event()
        release = asyncio.Event()
        fail = True

        async def gated_submit(submission):
            entered.set()
            await release.wait()
            return await original_submit(submission)

        def commit(**kwargs):
            if fail:
                raise RuntimeError("injected accepted append recovery")
            return original_commit(**kwargs)

        harness.port.submit = gated_submit
        harness.runner.plugin.commit_append_plan = commit
        try:
            # Each batch has one accepted window, with its tail waiting on
            # AppendAttempt's predecessor. A failed tail rolls back every later
            # reservation before a subsequent wire-order recovery can execute.
            for epoch, count in [(0, 4), (1, 3)]:
                entered.clear()
                release.clear()
                harness.submit(_append(6400))
                await entered.wait()
                for _ in range(count - 1):
                    harness.submit(_append(6400))
                await _wait_for_append_tasks(harness, count)
                release.set()
                await harness.settle()
                assert harness.runner.session.epoch == epoch + 1
                assert not harness.runner.tasks.append_tasks
                assert not harness.runner.model_state.audio_buffer.has_reserved()
            history = harness.runner.plugin.histories[harness.runner.session.session_id]
            assert len(harness.port.submissions) == 2
            assert len(history.audio_windows) == 2
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 5 * 6400 * 4
            assert harness.runner.model._append_recovery_epoch == 1

            fail = False
            await harness.run(_append(6400))
            assert [s.context.fence.epoch for s in harness.port.submissions] == [0, 1, 2]
            assert harness.port.submissions[-1].already_submitted is False
            assert len(history.audio_windows) == 3
            assert harness.port.submissions[-1].context.stage_sampling_params.max_tokens == 29
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 5 * 6400 * 4

            # Explicit cancel invalidates the reservation itself. Its later
            # stale-task compensation cannot restore intentionally dropped PCM.
            buffer = harness.runner.model_state.audio_buffer
            reservation = buffer.prepare_append(
                {"audio": "", "format": "pcm_f32le", "sample_rate_hz": 16000},
                operation_id="cancelled-queued-reservation",
                chunk_period_ms=400,
                allow_emit=True,
            )
            assert reservation is not None and reservation.active
            await harness.run(commands.CancelInput())
            assert not reservation.active
            reservation.rollback()
            assert buffer.pending_byte_count == 0 and not buffer.has_reserved()
            assert harness.runner.session.epoch == 3
        finally:
            release.set()
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_cancel_during_accepted_compensation_retries_abort_for_older_owner():
    from tools.lychee_session_lifecycle_probe import _append, _open
    from vllm_omni.engine.duplex import commands

    async def exercise():
        harness = await _open()
        original_cleanup = harness.port.cleanup
        original_commit = harness.runner.plugin.commit_append_plan
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()
        attempts = []

        async def cleanup(request_ids, *, abort=False):
            attempts.append((list(request_ids), abort))
            if len(attempts) == 1:
                cleanup_entered.set()
                await release_cleanup.wait()
            return await original_cleanup(request_ids, abort=abort)

        def commit(**kwargs):
            original_commit(**kwargs)
            raise RuntimeError("injected accepted failure while cancel races abort")

        harness.port.cleanup = cleanup
        harness.runner.plugin.commit_append_plan = commit
        try:
            harness.submit(_append(6400))
            await cleanup_entered.wait()
            assert harness.runner.session.epoch == 1
            old_owner = harness.port.submissions[-1].context.request_id
            await harness.run(commands.CancelInput())
            assert harness.runner.session.epoch == 2
            assert len(attempts) == 2
            assert attempts == [([old_owner], True), ([old_owner], True)]
            assert harness.port.cleanups == [([old_owner], True)]
            assert not harness.runner.session.request_resources
            assert harness.runner.plugin.histories[harness.runner.session.session_id].request_ids == set()
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
        finally:
            release_cleanup.set()
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_returned_submit_restores_cancel_released_owner_record_for_abort_retry():
    from tools.lychee_session_lifecycle_probe import _append, _open

    async def exercise():
        harness = await _open()
        original_submit = harness.port.submit
        original_cleanup = harness.port.cleanup
        workers = set()
        attempts = []

        async def submit(submission):
            owner = submission.context.request_id
            workers.add(owner)
            result = await original_submit(submission)
            session = harness.runner.session
            session.barge_in()
            session.release_resources_for_request_ids([owner])
            assert not session.request_resources
            return result

        async def cleanup(request_ids, *, abort=False):
            attempts.append((list(request_ids), abort))
            if len(attempts) == 1:
                assert all(resource.submitted for resource in harness.runner.session.request_resources.values())
                assert set(request_ids) == workers
                raise RuntimeError("injected abort failure after cancelled reservation was released")
            await original_cleanup(request_ids, abort=abort)
            if abort:
                workers.difference_update(request_ids)

        harness.port.submit = submit
        harness.port.cleanup = cleanup
        try:
            await harness.run(_append(6400))
            await harness.settle()
            owner = harness.port.submissions[-1].context.request_id
            assert len(attempts) >= 2 and all(ids == [owner] and abort for ids, abort in attempts)
            assert workers == set()
            assert not harness.runner.session.request_resources
            assert harness.runner.session.session_id not in harness.manager.runners
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())
