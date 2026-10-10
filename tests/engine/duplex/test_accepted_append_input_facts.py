# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Accepted model inputs survive an append that cannot commit its session binding.

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.engine.duplex.test_session_runner import close_harness, open_harness
from tools.lychee_session_lifecycle_probe import _append, _open
from vllm_omni.engine.duplex.commands import CancelInput
from vllm_omni.engine.duplex.contracts import DuplexStageSubmissionResult

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["canceled", "foreign_owner", "foreign_stage", "hook_failure"])
async def test_accepted_input_facts_are_recorded_before_expected_owner_compensation(monkeypatch, failure):
    harness = await open_harness()
    try:
        session = harness.session
        plugin = harness.manager.plugin
        expected_owner = harness.stage0_request_id()
        expected_fence = session.fence
        trace = []
        recorded = []
        newer_owner = None
        original_cleanup = harness.port.cleanup
        commit = Mock(side_effect=AssertionError("An invalid accepted append committed a binding"))
        monkeypatch.setattr(plugin, "commit_append_plan", commit)

        def record(*, request_id, fence, plan):
            trace.append("record")
            assert request_id == expected_owner and fence == expected_fence
            assert session.request_resources[(0, expected_owner)].submitted
            assert plan.prompt["model_intermediate_buffer"]["request_id"] == expected_owner
            recorded.append((request_id, fence, plan))
            if failure == "hook_failure":
                raise ValueError("injected accepted input recording failure")

        async def submit(submission):
            nonlocal newer_owner
            harness.port.submissions.append(submission)
            result = DuplexStageSubmissionResult(request_id=expected_owner, stage_id=0, replica_id=0)
            if failure == "canceled":
                session.barge_in()
                session.release_resources_for_request_ids([expected_owner])
                newer_owner = harness.stage0_request_id()
                session.bind_stage_request(0, newer_owner, fence=session.fence)
                session.bind_request(newer_owner)
            elif failure == "foreign_owner":
                result = replace(result, request_id="untrusted-returned-owner")
            elif failure == "foreign_stage":
                result = replace(result, stage_id=7)
            return result

        async def cleanup(request_ids, *, abort=False):
            trace.append("cleanup")
            assert recorded and request_ids == [expected_owner] and abort
            await original_cleanup(request_ids, abort=abort)

        monkeypatch.setattr(plugin, "record_accepted_append_plan", record)
        monkeypatch.setattr(harness.port, "submit", submit)
        monkeypatch.setattr(harness.port, "cleanup", cleanup)
        result = await harness.runner.model._append_via_data_plane(
            {"type": "audio"}, final=False, operation_id="accepted-facts", expected_epoch=expected_fence.epoch
        )
        assert result == {"accepted_append_recovered": True}
        assert trace == ["record", "cleanup"]
        assert len(recorded) == 1
        commit.assert_not_called()
        assert expected_owner not in session.resource_request_ids()
        assert plugin.data_plane.is_terminal(expected_owner)
        assert not plugin.data_plane.is_terminal("untrusted-returned-owner")
        if newer_owner is not None:
            assert session.active_request_id == newer_owner
            assert newer_owner in session.resource_request_ids()
            assert not plugin.data_plane.is_terminal(newer_owner)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_existing_plugin_without_accepted_input_override_commits_normally():
    harness = await open_harness()
    try:
        result = await harness.runner.model._append_via_data_plane(
            {"type": "audio"}, final=False, operation_id="default-accepted-facts", expected_epoch=0
        )
        assert result["ok"]
        assert harness.session.stage_request_submitted(0, harness.stage0_request_id())
        assert not harness.port.cleanups
    finally:
        await close_harness(harness)


def _lychee_payload(ticks, *, epoch, speech=None):
    count = len(ticks)
    return {
        "lychee_tick": torch.tensor(ticks),
        "lychee_text_token_ids": torch.tensor([158358] * count),
        "lychee_speech_token_ids": torch.tensor(speech or [158359] * count),
        "lychee_control_token_ids": torch.tensor([158357] * count),
        "lychee_execution_epoch": torch.tensor([epoch] * count),
    }


@pytest.mark.asyncio
async def test_preack_output_cancel_ack_preserves_forced_historical_input_for_next_lychee_rebuild(monkeypatch):
    harness = await _open()
    try:
        await harness.run(_append(6400))
        session = harness.runner.session
        plugin = harness.runner.plugin
        history = plugin.histories[session.session_id]
        history.record_outputs(_lychee_payload(list(range(1, 9)), epoch=0, speech=[158359] * 7 + [151694]))
        original_owner = history.bound_request_id
        raw = deepcopy((history.raw_text, history.raw_speech, history.raw_control))
        original_frontier = len(history.ticks) - 1
        await harness.run(CancelInput())
        assert session.epoch == 1 and history.force_listen_at_frontier
        expected_owner = harness.manager.stage_request_id(session.fence, stage_id=0)
        payload = deepcopy(harness.port.submissions[0].prompt["model_intermediate_buffer"]["duplex"]["payload"])
        commit = Mock(side_effect=AssertionError("The canceled recovery committed a binding"))
        monkeypatch.setattr(plugin, "commit_append_plan", commit)

        async def submit(submission):
            harness.port.submissions.append(submission)
            snapshot = submission.prompt["model_intermediate_buffer"]["duplex"]["lychee_history"]
            assert snapshot["force_listen_at_frontier"] and snapshot["logical_ticks"][-1] == 8
            assert expected_owner not in history.request_ids
            output = SimpleNamespace(
                request_id=expected_owner,
                outputs=[SimpleNamespace(multimodal_output=_lychee_payload([9], epoch=1), finish_reason=None)],
            )
            await harness.runner.model._send_model_output_events({"data_plane_outputs": [output]}, expected_epoch=1)
            assert history.frontier_tick == 9 and history.bound_request_id == original_owner
            await harness.run(CancelInput())
            assert session.epoch == 2
            return DuplexStageSubmissionResult(request_id=expected_owner, stage_id=0, replica_id=0)

        monkeypatch.setattr(harness.port, "submit", submit)
        result = await harness.runner.model._append_via_data_plane(
            payload, final=False, operation_id="cancel-before-ack", expected_epoch=1
        )
        assert result == {"accepted_append_recovered": True}
        commit.assert_not_called()
        assert harness.port.cleanups[-1] == ([expected_owner], True)
        assert expected_owner not in history.request_ids
        assert history.bound_request_id == original_owner and history.bound_execution_epoch == 0
        assert history.force_listen_at_frontier and history.pending_eos_rebuild_tick == 8
        assert history.frontier_tick == 9
        assert tuple(channel[original_frontier] for channel in (history.text, history.speech, history.control)) == (
            history.text_pad,
            history.speech_pad,
            history.sleep,
        )
        for actual, before in zip((history.raw_text, history.raw_speech, history.raw_control), raw):
            assert actual[: len(before)] == before

        next_plan = plugin.plan_append(
            request_id=harness.manager.stage_request_id(session.fence, stage_id=0),
            fence=session.fence,
            session_config=session.config.as_dict(),
            runtime_config=dict(session.runtime_config),
            seq=0,
            turn_seq=0,
            payload=payload,
            final=False,
            sampling_params=harness.port.sampling_defaults()[0],
        )
        rebuilt = next_plan.prompt["model_intermediate_buffer"]["duplex"]["lychee_history"]
        assert rebuilt["force_listen_at_frontier"] and rebuilt["logical_ticks"][-1] == 9
        assert tuple(
            rebuilt[field][original_frontier] for field in ("text_input_ids", "speech_input_ids", "control_input_ids")
        ) == (history.text_pad, history.speech_pad, history.sleep)
        assert tuple(
            rebuilt[field][original_frontier]
            for field in ("raw_text_output_ids", "raw_speech_output_ids", "raw_control_output_ids")
        ) == tuple(channel[original_frontier] for channel in raw)
    finally:
        await harness.manager.shutdown()
