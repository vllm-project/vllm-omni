"""Tests for streaming TTS text input via UPDATE_REQUEST mechanism."""
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.entrypoints.stage_utils import OmniStageTaskType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Scheduler: update_request_additional_info
# ---------------------------------------------------------------------------


def _make_scheduler():
    """Create a minimal OmniARScheduler-like object for unit testing."""
    from vllm.v1.request import RequestStatus

    sched = SimpleNamespace(
        requests={},
        running=deque(),
        _pending_additional_info_updates=defaultdict(list),
        _early_additional_info_updates=defaultdict(list),
    )

    # Bind methods from OmniARScheduler
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched.update_request_additional_info = (
        OmniARScheduler.update_request_additional_info.__get__(sched)
    )
    sched._resolve_streaming_request_id = (
        OmniARScheduler._resolve_streaming_request_id.__get__(sched)
    )
    sched._maybe_resume_request = (
        OmniARScheduler._maybe_resume_request.__get__(sched)
    )
    sched._flush_early_updates = (
        OmniARScheduler._flush_early_updates.__get__(sched)
    )
    sched._drain_pending_additional_info_updates = (
        OmniARScheduler._drain_pending_additional_info_updates.__get__(sched)
    )
    return sched


def _make_request(req_id, external_req_id=None, status=None):
    from vllm.v1.request import RequestStatus

    req = SimpleNamespace(
        request_id=req_id,
        external_req_id=external_req_id,
        status=status or RequestStatus.RUNNING,
    )
    req.is_finished = lambda: False
    return req


class TestSchedulerUpdateRequest:
    def test_update_goes_to_pending_when_request_exists(self):
        sched = _make_scheduler()
        req = _make_request("req-abc-1234", external_req_id="req-abc")
        sched.requests["req-abc-1234"] = req
        sched.running.append(req)

        sched.update_request_additional_info("req-abc", {"key": "val"})

        assert "req-abc-1234" in sched._pending_additional_info_updates
        assert len(sched._pending_additional_info_updates["req-abc-1234"]) == 1

    def test_update_goes_to_early_when_request_not_registered(self):
        sched = _make_scheduler()

        sched.update_request_additional_info("req-unknown", {"key": "val"})

        assert "req-unknown" in sched._early_additional_info_updates
        assert len(sched._pending_additional_info_updates) == 0

    def test_early_updates_flushed_on_resolve(self):
        sched = _make_scheduler()

        # Updates arrive before request registers
        sched.update_request_additional_info("req-abc", {"k": 1})
        sched.update_request_additional_info("req-abc", {"k": 2})
        assert len(sched._early_additional_info_updates["req-abc"]) == 2

        # Request registers
        req = _make_request("req-abc-5678", external_req_id="req-abc")
        sched.requests["req-abc-5678"] = req
        sched.running.append(req)

        sched._flush_early_updates()

        assert len(sched._early_additional_info_updates) == 0
        assert len(sched._pending_additional_info_updates["req-abc-5678"]) == 2

    def test_drain_clears_pending(self):
        sched = _make_scheduler()
        sched._pending_additional_info_updates["req-1"].append({"a": 1})
        sched._pending_additional_info_updates["req-1"].append({"b": 2})

        result = sched._drain_pending_additional_info_updates()

        assert "req-1" in result
        assert len(result["req-1"]) == 2
        assert len(sched._pending_additional_info_updates) == 0

    def test_drain_returns_empty_when_nothing_pending(self):
        sched = _make_scheduler()
        assert sched._drain_pending_additional_info_updates() == {}

    def test_resume_paused_request_on_update(self):
        from vllm.v1.request import RequestStatus

        sched = _make_scheduler()
        req = _make_request(
            "req-abc-1234",
            external_req_id="req-abc",
            status=RequestStatus.WAITING_FOR_CHUNK,
        )
        sched.requests["req-abc-1234"] = req

        sched.update_request_additional_info("req-abc", {"key": "val"})

        assert req.status == RequestStatus.RUNNING
        assert req in sched.running

    def test_resolve_by_external_req_id(self):
        sched = _make_scheduler()
        req = _make_request("req-abc-suffix", external_req_id="req-abc")
        sched.requests["req-abc-suffix"] = req

        resolved = sched._resolve_streaming_request_id("req-abc")
        assert resolved == "req-abc-suffix"

    def test_resolve_returns_none_for_unknown(self):
        sched = _make_scheduler()
        assert sched._resolve_streaming_request_id("unknown") is None


# ---------------------------------------------------------------------------
# Model runner: _APPEND_KEYS and merge semantics
# ---------------------------------------------------------------------------


class TestModelRunnerMerge:
    def test_append_keys_concatenate(self):
        from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

        # Verify _APPEND_KEYS contains the streaming key
        assert "streaming_text_token_ids" in OmniGPUModelRunner._APPEND_KEYS

    def test_merge_appends_for_append_keys(self):
        """Simulate the merge behavior for streaming_text_token_ids."""
        # The merge logic: if key in _APPEND_KEYS and both are lists, concatenate
        existing = {"streaming_text_token_ids": [100, 200]}
        new_update = {"streaming_text_token_ids": [300, 400]}

        # Simulate merge
        from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

        merged = dict(existing)
        for k, v in new_update.items():
            if (
                k in OmniGPUModelRunner._APPEND_KEYS
                and isinstance(v, list)
                and isinstance(merged.get(k), list)
            ):
                merged[k] = merged[k] + v
            else:
                merged[k] = v

        assert merged["streaming_text_token_ids"] == [100, 200, 300, 400]

    def test_merge_replaces_when_existing_is_none(self):
        """After clearing (None), new update should set, not append."""
        existing = {"streaming_text_token_ids": None}
        new_update = {"streaming_text_token_ids": [500]}

        from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

        merged = dict(existing)
        for k, v in new_update.items():
            if (
                k in OmniGPUModelRunner._APPEND_KEYS
                and isinstance(v, list)
                and isinstance(merged.get(k), list)
            ):
                merged[k] = merged[k] + v
            else:
                merged[k] = v

        assert merged["streaming_text_token_ids"] == [500]


# ---------------------------------------------------------------------------
# OmniSchedulerOutput: additional_information_updates field
# ---------------------------------------------------------------------------


class TestSchedulerOutput:
    def test_additional_information_updates_default_empty(self):
        from vllm.v1.core.sched.output import SchedulerOutput

        base_fields = {name: None for name in SchedulerOutput.__dataclass_fields__}
        base_fields["num_scheduled_tokens"] = {}
        base_fields["scheduled_new_reqs"] = []
        base_fields["total_num_scheduled_tokens"] = 0
        base_fields["scheduled_spec_decode_tokens"] = {}
        base_fields["scheduled_encoder_inputs"] = {}
        base_fields["num_common_prefix_blocks"] = [0]
        base_fields["finished_req_ids"] = []
        base_fields["free_encoder_mm_hashes"] = []
        base_fields["preempted_req_ids"] = []

        output = OmniSchedulerOutput(
            **base_fields,
            finished_requests_needing_kv_transfer={},
        )
        assert output.additional_information_updates == {}


# ---------------------------------------------------------------------------
# OmniModelRunnerOutput: streaming_pause_req_ids field
# ---------------------------------------------------------------------------


class TestModelRunnerOutput:
    def test_streaming_pause_req_ids_default_none(self):
        from vllm_omni.outputs import OmniModelRunnerOutput

        output = OmniModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
        )
        assert output.streaming_pause_req_ids is None


# ---------------------------------------------------------------------------
# OmniStageTaskType: UPDATE variant
# ---------------------------------------------------------------------------


class TestStageTaskType:
    def test_update_task_type_exists(self):
        assert hasattr(OmniStageTaskType, "UPDATE")
        assert OmniStageTaskType.UPDATE.value == "update"


# ---------------------------------------------------------------------------
# AsyncOmni: update_request routing
# ---------------------------------------------------------------------------


class TestAsyncOmniUpdateRequest:
    def test_update_request_submits_to_stage(self):
        stage = MagicMock()
        omni = SimpleNamespace(stage_list=[stage])

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        AsyncOmni.update_request(omni, "req-1", {"key": "val"}, stage_id=0)

        stage.submit.assert_called_once()
        call_args = stage.submit.call_args[0][0]
        assert call_args["type"] == OmniStageTaskType.UPDATE
        assert call_args["request_id"] == "req-1"
        assert call_args["update"] == {"key": "val"}


# ---------------------------------------------------------------------------
# Patch: EngineCoreRequestType.UPDATE
# ---------------------------------------------------------------------------


class TestEngineCoreRequestTypePatch:
    def test_update_type_exists(self):
        from vllm.v1.engine import EngineCoreRequestType

        assert hasattr(EngineCoreRequestType, "UPDATE")
        assert EngineCoreRequestType.UPDATE.value == b"\x05"
