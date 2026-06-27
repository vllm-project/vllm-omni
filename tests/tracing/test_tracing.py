# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_omni.diffusion.output_formatter import DiffusionStepTimings
from vllm_omni.metrics.stats import PipelineTimings
from vllm_omni.tracing import OmniSpanAttributes

pytestmark = [pytest.mark.cpu]


# ------------------------------------------------------------------ #
#  PipelineTimings
# ------------------------------------------------------------------ #


def test_pipeline_timings_asdict():
    pt = PipelineTimings(
        queue_wait_ms=12.5,
        preprocess_ms=3.2,
        ar2diffusion_ms=0.8,
    )
    d = dataclasses.asdict(pt)
    assert d == {
        "queue_wait_ms": 12.5,
        "preprocess_ms": 3.2,
        "ar2diffusion_ms": 0.8,
    }


# ------------------------------------------------------------------ #
#  DiffusionStepTimings
# ------------------------------------------------------------------ #


def test_step_start_ts_default():
    timings = DiffusionStepTimings(
        preprocess_time_s=0.01,
        exec_time_s=0.02,
        postprocess_time_s=0.03,
        total_time_ms=60.0,
    )
    assert timings.step_start_ts == 0.0


# ------------------------------------------------------------------ #
#  OmniSchedulerMixin.add_request
# ------------------------------------------------------------------ #


def test_add_request_sets_first_chunk_ts_for_new_request():
    from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

    class _BaseScheduler:
        def __init__(self):
            self.requests = {}

        def add_request(self, request):
            self.requests[request.request_id] = request

    class _FakeScheduler(OmniSchedulerMixin, _BaseScheduler):
        def __init__(self):
            self.requests = {}

    scheduler = _FakeScheduler()
    request = SimpleNamespace(request_id="req-1")

    before = time.time()
    scheduler.add_request(request)
    after = time.time()

    assert hasattr(request, "first_chunk_received_ts")
    assert before <= request.first_chunk_received_ts <= after


# ------------------------------------------------------------------ #
#  _emit_llm_processing_span
# ------------------------------------------------------------------ #


def _make_output_processor(**kwargs):
    """Build a MultimodalOutputProcessor with tracing wired up."""
    from vllm_omni.engine.output_processor import MultimodalOutputProcessor

    defaults = dict(
        tokenizer=None,
        log_stats=False,
        tracing_enabled=True,
        stage_id=1,
        stage_name="talker",
        replica_id=0,
    )
    defaults.update(kwargs)
    return MultimodalOutputProcessor(**defaults)


def _make_engine_core_output(first_chunk_ts=None, trace_headers=None):
    return SimpleNamespace(
        first_chunk_received_ts=first_chunk_ts,
        trace_headers=trace_headers or {},
    )


def _make_req_state(
    scheduled_ts=0.0,
    first_token_ts=0.0,
    last_token_ts=0.0,
):
    stats = SimpleNamespace(
        scheduled_ts=scheduled_ts,
        first_token_ts=first_token_ts,
        last_token_ts=last_token_ts,
        arrival_time=0.0,
    )
    return SimpleNamespace(
        native_text_stats=stats,
        stats=stats,
        external_req_id="req-1",
    )


@patch("vllm_omni.engine.output_processor.instrument_manual")
@patch(
    "vllm_omni.engine.output_processor.extract_trace_context",
    return_value=None,
)
def test_llm_span_skipped_when_no_first_chunk_ts(_mock_extract, mock_instrument):
    proc = _make_output_processor()
    output = _make_engine_core_output(first_chunk_ts=None)
    req_state = _make_req_state(scheduled_ts=1.0, last_token_ts=2.0)

    proc._emit_llm_processing_span(output, req_state)

    mock_instrument.assert_not_called()


@patch("vllm_omni.engine.output_processor.instrument_manual")
@patch(
    "vllm_omni.engine.output_processor.extract_trace_context",
    return_value=None,
)
def test_llm_span_uses_scheduled_to_last_token_duration(_mock_extract, mock_instrument):
    proc = _make_output_processor(stage_id=0, stage_name="thinker", replica_id=2)
    first_chunk_ts = 1700000000.5
    scheduled_ts = 100.0
    last_token_ts = 100.25

    output = _make_engine_core_output(first_chunk_ts=first_chunk_ts)
    req_state = _make_req_state(
        scheduled_ts=scheduled_ts,
        last_token_ts=last_token_ts,
    )

    proc._emit_llm_processing_span(output, req_state)

    mock_instrument.assert_called_once()
    call_kwargs = mock_instrument.call_args
    assert call_kwargs.kwargs["span_name"] == "llm_processing"

    duration = last_token_ts - scheduled_ts  # 0.25s
    expected_start_ns = int(first_chunk_ts * 1e9)
    expected_end_ns = int((first_chunk_ts + duration) * 1e9)
    assert call_kwargs.kwargs["start_time"] == expected_start_ns
    assert call_kwargs.kwargs["end_time"] == expected_end_ns

    attrs = call_kwargs.kwargs["attributes"]
    assert attrs[OmniSpanAttributes.STAGE_ID] == 0
    assert attrs[OmniSpanAttributes.STAGE_NAME] == "thinker"
    assert attrs[OmniSpanAttributes.STAGE_REPLICA_ID] == 2


# ------------------------------------------------------------------ #
#  _emit_diffusion_span
# ------------------------------------------------------------------ #


@patch("vllm_omni.entrypoints.omni_base.instrument_manual")
@patch(
    "vllm_omni.entrypoints.omni_base.extract_trace_context",
    return_value=None,
)
def test_diffusion_span_uses_step_start_ts(_mock_extract, mock_instrument):
    from vllm_omni.entrypoints.omni_base import OmniBase

    step_start = 1700000000.123
    total_ms = 500.0

    result = SimpleNamespace(
        trace_headers={},
    )
    stage_metrics = SimpleNamespace(
        diffusion_metrics={
            "diffusion_engine_total_time_ms": total_ms,
            "step_start_ts": step_start,
            "preprocess_time_ms": 10.0,
            "diffusion_engine_exec_time_ms": 480.0,
            "postprocess_time_ms": 10.0,
        },
        stage_id=2,
        replica_id=0,
    )

    # Call the unbound method with a fake self — we only need the method
    # logic, not the full OmniBase instance.
    fake_self = SimpleNamespace()
    OmniBase._emit_diffusion_span(fake_self, result, stage_metrics)

    mock_instrument.assert_called_once()
    call_kwargs = mock_instrument.call_args

    expected_start_ns = int(step_start * 1e9)
    expected_end_ns = int((step_start + total_ms / 1000) * 1e9)
    assert call_kwargs.kwargs["start_time"] == expected_start_ns
    assert call_kwargs.kwargs["end_time"] == expected_end_ns
    assert call_kwargs.kwargs["span_name"] == "diffusion_request"

    attrs = call_kwargs.kwargs["attributes"]
    assert attrs[OmniSpanAttributes.STAGE_ID] == 2
    assert attrs[OmniSpanAttributes.STAGE_NAME] == "diffusion"
    assert attrs[OmniSpanAttributes.STAGE_REPLICA_ID] == 0
    assert attrs[OmniSpanAttributes.DIFFUSION_PREPROCESS_MS] == 10.0
    assert attrs[OmniSpanAttributes.DIFFUSION_EXEC_MS] == 480.0
    assert attrs[OmniSpanAttributes.DIFFUSION_TOTAL_MS] == total_ms
