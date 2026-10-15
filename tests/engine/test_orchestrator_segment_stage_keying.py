# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming segment-boundary state must be keyed by stage.

``OrchestratorBase._orchestration_loop`` polls every stage into the same
``OrchestratorRequestState``. While the segment-boundary fields lived in a single
flat slot per request, whichever stage polled most recently overwrote the others,
and the per-stage consumers then read back a different stage's boundary.

The duplex consumer of the per-stage segment (``DuplexOrchestrator``'s output
context) is covered in ``test_duplex_orchestrator.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    StreamingSegmentState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

THINKER_STAGE = 0
TALKER_STAGE = 1


@dataclass
class _StageMetrics:
    pipeline_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _StageOutput:
    """Stand-in for a processed stage output, with the fields the router reads."""

    request_id: str
    finished: bool = False
    error: object | None = None


class _RecordingPool:
    """Stage pool that records the requests it was asked to build metrics for."""

    def __init__(self) -> None:
        self.built_for: list[str] = []

    def build_stage_metrics(self, outputs: list[_StageOutput], **_kwargs: object) -> _StageMetrics:
        self.built_for.extend(output.request_id for output in outputs)
        return _StageMetrics()


def _streaming_req_state(request_id: str = "req-streaming") -> OrchestratorRequestState:
    req_state = OrchestratorRequestState(
        request_id=request_id,
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=TALKER_STAGE,
        session_owned=True,
    )
    req_state.streaming.enabled = True
    return req_state


def test_segment_accessor_reads_only_its_own_stage() -> None:
    req_state = _streaming_req_state()
    req_state.streaming.segments[THINKER_STAGE] = StreamingSegmentState(
        finished=True,
        token_ids=[11, 22],
        output_metadata={"stage": "thinker"},
    )
    # The talker polls after the thinker and is still mid-segment.
    req_state.streaming.segments[TALKER_STAGE] = StreamingSegmentState(finished=False)

    thinker = req_state.streaming.segment(THINKER_STAGE)
    talker = req_state.streaming.segment(TALKER_STAGE)

    # These are the assertions that fail on the flat field: the talker is written
    # second, so the thinker loses its boundary to whichever stage polled last.
    assert thinker.finished is True
    assert thinker.token_ids == [11, 22]
    assert thinker.output_metadata == {"stage": "thinker"}

    assert talker.finished is False
    assert talker.token_ids == []
    assert talker.output_metadata == {}

    # A stage that never reported a boundary gets a fresh, unshared segment.
    unreported = req_state.streaming.segment(2)
    unreported.output_metadata["poisoned"] = True
    assert req_state.streaming.segment(2).output_metadata == {}


@pytest.mark.asyncio
async def test_segment_boundary_on_one_stage_does_not_build_metrics_on_another() -> None:
    req_state = _streaming_req_state()
    req_state.streaming.segments[THINKER_STAGE] = StreamingSegmentState(finished=True)

    talker_pool = _RecordingPool()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.request_states = {req_state.request_id: req_state}
    orchestrator.stage_pools = [_RecordingPool(), talker_pool]

    routed: list[_StageMetrics | None] = []

    async def record_route(
        _stage: int,
        _replica: int,
        _output: _StageOutput,
        _state: OrchestratorRequestState,
        metrics: _StageMetrics | None,
    ) -> None:
        routed.append(metrics)

    orchestrator._route_output = record_route

    # The talker's own output is neither finished nor at a segment boundary, so
    # it must not be treated as one just because the thinker reached one.
    await orchestrator._handle_processed_outputs(
        TALKER_STAGE,
        0,
        [_StageOutput(request_id=req_state.request_id)],
    )

    assert talker_pool.built_for == []
    assert routed == [None]
