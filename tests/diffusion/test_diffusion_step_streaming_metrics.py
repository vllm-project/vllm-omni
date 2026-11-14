# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request timing keys that ``DiffusionEngine.step_streaming`` attaches.

An interval that was not observed is left out of the output metrics instead of
being written as ``0.0``; a measured ``0`` stays. The tests drive
``step_streaming`` on CPU with a stubbed scheduler stream and executor, then
follow the keys through ``OrchestratorAggregator`` into the per-request
``StageRequestStats`` record and the ``stage_table`` rows that
``build_and_log_summary`` returns.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading

import pytest

from vllm_omni.diffusion import diffusion_engine as diffusion_engine_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.metrics.stats import StageRequestStats, StageStats
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_ALWAYS_SET = {"diffusion_engine_exec_time_ms", "postprocess_time_ms"}
_OPTIONAL = {
    "preprocess_time_ms",
    "output_ready_wait_time_ms",
    "scheduler_queue_wait_ms",
    "forward_time_ms",
    "vae_decode_time_ms",
    "kv_recv_time_ms",
}


def _make_request(*, scheduler_queue_wait_ms: float | None = None) -> OmniDiffusionRequest:
    request = OmniDiffusionRequest(
        prompt="prompt",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
        request_id="req-diffusion",
    )
    # Set by the diffusion scheduler at the first admission.
    request.scheduler_queue_wait_ms = scheduler_queue_wait_ms
    return request


class _ReadyOutputExecutor:
    """Resolves async outputs the way ``DiffusionExecutor.wait_output_ready``
    does, from outputs that are already available."""

    def __init__(self, ready_outputs: dict[str, DiffusionOutput]) -> None:
        self._ready_outputs = ready_outputs

    def wait_output_ready(self, async_output_id: str) -> concurrent.futures.Future[DiffusionOutput]:
        future: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
        future.set_result(self._ready_outputs[async_output_id])
        return future


def _make_engine(
    outputs: list[DiffusionOutput],
    *,
    pre_process_func=None,
    ready_outputs: dict[str, DiffusionOutput] | None = None,
) -> DiffusionEngine:
    """An engine whose scheduler delivers ``outputs`` to the real output
    stream at admission and whose executor resolves async outputs from
    ``ready_outputs``. The stream materializes async outputs and records
    ``output_ready_wait_time`` before ``step_streaming`` sees them."""
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.pre_process_func = pre_process_func
    engine._scheduler_num_waiting_reqs = 0
    engine.main_loop = None
    engine._cv = threading.Condition()
    engine._out_streams = {}
    engine._unclaimed_async_outputs = {}
    engine._shutdown_output_futures = {}
    engine._streaming_playback = None

    async def _no_background_loop() -> None:
        return None

    def _admit(request: OmniDiffusionRequest) -> str:
        engine._out_streams[request.request_id] = asyncio.Queue()
        for output in outputs:
            engine._put_output(request.request_id, output)
        return request.request_id

    engine._check_and_start_background_loop = _no_background_loop
    engine._add_prepared_request = _admit
    engine.postprocess_output = lambda request, output: [OmniRequestOutput(request_id=request.request_id)]
    engine.executor = _ReadyOutputExecutor(ready_outputs or {})
    return engine


async def _collect_metrics(engine: DiffusionEngine, request: OmniDiffusionRequest) -> list[dict]:
    return [out.metrics async for batch in engine.step_streaming(request) for out in batch]


@pytest.mark.asyncio
async def test_unobserved_intervals_are_left_out() -> None:
    """No pre-processor, a synchronous output, profiler off, no KV, and no
    scheduler stamp: only the always-measured keys are written."""
    engine = _make_engine([DiffusionOutput()])

    (metrics,) = await _collect_metrics(engine, _make_request())

    assert _ALWAYS_SET <= metrics.keys()
    assert not (_OPTIONAL & metrics.keys())
    assert metrics["diffusion_engine_exec_time_ms"] >= 0.0
    assert metrics["postprocess_time_ms"] >= 0.0


@pytest.mark.asyncio
async def test_observed_intervals_are_reported() -> None:
    def _pre_process(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return request

    ready = DiffusionOutput(
        stage_durations={"DemoPipeline.diffuse": 0.5, "DemoPipeline.vae.decode": 0.25},
        kv_recv_ms=12.0,
    )
    engine = _make_engine(
        [DiffusionOutput(async_output_id="out-1")],
        pre_process_func=_pre_process,
        ready_outputs={"out-1": ready},
    )

    (metrics,) = await _collect_metrics(engine, _make_request(scheduler_queue_wait_ms=3.0))

    assert (_ALWAYS_SET | _OPTIONAL) <= metrics.keys()
    assert metrics["preprocess_time_ms"] >= 0.0
    assert metrics["output_ready_wait_time_ms"] >= 0.0
    assert ready.output_ready_wait_time is not None
    assert metrics["scheduler_queue_wait_ms"] == pytest.approx(3.0)
    # Profiler and KV values come from the output the async wait returned.
    assert metrics["forward_time_ms"] == pytest.approx(500.0)
    assert metrics["vae_decode_time_ms"] == pytest.approx(250.0)
    assert metrics["kv_recv_time_ms"] == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_output_without_async_wait_leaves_wait_unset() -> None:
    """The stream sets ``output_ready_wait_time`` only when it waited for an
    async output, so a synchronous output does not report a 0 ms wait."""
    output = DiffusionOutput()
    engine = _make_engine([output])

    (metrics,) = await _collect_metrics(engine, _make_request())

    assert output.output_ready_wait_time is None
    assert "output_ready_wait_time_ms" not in metrics


@pytest.mark.asyncio
async def test_measured_zero_queue_wait_is_kept() -> None:
    engine = _make_engine([DiffusionOutput()])

    (metrics,) = await _collect_metrics(engine, _make_request(scheduler_queue_wait_ms=0.0))

    assert metrics["scheduler_queue_wait_ms"] == 0.0


@pytest.mark.asyncio
async def test_output_ready_wait_is_per_output() -> None:
    """A streamed request can mix synchronous and async outputs; each output
    reports only the wait it had."""
    engine = _make_engine(
        [DiffusionOutput(finished=False), DiffusionOutput(async_output_id="out-2")],
        ready_outputs={"out-2": DiffusionOutput()},
    )

    first, second = await _collect_metrics(engine, _make_request())

    assert "output_ready_wait_time_ms" not in first
    assert second["output_ready_wait_time_ms"] >= 0.0


@pytest.mark.asyncio
async def test_output_ready_wait_does_not_carry_to_the_next_output() -> None:
    """A synchronous output after an async one does not inherit its wait."""
    engine = _make_engine(
        [DiffusionOutput(async_output_id="out-1", finished=False), DiffusionOutput()],
        ready_outputs={"out-1": DiffusionOutput(finished=False)},
    )

    first, second = await _collect_metrics(engine, _make_request())

    assert first["output_ready_wait_time_ms"] >= 0.0
    assert "output_ready_wait_time_ms" not in second


@pytest.mark.asyncio
async def test_debug_breakdown_formats_unobserved_intervals() -> None:
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = diffusion_engine_module.logger
    handler = _Collect(level=logging.DEBUG)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        await _collect_metrics(_make_engine([DiffusionOutput()]), _make_request())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    breakdown = [msg for msg in records if "step_streaming breakdown" in msg]
    assert len(breakdown) == 1
    assert "preprocess=0.00 ms" in breakdown[0]
    assert "output_ready_wait=0.00 ms" in breakdown[0]


def _stage_stats() -> StageRequestStats:
    return StageRequestStats(
        batch_id=1,
        batch_size=1,
        num_tokens_in=0,
        num_tokens_out=0,
        stage_gen_time_ms=40.0,
        rx_transfer_bytes=0,
        rx_decode_time_ms=0.0,
        rx_in_flight_time_ms=0.0,
        stage_stats=StageStats(),
    )


@pytest.mark.asyncio
async def test_unobserved_intervals_stay_out_of_the_stage_record() -> None:
    """Follow the engine keys through the aggregator into the per-request
    record and the ``stage_table`` rows that ``build_and_log_summary``
    returns."""
    engine = _make_engine([DiffusionOutput()])
    request = _make_request(scheduler_queue_wait_ms=2.0)
    outputs = [out async for batch in engine.step_streaming(request) for out in batch]

    aggregator = OrchestratorAggregator(num_stages=1, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=0)
    aggregator.accumulate_diffusion_metrics("diffusion", "req-diffusion", outputs)
    aggregator.on_stage_metrics(0, "req-diffusion", _stage_stats(), "image")

    (event,) = aggregator.stage_events["req-diffusion"]
    assert event.diffusion_metrics is not None
    assert "preprocess_time_s" not in event.diffusion_metrics
    assert "output_ready_wait_time_s" not in event.diffusion_metrics
    assert event.diffusion_metrics["scheduler_queue_wait_s"] == pytest.approx(0.002)
    assert "diffusion_engine_exec_time_s" in event.diffusion_metrics

    summary = aggregator.build_and_log_summary()
    (entry,) = summary["stage_table"]
    (row,) = entry["stages"]
    assert "preprocess_time_s" not in row
    assert "output_ready_wait_time_s" not in row
    assert row["scheduler_queue_wait_s"] == pytest.approx(0.002)
