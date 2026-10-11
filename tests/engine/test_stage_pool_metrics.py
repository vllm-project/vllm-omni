# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("output_kind", [RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA])
def test_stage_pool_resumable_metrics_do_not_recount_prior_segments(monkeypatch, output_kind) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    params = SamplingParams(output_kind=output_kind)
    totals = []
    for segment in range(1, 4):
        now = 100.0 + segment
        monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: now)
        frames = 24000 * (segment if output_kind == RequestOutputKind.CUMULATIVE else 1)
        output = SimpleNamespace(
            request_id="req-stream",
            multimodal_output={"audio": SimpleNamespace(shape=(frames,)), "sample_rate": 24000},
            outputs=[],
        )
        pool.record_output_timestamps([output], output_ts=now)
        metrics = pool.build_stage_metrics(
            [output], submit_ts=100.0, request_timestamp=100.0, replica_id=0, sampling_params=params, incremental=True
        )
        totals.append(metrics)

    assert [m.audio_generated_frames for m in totals] == [24000, 24000, 24000]
    assert [m.output_unit_count for m in totals] == [24000, 24000, 24000]
    assert sum(m.audio_duration_s for m in totals) == pytest.approx(3.0)
    assert sum(m.stage_gen_time_ms for m in totals) == pytest.approx(3000.0)


def test_stage_pool_cumulative_audio_metrics_release_request_watermark(monkeypatch) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    params = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
    output = SimpleNamespace(
        request_id="req-stream",
        multimodal_output={"audio": SimpleNamespace(shape=(24000,)), "sample_rate": 24000},
        outputs=[],
    )
    durations = []
    for now in [101.0, 102.0]:
        monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: now)
        pool.record_output_timestamps([output], output_ts=now)
        metrics = pool.build_stage_metrics(
            [output], submit_ts=100.0, request_timestamp=100.0, replica_id=0, sampling_params=params
        )
        durations.append(metrics.audio_duration_s)
    assert durations == [1.0, 0.0]

    pool.release_binding("req-stream")
    monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: 201.0)
    pool.record_output_timestamps([output], output_ts=201.0)
    restarted = pool.build_stage_metrics(
        [output], submit_ts=200.0, request_timestamp=200.0, replica_id=0, sampling_params=params
    )
    assert restarted.audio_duration_s == pytest.approx(1.0)
    assert restarted.stage_gen_time_ms == pytest.approx(1000.0)


def test_duplex_response_audio_frames_match_serialized_stage_duration(monkeypatch) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    session = DuplexEngineSession(session_id="test", config=DuplexSessionConfig(model="test"), num_stages=3)
    session.begin_response()
    monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: 101.0)
    for index in range(2):
        output = SimpleNamespace(
            request_id=f"req-{index}",
            multimodal_output={"audio": SimpleNamespace(shape=(24000,)), "sample_rate": 24000},
            outputs=[],
        )
        pool.record_output_timestamps([output], output_ts=101.0)
        metrics = pool.build_stage_metrics(
            [output],
            submit_ts=100.0,
            request_timestamp=100.0,
            replica_id=0,
            sampling_params=SamplingParams(output_kind=RequestOutputKind.CUMULATIVE),
        )
        snapshot = OrchestratorAggregator._merge_stage_metric_event(None, metrics)
        total = session.accumulate_response_stage_metrics({"2": snapshot})["2"]
    assert total["audio_frames"] == 48000
    assert total["output_unit_count"] == 48000
    assert total["audio_duration_s"] == pytest.approx(total["audio_frames"] / total["audio_sample_rate"])


@pytest.mark.asyncio
@pytest.mark.parametrize("output_kind", [RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA])
async def test_session_audio_metrics_stay_with_response_before_interrupt(monkeypatch, output_kind) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    state = OrchestratorRequestState(
        request_id="req-stream",
        session_owned=True,
        sampling_params_list=[None, None, SamplingParams(output_kind=output_kind)],
        final_stage_id=2,
        stage_submit_ts={2: 100.0},
        request_timestamp=100.0,
    )
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [None, None, pool]
    orchestrator.request_states = {state.request_id: state}
    session = DuplexEngineSession(session_id="test", config=DuplexSessionConfig(model="test"), num_stages=3)
    session.begin_response()
    routed = []

    async def route(_stage_id, _replica_id, _output, _req_state, metrics):
        assert metrics is not None, "Audio must be accounted before the response can be interrupted"
        snapshot = OrchestratorAggregator._merge_stage_metric_event(None, metrics)
        session.accumulate_response_stage_metrics({"2": snapshot})
        routed.append(metrics)

    orchestrator._route_output = route
    cumulative = 0
    now = 100.0

    async def deliver(frames, *, finished=False):
        nonlocal cumulative, now
        cumulative += frames
        now += 1.0
        monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: now)
        output = SimpleNamespace(
            request_id=state.request_id,
            error=None,
            finished=finished,
            multimodal_output={
                "audio": SimpleNamespace(
                    shape=(cumulative if output_kind == RequestOutputKind.CUMULATIVE else frames,)
                ),
                "sample_rate": 24000,
            },
            outputs=[],
        )
        pool.record_output_timestamps([output], output_ts=now)
        await orchestrator._handle_processed_outputs(2, 0, [output])

    # Real failing wire lengths: 9.84 s cleared before native interruption;
    # the same persistent stage request then delivers a new 5.92 s response.
    for frames in [20160, *([24000] * 9)]:
        await deliver(frames)
    cancelled = session.accumulate_response_stage_metrics(None)["2"]
    assert cancelled["audio_frames"] == 236160
    session.end_response(commit_text=False, preserve_request=True)
    session.begin_response()
    for frames in [20160, *([24000] * 4), 25920]:
        await deliver(frames)
    await deliver(0, finished=True)
    completed = session.accumulate_response_stage_metrics(None)["2"]
    assert completed["audio_frames"] == completed["output_unit_count"] == 142080
    assert completed["audio_duration_s"] == pytest.approx(5.92)
    assert sum(metric.audio_generated_frames for metric in routed) == 378240
    assert routed[-1].audio_generated_frames == 0
    assert sum(metric.stage_gen_time_ms for metric in routed) == pytest.approx(17000.0)
    assert sum(len(metric.inter_output_latencies_ms) for metric in routed) == 16
    assert sum(sum(metric.inter_output_latencies_ms) for metric in routed) == pytest.approx(16000.0)


@pytest.mark.parametrize("collect_stage_metrics", [True, False])
def test_incremental_audio_metrics_preserve_integrated_text_counts_and_intervals(
    monkeypatch, collect_stage_metrics
) -> None:
    pool = StagePool(
        0,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    pool.collect_stage_metrics = collect_stage_metrics
    metrics: list[StageRequestStats] = []
    for count in [1, 2, 3, 3]:
        now = 100.0 + len(metrics) + 1
        monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: now)
        output = SimpleNamespace(
            request_id="audio-and-text",
            prompt_token_ids=[1, 2, 3, 4],
            multimodal_output={"audio": SimpleNamespace(shape=(24000 * count,)), "sample_rate": 24000},
            outputs=[SimpleNamespace(token_ids=list(range(count)))],
        )
        pool.record_output_timestamps([output], output_ts=now)
        metrics.append(
            pool.build_stage_metrics(
                [output],
                submit_ts=100.0,
                request_timestamp=100.0,
                replica_id=0,
                sampling_params=SamplingParams(output_kind=RequestOutputKind.CUMULATIVE),
                incremental=True,
            )
        )
    assert sum(item.num_tokens_in for item in metrics) == 4
    assert sum(item.num_tokens_out for item in metrics) == 3
    assert sum(item.stage_gen_time_ms for item in metrics) == pytest.approx(4000.0)
    assert metrics[-1].stage_stats.total_token == 3
    if collect_stage_metrics:
        assert sum(item.audio_generated_frames for item in metrics) == 72000
        assert sum(len(item.inter_output_latencies_ms) for item in metrics) == 3
    else:
        assert all(item.audio_generated_frames == 0 for item in metrics)
        assert all(item.output_unit_count == 0 for item in metrics)
        assert all(item.inter_output_latencies_ms is None for item in metrics)
    pool.release_binding("audio-and-text")
    monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: 201.0)
    pool.record_output_timestamps([output], output_ts=201.0)
    restarted = pool.build_stage_metrics(
        [output],
        submit_ts=200.0,
        request_timestamp=200.0,
        replica_id=0,
        sampling_params=SamplingParams(output_kind=RequestOutputKind.CUMULATIVE),
        incremental=True,
    )
    assert restarted.num_tokens_in == 4 and restarted.num_tokens_out == 3
    assert restarted.audio_generated_frames == (72000 if collect_stage_metrics else 0)
    assert restarted.stage_gen_time_ms == pytest.approx(1000.0)
    assert restarted.inter_output_latencies_ms == ([] if collect_stage_metrics else None)
