# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the realtime AR-Diffusion benchmark harness.

The harness is driven on a virtual clock against fake sessions, so the load
model, deadline model and metric definitions are verified without an engine,
a device or a checkpoint.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.ar_diffusion.realtime_harness import (
    ArrivalPlan,
    BenchmarkConfig,
    burst_arrivals,
    poisson_arrivals,
    run_benchmark,
)
from benchmarks.ar_diffusion.realtime_metrics import (
    ChunkEvent,
    LoadMode,
    SessionRecord,
    WorkloadProfile,
    chunk_deadlines,
    compare_runs,
    load_profile_from_spec,
    percentile,
    summarize_run,
    summarize_session,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# 3 latent frames x VAE temporal factor 4 = 12 raw frames, 16 fps playout, so
# one chunk is worth 750 ms. Declared here rather than imported from a model.
PROFILE = WorkloadProfile(frames_per_chunk=12, target_fps=16.0)


class VirtualClock:
    """Deterministic clock that advances only when a sleeper is released."""

    def __init__(self) -> None:
        self.now = 0.0
        self._waiters: list[tuple[float, asyncio.Future]] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            return
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._waiters.append((self.now + delay, future))
        await future

    async def run_until_idle(self, task: asyncio.Task) -> None:
        """Advance to each next wakeup until ``task`` completes."""
        while not task.done():
            await asyncio.sleep(0)
            if task.done():
                break
            if not self._waiters:
                await asyncio.sleep(0)
                if not self._waiters:
                    break
                continue
            self._waiters.sort(key=lambda item: item[0])
            when, future = self._waiters.pop(0)
            self.now = max(self.now, when)
            if not future.done():
                future.set_result(None)


class FakeSession:
    """A session whose tick takes a fixed amount of virtual time."""

    def __init__(self, clock: VirtualClock, *, tick_s: float, fail_at: int | None = None) -> None:
        self._clock = clock
        self._tick_s = tick_s
        self._fail_at = fail_at
        self._calls = 0
        self.closed = False

    async def next_chunk(self) -> str:
        if self._fail_at is not None and self._calls == self._fail_at:
            raise RuntimeError("fake rollout lost")
        self._calls += 1
        await self._clock.sleep(self._tick_s)
        return "chunk"

    async def close(self) -> None:
        self.closed = True


def make_factory(clock: VirtualClock, *, tick_s: float, fail_at: int | None = None):
    created: dict[str, FakeSession] = {}

    async def factory(session_id: str) -> FakeSession:
        session = FakeSession(clock, tick_s=tick_s, fail_at=fail_at)
        created[session_id] = session
        return session

    return factory, created


async def drive(config: BenchmarkConfig, factory, clock: VirtualClock, **kwargs):
    task = asyncio.ensure_future(run_benchmark(config, factory, clock=clock.time, sleep=clock.sleep, **kwargs))
    await clock.run_until_idle(task)
    return await task


# --------------------------------------------------------------------------
# Metric definitions
# --------------------------------------------------------------------------


def test_release_period_comes_from_the_profile_not_a_flag() -> None:
    assert PROFILE.release_period_s == pytest.approx(0.75)
    assert WorkloadProfile(frames_per_chunk=12, target_fps=24.0).release_period_s == pytest.approx(0.5)


def test_profile_is_built_from_a_declared_spec() -> None:
    profile = load_profile_from_spec(
        {"frames_per_block": 3, "vae_temporal_factor": 4, "resident_bytes_per_session": 1024},
        target_fps=16.0,
        causal=False,
    )
    assert profile.frames_per_chunk == 12
    assert profile.resident_bytes_per_session == 1024
    with pytest.raises(ValueError, match="frames_per_block"):
        load_profile_from_spec({"vae_temporal_factor": 4}, target_fps=16.0)


def test_percentile_reports_an_observed_value() -> None:
    values = [1.0, 2.0, 3.0, 100.0]
    assert percentile(values, 0.5) == 2.0
    assert percentile(values, 0.99) == 100.0
    assert percentile([], 0.5) is None


def _record(
    session_id: str,
    readies: list[float],
    *,
    submit_gap: float = 0.0,
    latencies: list[float] | None = None,
) -> SessionRecord:
    """Build a record from ready times, with per-chunk latency if needed."""
    gaps = latencies if latencies is not None else [submit_gap] * len(readies)
    if len(gaps) != len(readies):
        raise ValueError("latencies must match readies.")
    return SessionRecord(
        session_id=session_id,
        t_start=0.0,
        events=tuple(
            ChunkEvent(session_id=session_id, chunk_index=i, t_submit=t - gap, t_ready=t)
            for i, (t, gap) in enumerate(zip(readies, gaps, strict=True))
        ),
    )


def test_deadlines_anchor_on_the_buffer_and_skip_the_prebuffer() -> None:
    record = _record("s", [1.0, 1.5, 2.0])
    deadlines = chunk_deadlines(record, PROFILE)
    # buffer_chunks=1: chunk 0 fixes the grid and has no deadline of its own.
    assert 0 not in deadlines
    assert deadlines[1] == pytest.approx(1.75)
    assert deadlines[2] == pytest.approx(2.50)


def test_deeper_buffer_grants_real_slack() -> None:
    """Playback starts at the anchor holding two chunks, so chunk 2 is due
    after both have played -- not one period after the anchor arrived."""
    profile = WorkloadProfile(frames_per_chunk=12, target_fps=16.0, buffer_chunks=2)
    record = _record("s", [1.0, 1.5, 2.0])
    deadlines = chunk_deadlines(record, profile)
    assert set(deadlines) == {2}
    assert deadlines[2] == pytest.approx(1.5 + 24 / 16.0)


def test_cpr_counts_only_chunks_that_had_a_deadline() -> None:
    # chunk 1 due at 1.75 arrives at 1.5 (met); chunk 2 due at 2.5 arrives at 9.0 (missed).
    record = _record("s", [1.0, 1.5, 9.0])
    summary = summarize_session(record, PROFILE, mode=LoadMode.PACED)
    assert summary.deadline_chunks == 2
    assert summary.deadline_misses == 1
    assert summary.continuous_play_ratio == pytest.approx(0.5)


def test_saturating_mode_reports_no_continuity_metrics() -> None:
    """SLO is only defined under pacing; a ceiling run must not imply one."""
    record = _record("s", [1.0, 1.5, 9.0])
    summary = summarize_session(record, PROFILE, mode=LoadMode.SATURATING)
    assert summary.continuous_play_ratio is None
    assert summary.deadline_chunks == 0


def test_rtf_compares_wall_time_against_video_time() -> None:
    # 4 chunks x 0.75 s of video = 3.0 s; produced in 1.5 s of wall time.
    record = _record("s", [0.375, 0.75, 1.125, 1.5], submit_gap=0.375)
    summary = summarize_session(record, PROFILE, mode=LoadMode.SATURATING)
    assert summary.rtf == pytest.approx((1.5 - 0.0) / 3.0)


CAUSAL = WorkloadProfile(frames_per_chunk=12, frames_per_first_chunk=9, target_fps=16.0)


def test_causal_decoder_delivers_a_shorter_opening_chunk() -> None:
    """(n - 1) * factor + 1 for the opening block, n * factor after it.

    3 latent frames at temporal factor 4 is 9 raw frames the first time and 12
    every time after, so K chunks deliver 12K - 3 frames, not 12K.
    """
    assert CAUSAL.is_causal is True
    assert CAUSAL.frames_at(0) == 9
    assert CAUSAL.frames_at(1) == CAUSAL.frames_at(7) == 12
    assert CAUSAL.cumulative_frames(10) == 117  # matches the pipeline's 117-frame horizon
    assert CAUSAL.cumulative_frames(0) == 0

    summary = summarize_session(_record("s", [1.0, 2.0, 3.0]), CAUSAL, mode=LoadMode.SATURATING)
    assert summary.frames == 9 + 12 + 12
    # The uniform profile over-counts the same three chunks.
    assert summarize_session(_record("s", [1.0, 2.0, 3.0]), PROFILE, mode=LoadMode.SATURATING).frames == 36


def test_non_causal_profile_is_the_default_and_stays_uniform() -> None:
    assert PROFILE.is_causal is False
    assert PROFILE.frames_at(0) == PROFILE.frames_at(5) == 12
    assert PROFILE.cumulative_frames(3) == 36


def test_causal_opening_chunk_is_not_credited_with_a_full_period() -> None:
    """The opening chunk buys 9/16 s of playback, so chunk 1 is due sooner."""
    record = _record("s", [1.0, 1.5, 2.0])
    assert chunk_deadlines(record, CAUSAL)[1] == pytest.approx(1.0 + 9 / 16.0)
    assert chunk_deadlines(record, PROFILE)[1] == pytest.approx(1.0 + 12 / 16.0)


def test_causal_profile_from_spec_matches_the_pipeline_geometry() -> None:
    spec = {"frames_per_block": 3, "vae_temporal_factor": 4}
    causal = load_profile_from_spec(spec, target_fps=16.0)
    assert (causal.frames_per_first_chunk, causal.frames_per_chunk) == (9, 12)
    uniform = load_profile_from_spec(spec, target_fps=16.0, causal=False)
    assert (uniform.frames_per_first_chunk, uniform.frames_per_chunk) == (12, 12)


def test_cpr_is_macro_averaged_so_a_lagging_session_is_not_diluted() -> None:
    """A session that produced few chunks must weigh the same as a busy one."""
    good = _record("good", [1.0] + [1.0 + 0.75 * i for i in range(1, 21)])
    bad = _record("bad", [1.0, 99.0])
    run = summarize_run([good, bad], PROFILE, mode=LoadMode.PACED, wall_s=100.0)
    # Micro-averaging over all chunks would report ~0.95; macro-averaging halves it.
    assert run.continuous_play_ratio == pytest.approx(0.5, abs=0.02)


def test_worst_case_latency_is_across_sessions_not_per_session() -> None:
    run = summarize_run(
        [
            _record("a", [1.0, 2.0], latencies=[1.0, 1.0]),
            _record("b", [1.0, 30.0], latencies=[1.0, 29.0]),
        ],
        PROFILE,
        mode=LoadMode.SATURATING,
        wall_s=30.0,
    )
    assert run.worst_case_chunk_latency_s == pytest.approx(29.0)


def test_frames_per_gpu_second_is_normalized_by_device_count() -> None:
    run = summarize_run([_record("a", [1.0, 2.0])], PROFILE, mode=LoadMode.SATURATING, wall_s=2.0, num_gpus=4)
    assert run.generated_fps == pytest.approx(12.0)
    assert run.frames_per_gpu_second == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saturating_mode_issues_the_next_tick_immediately() -> None:
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=0.25)
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.SATURATING,
        chunks_per_session=4,
        arrivals=burst_arrivals(1),
    )
    run = await drive(config, factory, clock)
    summary = run.sessions[0]
    assert summary.chunks == 4
    # Back to back: 4 x 0.25 s, not paced out to 4 x 0.75 s.
    assert run.wall_s == pytest.approx(1.0)
    assert summary.rtf == pytest.approx(1.0 / 3.0)


@pytest.mark.asyncio
async def test_paced_mode_holds_the_release_period_when_generation_is_faster() -> None:
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=0.25)
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.PACED,
        chunks_per_session=4,
        arrivals=burst_arrivals(1),
    )
    run = await drive(config, factory, clock)
    # Chunk k is not submitted before k x 0.75 s, so the run spans the grid.
    assert run.wall_s == pytest.approx(3.0 * 0.75 + 0.25)
    assert run.sessions[0].continuous_play_ratio == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_paced_mode_records_misses_when_generation_is_too_slow() -> None:
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=2.0)
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.PACED,
        chunks_per_session=4,
        arrivals=burst_arrivals(1),
    )
    run = await drive(config, factory, clock)
    summary = run.sessions[0]
    assert summary.rtf > 1.0
    assert summary.continuous_play_ratio == pytest.approx(0.0)
    assert summary.deadline_misses == summary.deadline_chunks == 3


@pytest.mark.asyncio
async def test_a_lost_session_is_recorded_as_a_result_not_a_crash() -> None:
    clock = VirtualClock()
    factory, created = make_factory(clock, tick_s=0.25, fail_at=2)
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.SATURATING,
        chunks_per_session=5,
        arrivals=burst_arrivals(1),
    )
    run = await drive(config, factory, clock)
    assert run.sessions_lost == 1
    assert run.sessions[0].chunks == 2
    assert "fake rollout lost" in run.sessions[0].lost_reason
    # The session is still released even though its rollout was lost.
    assert created["bench-0"].closed is True


@pytest.mark.asyncio
async def test_peak_concurrency_is_reported_and_reflects_arrivals() -> None:
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=0.25)
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.SATURATING,
        chunks_per_session=2,
        arrivals=burst_arrivals(3),
    )
    run = await drive(config, factory, clock)
    assert run.peak_concurrent_sessions == 3
    assert len(run.sessions) == 3


@pytest.mark.asyncio
async def test_staggered_arrivals_lower_peak_concurrency() -> None:
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=0.25)
    # Second session arrives after the first has finished its two ticks.
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.SATURATING,
        chunks_per_session=2,
        arrivals=ArrivalPlan((0.0, 10.0)),
    )
    run = await drive(config, factory, clock)
    assert run.peak_concurrent_sessions == 1


def test_poisson_arrivals_are_deterministic_and_ordered() -> None:
    first = poisson_arrivals(5, rate_per_s=1.0, seed=7)
    assert first.offsets_s == poisson_arrivals(5, rate_per_s=1.0, seed=7).offsets_s
    assert list(first.offsets_s) == sorted(first.offsets_s)
    assert first.offsets_s != poisson_arrivals(5, rate_per_s=1.0, seed=8).offsets_s


def test_arrival_plan_rejects_unordered_offsets() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        ArrivalPlan((1.0, 0.0))


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_jsonl_carries_deadlines_in_paced_mode(tmp_path) -> None:
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=0.25)
    config = BenchmarkConfig(
        profile=PROFILE,
        mode=LoadMode.PACED,
        chunks_per_session=3,
        arrivals=burst_arrivals(1),
        events_dir=tmp_path / "events",
    )
    await drive(config, factory, clock)
    lines = (tmp_path / "events" / "bench-0.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert "deadline" not in records[0]  # prebuffer chunk
    assert records[1]["met_deadline"] is True
    assert set(records[0]) >= {"session_id", "chunk_index", "t_submit", "t_ready", "latency_s"}


def test_run_summary_is_json_serializable_and_states_its_conditions() -> None:
    run = summarize_run(
        [_record("a", [1.0, 2.0], submit_gap=0.5)],
        PROFILE,
        mode=LoadMode.PACED,
        wall_s=2.0,
        num_gpus=2,
        notes=("checkpoint=X", "resolution=480x832"),
    )
    payload = json.dumps(run.to_dict())
    restored = json.loads(payload)
    assert restored["mode"] == "paced"
    assert restored["num_gpus"] == 2
    assert restored["profile"]["release_period_s"] == pytest.approx(0.75)
    assert restored["notes"] == ["checkpoint=X", "resolution=480x832"]


# --------------------------------------------------------------------------
# The N=1 vs N=2 comparison this harness exists to produce
# --------------------------------------------------------------------------


def test_serialized_ticks_keep_aggregate_fps_flat_and_double_rtf() -> None:
    """The expected N=2 result when state concurrency is 2 but execution is 1.

    Two sessions resident, ticks serialized: the device produces the same
    frames per second, and each session sees twice its solo RTF.
    """
    # Long enough for steady state: the first chunk's head start (it meets no
    # contention) washes out, which is why the 2x relation is asymptotic.
    k = 20
    tick = 0.5
    solo = summarize_run(
        [_record("a", [tick * (i + 1) for i in range(k)], submit_gap=tick)],
        PROFILE,
        mode=LoadMode.SATURATING,
        wall_s=tick * k,
    )
    # Interleaved: a chunk now waits behind one other session's tick.
    a_ready = [tick + i * 2 * tick for i in range(k)]
    b_ready = [2 * tick + i * 2 * tick for i in range(k)]
    pair = summarize_run(
        [
            _record("a", a_ready, latencies=[tick] + [2 * tick] * (k - 1)),
            _record("b", b_ready, latencies=[2 * tick] * k),
        ],
        PROFILE,
        mode=LoadMode.SATURATING,
        wall_s=2 * tick * k,
    )
    assert pair.generated_fps == pytest.approx(solo.generated_fps)
    assert pair.sessions[0].rtf == pytest.approx(solo.sessions[0].rtf * 2, rel=0.05)

    delta = compare_runs(solo, pair)
    assert delta["generated_fps_ratio"] == pytest.approx(1.0)
    # Latency grew only by queueing behind one other session, so effectively no
    # switching cost is left over.
    assert abs(delta["per_tick_switching_cost_s"]) < 0.05 * tick


def test_switching_cost_is_the_latency_beyond_pure_queueing() -> None:
    """A measurable drop beyond queueing is the quantity batching has to beat."""
    solo = summarize_run(
        [_record("a", [0.5, 1.0], submit_gap=0.5)],
        PROFILE,
        mode=LoadMode.SATURATING,
        wall_s=1.0,
    )
    # Each chunk costs 1.2 s instead of the 1.0 s pure serialization predicts.
    pair = summarize_run(
        [
            _record("a", [1.2, 2.4], submit_gap=1.2),
            _record("b", [1.2, 2.4], submit_gap=1.2),
        ],
        PROFILE,
        mode=LoadMode.SATURATING,
        wall_s=2.4,
    )
    delta = compare_runs(solo, pair)
    assert delta["per_tick_switching_cost_s"] == pytest.approx(0.2)
    assert delta["generated_fps_ratio"] < 1.0


# --------------------------------------------------------------------------
# CLI argument -> config path (no engine, no device)
# --------------------------------------------------------------------------


def _args(*argv: str):
    from benchmarks.ar_diffusion.run_realtime_benchmark import parse_args

    return parse_args(["--model", "m", "--prompt", "p", "--note", "hw=test", *argv])


def test_cli_defaults_to_a_single_session_burst() -> None:
    from benchmarks.ar_diffusion.run_realtime_benchmark import build_config

    config = build_config(_args(), frames_per_chunk=12)
    assert config.mode is LoadMode.SATURATING
    assert len(config.arrivals) == 1
    assert config.arrivals.offsets_s == (0.0,)
    assert config.profile.release_period_s == pytest.approx(0.75)


def test_cli_requires_run_conditions() -> None:
    """A latency without model, hardware and resolution is not a result."""
    from benchmarks.ar_diffusion.run_realtime_benchmark import build_config, parse_args

    bare = parse_args(["--model", "m", "--prompt", "p"])
    with pytest.raises(ValueError, match="--note is required"):
        build_config(bare, frames_per_chunk=12)


def test_release_period_override_keeps_one_source_of_truth() -> None:
    from benchmarks.ar_diffusion.run_realtime_benchmark import build_config

    config = build_config(_args("--release-period", "0.5"), frames_per_chunk=12)
    assert config.profile.release_period_s == pytest.approx(0.5)
    assert config.profile.target_fps == pytest.approx(24.0)


def test_arrival_rate_switches_to_poisson() -> None:
    from benchmarks.ar_diffusion.run_realtime_benchmark import build_config

    config = build_config(_args("--num-sessions", "4", "--arrival-rate", "1.0"), frames_per_chunk=12)
    assert len(config.arrivals) == 4
    assert config.arrivals.offsets_s[0] == 0.0
    assert any(offset > 0 for offset in config.arrivals.offsets_s)


def test_frames_per_chunk_needs_a_mapping_the_spec_does_not_declare() -> None:
    """ARDiffusionKVCacheSpec counts latent frames and declares no conversion.

    The harness takes it from the caller instead of guessing. Note the result
    is a pair, not a scalar: the conversion is causal, so a single declared
    integer could not express it.
    """
    from benchmarks.ar_diffusion.run_realtime_benchmark import frames_per_chunk_from_spec

    class Spec:
        frames_per_block = 3

    assert frames_per_chunk_from_spec(Spec()) == (3, 3)
    assert frames_per_chunk_from_spec(Spec(), vae_temporal_factor=4) == (12, 9)
    assert frames_per_chunk_from_spec(Spec(), vae_temporal_factor=4, causal=False) == (12, 12)
    with pytest.raises(ValueError, match="vae_temporal_factor"):
        frames_per_chunk_from_spec(Spec(), vae_temporal_factor=0)

    class NoSpec:
        pass

    with pytest.raises(ValueError, match="frames_per_block"):
        frames_per_chunk_from_spec(NoSpec())


# --------------------------------------------------------------------------
# Vertical path: latents -> incremental decode -> delivered frames
# --------------------------------------------------------------------------


class FakeDecodeState:
    """Bounded cache stand-in: grows once, then plateaus."""

    def __init__(self) -> None:
        self.chunks = 0

    def nbytes(self) -> int:
        return 0 if self.chunks == 0 else 4096


class FakeChunkDecoder:
    """Applies the causal geometry the real decoder has, without torch."""

    def __init__(self, clock: VirtualClock, *, decode_s: float, factor: int = 4) -> None:
        self._clock = clock
        self._decode_s = decode_s
        self._factor = factor
        self.released: list[FakeDecodeState] = []

    def new_decode_state(self, session_id: str) -> FakeDecodeState:
        return FakeDecodeState()

    def decode_chunk(self, latent, state: FakeDecodeState):
        n = latent["latent_frames"]
        frames = (n - 1) * self._factor + 1 if state.chunks == 0 else n * self._factor
        state.chunks += 1
        # Charge decode time on the same virtual clock the driver reads.
        self._clock.now += self._decode_s
        return type("Frames", (), {"shape": (1, 3, frames, 8, 8)})()

    def release(self, state: FakeDecodeState) -> None:
        self.released.append(state)


class FakeLatentSession:
    def __init__(self, clock: VirtualClock, *, generate_s: float, latent_frames: int = 3) -> None:
        self._clock = clock
        self._generate_s = generate_s
        self._latent_frames = latent_frames
        self.closed = False

    async def next_chunk(self):
        self._clock.now += self._generate_s
        return {"latent_frames": self._latent_frames}

    async def close(self) -> None:
        self.closed = True


def _decoding_factory(clock: VirtualClock, *, generate_s: float, decode_s: float):
    from benchmarks.ar_diffusion.decoding_session import DecodingSession

    decoder = FakeChunkDecoder(clock, decode_s=decode_s)
    inner: dict[str, FakeLatentSession] = {}

    async def factory(session_id: str):
        session = FakeLatentSession(clock, generate_s=generate_s)
        inner[session_id] = session
        return DecodingSession(
            inner=session,
            decoder=decoder,
            session_id=session_id,
            clock=clock.time,
        )

    return factory, decoder, inner


@pytest.mark.asyncio
async def test_ttfc_covers_decode_because_a_latent_is_not_a_frame() -> None:
    """The gate is time to first *frame*, so decode is inside the measured path."""
    clock = VirtualClock()
    factory, _, _ = _decoding_factory(clock, generate_s=0.4, decode_s=0.1)
    config = BenchmarkConfig(
        profile=PROFILE, mode=LoadMode.SATURATING, chunks_per_session=3, arrivals=burst_arrivals(1)
    )
    run = await drive(config, factory, clock)
    assert run.sessions[0].ttfc_s == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_decode_share_is_the_headroom_overlap_can_recover() -> None:
    clock = VirtualClock()
    factory, _, _ = _decoding_factory(clock, generate_s=0.4, decode_s=0.1)
    config = BenchmarkConfig(
        profile=PROFILE, mode=LoadMode.SATURATING, chunks_per_session=4, arrivals=burst_arrivals(1)
    )
    run = await drive(config, factory, clock)
    summary = run.sessions[0]
    assert summary.generate_s_total == pytest.approx(4 * 0.4)
    assert summary.decode_s_total == pytest.approx(4 * 0.1)
    assert summary.decode_share == pytest.approx(0.2)
    assert run.decode_share == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_delivered_frames_are_measured_not_assumed() -> None:
    """The decoder's temporal geometry is not declared, so it is measured.

    The profile here is the uniform one, which would report 12 frames for every
    chunk; the decoder actually delivers 9 then 12, and the summary must follow
    the decoder.
    """
    clock = VirtualClock()
    factory, _, _ = _decoding_factory(clock, generate_s=0.1, decode_s=0.05)
    config = BenchmarkConfig(
        profile=PROFILE, mode=LoadMode.SATURATING, chunks_per_session=3, arrivals=burst_arrivals(1)
    )
    run = await drive(config, factory, clock)
    assert run.sessions[0].frames == 9 + 12 + 12
    assert PROFILE.cumulative_frames(3) == 36  # what assuming would have given


@pytest.mark.asyncio
async def test_resident_decoder_bytes_reach_the_summary() -> None:
    """Admission needs this term, and nothing else in the run reports it."""
    clock = VirtualClock()
    factory, _, _ = _decoding_factory(clock, generate_s=0.1, decode_s=0.05)
    config = BenchmarkConfig(
        profile=PROFILE, mode=LoadMode.SATURATING, chunks_per_session=3, arrivals=burst_arrivals(2)
    )
    run = await drive(config, factory, clock)
    assert run.resident_decoder_bytes_per_session == 4096
    assert json.loads(json.dumps(run.to_dict()))["resident_decoder_bytes_per_session"] == 4096


@pytest.mark.asyncio
async def test_closing_a_session_releases_decoder_state_with_it() -> None:
    """The cache has no recompute source, so it must not outlive the session."""
    clock = VirtualClock()
    factory, decoder, inner = _decoding_factory(clock, generate_s=0.1, decode_s=0.05)
    config = BenchmarkConfig(
        profile=PROFILE, mode=LoadMode.SATURATING, chunks_per_session=2, arrivals=burst_arrivals(2)
    )
    await drive(config, factory, clock)
    assert len(decoder.released) == 2
    assert all(session.closed for session in inner.values())


@pytest.mark.asyncio
async def test_a_latent_only_session_still_works_and_falls_back_to_the_profile() -> None:
    """Sessions without decode report no split, and the profile fills in frames."""
    clock = VirtualClock()
    factory, _ = make_factory(clock, tick_s=0.25)
    config = BenchmarkConfig(
        profile=PROFILE, mode=LoadMode.SATURATING, chunks_per_session=3, arrivals=burst_arrivals(1)
    )
    run = await drive(config, factory, clock)
    summary = run.sessions[0]
    assert summary.decode_share is None
    assert summary.generate_s_total is None
    assert summary.frames == 36
    assert run.resident_decoder_bytes_per_session is None
