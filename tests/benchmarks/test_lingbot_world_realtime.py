# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the LingBot-World realtime streaming benchmark.

The arithmetic is tested against the server's own formulas rather than against
hand-copied expectations, and the protocol handling runs against a real
WebSocket server that replays ``/v1/realtime/video`` with scripted delays -- so
a benchmark number can be wrong only if the server itself is, not because the
client parsed incorrectly an event or timed the wrong edge.
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import asyncio
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_world.benchmark_lingbot_world_realtime import (
    BenchmarkError,
    build_workload,
    print_report,
    endpoint_url,
    parse_args,
    resolve_image_reference,
    run_session,
)
from benchmarks.lingbot_world.workload import (
    DEFAULT_WARMUP_CHUNKS,
    FRAMES_PER_BLOCK,
    ChunkRecord,
    PromptUpdate,
    Workload,
    aggregate_metrics,
    build_camera_script,
    chunks_for_num_frames,
    compute_metrics,
    default_workload,
    load_workload,
    num_frames_for_chunks,
    percentile,
    pixel_frames_in_chunk,
    simulate_playback,
    steady_chunk_deadline_ms,
)

_IMAGE = "data:image/png;base64,Zm9v"


def _workload(num_chunks: int = 8, fps: int = 16) -> Workload:
    return default_workload(image_reference=_IMAGE, num_chunks=num_chunks, fps=fps)


# --------------------------------------------------------------------------- #
# Chunk and frame arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_chunks", list(range(1, 33)))
def test_frame_count_round_trips_through_the_server_formula(num_chunks: int) -> None:
    """``num_frames`` must land on a rollout of exactly the requested chunks.

    The server derives chunk count as ``((num_frames - 1) // 4 + 1) // 3``; if
    the benchmark's inverse is off by one frame the run silently generates a
    different number of blocks than it reports.
    """
    assert chunks_for_num_frames(num_frames_for_chunks(num_chunks)) == num_chunks


def test_recipe_frame_counts_match_the_documented_chunk_counts() -> None:
    # The recipe states three chunks for num_frames 33 and seven for 81.
    assert chunks_for_num_frames(33) == 3
    assert chunks_for_num_frames(81) == 7
    assert num_frames_for_chunks(3) == 33
    assert num_frames_for_chunks(7) == 81


def test_first_chunk_is_short_because_the_decoder_expands_it_once() -> None:
    assert pixel_frames_in_chunk(0) == 9
    assert pixel_frames_in_chunk(1) == 12
    assert pixel_frames_in_chunk(7) == 12
    # A whole rollout's frames are the sum of its chunks' frames.
    assert sum(pixel_frames_in_chunk(index) for index in range(5)) == num_frames_for_chunks(5)


def test_camera_script_holds_one_three_entry_action_list_per_chunk() -> None:
    script = build_camera_script(4, "orbit")
    assert len(script) == 4
    assert all(len(actions) == FRAMES_PER_BLOCK for actions in script)
    # orbit advances through the action keys so successive chunks differ.
    assert script[0] != script[1]


def test_unknown_camera_pattern_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown camera pattern"):
        build_camera_script(2, "sideways")


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #


def test_session_start_payload_matches_the_documented_protocol() -> None:
    workload = _workload(num_chunks=3)
    payload = workload.session_start_payload("robbyant/lingbot-world-v2-14b-causal-fast-diffusers")
    assert payload["type"] == "session.start"
    assert payload["num_frames"] == 33
    assert payload["image_reference"] == {"image_url": _IMAGE}
    assert payload["extra_params"]["flow_shift"] == 5.0
    assert len(payload["extra_params"]["camera_action_script"]) == 3


def test_workload_rejects_a_camera_script_with_the_wrong_block_width() -> None:
    with pytest.raises(ValueError, match="exactly 3 action lists"):
        Workload(prompt="p", image_reference=_IMAGE, camera_script=[[["w"], ["w"]]])


def test_workload_rejects_geometry_the_cache_cannot_serve() -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        Workload(prompt="p", image_reference=_IMAGE, camera_script=build_camera_script(1, "forward"), width=831)


def test_prompt_update_beyond_the_rollout_is_rejected() -> None:
    with pytest.raises(ValueError, match="never fires"):
        Workload(
            prompt="p",
            image_reference=_IMAGE,
            camera_script=build_camera_script(2, "forward"),
            prompt_updates=(PromptUpdate(after_chunk=5, prompt="later"),),
        )


@pytest.mark.parametrize("from_file", [False, True])
def test_duplicate_prompt_update_boundaries_are_rejected(tmp_path: Path, from_file: bool) -> None:
    updates = [
        {"after_chunk": 1, "prompt": "Rain begins"},
        {"after_chunk": 0, "prompt": "Clouds gather"},
        {"after_chunk": 1, "prompt": "Rain stops"},
    ]
    with pytest.raises(ValueError, match="Duplicate prompt update after chunk 1"):
        if from_file:
            path = tmp_path / "rollout.json"
            path.write_text(json.dumps({"image": _IMAGE, "num_chunks": 3, "prompt_updates": updates}))
            build_workload(parse_args(["--workload", str(path)]))
        else:
            Workload(
                prompt="p",
                image_reference=_IMAGE,
                camera_script=build_camera_script(3, "forward"),
                prompt_updates=(
                    PromptUpdate(after_chunk=1, prompt="Rain begins"),
                    PromptUpdate(after_chunk=0, prompt="Clouds gather"),
                    PromptUpdate(after_chunk=1, prompt="Rain stops"),
                ),
            )


def test_workload_file_is_loaded_and_cli_overrides_win(tmp_path: Path) -> None:
    path = tmp_path / "rollout.json"
    path.write_text(
        json.dumps(
            {
                "prompt": "A quiet street",
                "image": "https://example.invalid/frame.png",
                "num_chunks": 4,
                "camera_pattern": "hold",
                "fps": 12,
                "seed": 7,
                "prompt_updates": [{"after_chunk": 1, "prompt": "Rain begins", "transition_chunks": 2}],
            }
        )
    )
    workload = load_workload(path, overrides={"fps": 24, "seed": None})
    assert workload.prompt == "A quiet street"
    assert workload.num_chunks == 4
    assert workload.fps == 24, "an explicit CLI value must beat the file"
    assert workload.seed == 7, "a None override must leave the file value alone"
    assert workload.prompt_updates[0].transition_chunks == 2


def test_workload_file_without_a_rollout_length_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({"image": "https://example.invalid/frame.png"}))
    with pytest.raises(ValueError, match="camera_action_script or num_chunks"):
        load_workload(path)


def test_local_image_is_inlined_as_a_data_url(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    reference = resolve_image_reference(str(image))
    assert reference.startswith("data:image/png;base64,")
    assert resolve_image_reference("https://example.invalid/a.png") == "https://example.invalid/a.png"
    with pytest.raises(BenchmarkError, match="neither a URL nor a readable file"):
        resolve_image_reference(str(tmp_path / "missing.png"))


# --------------------------------------------------------------------------- #
# Metric math
# --------------------------------------------------------------------------- #


def test_rtf_is_a_cost_not_a_speed() -> None:
    """RTF must match vLLM-Omni's own formula, where lower is better.

    ``vllm_omni/metrics/definitions.py::compute_audio_rtf`` is
    ``latency / duration`` with the docstring "SLO red line < 1", and
    ``benchmarks/patch/patch.py`` sets ``video_rtf = generation_s /
    video_duration``. Emitting the reciprocal under the same field name would
    silently invert every dashboard that aggregates it, so the direction is
    pinned here rather than left to a comment.
    """
    # Two chunks, 9 + 12 = 21 frames at 12 fps = 1.75 s of video, generated in 3.5 s.
    slow = compute_metrics(_records([1.0, 2.5]), fps=12, warmup_chunks=0, session_wall_s=3.5)
    assert slow["video_seconds"] == pytest.approx(1.75)
    assert slow["video_rtf"] == pytest.approx(2.0), "twice as slow as real time"
    assert slow["video_rtfx"] == pytest.approx(0.5)

    fast = compute_metrics(_records([0.4, 0.475]), fps=12, warmup_chunks=0, session_wall_s=0.875)
    assert fast["video_rtf"] == pytest.approx(0.5), "twice as fast as real time"
    assert fast["video_rtf"] < 1.0 < slow["video_rtf"]
    assert fast["video_rtf"] * fast["video_rtfx"] == pytest.approx(1.0)


def test_steady_rtf_is_the_chunk_interval_over_its_deadline() -> None:
    """A steady chunk's RTF is just its interval divided by its video duration."""
    metrics = compute_metrics(_records([1.0] + [1.357] * 15), fps=12, warmup_chunks=6)
    # 12 frames at 12 fps = 1000 ms of video per steady chunk.
    assert metrics["chunk_deadline_ms"] == pytest.approx(1000.0)
    assert metrics["steady_video_rtf"] == pytest.approx(1.357, abs=1e-3)


def test_percentile_interpolates_so_short_runs_still_report_tails() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 1.0) == 40.0
    assert percentile(values, 0.5) == 25.0
    assert percentile([5.0], 0.99) == 5.0


def _records(intervals: list[float]) -> list[ChunkRecord]:
    """One record per entry; ``intervals[0]`` is the TTFC, the rest are gaps."""
    records: list[ChunkRecord] = []
    arrival = 0.0
    for index, interval in enumerate(intervals):
        arrival += interval
        records.append(
            ChunkRecord(
                index=index,
                arrival_s=arrival,
                inter_arrival_s=interval,
                byte_length=1000,
                num_frames=pixel_frames_in_chunk(index),
            )
        )
    return records


def test_steady_state_excludes_the_unsaturated_attention_window() -> None:
    """Warmup chunks must not drag the steady-state mean down.

    The first six chunks attend over a growing window, so they are genuinely
    cheaper. A benchmark that averages them in reports a cadence the session
    will never sustain.
    """
    intervals = [1.0] + [0.2] * (DEFAULT_WARMUP_CHUNKS - 1) + [0.8] * 4
    metrics = compute_metrics(_records(intervals), fps=16, warmup_chunks=DEFAULT_WARMUP_CHUNKS)
    assert metrics["interval_steady"]["count"] == 3
    assert metrics["interval_steady"]["mean_ms"] == pytest.approx(800.0)
    assert metrics["interval_all"]["mean_ms"] < metrics["interval_steady"]["mean_ms"]


def test_short_rollouts_report_that_steady_state_was_never_reached() -> None:
    metrics = compute_metrics(_records([1.0, 0.5, 0.5]), fps=16, warmup_chunks=DEFAULT_WARMUP_CHUNKS)
    assert metrics["steady_state_reached"] is False
    assert metrics["interval_steady"] == {}
    assert "slo_attainment" not in metrics


def test_real_time_factor_and_deadline_use_the_true_frame_counts() -> None:
    # Eight chunks: 9 + 7*12 = 93 frames at 16 fps = 5.8125 s of video.
    metrics = compute_metrics(_records([1.0] + [0.5] * 7), fps=16, warmup_chunks=0, session_wall_s=5.8125)
    assert metrics["total_frames"] == 93
    assert metrics["video_seconds"] == pytest.approx(5.8125)
    assert metrics["video_rtf"] == pytest.approx(1.0)
    assert metrics["video_rtfx"] == pytest.approx(1.0)
    # A steady chunk carries 12 frames, so real time allows 750 ms at 16 fps.
    assert metrics["chunk_deadline_ms"] == pytest.approx(750.0)
    assert steady_chunk_deadline_ms(12) == pytest.approx(1000.0)


def test_slo_attainment_counts_only_steady_chunks() -> None:
    intervals = [1.0] + [0.1] * 5 + [0.7, 0.9, 0.7, 0.9]
    metrics = compute_metrics(_records(intervals), fps=16, warmup_chunks=6, slo_ms=750.0)
    assert metrics["slo_attainment"] == pytest.approx(2 / 3)
    assert metrics["slo_violations"] == 1


def test_playback_simulation_finds_stalls_a_percentile_would_hide() -> None:
    """One late chunk behind a one-chunk buffer is a visible stall."""
    # 12 frames at 16 fps = 750 ms of playback per chunk; chunk 3 takes 2 s.
    intervals = [1.0, 0.5, 0.5, 2.0, 0.5]
    result = simulate_playback(_records(intervals), fps=16, buffer_chunks=1)
    assert result.underrun_count >= 1
    assert result.first_underrun_chunk == 3
    assert result.total_stall_s > 0.0


def test_a_deeper_buffer_absorbs_the_same_jitter() -> None:
    intervals = [1.0, 0.5, 0.5, 1.4, 0.2, 0.2]
    fast = simulate_playback(_records(intervals), fps=16, buffer_chunks=1)
    buffered = simulate_playback(_records(intervals), fps=16, buffer_chunks=3)
    assert buffered.underrun_count < fast.underrun_count


def test_playback_rejects_a_buffer_deeper_than_the_rollout() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        simulate_playback(_records([1.0, 0.5]), fps=16, buffer_chunks=9)


def test_aggregate_pools_sessions_and_reports_spread() -> None:
    fast = compute_metrics(_records([1.0] + [0.1] * 5 + [0.6] * 4), fps=16, warmup_chunks=6)
    slow = compute_metrics(_records([1.0] + [0.1] * 5 + [0.9] * 4), fps=16, warmup_chunks=6)
    aggregate = aggregate_metrics([fast, slow], fps=16)
    assert aggregate["sessions"] == 2
    assert aggregate["steady_interval_ms_mean"] == pytest.approx(750.0)
    assert aggregate["steady_interval_ms_spread"] == pytest.approx(300.0)
    assert aggregate["video_rtf_best"] <= aggregate["video_rtf_mean"] <= aggregate["video_rtf_worst"]


def test_single_session_aggregate_is_the_session_itself() -> None:
    metrics = compute_metrics(_records([1.0] + [0.5] * 3), fps=16, warmup_chunks=0)
    aggregate = aggregate_metrics([metrics], fps=16)
    assert aggregate["sessions"] == 1
    assert aggregate["video_rtf"] == metrics["video_rtf"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_endpoint_url_normalizes_http_and_appends_the_ws_path() -> None:
    assert endpoint_url(parse_args([])) == "ws://127.0.0.1:8000/v1/realtime/video"
    assert endpoint_url(parse_args(["--host", "h", "--port", "9"])) == "ws://h:9/v1/realtime/video"
    args = parse_args(["--base-url", "http://example.invalid:8099"])
    assert endpoint_url(args) == "ws://example.invalid:8099/v1/realtime/video"
    args = parse_args(["--base-url", "ws://example.invalid:8099/v1/realtime/video"])
    assert endpoint_url(args) == "ws://example.invalid:8099/v1/realtime/video"


def test_cli_rejects_a_zero_chunk_buffer() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--playback-buffer-chunks", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--target-fps", "0"])


def test_the_real_time_basis_is_independent_of_the_mux_label() -> None:
    """The checkpoint declares no frame rate, so the basis must be statable.

    The same measured cadence has RTF 1.809 at 16 fps and 1.357 at 12 fps;
    tying that verdict to the fps sent to the muxer would let a playback label
    silently decide whether a run passes.
    """
    records = _records([1.0] + [1.357] * 15)
    at_16 = compute_metrics(records, fps=16, warmup_chunks=6)
    at_12 = compute_metrics(records, fps=12, warmup_chunks=6)
    assert at_16["chunk_deadline_ms"] == pytest.approx(750.0)
    assert at_12["chunk_deadline_ms"] == pytest.approx(1000.0)
    # RTF is cost, so the harsher 16 fps basis scores WORSE (higher).
    assert at_16["video_rtf"] > at_12["video_rtf"]
    # Same wall clock, same bytes: only the basis moved.
    assert at_16["video_rtf"] / at_12["video_rtf"] == pytest.approx(16 / 12)
    assert parse_args([]).target_fps is None, "default defers to the workload fps"


# --------------------------------------------------------------------------- #
# Protocol handling against a scripted server
# --------------------------------------------------------------------------- #


def _serve(script):
    """Return an async context manager yielding a ws:// URL for ``script``."""
    from websockets.asyncio.server import Server, serve

    class _Server:
        def __init__(self) -> None:
            self.server: Server | None = None
            self.start_payloads: list[dict] = []

        async def __aenter__(self) -> str:
            async def handler(websocket) -> None:
                self.start_payloads.append(json.loads(await websocket.recv()))
                await script(websocket, self.start_payloads[-1])

            self.server = await serve(handler, "127.0.0.1", 0)
            port = next(iter(self.server.sockets)).getsockname()[1]
            return f"ws://127.0.0.1:{port}/v1/realtime/video"

        async def __aexit__(self, *exc) -> None:
            assert self.server is not None
            self.server.close()
            await self.server.wait_closed()

    return _Server()


async def _emit_rollout(websocket, payload, *, intervals=None, chunks=None, trailer=True) -> None:
    """Replay the documented event order: start, metadata+binary per chunk, done."""
    num_chunks = chunks if chunks is not None else len(payload["extra_params"]["camera_action_script"])
    await websocket.send(json.dumps({"type": "video.start", "request_id": "req-1", "format": "m4s"}))
    for index in range(num_chunks):
        if intervals is not None:
            await asyncio.sleep(intervals[index])
        frames = pixel_frames_in_chunk(index)
        body = bytes([index % 256]) * (100 + index)
        await websocket.send(
            json.dumps(
                {
                    "type": "video.chunk_metadata",
                    "request_id": "req-1",
                    "kind": "media",
                    "num_frames": frames,
                    "byte_length": len(body),
                    "generation_chunk_index": index,
                }
            )
        )
        await websocket.send(body)
    if trailer:
        tail = b"\x00" * 8
        await websocket.send(
            json.dumps(
                {
                    "type": "video.chunk_metadata",
                    "request_id": "req-1",
                    "kind": "trailer",
                    "num_frames": 0,
                    "byte_length": len(tail),
                    "generation_chunk_index": None,
                }
            )
        )
        await websocket.send(tail)
    await websocket.send(json.dumps({"type": "session.done", "stopped": False, "chunks": num_chunks}))


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


def test_session_records_one_chunk_per_block_and_ignores_the_trailer() -> None:
    workload = _workload(num_chunks=4)

    async def scenario() -> None:
        server = _serve(_emit_rollout)
        url = await server.__aenter__()
        try:
            result = await run_session(
                url,
                "model",
                workload,
                first_chunk_timeout=10,
                chunk_timeout=10,
                ping_interval=0,
                collect_media=True,
                print_chunks=False,
            )
        finally:
            await server.__aexit__(None, None, None)

        assert [record.index for record in result.records] == [0, 1, 2, 3]
        assert [record.num_frames for record in result.records] == [9, 12, 12, 12]
        assert result.request_id == "req-1"
        assert result.ttfc_s is not None and result.ttfc_s > 0
        # The trailer's bytes are kept for remuxing but carry no chunk record.
        assert len(result.media) == sum(record.byte_length for record in result.records) + 8
        assert server.start_payloads[0]["num_frames"] == workload.num_frames

    _run(scenario())


def test_measured_intervals_track_the_server_cadence() -> None:
    workload = _workload(num_chunks=3)

    async def script(websocket, payload) -> None:
        await _emit_rollout(websocket, payload, intervals=[0.05, 0.30, 0.05])

    async def scenario() -> None:
        server = _serve(script)
        url = await server.__aenter__()
        try:
            result = await run_session(
                url,
                "model",
                workload,
                first_chunk_timeout=10,
                chunk_timeout=10,
                ping_interval=0,
                collect_media=False,
                print_chunks=False,
            )
        finally:
            await server.__aexit__(None, None, None)

        intervals = [record.inter_arrival_s for record in result.records]
        assert intervals[1] == pytest.approx(0.30, abs=0.15)
        assert intervals[2] < intervals[1], "a fast chunk must not inherit the slow one's interval"
        assert result.records[0].arrival_s == pytest.approx(result.ttfc_s)

    _run(scenario())


def test_a_short_rollout_fails_loudly_instead_of_reporting_a_partial_cadence() -> None:
    workload = _workload(num_chunks=4)

    async def script(websocket, payload) -> None:
        await _emit_rollout(websocket, payload, chunks=2)

    async def scenario() -> None:
        server = _serve(script)
        url = await server.__aenter__()
        try:
            with pytest.raises(BenchmarkError, match="Expected 4 chunks"):
                await run_session(
                    url,
                    "model",
                    workload,
                    first_chunk_timeout=10,
                    chunk_timeout=10,
                    ping_interval=0,
                    collect_media=False,
                    print_chunks=False,
                )
        finally:
            await server.__aexit__(None, None, None)

    _run(scenario())


def test_a_server_error_event_aborts_the_session() -> None:
    workload = _workload(num_chunks=2)

    async def script(websocket, payload) -> None:
        await websocket.send(json.dumps({"type": "error", "message": "resolution mismatch"}))

    async def scenario() -> None:
        server = _serve(script)
        url = await server.__aenter__()
        try:
            with pytest.raises(BenchmarkError, match="resolution mismatch.*requested 2 chunks"):
                await run_session(
                    url,
                    "model",
                    workload,
                    first_chunk_timeout=10,
                    chunk_timeout=10,
                    ping_interval=0,
                    collect_media=False,
                    print_chunks=False,
                )
        finally:
            await server.__aexit__(None, None, None)

    _run(scenario())


def test_a_stalled_server_times_out_rather_than_hanging() -> None:
    workload = _workload(num_chunks=2)

    async def script(websocket, payload) -> None:
        await websocket.send(json.dumps({"type": "video.start", "request_id": "req-1", "format": "m4s"}))
        await asyncio.sleep(5)

    async def scenario() -> None:
        server = _serve(script)
        url = await server.__aenter__()
        try:
            with pytest.raises(BenchmarkError, match="No media chunk for"):
                await run_session(
                    url,
                    "model",
                    workload,
                    first_chunk_timeout=0.4,
                    chunk_timeout=0.4,
                    ping_interval=0,
                    collect_media=False,
                    print_chunks=False,
                )
        finally:
            await server.__aexit__(None, None, None)

    _run(scenario())


def test_keepalive_pings_reach_the_server_during_a_slow_first_chunk() -> None:
    """A compiled first block can outlast the server's stall timeout."""
    workload = _workload(num_chunks=1)
    seen: list[str] = []

    async def script(websocket, payload) -> None:
        await websocket.send(json.dumps({"type": "video.start", "request_id": "req-1", "format": "m4s"}))

        async def drain() -> None:
            async for message in websocket:
                event = json.loads(message)
                seen.append(event["type"])
                if event["type"] == "session.ping":
                    await websocket.send(json.dumps({"type": "session.pong"}))

        drainer = asyncio.create_task(drain())
        await asyncio.sleep(0.35)
        await _emit_rollout(websocket, payload, trailer=False)
        drainer.cancel()

    async def scenario() -> None:
        server = _serve(script)
        url = await server.__aenter__()
        try:
            result = await run_session(
                url,
                "model",
                workload,
                first_chunk_timeout=10,
                chunk_timeout=10,
                ping_interval=0.1,
                collect_media=False,
                print_chunks=False,
            )
        finally:
            await server.__aexit__(None, None, None)
        assert len(result.records) == 1
        assert "session.ping" in seen

    _run(scenario())


def test_prompt_updates_are_sent_on_their_chunk_boundary() -> None:
    workload = Workload(
        prompt="start",
        image_reference=_IMAGE,
        camera_script=build_camera_script(3, "forward"),
        prompt_updates=(PromptUpdate(after_chunk=0, prompt="Rain begins", transition_chunks=1),),
    )
    received: list[dict] = []

    async def script(websocket, payload) -> None:
        async def drain() -> None:
            async for message in websocket:
                received.append(json.loads(message))

        drainer = asyncio.create_task(drain())
        await _emit_rollout(websocket, payload, intervals=[0.02, 0.15, 0.02], trailer=False)
        await asyncio.sleep(0.05)
        drainer.cancel()

    async def scenario() -> None:
        server = _serve(script)
        url = await server.__aenter__()
        try:
            await run_session(
                url,
                "model",
                workload,
                first_chunk_timeout=10,
                chunk_timeout=10,
                ping_interval=0,
                collect_media=False,
                print_chunks=False,
            )
        finally:
            await server.__aexit__(None, None, None)
        interactions = [event for event in received if event.get("type") == "session.interaction"]
        assert len(interactions) == 1
        assert interactions[0]["interaction"]["event"]["prompt"] == "Rain begins"
        assert interactions[0]["interaction"]["transition_chunks"] == 1

    _run(scenario())


@pytest.mark.parametrize("delivered", [0, 1])
def test_pongs_do_not_extend_media_deadlines(delivered) -> None:
    seen = []

    async def script(websocket, payload) -> None:
        if delivered:
            await websocket.send(
                json.dumps(
                    {
                        "type": "video.chunk_metadata",
                        "kind": "media",
                        "generation_chunk_index": 0,
                        "num_frames": 9,
                    }
                )
            )
            await websocket.send(b"media")
        async for message in websocket:
            seen.append(json.loads(message)["type"])
            await websocket.send(json.dumps({"type": "session.pong"}))

    async def scenario() -> None:
        async with _serve(script) as url:
            with pytest.raises(BenchmarkError, match=f"after {delivered} chunk"):
                await asyncio.wait_for(
                    run_session(
                        url,
                        "model",
                        _workload(num_chunks=2),
                        first_chunk_timeout=0.25,
                        chunk_timeout=0.25,
                        ping_interval=0.03,
                        collect_media=False,
                        print_chunks=False,
                    ),
                    timeout=2,
                )
        assert "session.ping" in seen

    _run(scenario())


def test_terminal_chunk_is_only_excluded_from_steady_metrics() -> None:
    records = _records([1.0] + [0.1] * 5 + [1.2, 0.8, 0.1])
    metrics = compute_metrics(records, fps=12, warmup_chunks=6)
    assert metrics["steady_chunks"] == 2
    assert metrics["interval_steady"]["mean_ms"] == pytest.approx(1000)
    assert metrics["steady_video_rtf"] == pytest.approx(1)
    assert metrics["slo_attainment"] == pytest.approx(0.5)
    assert metrics["interval_all"]["count"] == 8
    assert metrics["total_frames"] == 105
    assert metrics["wall_seconds"] == pytest.approx(records[-1].arrival_s)
    assert metrics["playback"]["total_stall_ms"] == pytest.approx(
        simulate_playback(records, fps=12).total_stall_s * 1000
    )


def test_terminal_chunk_alone_does_not_establish_steady_state() -> None:
    metrics = compute_metrics(_records([1.0] * 7), fps=12, warmup_chunks=6)
    assert not metrics["steady_state_reached"]
    assert "steady_video_rtf" not in metrics


def test_aggregate_preserves_custom_deadline_and_weights_intervals() -> None:
    fast = compute_metrics(_records([1.0, 0.5, 0.1]), fps=12, warmup_chunks=0, slo_ms=700)
    slow = compute_metrics(_records([1.0, 0.9, 0.9, 0.9, 0.1]), fps=12, warmup_chunks=0, slo_ms=700)
    aggregate = aggregate_metrics([fast, slow], fps=12)
    assert aggregate["chunk_deadline_ms"] == 700
    assert aggregate["steady_interval_ms_mean"] == pytest.approx(800)
    slow["chunk_deadline_ms"] = 800
    with pytest.raises(ValueError, match="different chunk deadlines"):
        aggregate_metrics([fast, slow], fps=12)


@pytest.mark.parametrize(
    "flags, expected_shift, expected_negative",
    [
        ([], 7.0, "file negative"),
        (["--flow-shift", "5", "--negative-prompt", "CLI negative"], 5.0, "CLI negative"),
        (["--negative-prompt", ""], 7.0, ""),
    ],
)
def test_cli_workload_generation_overrides(tmp_path, flags, expected_shift, expected_negative) -> None:
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps(
            {
                "image": _IMAGE,
                "num_chunks": 3,
                "flow_shift": 7.0,
                "negative_prompt": "file negative",
            }
        )
    )
    workload = build_workload(parse_args(["--workload", str(path), *flags]))
    assert workload.flow_shift == expected_shift
    assert workload.negative_prompt == expected_negative


def test_builtin_workload_keeps_default_flow_shift() -> None:
    assert build_workload(parse_args(["--image", _IMAGE])).flow_shift == 5.0


def test_short_sample_report_qualifies_tail_percentiles(capsys) -> None:
    metrics = compute_metrics(_records([1.0] * 16), fps=12)
    print_report(metrics, workload=_workload(num_chunks=16), endpoint="ws://test", sessions=1, target_fps=12)
    assert "not reliable tail estimates" in capsys.readouterr().out
