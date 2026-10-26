# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import GenerateSampleResult
from vllm_omni.benchmarks.duplex_session_metrics import (
    DUPLEX_METRICS_FILENAME,
    build_duplex_metrics_report,
    collect_duplex_session_metrics,
    merge_duplex_metrics_report,
    read_duplex_metrics_report,
)
from vllm_omni.clients.duplex import EventCollector
from vllm_omni.entrypoints.cli.benchmark import omni_duplex_eval as cli
from vllm_omni.entrypoints.cli.benchmark.omni_duplex_eval import OmniDuplexEvalSubcommand

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_cli_generate_evaluate_summarize_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "split": "PR_correction",
                    "question_text": "What changed?",
                    "answer1": "The object moved.",
                }
            ]
        ),
        encoding="utf-8",
    )
    response_root = tmp_path / "responses"
    score_root = tmp_path / "scores"

    async def fake_generate(sample, *, output_root, **kwargs):
        output = Path(output_root) / sample.split / f"{sample.id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([{"sentence": "It moved.", "start": 0, "end": 1}]), encoding="utf-8")
        output.with_name(output.stem + ".meta.json").write_text(json.dumps({"clock": "media"}), encoding="utf-8")
        return GenerateSampleResult(
            output=output,
            request_metrics=[
                {
                    "sample_id": sample.id,
                    "split": sample.split,
                    "ttft_ms": 10.0,
                    "ttfp_ms": 20.0,
                    "rtf": 0.5,
                }
            ],
            session_metrics={
                "sample_id": sample.id,
                "split": sample.split,
                "stream_ttft_ms": 10.0,
                "stream_ttfp_ms": 20.0,
                "stream_rtf": 0.5,
            },
        )

    class FakeJudge:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *args, **kwargs):
            return '{"success_score": 1, "is_relevant": 1}'

    monkeypatch.setattr(cli, "generate_sample", fake_generate)
    monkeypatch.setattr(cli, "DuplexJudge", FakeJudge)

    parser = argparse.ArgumentParser()
    OmniDuplexEvalSubcommand.add_cli_args(parser)

    common = ["--dataset", str(manifest), "--family", "pr"]
    generate = parser.parse_args(
        [
            "generate",
            *common,
            "--model",
            "mock",
            "--ref-audio",
            str(manifest),
            "--response-root",
            str(response_root),
            "--concurrency",
            "2",
        ]
    )
    OmniDuplexEvalSubcommand.cmd(generate)
    duplex_metrics = json.loads((response_root / DUPLEX_METRICS_FILENAME).read_text(encoding="utf-8"))
    assert duplex_metrics["duplex_request_metrics"] == [
        {
            "sample_id": "sample-1",
            "split": "PR_correction",
            "ttft_ms": 10.0,
            "ttfp_ms": 20.0,
            "rtf": 0.5,
        }
    ]
    assert duplex_metrics["duplex_stream_ttft_ms"]["mean"] == 10.0
    assert duplex_metrics["duplex_stream_ttfp_ms"]["mean"] == 20.0
    assert duplex_metrics["duplex_stream_rtf"]["mean"] == 0.5
    evaluate = parser.parse_args(
        [
            "evaluate",
            *common,
            "--response-root",
            str(response_root),
            "--score-root",
            str(score_root),
            "--judge-model",
            "mock-judge",
            "--eval-workers",
            "2",
        ]
    )
    OmniDuplexEvalSubcommand.cmd(evaluate)
    OmniDuplexEvalSubcommand.cmd(parser.parse_args(["summarize", "--score-root", str(score_root)]))

    summary = json.loads(capsys.readouterr().out)
    assert summary["samples"] == 1
    assert summary["pr"]["mean_all_success"] == 1.0


def _cached_sample(tmp_path: Path, meta: str | None, split: str = "PR_correction") -> DuplexSample:
    sample = DuplexSample.from_row({"id": "sample", "split": split})
    output = tmp_path / sample.split / f"{sample.id}.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({"sentences": [], "chunks": []}), encoding="utf-8")
    if meta is not None:
        output.with_name(f"{sample.id}.meta.json").write_text(meta, encoding="utf-8")
    return sample


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta,mix",
    [
        ('{"mix": "question"}', "question"),
        ('{"mix": "soundtrack"}', "soundtrack"),
        ("{}", "question"),
        (None, "question"),
    ],
)
async def test_generate_reuses_cached_sample_for_the_same_mix(tmp_path, monkeypatch, meta, mix):
    sample = _cached_sample(tmp_path, meta)

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("a cached sample with the same mix must be reused before preparation")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    result = await runner.generate_sample(
        sample,
        url="ws://unused",
        model="mock",
        ref_audio=None,
        output_root=tmp_path,
        mix=mix,
    )
    assert result.output == tmp_path / sample.split / f"{sample.id}.json"
    assert result.metadata == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta,mix",
    [
        ('{"mix": "question"}', "soundtrack"),
        ('{"mix": "soundtrack"}', "question"),
        ("{}", "soundtrack"),
        (None, "soundtrack"),
    ],
)
async def test_generate_refuses_cached_sample_recorded_with_another_mix(tmp_path, monkeypatch, meta, mix):
    sample = _cached_sample(tmp_path, meta)

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("a mix mismatch must be rejected before preparation")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    with pytest.raises(ValueError, match="generated with mix="):
        await runner.generate_sample(
            sample,
            url="ws://unused",
            model="mock",
            ref_audio=None,
            output_root=tmp_path,
            mix=mix,
        )


@pytest.mark.asyncio
async def test_generate_rejects_unknown_mix_before_reusing_cache(tmp_path, monkeypatch):
    sample = _cached_sample(tmp_path, '{"mix": "question"}')

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("an unknown mix must be rejected before the cached artifact is reused")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    with pytest.raises(ValueError, match="mix must be one of"):
        await runner.generate_sample(
            sample,
            url="ws://unused",
            model="mock",
            ref_audio=None,
            output_root=tmp_path,
            mix="karaoke",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["[]", '{"sentences": [{"sentence": "Old.", "start": 1, "end": 1}]}'])
async def test_generate_reuses_rtd_cached_response_without_chunks(tmp_path, monkeypatch, body):
    sample = _cached_sample(tmp_path, '{"mix": "question"}', split="RTD_OCR")
    output = tmp_path / sample.split / f"{sample.id}.json"
    output.write_text(body, encoding="utf-8")

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("RTD scores sentences, so a cached sentence list must be reused")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    result = await runner.generate_sample(
        sample,
        url="ws://unused",
        model="mock",
        ref_audio=None,
        output_root=tmp_path,
        mix="question",
    )
    assert result.output == output


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["[]", '{"sentences": [{"sentence": "Old.", "start": 1, "end": 1}]}'])
async def test_generate_refuses_cached_response_without_chunks(tmp_path, monkeypatch, body):
    sample = _cached_sample(tmp_path, '{"mix": "question"}')
    output = tmp_path / sample.split / f"{sample.id}.json"
    output.write_text(body, encoding="utf-8")

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("a response without chunk times must be rejected before preparation")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    with pytest.raises(ValueError, match="no chunk times"):
        await runner.generate_sample(
            sample,
            url="ws://unused",
            model="mock",
            ref_audio=None,
            output_root=tmp_path,
            mix="question",
        )


@pytest.mark.asyncio
async def test_generate_rejects_unreadable_cached_mix(tmp_path, monkeypatch):
    sample = _cached_sample(tmp_path, "{")

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("unreadable metadata must be rejected before preparation")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    with pytest.raises(ValueError, match="cannot read mix"):
        await runner.generate_sample(
            sample,
            url="ws://unused",
            model="mock",
            ref_audio=None,
            output_root=tmp_path,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_generate_rejects_unpaced_server_vad_before_preparation_or_cache(tmp_path, monkeypatch, cached):
    sample = DuplexSample.from_row({"id": "sample", "split": "PR_correction"})
    if cached:
        output = tmp_path / sample.split / f"{sample.id}.json"
        output.parent.mkdir()
        output.write_text("[]")

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("invalid pacing must be rejected before preparing media")

    monkeypatch.setattr(runner, "prepare_sample", unexpected_preparation)
    with pytest.raises(ValueError, match="server_vad requires pace='realtime'"):
        await runner.generate_sample(
            sample,
            url="ws://unused",
            model="mock",
            ref_audio=None,
            output_root=tmp_path,
            turn_detection="server_vad",
            pace="as-fast-as-possible",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("pace", ["realtime", "as-fast-as-possible"])
async def test_generate_exercises_realtime_socket_and_media_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pace: str
):
    import websockets

    received = []

    async def handler(websocket):
        response_started = False
        async for raw in websocket:
            event = json.loads(raw)
            received.append(event)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.append" and not response_started:
                response_started = True
                await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r1"}}))
                await websocket.send(
                    json.dumps({"type": "response.output_text.delta", "response_id": "r1", "delta": "Done."})
                )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.output_audio.delta",
                            "response_id": "r1",
                            "delta": base64.b64encode(b"\0\0" * 240).decode(),
                            "sample_rate_hz": 24_000,
                        }
                    )
                )
            elif event["type"] == "input_audio_buffer.commit":
                # The reply opened during the clip is not the residual decision.
                await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
                await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r1"}}))
                listen = {"type": "response.listen", "response": {"metadata": {"model_listen": True}}}
                await websocket.send(json.dumps(listen))
            elif event["type"] == "session.close":
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    monkeypatch.setattr(runner, "_COMPLETION_SETTLE_S", 0.05)
    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: b"\0\0" * (16_000 * 8 // 10))
    monkeypatch.setattr(runner, "video_duration", lambda path: 0.8)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter([(0.0, b"jpeg")]))
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    ref = tmp_path / "ref.wav"
    for path in (audio, video, ref):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "PR_correction", "pr", "correction", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=ref,
            output_root=tmp_path / "responses",
            pace=pace,
        )

    assert json.loads(result.output.read_text(encoding="utf-8")) == {
        "sentences": [{"sentence": "Done.", "start": 0.8, "end": 0.8}],
        "chunks": [{"text": "Done.", "current_time": 0.8}],
    }
    meta = json.loads(result.output.with_name("sample.meta.json").read_text(encoding="utf-8"))
    assert meta["response_done"] is True
    assert meta["drain_timeout"] is None
    assert meta["unit_ms"] == 1000
    assert meta["clock"] == ("media" if pace == "realtime" else "invalid")
    assert len([event for event in received if event["type"] == "input_audio_buffer.append"]) == 1
    assert "duplex_request_metrics" not in meta
    assert result.request_metrics
    assert result.request_metrics[0]["sample_id"] == "sample"
    assert result.session_metrics["stream_ttfp_ms"] is not None
    assert any(event["type"] == "playback.ack" for event in received)


@pytest.mark.asyncio
async def test_generate_finishes_a_silent_session_on_the_post_commit_listen(tmp_path: Path, monkeypatch):
    import websockets

    appended = []

    async def handler(websocket):
        async for raw in websocket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.append":
                appended.append(base64.b64decode(event["audio"]))
                # A decision for an earlier unit does not end the session.
                listen = {"type": "response.listen", "response": {"metadata": {"model_listen": True}}}
                await websocket.send(json.dumps(listen))
            elif event["type"] == "input_audio_buffer.commit":
                await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
                await websocket.send(json.dumps(listen))
            elif event["type"] == "session.close":
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    # Exactly one model unit: the runner keeps a residual for the commit to flush.
    monkeypatch.setattr(runner, "_COMPLETION_SETTLE_S", 0.1)
    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: b"\1\0" * 16_000)
    monkeypatch.setattr(runner, "video_duration", lambda path: 1.0)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter([(0.0, b"jpeg")]))
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    for path in (audio, video):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "PR_post_event_reminder", "pr", "post_event_reminder", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=None,
            output_root=tmp_path / "responses",
            pace="as-fast-as-possible",
        )

    assert result.error == ""
    assert sum(len(chunk) for chunk in appended) == 2 * 16_000 - 2
    assert json.loads(result.output.read_text(encoding="utf-8")) == {"sentences": [], "chunks": []}
    meta = json.loads(result.output.with_name("sample.meta.json").read_text(encoding="utf-8"))
    assert meta["response_done"] is True and meta["drain_timeout"] is None


@pytest.mark.asyncio
async def test_generate_waits_for_a_response_created_after_the_post_commit_listen(tmp_path: Path, monkeypatch):
    import websockets

    async def late_response(websocket):
        # Stage1 output of an earlier speak decision trails the listen.
        await asyncio.sleep(0.2)
        await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r1"}}))
        await websocket.send(json.dumps({"type": "response.output_text.delta", "response_id": "r1", "delta": "Late."}))
        await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r1"}}))

    async def handler(websocket):
        tasks = []
        async for raw in websocket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.commit":
                await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
                listen = {"type": "response.listen", "response": {"metadata": {"model_listen": True}}}
                await websocket.send(json.dumps(listen))
                tasks.append(asyncio.create_task(late_response(websocket)))
            elif event["type"] == "session.close":
                for task in tasks:
                    task.cancel()
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    monkeypatch.setattr(runner, "_COMPLETION_SETTLE_S", 0.5)
    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: b"\1\0" * 16_000)
    monkeypatch.setattr(runner, "video_duration", lambda path: 1.0)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter([(0.0, b"jpeg")]))
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    for path in (audio, video):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "PR_post_event_reminder", "pr", "post_event_reminder", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=None,
            output_root=tmp_path / "responses",
            pace="as-fast-as-possible",
        )

    assert result.error == ""
    assert [chunk["text"] for chunk in json.loads(result.output.read_text(encoding="utf-8"))["chunks"]] == ["Late."]


@pytest.mark.asyncio
async def test_generate_keeps_a_response_opened_after_an_earlier_response_done(tmp_path: Path, monkeypatch):
    import websockets

    async def late_response(websocket):
        # Stage1 output of a speak decision trails the post-commit listen.
        await asyncio.sleep(0.2)
        await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r2"}}))
        await websocket.send(json.dumps({"type": "response.output_text.delta", "response_id": "r2", "delta": "Late."}))
        await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r2"}}))

    async def handler(websocket):
        tasks = []
        early_sent = False
        async for raw in websocket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.append" and not early_sent:
                early_sent = True
                await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r1"}}))
                await websocket.send(
                    json.dumps({"type": "response.output_text.delta", "response_id": "r1", "delta": "Early."})
                )
                await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r1"}}))
            elif event["type"] == "input_audio_buffer.commit":
                await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
                listen = {"type": "response.listen", "response": {"metadata": {"model_listen": True}}}
                await websocket.send(json.dumps(listen))
                tasks.append(asyncio.create_task(late_response(websocket)))
            elif event["type"] == "session.close":
                for task in tasks:
                    task.cancel()
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    monkeypatch.setattr(runner, "_COMPLETION_SETTLE_S", 0.5)
    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: b"\1\0" * 16_000)
    monkeypatch.setattr(runner, "video_duration", lambda path: 1.0)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter([(0.0, b"jpeg")]))
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    for path in (audio, video):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "PR_post_event_reminder", "pr", "post_event_reminder", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=None,
            output_root=tmp_path / "responses",
            pace="as-fast-as-possible",
        )

    assert result.error == ""
    text = "".join(chunk["text"] for chunk in json.loads(result.output.read_text(encoding="utf-8"))["chunks"])
    assert text == "Early.Late."


def test_residual_unit_decision_follows_the_committed_event():
    created = {"type": "response.created", "response": {"id": "r1"}}
    done = {"type": "response.done", "response": {"id": "r1", "status": "completed"}}
    committed = {"type": "input_audio_buffer.committed"}
    listen = {"type": "response.listen", "response": {"metadata": {"model_listen": True}}}
    late_created = {"type": "response.created", "response": {"id": "r2"}}
    late_done = {"type": "response.done", "response": {"id": "r2", "status": "completed"}}

    assert runner._residual_unit_decided([created, done, committed], 0) is False
    assert runner._residual_unit_decided([created, committed, done], 0) is False
    assert runner._residual_unit_decided([created, committed, done], 0, settled_silence_is_terminal=True) is True
    assert runner._residual_unit_decided([created, committed, listen, done], 0) is False
    assert runner._residual_unit_decided([created, committed, done, listen], 0) is True
    assert runner._residual_unit_decided([created, committed, late_created, late_done, done], 0) is True
    assert runner._residual_unit_decided([committed, listen], 0) is True
    assert runner._residual_unit_decided([created, done], 0) is False
    cancelled = {"type": "response.done", "response": {"id": "r1", "status": "cancelled"}}
    failed = {"type": "response.done", "response": {"id": "r1", "status": "failed"}}
    late_cancelled = {"type": "response.done", "response": {"id": "r2", "status": "cancelled"}}
    late_failed = {"type": "response.done", "response": {"id": "r2", "status": "failed"}}
    # Barge-in can cancel the open reply and still open the residual afterwards.
    assert runner._residual_unit_decided([created, committed, cancelled], 0) is False
    assert runner._residual_unit_decided([created, committed, cancelled], 0, settled_silence_is_terminal=True) is True
    assert runner._residual_unit_decided([created, committed, cancelled, late_created, late_done], 0) is True
    assert runner._residual_unit_decided([committed, late_created, late_cancelled], 0) is False
    assert (
        runner._residual_unit_decided([committed, late_created, late_cancelled], 0, settled_silence_is_terminal=True)
        is True
    )
    # A failed reply is terminal: nothing is promoted after it.
    assert runner._residual_unit_decided([created, committed, failed], 0) is True
    assert runner._residual_unit_decided([committed, late_created, late_failed], 0) is True
    state = runner._completion_diagnostics(
        [created, committed, cancelled],
        0,
        response_ids=["r1"],
        stable_since=time.monotonic() - runner._COMPLETION_SETTLE_S - 1,
    )
    assert state["committed_observed"] is True
    assert state["open_response_ids_at_commit"] == ["r1"]
    assert state["unfinished_response_ids"] == []
    assert state["quiet_window_satisfied"] is True
    assert state["normal_terminal_decision"] is False
    assert state["settled_silence_terminal_decision"] is True
    assert state["completion_state_events"][-1] == {
        "type": "response.done",
        "response_id": "r1",
        "response_status": "cancelled",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "cancelled"])
async def test_generate_accepts_settled_terminal_silence_only_after_the_drain_timeout(
    tmp_path: Path, monkeypatch, status: str
):
    import websockets

    timeout_observed = False

    async def full_timeout(predicate, *, timeout_s: float, label: str):
        nonlocal timeout_observed
        assert timeout_s >= 20.0 and label == "response.done"
        await asyncio.sleep(0.02)
        assert predicate() is False
        await asyncio.sleep(0.1)
        assert predicate() is False
        timeout_observed = True
        raise TimeoutError(f"Timed out waiting for {label}")

    async def handler(websocket):
        opened = False
        async for raw in websocket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.append" and not opened:
                opened = True
                await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r1"}}))
                await websocket.send(
                    json.dumps({"type": "response.output_text.delta", "response_id": "r1", "delta": "Interrupted."})
                )
            elif event["type"] == "input_audio_buffer.commit":
                await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
                await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r1", "status": status}}))
            elif event["type"] == "session.close":
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    monkeypatch.setattr(runner, "wait_for", full_timeout)
    monkeypatch.setattr(runner, "_COMPLETION_SETTLE_S", 0.05)
    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: b"\1\0" * 16_000)
    monkeypatch.setattr(runner, "video_duration", lambda path: 1.0)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter([(0.0, b"jpeg")]))
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    for path in (audio, video):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "PR_post_event_reminder", "pr", "post_event_reminder", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=None,
            output_root=tmp_path / "responses",
            pace="as-fast-as-possible",
        )

    assert timeout_observed
    assert result.error == ""
    assert result.metadata["response_done"] is True
    assert result.metadata["drain_timeout"] is None
    assert result.metadata["unfinished_response_ids"] == []
    assert [row["sentence"] for row in result.timed_sentences] == ["Interrupted."]


@pytest.mark.asyncio
async def test_generate_waits_for_a_response_opened_after_an_in_flight_reply(tmp_path: Path, monkeypatch):
    import websockets

    async def late_response(websocket):
        # The residual unit decides only after the in-flight reply finishes.
        await asyncio.sleep(0.5)
        await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r2"}}))
        await websocket.send(json.dumps({"type": "response.output_text.delta", "response_id": "r2", "delta": "Late."}))
        await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r2", "status": "completed"}}))

    async def handler(websocket):
        tasks = []
        opened = False
        async for raw in websocket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.append" and not opened:
                opened = True
                await websocket.send(json.dumps({"type": "response.created", "response": {"id": "r1"}}))
                await websocket.send(
                    json.dumps({"type": "response.output_text.delta", "response_id": "r1", "delta": "Early."})
                )
            elif event["type"] == "input_audio_buffer.commit":
                await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
                await websocket.send(
                    json.dumps({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
                )
                tasks.append(asyncio.create_task(late_response(websocket)))
            elif event["type"] == "session.close":
                for task in tasks:
                    task.cancel()
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    monkeypatch.setattr(runner, "_COMPLETION_SETTLE_S", 0.2)
    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: b"\1\0" * 16_000)
    monkeypatch.setattr(runner, "video_duration", lambda path: 1.0)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter([(0.0, b"jpeg")]))
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    for path in (audio, video):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "PR_post_event_reminder", "pr", "post_event_reminder", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=None,
            output_root=tmp_path / "responses",
            pace="as-fast-as-possible",
        )

    assert result.error == ""
    text = "".join(chunk["text"] for chunk in json.loads(result.output.read_text(encoding="utf-8"))["chunks"])
    assert text == "Early.Late."


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_frames,silence_frames", [(35, 19), (36, 16)])
async def test_generate_server_vad_sends_image_items_and_waits_for_server_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, speech_frames: int, silence_frames: int
):
    import numpy as np
    import websockets

    from vllm_omni.engine.duplex.turn_detection import ServerTurnDetector, TurnDetectionConfig

    received = []
    commits = []
    caps = {"required_input_modalities": ["audio"], "optional_input_modalities": [], "supports_image_input": True}
    pcm = (
        np.concatenate([np.ones(speech_frames * 512), np.zeros(silence_frames * 512), np.ones(21 * 512)])
        .astype("<i2")
        .tobytes()
    )
    frames = [(float(index), f"jpeg-{index}".encode()) for index in range(3)]

    async def handler(websocket):
        detector = ServerTurnDetector(TurnDetectionConfig())
        # Exercise the real framing/endpoint state machine without ONNX weights.
        monkeypatch.setattr(detector.vad, "_frame_scorer", lambda frame: 0.99 if np.any(frame) else 0.01)
        async for raw in websocket:
            event = json.loads(raw)
            received.append(event)
            if event["type"] == "session.update":
                await websocket.send(json.dumps({"type": "session.created", "session": {"capabilities": caps}}))
            elif event["type"] == "input_audio_buffer.append":
                detected = detector.process(event["audio"], fmt="pcm16", sample_rate_hz=event["sample_rate_hz"])
                if detected.speech_started:
                    await websocket.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
                if detected.speech_stopped:
                    await websocket.send(json.dumps({"type": "input_audio_buffer.speech_stopped"}))
                if detected.should_commit:
                    commits.append(event["audio_end_ms"])
                    response_id = f"r{len(commits)}"
                    for turn_event in (
                        {"type": "response.created", "response": {"id": response_id}},
                        {"type": "response.output_text.delta", "response_id": response_id, "delta": "Now."},
                        {
                            "type": "response.output_audio.delta",
                            "response_id": response_id,
                            "delta": base64.b64encode(b"\0\0" * 240).decode(),
                            "sample_rate_hz": 24_000,
                        },
                        {"type": "response.done", "response": {"id": response_id}},
                    ):
                        await websocket.send(json.dumps(turn_event))
            elif event["type"] == "session.close":
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

    monkeypatch.setattr(runner, "read_audio_pcm16", lambda path: pcm)
    monkeypatch.setattr(runner, "video_duration", lambda path: len(pcm) / 32_000)
    monkeypatch.setattr(runner, "iter_jpegs", lambda *args, **kwargs: iter(frames))
    monkeypatch.setattr(runner, "_SERVER_TURN_SETTLE_S", 0.05)
    audio = tmp_path / "question.wav"
    video = tmp_path / "video.mp4"
    for path in (audio, video):
        path.write_bytes(b"data")
    sample = DuplexSample("sample", "RTD_counting", "rtd", "counting", video, audio)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=None,
            output_root=tmp_path / "responses",
            instructions=None,
            turn_detection="server_vad",
        )

    # A large append can contain both an endpoint and a new onset. The first
    # endpoint must reach ServerTurnDetector before the next speech starts.
    assert len(commits) == 2, commits
    assert commits == [(speech_frames + 16) * 32, (speech_frames + silence_frames + 21 + 16) * 32]
    assert len(result.request_metrics) == 2
    session = next(event["session"] for event in received if event["type"] == "session.update")
    assert session["turn_detection"] == {"type": "server_vad"}
    assert session["extra_body"]["auto_response"] is False
    assert "ref_audio" not in session and "instructions" not in session
    assert not any(event["type"] == "input_audio_buffer.commit" for event in received)
    appends = [event for event in received if event["type"] == "input_audio_buffer.append"]
    chunks = [base64.b64decode(event["audio"]) for event in appends]
    assert max(map(len, chunks)) <= 512 * 2
    assert b"".join(chunks) == pcm + bytes(24_000 * 2)
    assert not any("video_frames" in event for event in appends)
    images = [event for event in received if event["type"] == "conversation.item.create"]
    assert [event["item"]["content"][0]["image_url"] for event in images] == [
        "data:image/jpeg;base64," + base64.b64encode(frame).decode() for _, frame in frames
    ]
    assert [sentence["sentence"] for sentence in result.timed_sentences] == ["Now.", "Now."]
    meta = json.loads(result.output.with_name("sample.meta.json").read_text(encoding="utf-8"))
    assert meta["response_done"] is True and meta["drain_timeout"] is None
    assert meta["turn_detection"] == "server_vad" and meta["frame_transport"] == "image_items"
    assert meta["unit_ms"] == 32
    assert meta["ref_audio_sha256"] is None


def _pr_manifest(tmp_path: Path, sample_ids: tuple[str, ...]) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": sample_id,
                    "split": "PR_correction",
                    "question_text": "What changed?",
                    "answer1": "The object moved.",
                }
                for sample_id in sample_ids
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def _run_generate_cli(manifest: Path, response_root: Path) -> None:
    parser = argparse.ArgumentParser()
    OmniDuplexEvalSubcommand.add_cli_args(parser)
    OmniDuplexEvalSubcommand.cmd(
        parser.parse_args(
            [
                "generate",
                "--dataset",
                str(manifest),
                "--family",
                "pr",
                "--model",
                "mock",
                "--ref-audio",
                str(manifest),
                "--response-root",
                str(response_root),
            ]
        )
    )


def _metrics_for(sample_id: str, *, ttft_ms: float, ttfp_ms: float, rtf: float) -> GenerateSampleResult:
    split = "PR_correction"
    return GenerateSampleResult(
        output=Path(split) / f"{sample_id}.json",
        request_metrics=[{"sample_id": sample_id, "split": split, "ttft_ms": ttft_ms, "ttfp_ms": ttfp_ms, "rtf": rtf}],
        session_metrics={
            "sample_id": sample_id,
            "split": split,
            "stream_ttft_ms": ttft_ms,
            "stream_ttfp_ms": ttfp_ms,
            "stream_rtf": rtf,
        },
    )


def _fake_generate(results: dict[str, GenerateSampleResult | None]):
    async def generate(
        sample: DuplexSample,
        *,
        output_root: str | Path,
        url: str,
        model: str,
        ref_audio: str | Path,
        fps: float = 1.0,
        mix: str = "question",
        pace: str = "realtime",
        clock: str = "media",
        overwrite: bool = False,
        unit_ms: int = 1000,
        instructions: str | None = None,
        turn_detection: str = "none",
    ) -> GenerateSampleResult:
        _ = (url, model, ref_audio, fps, mix, pace, clock, overwrite, unit_ms, instructions, turn_detection)
        result = results[sample.id]
        if result is None:
            return GenerateSampleResult(output=Path(output_root) / sample.split / f"{sample.id}.json")
        return result

    return generate


def test_merge_duplex_metrics_report_replaces_only_incoming_sample_keys():
    existing = build_duplex_metrics_report(
        request_metrics=[
            {"sample_id": "kept", "split": "PR_correction", "ttft_ms": 11.0},
            {"sample_id": "replaced", "split": "PR_correction", "ttft_ms": 12.0},
        ],
        session_metrics=[
            {"sample_id": "kept", "split": "PR_correction", "stream_ttft_ms": 11.0},
            {"sample_id": "replaced", "split": "PR_correction", "stream_ttft_ms": 12.0},
        ],
    )
    merged = merge_duplex_metrics_report(
        existing,
        request_metrics=[{"sample_id": "replaced", "split": "PR_correction", "ttft_ms": 99.0}],
        session_metrics=[{"sample_id": "replaced", "split": "PR_correction", "stream_ttft_ms": 99.0}],
    )
    assert [row["sample_id"] for row in merged["duplex_request_metrics"]] == ["kept", "replaced"]
    assert merged["duplex_request_metrics"][1]["ttft_ms"] == 99.0
    assert merged["duplex_stream_ttft_ms"]["mean"] == 55.0


def test_read_duplex_metrics_report_returns_none_when_missing(tmp_path: Path):
    assert read_duplex_metrics_report(tmp_path / DUPLEX_METRICS_FILENAME) is None


def test_read_duplex_metrics_report_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / DUPLEX_METRICS_FILENAME
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_duplex_metrics_report(path)


def test_merge_duplex_metrics_report_without_existing_is_passthrough():
    merged = merge_duplex_metrics_report(
        None,
        request_metrics=[{"sample_id": "a", "split": "PR_correction", "ttft_ms": 1.0}],
        session_metrics=[{"sample_id": "a", "split": "PR_correction", "stream_ttft_ms": 1.0}],
    )
    assert merged["duplex_request_metrics"][0]["ttft_ms"] == 1.0
    assert merged["duplex_stream_ttft_ms"]["mean"] == 1.0


def test_merge_duplex_metrics_report_rejects_non_object_rows():
    with pytest.raises(ValueError, match="duplex_request_metrics entries must be JSON objects"):
        merge_duplex_metrics_report(
            {"duplex_request_metrics": ["bad"]},
            request_metrics=[],
            session_metrics=[],
        )


def test_cli_generate_all_skipped_keeps_existing_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = _pr_manifest(tmp_path, ("sample-1",))
    response_root = tmp_path / "responses"
    monkeypatch.setattr(
        cli,
        "generate_sample",
        _fake_generate({"sample-1": _metrics_for("sample-1", ttft_ms=10.0, ttfp_ms=20.0, rtf=0.5)}),
    )
    _run_generate_cli(manifest, response_root)
    monkeypatch.setattr(cli, "generate_sample", _fake_generate({"sample-1": None}))
    _run_generate_cli(manifest, response_root)
    duplex_metrics = json.loads((response_root / DUPLEX_METRICS_FILENAME).read_text(encoding="utf-8"))
    assert duplex_metrics["duplex_request_metrics"][0]["ttft_ms"] == 10.0
    assert duplex_metrics["duplex_stream_ttft_ms"]["mean"] == 10.0
    assert duplex_metrics["duplex_stream_ttfp_ms"]["mean"] == 20.0
    assert duplex_metrics["duplex_stream_rtf"]["mean"] == 0.5


def test_cli_generate_partial_resume_merges_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = _pr_manifest(tmp_path, ("sample-1", "sample-2"))
    response_root = tmp_path / "responses"
    monkeypatch.setattr(
        cli,
        "generate_sample",
        _fake_generate(
            {
                "sample-1": _metrics_for("sample-1", ttft_ms=10.0, ttfp_ms=20.0, rtf=0.5),
                "sample-2": _metrics_for("sample-2", ttft_ms=30.0, ttfp_ms=40.0, rtf=0.25),
            }
        ),
    )
    _run_generate_cli(manifest, response_root)
    monkeypatch.setattr(
        cli,
        "generate_sample",
        _fake_generate(
            {
                "sample-1": None,
                "sample-2": _metrics_for("sample-2", ttft_ms=90.0, ttfp_ms=80.0, rtf=1.5),
            }
        ),
    )
    _run_generate_cli(manifest, response_root)
    duplex_metrics = json.loads((response_root / DUPLEX_METRICS_FILENAME).read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in duplex_metrics["duplex_request_metrics"]}
    assert by_id["sample-1"]["ttft_ms"] == 10.0
    assert by_id["sample-2"]["ttft_ms"] == 90.0
    assert duplex_metrics["duplex_stream_ttft_ms"]["mean"] == 50.0
    assert duplex_metrics["duplex_stream_ttfp_ms"]["mean"] == 50.0
    assert duplex_metrics["duplex_stream_rtf"]["mean"] == 1.0


def test_collect_duplex_session_metrics_matches_omniinteract_window():
    collector = EventCollector()
    audio = {
        "type": "response.output_audio.delta",
        "response_id": "r1",
        "format": "pcm16",
        "delta": base64.b64encode(bytes((1, 0)) * 2400).decode(),
        "sample_rate_hz": 24_000,
        "metadata": {"audio_duration_ms": 100},
    }
    collector.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=10.1)
    collector.add(
        {"type": "response.output_text.delta", "response_id": "r1", "delta": "Done."},
        received_at_s=10.2,
    )
    collector.add(audio, received_at_s=10.3)
    collector.add({"type": "response.done", "response": {"id": "r1"}}, received_at_s=10.4)
    collector.note_client_append({"sent_at_s": 10.05, "event_id": "append-1"})
    bundle = collect_duplex_session_metrics(collector, stream_start=10.0, session_id="sample")
    assert bundle.request_metrics[0]["ttft_ms"] == 100.0
    assert bundle.request_metrics[0]["ttfp_ms"] == 200.0
    assert bundle.request_metrics[0]["rtf"] == 2.0
    assert bundle.request_metrics[0]["client_ttfp_ms"] == 250.0
    assert bundle.request_metrics[0]["client_audio_generation_ms"] == 250.0
    assert bundle.request_metrics[0]["client_rtf"] == 2.5
    assert bundle.session_metrics["stream_ttft_ms"] == 200.0
    assert bundle.session_metrics["stream_ttfp_ms"] == 300.0
    report = build_duplex_metrics_report(
        request_metrics=bundle.request_metrics,
        session_metrics=[bundle.session_metrics],
    )
    assert report["duplex_stream_ttft_ms"]["mean"] == 200.0
    assert report["duplex_stream_ttfp_ms"]["mean"] == 300.0


def test_event_collector_from_realtime_copies_client_appends():
    from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

    source = RealtimeEventCollector()
    stamp = {"sent_at_s": 9.8, "event_id": "evt-1"}
    source.note_client_append(stamp)
    source.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=10.0)
    copied = runner._event_collector_from_realtime(source)
    assert copied.client_appends == [stamp]
    assert copied.response_ids == ["r1"]


def test_collect_duplex_session_metrics_keeps_all_engine_stages():
    collector = EventCollector()
    audio = {
        "type": "response.output_audio.delta",
        "response_id": "r1",
        "format": "pcm16",
        "delta": base64.b64encode(bytes((1, 0)) * 2400).decode(),
        "sample_rate_hz": 24_000,
        "metadata": {
            "audio_duration_ms": 100,
            "vllm_omni": {
                "stage_metrics": {
                    "0": {
                        "final_output_type": "text",
                        "num_tokens_out": 4,
                        "vllm_ttft_ms": 12.0,
                        "vllm_tpot_ms": 3.0,
                        "vllm_itls_ms": [3.0],
                    },
                    "1": {
                        "output_unit_type": "stream",
                        "output_unit_count": 8,
                        "serving_time_to_first_output_ms": 40.0,
                        "time_per_output_unit_ms": 5.0,
                        "inter_output_latencies_ms": [5.0],
                    },
                    "2": {
                        "final_output_type": "audio",
                        "serving_time_to_first_output_ms": 80.0,
                    },
                }
            },
        },
    }
    collector.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=10.1)
    collector.add(
        {"type": "response.output_text.delta", "response_id": "r1", "delta": "Done."},
        received_at_s=10.2,
    )
    collector.add(audio, received_at_s=10.3)
    collector.add({"type": "response.done", "response": {"id": "r1"}}, received_at_s=10.4)
    bundle = collect_duplex_session_metrics(collector, stream_start=10.0, session_id="sample")
    stages = bundle.request_metrics[0]["stages"]
    assert list(stages) == ["0", "1", "2"]
    assert stages["0"] == bundle.request_metrics[0]["stage0_tokens"]
    assert stages["1"]["ttfc_ms"] == 40.0
    assert stages["1"]["tpop_ms"] == 5.0
    assert stages["2"]["ttfp_ms"] == 80.0
    assert bundle.session_metrics["stages"]["0"]["tpot_ms"]["mean"] == 3.0
    assert bundle.session_metrics["stages"]["2"]["ttfp_ms"]["mean"] == 80.0
    report = build_duplex_metrics_report(
        request_metrics=bundle.request_metrics,
        session_metrics=[bundle.session_metrics],
    )
    assert "duplex_stream_stages" not in report


def test_judge_exercises_openai_http_schema():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append((self.path, self.headers["Authorization"], json.loads(self.rfile.read(length))))
            payload = json.dumps({"choices": [{"message": {"content": '{"success_score": 1}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        judge = DuplexJudge(f"http://127.0.0.1:{server.server_port}", "judge", api_key="token")
        assert judge.chat("prompt") == '{"success_score": 1}'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    path, authorization, payload = requests[0]
    assert path == "/v1/chat/completions"
    assert authorization == "Bearer token"
    assert payload["model"] == "judge"
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
