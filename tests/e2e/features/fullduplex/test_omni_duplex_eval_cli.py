# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import base64
import json
import threading
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
                "global_ttft_ms": 10.0,
                "global_ttfp_ms": 20.0,
                "global_rtf": 0.5,
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
    assert duplex_metrics["mean_duplex_global_ttft_ms"] == 10.0
    assert duplex_metrics["mean_duplex_global_ttfp_ms"] == 20.0
    assert duplex_metrics["mean_duplex_global_rtf"] == 0.5
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


@pytest.mark.asyncio
async def test_generate_exercises_realtime_socket_and_media_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
                await websocket.send(json.dumps({"type": "response.done", "response": {"id": "r1"}}))
            elif event["type"] == "session.close":
                await websocket.send(json.dumps({"type": "session.closed"}))
                return

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
        )

    assert json.loads(result.output.read_text(encoding="utf-8")) == [{"sentence": "Done.", "start": 0.8, "end": 0.8}]
    meta = json.loads(result.output.with_name("sample.meta.json").read_text(encoding="utf-8"))
    assert meta["response_done"] is True
    assert meta["drain_timeout"] is None
    assert "duplex_request_metrics" not in meta
    assert result.request_metrics
    assert result.request_metrics[0]["sample_id"] == "sample"
    assert result.session_metrics["global_ttfp_ms"] is not None
    assert any(event["type"] == "playback.ack" for event in received)


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
            "global_ttft_ms": ttft_ms,
            "global_ttfp_ms": ttfp_ms,
            "global_rtf": rtf,
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
    ) -> GenerateSampleResult:
        _ = (url, model, ref_audio, fps, mix, pace, clock, overwrite, unit_ms)
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
            {"sample_id": "kept", "split": "PR_correction", "global_ttft_ms": 11.0},
            {"sample_id": "replaced", "split": "PR_correction", "global_ttft_ms": 12.0},
        ],
    )
    merged = merge_duplex_metrics_report(
        existing,
        request_metrics=[{"sample_id": "replaced", "split": "PR_correction", "ttft_ms": 99.0}],
        session_metrics=[{"sample_id": "replaced", "split": "PR_correction", "global_ttft_ms": 99.0}],
    )
    assert [row["sample_id"] for row in merged["duplex_request_metrics"]] == ["kept", "replaced"]
    assert merged["duplex_request_metrics"][1]["ttft_ms"] == 99.0
    assert merged["mean_duplex_global_ttft_ms"] == 55.0


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
        session_metrics=[{"sample_id": "a", "split": "PR_correction", "global_ttft_ms": 1.0}],
    )
    assert merged["duplex_request_metrics"][0]["ttft_ms"] == 1.0
    assert merged["mean_duplex_global_ttft_ms"] == 1.0


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
    assert duplex_metrics["mean_duplex_global_ttft_ms"] == 10.0
    assert duplex_metrics["mean_duplex_global_ttfp_ms"] == 20.0
    assert duplex_metrics["mean_duplex_global_rtf"] == 0.5


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
    assert duplex_metrics["mean_duplex_global_ttft_ms"] == 50.0
    assert duplex_metrics["mean_duplex_global_ttfp_ms"] == 50.0
    assert duplex_metrics["mean_duplex_global_rtf"] == 1.0


def test_collect_duplex_session_metrics_matches_omniinteract_window():
    collector = EventCollector()
    audio = {
        "type": "response.audio.delta",
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
    bundle = collect_duplex_session_metrics(collector, stream_start=10.0, session_id="sample")
    assert bundle.request_metrics[0]["ttft_ms"] == 100.0
    assert bundle.request_metrics[0]["ttfp_ms"] == 200.0
    assert bundle.request_metrics[0]["rtf"] == 2.0
    assert bundle.session_metrics["global_ttft_ms"] == 200.0
    assert bundle.session_metrics["global_ttfp_ms"] == 300.0
    report = build_duplex_metrics_report(
        request_metrics=bundle.request_metrics,
        session_metrics=[bundle.session_metrics],
    )
    assert report["mean_duplex_global_ttft_ms"] == 200.0
    assert report["mean_duplex_global_ttfp_ms"] == 300.0


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
