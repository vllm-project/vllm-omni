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
        return output

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
                            "type": "response.audio.delta",
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
        output = await runner.generate_sample(
            sample,
            url=f"ws://127.0.0.1:{port}/v1/realtime?duplex=1",
            model="mock",
            ref_audio=ref,
            output_root=tmp_path / "responses",
        )

    assert json.loads(output.read_text(encoding="utf-8")) == [{"sentence": "Done.", "start": 0.8, "end": 0.8}]
    meta = json.loads(output.with_name("sample.meta.json").read_text(encoding="utf-8"))
    assert meta["response_done"] is True
    assert meta["drain_timeout"] is None
    assert any(event["type"] == "playback.ack" for event in received)


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
