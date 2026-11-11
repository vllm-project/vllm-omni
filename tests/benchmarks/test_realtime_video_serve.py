# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the real serving lifecycle with a local WS protocol peer."""

import asyncio
import json
import math
import statistics
from pathlib import Path

import pytest
from websockets.asyncio.server import serve

from vllm_omni.benchmarks.realtime_video.serving import build_serve_args
from vllm_omni.benchmarks.realtime_video.workload import pixel_frames_in_chunk
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


@pytest.mark.parametrize("failure", [None, "warmup", "measured", "save-video"])
def test_real_serve_entry_saves_video_sessions_and_failed_warmups(monkeypatch, tmp_path: Path, failure):
    """Warmup and measured raw streams traverse the actual scheduler/writer."""
    import vllm.benchmarks.serve as upstream

    def no_tokenizer(*args, **kwargs):
        pytest.fail("Realtime video must not initialize a text tokenizer")

    monkeypatch.setattr(upstream, "get_tokenizer", no_tokenizer)
    observed = []

    async def scenario():
        async def handler(websocket):
            payload = json.loads(await websocket.recv())
            count = len(payload["extra_params"]["camera_action_script"])
            observed.append(count)
            request_id = f"request-{len(observed)}"
            await websocket.send(json.dumps({"type": "video.start", "request_id": request_id, "format": "m4s"}))
            failing = (failure == "warmup" and len(observed) == 1) or (failure == "measured" and len(observed) == 3)
            for index in range(count):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "video.chunk_metadata",
                            "request_id": request_id,
                            "kind": "media",
                            "num_frames": pixel_frames_in_chunk(index),
                            "byte_length": 1,
                            "generation_chunk_index": index,
                        }
                    )
                )
                await websocket.send(b"x")
            if failing:
                # Even a full set of chunks is invalid without a successful terminal event.
                await websocket.send(json.dumps({"type": "error", "message": "deliberate WS failure"}))
            else:
                await websocket.send(json.dumps({"type": "session.done", "stopped": False, "chunks": count}))

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            args = build_serve_args(
                [
                    "--base-url",
                    f"ws://127.0.0.1:{port}/v1/realtime/video",
                    "--model",
                    "test-video",
                    "--num-prompts",
                    "2",
                    "--num-warmups",
                    "1",
                    "--realtime-video-num-chunks",
                    "10",
                    "--realtime-video-warmup-num-chunks",
                    "8",
                    "--realtime-video-image",
                    "data:image/png;base64,eA==",
                    "--realtime-video-ping-interval",
                    "0",
                    "--disable-tqdm",
                    "--save-result",
                    "--result-dir",
                    str(tmp_path),
                    "--result-filename",
                    "result.json",
                ]
            )
            if failure == "save-video":
                blocked_parent = tmp_path / "parent-file"
                blocked_parent.write_text("not a directory")
                args.realtime_video_save_video = str(blocked_parent / "video.mp4")
            if failure:
                with pytest.raises(RuntimeError, match="Realtime video benchmark failed"):
                    await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)
            else:
                await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    saved = json.loads((tmp_path / "result.json").read_text())
    video = saved["realtime_video"]
    assert saved["backend"] == "realtime-video"
    assert saved["dataset_name"] == "realtime-video"
    assert "total_output_tokens" not in saved and "mean_ttft_ms" not in saved and "itls" not in saved
    assert len(video["warmups"]) == 1
    warmup = video["warmups"][0]
    assert warmup["request_id"] == "request-1" and len(warmup["chunks"]) == 8
    assert warmup["measured"] is False
    assert video["passed"] is (failure is None)
    if failure == "warmup":
        assert observed == [8]
        assert video["sessions"] == [] and saved["completed"] == 0
        assert video["failed_warmups"] == 1 and warmup["success"] is False
    else:
        assert observed == [8, 10, 10]  # No hidden full-rollout ready request.
        assert len(video["sessions"]) == 2
        assert all(len(session["chunks"]) == 10 for session in video["sessions"])
        assert video["sessions"][0]["metrics"]["steady_chunks"] == 3
        assert video["sessions"][0]["metrics"]["total_frames"] == 117
        assert "ttfc_ms" in video["sessions"][0]
        assert saved["completed"] == (1 if failure == "measured" else 2)
        if failure == "measured":
            assert video["sessions"][-1]["success"] is False
    if failure == "save-video":
        assert all(session["success"] and session["metrics"] for session in video["sessions"])
        assert video["metrics"] and video["finalization_errors"]
        assert "save video:" in video["errors"][0]
    elif failure:
        assert video["errors"] and "deliberate WS failure" in video["errors"][0]
    if failure in (None, "save-video"):
        sessions = video["sessions"]
        assert saved["request_throughput"] == pytest.approx(2 / saved["duration"])
        assert math.isfinite(saved["mean_e2el_ms"]) and saved["mean_e2el_ms"] > 0
        assert saved["mean_video_ttfc_ms"] == pytest.approx(statistics.mean(row["ttfc_ms"] for row in sessions))
        assert saved["min_video_fps"] == min(row["metrics"]["frames_per_second"] for row in sessions)
        assert saved["mean_video_rtf"] == statistics.mean(row["metrics"]["video_rtf"] for row in sessions)
        intervals = [chunk["inter_arrival_ms"] for row in sessions for chunk in row["chunks"][6:-1]]
        assert saved["mean_video_steady_interval_ms"] == pytest.approx(statistics.mean(intervals))
        assert "whole measured rollouts" in saved["video_metric_scope"]
    else:
        assert "mean_e2el_ms" not in saved and "mean_video_steady_interval_ms" not in saved


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-concurrency", "2"],
        ["--request-rate", "1"],
        ["--probe-request-rate", "1"],
        ["--self-timed"],
        ["--ready-check-timeout-sec", "5"],
        ["--profile"],
        ["--ramp-up-strategy", "linear"],
        ["--plot-timeline"],
        ["--plot-dataset-stats"],
    ],
)
def test_realtime_video_rejects_modes_that_change_the_rollout_population(flags):
    with pytest.raises(ValueError, match="realtime-video"):
        build_serve_args(flags)


def test_realtime_video_defaults_are_safe_and_do_not_change_text_defaults():
    from vllm.benchmarks.serve import add_cli_args

    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    args = build_serve_args([])
    assert (args.num_prompts, args.num_warmups, args.max_concurrency) == (1, 0, 1)
    assert args.skip_tokenizer_init and args.ready_check_timeout_sec == 0
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    text_args = parser.parse_args([])
    baseline_parser = TrackingArgumentParser()
    add_cli_args(baseline_parser)
    baseline_args = baseline_parser.parse_args([])
    assert text_args.num_prompts == 1000
    assert text_args.ready_check_timeout_sec == baseline_args.ready_check_timeout_sec
    assert text_args.skip_tokenizer_init == baseline_args.skip_tokenizer_init


async def _emit_transport_rollout(websocket):
    payload = json.loads(await websocket.recv())
    count = len(payload["extra_params"]["camera_action_script"])
    await websocket.send(json.dumps({"type": "video.start", "request_id": "transport-test", "format": "m4s"}))
    for index in range(count):
        await websocket.send(
            json.dumps(
                {
                    "type": "video.chunk_metadata",
                    "kind": "media",
                    "generation_chunk_index": index,
                    "num_frames": pixel_frames_in_chunk(index),
                    "byte_length": 1,
                }
            )
        )
        await websocket.send(b"x")
    await websocket.send(json.dumps({"type": "session.done", "stopped": False, "chunks": count}))


@pytest.mark.parametrize("with_header", [False, True])
def test_shared_header_reaches_authenticated_ws_warmup_and_measurement(tmp_path, with_header):
    from http import HTTPStatus

    observed_headers = []

    async def scenario():
        async def authenticate(connection, request):
            value = request.headers.get("Authorization")
            observed_headers.append(value)
            if value != "Bearer local-test-only":
                return connection.respond(HTTPStatus.UNAUTHORIZED, "Header required")
            return None

        async with serve(_emit_transport_rollout, "127.0.0.1", 0, process_request=authenticate) as server:
            port = server.sockets[0].getsockname()[1]
            flags = ["--header", "Authorization=Bearer local-test-only"] if with_header else []
            args = build_serve_args(
                [
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--model",
                    "test-video",
                    "--num-prompts",
                    "1",
                    "--num-warmups",
                    "1",
                    "--realtime-video-num-chunks",
                    "10",
                    "--realtime-video-warmup-num-chunks",
                    "8",
                    "--realtime-video-ping-interval",
                    "0",
                    "--realtime-video-image",
                    "data:image/png;base64,eA==",
                    "--disable-tqdm",
                    "--save-result",
                    "--result-dir",
                    str(tmp_path),
                    "--result-filename",
                    "result.json",
                    *flags,
                ]
            )
            if with_header:
                await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)
            else:
                with pytest.raises(RuntimeError, match="Realtime video benchmark failed"):
                    await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    raw = (tmp_path / "result.json").read_text()
    document = json.loads(raw)["realtime_video"]
    assert document["passed"] is with_header
    expected_headers: list[str | None] = ["Bearer local-test-only"] * 2 if with_header else [None]
    assert observed_headers == expected_headers
    assert "local-test-only" not in raw and "Authorization" not in raw
    if with_header:
        assert [len(row["chunks"]) for row in document["warmups"]] == [8]
        assert [len(row["chunks"]) for row in document["sessions"]] == [10]


@pytest.mark.parametrize("insecure", [False, True])
def test_shared_insecure_changes_certificate_verification_without_disabling_wss(tmp_path, insecure):
    import ssl
    import subprocess

    cert, key = tmp_path / "certificate.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=localhost",
            "-days",
            "1",
            "-addext",
            "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    encrypted_connections = []

    async def scenario():
        async def handler(websocket):
            encrypted_connections.append(websocket.transport.get_extra_info("ssl_object") is not None)
            await _emit_transport_rollout(websocket)

        async with serve(handler, "127.0.0.1", 0, ssl=context) as server:
            port = server.sockets[0].getsockname()[1]
            args = build_serve_args(
                [
                    "--base-url",
                    f"wss://127.0.0.1:{port}/v1/realtime/video",
                    "--model",
                    "test-video",
                    "--num-prompts",
                    "1",
                    "--realtime-video-num-chunks",
                    "10",
                    "--realtime-video-image",
                    "data:image/png;base64,eA==",
                    "--realtime-video-ping-interval",
                    "0",
                    "--disable-tqdm",
                    "--save-result",
                    "--result-dir",
                    str(tmp_path),
                    "--result-filename",
                    "result.json",
                    *(["--insecure"] if insecure else []),
                ]
            )
            if insecure:
                await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)
            else:
                with pytest.raises(RuntimeError, match="Realtime video benchmark failed"):
                    await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    document = json.loads((tmp_path / "result.json").read_text())["realtime_video"]
    assert document["config"]["insecure"] is insecure
    assert document["passed"] is insecure
    assert encrypted_connections == ([True] if insecure else [])
    assert document["endpoint"].startswith("wss://")
    if not insecure:
        assert any("CERTIFICATE_VERIFY_FAILED" in error for error in document["errors"])


def test_video_binding_and_short_warmup_preserve_dynamic_request_and_sample_attributes():
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

    from vllm_omni.benchmarks.realtime_video.serving import (
        bind_realtime_video,
        get_realtime_video_samples,
        make_realtime_video_warmup,
    )

    args = build_serve_args(
        [
            "--realtime-video-image",
            "data:image/png;base64,eA==",
            "--realtime-video-num-chunks",
            "10",
            "--realtime-video-warmup-num-chunks",
            "8",
        ]
    )
    [sample] = get_realtime_video_samples(args)
    future_metadata = object()
    setattr(sample, "future_sample_metadata", future_metadata)
    request = RequestFuncInput(
        prompt="",
        api_url="http://localhost/v1/realtime/video",
        prompt_len=0,
        output_len=0,
        model="test",
        extra_headers={"X-Test": "local"},
    )
    setattr(request, "future_transport_metadata", future_metadata)
    bound = bind_realtime_video(sample, request)
    warmup = make_realtime_video_warmup(bound)
    assert getattr(bound, "future_transport_metadata") is future_metadata
    assert getattr(warmup, "future_transport_metadata") is future_metadata
    assert getattr(warmup, "video_sample").future_sample_metadata is future_metadata
    assert warmup.extra_headers == {"X-Test": "local"}
    assert getattr(bound, "video_sample").workload.num_chunks == 10
    assert getattr(warmup, "video_sample").workload.num_chunks == 8
    assert not hasattr(request, "video_sample") and getattr(bound, "measured") is True
    assert getattr(warmup, "measured") is False
