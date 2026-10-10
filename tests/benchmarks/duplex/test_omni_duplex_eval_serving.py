# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU coverage of the public serving benchmark and shared DuplexEval runner."""

import asyncio
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pybase64 as base64
import pytest
import websockets
from aiohttp import web
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex import omni_duplex_eval_serving as adapter
from vllm_omni.benchmarks.patch import patch
from vllm_omni.entrypoints.cli.benchmark.cli_args import preprocess_serve_args
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _args(tmp_path, *extra):
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"reference")
    manifest = tmp_path / "samples.json"
    manifest.write_text(
        json.dumps(
            [
                {"id": str(i), "split": "PR_correction", "video": "video.mp4", "question_audio": "question.wav"}
                for i in range(2)
            ]
        )
    )
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    return parser.parse_args(
        [
            "--model",
            "mock",
            "--backend",
            "openai-realtime-duplex",
            "--dataset-name",
            "omni-duplex-eval",
            "--dataset-path",
            str(manifest),
            "--endpoint",
            "/v1/realtime",
            "--duplex-eval-ref-audio",
            str(ref),
            "--duplex-eval-response-root",
            str(tmp_path / "responses"),
            "--disable-shuffle",
            *extra,
        ]
    )


@pytest.fixture
def prepared(monkeypatch):
    value = runner.PreparedSample(b"\0\0" * 160, ((0.0, b"jpeg"),), 0.01, "data:audio/wav;base64,cmVm", "hash")
    monkeypatch.setattr(adapter, "prepare_sample", lambda *args, **kwargs: value)
    return value


def _samples(tmp_path):
    args = _args(tmp_path)
    preprocess_serve_args(args)
    return patch.get_samples(args, None)


def _request(sample, *, measured=True):
    request = RequestFuncInput(
        model="mock",
        model_name="served-model",
        prompt="",
        api_url="http://localhost:8000/v1/realtime",
        prompt_len=0,
        output_len=0,
        request_id=sample.request_id if measured else None,
    )
    adapter.attach_duplex_eval(sample, request)
    return request


def _result(sample):
    return runner.GenerateSampleResult(
        output=sample.response_root / sample.sample.split / f"{sample.sample.id}.json",
        request_metrics=[{"stage0_tokens": {"output_token_count": 3, "itls_ms": [10.0, 30.0]}}],
        session_metrics={"ttft_ms": {"mean": 100.0}, "ttfp_ms": {"mean": 200.0}, "rtf": {"mean": 0.5}},
        timed_sentences=[{"sentence": "Done.", "start": 0.01, "end": 0.01}],
        metadata={"response_done": True, "clock": "media"},
        output_tokens=3,
        audio_bytes=48000,
    )


@pytest.mark.parametrize("count,expected", [(None, 2), ("0", 2), ("1", 1), ("100", 2)])
def test_selection_uses_standard_prompt_count(tmp_path, prepared, count, expected):
    args = _args(tmp_path, *([] if count is None else ["--num-prompts", count]))
    preprocess_serve_args(args)
    requests = patch.get_samples(args, None)
    assert len(requests) == args.num_prompts == expected
    assert args.max_concurrency == 1
    assert all(request.prepared is prepared for request in requests)
    assert len({request.request_id for request in requests}) == expected


def test_repeated_sweep_runs_keep_artifacts_isolated(tmp_path, prepared):
    args = _args(tmp_path)
    preprocess_serve_args(args)
    parent = args.duplex_eval_response_root
    roots = []
    for concurrency in (1, 2):
        args.max_concurrency = concurrency
        samples = patch.get_samples(args, None)
        outputs = []
        for sample in samples:
            output = patch.MixRequestFuncOutput(success=True)
            output.duplex_eval_result = replace(_result(sample), timed_sentences=[{"sentence": str(concurrency)}])
            outputs.append(output)
        summary = adapter.finalize_duplex_eval(samples, outputs)
        root = Path(summary["response_root"])
        assert summary["published"] == len(samples) and not summary["artifact_errors"]
        assert all(sample.response_root == root for sample in samples)
        assert args.duplex_eval_response_root == parent
        roots.append(root)
    assert roots[0] != roots[1]
    assert all(root.parent == parent for root in roots)
    for concurrency, root in enumerate(roots, start=1):
        assert json.loads((root / "PR_correction/0.json").read_text()) == [{"sentence": str(concurrency)}]


@pytest.mark.asyncio
async def test_standalone_preparation_keeps_event_loop_responsive(tmp_path, prepared, monkeypatch):
    sample = _samples(tmp_path)[0]
    loop = asyncio.get_running_loop()
    preparing = asyncio.Event()
    loop_progress = threading.Event()

    def slow_prepare(*args, **kwargs):
        loop.call_soon_threadsafe(preparing.set)
        # Only the event-loop coroutine below can release preparation. On the
        # old synchronous path it cannot run until this bounded wait fails.
        assert loop_progress.wait(5), "media preparation blocked the event loop"
        return prepared

    monkeypatch.setattr(runner, "prepare_sample", slow_prepare)

    async def handler(socket):
        async for raw in socket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await socket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.commit":
                await socket.send(json.dumps({"type": "response.created", "response": {"id": "r1"}}))
                await socket.send(json.dumps({"type": "response.done", "response": {"id": "r1"}}))
            elif event["type"] == "session.close":
                await socket.send(json.dumps({"type": "session.closed"}))
                return

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        task = asyncio.create_task(
            runner.generate_sample(
                sample.sample,
                url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime",
                model="mock",
                ref_audio=sample.ref_audio,
                output_root=sample.response_root,
            )
        )
        try:
            await asyncio.wait_for(preparing.wait(), timeout=10)
        finally:
            loop_progress.set()
        result = await asyncio.wait_for(task, timeout=10)
    assert not result.error and result.metadata["response_done"]
    assert result.output.exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--backend", "openai"),
        ("--endpoint", "/v1/chat/completions"),
        ("--num-prompts", "-1"),
        ("--max-concurrency", "0"),
        ("--probe-request-rate", "1"),
    ],
)
def test_invalid_cli_combinations(tmp_path, option, value):
    with pytest.raises(ValueError):
        preprocess_serve_args(_args(tmp_path, option, value))


def test_evaluation_requires_a_judge_model(tmp_path):
    with pytest.raises(ValueError, match="--duplex-eval-judge-model"):
        preprocess_serve_args(_args(tmp_path, "--duplex-eval-evaluate"))


@pytest.mark.parametrize(
    "option,value",
    [
        ("--duplex-eval-judge-timeout-s", "nan"),
        ("--duplex-eval-judge-fps", "0"),
        ("--duplex-eval-window-size", "-1"),
        ("--duplex-eval-eval-workers", "0"),
    ],
)
def test_evaluation_rejects_invalid_options(tmp_path, option, value):
    with pytest.raises(SystemExit):
        _args(tmp_path, option, value)


def test_evaluation_reuses_selected_samples_and_existing_scores(tmp_path, prepared, monkeypatch):
    args = _args(
        tmp_path,
        "--duplex-eval-evaluate",
        "--duplex-eval-judge-model",
        "judge",
        "--duplex-eval-ids",
        "1",
        "--duplex-eval-judge-video-mode",
        "frame-sample",
        "--duplex-eval-judge-fps",
        "3",
        "--duplex-eval-window-size",
        "12",
        "--duplex-eval-judge-timeout-s",
        "0.5",
        "--duplex-eval-eval-workers",
        "2",
    )
    preprocess_serve_args(args)
    samples = patch.get_samples(args, None)
    assert [request.sample.id for request in samples] == ["1"]
    observed = []
    evaluate = adapter.evaluate_sample

    def record(sample, response, score, judge, **kwargs):
        observed.append((sample.id, judge.timeout, kwargs))
        return evaluate(sample, response, score, judge, **kwargs)

    monkeypatch.setattr(adapter, "evaluate_sample", record)
    monkeypatch.setattr(adapter.DuplexJudge, "chat", lambda *args, **kwargs: '{"success_score": 1}')
    output = patch.MixRequestFuncOutput(success=True)
    output.duplex_eval_result = _result(samples[0])
    summary = adapter.finalize_duplex_eval(samples, [output])
    accuracy = summary["accuracy"]
    assert observed == [("1", 0.5, {"judge_fps": 3, "judge_video_mode": "frame-sample", "window_size": 12.0})]
    assert accuracy["status"] == "completed" and accuracy["evaluated"] == accuracy["total"] == 1
    score_root = Path(accuracy["score_root"])
    assert not (score_root / "PR_correction/0.json").exists()
    assert adapter.summarize_scores(score_root)["pr"] == accuracy["pr"]
    assert accuracy["pr"]["mean_all_success"] == 1
    assert json.loads((score_root / "evaluation_summary.json").read_text()) == accuracy


@pytest.mark.parametrize("failure", ["generation", "publication", "judge", "clock"])
def test_evaluation_reports_incomplete_coverage(tmp_path, prepared, monkeypatch, failure):
    samples = [
        replace(sample, evaluation=adapter.DuplexEvalEvaluation("http://judge", "judge"))
        for sample in _samples(tmp_path)
    ]
    outputs = [patch.MixRequestFuncOutput(success=True) for _ in samples]
    for sample, output in zip(samples, outputs, strict=True):
        output.duplex_eval_result = _result(sample)
    if failure == "generation":
        outputs[0].success = False
    if failure == "clock":
        outputs[0].duplex_eval_result.metadata["clock"] = "invalid"
    publish = adapter.write_sample_result

    def write(result):
        if failure == "publication" and result.output.stem == "0":
            raise OSError("disk full")
        publish(result)

    calls = 0

    def judge_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "judge" and calls == 1:
            raise ConnectionError("judge unavailable")
        return '{"success_score": 1}'

    monkeypatch.setattr(adapter, "write_sample_result", write)
    monkeypatch.setattr(adapter.DuplexJudge, "chat", judge_chat)
    accuracy = adapter.finalize_duplex_eval(samples, outputs)["accuracy"]
    assert accuracy["status"] == "partial" and accuracy["total"] == 2
    assert accuracy["evaluated"] == accuracy["samples"] == 1
    assert accuracy["skipped"] == int(failure in {"generation", "publication"})
    assert accuracy["failed"] == len(accuracy["errors"]) == int(failure in {"judge", "clock"})
    assert outputs[1].success


def test_evaluation_summary_write_failure_preserves_service_results(tmp_path, prepared, monkeypatch):
    sample = replace(_samples(tmp_path)[0], evaluation=adapter.DuplexEvalEvaluation("http://judge", "judge"))
    output = patch.MixRequestFuncOutput(success=True)
    output.duplex_eval_result = _result(sample)
    write = Path.write_text

    def fail_summary(path, *args, **kwargs):
        if path.name == "evaluation_summary.json":
            raise OSError("disk full")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_summary)
    monkeypatch.setattr(adapter.DuplexJudge, "chat", lambda *args, **kwargs: '{"success_score": 1}')
    summary = adapter.finalize_duplex_eval([sample], [output])
    assert summary["published"] == 1 and output.success
    assert summary["accuracy"] == {"status": "failed", "error": "disk full"}


@pytest.mark.parametrize("sample_id", ["same", "../escape"])
def test_rejects_colliding_or_unsafe_artifact_paths(tmp_path, prepared, sample_id):
    args = _args(tmp_path)
    manifest = Path(args.dataset_path)
    rows = json.loads(manifest.read_text())
    for row in rows:
        row["id"] = sample_id
    manifest.write_text(json.dumps(rows if sample_id == "same" else rows[:1]))
    preprocess_serve_args(args)
    with pytest.raises(ValueError, match="unique|path components"):
        patch.get_samples(args, None)
    assert not args.duplex_eval_response_root.exists()


@pytest.mark.asyncio
async def test_adapter_preserves_metrics_and_forwards_request(tmp_path, prepared, monkeypatch):
    sample = _samples(tmp_path)[0]
    expected = _result(sample)
    calls = []

    async def generate(case, **kwargs):
        calls.append((case, kwargs))
        return expected

    monkeypatch.setattr(runner, "generate_sample", generate)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    request = _request(sample)
    request.extra_headers = {"X-Custom": "test"}
    request.extra_body = {"temperature": 0.2}
    output = await patch.async_request_openai_realtime_duplex(request, session=None)
    assert output.success and output.generated_text == "Done."
    assert output.ttft == 0.1 and output.audio_ttfp == 0.2 and output.audio_rtf == 0.5
    assert output.audio_duration == 1 and output.output_tokens == 3
    assert output.itl == [0.01, 0.03] and output.tpot_measured
    assert output.text_latency == pytest.approx(0.14)
    assert output.duplex_session_metrics is expected.session_metrics
    case, kwargs = calls[0]
    assert case is sample.sample and kwargs["prepared"] is prepared
    assert kwargs["model"] == "served-model" and kwargs["url"] == request.api_url
    assert kwargs["additional_headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["additional_headers"]["X-Custom"] == "test"
    assert kwargs["extra_body"] == request.extra_body and kwargs["defer_artifacts"]
    assert not expected.output.exists()
    summary = adapter.finalize_duplex_eval([sample], [output])
    assert summary["published"] == 1 and not summary["artifact_errors"]
    assert json.loads(expected.output.read_text()) == expected.timed_sentences


def test_rejects_response_metadata_collision(tmp_path, prepared):
    args = _args(tmp_path)
    path = Path(args.dataset_path)
    rows = json.loads(path.read_text())
    rows[0]["id"], rows[1]["id"] = "x", "x.meta"
    path.write_text(json.dumps(rows))
    preprocess_serve_args(args)
    with pytest.raises(ValueError, match="must not collide"):
        patch.get_samples(args, None)
    assert not args.duplex_eval_response_root.exists()


@pytest.mark.asyncio
async def test_warmup_does_not_publish_or_skip(tmp_path, prepared, monkeypatch):
    sample = _samples(tmp_path)[0]
    result = _result(sample)
    calls = 0

    async def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return result

    monkeypatch.setattr(runner, "generate_sample", generate)
    warmups = await asyncio.gather(
        *[patch.async_request_openai_realtime_duplex(_request(sample, measured=False), session=None) for _ in range(3)]
    )
    assert all(output.success and not hasattr(output, "duplex_eval_result") for output in warmups)
    assert not result.output.exists()
    output = await patch.async_request_openai_realtime_duplex(_request(sample), session=None)
    assert calls == 4 and output.success


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["protocol", "drain", "exception", "timeout"])
async def test_failed_sessions_are_not_judge_inputs(tmp_path, prepared, monkeypatch, failure):
    sample = _samples(tmp_path)[0]

    async def generate(*args, **kwargs):
        if failure == "exception":
            raise ConnectionError("socket closed")
        if failure == "timeout":
            await asyncio.sleep(10)
        return replace(_result(sample), error="session failed", metadata={"response_done": failure != "drain"})

    monkeypatch.setattr(runner, "generate_sample", generate)
    monkeypatch.setattr(patch, "_omni_request_timeout_s", lambda: 0.01)
    output = await patch.async_request_openai_realtime_duplex(_request(sample), session=None)
    assert not output.success and output.error
    assert adapter.finalize_duplex_eval([sample], [output])["published"] == 0
    assert not _result(sample).output.exists()


@pytest.mark.asyncio
async def test_missing_token_timing_is_not_invented(tmp_path, prepared, monkeypatch):
    sample = _samples(tmp_path)[0]
    result = replace(_result(sample), request_metrics=[], output_tokens=0, session_metrics={})

    async def generate(*args, **kwargs):
        return result

    monkeypatch.setattr(runner, "generate_sample", generate)
    output = await patch.async_request_openai_realtime_duplex(_request(sample), session=None)
    assert output.success and output.itl == [] and not output.tpot_measured
    assert output.duplex_session_metrics == {}


@pytest.mark.asyncio
async def test_cancelled_request_propagates_and_cleans_up(tmp_path, prepared, monkeypatch):
    sample = _samples(tmp_path)[0]
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def generate(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(runner, "generate_sample", generate)
    task = asyncio.create_task(patch.async_request_openai_realtime_duplex(_request(sample), session=None))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and not _result(sample).output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code,success", [("playback_ack_too_late", True), ("invalid_session", False)])
async def test_socket_errors_match_omniinteract_policy(tmp_path, prepared, error_code, success):
    sample = _samples(tmp_path)[0]

    async def handler(socket):
        async for raw in socket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                await socket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.commit":
                for response in [
                    {"type": "response.created", "response": {"id": "r1"}},
                    {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
                    {"type": "error", "error": {"code": error_code}},
                ]:
                    await socket.send(json.dumps(response))
            elif event["type"] == "session.close":
                await socket.send(json.dumps({"type": "session.closed"}))
                return

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        request = _request(sample)
        request.api_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        output = await patch.async_request_openai_realtime_duplex(request, session=None)
    assert output.success is success
    assert adapter.finalize_duplex_eval([sample], [output])["published"] == int(success)


def test_artifact_io_failure_preserves_service_results(tmp_path, prepared, monkeypatch):
    samples = _samples(tmp_path)
    outputs = [patch.MixRequestFuncOutput(success=True) for _ in samples]
    for sample, output in zip(samples, outputs, strict=True):
        output.duplex_eval_result = _result(sample)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(adapter, "write_sample_result", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    summary = adapter.finalize_duplex_eval(samples, outputs)
    assert summary["published"] == 0 and len(summary["artifact_errors"]) == 3
    assert all(output.success for output in outputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_status", [None, 200, 503])
async def test_benchmark_socket_lifecycle_excludes_warmups(tmp_path, prepared, monkeypatch, capsys, judge_status):
    samples = _samples(tmp_path)
    configured: list[dict] = []
    judge_calls = []

    class BenchmarkTime:
        offset = 0.0
        monotonic = staticmethod(time.monotonic)

        @staticmethod
        def perf_counter():
            return time.perf_counter() + BenchmarkTime.offset

    monkeypatch.setattr(patch, "time", BenchmarkTime)

    async def judge_handler(request):
        # Every readiness, warmup and measured session has finished before judging.
        assert len(configured) == 5
        assert request.headers["Authorization"] == "Bearer judge-key"
        body = await request.json()
        assert body["model"] == "judge"
        judge_calls.append(body)
        # Model an hour of judge work without a real wait. It must not change
        # the previously captured benchmark duration or throughput denominator.
        BenchmarkTime.offset = 3600.0
        return web.json_response(
            {"choices": [{"message": {"content": json.dumps({"success_score": len(judge_calls) % 2})}}]},
            status=judge_status,
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", judge_handler)
    judge_server = web.AppRunner(app)
    await judge_server.setup()
    await web.TCPSite(judge_server, "127.0.0.1", 0).start()
    if judge_status is not None:
        options = adapter.DuplexEvalEvaluation(
            f"http://127.0.0.1:{judge_server.addresses[0][1]}/v1", "judge", api_key="judge-key", timeout_s=2
        )
        samples = [replace(sample, evaluation=options) for sample in samples]

    async def handler(socket):
        async for raw in socket:
            event = json.loads(raw)
            if event["type"] == "session.update":
                configured.append(event["session"])
                await socket.send(json.dumps({"type": "session.created"}))
            elif event["type"] == "input_audio_buffer.commit":
                for response in [
                    {"type": "response.created", "response": {"id": "r1"}},
                    {"type": "response.output_text.delta", "response_id": "r1", "delta": "Done."},
                    {
                        "type": "response.output_audio.delta",
                        "response_id": "r1",
                        "sample_rate_hz": 24000,
                        "delta": base64.b64encode(b"\0\0" * 240).decode(),
                    },
                    {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
                ]:
                    await socket.send(json.dumps(response))
            elif event["type"] == "session.close":
                await socket.send(json.dumps({"type": "session.closed"}))
                return

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            result = await patch.benchmark(
                task_type=patch.TaskType.GENERATION,
                endpoint_type="openai-realtime-duplex",
                api_url=f"{url}/v1/realtime",
                base_url=url,
                model_id="mock",
                model_name="mock",
                tokenizer=None,
                input_requests=samples,
                logprobs=None,
                request_rate=float("inf"),
                burstiness=1.0,
                disable_tqdm=True,
                num_warmups=2,
                profile=False,
                selected_percentile_metrics=["ttft", "e2el"],
                selected_percentiles=[50, 99],
                ignore_eos=False,
                goodput_config_dict={},
                max_concurrency=2,
                lora_modules=None,
                extra_headers=None,
                extra_body={"custom": "test"},
                ready_check_timeout_sec=10,
            )
        finally:
            await judge_server.cleanup()
    assert len(configured) == 5  # one readiness, two warmups, two measured sessions
    assert all(session["extra_body"]["custom"] == "test" for session in configured)
    assert result["completed"] == 2
    assert len(result["duplex_session_metrics"]) == 2
    assert len(result["duplex_request_metrics"]) == 2
    assert result["omni_duplex_eval"]["published"] == 2
    root = samples[0].response_root
    assert len(list(root.glob("*/*.meta.json"))) == 2
    assert json.loads((root / "PR_correction/0.json").read_text()) == [
        {"sentence": "Done.", "start": 0.01, "end": 0.01}
    ]
    report = json.loads((root / "duplex_metrics.json").read_text())
    assert len(report["duplex_session_metrics"]) == 2
    assert 0 <= result["duration"] < 3600
    if judge_status is None:
        assert not judge_calls and "accuracy" not in result["omni_duplex_eval"]
        assert "Omni-DuplexEval accuracy:" not in capsys.readouterr().out
    else:
        assert len(judge_calls) == 2  # no scoring for readiness or warmups
        accuracy = result["omni_duplex_eval"]["accuracy"]
        assert accuracy["status"] == ("completed" if judge_status == 200 else "failed")
        assert accuracy["total"] == 2 and accuracy["skipped"] == 0
        assert accuracy["evaluated"] == (2 if judge_status == 200 else 0)
        if judge_status == 200:
            assert accuracy["pr"]["mean_all_success"] == 0.5
        else:
            assert accuracy["failed"] == 2 and "pr" not in accuracy
        assert "Omni-DuplexEval accuracy:" in capsys.readouterr().out
