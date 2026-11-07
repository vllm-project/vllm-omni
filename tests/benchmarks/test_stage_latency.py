# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import csv
import json
import socket

import pytest
from aiohttp import web

from benchmarks.diffusion import backends
from benchmarks.diffusion.backends import RequestFuncInput, RequestFuncOutput
from benchmarks.diffusion.bench_stage_latency import build_parser, make_record, run, save_results

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def output(**kwargs) -> RequestFuncOutput:
    return RequestFuncOutput(
        **{"success": True, "error": "", "latency": 5.0, "stage_durations": {"text": 1.0, "gen": 3.0}, **kwargs}
    )


def test_records_preserve_whole_phase_times(tmp_path):
    records = [make_record(0, output(), "text", "gen"), make_record(1, output(latency=7.0), "text", "gen")]
    prefix = tmp_path / "measurements"
    save_results(prefix, {"tp": 2}, records)
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["records"] == records
    assert report["summary_seconds"]["end_to_end_seconds"] == {"mean": 6.0, "median": 6.0}
    with prefix.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert float(rows[0]["end_to_end_seconds"]) == 5.0  # not text + generation


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_stage_times_are_errors(bad):
    with pytest.raises(ValueError, match="Invalid duration"):
        make_record(0, output(stage_durations={"text": bad, "gen": 3.0}), "text", "gen")


def test_missing_metrics_and_failed_requests_are_errors():
    with pytest.raises(ValueError, match="Missing stage"):
        make_record(0, output(stage_durations={}), "text", "gen")
    with pytest.raises(RuntimeError, match="kernel failure"):
        make_record(0, output(success=False, error="kernel failure"), "text", "gen")


def test_sequential_run_excludes_warmup_and_requests_one_image(monkeypatch, tmp_path):
    calls = []

    async def request(req, session):
        assert isinstance(req, RequestFuncInput)
        calls.append(req)
        return output(latency=float(len(calls)))

    monkeypatch.setattr(backends, "async_request_chat_completions", request)
    args = build_parser().parse_args(
        [
            "--model",
            "test-model",
            "--prompt",
            "test",
            "--width",
            "2560",
            "--height",
            "1760",
            "--num-inference-steps",
            "8",
            "--text-stage",
            "text",
            "--generation-stage",
            "gen",
            "--warmup",
            "2",
            "--repeats",
            "3",
            "--output-prefix",
            str(tmp_path / "run"),
        ]
    )
    asyncio.run(run(args))
    report = json.loads((tmp_path / "run.json").read_text())
    assert [row["end_to_end_seconds"] for row in report["records"]] == [3.0, 4.0, 5.0]
    assert len(calls) == 5
    assert all(call.extra_body["num_outputs_per_prompt"] == 1 for call in calls)
    assert report["configuration"]["concurrency"] == 1
    args.extra_body = {"width": 1024}
    with pytest.raises(ValueError, match="conflicts"):
        asyncio.run(run(args))
    assert len(calls) == 5


@pytest.mark.parametrize("flag", ["--warmup", "--repeats"])
def test_warmup_and_measurement_required(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, "0"])


@pytest.mark.parametrize("metrics_location", ["content", "top_level"])
def test_real_http_backend_preserves_metrics_and_discards_warmup(tmp_path, metrics_location):
    async def exercise():
        calls = []
        active = 0
        peak_active = 0

        async def handle(request):
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            payload = await request.json()
            calls.append(payload)
            # Yield so concurrent requests would overlap if the runner regresses.
            await asyncio.sleep(0)
            durations = {"text": float(len(calls)), "gen": 0.75}
            item: dict[str, object] = {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}
            response: dict[str, object] = {"choices": [{"message": {"content": [item]}}]}
            if metrics_location == "content":
                item["stage_durations"] = durations
            else:
                response["metrics"] = {"stage_durations": durations}
            active -= 1
            return web.json_response(response)

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            site = web.SockSite(runner, sock)
            await site.start()
            try:
                args = build_parser().parse_args(
                    [
                        "--base-url",
                        f"http://127.0.0.1:{port}",
                        "--model",
                        "test-model",
                        "--prompt",
                        "A mountain lake",
                        "--width",
                        "2560",
                        "--height",
                        "1760",
                        "--num-inference-steps",
                        "8",
                        "--text-stage",
                        "text",
                        "--generation-stage",
                        "gen",
                        "--warmup",
                        "2",
                        "--repeats",
                        "3",
                        "--output-prefix",
                        str(tmp_path / "http"),
                    ]
                )
                await run(args)
            finally:
                await runner.cleanup()
        assert len(calls) == 5
        assert peak_active == 1
        for payload in calls:
            assert payload["model"] == "test-model"
            assert payload["messages"] == [{"role": "user", "content": "A mountain lake"}]
            assert payload["extra_body"] == {
                "return_stage_metrics": True,
                "num_outputs_per_prompt": 1,
                "width": 2560,
                "height": 1760,
                "num_inference_steps": 8,
                "seed": 42,
            }
        report = json.loads((tmp_path / "http.json").read_text())
        assert [row["text_seconds"] for row in report["records"]] == [3.0, 4.0, 5.0]
        assert all(row["generation_seconds"] == 0.75 for row in report["records"])
        assert all(row["end_to_end_seconds"] > 0 for row in report["records"])

    asyncio.run(exercise())
