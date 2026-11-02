# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import importlib
import sys
from dataclasses import dataclass, field

import pytest
import pytest_asyncio
from aiohttp import web

from benchmarks.diffusion import backends

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu, pytest.mark.asyncio]


@dataclass
class _ProfilerServer:
    base_url: str
    events: list[tuple[str, dict]] = field(default_factory=list)
    start_status: int = 200
    stop_status: int = 200
    warmup_status: int = 200
    block_generation: bool = False
    generation_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_generation: asyncio.Event = field(default_factory=asyncio.Event)


@pytest_asyncio.fixture
async def profiler_server(unused_tcp_port):
    server = _ProfilerServer(f"http://127.0.0.1:{unused_tcp_port}")

    async def profile(request):
        action = request.path.removeprefix("/").removesuffix("_profile")
        server.events.append((action, await request.json()))
        status = server.start_status if action == "start" else server.stop_status
        return web.json_response({"status": "SUCCESS", "detail": "private server details"}, status=status)

    async def generate(request):
        payload = await request.json()
        warmup = payload["num_inference_steps"] == 1
        server.events.append(("warmup" if warmup else "generate", payload))
        if not warmup and server.block_generation:
            server.generation_started.set()
            await server.release_generation.wait()
        return web.json_response({"data": []}, status=server.warmup_status if warmup else 200)

    async def video(request):
        await request.read()
        if request.method == "POST":
            server.events.append(("submit", {}))
            return web.json_response({"id": "job", "status": "queued"})
        if request.path.endswith("/content"):
            server.events.append(("content", {}))
            return web.Response(body=b"video")
        if request.method == "DELETE":
            server.events.append(("delete", {}))
            return web.json_response({})
        server.events.append(("poll", {}))
        return web.json_response({"id": "job", "status": "completed"})

    app = web.Application()
    app.router.add_post("/start_profile", profile)
    app.router.add_post("/stop_profile", profile)
    app.router.add_post("/v1/images/generations", generate)
    app.router.add_post("/v1/videos", video)
    app.router.add_route("*", "/v1/videos/job", video)
    app.router.add_get("/v1/videos/job/content", video)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    try:
        yield server
    finally:
        server.release_generation.set()
        await runner.cleanup()


@pytest.fixture
def diffusion_benchmark(monkeypatch):
    monkeypatch.setitem(sys.modules, "backends", backends)
    return importlib.import_module("benchmarks.diffusion.diffusion_benchmark_serving")


@pytest.fixture
def benchmark_args(diffusion_benchmark, profiler_server):
    args = diffusion_benchmark.build_parser().parse_args(
        [
            "--base-url",
            profiler_server.base_url,
            "--model",
            "test-model",
            "--dataset",
            "random",
            "--task",
            "t2i",
            "--endpoint",
            "/v1/images/generations",
            "--num-prompts",
            "2",
            "--num-inference-steps",
            "4",
            "--warmup-requests",
            "1",
            "--warmup-num-inference-steps",
            "1",
            "--disable-tqdm",
        ]
    )
    # Also exercise benchmark() directly against the pre-feature implementation.
    args.enable_profiler = True
    args.profiler_stages = None
    return args


async def test_profiler_cli_defaults_are_disabled(diffusion_benchmark):
    args = diffusion_benchmark.build_parser().parse_args([])
    assert args.enable_profiler is False
    assert args.profiler_stages is None


async def test_profiler_cli_selects_stages(diffusion_benchmark):
    args = diffusion_benchmark.build_parser().parse_args(["--enable-profiler", "--profiler-stages", "0", "2"])
    assert args.enable_profiler is True
    assert args.profiler_stages == [0, 2]


@pytest.mark.parametrize("stages", [None, [0], [0, 2]])
async def test_profiler_brackets_measured_http_requests(
    diffusion_benchmark, profiler_server, benchmark_args, stages, tmp_path
):
    benchmark_args.profiler_stages = stages
    benchmark_args.output_file = str(tmp_path / "metrics.json")

    await diffusion_benchmark.benchmark(benchmark_args)

    assert [event for event, _ in profiler_server.events] == ["warmup", "start", "generate", "generate", "stop"]
    controls = [payload for event, payload in profiler_server.events if event in {"start", "stop"}]
    assert controls == ([{}] * 2 if stages is None else [{"stages": stages}] * 2)
    metrics = diffusion_benchmark.json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["completed_requests"] == 2


async def test_disabled_profiler_sends_only_generation_requests(diffusion_benchmark, profiler_server, benchmark_args):
    benchmark_args.enable_profiler = False

    await diffusion_benchmark.benchmark(benchmark_args)

    assert [event for event, _ in profiler_server.events] == ["warmup", "generate", "generate"]


async def test_profiler_covers_async_video_job_and_cleanup(diffusion_benchmark, profiler_server, benchmark_args):
    benchmark_args.task = "t2v"
    benchmark_args.endpoint = "/v1/videos"
    benchmark_args.num_prompts = 1
    benchmark_args.warmup_requests = 0

    await diffusion_benchmark.benchmark(benchmark_args)

    assert [event for event, _ in profiler_server.events] == ["start", "submit", "poll", "content", "delete", "stop"]


@pytest.mark.parametrize("stages, enabled", [([0], False), ([-1], True)])
async def test_invalid_profiler_options_fail_before_http(
    diffusion_benchmark, profiler_server, benchmark_args, stages, enabled
):
    benchmark_args.profiler_stages = stages
    benchmark_args.enable_profiler = enabled

    with pytest.raises(ValueError, match="--profiler-stages"):
        await diffusion_benchmark.benchmark(benchmark_args)

    assert profiler_server.events == []


async def test_warmup_failure_does_not_start_profiler(diffusion_benchmark, profiler_server, benchmark_args):
    profiler_server.warmup_status = 503

    with pytest.raises(RuntimeError, match="warmup requests failed"):
        await diffusion_benchmark.benchmark(benchmark_args)

    assert [event for event, _ in profiler_server.events] == ["warmup"]


@pytest.mark.parametrize("action", ["start", "stop"])
async def test_profiler_http_error_is_reported(diffusion_benchmark, profiler_server, benchmark_args, action):
    setattr(profiler_server, action + "_status", 503)

    with pytest.raises(RuntimeError, match=rf"{action}.*HTTP 503") as error:
        await diffusion_benchmark.benchmark(benchmark_args)

    assert "private server details" not in str(error.value)
    events = [event for event, _ in profiler_server.events]
    assert events == (["warmup", "start"] if action == "start" else ["warmup", "start", "generate", "generate", "stop"])


async def test_profiler_stops_after_measurement_error(
    diffusion_benchmark, profiler_server, benchmark_args, monkeypatch
):
    request_function, endpoint = diffusion_benchmark.backends_function_mapping["2i"][benchmark_args.endpoint]

    async def request_then_fail(request, session, progress):
        result = await request_function(request, session, progress)
        if request.num_inference_steps != 1:
            raise RuntimeError("measurement failed")
        return result

    monkeypatch.setitem(
        diffusion_benchmark.backends_function_mapping["2i"], benchmark_args.endpoint, (request_then_fail, endpoint)
    )

    with pytest.raises(RuntimeError, match="measurement failed"):
        await diffusion_benchmark.benchmark(benchmark_args)

    assert profiler_server.events[-1] == ("stop", {})


async def test_profiler_stops_when_benchmark_is_cancelled(diffusion_benchmark, profiler_server, benchmark_args):
    benchmark_args.warmup_requests = 0
    benchmark_args.num_prompts = 1
    profiler_server.block_generation = True
    task = asyncio.create_task(diffusion_benchmark.benchmark(benchmark_args))
    try:
        await asyncio.wait_for(profiler_server.generation_started.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert [event for event, _ in profiler_server.events] == ["start", "generate", "stop"]
