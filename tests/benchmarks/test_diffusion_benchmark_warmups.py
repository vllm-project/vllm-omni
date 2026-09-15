# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import sys
from argparse import Namespace

import pytest

from benchmarks.diffusion import backends
from benchmarks.diffusion.backends import RequestFuncInput, RequestFuncOutput

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu, pytest.mark.asyncio]


@pytest.fixture
def diffusion_benchmark(monkeypatch):
    # The standalone script imports its sibling backend module by name.
    monkeypatch.setitem(sys.modules, "backends", backends)
    return importlib.import_module("benchmarks.diffusion.diffusion_benchmark_serving")


@pytest.fixture
def warmup_args():
    return Namespace(
        warmup_requests=2,
        warmup_concurrency=2,
        warmup_num_inference_steps=4,
        task="t2i",
    )


@pytest.fixture
def input_requests():
    return [
        RequestFuncInput(
            prompt=prompt,
            api_url="http://test.local/v1/chat/completions",
            model="test-model",
            num_inference_steps=20,
        )
        for prompt in ("first", "second")
    ]


async def test_disabled_warmup_sends_no_requests(diffusion_benchmark, warmup_args, input_requests):
    warmup_args.warmup_requests = 0

    async def unexpected_request(*args):
        pytest.fail("disabled warmup must not send requests")

    pairs = await diffusion_benchmark._run_warmups(input_requests, warmup_args, None, unexpected_request)

    assert pairs == []


async def test_successful_warmups_preserve_request_output_pairs(diffusion_benchmark, warmup_args, input_requests):
    outputs = [RequestFuncOutput(success=True, latency=0.5), RequestFuncOutput(success=True, latency=1.0)]
    pending_outputs = iter(outputs)

    async def successful_request(req, session, pbar):
        return next(pending_outputs)

    pairs = await diffusion_benchmark._run_warmups(input_requests, warmup_args, None, successful_request)

    assert [request.prompt for request, _ in pairs] == ["first", "second"]
    assert [request.num_inference_steps for request, _ in pairs] == [4, 4]
    assert [output for _, output in pairs] == outputs
    assert [request.num_inference_steps for request in input_requests] == [20, 20]


@pytest.mark.parametrize("failed_count", [1, 2], ids=["partial-failure", "all-failed"])
@pytest.mark.parametrize(
    "error",
    ["HTTP 503: private prompt rejected", "Cannot connect to https://test.local/?token=private"],
    ids=["http-error", "transport-error"],
)
async def test_failed_warmups_abort_without_exposing_response_errors(
    diffusion_benchmark, warmup_args, input_requests, failed_count, error
):
    outputs = iter(
        [RequestFuncOutput(success=False, error=error) for _ in range(failed_count)]
        + [RequestFuncOutput(success=True) for _ in range(2 - failed_count)]
    )

    async def request(req, session, pbar):
        return next(outputs)

    with pytest.raises(RuntimeError, match=f"{failed_count}/2 warmup requests failed") as exc_info:
        await diffusion_benchmark._run_warmups(input_requests, warmup_args, None, request)

    assert error not in str(exc_info.value)
    assert "server logs" in str(exc_info.value)


async def test_benchmark_does_not_start_measurement_after_failed_warmup(
    monkeypatch, mocker, diffusion_benchmark, warmup_args, input_requests
):
    args = Namespace(
        **vars(warmup_args),
        base_url="http://test.local",
        endpoint="/v1/chat/completions",
        dataset="random",
        model="test-model",
        enable_negative_prompt=False,
        return_stage_metrics=False,
        extra_body=None,
        max_concurrency=1,
        disable_tqdm=True,
        slo=False,
    )
    sent = []

    async def failed_request(req, session, pbar):
        sent.append(req)
        return RequestFuncOutput(success=False, error="HTTP 503: unavailable")

    monkeypatch.setitem(
        diffusion_benchmark.backends_function_mapping["2i"],
        args.endpoint,
        (failed_request, args.endpoint),
    )
    dataset = mocker.Mock(spec=diffusion_benchmark.RandomDataset)
    dataset.get_requests.return_value = input_requests
    monkeypatch.setattr(diffusion_benchmark, "RandomDataset", mocker.Mock(return_value=dataset))
    timer = mocker.Mock(spec=diffusion_benchmark.time)
    timer.perf_counter.side_effect = AssertionError("measurement started after failed warmup")
    monkeypatch.setattr(diffusion_benchmark, "time", timer)

    with pytest.raises(RuntimeError, match="2/2 warmup requests failed"):
        await diffusion_benchmark.benchmark(args)

    assert len(sent) == warmup_args.warmup_requests
