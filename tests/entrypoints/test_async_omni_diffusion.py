# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_omni.diffusion.stage_diffusion_client as stage_diffusion_client_module
from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_diffusion_od_config_returns_direct_config():
    diffusion = object.__new__(AsyncOmniDiffusion)
    diffusion.od_config = object()

    assert diffusion.get_diffusion_od_config() is diffusion.od_config


def test_async_omni_diffusion_generate_aborts_engine_on_cancel():
    async def run_test():
        started = threading.Event()
        release = threading.Event()
        abort = Mock()

        def step(request):
            del request
            started.set()
            release.wait(timeout=5)
            return [SimpleNamespace(request_id="req-1")]

        diffusion = object.__new__(AsyncOmniDiffusion)
        diffusion.engine = SimpleNamespace(step=step, abort=abort)
        diffusion._executor = ThreadPoolExecutor(max_workers=1)

        task = asyncio.create_task(
            diffusion.generate(
                prompt="hello",
                sampling_params=OmniDiffusionSamplingParams(),
                request_id="req-1",
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            diffusion._executor.shutdown(wait=True)

        abort.assert_called_once_with("req-1")

    asyncio.run(run_test())


def test_stage_diffusion_client_abort_requests_forwards_to_engine():
    async def run_test():
        aborted_request_ids: list[list[str]] = []

        async def abort(request_ids):
            aborted_request_ids.append(request_ids)

        client = object.__new__(StageDiffusionClient)
        client._engine = SimpleNamespace(abort=abort)
        client._tasks = {}

        task = asyncio.create_task(asyncio.sleep(60))
        client._tasks["req-1"] = task

        await client.abort_requests_async(["req-1", "req-2"])

        with pytest.raises(asyncio.CancelledError):
            await task
        assert client._tasks == {}
        assert aborted_request_ids == [["req-1", "req-2"]]

    asyncio.run(run_test())


def test_stage_diffusion_client_run_treats_abort_as_normal_path(monkeypatch):
    async def run_test():
        async def generate(prompt, sampling_params, request_id):
            del prompt, sampling_params
            raise DiffusionRequestAbortedError(f"Request {request_id} aborted.")

        info = Mock()
        exception = Mock()
        monkeypatch.setattr(stage_diffusion_client_module.logger, "info", info)
        monkeypatch.setattr(stage_diffusion_client_module.logger, "exception", exception)

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 3
        client._engine = SimpleNamespace(generate=generate)
        client._output_queue = asyncio.Queue()
        client._tasks = {"req-1": object()}

        await client._run("req-1", "prompt", OmniDiffusionSamplingParams())

        assert client._output_queue.empty()
        assert client._tasks == {}
        info.assert_called_once()
        exception.assert_not_called()

    asyncio.run(run_test())
