# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for the last step request terminating before engine idle."""

import asyncio
import threading
import weakref

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Pipeline(torch.nn.Module):
    supports_step_execution = True

    def prepare_encode(self, state):
        state.timesteps = torch.arange(2)
        state.latents = torch.zeros(1, 2)
        state.extra["private"] = torch.ones(16)
        # Never retain the state or private tensor in the test itself.
        self.refs = (weakref.ref(state), weakref.ref(state.extra["private"]))

    def denoise_step(self, batch, **kwargs):
        for state in batch.states:
            state.extra["private"].add_(1)
        return torch.ones_like(batch.latents)

    def step_scheduler(self, state, noise_pred):
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state):
        return DiffusionOutput(output=state.latents)


@pytest.fixture
def runtime(monkeypatch):
    def init_cpu(worker):
        worker.device = torch.device("cpu")
        worker.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))

    def load_pipeline(worker, **kwargs):
        worker.model_runner.pipeline = _Pipeline()
        worker.model_runner.kv_transfer_manager = None  # No distributed KV transfer in this CPU test.

    # Bypass hardware/distributed initialization and model loading, not request cleanup.
    monkeypatch.setattr(DiffusionWorker, "init_device", init_cpu)
    monkeypatch.setattr(DiffusionWorker, "load_model", load_pipeline)
    for method in ("is_available", "max_memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(current_omni_platform, method, lambda: 0)
    engine = DiffusionEngine(OmniDiffusionConfig(model="test", step_execution=True, distributed_executor_backend="uni"))
    idle = threading.Event()
    wait = engine._cv.wait

    def observe_idle(timeout=None):
        if not engine.scheduler.has_requests():
            idle.set()
        return wait(timeout)

    monkeypatch.setattr(engine._cv, "wait", observe_idle)
    try:
        yield engine, idle
    finally:
        engine.close()  # Only after the test observed the idle boundary.


async def _run_to_idle(runtime, *, aborted=False):
    engine, idle = runtime
    engine.add_request(
        OmniDiffusionRequest(
            request_id="probe", prompt="test", sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2)
        )
    )
    await engine._check_and_start_background_loop()
    assert await asyncio.to_thread(idle.wait, 10), "engine never reached idle"
    with engine._cv:
        assert not engine.scheduler.has_requests()
    assert not engine._out_streams["probe"].empty(), "terminal output missing at idle"
    async for output in engine.get_output_stream("probe"):
        assert output.finished and output.aborted == aborted and output.error is None
        if not aborted:
            assert torch.equal(output.output, torch.full((1, 2), 2.0))
    pipeline = engine.executor.driver_worker.worker.model_runner.pipeline
    assert all(ref() is None for ref in pipeline.refs), "request resources retained at idle"


@pytest.mark.asyncio
async def test_completion_releases_private_resources_at_idle(runtime):
    await _run_to_idle(runtime)


@pytest.mark.asyncio
async def test_last_abort_delivers_terminal_output_and_releases_resources(runtime, monkeypatch):
    engine, _ = runtime
    emit = engine._emit_outputs

    def abort_between_steps(*args, **kwargs):
        emit(*args, **kwargs)
        engine.abort("probe")

    monkeypatch.setattr(engine, "_emit_outputs", abort_between_steps)
    await _run_to_idle(runtime, aborted=True)
