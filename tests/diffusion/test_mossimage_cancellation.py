# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig

from tests.diffusion.test_mossimage_slg import pipeline, request, sampling
from vllm_omni.diffusion.cancellation import RequestCancellationRegistry, request_cancellation_scope
from vllm_omni.diffusion.data import DiffusionRequestAbortedError, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, supports_request_cancellation
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("step_execution", [False, True])
def test_cooperative_cancellation_is_full_forward_only(step_execution):
    config = OmniDiffusionConfig(model="test", model_class_name="MOSSImagePipeline", step_execution=step_execution)
    assert supports_request_cancellation(config) is (not step_execution)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_ids", [("cancelled",), ("cancelled", "live")])
async def test_step_execution_uses_scheduler_cancellation(monkeypatch, cancel_ids):
    model = pipeline()

    def init_cpu(worker):
        worker.device = torch.device("cpu")
        worker.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))

    def load_pipeline(worker, **kwargs):
        worker.model_runner.pipeline = model
        worker.model_runner.kv_transfer_manager = None

    monkeypatch.setattr(DiffusionWorker, "init_device", init_cpu)
    monkeypatch.setattr(DiffusionWorker, "load_model", load_pipeline)
    for method in ("is_available", "max_memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(current_omni_platform, method, lambda: 0)
    engine = DiffusionEngine(
        OmniDiffusionConfig(
            model="test",
            model_class_name="MOSSImagePipeline",
            step_execution=True,
            distributed_executor_backend="uni",
            max_num_seqs=2,
        )
    )
    batches = []
    denoise_step = model.denoise_step

    def abort_during_step(batch, **kwargs):
        batches.append(list(batch.request_ids))
        result = denoise_step(batch, **kwargs)
        engine.abort(cancel_ids)
        return result

    monkeypatch.setattr(model, "denoise_step", abort_during_step)

    async def collect(request_id):
        return [output async for output in engine.get_output_stream(request_id)]

    try:
        assert engine._request_cancellations is None
        for request_id in ("cancelled", "live"):
            params = sampling()
            params.num_inference_steps = 2
            req = request(params, request_id)
            engine.add_request(req)
            assert req.cancellation_signal is None
        await engine._check_and_start_background_loop()
        cancelled, live = await asyncio.wait_for(asyncio.gather(collect("cancelled"), collect("live")), timeout=10)
        assert len(cancelled) == len(live) == 1
        assert cancelled[0].finished and cancelled[0].aborted
        assert live[0].finished and live[0].aborted == ("live" in cancel_ids)
        assert live[0].error is None
        if "live" in cancel_ids:
            assert batches == [["cancelled", "live"]]
        else:
            assert live[0].output is not None
            assert batches == [["cancelled", "live"], ["live"]]
    finally:
        engine.close()


@pytest.mark.parametrize("cancel_after_step", [0, 1, 2])
def test_cancelled_forward_stops_before_more_steps_or_decode(monkeypatch, mocker, cancel_after_step):
    model = pipeline()
    params = sampling()
    params.num_inference_steps = 2
    registry = RequestCancellationRegistry()
    signal = registry.create("req")
    synchronize = mocker.Mock()
    monkeypatch.setattr(current_omni_platform, "synchronize", synchronize)
    decode = mocker.Mock()
    model._vae_decode = decode
    original_step = model.scheduler_step_maybe_with_cfg
    steps = 0

    def step(*args):
        nonlocal steps
        result = original_step(*args)
        steps += 1
        if steps == cancel_after_step:
            registry.cancel(["req"])
        return result

    model.scheduler_step_maybe_with_cfg = step
    try:
        if cancel_after_step == 0:
            registry.cancel(["req"])
        with request_cancellation_scope([signal]), pytest.raises(DiffusionRequestAbortedError):
            model(DiffusionRequestBatch([request(params)]))
        assert steps == cancel_after_step
        decode.assert_not_called()
        synchronize.assert_called_once_with()
        assert model.supports_request_cancellation
    finally:
        registry.close()


def test_cancelled_request_does_not_interrupt_live_batch_peer(monkeypatch, mocker):
    model = pipeline()
    registry = RequestCancellationRegistry()
    signals = [registry.create(request_id) for request_id in ("cancelled", "live")]
    synchronize = mocker.Mock()
    monkeypatch.setattr(current_omni_platform, "synchronize", synchronize)
    try:
        registry.cancel(["cancelled"])
        with request_cancellation_scope(signals):
            outputs = model(DiffusionRequestBatch([request(sampling(), "cancelled"), request(sampling(), "live")]))
        assert len(outputs) == 2
        assert all(output.output is not None for output in outputs)
        synchronize.assert_not_called()
    finally:
        registry.close()
