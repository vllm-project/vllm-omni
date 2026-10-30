# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.diffusion.test_mossimage_slg import pipeline, request, sampling
from vllm_omni.diffusion.cancellation import RequestCancellationRegistry, request_cancellation_scope
from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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
