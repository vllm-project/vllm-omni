# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import vllm.envs as envs
from vllm import SamplingParams

from vllm_omni.diffusion.batch_invariance import DIFFUSION_BATCH_INVARIANT_ENV
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _unset_diffusion_switch(monkeypatch):
    """Stop an inherited explicit switch from silently disabling the batch-invariant cases."""
    monkeypatch.delenv(DIFFUSION_BATCH_INVARIANT_ENV, raising=False)


class _DummyDiffusionStage:
    stage_type = "diffusion"
    final_output = True

    def __init__(self) -> None:
        self.calls = []

    async def add_request_async(self, request_id, prompt, sampling_params, kv_sender_info=None):
        self.calls.append((request_id, prompt, sampling_params, kv_sender_info))


class _GeneratorInitializingDiffusionStage(_DummyDiffusionStage):
    async def add_request_async(self, request_id, prompt, sampling_params, kv_sender_info=None):
        if sampling_params.generator is None:
            sampling_params.generator = torch.Generator(device="cpu").manual_seed(sampling_params.seed)
        await super().add_request_async(request_id, prompt, sampling_params, kv_sender_info)


@pytest.mark.asyncio
async def test_plain_sampling_seed_survives_initial_and_streaming_normalization(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    stage = _DummyDiffusionStage()
    pool = StagePool(0, stage)
    state = SimpleNamespace(sampling_params_list=[SamplingParams(seed=-2)])

    await pool.submit_initial("request-test", state, "prompt")
    await pool.submit_update("request-test", state, "prompt-update")

    assert [call[2].seed for call in stage.calls] == [-2, -2]
    assert all(isinstance(call[2], OmniDiffusionSamplingParams) for call in stage.calls)


@pytest.mark.asyncio
async def test_feature_off_plain_sampling_params_keep_caller_seed(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    stage = _DummyDiffusionStage()
    pool = StagePool(0, stage)
    state = SimpleNamespace(sampling_params_list=[SamplingParams(seed=17)])

    await pool.submit_initial("request-test", state, "prompt")
    await pool.submit_update("request-test", state, "prompt-update")

    assert [call[2].seed for call in stage.calls] == [17, 17]
    assert all(isinstance(call[2], OmniDiffusionSamplingParams) for call in stage.calls)


@pytest.mark.asyncio
async def test_feature_off_native_omni_sampling_params_preserve_seed(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    stage = _DummyDiffusionStage()
    pool = StagePool(0, stage)
    params = OmniDiffusionSamplingParams(seed=17)
    state = SimpleNamespace(sampling_params_list=[params])

    await pool.submit_initial("request-test", state, "prompt")
    await pool.submit_update("request-test", state, "prompt-update")

    assert [call[2].seed for call in stage.calls] == [17, 17]
    assert all(call[2] is params for call in stage.calls)


@pytest.mark.asyncio
async def test_batch_invariant_validation_runs_before_replica_binding(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    stage = _DummyDiffusionStage()
    pool = StagePool(0, stage)
    state = SimpleNamespace(sampling_params_list=[OmniDiffusionSamplingParams()])

    with pytest.raises(ValueError, match="explicit integer seed"):
        await pool.submit_initial("request-test", state, "prompt")

    assert stage.calls == []
    assert pool.get_bound_replica_id("request-test") is None


@pytest.mark.asyncio
async def test_batch_invariant_stage_pool_rejects_generator_only(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    stage = _DummyDiffusionStage()
    pool = StagePool(0, stage)
    state = SimpleNamespace(
        sampling_params_list=[OmniDiffusionSamplingParams(generator=torch.Generator(device="cpu").manual_seed(7))]
    )

    with pytest.raises(ValueError, match="does not accept generator"):
        await pool.submit_initial("request-test", state, "prompt")

    assert stage.calls == []


@pytest.mark.asyncio
async def test_batch_invariant_dispatch_isolates_internal_generator_mutation(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    stage = _GeneratorInitializingDiffusionStage()
    pool = StagePool(0, stage)
    params = OmniDiffusionSamplingParams(seed=7)
    state = SimpleNamespace(sampling_params_list=[params])

    await pool.submit_initial("request-test", state, "prompt")
    await pool.submit_update("request-test", state, "prompt-update-1")
    await pool.submit_update("request-test", state, "prompt-update-2")

    dispatched_params = [call[2] for call in stage.calls]
    assert [item.seed for item in dispatched_params] == [7, 7, 7]
    assert all(item.generator.initial_seed() == 7 for item in dispatched_params)
    assert len({id(item) for item in dispatched_params}) == 3
    assert params.generator is None
