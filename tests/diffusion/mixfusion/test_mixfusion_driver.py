# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The driver executes approved batches and never regroups rejected inputs."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm_omni.diffusion.mixfusion.chunked import ChunkedJointAdapter
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.plan import MixFusionRoot
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionBatchSpec
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _Pipeline:
    interrupt: bool = False
    mixfusion_driver: MixFusionDriver | None = None


def _driver():
    pipeline = _Pipeline()
    transformer = nn.Identity()
    transformer._mixfusion_plan = {}
    transformer._mixfusion_hooks_applied = True
    pipeline.mixfusion_driver = MixFusionDriver(pipeline, transformer, ChunkedJointAdapter(MixFusionRoot()))
    return pipeline.mixfusion_driver


def _states(lengths=(8, 12)):
    return [
        StepRequestState(
            request_id=str(i),
            sampling=OmniDiffusionSamplingParams(),
            latents=torch.zeros(1, length, 4),
            timesteps=torch.tensor([500.0]),
        )
        for i, length in enumerate(lengths)
    ]


def _spec(states, chunk=4):
    return MixFusionBatchSpec(chunk, {state.request_id: tuple(state.latents.shape) for state in states})


def test_dense_execution_never_arms_packing():
    states = _states((8, 8))
    batch = InputBatch.make_batch(states)
    with override_forward_context(ForwardContext()):
        result = _driver().run(batch, states, lambda b, **kw: (b.latents, get_forward_context().mixfusion_step))
    assert result[0].shape == (2, 8, 4)
    assert result[1] is None


def test_approved_batch_uses_scheduler_chunk_and_restores_context_on_error():
    states = _states()
    batch = InputBatch.make_batch(states, mixfusion_batch=_spec(states, chunk=2))
    calls = []

    def dense_step(batch, **kwargs):
        calls.append(get_forward_context().mixfusion_step.chunk_size)
        raise RuntimeError("model failure")

    with override_forward_context(ForwardContext()):
        with pytest.raises(RuntimeError, match="model failure"):
            _driver().run(batch, states, dense_step)
        assert get_forward_context().mixfusion_step is None
    assert calls == [2]


def test_bad_shape_fails_before_any_forward():
    states = _states()
    spec = _spec(states)
    states[1].latents = torch.zeros(1, 16, 4)
    batch = InputBatch.make_batch(states, mixfusion_batch=spec)
    with override_forward_context(ForwardContext()):
        with pytest.raises(ValueError, match="request 1: scheduled latent shape"):
            _driver().run(batch, states, lambda *a, **k: pytest.fail("must not run or fall back"))


def test_failed_request_subset_keeps_the_scheduled_chunk_size():
    states = _states()
    spec = _spec(states)
    batch = InputBatch.make_batch(states[1:], mixfusion_batch=spec)
    with override_forward_context(ForwardContext()):
        result = _driver().run(batch, states[1:], lambda b, **k: get_forward_context().mixfusion_step.chunk_size)
    assert result == 4  # The survivor's own GCD would be 12.


def test_runtime_hook_change_is_an_error():
    driver = _driver()
    states = _states()
    batch = InputBatch.make_batch(states, mixfusion_batch=_spec(states))
    driver.transformer.register_forward_pre_hook(lambda module, args: args)
    with pytest.raises(ValueError, match="foreign hooks"):
        driver.validate(batch)


def test_input_batch_cache_tracks_execution_mode_and_chunk_changes():
    states = _states((8, 8))
    dense = InputBatch.make_batch(states)
    packed = InputBatch.make_batch(states, cached_batch=dense, mixfusion_batch=_spec(states))
    assert isinstance(packed.latents, list)
    smaller_chunks = _spec(states, chunk=2)
    cached = InputBatch.make_batch(states, cached_batch=packed, mixfusion_batch=smaller_chunks)
    assert cached is packed and cached.mixfusion_batch is smaller_chunks
    dense_again = InputBatch.make_batch(states, cached_batch=cached)
    assert isinstance(dense_again.latents, torch.Tensor)


def test_remote_contract_failure_prevents_local_forward(monkeypatch):
    import vllm_omni.diffusion.mixfusion.driver as driver_module

    monkeypatch.setattr(driver_module, "_any_rank_failed", lambda local: True)
    states = _states()
    batch = InputBatch.make_batch(states, mixfusion_batch=_spec(states))
    with pytest.raises(RuntimeError, match="another DiT rank"):
        _driver().run(batch, states, lambda *a, **k: pytest.fail("must not enter model collectives"))


def test_contract_failure_reduction_uses_cpu_group(monkeypatch):
    import vllm_omni.diffusion.distributed.parallel_state as parallel_state
    import vllm_omni.diffusion.mixfusion.driver as driver_module

    @dataclass
    class World:
        world_size: int = 2
        cpu_group: object = None

    group = object()
    monkeypatch.setattr(driver_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: World(cpu_group=group))
    calls = []

    def reduce(failed, *, op, group):
        calls.append((failed.device.type, op, group))
        failed.fill_(1)  # A peer rejected its execution contract.

    monkeypatch.setattr(driver_module.dist, "all_reduce", reduce)
    assert driver_module._any_rank_failed(False)
    assert calls == [("cpu", driver_module.dist.ReduceOp.MAX, group)]
