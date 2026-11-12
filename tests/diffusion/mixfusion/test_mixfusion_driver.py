# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The driver executes approved batches and never regroups rejected inputs."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm_omni.diffusion.mixfusion.adapter import PackedInputs
from vllm_omni.diffusion.mixfusion.chunked import ChunkedJointAdapter
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.plan import MixFusionRoot
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionRequestInfo
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
            mixfusion_info=MixFusionRequestInfo(1, length, 1, 128),
        )
        for i, length in enumerate(lengths)
    ]


def test_dense_execution_never_arms_packing():
    states = _states((8, 8))
    batch = InputBatch.make_batch(states)
    with override_forward_context(ForwardContext()):
        result = _driver().run(batch, states, lambda b, **kw: (b.latents, get_forward_context().mixfusion_step))
    assert result[0].shape == (2, 8, 4)
    assert result[1] is None


def test_approved_batch_arms_empty_context_and_restores_it_on_error():
    states = _states()
    batch = InputBatch.make_batch(states, use_mixfusion=True)
    calls = []

    def dense_step(batch, **kwargs):
        calls.append(get_forward_context().mixfusion_step.plan)
        raise RuntimeError("model failure")

    with override_forward_context(ForwardContext()):
        with pytest.raises(RuntimeError, match="model failure"):
            _driver().run(batch, states, dense_step)
        assert get_forward_context().mixfusion_step is None
    assert calls == [None]


def test_bad_shape_fails_before_any_forward():
    states = _states()
    states[1].latents = torch.zeros(1, 16, 4)
    batch = InputBatch.make_batch(states, use_mixfusion=True)
    with override_forward_context(ForwardContext()):
        with pytest.raises(ValueError, match="request 1: actual latents disagree"):
            _driver().run(batch, states, lambda *a, **k: pytest.fail("must not run or fall back"))


def test_multi_image_admission_is_rejected():
    driver = _driver()
    states = _states()
    states[0].mixfusion_info = MixFusionRequestInfo(2, 8, 1, 128)
    states[0].latents = torch.zeros(2, 8, 4)
    batch = InputBatch.make_batch(states, use_mixfusion=True)
    with pytest.raises(ValueError, match="one image per request"):
        driver.validate(batch)


def test_armed_step_releases_its_packed_bundle_on_exit():
    ctx = ForwardContext()
    with override_forward_context(ctx):
        with _driver().armed():
            step = get_forward_context().mixfusion_step
            packed = PackedInputs(args=(), kwargs={"hidden_states": torch.zeros(1, 8, 4)}, plan=SimpleNamespace())
            step.packed, step.plan = packed, packed.plan
        assert get_forward_context().mixfusion_step is None
    assert step.packed is None and step.plan is None


def test_runtime_hook_change_is_an_error():
    driver = _driver()
    states = _states()
    batch = InputBatch.make_batch(states, use_mixfusion=True)
    driver.transformer.register_forward_pre_hook(lambda module, args: args)
    with pytest.raises(ValueError, match="foreign hooks"):
        driver.validate(batch)


def test_input_batch_cache_tracks_execution_mode():
    states = _states((8, 8))
    dense = InputBatch.make_batch(states)
    packed = InputBatch.make_batch(states, cached_batch=dense, use_mixfusion=True)
    assert isinstance(packed.latents, list)
    cached = InputBatch.make_batch(states, cached_batch=packed, use_mixfusion=True)
    assert cached is packed and cached.use_mixfusion
    dense_again = InputBatch.make_batch(states, cached_batch=cached)
    assert isinstance(dense_again.latents, torch.Tensor)


def test_remote_contract_failure_prevents_local_forward(monkeypatch):
    import vllm_omni.diffusion.mixfusion.driver as driver_module

    monkeypatch.setattr(driver_module, "_any_rank_failed", lambda local: True)
    states = _states()
    batch = InputBatch.make_batch(states, use_mixfusion=True)
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
