# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Executable contracts for the two behaviors the generic cutover changed.

Placement now follows resolved block identity, and the backends run on the
resolved plan alone. Declaration parsing, selection errors, rollback,
residency and enable/disable cycles are covered by ``test_plan_resolver.py``,
``test_layerwise_backend.py`` and ``test_sequential_backend.py``.
"""

from typing import ClassVar

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader import layerwise_backend, sequential_backend
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

STRATEGIES = [OffloadStrategy.MODEL_LEVEL, OffloadStrategy.LAYER_WISE]
CPU = torch.device("cpu")


class _Block(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.register_buffer("scale", torch.tensor(0.5))

    def forward(self, x):
        return super().forward(x).tanh() * self.scale


class _Stack(nn.Module):
    """Two block containers plus non-block state that must stay resident."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.tail = nn.ModuleList([_Block() for _ in range(2)])
        self.bias = nn.Parameter(torch.ones(4))
        self.register_buffer("scale", torch.tensor(0.25))
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        for block in (*self.blocks, *self.tail):
            x = block(x)
        return self.proj(x) * self.scale + self.bias


class _Pipeline(nn.Module):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["resident"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks", "tail")},
        encoder_block_attrs={"text_encoder": ("blocks", "tail")},
    )

    def __init__(self):
        super().__init__()
        self.transformer = _Stack()
        self.text_encoder = _Stack()
        self.image_encoder = nn.Linear(4, 4)
        self.vae = nn.Linear(4, 4)
        self.resident = nn.Linear(4, 4)

    def forward(self, x):
        return self.resident(self.vae(self.transformer(self.text_encoder(self.image_encoder(x)))))


@pytest.fixture(params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)])
def execution_device(request, monkeypatch):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for real transfer/residency coverage")
        return torch.device("cuda:0")
    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)
    monkeypatch.setattr(current_omni_platform, "get_free_memory", lambda: 0)
    return CPU


def _config(strategy, components):
    return OffloadConfig(strategy=strategy, components=components, pin_cpu_memory=False)


def _backend(config, device):
    kind = ModelLevelOffloadBackend if config.strategy is OffloadStrategy.MODEL_LEVEL else LayerWiseOffloadBackend
    return kind(config, device)


def _assert_no_offload_hooks(pipeline):
    for module in pipeline.modules():
        registry = getattr(module, "_hook_registry", None)
        if registry is not None:
            assert registry.get_hook("sequential_offload") is None
            assert registry.get_hook("layerwise_offload") is None
        assert not getattr(module, "_omni_layerwise_enabled", False)


@torch.inference_mode()
def test_alias_of_a_streamed_block_is_not_placed(execution_device):
    """Placement moves DiT state by resolved block identity, not by attribute.

    An attribute aliasing a streamed block must stay with its ring: copying it
    to the device makes the ring record the device as that block's home and
    strands those weights there after teardown.
    """
    pipeline = _Pipeline()
    pipeline.transformer.proj.weight = pipeline.transformer.blocks[1].weight
    original = pipeline.transformer.proj.weight.clone()
    backend = _backend(_config(OffloadStrategy.LAYER_WISE, frozenset({"dit"})), execution_device)

    backend.enable(pipeline)
    try:
        assert pipeline.transformer.proj.weight is pipeline.transformer.blocks[1].weight
        assert pipeline.transformer.proj.weight.device == CPU
        # Non-block state around the alias is still placed.
        assert pipeline.transformer.proj.bias.device == execution_device
        assert pipeline.transformer.bias.device == execution_device
    finally:
        backend.disable()

    assert pipeline.transformer.proj.weight.device == CPU
    torch.testing.assert_close(pipeline.transformer.proj.weight, original)


@pytest.mark.parametrize("strategy", STRATEGIES)
@torch.inference_mode()
def test_backends_execute_the_resolved_plan_alone(execution_device, strategy, monkeypatch):
    pipeline = _Pipeline().to(execution_device)
    x = torch.randn(2, 4, device=execution_device)
    expected = pipeline(x)
    config = _config(strategy, frozenset({"dit", "text_encoder"}))
    resolved = resolve_offload_plan(pipeline, config)

    def forbidden_read(*args, **kwargs):
        pytest.fail("Backend reinterpreted topology after plan resolution")

    # Once resolved, neither declarations nor selector helpers are an input to
    # backend execution. Transport options stay readable.
    monkeypatch.setattr(config, "offloads", forbidden_read)
    monkeypatch.setattr(config, "should_offload_encoder", forbidden_read)
    monkeypatch.setattr(pipeline, "_offload_plan", None)
    backend_module = sequential_backend if strategy is OffloadStrategy.MODEL_LEVEL else layerwise_backend
    monkeypatch.setattr(backend_module, "resolve_offload_plan", lambda *_: resolved)

    backend = _backend(config, execution_device)
    try:
        backend.enable(pipeline)
        torch.testing.assert_close(pipeline(x), expected)
    finally:
        backend.disable()
    _assert_no_offload_hooks(pipeline)
