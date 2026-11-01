# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for phase-aware DLO; no accelerator kernels required."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as backend_module
from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class PrefixMethod:
    def get_layerwise_prefix_weights(self, layer):
        return 5, {"master": layer.master}, ("weight", "weight_scale")


class PrefixBlock(nn.Module):
    def __init__(self, offset=1):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 32 + offset, False)
        self.register_buffer("weight_scale", torch.ones(4))
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.bfloat16), False)
        self.master = self.weight.detach().to(torch.bfloat16).clone()
        self.quant_method = PrefixMethod()

    def forward(self, x):
        if backend_module._in_prefix(5):
            assert self.weight.numel() == self.weight_scale.numel() == 0
            return x @ self.master + self.bias
        assert self.master.numel() == 0
        return (x.float() @ self.weight + self.bias).to(x.dtype)


@pytest.fixture
def runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, backend_module.current_omni_platform, synchronize=True)
    phase = SimpleNamespace(step=0)
    monkeypatch.setattr(
        backend_module, "_in_prefix", lambda k: bool(k and phase.step is not None and 0 <= phase.step < k)
    )
    return phase


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_ring_phase_boundary_cfg_and_request_reset(runtime, count):
    blocks = nn.ModuleList([PrefixBlock(i + 1) for i in range(count)])
    expected = [(b.weight.detach().clone(), b.master.clone(), b.bias.detach().clone()) for b in blocks]
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE, pin_cpu_memory=False, dlo_use_allgather=False),
        torch.device("cpu"),
    )
    hooks = backend._install_hook_group(blocks, "dit")
    buffers = backend._allocate_shared_buffers(hooks)
    tracker = [-1, -1]
    for hook in hooks:
        hook.gpu_buffers = buffers
        hook._shared_slot_group = tracker
        hook._group_id = 0
    assert all(b.master.numel() == 0 for b in blocks)
    for step in [0, 0, 4, 4, 5, 5, 49, None, 0, 5]:
        runtime.step = step
        x = torch.ones(2, 4, dtype=torch.bfloat16) / 8
        reference = x.clone()
        for block, (weight, master, bias) in zip(blocks, expected):
            reference = (
                (reference @ master + bias)
                if step is not None and step < 5
                else (reference.float() @ weight + bias).to(x.dtype)
            )
            x = block(x)
        torch.testing.assert_close(x, reference, rtol=0, atol=0)
    for hook in hooks:
        hook.restore_next_block_to_cpu()
    for block, (weight, master, bias) in zip(blocks, expected):
        torch.testing.assert_close(block.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(block.master, master, rtol=0, atol=0)
        torch.testing.assert_close(block.bias, bias, rtol=0, atol=0)
        assert not block._layerwise_prefix_managed


def test_allgather_prefix_and_quantized_payloads(runtime, monkeypatch):
    blocks = [PrefixBlock(), PrefixBlock()]
    hooks = []
    for rank, block in enumerate(blocks):
        hook = DistributedLayerwiseOffloadHook(nn.Module(), torch.device("cpu"), object(), 2, rank, pin_memory=False)
        hook.next_block = block
        hook.initialize_hook(nn.Module())
        hooks.append(hook)
    shared = DistributedLayerwiseOffloadBackend._allocate_shared_buffers(hooks)
    shards = DistributedLayerwiseOffloadBackend._allocate_shared_shard_buffers(hooks)
    for hook in hooks:
        hook.gpu_buffers, hook.gpu_shard_buffers = shared, shards

    assert all(hook.manifest is not None for hook in hooks)
    assert all(hook._prefix_payload["manifest"] is not None for hook in hooks)

    def gather(out, local, group):
        payload = hooks[1]._prefix_payload["cpu_shards"] if runtime.step < 5 else hooks[1].cpu_shards
        out.copy_(torch.cat([local, payload[local.dtype]]))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    for step in [4, 5, 0]:
        runtime.step = step
        hooks[0].prefetch_layer(0)
        assert hooks[0]._prefetched_prefix == (step < 5)
        template = PrefixBlock()
        if step < 5:
            torch.testing.assert_close(blocks[0].master, template.master, rtol=0, atol=0)
        else:
            torch.testing.assert_close(blocks[0].weight, template.weight, rtol=0, atol=0)
        backend_module.clear_tensor_storage(
            [blocks[0].master, blocks[0].weight, blocks[0].weight_scale, blocks[0].bias]
        )

    def restore_gather(out, local, group):
        candidate = hooks[1]._prefix_payload["cpu_shards"]
        if local.dtype not in candidate or local.numel() != candidate[local.dtype].numel():
            candidate = hooks[1].cpu_shards
        out.copy_(torch.cat([local, candidate[local.dtype]]))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", restore_gather)
    hooks[0].restore_next_block_to_cpu()
    template = PrefixBlock()
    torch.testing.assert_close(blocks[0].master, template.master, rtol=0, atol=0)
    torch.testing.assert_close(blocks[0].weight, template.weight, rtol=0, atol=0)
    assert not blocks[0]._layerwise_prefix_managed


def test_shared_groups_cannot_reuse_stale_prefix(runtime):
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE, pin_cpu_memory=False, dlo_use_allgather=False),
        torch.device("cpu"),
    )
    groups = [nn.ModuleList([PrefixBlock(i + 1) for i in range(n)]) for n in (2, 3)]
    hooks = [backend._install_hook_group(blocks, "dit") for blocks in groups]
    buffers = backend._allocate_shared_buffers([h for ring in hooks for h in ring])
    tracker = [-1, -1]
    for group_id, ring in enumerate(hooks):
        for h in ring:
            h.gpu_buffers, h._shared_slot_group, h._group_id = buffers, tracker, group_id
    for step in [0, 4, 5, 0]:
        runtime.step = step
        for blocks in groups:
            x = reference = torch.ones(1, 4, dtype=torch.bfloat16) / 8
            for i, block in enumerate(blocks):
                original = PrefixBlock(i + 1)
                reference = (
                    reference @ original.master + original.bias
                    if step < 5
                    else (reference.float() @ original.weight + original.bias).to(x.dtype)
                )
                x = block(x)
            torch.testing.assert_close(x, reference, rtol=0, atol=0)


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_prefix_pinned_staging_is_shared_and_phase_safe(runtime, monkeypatch, count):
    empty, zeros = torch.empty, torch.zeros
    pinned_sizes = []

    def fake_empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        result = empty(*args, **kwargs)
        if pinned:
            pinned_sizes.append(result.numel() * result.element_size())
        return result

    def fake_zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch, "zeros", fake_zeros)
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE, pin_cpu_memory=True, dlo_use_allgather=False),
        torch.device("cpu"),
    )
    groups = [nn.ModuleList([PrefixBlock(i + 1) for i in range(n)]) for n in (count, 3)]
    rings = [backend._install_hook_group(blocks, "dit") for blocks in groups]
    hooks = [h for ring in rings for h in ring]
    DistributedLayerwiseOffloadHook.configure_prefix_staging(hooks)
    assert sum(pinned_sizes) == 2 * (16 + 4) * 2
    assert all(h._prefix_staging_buffers is hooks[0]._prefix_staging_buffers for h in hooks)
    assert all(h._prefix_staging_events is hooks[0]._prefix_staging_events for h in hooks)
    assert all(not t.is_pinned() for h in hooks for t in h._prefix_payload["cpu_shards"].values())
    waited = []
    hooks[0]._prefix_staging_events[0] = SimpleNamespace(synchronize=lambda: waited.append(True))
    hooks[0]._stage_prefix_shards(0)
    assert waited == [True]
    buffers = backend._allocate_shared_buffers(hooks)
    tracker = [-1, -1]
    for group_id, ring in enumerate(rings):
        for h in ring:
            h.gpu_buffers, h._shared_slot_group, h._group_id = buffers, tracker, group_id
    for step in (0, 4, 4, 5, 49, 0, 5):
        runtime.step = step
        for blocks in groups:
            x = reference = torch.ones(2, 4, dtype=torch.bfloat16) / 8
            for i, block in enumerate(blocks):
                original = PrefixBlock(i + 1)
                reference = (
                    reference @ original.master + original.bias
                    if step < 5
                    else (reference.float() @ original.weight + original.bias).to(x.dtype)
                )
                x = block(x)
            torch.testing.assert_close(x, reference, rtol=0, atol=0)
