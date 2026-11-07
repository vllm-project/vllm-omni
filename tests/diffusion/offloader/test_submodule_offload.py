# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise submodule weight ownership without Cache-DiT or reference KV."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.block_discovery import get_blocks_from_dit
from vllm_omni.diffusion.offloader.config import OffloadStrategy, offload_streams_blocks, resolve_offload
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import current_omni_platform
from vllm_omni.diffusion.offloader.submodule_offload_backend import SubmoduleOffloadBackend

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class TinyBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.adaln_proj = nn.Linear(4, 4)
        self.attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 4))
        self.norm = nn.LayerNorm(4)
        with torch.no_grad():
            for param in self.parameters():
                param.add_(index * 0.05)

    def forward(self, x):
        return x + self.mlp(self.attn(self.adaln_proj(self.norm(x))))


class TinyDiT(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, count):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(index) for index in range(count)])

    def forward(self, x, indices=None):
        for index in range(len(self.blocks)) if indices is None else indices:
            x = self.blocks[index](x)
        return x


@pytest.fixture
def runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)


@pytest.mark.parametrize("count", [3, 4])
def test_repeated_forward_skipped_blocks_and_restore(runtime, count):
    torch.manual_seed(17)
    pipeline = nn.Module()
    pipeline.transformer = TinyDiT(count)
    baseline = deepcopy(pipeline.transformer)
    backend = SubmoduleOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.SUBMODULE,
            pin_cpu_memory=False,
            dlo_use_allgather=False,
            components=frozenset({"dit"}),
        ),
        torch.device("cpu"),
    )
    backend.enable(pipeline)
    hooks = backend._all_hook_groups[0]
    assert len(hooks) == count * 3
    assert all(hook.gpu_buffers is hooks[0].gpu_buffers for hook in hooks)
    assert len(hooks[0].gpu_buffers) == 2
    largest = max(
        sum(t.numel() for t in unit.parameters())
        for block in baseline.blocks
        for unit in (block.adaln_proj, block.attn, block.mlp)
    )
    assert hooks[0].gpu_buffers[0][torch.float32].numel() == largest
    assert all(block.norm.weight.numel() == 4 for block in pipeline.transformer.blocks)

    x = torch.randn(2, 4)
    with torch.no_grad():
        # A forward may execute only a subset of the physical block stack.
        for indices in (None, [0, count - 1], None, [1, count - 1], None):
            torch.testing.assert_close(pipeline.transformer(x, indices), baseline(x, indices))
    backend.disable()
    for name, param in pipeline.transformer.named_parameters():
        torch.testing.assert_close(param, dict(baseline.named_parameters())[name])


def test_discover_plain_blocks():
    dit = TinyDiT(4)
    attrs, blocks = get_blocks_from_dit(dit)
    assert attrs == ["blocks"]
    assert blocks == list(dit.blocks)


def test_submodule_config_and_factory(runtime, monkeypatch):
    od_config = SimpleNamespace(enable_submodule_offload=True, dlo_chunk_size_mb=64)
    resolved = resolve_offload(od_config)
    assert resolved.strategy is OffloadStrategy.SUBMODULE
    assert resolved.any_allgather
    assert offload_streams_blocks(od_config)
    assert OffloadConfig.from_od_config(od_config).strategy is OffloadStrategy.SUBMODULE
    monkeypatch.setattr(current_omni_platform, "supports_cpu_offload", lambda: True)
    monkeypatch.setattr(current_omni_platform, "get_device_count", lambda: 1)
    backend = get_offload_backend(od_config, torch.device("cpu"))
    assert isinstance(backend, SubmoduleOffloadBackend)


def test_incompatible_auxiliary_stack_keeps_block_hooks(runtime):
    backend = SubmoduleOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.SUBMODULE, pin_cpu_memory=False, dlo_use_allgather=False),
        torch.device("cpu"),
    )
    blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    hooks = backend._install_hook_group(blocks, "dit")
    assert len(hooks) == 2
    backend.disable()
