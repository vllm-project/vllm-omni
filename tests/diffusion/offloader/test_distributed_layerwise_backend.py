# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DistributedLayerwiseOffloadHook and backend utilities."""

import gc
import json
import os
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as dist_backend_module
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class DummyEvent:
    def record(self, _stream) -> None:
        return None


@contextmanager
def dummy_stream(_stream):
    yield None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()


@pytest.fixture(scope="module")
def dist_group():
    master_port = _find_free_port()
    _set_dist_env(rank=0, world_size=1, master_port=master_port)

    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        _cleanup_distributed()


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    monkeypatch.setattr(dist_backend_module.current_omni_platform, "Stream", DummyStream)
    monkeypatch.setattr(dist_backend_module.current_omni_platform, "Event", DummyEvent)
    monkeypatch.setattr(dist_backend_module.current_omni_platform, "current_stream", lambda: DummyStream())
    monkeypatch.setattr(dist_backend_module.current_omni_platform, "stream", dummy_stream)


class TinyBlock(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        mesh = DeviceMesh("cpu", [0])
        dtensor = DTensor.from_local(values, mesh, [Replicate()])
        self.weight = nn.Parameter(dtensor)


def _make_values(start: float) -> torch.Tensor:
    return torch.arange(start, start + 4, dtype=torch.float32)


class TestDistributedLayerwiseOffloadHook:
    def test_shard_and_pin_single_rank(self, dist_group, patched_offload_runtime):
        """With dp_size=1, the shard should equal the full weights."""
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        # With dp_size=1, the CPU shard should contain all 4 elements
        assert next_block.weight.dtype in hook.cpu_shards
        shard = hook.cpu_shards[next_block.weight.dtype]
        assert shard.numel() == 4
        assert torch.equal(shard, _make_values(10.0))

    def test_dtensor_wrapper_preserved_across_prefetch_and_offload(self, dist_group, patched_offload_runtime):
        """DTensor wrapper should be preserved through prefetch/offload cycle."""
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        # After init, next_block weights should be placeholders
        assert isinstance(next_block.weight, DTensor)
        assert next_block.weight.to_local().is_meta

        # Prefetch into slot 0
        hook.prefetch_layer(slot=0, non_blocking=False)

        # After prefetch, next_block weights should be materialized
        assert isinstance(next_block.weight, DTensor)
        assert torch.equal(next_block.weight.to_local(), _make_values(10.0))

        # Offload current block
        hook.offload_layer()
        assert isinstance(current_block.weight, DTensor)
        assert current_block.weight.to_local().is_meta
        assert not hook.is_materialized

    def test_double_buffer_slot_swapping(self, dist_group, patched_offload_runtime):
        """Verify slot swapping works correctly after each layer."""
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        assert hook.current_slot == 0

        # Prefetch into slot 1 (next slot)
        hook.prefetch_layer(slot=1, non_blocking=False)

        # Simulate post_forward: offload + swap
        hook.offload_layer()
        hook.current_slot = 1 - hook.current_slot

        assert hook.current_slot == 1

        # Next iteration: prefetch into slot 0
        hook.prefetch_layer(slot=0, non_blocking=False)
        hook.offload_layer()
        hook.current_slot = 1 - hook.current_slot

        assert hook.current_slot == 0

    def test_device_buffers_allocated(self, dist_group, patched_offload_runtime):
        """Verify exactly two device buffers are allocated."""
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        # Two slots should be allocated
        assert hook.gpu_buffers[0] is not None
        assert hook.gpu_buffers[1] is not None

        # Each slot should have one dtype entry (float32)
        assert len(hook.gpu_buffers[0]) == 1
        assert len(hook.gpu_buffers[1]) == 1

        # Buffer size should match total numel
        dtype = next_block.weight.dtype
        assert hook.gpu_buffers[0][dtype].numel() == 4
        assert hook.gpu_buffers[1][dtype].numel() == 4

    def test_sharding_multiple_ranks(self, dist_group, patched_offload_runtime):
        """Verify weight sharding splits correctly across ranks."""
        block = nn.Module()
        block.weight = nn.Parameter(torch.arange(8, dtype=torch.float32))

        next_block = nn.Module()
        next_block.weight = nn.Parameter(torch.arange(100, 108, dtype=torch.float32))

        # Simulate dp_size=4, rank=1
        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=4,
            rank=1,
            pin_memory=False,
        )

        hook.initialize_hook(block)

        shard = hook.cpu_shards[torch.float32]
        # 8 elements / 4 ranks = 2 elements per shard
        assert shard.numel() == 2
        # Rank 1 should have elements [102, 103]
        assert torch.equal(shard, torch.tensor([102.0, 103.0]))

    def test_sharding_with_remainder(self, dist_group, patched_offload_runtime):
        """Verify sharding handles non-even division with padding."""
        block = nn.Module()
        block.weight = nn.Parameter(torch.arange(3, dtype=torch.float32))

        next_block = nn.Module()
        next_block.weight = nn.Parameter(torch.arange(100, 103, dtype=torch.float32))

        # 3 elements, dp_size=2: both ranks get ceil(3/2)=2 elements (padded)
        hook0 = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=2,
            rank=0,
            pin_memory=False,
        )
        # Need fresh next_block for each hook since init replaces tensors
        next_block0 = nn.Module()
        next_block0.weight = nn.Parameter(torch.arange(100, 103, dtype=torch.float32))
        hook0.next_block = next_block0
        hook0.initialize_hook(block)

        shard0 = hook0.cpu_shards[torch.float32]
        assert shard0.numel() == 2
        assert torch.equal(shard0, torch.tensor([100.0, 101.0]))

        next_block1 = nn.Module()
        next_block1.weight = nn.Parameter(torch.arange(100, 103, dtype=torch.float32))
        block1 = nn.Module()
        block1.weight = nn.Parameter(torch.arange(3, dtype=torch.float32))
        hook1 = DistributedLayerwiseOffloadHook(
            next_block=next_block1,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=2,
            rank=1,
            pin_memory=False,
        )
        hook1.initialize_hook(block1)

        shard1 = hook1.cpu_shards[torch.float32]
        # Equal-sized shards: rank 1 gets [102, 0] (zero-padded)
        assert shard1.numel() == 2
        assert torch.equal(shard1, torch.tensor([102.0, 0.0]))


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))


class _SingleBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _MultiBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    def __init__(self, num_transformer: int = 2, num_single: int = 2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_transformer)])
        self.single_transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_single)])


class _EmptyBlocksModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([])


class _NoAttrsModel(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _MmapPostLoadModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.time_embedder = nn.Linear(2, 2, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
        self.post_load_calls = 0

    def post_load_weights(self) -> None:
        self.time_embedder.to(torch.float32)
        self.post_load_calls += 1


class _MmapPostLoadPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.device("meta"):
            self.transformer = _MmapPostLoadModel()

    @staticmethod
    def _remap_ckpt_key(key: str) -> str:
        return key


class TestMmapWeightLoading:
    def test_runs_model_post_load_hook(self, tmp_path, patched_offload_runtime):
        pipeline = _MmapPostLoadPipeline()
        weights = {name: torch.ones(param.shape, dtype=torch.bfloat16) for name, param in pipeline.named_parameters()}
        weight_file = tmp_path / "model.safetensors"
        save_file(weights, str(weight_file))
        weight_map = {name: weight_file.name for name in weights}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

        backend = DistributedLayerwiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=False,
                model_path=str(tmp_path),
            ),
            torch.device("cpu"),
        )
        modules = SimpleNamespace(
            dits=[pipeline.transformer],
            dit_names=["transformer"],
        )

        backend._load_weights_via_mmap(pipeline, modules)

        assert pipeline.transformer.post_load_calls == 1
        assert pipeline.transformer.time_embedder.weight.dtype == torch.float32
        assert pipeline.transformer.blocks[0].weight.dtype == torch.bfloat16


class TestGetBlocksFromDit:
    def test_get_blocks_from_dit_single_block_attr(self):
        model = _SingleBlockModel(num_blocks=3)
        attr_names, blocks = DistributedLayerwiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 3
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_multi_block_attrs(self):
        model = _MultiBlockModel(num_transformer=2, num_single=3)
        attr_names, blocks = DistributedLayerwiseOffloadBackend.get_blocks_from_dit(model)
        assert set(attr_names) == {"transformer_blocks", "single_transformer_blocks"}
        assert len(blocks) == 5

    def test_get_blocks_from_dit_empty_blocks(self):
        model = _EmptyBlocksModel()
        attr_names, blocks = DistributedLayerwiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_no_attrs_defined(self):
        model = _NoAttrsModel(num_blocks=3)
        attr_names, blocks = DistributedLayerwiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []


class TestGetBlocksAttrNames:
    def test_get_blocks_attr_names_new_format(self):
        model = _MultiBlockModel()
        attrs = DistributedLayerwiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == ["transformer_blocks", "single_transformer_blocks"]

    def test_get_blocks_attr_names_no_attrs(self):
        model = _NoAttrsModel()
        attrs = DistributedLayerwiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == []

    def test_set_blocks_attr_names(self):
        model = _NoAttrsModel()
        DistributedLayerwiseOffloadBackend.set_blocks_attr_names(model, ["new_blocks"])
        assert hasattr(model.__class__, "_layerwise_offload_blocks_attrs")
        assert model.__class__._layerwise_offload_blocks_attrs == ["new_blocks"]


class TestCrossGroupSharedBuffer:
    """Regression test: when multiple DiT groups share the same 2 GPU
    buffers, the first block of each group must always sync-prefetch on
    entry, because another group may have overwritten the shared slot.
    """

    def test_group_first_hook_forces_sync_prefetch(self, dist_group, patched_offload_runtime):
        """Verify _is_group_first=True forces sync-prefetch in pre_forward
        even when is_materialized returns True."""
        block_a = TinyBlock(_make_values(1.0))
        block_b = TinyBlock(_make_values(10.0))

        hook_a = DistributedLayerwiseOffloadHook(
            next_block=block_b,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            pin_memory=False,
        )
        hook_a.initialize_hook(block_a)

        # Prefetch to make block_a "materialized" (non-empty params)
        hook_a.prefetch_layer(hook_a.current_slot, non_blocking=False)
        assert hook_a.is_materialized

        # Track sync-prefetch calls
        sync_called = [False]
        orig = hook_a.prefetch_layer

        def tracking(slot, non_blocking=True):
            if not non_blocking:
                sync_called[0] = True
            orig(slot, non_blocking=non_blocking)

        hook_a.prefetch_layer = tracking

        # Case 1: _is_group_first=False, materialized → no sync-prefetch
        hook_a._is_group_first = False
        sync_called[0] = False
        hook_a.pre_forward(block_a)
        assert not sync_called[0], "Materialized block should not sync-prefetch without _is_group_first"

        # Case 2: _is_group_first=True, materialized → MUST sync-prefetch
        hook_a._is_group_first = True
        hook_a._prev_hook = hook_a
        sync_called[0] = False
        hook_a.pre_forward(block_a)
        assert sync_called[0], "_is_group_first must force sync-prefetch even when materialized"
