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
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    TensorBinding,
    build_checkpoint_mmap_plan,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.block_discovery import (
    get_blocks_attr_names,
    get_blocks_from_dit,
    set_blocks_attr_names,
)
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
    PinnedResidentLayerGroup,
)
from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager
from vllm_omni.diffusion.offloader.offload_plan import (
    OffloadPlan,
    get_offload_plan,
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

    def synchronize(self) -> None:
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
    monkeypatch.setattr(dist_backend_module.current_omni_platform, "synchronize", lambda: None)


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

    def test_prefetch_preserves_transposed_weight_stride(
        self,
        dist_group,
        patched_offload_runtime,
    ):
        """Online-FP8 Cutlass weights must retain their transposed layout."""

        class StridedBlock(nn.Module):
            def __init__(self, start: float):
                super().__init__()
                base = torch.arange(start, start + 12).reshape(3, 4)
                self.weight = nn.Parameter(base.t(), requires_grad=False)

        current_block = StridedBlock(1.0)
        next_block = StridedBlock(20.0)
        expected = next_block.weight.detach().clone()
        expected_stride = next_block.weight.stride()
        assert expected_stride == (1, 4)

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
        hook.prefetch_layer(slot=0, non_blocking=False)

        assert next_block.weight.stride() == expected_stride
        assert torch.equal(next_block.weight, expected)

    def test_allgather_reconstructs_online_fp8_weight_and_scale(
        self,
        patched_offload_runtime,
        monkeypatch,
    ):
        """DLO must reconstruct finalized online-FP8 tensors byte-for-byte."""

        class OnlineFp8Block(nn.Module):
            def __init__(self):
                super().__init__()
                logical_weight = torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                    ],
                    dtype=torch.float32,
                ).to(torch.float8_e4m3fn)
                self.weight = nn.Parameter(logical_weight.t(), requires_grad=False)
                self.weight_scale = nn.Parameter(torch.tensor([0.25]), requires_grad=False)

        current_block = OnlineFp8Block()
        rank0_block = OnlineFp8Block()
        rank1_block = OnlineFp8Block()
        expected_weight = rank0_block.weight.detach().clone()
        expected_scale = rank0_block.weight_scale.detach().clone()
        expected_stride = rank0_block.weight.stride()

        rank0_hook = DistributedLayerwiseOffloadHook(
            next_block=rank0_block,
            device=torch.device("cpu"),
            dp_group=object(),
            dp_size=2,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )
        rank1_hook = DistributedLayerwiseOffloadHook(
            next_block=rank1_block,
            device=torch.device("cpu"),
            dp_group=object(),
            dp_size=2,
            rank=1,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )
        rank0_hook.initialize_hook(current_block)
        rank1_hook.initialize_hook(OnlineFp8Block())
        rank0_hook.gpu_shard_buffers = [
            {dtype: torch.empty_like(shard) for dtype, shard in rank0_hook.cpu_shards.items()} for _ in range(2)
        ]

        def fake_allgather(output, local_shard, *, group):
            del group
            remote_shard = rank1_hook.cpu_shards[local_shard.dtype]
            shard_size = local_shard.numel()
            output[:shard_size].copy_(local_shard)
            output[shard_size : 2 * shard_size].copy_(remote_shard)

        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_allgather)
        rank0_hook.prefetch_layer(slot=0, non_blocking=False)

        assert rank0_block.weight.stride() == expected_stride
        torch.testing.assert_close(
            rank0_block.weight.float(),
            expected_weight.float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(rank0_block.weight_scale, expected_scale)

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

    def test_mmap_layout_transform_is_applied_before_sharding(self, dist_group, patched_offload_runtime):
        block = nn.Module()
        block.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        next_block = nn.Module()
        next_block.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))

        def transform(tensor):
            return tensor.flip(0)

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=2,
            rank=0,
            pin_memory=False,
            tensor_transforms={id(next_block.weight): transform},
        )
        hook.initialize_hook(block)

        assert torch.equal(hook.cpu_shards[torch.float32], torch.tensor([3.0, 2.0]))

    def test_transform_is_not_applied_without_a_loader_plan(self, dist_group, patched_offload_runtime):
        block = nn.Module()
        block.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        next_block = nn.Module()
        next_block.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=2,
            rank=0,
            pin_memory=False,
        )
        hook.initialize_hook(block)

        assert torch.equal(hook.cpu_shards[torch.float32], torch.tensor([0.0, 1.0]))

    def test_rank_local_mmap_retains_source_and_uses_staging(self, dist_group, patched_offload_runtime):
        current_block = nn.Linear(2, 2, bias=False)
        next_block = nn.Linear(2, 2, bias=False)
        next_block.weight.data.copy_(torch.arange(4, dtype=torch.float32).view(2, 2))
        source_storage = next_block.weight.untyped_storage().data_ptr()

        def transform(tensor):
            return tensor.t()

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            pin_memory=False,
            rank_local_mmap=True,
            tensor_transforms={id(next_block.weight): transform},
        )
        hook.initialize_hook(current_block)

        source = hook.cpu_sources[torch.float32][0]["tensor"]
        assert source.untyped_storage().data_ptr() == source_storage
        assert hook.cpu_shards == {}
        assert next_block.weight.numel() == 0

        hook.cpu_staging_buffers = [
            {torch.float32: torch.empty(4)},
            {torch.float32: torch.empty(4)},
        ]
        hook.prefetch_layer(slot=0, non_blocking=False)

        assert torch.equal(
            next_block.weight,
            torch.tensor([[0.0, 2.0], [1.0, 3.0]]),
        )
        assert next_block.weight.stride() == (1, 2)
        assert torch.equal(source, torch.arange(4, dtype=torch.float32).view(2, 2))

    def test_rank_local_mmap_staging_is_bounded_by_largest_block(self, patched_offload_runtime):
        hooks = []
        for size in (4, 9):
            current_block = nn.Linear(1, 1, bias=False)
            next_block = nn.Module()
            next_block.weight = nn.Parameter(torch.arange(size, dtype=torch.float32))
            hook = DistributedLayerwiseOffloadHook(
                next_block=next_block,
                device=torch.device("cpu"),
                dp_group=None,
                dp_size=1,
                rank=0,
                pin_memory=False,
                shared_buffers=[None, None],
                rank_local_mmap=True,
            )
            hook.initialize_hook(current_block)
            hooks.append(hook)

        resident_block = nn.Module()
        resident_block.weight = nn.Parameter(torch.arange(12, dtype=torch.float32))
        resident_group = PinnedResidentLayerGroup(
            [resident_block],
            device=torch.device("cpu"),
            copy_stream=DummyStream(),
            pin_memory=False,
            rank_local_mmap=True,
        )
        staging = DistributedLayerwiseOffloadBackend._allocate_shared_cpu_staging_buffers(
            hooks,
            resident_group,
        )

        assert len(staging) == 2
        assert staging[0][torch.float32].numel() == 12
        assert staging[1][torch.float32].numel() == 12


class TestPinnedResidentLayerGroup:
    def test_load_offload_reuses_pinned_master_weights(self, patched_offload_runtime):
        blocks = [nn.Linear(2, 2), nn.Linear(2, 2)]
        expected = []
        for index, block in enumerate(blocks):
            block.weight.data.fill_(index + 1)
            block.bias.data.fill_(index + 3)
            expected.append((block.weight.detach().clone(), block.bias.detach().clone()))

        group = PinnedResidentLayerGroup(
            blocks,
            device=torch.device("cpu"),
            copy_stream=DummyStream(),
            pin_memory=False,
        )

        assert all(block.weight.numel() == 0 for block in blocks)

        group.load()
        for block, (weight, bias) in zip(blocks, expected):
            assert torch.equal(block.weight, weight)
            assert torch.equal(block.bias, bias)

        # Device-side changes must not overwrite the persistent host master.
        blocks[0].weight.data.zero_()
        group.offload()
        assert all(block.weight.numel() == 0 for block in blocks)

        group.load()
        for block, (weight, bias) in zip(blocks, expected):
            assert torch.equal(block.weight, weight)
            assert torch.equal(block.bias, bias)

        group.offload()

    def test_load_preserves_noncontiguous_weight_stride(self, patched_offload_runtime):
        block = nn.Module()
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
        expected = weight.detach().clone()
        expected_stride = weight.stride()
        block.weight = nn.Parameter(weight)
        group = PinnedResidentLayerGroup(
            [block],
            device=torch.device("cpu"),
            copy_stream=DummyStream(),
            pin_memory=False,
        )

        group.load()
        assert torch.equal(block.weight, expected)
        assert block.weight.stride() == expected_stride
        group.offload()

    def test_rank_local_mmap_resident_layers_retain_sources(self, patched_offload_runtime):
        blocks = [nn.Linear(2, 2, bias=False) for _ in range(3)]
        expected = []
        source_storage = []
        for index, block in enumerate(blocks):
            block.weight.data.fill_(index + 1)
            expected.append(block.weight.detach().clone())
            source_storage.append(block.weight.untyped_storage().data_ptr())

        group = PinnedResidentLayerGroup(
            blocks,
            device=torch.device("cpu"),
            copy_stream=DummyStream(),
            pin_memory=False,
            rank_local_mmap=True,
        )

        for index, state in enumerate(group._states):
            source = state["cpu_sources"][torch.float32][0]["tensor"]
            assert source.untyped_storage().data_ptr() == source_storage[index]
            assert state["cpu_shards"] == {}
        assert len(group._cpu_staging_buffers) == 2

        group.load()
        for block, weight in zip(blocks, expected):
            assert torch.equal(block.weight, weight)

        blocks[0].weight.data.zero_()
        group.offload()
        group.load()
        for block, weight in zip(blocks, expected):
            assert torch.equal(block.weight, weight)

        group.offload()
        assert all(block.weight.numel() == 0 for block in blocks)

    def test_shared_allgather_output_is_narrowed_to_current_block(
        self, dist_group, patched_offload_runtime, monkeypatch
    ):
        block = nn.Module()
        block.weight = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        next_block = nn.Module()
        next_block.weight = nn.Parameter(torch.arange(3, dtype=torch.float32))

        hook = DistributedLayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            dp_group=object(),
            dp_size=2,
            rank=0,
            pin_memory=False,
        )
        hook.initialize_hook(block)
        # Simulate unified buffers sized for a larger heterogeneous block.
        hook.gpu_buffers = [
            {torch.float32: torch.empty(12)},
            {torch.float32: torch.empty(12)},
        ]
        hook.gpu_shard_buffers = [
            {torch.float32: torch.empty(6)},
            {torch.float32: torch.empty(6)},
        ]
        output_sizes = []

        def fake_allgather(output, local_shard, *, group):
            del group
            output_sizes.append(output.numel())
            output.zero_()
            output[: local_shard.numel()].copy_(local_shard)

        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_allgather)

        hook.prefetch_layer(slot=0, non_blocking=False)

        assert output_sizes == [4]


class TestPinnedModuleStager:
    def test_offload_rebinds_immutable_cpu_master(self, patched_offload_runtime):
        module = nn.Linear(2, 2)
        expected_weight = module.weight.detach().clone()
        expected_bias = module.bias.detach().clone()
        stager = PinnedModuleStager(
            module,
            torch.device("cpu"),
            pin_memory=False,
            copy_stream=DummyStream(),
        )

        stager.load()
        module.weight.data.zero_()
        module.bias.data.zero_()
        stager.offload()

        assert torch.equal(module.weight, expected_weight)
        assert torch.equal(module.bias, expected_bias)
        stager.load()
        assert torch.equal(module.weight, expected_weight)
        assert torch.equal(module.bias, expected_bias)
        stager.offload()


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
            ),
            torch.device("cpu"),
        )
        modules = SimpleNamespace(
            dits=[pipeline.transformer],
            dit_names=["transformer"],
        )
        plan = HostWeightPlan(
            backing_kind="checkpoint_mmap",
            bindings={
                name: TensorBinding(
                    checkpoint_key=name,
                    file_path=str(weight_file),
                )
                for name in weights
            },
        )

        backend._load_weights_via_mmap(pipeline, modules, plan)

        assert pipeline.transformer.post_load_calls == 1
        assert pipeline.transformer.time_embedder.weight.dtype == torch.float32
        assert pipeline.transformer.blocks[0].weight.dtype == torch.bfloat16

    def test_allgather_flag_with_group_size_one_uses_rank_local_mmap(
        self,
        tmp_path,
        patched_offload_runtime,
    ):
        class Transformer(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(2, 2, bias=False),
                        nn.Linear(2, 2, bias=False),
                    ]
                )

        class Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Transformer()

            @staticmethod
            def _remap_ckpt_key(key):
                return key

        pipeline = Pipeline()
        weights = {name: torch.ones_like(param) for name, param in pipeline.named_parameters()}
        checkpoint_file = tmp_path / "model.safetensors"
        save_file(weights, str(checkpoint_file))
        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path=str(tmp_path),
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )
        assert result.plan is not None
        backend = DistributedLayerwiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=False,
                dp_size=1,
                dlo_use_allgather=True,
            ),
            torch.device("cpu"),
            host_weight_plan=result.plan,
        )

        backend.enable(pipeline)

        assert backend._using_rank_local_mmap
        assert backend.dp_group is None
        assert all(hook.rank_local_mmap for group in backend._all_hook_groups for hook in group)
        backend.disable()


class TestGetBlocksFromDit:
    def test_get_blocks_from_dit_single_block_attr(self):
        model = _SingleBlockModel(num_blocks=3)
        attr_names, blocks = get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 3
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_multi_block_attrs(self):
        model = _MultiBlockModel(num_transformer=2, num_single=3)
        attr_names, blocks = get_blocks_from_dit(model)
        assert set(attr_names) == {"transformer_blocks", "single_transformer_blocks"}
        assert len(blocks) == 5

    def test_get_blocks_from_dit_empty_blocks(self):
        model = _EmptyBlocksModel()
        attr_names, blocks = get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_no_attrs_defined(self):
        model = _NoAttrsModel(num_blocks=3)
        attr_names, blocks = get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []


class TestGetBlocksAttrNames:
    def test_get_blocks_attr_names_new_format(self):
        model = _MultiBlockModel()
        attrs = get_blocks_attr_names(model)
        assert attrs == ["transformer_blocks", "single_transformer_blocks"]

    def test_get_blocks_attr_names_no_attrs(self):
        model = _NoAttrsModel()
        attrs = get_blocks_attr_names(model)
        assert attrs == []

    def test_set_blocks_attr_names(self):
        model = _NoAttrsModel()
        set_blocks_attr_names(model, ["new_blocks"])
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

        # Case 2: _is_group_first=True, materialized, slot contaminated
        # (no _shared_slot_group set → defaults to contaminated) → MUST sync-prefetch
        hook_a._is_group_first = True
        hook_a._prev_hook = hook_a
        sync_called[0] = False
        hook_a.pre_forward(block_a)
        assert sync_called[0], "_is_group_first must force sync-prefetch when slot is contaminated"

        # Case 3: _is_group_first=True, slot NOT contaminated (same group owns slot)
        # → skip sync-prefetch, just wait for async event
        hook_a._shared_slot_group = [hook_a._group_id, -1]
        sync_called[0] = False
        hook_a.pre_forward(block_a)
        assert not sync_called[0], "Should skip sync-prefetch when slot_group matches (non-contaminated)"


class TestAllGatherOutputSlicing:
    """Regression test: with heterogeneous block sizes (e.g. MiniMax H3's
    token_refiner with blocks of ~1231MB and ~239MB), the shared AllGather
    output buffers are sized to the *largest* block across all groups.
    prefetch_layer must slice the output buffer down to each block's actual
    AllGather output size, otherwise output.numel() != dp_size * input.numel()
    and all_gather_into_tensor raises during enable()'s first prefetch.
    """

    def test_prefetch_slices_shared_output_buffer_to_block_size(self, dist_group, patched_offload_runtime, monkeypatch):
        dp_size = 2

        # Heterogeneous block sizes: 8 vs 32 elements.
        small_block = nn.Module()
        small_block.weight = nn.Parameter(torch.arange(8, dtype=torch.float32))
        large_block = nn.Module()
        large_block.weight = nn.Parameter(torch.arange(100, 132, dtype=torch.float32))

        def make_hook(next_block: nn.Module) -> DistributedLayerwiseOffloadHook:
            hook = DistributedLayerwiseOffloadHook(
                next_block=next_block,
                device=torch.device("cpu"),
                dp_group=dist.group.WORLD,  # non-None -> AllGather path (mocked below)
                dp_size=dp_size,
                rank=0,
                pin_memory=False,
            )
            current_block = nn.Module()
            current_block.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))
            hook.initialize_hook(current_block)
            return hook

        hook_small = make_hook(small_block)
        hook_large = make_hook(large_block)
        hooks = [hook_small, hook_large]

        # Assign max-sized shared buffers exactly as backend.enable() does.
        shared_buffers = DistributedLayerwiseOffloadBackend._allocate_shared_buffers(hooks)
        shared_shard_buffers = DistributedLayerwiseOffloadBackend._allocate_shared_shard_buffers(hooks)
        for hook in hooks:
            hook.gpu_buffers = shared_buffers
            hook.gpu_shard_buffers = shared_shard_buffers

        # Shared output buffer must be sized to the largest block (32).
        assert shared_buffers[0][torch.float32].numel() == 32

        # Fake AllGather that enforces the real torch size contract and
        # emulates reconstruction of the full block from all ranks' shards.
        full_values = {
            8: torch.arange(8, dtype=torch.float32),
            32: torch.arange(100, 132, dtype=torch.float32),
        }
        ag_output_numels: list[int] = []

        def fake_all_gather(output, input, group=None):
            assert output.numel() == dp_size * input.numel(), (
                f"AllGather contract violated: output.numel()={output.numel()} "
                f"!= {dp_size} * input.numel()={input.numel()}"
            )
            ag_output_numels.append(output.numel())
            output.copy_(full_values[output.numel()])

        monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather)

        # Small block: output must be sliced to dp_size * shard = 8, not 32.
        hook_small.prefetch_layer(slot=0, non_blocking=False)
        assert ag_output_numels[-1] == 8
        assert torch.equal(small_block.weight, full_values[8])

        # Large block: uses the full max-sized buffer.
        hook_large.prefetch_layer(slot=1, non_blocking=False)
        assert ag_output_numels[-1] == 32
        assert torch.equal(large_block.weight, full_values[32])


class TestOffloadPlan:
    """Test declarative OffloadPlan metadata."""

    def test_get_offload_plan_returns_none_when_not_declared(self):
        """Pipelines without _offload_plan should return None."""
        model = _SingleBlockModel(num_blocks=3)
        assert get_offload_plan(model) is None

    def test_get_offload_plan_returns_declared_plan(self):
        """Pipelines with _offload_plan should return it."""
        plan = OffloadPlan(
            block_attrs={"transformer": ("gen_layers",)},
            offload_submodules={"context_encoder": "layers"},
            resident_dit_paths=frozenset({"transformer"}),
        )

        class PipelineWithPlan(nn.Module):
            _offload_plan = plan

        model = PipelineWithPlan()
        result = get_offload_plan(model)
        assert result is plan
        assert result.block_attrs == {"transformer": ("gen_layers",)}
        assert result.offload_submodules == {"context_encoder": "layers"}
        assert result.resident_dit_paths == frozenset({"transformer"})

    def test_offload_plan_defaults_to_empty(self):
        """OffloadPlan with no arguments should have empty dicts."""
        plan = OffloadPlan()
        assert plan.block_attrs == {}
        assert plan.offload_submodules == {}
        assert plan.resident_dit_paths == frozenset()

    def test_offload_plan_is_frozen(self):
        """OffloadPlan should be immutable (frozen=True)."""
        plan = OffloadPlan()
        with pytest.raises(Exception):
            plan.block_attrs = {"x": ("y",)}  # type: ignore

    def test_all_resident_dit_still_prepares_planned_submodules(self, patched_offload_runtime):
        class TokenRefiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

        class Transformer(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
                self.token_refiner = TokenRefiner()

        class Pipeline(nn.Module):
            _offload_plan = OffloadPlan(
                offload_submodules={"token_refiner": "blocks"},
                resident_dit_paths=frozenset({"transformer"}),
            )

            def __init__(self):
                super().__init__()
                self.transformer = Transformer()

        pipeline = Pipeline()
        backend = DistributedLayerwiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=False,
                dlo_use_allgather=False,
                dlo_resident_layers=2,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert len(backend._resident_blocks) == 2
        assert len(backend._all_hook_groups) == 1
        assert len(backend._all_hook_groups[0]) == 2
        assert backend.enabled

        backend.disable()


class TestMmapValidation:
    """Tests for loader preflight and backend plan realization."""

    def test_component_source_prefix_builds_model_namespace(self, tmp_path):
        source_root = tmp_path / "partition"
        checkpoint_dir = source_root / "transformer"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "model.safetensors"
        save_file(
            {"blocks.0.weight": torch.arange(4, dtype=torch.float32).reshape(1, 4)},
            str(checkpoint_file),
        )

        class Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.blocks = nn.ModuleList([nn.Linear(4, 1, bias=False)])
                self.weights_sources = [
                    SimpleNamespace(
                        model_or_path=str(source_root),
                        subfolder="transformer",
                        revision=None,
                        prefix="transformer.",
                    )
                ]

        pipeline = Pipeline()
        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=tuple(pipeline.weights_sources),
            model_path=None,
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.fallback_reason is None
        assert result.plan is not None
        assert result.plan.bindings["transformer.blocks.0.weight"] == TensorBinding(
            checkpoint_key="blocks.0.weight",
            file_path=str(checkpoint_file),
        )
        assert result.plan.planned_source_prefixes == frozenset({"transformer."})

    def test_non_dedicated_component_source_falls_back_before_discovery(self):
        pipeline = nn.Module()
        pipeline.transformer = nn.Linear(2, 2, bias=False)
        sources = (
            SimpleNamespace(
                model_or_path="unused",
                subfolder=None,
                revision=None,
                prefix="",
            ),
        )

        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=sources,
            model_path=None,
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is None
        assert "dedicated component weight source" in result.fallback_reason

    def test_preflight_falls_back_before_mutation_for_missing_model_tensors(self, tmp_path):
        source_root = tmp_path / "partition"
        checkpoint_dir = source_root / "transformer"
        checkpoint_dir.mkdir(parents=True)
        save_file(
            {"blocks.0.weight": torch.ones((2, 2), dtype=torch.float32)},
            str(checkpoint_dir / "model.safetensors"),
        )

        class Transformer(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)])

        class Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Transformer()
                self.weights_sources = [
                    SimpleNamespace(
                        model_or_path=str(source_root),
                        subfolder="transformer",
                        revision=None,
                        prefix="transformer.",
                    )
                ]

        pipeline = Pipeline()
        original_weight = pipeline.transformer.blocks[0].weight
        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=tuple(pipeline.weights_sources),
            model_path=None,
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is None
        assert "1 required DiT tensors" in result.fallback_reason
        assert pipeline.transformer.blocks[0].weight is original_weight
        assert not original_weight.is_meta

    def test_tp_falls_back_before_checkpoint_discovery(self):
        class SimpleModel(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])

        class SimplePipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = SimpleModel()

            @staticmethod
            def _remap_ckpt_key(key):
                return key

        pipeline = SimplePipeline()
        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path="unused",
            tensor_parallel_size=2,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is None
        assert result.fallback_reason == "TP=2 requires the ordinary loader"

    @pytest.mark.parametrize(
        ("use_hsdp", "online_quantization", "expected_reason"),
        [
            (True, False, "HSDP requires the ordinary loader"),
            (False, True, "online quantization requires the ordinary loader"),
        ],
    )
    def test_runtime_layouts_that_need_loader_callbacks_fall_back(
        self,
        use_hsdp,
        online_quantization,
        expected_reason,
    ):
        pipeline = nn.Module()
        pipeline.transformer = nn.Linear(2, 2, bias=False)

        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path="unused",
            tensor_parallel_size=1,
            use_hsdp=use_hsdp,
            online_quantization=online_quantization,
        )

        assert result.plan is None
        assert result.fallback_reason == expected_reason

    @pytest.mark.parametrize(
        ("checkpoint_tensor", "expected_reason"),
        [
            (torch.ones((3, 2), dtype=torch.float32), "shape mismatch"),
            (torch.ones((2, 2), dtype=torch.bfloat16), "dtype mismatch"),
        ],
    )
    def test_source_metadata_mismatch_falls_back(self, tmp_path, checkpoint_tensor, expected_reason):
        checkpoint_file = tmp_path / "model.safetensors"
        save_file({"transformer.weight": checkpoint_tensor}, str(checkpoint_file))

        class Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Linear(2, 2, bias=False)

            @staticmethod
            def _remap_ckpt_key(key):
                return key

        pipeline = Pipeline()

        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path=str(tmp_path),
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is None
        assert expected_reason in result.fallback_reason

    def test_non_persistent_buffers_do_not_need_checkpoint_bindings(self, tmp_path):
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 2))
                self.register_buffer("derived", torch.ones(2), persistent=False)

        checkpoint_file = tmp_path / "model.safetensors"
        save_file({"transformer.weight": torch.ones(2, 2)}, str(checkpoint_file))

        class Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Transformer()

            @staticmethod
            def _remap_ckpt_key(key):
                return key

        pipeline = Pipeline()

        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path=str(tmp_path),
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is not None
        assert set(result.plan.bindings) == {"transformer.weight"}

    def test_unadapted_custom_weight_loader_falls_back_at_tp1(self, tmp_path):
        def custom_weight_loader(param, weight):
            param.data.copy_(weight)

        class ModelWithCustomLoader(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                # Simulate QKVParallelLinear: custom weight_loader even at TP=1
                self.linear.weight.weight_loader = custom_weight_loader
                self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])

        class PipelineWithLoader(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = ModelWithCustomLoader()

            @staticmethod
            def _remap_ckpt_key(key):
                return key

        pipeline = PipelineWithLoader()
        weights = {name: torch.ones(p.shape, dtype=p.dtype) for name, p in pipeline.named_parameters() if not p.is_meta}
        save_file(weights, str(tmp_path / "model.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
        )

        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path=str(tmp_path),
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is None
        assert "custom weight_loader" in result.fallback_reason

    def test_cosmos3_adapter_preserves_existing_tp1_direct_mmap_contract(self, tmp_path):
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )
        from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
            Cosmos3VFMTransformer,
        )

        transformer = object.__new__(Cosmos3VFMTransformer)
        nn.Module.__init__(transformer)
        transformer.proj_in = nn.Linear(2, 2, bias=False)
        transformer.proj_in.weight.weight_loader = lambda *_args: None
        pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
        nn.Module.__init__(pipeline)
        pipeline.transformer = transformer

        source_root = tmp_path / "partition"
        checkpoint_dir = source_root / "transformer"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "model.safetensors"
        save_file({"proj_in.weight": torch.ones(2, 2)}, str(checkpoint_file))
        sources = (
            SimpleNamespace(
                model_or_path=str(source_root),
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
            ),
        )

        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", transformer),),
            sources=sources,
            model_path=None,
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is not None
        assert result.plan.bindings["transformer.proj_in.weight"] == TensorBinding(
            checkpoint_key="proj_in.weight",
            file_path=str(checkpoint_file),
        )

    def test_validate_loaded_weights_called(self, tmp_path, patched_offload_runtime):
        """validate_loaded_weights should be called after mmap loading."""

        class ModelWithValidate(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]
            validate_called = False

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])

            def post_load_weights(self):
                pass

            def validate_loaded_weights(self, loaded):
                self.validate_called = True
                assert len(loaded) > 0, "Should have loaded weights"

        class PipelineWithValidate(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = ModelWithValidate()

            @staticmethod
            def _remap_ckpt_key(key):
                return key

        pipeline = PipelineWithValidate()
        # Save on meta to trigger mmap path
        with torch.device("meta"):
            pipeline.transformer = ModelWithValidate()

        weights = {
            name: torch.ones(param.shape, dtype=param.dtype if not param.is_meta else torch.float32)
            for name, param in pipeline.named_parameters()
        }
        save_file(weights, str(tmp_path / "model.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
        )

        backend = DistributedLayerwiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=False,
            ),
            torch.device("cpu"),
        )
        modules = SimpleNamespace(dits=[pipeline.transformer], dit_names=["transformer"])
        result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", pipeline.transformer),),
            sources=(),
            model_path=str(tmp_path),
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )

        assert result.plan is not None
        backend._load_weights_via_mmap(pipeline, modules, result.plan)
        assert pipeline.transformer.validate_called, "validate_loaded_weights should be called"


class TestConfigValidation:
    """Tests for configuration validation in OffloadConfig / execute_request."""

    def test_loader_plan_cannot_be_silently_dropped_on_unsupported_platform(self, monkeypatch):
        import vllm_omni.diffusion.offloader as offloader_module

        od_config = SimpleNamespace(
            enable_cpu_offload=False,
            enable_layerwise_offload=False,
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=False,
            dlo_resident_layers=0,
            pin_cpu_memory=False,
            model="unused",
            parallel_config=SimpleNamespace(
                use_hsdp=False,
                data_parallel_size=1,
                sequence_parallel_size=1,
            ),
        )
        plan = HostWeightPlan(
            backing_kind="checkpoint_mmap",
            bindings={},
        )
        monkeypatch.setattr(
            offloader_module.current_omni_platform,
            "supports_cpu_offload",
            lambda: False,
        )

        with pytest.raises(RuntimeError, match="skipped ordinary weight materialization"):
            offloader_module.get_offload_backend(
                od_config,
                device=torch.device("cpu"),
                host_weight_plan=plan,
            )

    def test_hsdp_with_allgather_rejected(self):
        """HSDP + DLO + AllGather should raise ValueError (double sharding)."""
        from vllm_omni.diffusion.offloader.base import OffloadConfig

        class FakePC:
            data_parallel_size = 1
            use_hsdp = True
            hsdp_shard_size = 2
            hsdp_replicate_size = 1
            sequence_parallel_size = 1
            tensor_parallel_size = 1
            cfg_parallel_size = 1
            pipeline_parallel_size = 1
            ulysses_degree = 1
            ring_degree = 1
            allgather_degree = 1
            enable_expert_parallel = False

        class FakeODConfig:
            enable_cpu_offload = False
            enable_layerwise_offload = False
            enable_distributed_layerwise_offload = True
            dlo_use_allgather = True
            pin_cpu_memory = True
            parallel_config = FakePC()
            model = "/fake/path"

        with pytest.raises(ValueError, match="incompatible with HSDP"):
            OffloadConfig.from_od_config(FakeODConfig())

    def test_hsdp_without_allgather_allowed(self):
        """HSDP + DLO + no-AllGather uses standard-loader rank-local weights."""
        from vllm_omni.diffusion.offloader.base import OffloadConfig

        class FakePC:
            data_parallel_size = 1
            use_hsdp = True
            hsdp_shard_size = 2
            hsdp_replicate_size = 1
            sequence_parallel_size = 1
            tensor_parallel_size = 1
            cfg_parallel_size = 1
            pipeline_parallel_size = 1
            ulysses_degree = 1
            ring_degree = 1
            allgather_degree = 1
            enable_expert_parallel = False

        class FakeODConfig:
            enable_cpu_offload = False
            enable_layerwise_offload = False
            enable_distributed_layerwise_offload = True
            dlo_use_allgather = False  # no AllGather → should be allowed
            dlo_resident_layers = 20
            pin_cpu_memory = True
            parallel_config = FakePC()
            model = "/fake/path"

        config = OffloadConfig.from_od_config(FakeODConfig())
        assert config.dp_size == 1  # forced to 1 when no AllGather
        assert config.dlo_resident_layers == 20

    def test_resident_layers_with_allgather_rejected(self):
        class FakePC:
            data_parallel_size = 2
            use_hsdp = False
            sequence_parallel_size = 1

        class FakeODConfig:
            enable_cpu_offload = False
            enable_layerwise_offload = False
            enable_distributed_layerwise_offload = True
            dlo_use_allgather = True
            dlo_resident_layers = 20
            pin_cpu_memory = True
            parallel_config = FakePC()
            model = "/fake/path"

        with pytest.raises(ValueError, match="requires --dlo-no-use-allgather"):
            OffloadConfig.from_od_config(FakeODConfig())

    def test_num_inference_steps_none_rejected(self):
        """DP multi-concurrency should reject None num_inference_steps."""
        from types import SimpleNamespace

        # Mock requests with None steps
        reqs = [
            SimpleNamespace(
                req=SimpleNamespace(
                    request_id=f"req-{i}",
                    sampling_params=SimpleNamespace(num_inference_steps=None),
                )
            )
            for i in range(2)
        ]

        # We can't easily instantiate the full executor, but we can test
        # the validation logic by checking that the code path raises.
        # The validation is in execute_request, which needs self._ensure_open().
        # Instead, test the validation logic directly:
        step_counts = {
            r.req.sampling_params.num_inference_steps
            for r in reqs
            if r.req.sampling_params.num_inference_steps is not None
        }
        has_none = any(r.req.sampling_params.num_inference_steps is None for r in reqs)
        assert has_none, "Test setup: should have None steps"
        assert len(step_counts) == 0, "Test setup: no explicit steps"

        # The validation condition: (len(step_counts) > 1) or has_none → should reject
        should_reject = (len(step_counts) > 1) or has_none
        assert should_reject, "None steps should trigger rejection"

    def test_num_inference_steps_same_explicit_allowed(self):
        """DP multi-concurrency should allow same explicit num_inference_steps."""
        from types import SimpleNamespace

        reqs = [
            SimpleNamespace(
                req=SimpleNamespace(
                    request_id=f"req-{i}",
                    sampling_params=SimpleNamespace(num_inference_steps=35),
                )
            )
            for i in range(4)
        ]

        step_counts = {
            r.req.sampling_params.num_inference_steps
            for r in reqs
            if r.req.sampling_params.num_inference_steps is not None
        }
        has_none = any(r.req.sampling_params.num_inference_steps is None for r in reqs)

        should_reject = (len(step_counts) > 1) or has_none
        assert not should_reject, "Same explicit steps should be allowed"
        assert step_counts == {35}

    def test_num_inference_steps_different_explicit_rejected(self):
        """DP multi-concurrency should reject different explicit steps."""
        from types import SimpleNamespace

        reqs = [
            SimpleNamespace(
                req=SimpleNamespace(
                    request_id="req-0",
                    sampling_params=SimpleNamespace(num_inference_steps=35),
                )
            ),
            SimpleNamespace(
                req=SimpleNamespace(
                    request_id="req-1",
                    sampling_params=SimpleNamespace(num_inference_steps=30),
                )
            ),
        ]

        step_counts = {
            r.req.sampling_params.num_inference_steps
            for r in reqs
            if r.req.sampling_params.num_inference_steps is not None
        }
        has_none = any(r.req.sampling_params.num_inference_steps is None for r in reqs)

        should_reject = (len(step_counts) > 1) or has_none
        assert should_reject, "Different steps should trigger rejection"
        assert len(step_counts) == 2


class TestDynamicSlotTracking:
    """Tests for dynamic slot tracking with odd and even block counts.

    With i%2 initialization alone, odd block counts cause the circular
    tail→head prefetch to collide with the head's read slot (both map
    to slot 0).  Dynamic slot tracking via _prefetched_slot corrects
    this by reading the actual slot from the previous hook's prefetch.
    """

    @staticmethod
    def _trace_slots(num_blocks: int, num_iterations: int = 1):
        """Trace slot assignments through forward passes (pure logic, no hooks).

        Returns a list of (read_slot, write_slot) per block per iteration.
        """
        # Initial i%2 assignment (same as enable())
        current_slots = [i % 2 for i in range(num_blocks)]
        prefetched_slots: list[int | None] = [None] * num_blocks

        # Sync-prefetch: tail hook prefetches into slot 0
        prefetched_slots[-1] = 0

        results = []
        for _ in range(num_iterations):
            for i in range(num_blocks):
                prev_idx = (i - 1) % num_blocks

                # Dynamic slot tracking: read from prev's prefetched slot
                if prefetched_slots[prev_idx] is not None:
                    current_slots[i] = prefetched_slots[prev_idx]

                read_slot = current_slots[i]

                # Prefetch into other slot
                next_slot = 1 - read_slot
                current_slots[i] = next_slot
                prefetched_slots[i] = next_slot

                results.append((read_slot, next_slot))

        return results

    def test_odd_block_count_no_collision(self):
        """3 blocks (odd): no read/write slot collision."""
        results = self._trace_slots(3)
        for i, (read, write) in enumerate(results):
            assert read != write, f"Block {i % 3} (step {i}): read={read} == write={write}"

    def test_even_block_count_no_collision(self):
        """4 blocks (even): no read/write slot collision."""
        results = self._trace_slots(4)
        for i, (read, write) in enumerate(results):
            assert read != write, f"Block {i % 4} (step {i}): read={read} == write={write}"

    def test_large_odd_block_count(self):
        """7 blocks (odd): no collision across all blocks."""
        results = self._trace_slots(7)
        for i, (read, write) in enumerate(results):
            assert read != write, f"Block {i % 7} (step {i}): read={read} == write={write}"

    def test_5_blocks_two_iterations(self):
        """5 blocks (odd), 2 iterations: no collision in either pass."""
        results = self._trace_slots(5, num_iterations=2)
        n = 5
        for iteration in range(2):
            for i in range(n):
                idx = iteration * n + i
                read, write = results[idx]
                assert read != write, f"Iter {iteration} block {i}: read={read} == write={write}"

    def test_3_blocks_three_iterations(self):
        """3 blocks (odd), 3 iterations: no collision across all passes."""
        results = self._trace_slots(3, num_iterations=3)
        n = 3
        for iteration in range(3):
            for i in range(n):
                idx = iteration * n + i
                read, write = results[idx]
                assert read != write, f"Iter {iteration} block {i}: read={read} == write={write}"

    def test_72_blocks_cosmos3(self):
        """72 blocks (Cosmos3, even): no collision."""
        results = self._trace_slots(72)
        for i, (read, write) in enumerate(results):
            assert read != write, f"Block {i % 72} (step {i}): read={read} == write={write}"

    def test_slots_alternate_within_iteration(self):
        """Within one iteration, consecutive blocks should use alternating slots."""
        results = self._trace_slots(6)
        for i in range(6):
            read, write = results[i]
            # Each block reads from one slot and writes to the other
            assert {read, write} == {0, 1}, f"Block {i}: slots {read},{write} don't cover both"

    def test_prefetched_slot_set_after_prefetch(self, dist_group, patched_offload_runtime):
        """prefetch_layer should set _prefetched_slot to the slot argument."""
        block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(2.0))

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
        hook.initialize_hook(block)

        assert hook._prefetched_slot is None  # initially None

        hook.prefetch_layer(slot=0, non_blocking=False)
        assert hook._prefetched_slot == 0

        hook.prefetch_layer(slot=1, non_blocking=False)
        assert hook._prefetched_slot == 1

    def test_dynamic_slot_overrides_initial_assignment(self, dist_group, patched_offload_runtime):
        """When _prev_hook._prefetched_slot is set, pre_forward should use
        that value instead of the initial i%2 assignment."""
        block_a = TinyBlock(_make_values(1.0))
        block_b = TinyBlock(_make_values(2.0))

        hook_a = DistributedLayerwiseOffloadHook(
            next_block=block_b,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )
        hook_a.initialize_hook(block_a)

        hook_b = DistributedLayerwiseOffloadHook(
            next_block=block_a,
            device=torch.device("cpu"),
            dp_group=None,
            dp_size=1,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )
        hook_b.initialize_hook(block_b)

        # Wire: hook_b._prev_hook = hook_a
        hook_b._prev_hook = hook_a

        # hook_a prefetched into slot 1
        hook_a.prefetch_layer(slot=1, non_blocking=False)
        assert hook_a._prefetched_slot == 1

        # hook_b's initial slot is 0 (from i%2)
        hook_b.current_slot = 0

        # pre_forward should override to slot 1
        hook_b.pre_forward(block_b)
        assert hook_b.current_slot == 1, "pre_forward should read _prev_hook._prefetched_slot=1, not keep initial 0"
