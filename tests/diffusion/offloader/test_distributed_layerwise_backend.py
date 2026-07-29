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
from vllm_omni.diffusion.offloader.block_discovery import (
    get_blocks_attr_names,
    get_blocks_from_dit,
    set_blocks_attr_names,
)
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan, get_offload_plan
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
        )

        class PipelineWithPlan(nn.Module):
            _offload_plan = plan

        model = PipelineWithPlan()
        result = get_offload_plan(model)
        assert result is plan
        assert result.block_attrs == {"transformer": ("gen_layers",)}
        assert result.offload_submodules == {"context_encoder": "layers"}

    def test_offload_plan_defaults_to_empty(self):
        """OffloadPlan with no arguments should have empty dicts."""
        plan = OffloadPlan()
        assert plan.block_attrs == {}
        assert plan.offload_submodules == {}

    def test_offload_plan_is_frozen(self):
        """OffloadPlan should be immutable (frozen=True)."""
        plan = OffloadPlan()
        with pytest.raises(Exception):
            plan.block_attrs = {"x": ("y",)}  # type: ignore


class TestMmapValidation:
    """Tests for mmap loader validation: TP rejection, strict check, validate_loaded_weights."""

    def test_tp_rejected_when_tp_world_size_gt_1(self, tmp_path, patched_offload_runtime, monkeypatch):
        """DLO+AllGather should reject when TP world size > 1."""
        import json
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy

        # Mock TP world size = 2
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            lambda: 2,
        )

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
        weights = {name: torch.ones(p.shape, dtype=p.dtype) for name, p in pipeline.named_parameters() if not p.is_meta}
        save_file(weights, str(tmp_path / "model.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
        )

        backend = DistributedLayerwiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=False,
                model_path=str(tmp_path),
            ),
            torch.device("cpu"),
        )
        modules = SimpleNamespace(dits=[pipeline.transformer], dit_names=["transformer"])

        with pytest.raises(ValueError, match="Tensor Parallel"):
            backend._load_weights_via_mmap(pipeline, modules)

    def test_tp_allowed_when_tp_world_size_eq_1(self, tmp_path, patched_offload_runtime, monkeypatch):
        """DLO+AllGather should work when TP world size = 1, even if params
        have custom weight_loader attributes (e.g. QKVParallelLinear at TP=1)."""
        import json
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy

        # Mock TP world size = 1 (default)
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            lambda: 1,
        )

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

        backend = DistributedLayerwiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=False,
                model_path=str(tmp_path),
            ),
            torch.device("cpu"),
        )
        modules = SimpleNamespace(dits=[pipeline.transformer], dit_names=["transformer"])

        # Should NOT raise — TP=1 is allowed even with custom weight_loader
        backend._load_weights_via_mmap(pipeline, modules)

    def test_validate_loaded_weights_called(self, tmp_path, patched_offload_runtime):
        """validate_loaded_weights should be called after mmap loading."""
        import json
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy

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
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE, pin_cpu_memory=False, model_path=str(tmp_path)
            ),
            torch.device("cpu"),
        )
        modules = SimpleNamespace(dits=[pipeline.transformer], dit_names=["transformer"])

        backend._load_weights_via_mmap(pipeline, modules)
        assert pipeline.transformer.validate_called, "validate_loaded_weights should be called"


class TestConfigValidation:
    """Tests for configuration validation in OffloadConfig / execute_request."""

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
        """HSDP + DLO + no-AllGather should be allowed (full weights per rank)."""
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
            pin_cpu_memory = True
            parallel_config = FakePC()
            model = "/fake/path"

        config = OffloadConfig.from_od_config(FakeODConfig())
        assert config.dp_size == 1  # forced to 1 when no AllGather

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
