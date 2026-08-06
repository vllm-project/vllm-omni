# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LayerwiseOffloadHook and LayerWiseOffloadBackend utilities."""

import gc
import os
import socket
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend, LayerwiseOffloadHook
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
def patched_offload_runtime(mocker):
    mocker.patch.object(layerwise_backend_module.current_omni_platform, "Stream", DummyStream)
    mocker.patch.object(layerwise_backend_module.current_omni_platform, "Event", DummyEvent)
    mocker.patch.object(layerwise_backend_module.current_omni_platform, "current_stream", lambda: DummyStream())
    mocker.patch.object(layerwise_backend_module.current_omni_platform, "stream", dummy_stream)


class TinyBlock(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        mesh = DeviceMesh("cpu", [0])
        dtensor = DTensor.from_local(values, mesh, [Replicate()])
        self.weight = nn.Parameter(dtensor)


def _make_values(start: float) -> torch.Tensor:
    return torch.arange(start, start + 4, dtype=torch.float32)


class TestLayerwiseOffloadHook:
    def test_dtensor_wrapper_is_preserved_across_prefetch_and_offload(self, dist_group, patched_offload_runtime):
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        assert isinstance(next_block.weight, DTensor)
        assert next_block.weight.to_local().is_meta
        assert next_block.weight.to_local().shape == torch.Size([4])
        assert hook.dtype_metadata[next_block.weight.dtype][0]["shape"] == torch.Size([4])

        hook.prefetch_layer(non_blocking=False)
        assert isinstance(next_block.weight, DTensor)
        assert torch.equal(next_block.weight.to_local(), _make_values(10.0))
        assert next_block.weight.to_local().shape == torch.Size([4])

        hook.offload_layer()
        assert isinstance(current_block.weight, DTensor)
        assert current_block.weight.to_local().is_meta
        assert current_block.weight.to_local().shape == torch.Size([4])
        assert not hook.is_materialized


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


class _InvalidAttrModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["nonexistent_blocks", "blocks"]

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _DeprecatedSingleAttrModel(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _NoAttrsModel(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class TestGetBlocksFromDit:
    def test_get_blocks_from_dit_single_block_attr(self):
        model = _SingleBlockModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 3
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_multi_block_attrs(self):
        model = _MultiBlockModel(num_transformer=2, num_single=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert set(attr_names) == {"transformer_blocks", "single_transformer_blocks"}
        assert len(blocks) == 5
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_empty_blocks(self):
        model = _EmptyBlocksModel()
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_invalid_attr_name(self):
        model = _InvalidAttrModel(num_blocks=2)
        with pytest.raises(
            AttributeError,
            match="Attribute 'nonexistent_blocks' declared in _layerwise_offload_blocks_attrs does not exist",
        ):
            LayerWiseOffloadBackend.get_blocks_from_dit(model)

    def test_get_blocks_from_dit_no_attrs_defined(self):
        model = _NoAttrsModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_deprecated_single_attr(self):
        model = _DeprecatedSingleAttrModel(num_blocks=2)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 2


class TestGetBlocksAttrNames:
    def test_get_blocks_attr_names_new_format(self):
        model = _MultiBlockModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == ["transformer_blocks", "single_transformer_blocks"]

    def test_get_blocks_attr_names_no_attrs(self):
        model = _NoAttrsModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == []

    def test_set_blocks_attr_names(self):
        model = _NoAttrsModel()
        LayerWiseOffloadBackend.set_blocks_attr_names(model, ["new_blocks"])
        assert hasattr(model.__class__, "_layerwise_offload_blocks_attrs")
        assert model.__class__._layerwise_offload_blocks_attrs == ["new_blocks"]


def _round_trip(params: dict, bufs: dict | None = None) -> dict[str, torch.Tensor]:
    """Serialise params+bufs to CPU via _to_cpu, then restore via the prefetch path.

    Returns a mapping of name -> restored tensor, matching the shapes and
    strides that LayerwiseOffloadHook.prefetch_layer would write back.
    """
    bufs = bufs or {}
    cpu_weights, metadata = LayerwiseOffloadHook._to_cpu(
        params, bufs, device=torch.device("cpu"), pin_memory=False
    )

    restored: dict[str, torch.Tensor] = {}
    for dtype, ordered_meta in metadata.items():
        flat = cpu_weights[dtype]
        for m in ordered_meta:
            base = flat[m["offset"] : m["offset"] + m["numel"]].view(m["shape"])
            restored[m["name"]] = base.t() if m.get("store_transposed") else base

    return restored


class TestNonContigRoundTrip:
    """_to_cpu / prefetch_layer must preserve the original strides and data.

    FP8 linear weights are often stored as F.pad(...).t() — a 2-D column-major
    (F-order) tensor. The old code called .flatten() (C-logical order) and
    restored with .view(shape), assigning C-major strides and causing the Cutlass
    W8A8 kernel to read garbage data.
    """

    def test_forder_weight_strides_are_preserved(self):
        """Restored tensor has the same strides as the original F-order weight."""
        w = torch.randn(8, 16, dtype=torch.float32).t().contiguous().t()
        assert w.t().is_contiguous()

        params = {"weight": nn.Parameter(w)}
        restored = _round_trip(params)

        assert restored["weight"].stride() == w.stride()

    def test_corder_weight_is_unaffected(self):
        """Standard C-contiguous weights pass through without transposition."""
        w = torch.randn(16, 32, dtype=torch.float32)
        assert w.is_contiguous()

        params = {"weight": nn.Parameter(w)}
        restored = _round_trip(params)

        assert torch.equal(restored["weight"], w)
        assert restored["weight"].stride() == w.stride()
        assert restored["weight"].is_contiguous()

    def test_1d_bias_is_unaffected(self):
        """1-D tensors (biases) are never transposed regardless of layout."""
        b = torch.randn(64, dtype=torch.float32)
        params = {"bias": nn.Parameter(b)}
        restored = _round_trip(params)

        assert torch.equal(restored["bias"], b)

    def test_mixed_forder_and_corder_in_same_block(self):
        """F-order weight and C-order bias coexist correctly in one serialisation."""
        W = torch.randn(8, 16, dtype=torch.float32)
        W_forder = W.t().contiguous().t()
        bias = torch.randn(8, dtype=torch.float32)

        params = {
            "weight": nn.Parameter(W_forder),
            "bias": nn.Parameter(bias),
        }
        restored = _round_trip(params)

        assert torch.equal(restored["weight"], W_forder)
        assert restored["weight"].stride() == W_forder.stride()
        assert torch.equal(restored["bias"], bias)

    def test_fp8_padded_weight_pattern(self):
        """Reproduce the exact FP8 padding pattern used by vLLM's linear layers.

        vLLM pads weight rows to a GEMM-friendly alignment then transposes:
            qweight = F.pad(weight, (0, pad_cols)).t()
        The result is F-order. Verify round-trip preserves data exactly.
        """
        rows, cols, pad_cols = 32, 60, 4  # pad to 64
        weight = torch.randn(rows, cols, dtype=torch.float32)
        qweight = torch.nn.functional.pad(weight, (0, pad_cols)).t()

        assert qweight.t().is_contiguous(), "padded+transposed weight should be F-order"
        assert not qweight.is_contiguous()

        params = {"qweight": nn.Parameter(qweight)}
        restored = _round_trip(params)

        assert torch.equal(restored["qweight"], qweight)
        assert restored["qweight"].stride() == qweight.stride()

    def test_hook_full_round_trip_with_forder_weight(self, monkeypatch):
        """End-to-end: LayerwiseOffloadHook initialize → prefetch preserves F-order weight."""

        monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Stream", DummyStream)
        monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Event", DummyEvent)
        monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "current_stream", lambda: DummyStream())
        monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "stream", dummy_stream)

        class FOrderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                w = torch.arange(24, dtype=torch.float32).reshape(4, 6)
                # Store as F-order (column-major) to mimic FP8 padded weight
                self.weight = nn.Parameter(w.t().contiguous().t())

        current_block = FOrderBlock()
        next_block = FOrderBlock()
        original_weight = next_block.weight.data.clone()
        original_stride = next_block.weight.data.stride()

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )
        hook.initialize_hook(current_block)

        # initialize_hook serialises the NEXT block to CPU; next_block's weight
        # becomes a zero-numel placeholder.
        assert next_block.weight.data.numel() == 0, "next block weight should be offloaded"

        hook.prefetch_layer(non_blocking=False)

        assert torch.equal(next_block.weight.data, original_weight), (
            "weight data corrupted after prefetch_layer"
        )
        assert next_block.weight.data.stride() == original_stride, (
            "strides changed after prefetch_layer"
        )
