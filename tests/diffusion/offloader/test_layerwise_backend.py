# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for LayerwiseOffloadHook and LayerWiseOffloadBackend utilities."""

import gc
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
from tests.helpers.runtime import get_distributed_init_method
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


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()


@pytest.fixture(scope="module")
def dist_group():
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_distributed_init_method())
    try:
        yield
    finally:
        _cleanup_distributed()


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Stream", DummyStream)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Event", DummyEvent)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "current_stream", lambda: DummyStream())
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "stream", dummy_stream)


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

    def test_prefetch_preserves_transposed_weight_stride(self, patched_offload_runtime):
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

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )
        hook.initialize_hook(current_block)
        hook.prefetch_layer(non_blocking=False)

        assert next_block.weight.stride() == expected_stride
        assert torch.equal(next_block.weight, expected)


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


class _TrackingEncoder(nn.Module):
    """An encoder that records whether it was pinned whole onto the device."""

    def __init__(self, n_blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(n_blocks)])
        self.norm = nn.Linear(4, 4)
        self.moves: list[str] = []

    def to(self, *args, **kwargs):  # noqa: A003 - mirrors nn.Module.to
        self.moves.append("to")
        return super().to(*args, **kwargs)


class _StageableEncoder(_TrackingEncoder):
    """An encoder whose pipeline loads and releases it around each use."""

    def load_to_device(self) -> None:
        self.moves.append("load")

    def offload_to_cpu(self) -> None:
        self.moves.append("offload")


def _encoder_pipeline(encoder: nn.Module, *, declared: bool, on_demand: bool = False) -> nn.Module:
    from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

    attrs = {"text_encoder": ("blocks",)} if declared else {}
    paths = frozenset({"text_encoder"}) if on_demand else frozenset()

    class Pipeline(nn.Module):
        _dit_modules = ["transformer"]
        _encoder_modules = ["text_encoder"]
        _vae_modules: list[str] = []
        _offload_plan = OffloadPlan(encoder_block_attrs=attrs, on_demand_component_paths=paths)

        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.text_encoder = encoder

    return Pipeline()


def _plain_backend() -> LayerWiseOffloadBackend:
    from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy

    return LayerWiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=False),
        torch.device("cpu"),
    )


def test_layerwise_backend_streams_declared_encoder_blocks_and_places_the_rest(
    patched_offload_runtime,
) -> None:
    """Blocks are paged; everything else still has to reach the device.

    Installing the hooks leaves each block parameter as an empty placeholder, so
    the follow-up ``to()`` only carries the non-block state -- norms, embeddings,
    projections -- which the first forward needs on the device. Skipping it is
    what the distributed backend does not do: it calls `_register_on_demand_hook`
    after installing the very same hooks.
    """
    encoder = _TrackingEncoder()
    backend = _plain_backend()

    backend.enable(_encoder_pipeline(encoder, declared=True))

    assert getattr(encoder, "_omni_layerwise_enabled", False) is True
    assert len(encoder._omni_layerwise_block_groups) == 1
    assert encoder.moves == ["to"], "non-block encoder state must still be placed on the device"
    assert not LayerwiseOffloadHook._is_materialized_tensor(encoder.blocks[0].weight), (
        "block weights must stay paged out; only the non-block state is placed"
    )
    backend.disable()


def test_layerwise_backend_parks_a_declared_on_demand_encoder_instead_of_placing_it(
    patched_offload_runtime,
) -> None:
    """An encoder the pipeline stages itself is left to that lifecycle."""
    encoder = _StageableEncoder()
    backend = _plain_backend()

    backend.enable(_encoder_pipeline(encoder, declared=True, on_demand=True))

    assert encoder.moves == ["offload"], "a pipeline-managed encoder must be parked, not placed"
    backend.disable()


def test_layerwise_backend_disable_tears_down_streamed_encoder_blocks(patched_offload_runtime) -> None:
    """disable() must remove the encoder hooks, restore weights and clear state.

    Leaving ``_omni_layerwise_enabled`` set makes the next enable() return early
    on its idempotence check without rebuilding the hooks, while the blocks are
    still placeholders.
    """
    encoder = _TrackingEncoder()
    backend = _plain_backend()
    pipeline = _encoder_pipeline(encoder, declared=True)

    backend.enable(pipeline)
    assert encoder._omni_layerwise_enabled is True

    backend.disable()

    assert getattr(encoder, "_omni_layerwise_enabled", False) is False
    assert encoder._omni_layerwise_hooks == []
    assert encoder._omni_layerwise_block_groups == []
    for block in encoder.blocks:
        registry = getattr(block, "_hook_registry", None)
        assert registry is None or LayerwiseOffloadHook._HOOK_NAME not in registry._hooks
        assert LayerwiseOffloadHook._is_materialized_tensor(block.weight), (
            "weights must be materialized again, not left as placeholders"
        )

    # A second cycle must rebuild rather than short-circuit on stale state.
    backend.enable(pipeline)
    assert encoder._omni_layerwise_enabled is True
    assert len(encoder._omni_layerwise_block_groups) == 1
    backend.disable()


def test_layerwise_backend_teardown_restores_weights_without_going_through_the_device(
    patched_offload_runtime, mocker
) -> None:
    """Teardown must not bring the whole encoder back onto the accelerator.

    `prefetch_layer()` is the only restore path that allocates on the backend's
    device, and it frees nothing, so using it for teardown would leave every
    block of a streamed stack resident at once -- the residency this backend
    exists to avoid, at the moment (shutdown, failed enable) when the headroom
    is least available. Teardown rebuilds on the host instead.

    Asserting the mechanism rather than a device string is deliberate: driving
    the backend at a fake device is not a faithful proxy, because
    `Module.to("meta")` replaces parameter objects instead of converting them
    in place, which no real accelerator transfer does.
    """
    encoder = _TrackingEncoder()
    backend = _plain_backend()
    spy = mocker.spy(LayerwiseOffloadHook, "prefetch_layer")

    backend.enable(_encoder_pipeline(encoder, declared=True))
    calls_after_enable = spy.call_count
    backend.disable()

    assert spy.call_count == calls_after_enable, "teardown must not restore through the device path"
    for block in encoder.blocks:
        assert LayerwiseOffloadHook._is_materialized_tensor(block.weight)
        assert block.weight.device.type == "cpu"
        assert block.weight.shape == (4, 4)


def test_layerwise_backend_tears_down_encoder_even_when_the_dit_has_no_streamable_blocks(
    patched_offload_runtime,
) -> None:
    """`enabled` only tracks DiT blocks, so teardown cannot be gated on it.

    A DiT without a streamable block list leaves `enabled` False while the
    encoder hooks are already installed; an `enabled`-gated disable() would
    return early and strand them.
    """
    from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

    encoder = _TrackingEncoder()

    class Pipeline(nn.Module):
        _dit_modules = ["transformer"]
        _encoder_modules = ["text_encoder"]
        _vae_modules: list[str] = []
        _offload_plan = OffloadPlan(encoder_block_attrs={"text_encoder": ("blocks",)})

        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            # A single block is not a streamable stack.
            self.transformer.blocks = nn.ModuleList([nn.Linear(4, 4)])
            self.text_encoder = encoder

    backend = _plain_backend()
    backend.enable(Pipeline())

    assert backend.enabled is False, "the DiT has nothing to stream"
    assert encoder._omni_layerwise_enabled is True, "the encoder was still streamed"

    backend.disable()

    assert getattr(encoder, "_omni_layerwise_enabled", False) is False
    assert encoder._omni_layerwise_hooks == []


def test_layerwise_backend_keeps_undeclared_encoder_resident(patched_offload_runtime) -> None:
    """Without a declaration there is nothing to page: keep today's behavior."""
    encoder = _TrackingEncoder()
    backend = _plain_backend()

    backend.enable(_encoder_pipeline(encoder, declared=False))

    assert getattr(encoder, "_omni_layerwise_enabled", False) is False
    assert encoder.moves == ["to"]
    backend.disable()


def test_layerwise_backend_keeps_encoder_resident_when_path_is_not_a_block_list(patched_offload_runtime) -> None:
    """A declared path that is not a streamable stack degrades through the backend."""
    from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

    encoder = _TrackingEncoder()
    pipeline = _encoder_pipeline(encoder, declared=True)
    # Declare a path that exists but is a plain module, not a block list.
    pipeline._offload_plan = OffloadPlan(encoder_block_attrs={"text_encoder": ("norm",)})
    backend = _plain_backend()

    backend.enable(pipeline)

    assert getattr(encoder, "_omni_layerwise_enabled", False) is False
    assert encoder.moves == ["to"], "the backend must fall back to placing the encoder"
    backend.disable()


class _TransposedEncoder(nn.Module):
    """An encoder whose block weights carry a transposed (non-contiguous) layout.

    Online-FP8 Cutlass weights are stored this way; ``_to_cpu`` records their
    ``stride`` and stages the physical layout, and ``prefetch_layer`` restores
    it with ``torch.as_strided`` (see
    ``test_prefetch_preserves_transposed_weight_stride``). Teardown has to do
    the same.
    """

    class _StridedBlock(nn.Module):
        def __init__(self, start: float) -> None:
            super().__init__()
            base = torch.arange(start, start + 12, dtype=torch.float32).reshape(3, 4)
            self.weight = nn.Parameter(base.t(), requires_grad=False)

    def __init__(self, n_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([self._StridedBlock(1.0 + 100.0 * i) for i in range(n_blocks)])
        self.norm = nn.Linear(4, 4)
        self.moves: list[str] = []

    def to(self, *args, **kwargs):  # noqa: A003 - mirrors nn.Module.to
        self.moves.append("to")
        return super().to(*args, **kwargs)


def test_layerwise_backend_teardown_preserves_transposed_block_weights(patched_offload_runtime) -> None:
    """disable() must rebuild the physical layout, not reinterpret the slice.

    ``restore_next_block_to_host`` reads back a flat slice of the staging
    buffer. ``_to_cpu`` wrote that slice in the weight's *physical* order, so
    reinterpreting it contiguously with ``.view(shape)`` silently scrambles the
    element order of any transposed weight. Rebuilding it with
    ``torch.as_strided(..., metadata["stride"])`` -- the same thing
    ``prefetch_layer`` does -- keeps the values and the layout.
    """
    encoder = _TransposedEncoder()
    expected = [block.weight.detach().clone() for block in encoder.blocks]
    expected_strides = [block.weight.stride() for block in encoder.blocks]
    assert expected_strides[0] == (1, 4), "the fixture must not be contiguous"

    backend = _plain_backend()
    pipeline = _encoder_pipeline(encoder, declared=True)

    backend.enable(pipeline)
    backend.disable()

    for block, want, want_stride in zip(encoder.blocks, expected, expected_strides):
        assert LayerwiseOffloadHook._is_materialized_tensor(block.weight)
        assert torch.equal(block.weight, want), "transposed weight came back with a scrambled element order"
        assert block.weight.stride() == want_stride
