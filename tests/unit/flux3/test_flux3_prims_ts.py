# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the Flux3 PrimsTS FP8 and NVFP4 adapters."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearMethodBase

from vllm_omni.diffusion.layers import prims_ts_nvfp4
from vllm_omni.diffusion.models.flux3 import flux3_prims_ts, flux3_prims_ts_config
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter_factory import (
    Flux3PrimsTSConfig,
    Flux3PrimsTSMode,
    Flux3PrimsTSSupport,
)
from vllm_omni.diffusion.models.flux3.flux3_prims_ts import (
    Flux3BlockKernelPlan,
    Flux3PrimsTSKernelAdapter,
    prepare_fp8_layer,
    prepare_nvfp4_layer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PassthroughMethod(LinearMethodBase):
    def __init__(self) -> None:
        self.processed = False
        self.out_dtype = torch.bfloat16

    def create_weights(self, *args, **kwargs) -> None:
        del args, kwargs

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.processed = True

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, bias
        return x


class _FakeFp8Method(_PassthroughMethod):
    pass


class _OtherMethod(_PassthroughMethod):
    pass


def _layer(
    *,
    input_size: int = 128,
    output_size: int = 512,
    method: LinearMethodBase | None = None,
) -> nn.Module:
    layer = nn.Module()
    layer.input_size_per_partition = input_size
    layer.output_size_per_partition = output_size
    layer.quant_method = method or _FakeFp8Method()
    layer.weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=False)
    return layer


def _adapter(mode: Flux3PrimsTSMode = Flux3PrimsTSMode.AUTO) -> Flux3PrimsTSKernelAdapter:
    quant_config = SimpleNamespace(LinearMethodCls=_FakeFp8Method)
    return Flux3PrimsTSKernelAdapter(
        lambda q, k, _v, _norm, _rope: (q, k),
        prims_ts_config=Flux3PrimsTSConfig(mode=mode),
        prims_ts_support=Flux3PrimsTSSupport(True),
        quant_config=quant_config,
        quant_format="fp8",
    )


def test_prims_ts_fused_qkv_targets_only_supported_blocks() -> None:
    adapter = _adapter()
    assert adapter.uses_fused_qkv("content_mode_blocks.image.0", 128)
    assert adapter.uses_fused_qkv("single_blocks.0", 128)
    assert not adapter.uses_fused_qkv("content_mode_blocks.image_cond.0", 128)
    assert not adapter.uses_fused_qkv("txt_mode_blocks.0", 128)
    assert not adapter.uses_fused_qkv("late_content_mode_blocks.image.0", 128)
    assert not adapter.uses_fused_qkv("content_mode_blocks.image.0", 32)


def _weight_layer() -> nn.Module:
    layer = nn.Module()
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    layer.weight = nn.Parameter(values.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.arange(1, 5, dtype=torch.float32), requires_grad=False)
    return layer


def test_prepared_weights_are_nonpersistent_buffers() -> None:
    layer = _weight_layer()

    prepare_fp8_layer(layer, "linear")

    assert flux3_prims_ts._FP8_WEIGHT_BUFFER not in layer.state_dict()
    assert flux3_prims_ts._FP8_SCALE_BUFFER not in layer.state_dict()


def test_quant_method_wrapper_prepares_after_base_processing() -> None:
    layer = _weight_layer()
    base_method = _FakeFp8Method()
    method = flux3_prims_ts._PrimsTSFp8LinearMethod(base_method, "linear")

    method.process_weights_after_loading(layer)

    assert base_method.processed is True
    assert flux3_prims_ts._FP8_WEIGHT_BUFFER in layer._buffers


def test_rope_cache_flattens_the_current_flux3_layout() -> None:
    rope = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    cache = flux3_prims_ts._rope_cache(rope)

    torch.testing.assert_close(cache, rope.reshape(6, 4))


def test_adapter_builds_one_immutable_plan_for_supported_image_block() -> None:
    block = SimpleNamespace(
        qkv=_layer(output_size=1536),
        attn_out=_layer(),
        mlp_in=_layer(output_size=1024),
        mlp_out=_layer(),
        norm=SimpleNamespace(query_norm=object()),
        head_dim=128,
        num_heads_local=4,
        num_kv_heads_local=4,
    )

    plan = _adapter().configure_block(block, "content_mode_blocks.image.0")

    assert plan.qkv_qknorm_rope is True
    assert plan.attention_output is True
    assert plan.mlp_swiglu is True
    assert plan.mlp_output is True
    assert isinstance(block.qkv.quant_method, flux3_prims_ts._PrimsTSFp8LinearMethod)
    assert isinstance(block.mlp_in.quant_method, flux3_prims_ts._PrimsTSFp8LinearMethod)
    with pytest.raises(AttributeError):
        plan.mlp_output = False


def test_adapter_skips_checkpoint_excluded_and_nontarget_linears() -> None:
    block = SimpleNamespace(
        qkv=_layer(method=_OtherMethod()),
        attn_out=_layer(method=_OtherMethod()),
        mlp_in=_layer(method=_OtherMethod()),
        mlp_out=_layer(method=_OtherMethod()),
        norm=SimpleNamespace(query_norm=object()),
        head_dim=128,
        num_heads_local=4,
        num_kv_heads_local=4,
    )
    adapter = _adapter()

    excluded_plan = adapter.configure_block(block, "content_mode_blocks.image.0")
    text_plan = adapter.configure_block(block, "content_mode_blocks.text.0")

    assert excluded_plan.enabled is False
    assert text_plan.enabled is False
    assert isinstance(block.qkv.quant_method, _OtherMethod)
    assert isinstance(block.mlp_in.quant_method, _OtherMethod)


def test_required_mode_rejects_model_without_compatible_blocks() -> None:
    with pytest.raises(RuntimeError, match="no compatible FP8 transformer blocks"):
        _adapter(Flux3PrimsTSMode.REQUIRED).finalize()


class _FakeNvFp4Method(_PassthroughMethod):
    def __init__(self, quant_config: object) -> None:
        super().__init__()
        self.quant_config = quant_config


def _nvfp4_layer(
    quant_config: object,
    *,
    output_size: int = 512,
    global_scale: float = 0.5,
    input_global_scale: float = 0.25,
) -> nn.Module:
    input_size = 256
    layer = _layer(
        input_size=input_size,
        output_size=output_size,
        method=_FakeNvFp4Method(quant_config),
    )
    layer.weight = nn.Parameter(
        torch.arange(output_size * (input_size // 2), dtype=torch.uint8).reshape(output_size, -1), requires_grad=False
    )
    logical_scales = torch.ones((output_size, input_size // 16), dtype=torch.float8_e4m3fn)
    packed_scales = prims_ts_nvfp4.nvfp4_scales_to_128x4(logical_scales)
    layer.weight_scale = nn.Parameter(
        packed_scales.view(torch.float8_e4m3fn).reshape_as(logical_scales), requires_grad=False
    )
    layer.weight_global_scale = nn.Parameter(torch.tensor([global_scale], dtype=torch.float32), requires_grad=False)
    layer.input_global_scale = nn.Parameter(
        torch.tensor([input_global_scale], dtype=torch.float32), requires_grad=False
    )
    layer.input_global_scale_inv = nn.Parameter(
        torch.tensor([1.0 / input_global_scale], dtype=torch.float32), requires_grad=False
    )
    return layer


def test_prepare_nvfp4_swiglu_interleaves_rows_without_mutating_base_weight() -> None:
    quant_config = object()
    layer = _nvfp4_layer(quant_config)
    original_weight = layer.weight.detach().clone()

    expected_weight, expected_scales = prims_ts_nvfp4.prepare_prims_ts_nvfp4_weights(
        layer.weight,
        layer.weight_scale,
        layout="swiglu",
    )
    prepare_nvfp4_layer(layer, "swiglu")

    torch.testing.assert_close(layer._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER], expected_weight)
    torch.testing.assert_close(layer._buffers[flux3_prims_ts._NVFP4_SCALE_BUFFER], expected_scales)
    torch.testing.assert_close(layer.weight, original_weight)
    assert flux3_prims_ts._NVFP4_WEIGHT_BUFFER not in layer.state_dict()
    assert flux3_prims_ts._NVFP4_SCALE_BUFFER not in layer.state_dict()


def test_nvfp4_prepared_linear_operator_is_cached_on_its_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = nn.Module()
    weight = torch.empty((16, 8), dtype=torch.uint8)
    weight_scale = torch.ones((16,), dtype=torch.float32, requires_grad=True)
    prepare_calls = []

    class _FakeGemm:
        def prepare_fp4_linear(self, *args, **kwargs):
            prepare_calls.append((args, kwargs))
            return object()

    monkeypatch.setattr(flux3_prims_ts, "flux3_nvfp4_gemm_config", lambda *args, **kwargs: "measured-config")
    monkeypatch.setattr(prims_ts_nvfp4, "_gemm", lambda: _FakeGemm())
    cache_id = prims_ts_nvfp4.register_prims_ts_nvfp4_layer(layer, config_provider="flux3")

    first = prims_ts_nvfp4._prepare_nvfp4_linear(cache_id, "linear", weight, weight_scale, 0.25, 0.5, 128)
    second = prims_ts_nvfp4._prepare_nvfp4_linear(cache_id, "linear", weight, weight_scale, 0.25, 0.5, 128)
    other_shape = prims_ts_nvfp4._prepare_nvfp4_linear(cache_id, "linear", weight, weight_scale, 0.25, 0.5, 256)

    assert first is second
    assert other_shape is not first
    assert len(prepare_calls) == 2
    assert len(getattr(layer, prims_ts_nvfp4._PREPARED_ATTR)["linear"]) == 2
    assert all(call_kwargs["config"] == "measured-config" for _, call_kwargs in prepare_calls)
    assert all(not args[0].requires_grad and not args[1].requires_grad for args, _ in prepare_calls)


def test_nvfp4_swiglu_prepares_packed_output_for_the_down_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = nn.Module()
    weight = torch.empty((32, 8), dtype=torch.uint8)
    weight_scale = torch.ones((32,), dtype=torch.float32, requires_grad=True)
    prepare_kwargs = []

    class _FakeGemm:
        def prepare_fp4_linear_swiglu(self, *args, **kwargs):
            prepare_kwargs.append((args, kwargs))
            return object()

    monkeypatch.setattr(flux3_prims_ts, "flux3_nvfp4_gemm_config", lambda *args, **kwargs: "measured-config")
    monkeypatch.setattr(prims_ts_nvfp4, "_gemm", lambda: _FakeGemm())
    cache_id = prims_ts_nvfp4.register_prims_ts_nvfp4_layer(layer, config_provider="flux3")
    prims_ts_nvfp4._prepare_nvfp4_linear(cache_id, "swiglu", weight, weight_scale, 0.25, 0.5, 128)

    _, kwargs = prepare_kwargs[0]
    assert kwargs["out_dtype"] is torch.uint8
    assert kwargs["config"] == "measured-config"
    assert prepare_kwargs[0][0][1].requires_grad is False
    torch.testing.assert_close(kwargs["output_quant_scale"], torch.ones(1))


def test_nvfp4_measured_config_selects_shape_specific_tactic(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.delenv(flux3_prims_ts_config.AUTOTUNE_ENV, raising=False)
    monkeypatch.setattr(flux3_prims_ts_config, "_config_type", lambda: _Config)
    monkeypatch.setattr(
        flux3_prims_ts_config.current_platform,
        "get_device_capability",
        lambda _device_index: SimpleNamespace(to_int=lambda: 100),
    )

    config = flux3_prims_ts_config.flux3_nvfp4_gemm_config(
        torch.device("cuda:0"), epilogue="linear", k=3072, n=9216, m=8192
    )

    assert config.kwargs["cluster_shape"] == (2, 2, 1)
    assert config.kwargs["tile_k"] == 256
    assert config.kwargs["tmem_overlap"] is True
    assert config.kwargs["ab_stages"] == 5
    assert config.kwargs["scheduler"] == "clc_dynamic"


def test_autotune_config_uses_flashinfer_defaults_for_fp8_and_nvfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(flux3_prims_ts_config.AUTOTUNE_ENV, "true")

    assert (
        flux3_prims_ts_config.flux3_fp8_gemm_config(
            torch.empty((1024, 3072), device="meta"),
            torch.empty((9216, 3072), device="meta"),
            "linear",
        )
        is None
    )
    assert (
        flux3_prims_ts_config.flux3_nvfp4_gemm_config(torch.device("cuda:0"), epilogue="linear", k=3072, n=9216, m=1024)
        is None
    )


def test_autotune_env_requires_a_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(flux3_prims_ts_config.AUTOTUNE_ENV, "sometimes")

    with pytest.raises(ValueError, match="must be a boolean"):
        flux3_prims_ts_config.prims_ts_autotune_enabled()


def test_fp8_measured_config_selects_shape_specific_tactic(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.delenv(flux3_prims_ts_config.AUTOTUNE_ENV, raising=False)
    monkeypatch.setattr(flux3_prims_ts_config, "_config_type", lambda: _Config)
    monkeypatch.setattr(
        flux3_prims_ts_config.current_platform,
        "get_device_capability",
        lambda _device_index: SimpleNamespace(to_int=lambda: 100),
    )

    config = flux3_prims_ts_config.flux3_fp8_gemm_config(
        torch.empty((8192, 3072), device="meta"),
        torch.empty((9216, 3072), device="meta"),
        "linear",
    )

    assert config.kwargs["operand_format"] == "fp8_e4m3"
    assert config.kwargs["cluster_shape"] == (2, 2, 1)
    assert config.kwargs["tile_k"] == 128
    assert config.kwargs["scheduler"] == "clc_dynamic"
    assert config.kwargs["ab_stages"] == 5


def test_autotune_context_persists_cache_by_model_device_and_rank(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from contextlib import nullcontext

    calls = []
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    autotuner = ModuleType("flashinfer.autotuner")

    def _autotune(enabled, **kwargs):
        calls.append((enabled, kwargs))
        return nullcontext()

    autotuner.autotune = _autotune
    flashinfer.autotuner = autotuner
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv(flux3_prims_ts_config.TUNING_BUCKETS_ENV, "1024, 8192")
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    context = flux3_prims_ts_config.flux3_prims_ts_autotune_context("model-id")

    assert context is not None
    enabled, kwargs = calls[0]
    assert enabled is True
    assert kwargs["tuning_buckets"] == (1024, 8192)
    assert kwargs["round_up"] is True
    cache_path = Path(kwargs["cache"])
    assert cache_path.name == "rank3.json"
    assert len(cache_path.parent.name) == 16
    assert cache_path.parent.parent.name == "NVIDIA_B200"
    assert cache_path.exists() is False

    context = flux3_prims_ts_config.flux3_prims_ts_autotune_context("model-id", tune_mode=False)

    assert context is not None
    enabled, kwargs = calls[1]
    assert enabled is False
    assert kwargs["cache"] == str(cache_path)
    assert kwargs["tuning_buckets"] == (1024, 8192)
    assert kwargs["round_up"] is True


def test_nvfp4_activation_quantization_matches_flashinfer_cute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = object()
    calls = []
    flashinfer = ModuleType("flashinfer")
    flashinfer.SfLayout = SimpleNamespace(layout_128x4=layout)

    def _quantize(activation, global_scale, **kwargs):
        calls.append((activation, global_scale, kwargs))
        return torch.empty((4, 8), dtype=torch.uint8), torch.empty((4, 2), dtype=torch.float8_e4m3fn)

    flashinfer.nvfp4_quantize = _quantize
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    activation = torch.randn((4, 16), dtype=torch.bfloat16, requires_grad=True)
    global_scale_inv = torch.ones(1, dtype=torch.float32, requires_grad=True)

    packed, scales = prims_ts_nvfp4._nvfp4_quantize_impl(activation, global_scale_inv)

    passed_activation, passed_scale, kwargs = calls[0]
    assert passed_activation.requires_grad is False
    assert passed_scale.requires_grad is False
    assert kwargs == {"sfLayout": layout, "do_shuffle": False, "backend": "cute-dsl"}
    assert packed.dtype is torch.uint8
    assert scales.dtype is torch.uint8
    assert scales.is_contiguous()


def test_nvfp4_quantize_op_passes_the_owner_scale_as_a_host_float(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = _nvfp4_layer(object(), input_global_scale=0.125)
    prims_ts_nvfp4.register_prims_ts_nvfp4_layer(layer, config_provider="flux3")
    prepare_nvfp4_layer(layer, "linear")
    calls = []
    monkeypatch.setattr(flux3_prims_ts, "nvfp4_quantize_128x4", lambda x, scale: calls.append((x, scale)) or (x, x))
    activation = torch.empty((4, layer.input_size_per_partition), dtype=torch.bfloat16)

    flux3_prims_ts._flux3_nvfp4_quantize_impl(activation, layer._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER])

    assert calls[0][0] is activation
    assert type(calls[0][1]) is float
    assert calls[0][1] == layer._buffers[flux3_prims_ts._NVFP4_INPUT_SCALE_INV_BUFFER].item() == 8.0


def test_nvfp4_quantize_accepts_a_host_float_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    flashinfer = ModuleType("flashinfer")
    flashinfer.SfLayout = SimpleNamespace(layout_128x4=object())
    flashinfer.nvfp4_quantize = lambda activation, global_scale, **kwargs: calls.append(global_scale) or (
        torch.empty((4, 8), dtype=torch.uint8),
        torch.empty((4, 2), dtype=torch.float8_e4m3fn),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    prims_ts_nvfp4.nvfp4_quantize_128x4(torch.randn((4, 16), dtype=torch.bfloat16), 0.5)

    assert calls == [0.5]


def test_nvfp4_row_linear_accepts_packed_swiglu_without_requantizing(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = nn.Module()
    layer.register_buffer(flux3_prims_ts._NVFP4_WEIGHT_BUFFER, torch.empty((16, 8), dtype=torch.uint8))
    layer.register_buffer(flux3_prims_ts._NVFP4_SCALE_BUFFER, torch.empty((16,), dtype=torch.uint8))
    layer.register_buffer(flux3_prims_ts._NVFP4_INPUT_SCALE_INV_BUFFER, torch.tensor([4.0], dtype=torch.float32))
    packed = torch.empty((4, 8), dtype=torch.uint8)
    scales = torch.empty((4,), dtype=torch.uint8)
    expected = torch.empty((4, 16), dtype=torch.bfloat16)
    calls = []

    def _linear_op(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(flux3_prims_ts, "_FLUX3_NVFP4_LINEAR_OP", _linear_op)
    method = flux3_prims_ts._PrimsTSNvFp4LinearMethod(_PassthroughMethod(), "linear")

    output = method.apply(layer, (packed, scales))

    assert output is expected
    args = calls[0]
    assert args[0] is packed and args[1] is scales
    assert args[2] is layer._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER]
    assert args[4] is True  # packed input: unit global scale inside the op


def test_nvfp4_linear_uses_checkpoint_input_scale_for_tensor_input(monkeypatch: pytest.MonkeyPatch) -> None:
    quant_config = object()
    layer = _nvfp4_layer(quant_config, input_global_scale=0.125)
    prims_ts_nvfp4.register_prims_ts_nvfp4_layer(layer, config_provider="flux3")
    prepare_nvfp4_layer(layer, "linear")
    hidden_states = torch.empty((2, 3, layer.input_size_per_partition), dtype=torch.bfloat16)
    quantized = (torch.empty(6, 1, dtype=torch.uint8), torch.empty(6, dtype=torch.uint8))
    expected = torch.empty((6, layer.output_size_per_partition), dtype=torch.bfloat16)
    quantize_calls, op_calls, impl_calls = [], [], []

    def _quantize(x, scale_inv):
        quantize_calls.append((x, scale_inv))
        return quantized

    def _linear_op(*args):
        op_calls.append(args)
        return flux3_prims_ts._flux3_nvfp4_linear_impl(*args)

    def _impl(*args):
        impl_calls.append(args)
        return expected

    monkeypatch.setattr(flux3_prims_ts, "_FLUX3_NVFP4_QUANTIZE_OP", _quantize)
    monkeypatch.setattr(flux3_prims_ts, "_FLUX3_NVFP4_LINEAR_OP", _linear_op)
    monkeypatch.setattr(flux3_prims_ts, "_nvfp4_linear_impl", _impl)

    output = flux3_prims_ts._nvfp4_linear(layer, hidden_states)

    assert output.shape == (2, 3, layer.output_size_per_partition)
    torch.testing.assert_close(layer._buffers[flux3_prims_ts._NVFP4_INPUT_SCALE_INV_BUFFER], torch.tensor([8.0]))
    assert quantize_calls[0][0].shape == (6, layer.input_size_per_partition)
    assert quantize_calls[0][1] is layer._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER]
    assert op_calls[0][4] is False
    # The op resolves the checkpoint scales from the registered weight.
    assert impl_calls[0][4] == 0.125
    assert impl_calls[0][5] == 0.5
    assert impl_calls[0][6] == prims_ts_nvfp4.get_prims_ts_nvfp4_cache_id(layer)


def test_nvfp4_adapter_prepares_fused_qkv_with_distinct_projection_scales() -> None:
    quant_config = object()
    qkv = _nvfp4_layer(quant_config, output_size=1536)
    qkv.weight_scale_2 = nn.Parameter(torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32), requires_grad=False)

    class _FusedNvFp4Method(_FakeNvFp4Method):
        def process_weights_after_loading(self, layer: nn.Module) -> None:
            self.processed = True
            torch.testing.assert_close(layer.weight_scale_2, torch.ones(3))
            layer.weight_global_scale = nn.Parameter(layer.weight_scale_2[:1].detach().clone(), requires_grad=False)
            layer.input_global_scale = nn.Parameter(torch.tensor([0.125], dtype=torch.float32), requires_grad=False)
            layer.input_global_scale_inv = nn.Parameter(torch.tensor([8.0], dtype=torch.float32), requires_grad=False)
            del layer.weight_scale_2

    qkv.quant_method = _FusedNvFp4Method(quant_config)
    qkv.num_heads = 4
    qkv.num_kv_heads = 4
    block = SimpleNamespace(
        qkv=qkv,
        attn_out=_nvfp4_layer(quant_config),
        mlp_in=_nvfp4_layer(quant_config, output_size=1024),
        mlp_out=_nvfp4_layer(quant_config),
        norm=SimpleNamespace(query_norm=object()),
        head_dim=128,
        num_heads_local=4,
        num_kv_heads_local=4,
    )
    adapter = Flux3PrimsTSKernelAdapter(
        lambda q, k, _v, _norm, _rope: (q, k),
        prims_ts_config=Flux3PrimsTSConfig(mode=Flux3PrimsTSMode.REQUIRED),
        prims_ts_support=Flux3PrimsTSSupport(True),
        quant_config=quant_config,
        quant_format="nvfp4",
    )

    plan = adapter.configure_block(block, "content_mode_blocks.image.0")
    qkv.quant_method.process_weights_after_loading(qkv)

    assert plan.qkv_qknorm_rope is True
    assert plan.mlp_swiglu is True
    assert plan.mlp_output is True
    for owner in (qkv, block.attn_out, block.mlp_in, block.mlp_out):
        assert prims_ts_nvfp4.get_prims_ts_nvfp4_cache_id(owner) == getattr(owner, prims_ts_nvfp4._CACHE_ID_ATTR)
    assert qkv._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER].shape == (1536, 128)
    torch.testing.assert_close(
        qkv._buffers[flux3_prims_ts._NVFP4_QKV_SCALE_BUFFER],
        torch.tensor([0.03125, 0.0625, 0.125]),
    )
    assert qkv._flux3_prims_ts_nvfp4_weight_global_scale == 1.0
    assert qkv._flux3_prims_ts_nvfp4_input_global_scale == 0.125
    torch.testing.assert_close(qkv._buffers[flux3_prims_ts._NVFP4_INPUT_SCALE_INV_BUFFER], torch.tensor([8.0]))
    # The checkpoint method (LoRA-wrapped blocks) dequantizes with the shared largest
    # multiplier; its output is corrected per projection.
    out = qkv.quant_method.apply(qkv, torch.ones(2, 1536, dtype=torch.bfloat16))
    assert out.dtype is torch.bfloat16
    torch.testing.assert_close(out[0], torch.tensor([0.25, 0.5, 1.0]).repeat_interleave(512).bfloat16())


def test_nvfp4_operands_follow_offloaded_weight_storage() -> None:
    """Layerwise offload swaps weight storage in place; lookups follow the new address."""
    quant_config = object()
    layers = [_nvfp4_layer(quant_config, input_global_scale=s) for s in (0.125, 0.25)]
    for layer in layers:
        prims_ts_nvfp4.register_prims_ts_nvfp4_layer(layer, config_provider="flux3")
        prepare_nvfp4_layer(layer, "linear")
    first, second = (layer._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER] for layer in layers)
    assert flux3_prims_ts._nvfp4_operands(first)[1] == 0.125

    # Swap storages: each weight now lives at the other one's old address.
    first_storage, second_storage = first.clone(), second.clone()
    first.set_(second_storage)
    second.set_(first_storage)

    assert flux3_prims_ts._nvfp4_operands(first)[1] == 0.125
    assert flux3_prims_ts._nvfp4_operands(second)[1] == 0.25
    with pytest.raises(RuntimeError, match="not registered"):
        flux3_prims_ts._nvfp4_operands(torch.empty_like(first))


def test_nvfp4_input_is_quantized_once_with_the_qkv_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    quant_config = object()
    block = SimpleNamespace(
        qkv=_nvfp4_layer(quant_config, input_global_scale=0.125),
        mlp_in=_nvfp4_layer(quant_config, input_global_scale=0.25),
        kernel_plan=Flux3BlockKernelPlan(qkv_linear=True, qkv_qknorm_rope=True, mlp_swiglu=True),
    )
    for layer in (block.qkv, block.mlp_in):
        prepare_nvfp4_layer(layer, "linear")
    adapter = _adapter()
    adapter.quant_format = "nvfp4"
    hidden_states = torch.empty(1, 4, 256)
    packed = (torch.empty(4, 128, dtype=torch.uint8), torch.empty(512, dtype=torch.uint8))
    calls = []

    def quantize(x, scale_inv):
        calls.append((x, scale_inv))
        return packed

    monkeypatch.setattr(flux3_prims_ts, "_FLUX3_NVFP4_QUANTIZE_OP", quantize)
    states, activation = adapter.prepare_input(block, hidden_states)

    assert states is hidden_states
    assert activation is packed
    assert len(calls) == 1
    assert calls[0][1] is block.qkv._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER]


@pytest.mark.parametrize("mlp_scale, requantized", [(0.125, False), (0.25, True)])
def test_nvfp4_swiglu_requantizes_only_for_a_different_scale(
    monkeypatch: pytest.MonkeyPatch, mlp_scale: float, requantized: bool
) -> None:
    quant_config = object()
    qkv = _nvfp4_layer(quant_config, input_global_scale=0.125)
    mlp_in = _nvfp4_layer(quant_config, input_global_scale=mlp_scale)
    for layer, layout in ((qkv, "linear"), (mlp_in, "swiglu")):
        prims_ts_nvfp4.register_prims_ts_nvfp4_layer(layer, config_provider="flux3")
        prepare_nvfp4_layer(layer, layout)
    shared = (torch.empty(4, 128, dtype=torch.uint8), torch.empty(512, dtype=torch.uint8))
    own = (torch.empty(4, 128, dtype=torch.uint8), torch.empty(512, dtype=torch.uint8))
    hidden_states = torch.empty(4, 256)
    quantize_calls, swiglu_calls = [], []

    def _quantize(x, scale_inv):
        quantize_calls.append((x, scale_inv))
        return own

    def _swiglu(*args):
        swiglu_calls.append(args)
        return own

    monkeypatch.setattr(flux3_prims_ts, "nvfp4_quantize_128x4", _quantize)
    monkeypatch.setattr(flux3_prims_ts, "_nvfp4_swiglu_impl", _swiglu)
    flux3_prims_ts._flux3_nvfp4_swiglu_impl(
        *shared,
        hidden_states,
        qkv._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER],
        mlp_in._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER],
        mlp_in._buffers[flux3_prims_ts._NVFP4_SCALE_BUFFER],
    )

    assert len(quantize_calls) == int(requantized)
    if requantized:
        # Requantized with the SwiGLU projection's own scale, as a host float.
        assert quantize_calls[0][1] == 1.0 / mlp_scale
    assert swiglu_calls[0][0] is (own if requantized else shared)[0]
    # The GEMM keeps the SwiGLU projection's own checkpoint scale.
    assert swiglu_calls[0][4] == mlp_scale


def test_nvfp4_mlp_passes_packed_swiglu_output_to_the_down_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    quant_config = object()
    block = SimpleNamespace(
        qkv=_nvfp4_layer(quant_config, input_global_scale=0.125),
        mlp_in=_nvfp4_layer(quant_config, input_global_scale=0.125),
        kernel_plan=Flux3BlockKernelPlan(qkv_linear=True, qkv_qknorm_rope=True, mlp_swiglu=True, mlp_output=True),
    )
    packed = (torch.empty(4, 128, dtype=torch.uint8), torch.empty(512, dtype=torch.uint8))
    hidden_states = torch.empty(1, 4, 256)
    swiglu_calls = []

    def _swiglu(*args):
        swiglu_calls.append(args)
        return packed

    block.mlp_out = lambda x: torch.empty(4, 256)
    monkeypatch.setattr(flux3_prims_ts, "_FLUX3_NVFP4_SWIGLU_OP", _swiglu)
    for layer in (block.qkv, block.mlp_in):
        prepare_nvfp4_layer(layer, "linear")
    adapter = _adapter()
    adapter.quant_format = "nvfp4"

    out = adapter.mlp(block, (hidden_states, packed))

    assert out.shape == (1, 4, 256)
    args = swiglu_calls[0]
    assert args[0] is packed[0] and args[1] is packed[1]
    assert args[3] is block.qkv._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER]
    assert args[4] is block.mlp_in._buffers[flux3_prims_ts._NVFP4_WEIGHT_BUFFER]


@pytest.mark.parametrize("quant_format", ["fp8", "nvfp4"])
@pytest.mark.parametrize("wrapped", ["qkv", "mlp_out"])
def test_lora_wrapped_block_calls_its_projection_modules(
    monkeypatch: pytest.MonkeyPatch, quant_format: str, wrapped: str
) -> None:
    class _LoRAWrapper(nn.Module):
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self.out = out

        def forward(self, x):
            calls.append((wrapped, x))
            return (self.out, None) if wrapped == "qkv" else self.out

    calls = []
    monkeypatch.setattr(flux3_prims_ts, "BaseLayerWithLoRA", _LoRAWrapper)

    def _unexpected(*args, **kwargs):
        raise AssertionError("fused PrimsTS path used for a LoRA-wrapped block")

    for name in ("_FLUX3_NVFP4_QUANTIZE_OP", "_FLUX3_NVFP4_SWIGLU_OP", "_FLUX3_NVFP4_QKV_QKNORM_ROPE_OP"):
        monkeypatch.setattr(flux3_prims_ts, name, _unexpected)
    monkeypatch.setattr(flux3_prims_ts, "prims_ts_fp8_quantize", _unexpected)
    hidden_states = torch.randn(1, 4, 256)
    qkv_out = torch.randn(1, 4, 3 * 2 * 128)
    projections = {
        "qkv": lambda x: (qkv_out, None),
        "attn_out": lambda x: x,
        "mlp_in": lambda x: torch.cat([x, x], dim=-1),
        "mlp_out": lambda x: x[..., :256],
    }
    projections[wrapped] = _LoRAWrapper(qkv_out if wrapped == "qkv" else torch.zeros(1, 4, 256))
    block = SimpleNamespace(
        **projections,
        mlp_act=lambda x: x[..., :256],
        norm=object(),
        head_dim=128,
        num_heads_local=2,
        num_kv_heads_local=2,
        kernel_plan=Flux3BlockKernelPlan(
            qkv_linear=True, qkv_qknorm_rope=True, attention_output=True, mlp_swiglu=True, mlp_output=True
        ),
    )
    adapter = _adapter()
    adapter.quant_format = quant_format

    prepared = adapter.prepare_input(block, hidden_states)
    q, k, v = adapter.qkv_qknorm_rope(block, prepared, torch.empty(1, 4, 128))
    mlp_out = adapter.mlp(block, prepared)

    assert prepared[0] is hidden_states and prepared[1] is None
    assert q.shape == k.shape == v.shape == (1, 4, 2, 128)
    torch.testing.assert_close(torch.cat([q, k, v], dim=2).flatten(2), qkv_out)
    assert mlp_out.shape == (1, 4, 256)
    assert [name for name, _ in calls] == [wrapped]


def test_nvfp4_prims_ts_rejects_unfused_qkv() -> None:
    quant_config = object()
    block = SimpleNamespace(
        qkv=SimpleNamespace(
            q_proj=_nvfp4_layer(quant_config),
            k_proj=_nvfp4_layer(quant_config),
            v_proj=_nvfp4_layer(quant_config),
        ),
        attn_out=_nvfp4_layer(quant_config),
        mlp_in=_nvfp4_layer(quant_config, output_size=1024),
        mlp_out=_nvfp4_layer(quant_config),
        norm=SimpleNamespace(query_norm=object()),
        head_dim=128,
        num_heads_local=4,
        num_kv_heads_local=4,
    )
    adapter = Flux3PrimsTSKernelAdapter(
        lambda q, k, _v, _norm, _rope: (q, k),
        prims_ts_config=Flux3PrimsTSConfig(mode=Flux3PrimsTSMode.REQUIRED),
        prims_ts_support=Flux3PrimsTSSupport(True),
        quant_config=quant_config,
        quant_format="nvfp4",
    )

    with pytest.raises(RuntimeError, match="requires fused QKV"):
        adapter.configure_block(block, "content_mode_blocks.image.0")


def test_nvfp4_adapter_keeps_swiglu_unfused_without_prims_ts_down_projection() -> None:
    quant_config = object()
    block = SimpleNamespace(
        qkv=_nvfp4_layer(quant_config, output_size=1536),
        attn_out=_nvfp4_layer(quant_config),
        mlp_in=_nvfp4_layer(quant_config, output_size=1024),
        mlp_out=_layer(method=_OtherMethod()),
        norm=SimpleNamespace(query_norm=object()),
        head_dim=128,
        num_heads_local=4,
        num_kv_heads_local=4,
    )
    adapter = Flux3PrimsTSKernelAdapter(
        lambda q, k, _v, _norm, _rope: (q, k),
        prims_ts_config=Flux3PrimsTSConfig(mode=Flux3PrimsTSMode.REQUIRED),
        prims_ts_support=Flux3PrimsTSSupport(True),
        quant_config=quant_config,
        quant_format="nvfp4",
    )

    plan = adapter.configure_block(block, "content_mode_blocks.image.0")

    assert plan.mlp_swiglu is False
    assert plan.mlp_output is False
    assert isinstance(block.mlp_in.quant_method, _FakeNvFp4Method)
