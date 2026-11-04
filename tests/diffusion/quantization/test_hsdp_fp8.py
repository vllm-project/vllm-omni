# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for HSDP/FSDP2 compatibility with online FP8 quantization."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.quantization.hsdp_fp8 import (
    _build_transposed_get_layer_params,
    prepare_fp8_layers_for_fsdp,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# --- shared test doubles ---


class _ToyKernel:
    """Minimal kernel that mirrors ``FP8ScaledMMLinearKernel``."""

    def __init__(self):
        self.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )


# --- helper to build a toy module simulating online-FP8 post-load state ---


def _make_fp8_toy_module(
    out_features: int = 16,
    in_features: int = 32,
    *,
    quant_method_cls=None,
):
    """Return a module whose weight is ``qweight.t()`` (non-contiguous) and whose
    ``quant_method`` is an ``Fp8LinearMethod`` with a ``_ToyKernel``.
    """
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

    if quant_method_cls is None:
        quant_method_cls = Fp8LinearMethod

    # Create the quant-method instance without calling __init__ (avoids
    # heavyweight upstream config dependencies).  isinstance() still works.
    qm = object.__new__(quant_method_cls)
    if isinstance(qm, ModelOptLinearMethod):
        qm.spec = QuantSpec(weight=kFp8StaticTensorSym, activation=kFp8StaticTensorSym)
        qm.kernel = _ToyKernel()
    else:
        qm.fp8_linear = _ToyKernel()

    # randn does not support float8 on CPU; zeros does and the test only
    # cares about shape/stride, not the actual weight values.
    qweight = torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
    weight = nn.Parameter(qweight.t())  # (in, out) non-contiguous column-major view

    module = nn.Module()
    module.quant_method = qm
    module.weight = weight
    module.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
    # Set input_scale / input_scale_ub to verify _get_layer_params
    # passes them through unchanged.
    module.input_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
    module.input_scale_ub = nn.Parameter(torch.ones(1, dtype=torch.float32))

    return module


# --- tests ---


def test_transposed_get_layer_params(monkeypatch: pytest.MonkeyPatch):
    import types

    kernel = _ToyKernel()

    # original bound method + patch + re-bind, matching prepare_fp8_layers_for_fsdp
    original_bound = kernel._get_layer_params
    patched_func = _build_transposed_get_layer_params(original_bound)
    monkeypatch.setattr(kernel, "_get_layer_params", types.MethodType(patched_func, kernel))

    module = _make_fp8_toy_module(16, 32)
    # Reproduce the storage rewrite from prepare_fp8_layers_for_fsdp:
    # weight was qweight.t()  →  .t() recovers qweight (row-major contiguous).
    module.weight = nn.Parameter(module.weight.data.t(), requires_grad=False)
    module._omni_fp8_fsdp_row_major = True
    assert module.weight.is_contiguous(), "qweight should be row-major contiguous"

    w, w_s, x_s, x_s_ub = kernel._get_layer_params(module)

    # patched method transposes row-major → column-major for Cutlass
    assert w.shape == (32, 16)
    assert not w.is_contiguous()
    # scales are passed through unchanged
    assert w_s is module.weight_scale
    assert x_s is module.input_scale
    assert x_s_ub is module.input_scale_ub


def test_rewrites_non_contiguous_weight():
    module = _make_fp8_toy_module(16, 32)
    assert not module.weight.is_contiguous()

    n_rewritten_layers = prepare_fp8_layers_for_fsdp(module)
    assert n_rewritten_layers == 1
    assert module.weight.is_contiguous()
    assert module.weight.shape == (16, 32)


def test_rewrites_per_tensor_online_fp8_weight():
    """Regression test for vLLM's new online FP8 frontend (#45463)."""
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    module = _make_fp8_toy_module(
        16,
        32,
        quant_method_cls=Fp8PerTensorOnlineLinearMethod,
    )

    n_rewritten_layers = prepare_fp8_layers_for_fsdp(module)

    assert n_rewritten_layers == 1
    assert module.weight.is_contiguous()
    assert module.weight.shape == (16, 32)
    w, *_ = module.quant_method.fp8_linear._get_layer_params(module)
    assert w.shape == (32, 16)
    assert w.stride() == (1, 32)


def test_patched_get_layer_params_returns_transposed_view():
    module = _make_fp8_toy_module(16, 32)
    prepare_fp8_layers_for_fsdp(module)

    kernel = module.quant_method.fp8_linear
    w, w_s, x_s, x_s_ub = kernel._get_layer_params(module)

    # Cutlass expects column-major B: (in, out), stride (1, in)
    assert w.shape == (32, 16)
    assert not w.is_contiguous()
    assert w.stride() == (1, 32)

    # scales pass through unchanged
    assert w_s is module.weight_scale
    assert x_s is module.input_scale
    assert x_s_ub is module.input_scale_ub


def test_contiguous_weights_skipped():
    module = _make_fp8_toy_module(16, 32)
    # make weight contiguous before calling
    module.weight = nn.Parameter(module.weight.data.t().contiguous())
    assert module.weight.is_contiguous()

    n_rewritten_layers = prepare_fp8_layers_for_fsdp(module)
    assert n_rewritten_layers == 0


def test_kernel_patched_only_once_per_instance():
    module = _make_fp8_toy_module(16, 32)
    kernel = module.quant_method.fp8_linear
    original = kernel._get_layer_params

    prepare_fp8_layers_for_fsdp(module)
    first_patched = kernel._get_layer_params

    # Second call with same kernel id must not stack patches
    n_rewritten_layers = prepare_fp8_layers_for_fsdp(module)
    assert n_rewritten_layers == 0
    assert kernel._get_layer_params is first_patched
    assert kernel._get_layer_params is not original


@pytest.mark.parametrize("wrapped", [False, True])
def test_modelopt_maps_preserves_both_weight_views(wrapped):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod

    from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
        Cosmos3MixedPrecisionConfig,
        Cosmos3MixedPrecisionRuntime,
    )
    from vllm_omni.diffusion.models.cosmos3.mixed_precision.runtime import Cosmos3MixedPrecisionLinearMethod
    from vllm_omni.diffusion.models.cosmos3.mixed_precision.strategy import Fp8W8A8W8A16Strategy

    layer = _make_fp8_toy_module(16, 32, quant_method_cls=ModelOptLinearMethod)
    layer.input_size_per_partition = 32
    layer.output_size_per_partition = 16
    values = (torch.arange(512).reshape(16, 32) % 17 - 8).to(torch.float8_e4m3fn)
    layer.weight = nn.Parameter(values.t(), requires_grad=False)
    layer.weight_scale.data.fill_(0.125)
    method = layer.quant_method
    strategy = Fp8W8A8W8A16Strategy()
    if wrapped:
        layer.quant_method = Cosmos3MixedPrecisionLinearMethod(
            method, strategy, Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig()), "gen_layers.0", "generation"
        )
    original_method = layer.quant_method
    dense = strategy.materialize(layer)
    x = torch.arange(64).reshape(2, 32).to(torch.bfloat16) / 32
    expected = strategy.apply_high(layer, x, None)
    assert prepare_fp8_layers_for_fsdp(layer) == 1
    assert layer.quant_method is original_method
    assert layer.weight.is_contiguous()
    torch.testing.assert_close(layer.weight.float(), values.float(), rtol=0, atol=0)
    torch.testing.assert_close(strategy.materialize(layer), dense, rtol=0, atol=0)
    torch.testing.assert_close(strategy.apply_high(layer, x, None), expected, rtol=0, atol=0)
    native_weight, *_ = method.kernel._get_layer_params(layer)
    torch.testing.assert_close(native_weight.float(), values.t().float(), rtol=0, atol=0)
    assert native_weight.stride() == (1, 32)
    assert prepare_fp8_layers_for_fsdp(layer) == 0

    # FSDP swaps parameter objects at gather/reshard boundaries. Neither path
    # may retain the pre-gather tensor or an old dequantized weight.
    layer.weight = nn.Parameter((values.float() * 2).to(torch.float8_e4m3fn), requires_grad=False)
    torch.testing.assert_close(strategy.materialize(layer), dense * 2, rtol=0, atol=0)
    native_weight, *_ = method.kernel._get_layer_params(layer)
    torch.testing.assert_close(native_weight.float(), values.t().float() * 2, rtol=0, atol=0)


def test_shared_kernel_adaptation_is_per_layer_and_idempotent():
    first = _make_fp8_toy_module(16, 32)
    second = _make_fp8_toy_module(16, 32)
    kernel = first.quant_method.fp8_linear
    second.quant_method.fp8_linear = kernel
    original_second = second.weight
    assert prepare_fp8_layers_for_fsdp(first) == 1
    # A shared kernel must not transpose a layer that has not been adapted.
    weight, *_ = kernel._get_layer_params(second)
    assert weight is original_second
    patched = kernel._get_layer_params
    # A later invocation must not stack a second kernel patch.
    assert prepare_fp8_layers_for_fsdp(second) == 1
    assert kernel._get_layer_params is patched
    for layer in (first, second):
        weight, *_ = kernel._get_layer_params(layer)
        assert weight.shape == (32, 16)
        assert weight.stride() == (1, 32)


def test_generic_quant_method_unwrapper_is_explicit_and_single_level():
    from types import SimpleNamespace

    from vllm_omni.diffusion.quantization.hsdp_fp8 import _unwrap_quant_method

    class Wrapper:
        def __init__(self, base):
            self.base = base

        def get_base_quant_method(self):
            return self.base

    base = object()
    wrapper = Wrapper(base)
    assert _unwrap_quant_method(None) is None
    assert _unwrap_quant_method(base) is base
    assert _unwrap_quant_method(wrapper) is base
    assert _unwrap_quant_method(Wrapper(wrapper)) is wrapper
    unrelated = SimpleNamespace(base_method=base)
    assert _unwrap_quant_method(unrelated) is unrelated


def test_fp8_adapter_accepts_generic_quant_method_wrapper():
    from types import SimpleNamespace

    layer = _make_fp8_toy_module(16, 32)
    method = layer.quant_method
    wrapper = SimpleNamespace(get_base_quant_method=lambda: method)
    layer.quant_method = wrapper
    assert prepare_fp8_layers_for_fsdp(layer) == 1
    assert layer.quant_method is wrapper
    assert layer.weight.is_contiguous()
    weight, *_ = method.fp8_linear._get_layer_params(layer)
    assert weight.shape == (32, 16)


def test_fp8_adapter_skips_other_modelopt_specs():
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic, kNvfp4Static

    layer = _make_fp8_toy_module(quant_method_cls=ModelOptLinearMethod)
    layer.quant_method.spec = QuantSpec(weight=kNvfp4Static, activation=kNvfp4Dynamic)
    original_weight = layer.weight
    original_accessor = layer.quant_method.kernel._get_layer_params
    assert prepare_fp8_layers_for_fsdp(layer) == 0
    assert layer.weight is original_weight
    assert layer.quant_method.kernel._get_layer_params == original_accessor
