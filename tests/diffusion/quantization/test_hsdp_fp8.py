# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


def _make_fp8_toy_module(out_features: int = 16, in_features: int = 32):
    """Return a module whose weight is ``qweight.t()`` (non-contiguous) and whose
    ``quant_method`` is an ``Fp8LinearMethod`` with a ``_ToyKernel``.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    # Create the quant-method instance without calling __init__ (avoids
    # heavyweight upstream config dependencies).  isinstance() still works.
    qm = object.__new__(Fp8LinearMethod)
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


def test_transposed_get_layer_params():
    import types

    kernel = _ToyKernel()

    # original bound method + patch + re-bind, matching prepare_fp8_layers_for_fsdp
    original_bound = kernel._get_layer_params
    patched_func = _build_transposed_get_layer_params(original_bound)
    kernel._get_layer_params = types.MethodType(patched_func, kernel)

    module = _make_fp8_toy_module(16, 32)
    # Reproduce the storage rewrite from prepare_fp8_layers_for_fsdp:
    # weight was qweight.t()  →  .t() recovers qweight (row-major contiguous).
    module.weight = nn.Parameter(module.weight.data.t(), requires_grad=False)
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
