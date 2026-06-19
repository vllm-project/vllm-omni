# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HSDP/FSDP2 compatibility for online FP8 quantization.

vllm's ``Fp8OnlineLinearMethod.process_weights_after_loading`` ends with
``layer.weight = qweight.t()`` so that the Cutlass FP8 GEMM kernel sees its
B operand as column-major ``[K, N]`` (the TN layout required by Hopper FP8
``wgmma`` instructions). The resulting tensor is a non-contiguous transpose
view of a ``(out_features, in_features)`` row-major storage.

FSDP2 ``fully_shard`` rejects non-contiguous parameters because dim-0 sharding
on a column-major view cannot be a contiguous memcpy. The two requirements are
fundamentally at odds with the same physical buffer.

This module reconciles them by separating the views: keep the parameter as
the underlying ``(out, in)`` row-major contiguous storage (FSDP-friendly),
and inject the equivalent ``.t()`` at the GEMM call site so the Cutlass
kernel still receives a column-major B (zero-copy stride flip).

The transpose is injected inside ``ScaledMMLinearKernel._get_layer_params``,
which is the single place where ``apply_weights`` reads the weight tensor.
The rest of ``apply_weights`` -- including its ``output_shape = w.shape[1]``
computation and the eventual ``apply_scaled_mm(B=w, ...)`` call -- then
operates on a tensor whose shape ``(K, N)`` and stride ``(1, K)`` match
what the upstream FP8 kernel expects. No other upstream code is overridden,
which keeps us robust against future changes inside ``apply_weights``.
"""

from __future__ import annotations

import types

from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.fp8 import Fp8OnlineLinearMethod

logger = init_logger(__name__)


def _build_transposed_get_layer_params(original_bound_method):
    """Return a ``_get_layer_params`` replacement that transposes ``w``.

    ``original_bound_method`` is the kernel instance's existing
    ``_get_layer_params`` (already bound to ``self``); we close over it and
    apply ``.t()`` to the returned weight, leaving scales untouched.
    """

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = original_bound_method(layer)
        return w.t(), w_s, x_s, x_s_ub

    return _get_layer_params


def prepare_fp8_layers_for_fsdp(model: nn.Module) -> int:
    """Make online-FP8 linear layers in ``model`` FSDP2-compatible.

    For every layer whose ``quant_method`` is an :class:`Fp8OnlineLinearMethod`
    and whose weight is currently a non-contiguous transpose view, this
    function:

    1. Replaces ``layer.weight`` with the underlying ``(out, in)`` row-major
       contiguous storage so FSDP2 ``fully_shard`` accepts it.
    2. Patches the per-layer GEMM kernel's bound ``_get_layer_params`` method
       to return a ``.t()`` view of the weight, so ``apply_weights`` and the
       downstream ``apply_scaled_mm`` continue to see a column-major
       ``(in, out)`` B with zero copies.

    Layers whose weight is already contiguous (e.g. Marlin FP8, offline-
    quantized checkpoints) or that use a different quant method are skipped.

    Returns:
        Number of layers rewritten.
    """
    n_patched = 0
    patched_kernel_ids: set[int] = set()
    for module in model.modules():
        qm = getattr(module, "quant_method", None)
        if not isinstance(qm, Fp8OnlineLinearMethod):
            continue

        weight = getattr(module, "weight", None)
        if weight is None or weight.is_contiguous():
            continue

        # ``weight`` here is ``qweight.t()`` (a non-contiguous (in, out) view
        # of an (out, in) row-major qweight storage). ``weight.t()`` recovers
        # that storage; ``.contiguous()`` is a zero-copy alias when the source
        # is already row-major contiguous, but defensively materializes if
        # upstream ever stacks another view.
        contig = weight.data.t().contiguous()
        new_param = nn.Parameter(contig, requires_grad=False)
        module.weight = new_param

        kernel = qm.fp8_linear
        # Each linear layer is expected to have its own kernel instance, but
        # guard against shared instances to avoid stacking multiple ``.t()``
        # patches on the same object (which would compose into identity).
        if id(kernel) not in patched_kernel_ids:
            original_get = kernel._get_layer_params
            kernel._get_layer_params = types.MethodType(_build_transposed_get_layer_params(original_get), kernel)
            patched_kernel_ids.add(id(kernel))

        n_patched += 1

    if n_patched:
        logger.info(
            "Rewrote %d FP8 linear layer(s) into FSDP2-compatible storage; "
            "transpose to column-major B is now applied at GEMM time.",
            n_patched,
        )
    return n_patched
