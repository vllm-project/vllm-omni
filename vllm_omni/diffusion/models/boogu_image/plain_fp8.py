# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Plain-tensor FP8 weight-only representation for offloaded Boogu-Image linears.
#
# Why this file exists
# --------------------
# Official Boogu-Image FP8 checkpoints (Boogu-Image-0.1-Base-fp8) serialize
# their DiT weights as torchao ``Float8Tensor`` subclass instances.  vLLM's
# torchao integration (``TorchAOLinearMethod`` with
# ``is_checkpoint_torchao_serialized=True``) keeps that subclass in place at
# runtime: ``convert_to_packed_tensor_based_on_current_hardware`` only packs
# ``Int4Tensor``, so the loaded ``layer.weight`` remains a Float8Tensor whose
# forward simply dequantizes and runs ``F.linear``.
#
# That subclass is fine for plain inference, but the offloaders
# (model-level and layerwise) rely on *plain tensor storage semantics*:
# ``flatten_physical_storage`` / ``clear_tensor_storage`` /
# ``set_tensor_storage`` reshape, slice, reassign ``.data`` and move between
# devices.  torchao's Float8Tensor dispatch table does not implement those
# aten primitives, so CPU offload + FP8 checkpoint fails at weight staging.
#
# Approach (model-side repack)
# ----------------------------
# A serialized Float8Tensor is fully described by two *plain* tensors
# (``__tensor_flatten__`` -> qdata + scale) plus small metadata:
#
#   qdata     fp8_e4m3fn [out, in]   (ordinary tensor, per-row block layout)
#   scale     fp32       [out, 1]    (ordinary tensor, row scales)
#   block_size == [1, in]            (per-row fine-grained quantization)
#
# We unpack each quantized linear into ``weight`` (qdata) + ``weight_scale``
# ordinary parameters and swap in :class:`PlainFloat8LinearMethod`, whose
# forward dequantizes with the exact same arithmetic as torchao's
# ``Float8Tensor.dequantize()`` and falls back to ``F.linear``.  The offload
# framework then treats every parameter as a plain tensor and needs zero
# changes.  Numerical equivalence with the torchao runtime path is verified
# in tests (bit-identical dequantize arithmetic, byte-identical outputs).
#
# Only the per-row block layout (``block_size == [1, in]``) is handled; any
# other block layout is left untouched so it fails loudly in the offloader
# instead of silently computing the wrong values.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

# Metadata carried by a torchao Float8Tensor that has no meaning once the
# tensor is unpacked into plain qdata + scale parameters.
_FLOAT8_TENSOR_METADATA_FIELDS = frozenset(
    {
        "qdata",
        "scale",
        "block_size",
        "mm_config",
        "act_quant_kwargs",
        "kernel_preference",
    }
)


class PlainFloat8LinearMethod:
    """Weight-only FP8 matmul over plain ``(qdata, scale)`` parameters.

    Consumes ``layer.weight`` (fp8_e4m3fn) and ``layer.weight_scale``
    (fp32, per-row) plus the layer attributes recorded by
    :func:`unpack_float8_linear` (``weight_dtype`` / ``weight_scale_shape``)
    and dequantizes with the same arithmetic torchao's ``Float8Tensor``
    runtime path uses (qdata -> fp32, broadcast-multiply by scale, cast to
    the original high-precision dtype), then runs ``F.linear``.
    """

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        scale = layer.weight_scale
        output_dtype = layer.weight_dtype
        # torchao's Float8Tensor.dequantize(): qdata -> fp32, expand scale to
        # the qdata shape, multiply, cast to the original dtype.
        high_prec = (weight.to(torch.float32) * scale).to(output_dtype)
        return F.linear(x, high_prec, bias)


def _is_torchao_float8_tensor(value: Any) -> bool:
    """True for torchao Float8Tensor instances without importing torchao.

    The class lives under ``torchao.quantization.quantize_.workflows.float8``
    (version-dependent); duck-typing on the flatten layout is what the
    serializer itself guarantees (``__tensor_flatten__`` -> qdata + scale).
    """
    if not isinstance(value, torch.Tensor):
        return False
    if type(value).__name__ != "Float8Tensor":
        return False
    if not type(value).__module__.startswith("torchao"):
        return False
    if not hasattr(value, "qdata") or not hasattr(value, "scale"):
        return False
    if not hasattr(value, "block_size"):
        return False
    return True


def _is_per_row_block(weight: Any) -> bool:
    block_size = getattr(weight, "block_size", None)
    # Per-row fine-grained layout: one scale per output row,
    # block_size == [1, in_features].
    if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
        return False
    if block_size[0] != 1 or block_size[1] != weight.shape[1]:
        return False
    return True


def unpack_float8_linear(layer: nn.Module) -> bool:
    """Unpack one quantized linear's Float8Tensor weight into plain tensors.

    Mutates ``layer`` in place: replaces ``weight`` (Float8Tensor subclass)
    with an ordinary fp8 parameter plus a new ordinary fp32 ``weight_scale``
    parameter, and swaps ``quant_method`` for :class:`PlainFloat8LinearMethod`.

    Returns True when the layer was repacked, False when it was already in
    plain form or carries an unsupported (non per-row) block layout.
    """
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return False
    if isinstance(quant_method, PlainFloat8LinearMethod):
        return False  # idempotent: already repacked
    weight = getattr(layer, "weight", None)
    if not _is_torchao_float8_tensor(weight):
        return False
    if not _is_per_row_block(weight):
        logger.warning(
            "Skip Float8Tensor unpack for %s: unsupported block_size=%s (only per-row [1, in] layout is supported)",
            getattr(layer, "prefix", "") or type(layer).__name__,
            getattr(weight, "block_size", None),
        )
        return False

    qdata = weight.qdata.detach()
    scale = weight.scale.detach()
    original_dtype = weight.dtype

    # Preserve loader-facing attributes (input_dim / output_dim / ...) that
    # set_weight_attrs attached to the subclass tensor.
    carried_attrs: dict[str, Any] = {}
    for name, value in vars(weight).items():
        if name not in _FLOAT8_TENSOR_METADATA_FIELDS and not name.startswith("_"):
            carried_attrs[name] = value

    weight_param = nn.Parameter(qdata, requires_grad=False)
    scale_param = nn.Parameter(scale, requires_grad=False)
    for name, value in carried_attrs.items():
        setattr(weight_param, name, value)
    # Record the scale layout shape ([out, 1]) so a future kernel-based
    # matmul path can interpret weight_scale without touching the subclass.
    setattr(weight_param, "weight_scale_shape", tuple(scale.shape))
    setattr(weight_param, "weight_dtype", original_dtype)

    layer.register_parameter("weight", weight_param)
    layer.register_parameter("weight_scale", scale_param)
    layer.weight_dtype = original_dtype  # type: ignore[attr-defined]
    layer.quant_method = PlainFloat8LinearMethod()  # type: ignore[attr-defined]
    return True


def repack_torchao_float8_linears(model: nn.Module) -> int:
    """Unpack every torchao Float8Tensor linear under ``model``.

    Returns the number of layers repacked.  Call after checkpoint weights
    have been fully loaded and before offload backends stage parameters
    (which requires plain storage semantics).
    """
    repacked = 0
    for layer in model.modules():
        if unpack_float8_linear(layer):
            repacked += 1
    if repacked:
        logger.info(
            "Unpacked %d torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters",
            repacked,
        )
    return repacked
