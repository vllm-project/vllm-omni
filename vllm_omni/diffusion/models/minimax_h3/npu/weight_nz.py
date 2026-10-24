# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pre-convert MiniMax-H3 DiT linear weights from ND to Ascend FRACTAL_NZ.

Setting::

    VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ=1

casts every eligible DiT linear weight into the ``ACL_FORMAT_FRACTAL_NZ``
layout once, right after the checkpoint has been loaded and the DiT is on the
device, so each denoising step's matmul reads an NZ weight instead of paying a
transdata on the weight side every call.

``npu_format_cast`` only rewrites the storage layout; the values are untouched,
so the matmul result is bit-identical to the ND path (measured maxdiff ``0.0``
for both bf16 and int8 on Ascend910).

Two regimes, and they behave very differently:

* **Quantized (int8)** -- uniformly better. The four DiT linears measured
  0.834 / 0.865 / 0.826 / 0.861 against the real ``npu_quant_matmul`` call, all
  bit-identical, with no shape regressing.
* **Dense bf16** -- shape dependent, and it nets out flat: 0.764x on out_proj
  and 0.902x on qkv, but 1.023x on fc1 and 1.144x on fc2.

The switch is opt-in and off by default.

Only the DiT is touched: the text encoder and the VAEs keep their ND weights.

Two prerequisites and one hazard are handled here:

* ``torch.npu.config.allow_internal_format`` must be ``True``. ``NPUPlatform``
  sets it for every NPU deployment (``platforms/npu/platform.py``); with it
  ``False``, ``npu_format_cast`` silently returns an ND tensor, so the cast is
  verified afterwards rather than assumed.
* The load-time sweep only takes unquantized linears: a quantized layer's
  method owns its weight layout, and mxfp4 casts its own weights to NZ. The
  offload path has no such guard and instead keys off the dtype set, which is
  why int8 reaches it there and not here.
* The offload path has two mechanisms (``NZ_MODE_ENV``). The default one
  permutes the checkpoint bytes on the host and re-interprets the uploaded
  buffer, so no format op runs on the upload path at all -- see the reactor
  section below. It covers bf16/fp16 and int8, whose fractal blocks are 16x16
  and 16x32 respectively.
* Offloaded DiTs cannot be cast here. Every offload backend relocates the
  parameters after this hook -- the layerwise ones flatten them to pinned host
  memory, the sequential one moves whole modules -- and an internal-format
  tensor crossing that boundary transdata's back to ND, while the storage
  descriptors only record logical strides and so would not notice. The one
  layer that can still work is the layerwise upload path, which re-points the
  parameters at a freshly filled device buffer every block: it calls
  ``maybe_cast_weight_to_nz`` there instead, at the price of one ND->NZ
  conversion per weight per upload. ``weight_nz_enabled`` is the shared
  opt-in, and ``nz_cast_takes_effect`` lets that path fail loudly rather than
  silently measure an unchanged baseline.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

logger = init_logger(__name__)

WEIGHT_NZ_ENV = "VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ"

# Selects how a weight becomes NZ.
#
#   reactor (default) -- permute the checkpoint bytes on the host, carry them
#       through the offload transport unchanged, and re-interpret the buffer
#       slot as FRACTAL_NZ at re-point time. No device-side format op exists on
#       the hot path, which is the entire point: the ND->NZ transdata an
#       offload upload used to emit contends with sparse attention and degrades
#       it.
#   cast -- the previous behaviour, one ``empty_with_format`` + ``copy_`` (a
#       transdata) per weight per upload. Kept as the fallback when the
#       pointer-backed storage constructors are unavailable, and for any weight
#       the reactor declines (a shape whose padded fractal footprint would not
#       fit the slot the ND layout reserved).
NZ_MODE_ENV = "VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ_MODE"

# ACL_FORMAT_FRACTAL_NZ, the format id ``torch_npu.npu_format_cast`` takes.
_ACL_FORMAT_FRACTAL_NZ = 29

# Fractal block size per dtype, as (N0, K0), for the host-side permutation.
# Measured against ``npu_format_cast`` on Ascend910: 16x16 for the 2-byte
# dtypes, 16x32 for int8. The paddings differ too -- bf16 rounds both axes to
# 16, int8 rounds N to 16 and K to 32 -- which is why this is a pair and not a
# single width.
_NZ_BLOCK = {
    torch.bfloat16: (16, 16),
    torch.float16: (16, 16),
    torch.int8: (16, 32),
}

# FRACTAL_NZ is a fractal layout (16x16 blocks; 16 elements per block row for
# bf16/fp16, 32 for int8). fp32 has no NZ matmul path at all.
#
# int8 is included because that is where the win actually is: measured on
# Ascend910 with the real npu_quant_matmul call (dynamic-quant activations,
# per-channel weight scale, bf16 output), the four DiT linears come out
# bit-identical and 13-17% faster on an NZ weight -- 0.834 / 0.865 / 0.826 /
# 0.861 -- uniformly, with no shape regressing. The bf16 dense path, by
# contrast, swings between 0.76x and 1.14x and nets out flat.
#
# fp8 is deliberately NOT here yet: its online methods pack their own layout
# (mxfp4 casts its weights itself), and no fp8 kernel has been checked against
# an NZ weight on this stack.
_NZ_DTYPES = (torch.bfloat16, torch.float16, torch.int8)


def _weight_nz_env_enabled() -> bool:
    return os.environ.get(WEIGHT_NZ_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def weight_nz_enabled() -> bool:
    """Public form of the opt-in, for callers outside this module."""
    return _weight_nz_env_enabled()


def maybe_cast_weight_to_nz(weight: torch.Tensor) -> torch.Tensor | None:
    """Cast one weight to FRACTAL_NZ, or return ``None`` when it should stay ND.

    The eligibility filter is deliberately shape/dtype-only: the layerwise
    offload path re-points parameters from a flattened device buffer and has no
    module handle, so it cannot consult ``quant_method``. The dtype set carries
    that decision instead -- see ``_NZ_DTYPES`` -- and the 1-D bias/scale
    tensors that travel in the same buffer fall out on the rank check.

    Quantized weights need no special handling here: ``process_weights_after_loading``
    has already put them in the layout the kernel wants (int8, ``(K, N)``), so
    the NZ cast applies to that tensor exactly as it does to a dense one.

    No readback check here: this runs once per weight per offload upload, and
    the caller verifies the cast takes effect once at setup instead.
    """
    if weight.dim() != 2 or weight.dtype not in _NZ_DTYPES or not _is_npu_weight(weight):
        return None
    import torch_npu

    try:
        # Allocate the NZ destination and copy into it, rather than
        # ``npu_format_cast(weight.contiguous(), 29)``. The cast silently
        # returns an ND tensor when the input does not own its storage, which
        # is exactly the layerwise offload case (the weight is an as_strided
        # view over a shared device buffer), and ``.contiguous()`` does not
        # help: an already-contiguous view is returned unchanged and is still
        # non-owning. Measured on Ascend910 -- the cast then costs a full
        # copy + allocation and produces ND, so the matmul sees no NZ at all.
        out = torch_npu.empty_with_format(
            tuple(weight.shape),
            dtype=weight.dtype,
            device=weight.device,
            acl_format=_ACL_FORMAT_FRACTAL_NZ,
        )
        out.copy_(weight)
        return out
    except Exception as exc:
        logger.warning("MiniMax-H3 weight NZ: could not transpose %s, keeping ND: %s", tuple(weight.shape), exc)
        return None


# --------------------------------------------------------------------------- #
#  Reactor: pre-permuted weights carried through the offload transport        #
# --------------------------------------------------------------------------- #
#
# ``maybe_cast_weight_to_nz`` above pays a device-side transdata per upload
# because the offload buffer holds ND bytes. The alternative is to permute once
# on the host, ship the permuted bytes, and only *reinterpret* the buffer at
# re-point time. That needs a tensor whose storage descriptor says FRACTAL_NZ
# while pointing at a slice of somebody else's buffer -- which is what the two
# private constructors below do. They are not documented, but they are not a
# hack either: torch_npu rebuilds tensors out of its own ACLGraph memory pool
# with exactly this pair (``dynamo/npugraph_ex/_acl_concrete_graph/utils.py``).
#
# Measured on Ascend910 (2026-10-12): the permuted bytes are byte-identical to
# ``npu_format_cast(w, 29)`` for every shape measured, the matmul result is
# bit-identical to both the ND and the device-cast path, and a profile of the
# hot path shows ``trans_TransData_0`` dropping to zero while the NZ matmul
# (``aclnnMatmulWeightNz``) is still selected.


def nz_reactor_available() -> bool:
    """Whether this torch_npu exposes the pointer-backed storage constructors."""
    try:
        import torch_npu

        return hasattr(torch_npu._C, "_construct_storage_from_data_pointer") and hasattr(
            torch_npu._C, "_construct_NPU_Tensor_From_Storage_And_Metadata"
        )
    except Exception:
        return False


def nz_mode() -> str:
    """Resolve the requested mechanism, defaulting to the transdata-free one."""
    requested = os.environ.get(NZ_MODE_ENV, "").strip().lower()
    if requested in ("cast", "reactor"):
        return requested
    if requested:
        logger.warning("%s=%r is not one of cast|reactor; using reactor", NZ_MODE_ENV, requested)
    return "reactor"


def nz_padded_numel(shape: tuple[int, int], block: tuple[int, int]) -> int:
    """Elements an NZ storage needs for ``shape``, rounded up per axis."""
    n, k = shape
    block_n, block_k = block
    return ((n + block_n - 1) // block_n * block_n) * ((k + block_k - 1) // block_k * block_k)


def nz_reactor_can_hold(shape: Sequence[int], stride: Sequence[int], dtype: torch.dtype) -> bool:
    """Whether pre-permuting ``shape``/``stride`` fits the slot it would replace.

    The transport is left untouched by this feature, so the permuted bytes have
    to occupy exactly the elements the ND layout used. That holds only for a
    plain contiguous 2-D weight whose padded fractal footprint equals ``N*K``.
    Every DiT linear in H3 satisfies it (all dims are multiples of 16); the
    guard is what keeps a future misaligned shape on the ``cast`` path instead
    of silently corrupting the buffer.
    """
    block = _NZ_BLOCK.get(dtype)
    if block is None or len(shape) != 2:
        return False
    n, k = shape
    if tuple(stride) != (k, 1):
        return False
    return nz_padded_numel((n, k), block) == n * k


def permute_host_weight_to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Reorder a host ``[N, K]`` weight into FRACTAL_NZ order.

    ``[N, K] -> [K1, N1, N0, K0]``, with ``(N0, K0)`` the dtype's fractal block.
    Runs on the checkpoint bytes while they are still in host memory, so it is
    off the critical path entirely.

    Both axes must divide evenly by their block, which ``nz_reactor_can_hold``
    has already established; there is deliberately no padding branch here,
    because a padded tensor would not fit the slot the ND layout reserved.
    """
    block_n, block_k = _NZ_BLOCK[weight.dtype]
    n, k = weight.shape
    n1, k1 = n // block_n, k // block_k
    return weight.reshape(n1, block_n, k1, block_k).permute(2, 0, 1, 3).contiguous().reshape(-1)


def nz_view_over_buffer(
    buffer: torch.Tensor,
    offset: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """View ``buffer[offset:offset+numel]`` as a zero-copy FRACTAL_NZ tensor.

    The result shares the buffer's memory and owns nothing: destroying it frees
    nothing, and it must not outlive ``buffer`` (measured -- once the buffer is
    released the caching allocator hands the address out again and the view
    reads whatever lands there, without raising).

    The base pointer is advanced instead of using ``storage_offset``: the
    constructors validate ``storage_offset * itemsize + required storage``
    against ``nbytes``, which an offset into a shared buffer fails.
    """
    import torch_npu

    n, k = shape
    metadata = {
        "size": (n, k),
        "stride": (k, 1),
        "nbytes": n * k * dtype.itemsize,
        "dtype": dtype,
        "data_ptr": buffer.data_ptr() + offset * dtype.itemsize,
        "storage_offset": 0,
        "device": buffer.device,
        "npu_format": _ACL_FORMAT_FRACTAL_NZ,
        "layout": torch.strided,
        "requires_grad": False,
    }
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)


def nz_reactor_takes_effect() -> bool:
    """Probe once that a buffer slice really re-reads as FRACTAL_NZ.

    Same reason as ``nz_cast_takes_effect``: without
    ``torch.npu.config.allow_internal_format`` the result would be an ordinary
    ND tensor, and the caller would measure no change and blame the feature.
    """
    try:
        shape = (16, 16)
        dtype = torch.bfloat16
        scratch = torch.zeros(shape[0] * shape[1], dtype=dtype, device="npu")
        return _npu_format(nz_view_over_buffer(scratch, 0, shape, dtype)) == _ACL_FORMAT_FRACTAL_NZ
    except Exception:
        return False


def _reactor_weight(module: nn.Module) -> torch.Tensor | None:
    """``module``'s weight when the reactor knows how to pre-permute it.

    Deliberately broader than ``_is_eligible_linear``: that filter exists to
    keep the load-time sweep away from quantized layers, whose method owns the
    weight layout. An offloaded weight has no such owner -- it arrives already
    in the layout the kernel asked for, and int8 is exactly the case worth
    covering, since that is the dtype an NZ weight actually gains on.

    Requiring ``LinearBase`` is what keeps the per-channel scales out: they are
    parameters of the same module, not modules of their own, so they are never
    visited here.
    """
    if not isinstance(module, LinearBase):
        return None
    weight = getattr(module, "weight", None)
    if weight is None or weight.dtype not in _NZ_BLOCK:
        return None
    return weight


def nz_reactor_weight_names(model: nn.Module) -> set[str]:
    """Names of ``model``'s parameters the reactor may pre-permute."""
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if _reactor_weight(module) is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        names.add(f"{prefix}weight")
    return names


def nz_cast_takes_effect() -> bool:
    """Probe once whether ``npu_format_cast`` actually produces NZ on this box.

    With ``torch.npu.config.allow_internal_format`` off the cast silently
    returns an ND tensor. Every caller of the offload-path cast checks this at
    setup so a no-op cannot be mistaken for "NZ measured, no gain".
    """
    import torch_npu

    try:
        probe = torch.zeros(16, 16, dtype=torch.bfloat16, device="npu")
        return _npu_format(torch_npu.npu_format_cast(probe, _ACL_FORMAT_FRACTAL_NZ)) == _ACL_FORMAT_FRACTAL_NZ
    except Exception:
        return False


def _npu_format(tensor: torch.Tensor) -> int | None:
    """Read a tensor's ACL format id, or ``None`` when it cannot be read."""
    import torch_npu

    try:
        return int(torch_npu.get_npu_format(tensor))
    except Exception:
        return None


def _is_npu_weight(weight: torch.Tensor) -> bool:
    """NZ is an Ascend storage layout, so only device-resident weights qualify.

    A weight left on the host belongs to an offload path, which the caller has
    already rejected; this is the second line of defence.
    """
    return weight.device.type == "npu"


def _is_eligible_linear(module: nn.Module) -> bool:
    """A vLLM linear whose weight layout no quant method owns.

    ``quant_config is None`` resolves to ``AscendUnquantizedLinearMethod`` on
    NPU, a subclass of ``UnquantizedLinearMethod``, so a single ``isinstance``
    accepts both the plain and the Ascend unquantized case while rejecting every
    quantized one (fp8/int8/mxfp8 build their own storage).
    """
    if not isinstance(module, LinearBase):
        return False
    return isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)


def install_minimax_h3_dit_weight_nz(model: nn.Module, od_config: Any | None = None) -> int:
    """Cast eligible DiT linear weights ND -> FRACTAL_NZ.

    Returns the number of weights actually left in the NZ layout, which is ``0``
    when the env opt-in is off, when an offload backend owns the parameters, or
    when nothing qualifies. Safe to call more than once: an already-NZ weight is
    left alone.
    """
    if not _weight_nz_env_enabled():
        return 0

    # Imported here rather than at module scope: this hook runs once per DiT at
    # load time, and the offload package pulls in the model loader.
    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
    from vllm_omni.diffusion.offloader.tensor_utils import is_dtensor

    if resolve_offload_strategy(od_config) is not OffloadStrategy.NONE:
        logger.info(
            "MiniMax-H3 DiT weight NZ deferred to the offload path (%s=%s): "
            "an offload backend owns the parameters and re-casts at upload time",
            WEIGHT_NZ_ENV,
            os.environ.get(WEIGHT_NZ_ENV),
        )
        return 0

    cast = 0
    skipped = 0
    for name, module in model.named_modules():
        if not _is_eligible_linear(module):
            continue
        weight = getattr(module, "weight", None)
        # DTensor shards (HSDP/FSDP2) are not plain 2-D parameters, and a 1-D
        # flat shard has no meaningful NZ layout.
        if weight is None or weight.dim() != 2 or is_dtensor(weight):
            skipped += 1
            continue
        if weight.dtype not in _NZ_DTYPES:
            # The fp32 residents (patch projections, time embedder, final layer)
            # stay ND; NZ does not support fp32.
            skipped += 1
            continue
        if not _is_npu_weight(weight):
            skipped += 1
            continue
        if _npu_format(weight) == _ACL_FORMAT_FRACTAL_NZ:
            skipped += 1
            continue
        cast_weight = maybe_cast_weight_to_nz(weight.data)
        if cast_weight is None:
            skipped += 1
            continue
        module.weight.data = cast_weight
        if _npu_format(module.weight) != _ACL_FORMAT_FRACTAL_NZ:
            # ``allow_internal_format`` is off, so the cast was silently dropped.
            logger.warning(
                "MiniMax-H3 DiT weight NZ: %s stayed ND; torch.npu.config.allow_internal_format is not enabled",
                name,
            )
            skipped += 1
            continue
        cast += 1

    logger.info(
        "MiniMax-H3 DiT weight NZ installed (%s=%s): %d linear weights now FRACTAL_NZ, %d left ND",
        WEIGHT_NZ_ENV,
        os.environ.get(WEIGHT_NZ_ENV),
        cast,
        skipped,
    )
    return cast


__all__ = [
    "NZ_MODE_ENV",
    "WEIGHT_NZ_ENV",
    "install_minimax_h3_dit_weight_nz",
    "maybe_cast_weight_to_nz",
    "nz_cast_takes_effect",
    "nz_mode",
    "nz_padded_numel",
    "nz_reactor_available",
    "nz_reactor_can_hold",
    "nz_reactor_takes_effect",
    "nz_reactor_weight_names",
    "nz_view_over_buffer",
    "permute_host_weight_to_nz",
    "weight_nz_enabled",
]
