# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only layout helpers for MiniMax-H3 tile-64 VSA.

The reference H3 path first zeroes a padded tile tensor and then evaluates an
advanced gather followed by indexed assignment::

    tiled = torch.zeros(...)
    tiled[:, non_pad] = compact[:, partition]

For long H3 documents the RHS is another full compact tensor.  The optional
CUDA paths below write every destination row exactly once and use cached
``destination row -> source row`` maps.  Negative map entries produce exact
zeros, so edge-tile, optional partner-tile, and packed-sequence suffix padding
retain the reference contract.

These are deliberately narrow, feature-gated primitives.  Routing, sparse
block selection, and attention arithmetic are unchanged.
"""

from __future__ import annotations

import os
import weakref

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

H3_VSA_FUSED_TILE_PACK_ENV = "VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK"
H3_VSA_FUSED_UNTILE_ENV = "VLLM_OMNI_FASTVIDEO_VSA_FUSED_UNTILE"
# Producer-owned marker that identifies an H3 target-video call whose resolved
# attention backend is FASTVIDEO_VSA.  It deliberately does not depend on the
# learned gate tensor being present: opt-in communication experiments use it
# to fail before their first collective if gate production regresses.
H3_VSA_ATTENTION_ACTIVE_KEY = "vsa_h3_attention_active"
_MAX_FUSED_ROW_WIDTH = 4096

# Avoid a device synchronization on every one of H3's 200 attention layers:
# CUDA kernels accept only maps that were range/uniqueness checked by the
# builders below. The weak reference prevents allocator pointer reuse from
# inheriting stale trust after a map is freed.
_VALIDATED_SOURCE_ROWS: dict[
    tuple[str, int | None, int],
    tuple[weakref.ReferenceType[torch.Tensor], int, str, int, int],
] = {}


def _source_rows_key(source_rows: torch.Tensor) -> tuple[str, int | None, int]:
    return (
        source_rows.device.type,
        source_rows.device.index,
        source_rows.data_ptr(),
    )


def _source_rows_version(source_rows: torch.Tensor) -> int | None:
    """Return the mutation version used by the trusted-map registry.

    Tensors created while ``torch.inference_mode`` is active intentionally do
    not have a version counter.  Such a tensor cannot participate safely in a
    cache whose trust contract depends on detecting in-place mutation, so it
    must fail closed instead of making ``._version`` raise during model setup.
    """
    if source_rows.is_inference():
        return None
    return source_rows._version


def _register_source_rows(
    source_rows: torch.Tensor,
    *,
    kind: str,
    source_bound: int,
    valid_rows: int,
) -> torch.Tensor:
    key = _source_rows_key(source_rows)
    version = _source_rows_version(source_rows)
    if version is None:
        # Do not let allocator pointer reuse inherit a previous tensor's trust
        # if an inference tensor is ever passed here outside the builders.
        _VALIDATED_SOURCE_ROWS.pop(key, None)
        return source_rows

    def cleanup(reference: weakref.ReferenceType[torch.Tensor]) -> None:
        current = _VALIDATED_SOURCE_ROWS.get(key)
        if current is not None and current[0] is reference:
            _VALIDATED_SOURCE_ROWS.pop(key, None)

    reference = weakref.ref(source_rows, cleanup)
    _VALIDATED_SOURCE_ROWS[key] = (
        reference,
        version,
        kind,
        source_bound,
        valid_rows,
    )
    return source_rows


def _source_rows_are_registered(
    source_rows: torch.Tensor,
    *,
    kind: str,
    source_bound: int,
    valid_rows: int | None = None,
) -> bool:
    entry = _VALIDATED_SOURCE_ROWS.get(_source_rows_key(source_rows))
    if entry is None:
        return False
    reference, version, registered_kind, registered_bound, registered_valid = entry
    current_version = _source_rows_version(source_rows)
    return (
        reference() is source_rows
        and current_version is not None
        and version == current_version
        and registered_kind == kind
        and registered_bound == source_bound
        and (valid_rows is None or registered_valid == valid_rows)
    )


def h3_vsa_fused_tile_pack_enabled() -> bool:
    """Return whether the opt-in H3 compact-to-tile CUDA path is requested."""
    raw = os.environ.get(H3_VSA_FUSED_TILE_PACK_ENV, "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def h3_vsa_fused_untile_enabled() -> bool:
    """Return whether the opt-in H3 tile-to-aligned CUDA path is requested."""
    raw = os.environ.get(H3_VSA_FUSED_UNTILE_ENV, "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def build_h3_tiled_source_rows(
    partition: torch.Tensor,
    non_pad: torch.Tensor,
    padded_rows: int,
) -> torch.Tensor:
    """Build the inverse map consumed by :func:`h3_vsa_tile_pack`.

    ``partition`` and ``non_pad`` encode the existing reference assignment
    ``dst[:, non_pad] = src[:, partition]``.  The returned int32 vector has
    length ``padded_rows``; ``-1`` denotes a destination padding row.

    Geometry is constructed once per cached H3 layout.  Strict validation is
    intentional: a malformed map must never reach an asynchronous CUDA
    kernel, where an out-of-bounds source row would poison the process.
    """
    if partition.ndim != 1 or non_pad.ndim != 1:
        raise ValueError("partition and non_pad must be one-dimensional")
    if partition.device != non_pad.device:
        raise ValueError("partition and non_pad must be on the same device")
    if partition.numel() != non_pad.numel():
        raise ValueError(f"partition and non_pad must have equal length, got {partition.numel()} and {non_pad.numel()}")
    if partition.dtype not in (torch.int32, torch.int64) or non_pad.dtype not in (torch.int32, torch.int64):
        raise TypeError("partition and non_pad must use int32 or int64 indices")
    if padded_rows < partition.numel():
        raise ValueError(f"padded_rows={padded_rows} cannot hold {partition.numel()} compact rows")
    if padded_rows > torch.iinfo(torch.int32).max:
        raise ValueError("padded_rows exceeds the int32 layout-map contract")

    compact_rows = partition.numel()
    if compact_rows:
        partition64 = partition.to(torch.int64)
        non_pad64 = non_pad.to(torch.int64)
        if int(partition64.min()) < 0 or int(partition64.max()) >= compact_rows:
            raise ValueError("partition contains an out-of-bounds compact source row")
        if int(non_pad64.min()) < 0 or int(non_pad64.max()) >= padded_rows:
            raise ValueError("non_pad contains an out-of-bounds padded destination row")
        if torch.unique(partition64).numel() != compact_rows:
            raise ValueError("partition must be a permutation of all compact source rows")
        if torch.unique(non_pad64).numel() != compact_rows:
            raise ValueError("non_pad destination rows must be unique")

    # Layout maps must retain a version counter even when model construction
    # runs under the outer inference-mode context.  The trusted-map registry
    # uses that counter to reject maps modified after geometry validation.
    with torch.inference_mode(False):
        source_rows = torch.full(
            (padded_rows,),
            -1,
            dtype=torch.int32,
            device=partition.device,
        )
        if compact_rows:
            source_rows.index_copy_(
                0,
                non_pad.to(torch.int64),
                partition.to(torch.int32),
            )
    return _register_source_rows(
        source_rows,
        kind="tile_pack",
        source_bound=compact_rows,
        valid_rows=compact_rows,
    )


def build_h3_aligned_untile_source_rows(
    untile: torch.Tensor,
    *,
    aligned_rows: int,
    tiled_rows: int,
) -> torch.Tensor:
    """Build the map for one-pass tile-64 output un-tiling and pad restore.

    ``untile[compact_row]`` is the source row in the VSA kernel's tile-64
    output.  The returned int32 map has ``aligned_rows`` entries: the valid
    prefix preserves ``untile`` exactly and every structural suffix row is
    ``-1``, which the CUDA kernel writes as an exact zero.
    """
    if untile.ndim != 1:
        raise ValueError("untile must be one-dimensional")
    if untile.dtype not in (torch.int32, torch.int64):
        raise TypeError("untile must use int32 or int64 indices")
    if aligned_rows < untile.numel():
        raise ValueError(f"aligned_rows={aligned_rows} cannot hold {untile.numel()} valid rows")
    if aligned_rows > torch.iinfo(torch.int32).max:
        raise ValueError("aligned_rows exceeds the int32 layout-map contract")
    if tiled_rows <= 0:
        raise ValueError(f"tiled_rows must be positive, got {tiled_rows}")

    if untile.numel():
        untile64 = untile.to(torch.int64)
        if int(untile64.min()) < 0 or int(untile64.max()) >= tiled_rows:
            raise ValueError("untile contains an out-of-bounds tiled source row")
        if torch.unique(untile64).numel() != untile.numel():
            raise ValueError("untile source rows must be unique")

    # See build_h3_tiled_source_rows: mutation tracking is part of the CUDA
    # trust contract, so the map itself must not be an inference tensor.
    with torch.inference_mode(False):
        source_rows = torch.full(
            (aligned_rows,),
            -1,
            dtype=torch.int32,
            device=untile.device,
        )
        if untile.numel():
            source_rows[: untile.numel()] = untile.to(torch.int32)
    return _register_source_rows(
        source_rows,
        kind="aligned_untile",
        source_bound=tiled_rows,
        valid_rows=untile.numel(),
    )


def _validate_h3_tile_pack_inputs(x: torch.Tensor, source_rows: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"H3 VSA tile pack expects [B,S,H,D], got {tuple(x.shape)}")
    if source_rows.ndim != 1 or source_rows.dtype != torch.int32:
        raise ValueError("source_rows must be a one-dimensional int32 tensor")
    if source_rows.device != x.device:
        raise ValueError("source_rows must be on the activation device")
    if x.shape[1] > source_rows.numel():
        raise ValueError(f"padded destination has {source_rows.numel()} rows but compact input has {x.shape[1]} rows")
    if source_rows.numel() == 0 and x.shape[1] != 0:
        raise ValueError("a non-empty compact input requires a non-empty destination map")


def h3_vsa_tile_pack_cuda_supported(x: torch.Tensor, source_rows: torch.Tensor) -> bool:
    """Whether the inference-only Triton implementation can safely run."""
    row_width = x.shape[2] * x.shape[3] if x.ndim == 4 else 0
    return (
        HAS_TRITON
        and x.is_cuda
        and x.ndim == 4
        and x.dtype in (torch.float16, torch.bfloat16)
        and not x.requires_grad
        and source_rows.is_cuda
        and source_rows.device == x.device
        and source_rows.ndim == 1
        and source_rows.dtype == torch.int32
        and source_rows.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] > 0
        and x.shape[2] > 0
        and x.shape[3] > 0
        and x.shape[1] <= source_rows.numel()
        and 0 < row_width <= _MAX_FUSED_ROW_WIDTH
        and _source_rows_are_registered(
            source_rows,
            kind="tile_pack",
            source_bound=x.shape[1],
            valid_rows=x.shape[1],
        )
    )


def h3_vsa_tile_pack_reference(x: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
    """Pure-Torch reference used by CPU tests and correctness qualification."""
    _validate_h3_tile_pack_inputs(x, source_rows)
    output = torch.zeros(
        (x.shape[0], source_rows.numel(), x.shape[2], x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )
    valid_destinations = torch.nonzero(source_rows >= 0, as_tuple=False).flatten()
    if valid_destinations.numel():
        compact_sources = source_rows.index_select(0, valid_destinations).to(torch.int64)
        if int(compact_sources.min()) < 0 or int(compact_sources.max()) >= x.shape[1]:
            raise ValueError("source_rows contains an out-of-bounds compact source row")
        if torch.unique(compact_sources).numel() != x.shape[1] or compact_sources.numel() != x.shape[1]:
            raise ValueError("source_rows must reference every compact source row exactly once")
        output.index_copy_(1, valid_destinations, x.index_select(1, compact_sources))
    elif x.shape[1]:
        raise ValueError("source_rows does not reference the non-empty compact input")
    return output


def _validate_h3_tile_untile_inputs(x: torch.Tensor, source_rows: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"H3 VSA tile untile expects [B,S,H,D], got {tuple(x.shape)}")
    if source_rows.ndim != 1 or source_rows.dtype != torch.int32:
        raise ValueError("source_rows must be a one-dimensional int32 tensor")
    if source_rows.device != x.device:
        raise ValueError("source_rows must be on the activation device")
    if source_rows.numel() == 0:
        raise ValueError("H3 VSA tile untile requires a non-empty destination map")


def _validate_h3_tile_untile_output(
    x: torch.Tensor,
    source_rows: torch.Tensor,
    out: torch.Tensor,
) -> None:
    expected_shape = (
        x.shape[0],
        source_rows.numel(),
        x.shape[2],
        x.shape[3],
    )
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"H3 untile output shape mismatch: expected {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != x.dtype or out.device != x.device:
        raise ValueError(
            "H3 VSA tile untile output dtype/device mismatch: "
            f"expected=({x.dtype}, {x.device}), observed=({out.dtype}, {out.device})"
        )
    if not out.is_contiguous():
        raise ValueError("H3 VSA tile untile output must be contiguous")
    if out.requires_grad:
        raise ValueError("H3 VSA tile untile output must not require gradients")
    if out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
        raise ValueError("H3 VSA tile untile input and output must not share storage")


def h3_vsa_tile_untile_cuda_supported(
    x: torch.Tensor,
    source_rows: torch.Tensor,
) -> bool:
    """Whether the one-pass tile-to-aligned CUDA implementation can run."""
    row_width = x.shape[2] * x.shape[3] if x.ndim == 4 else 0
    return (
        HAS_TRITON
        and x.is_cuda
        and x.ndim == 4
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16)
        and not x.requires_grad
        and source_rows.is_cuda
        and source_rows.device == x.device
        and source_rows.ndim == 1
        and source_rows.dtype == torch.int32
        and source_rows.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] > 0
        and x.shape[2] > 0
        and x.shape[3] > 0
        and source_rows.numel() > 0
        and 0 < row_width <= _MAX_FUSED_ROW_WIDTH
        and _source_rows_are_registered(
            source_rows,
            kind="aligned_untile",
            source_bound=x.shape[1],
        )
    )


def h3_vsa_tile_untile_reference(
    x: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Pure-Torch tile-to-aligned reference, including exact suffix zeros."""
    _validate_h3_tile_untile_inputs(x, source_rows)
    output = torch.zeros(
        (x.shape[0], source_rows.numel(), x.shape[2], x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )
    valid_destinations = torch.nonzero(source_rows >= 0, as_tuple=False).flatten()
    if valid_destinations.numel():
        tiled_sources = source_rows.index_select(0, valid_destinations).to(torch.int64)
        if int(tiled_sources.min()) < 0 or int(tiled_sources.max()) >= x.shape[1]:
            raise ValueError("source_rows contains an out-of-bounds tiled source row")
        if torch.unique(tiled_sources).numel() != tiled_sources.numel():
            raise ValueError("source_rows tiled source rows must be unique")
        output.index_copy_(1, valid_destinations, x.index_select(1, tiled_sources))
    return output


if HAS_TRITON:

    @triton.jit
    def _h3_vsa_tile_pack_kernel(
        source_ptr,
        source_rows_ptr,
        output_ptr,
        source_stride_batch,
        source_stride_seq,
        source_stride_head,
        source_stride_dim,
        padded_rows: tl.constexpr,
        head_dim: tl.constexpr,
        row_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        flat_destination_row = tl.program_id(0)
        batch_index = flat_destination_row // padded_rows
        destination_row = flat_destination_row - batch_index * padded_rows
        source_row = tl.load(source_rows_ptr + destination_row)

        feature_offsets = tl.arange(0, block_size)
        feature_mask = feature_offsets < row_width
        head_index = feature_offsets // head_dim
        dim_index = feature_offsets - head_index * head_dim
        valid_source = source_row >= 0
        source_offsets = (
            batch_index * source_stride_batch
            + source_row * source_stride_seq
            + head_index * source_stride_head
            + dim_index * source_stride_dim
        )
        values = tl.load(source_ptr + source_offsets, mask=feature_mask & valid_source, other=0.0)
        output_offsets = flat_destination_row * row_width + feature_offsets
        tl.store(output_ptr + output_offsets, values, mask=feature_mask)


def _h3_vsa_tile_pack_cuda_impl(x: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
    _validate_h3_tile_pack_inputs(x, source_rows)
    if not h3_vsa_tile_pack_cuda_supported(x, source_rows):
        raise RuntimeError("H3 VSA fused tile pack received an unsupported CUDA input")
    output = torch.empty(
        (x.shape[0], source_rows.numel(), x.shape[2], x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )
    row_width = x.shape[2] * x.shape[3]
    block_size = triton.next_power_of_2(row_width)
    _h3_vsa_tile_pack_kernel[(x.shape[0] * source_rows.numel(),)](
        x,
        source_rows,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        padded_rows=source_rows.numel(),
        head_dim=x.shape[3],
        row_width=row_width,
        block_size=block_size,
        num_warps=4 if block_size <= 256 else 8,
    )
    return output


def _h3_vsa_tile_untile_cuda_impl(
    x: torch.Tensor,
    source_rows: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_h3_tile_untile_inputs(x, source_rows)
    if not h3_vsa_tile_untile_cuda_supported(x, source_rows):
        raise RuntimeError("H3 VSA fused tile untile received an unsupported CUDA input")
    if out is None:
        output = torch.empty(
            (x.shape[0], source_rows.numel(), x.shape[2], x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )
    else:
        _validate_h3_tile_untile_output(x, source_rows, out)
        output = out
    row_width = x.shape[2] * x.shape[3]
    block_size = triton.next_power_of_2(row_width)
    _h3_vsa_tile_pack_kernel[(x.shape[0] * source_rows.numel(),)](
        x,
        source_rows,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        padded_rows=source_rows.numel(),
        head_dim=x.shape[3],
        row_width=row_width,
        block_size=block_size,
        num_warps=4 if block_size <= 256 else 8,
    )
    return output


if not hasattr(torch.ops.vllm_omni, "h3_vsa_tile_pack"):

    @torch.library.custom_op("vllm_omni::h3_vsa_tile_pack", mutates_args=())
    def _h3_vsa_tile_pack_op(x: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
        return _h3_vsa_tile_pack_cuda_impl(x, source_rows)

    @_h3_vsa_tile_pack_op.register_fake
    def _(x: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (x.shape[0], source_rows.shape[0], x.shape[2], x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )


if not hasattr(torch.ops.vllm_omni, "h3_vsa_tile_untile"):

    @torch.library.custom_op("vllm_omni::h3_vsa_tile_untile", mutates_args=())
    def _h3_vsa_tile_untile_op(
        x: torch.Tensor,
        source_rows: torch.Tensor,
    ) -> torch.Tensor:
        return _h3_vsa_tile_untile_cuda_impl(x, source_rows)

    @_h3_vsa_tile_untile_op.register_fake
    def _(x: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (x.shape[0], source_rows.shape[0], x.shape[2], x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )


def h3_vsa_tile_pack(x: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
    """Pack compact H3 rows into tile-64 order through the fused CUDA op."""
    _validate_h3_tile_pack_inputs(x, source_rows)
    if not h3_vsa_tile_pack_cuda_supported(x, source_rows):
        raise RuntimeError("H3 VSA fused tile pack is unavailable for this input")
    return torch.ops.vllm_omni.h3_vsa_tile_pack(x, source_rows)


def h3_vsa_tile_untile(
    x: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Un-tile H3 output into aligned packed order through the fused CUDA op."""
    _validate_h3_tile_untile_inputs(x, source_rows)
    if not h3_vsa_tile_untile_cuda_supported(x, source_rows):
        raise RuntimeError("H3 VSA fused tile untile is unavailable for this input")
    return torch.ops.vllm_omni.h3_vsa_tile_untile(x, source_rows)


@torch.compiler.disable
def h3_vsa_tile_untile_out(
    x: torch.Tensor,
    source_rows: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Un-tile directly into caller-owned eager output storage.

    This mutating form intentionally bypasses the functional custom op. H3's
    packed-attention island is already eager, and the caller-owned storage is
    a registered FlashInfer communication buffer that cannot be represented
    as a functional compiler allocation.
    """
    _validate_h3_tile_untile_inputs(x, source_rows)
    _validate_h3_tile_untile_output(x, source_rows, out)
    if not h3_vsa_tile_untile_cuda_supported(x, source_rows):
        raise RuntimeError("H3 VSA fused tile untile-out is unavailable for this input")
    return _h3_vsa_tile_untile_cuda_impl(x, source_rows, out=out)


__all__ = [
    "H3_VSA_ATTENTION_ACTIVE_KEY",
    "H3_VSA_FUSED_TILE_PACK_ENV",
    "H3_VSA_FUSED_UNTILE_ENV",
    "build_h3_aligned_untile_source_rows",
    "build_h3_tiled_source_rows",
    "h3_vsa_fused_tile_pack_enabled",
    "h3_vsa_fused_untile_enabled",
    "h3_vsa_tile_pack",
    "h3_vsa_tile_pack_cuda_supported",
    "h3_vsa_tile_pack_reference",
    "h3_vsa_tile_untile",
    "h3_vsa_tile_untile_out",
    "h3_vsa_tile_untile_cuda_supported",
    "h3_vsa_tile_untile_reference",
]
