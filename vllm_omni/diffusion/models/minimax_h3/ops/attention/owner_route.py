# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU route planning for MiniMax-H3 VSA coarse-output piggybacking.

The H3 VSA output is restored from tile-major order through an aligned
``destination row -> tiled source row`` map.  Under Ulysses, aligned output
rows are split into equal contiguous chunks.  A receiver therefore needs only
the coarse tiles referenced by its own chunk, rather than the full coarse
gate/output tensor.

This module builds that static owner route.  It deliberately contains no CUDA
or distributed communication code: the plan is constructed once on CPU and
can be consumed by a later producer-direct reverse-O implementation.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch

H3_VSA_TILE_ROWS = 64
_INT32_MAX = torch.iinfo(torch.int32).max

_TRUSTED_PLANS: dict[
    int,
    tuple[
        weakref.ReferenceType[H3VSAOwnerRoutePlan],
        tuple[weakref.ReferenceType[torch.Tensor], ...],
        tuple[int, ...],
        tuple[int, int, int, int],
        tuple[int, ...],
    ],
] = {}


@dataclass(frozen=True)
class H3VSAOwnerRoutePlan:
    """Static owner route for one aligned H3 VSA layout.

    ``owner_tile_ids[owner, :owner_tile_counts[owner]]`` lists the sorted,
    unique logical coarse tiles needed by that owner's sequence rows.  The
    remaining columns are exactly ``-1``.

    ``local_row_to_tail[owner, local_row]`` maps an effective local row to a
    relative slot in that owner's compact tail.  Structural aligned padding is
    represented by ``-1``.
    """

    owner_tile_ids: torch.Tensor
    owner_tile_counts: torch.Tensor
    local_row_to_tail: torch.Tensor
    logical_block_count: int

    @property
    def sp_world_size(self) -> int:
        return self.owner_tile_ids.shape[0]

    @property
    def local_rows(self) -> int:
        return self.local_row_to_tail.shape[1]

    @property
    def kmax(self) -> int:
        return self.owner_tile_ids.shape[1]

    @property
    def local_row_to_receive_row(self) -> torch.Tensor:
        """Map local rows to absolute rows in ``[local O; compact tail]``.

        The returned tensor has the same shape as ``local_row_to_tail``.
        Valid entries are ``local_rows + relative_tail_slot``; aligned padding
        remains ``-1``.
        """
        return torch.where(
            self.local_row_to_tail >= 0,
            self.local_row_to_tail + self.local_rows,
            self.local_row_to_tail,
        )

    @property
    def owner_tile_entries(self) -> int:
        """Total tile payload rows across all owners, before Kmax padding."""
        return int(self.owner_tile_counts.to(torch.int64).sum())

    @property
    def duplicate_owner_tile_entries(self) -> int:
        """Extra owner/tile entries caused by tiles spanning SP owners."""
        return self.owner_tile_entries - self.logical_block_count

    @property
    def cross_owner_tile_count(self) -> int:
        """Number of logical tiles referenced by more than one SP owner."""
        valid_tiles = self.owner_tile_ids[self.owner_tile_ids >= 0].to(torch.int64)
        frequencies = torch.bincount(valid_tiles, minlength=self.logical_block_count)
        return int(torch.count_nonzero(frequencies > 1))


def _register_owner_route_plan(plan: H3VSAOwnerRoutePlan) -> H3VSAOwnerRoutePlan:
    key = id(plan)

    def cleanup(reference: weakref.ReferenceType[H3VSAOwnerRoutePlan]) -> None:
        current = _TRUSTED_PLANS.get(key)
        if current is not None and current[0] is reference:
            _TRUSTED_PLANS.pop(key, None)

    tensors = (
        plan.owner_tile_ids,
        plan.owner_tile_counts,
        plan.local_row_to_tail,
    )
    plan_reference = weakref.ref(plan, cleanup)
    tensor_references = tuple(weakref.ref(tensor) for tensor in tensors)
    versions = tuple(tensor._version for tensor in tensors)
    geometry = (
        plan.sp_world_size,
        plan.local_rows,
        plan.kmax,
        plan.logical_block_count,
    )
    data_ptrs = tuple(tensor.data_ptr() for tensor in tensors)
    _TRUSTED_PLANS[key] = (
        plan_reference,
        tensor_references,
        versions,
        geometry,
        data_ptrs,
    )
    return plan


def h3_vsa_owner_route_plan_is_trusted(plan: H3VSAOwnerRoutePlan) -> bool:
    """Whether ``plan`` is an unmodified result of the validated builder."""
    if not isinstance(plan, H3VSAOwnerRoutePlan):
        return False
    entry = _TRUSTED_PLANS.get(id(plan))
    if entry is None:
        return False
    plan_reference, tensor_references, versions, geometry, data_ptrs = entry
    tensors = (
        plan.owner_tile_ids,
        plan.owner_tile_counts,
        plan.local_row_to_tail,
    )
    return (
        plan_reference() is plan
        and all(reference() is tensor for reference, tensor in zip(tensor_references, tensors))
        and versions == tuple(tensor._version for tensor in tensors)
        and data_ptrs == tuple(tensor.data_ptr() for tensor in tensors)
        and geometry
        == (
            plan.sp_world_size,
            plan.local_rows,
            plan.kmax,
            plan.logical_block_count,
        )
    )


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_aligned_source_rows(
    aligned_untile_source_rows: torch.Tensor,
    *,
    global_rows: int,
    logical_block_count: int,
) -> None:
    if aligned_untile_source_rows.device.type != "cpu":
        raise ValueError("aligned_untile_source_rows must be a CPU tensor")
    if aligned_untile_source_rows.ndim != 1:
        raise ValueError("aligned_untile_source_rows must be one-dimensional")
    if aligned_untile_source_rows.dtype != torch.int32:
        raise TypeError("aligned_untile_source_rows must use int32 indices")
    if not aligned_untile_source_rows.is_contiguous():
        raise ValueError("aligned_untile_source_rows must be contiguous")
    if aligned_untile_source_rows.numel() != global_rows:
        raise ValueError(
            "aligned_untile_source_rows length must equal global_rows, got "
            f"{aligned_untile_source_rows.numel()} and {global_rows}"
        )

    source_rows64 = aligned_untile_source_rows.to(torch.int64)
    if torch.any(source_rows64 < -1):
        raise ValueError("aligned_untile_source_rows may use only -1 for padding")

    valid_mask = source_rows64 >= 0
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        raise ValueError("aligned_untile_source_rows must contain valid rows")
    if not torch.all(valid_mask[:valid_count]) or torch.any(valid_mask[valid_count:]):
        raise ValueError("aligned padding must be a contiguous -1 suffix")

    valid_rows = source_rows64[:valid_count]
    tiled_rows = logical_block_count * H3_VSA_TILE_ROWS
    if int(valid_rows.max()) >= tiled_rows:
        raise ValueError("aligned_untile_source_rows contains an out-of-bounds tiled source row")
    if torch.unique(valid_rows).numel() != valid_count:
        raise ValueError("valid aligned_untile_source_rows must be globally unique")

    covered_tiles = torch.unique(
        torch.div(valid_rows, H3_VSA_TILE_ROWS, rounding_mode="floor"),
        sorted=True,
    )
    expected_tiles = torch.arange(logical_block_count, dtype=torch.int64)
    if not torch.equal(covered_tiles, expected_tiles):
        raise ValueError("aligned_untile_source_rows must cover every logical tile")


def _validate_owner_route_plan(
    plan: H3VSAOwnerRoutePlan,
    aligned_untile_source_rows: torch.Tensor,
) -> None:
    """Fail closed if a constructed route violates its source-map contract."""
    sp_world_size = plan.sp_world_size
    local_rows = plan.local_rows
    kmax = plan.kmax

    if plan.owner_tile_ids.dtype != torch.int32:
        raise RuntimeError("owner_tile_ids must use int32 indices")
    if plan.owner_tile_counts.dtype != torch.int32:
        raise RuntimeError("owner_tile_counts must use int32 indices")
    if plan.local_row_to_tail.dtype != torch.int32:
        raise RuntimeError("local_row_to_tail must use int32 indices")
    if plan.owner_tile_counts.shape != (sp_world_size,):
        raise RuntimeError("owner_tile_counts shape does not match owner_tile_ids")
    if plan.local_row_to_tail.shape != (sp_world_size, local_rows):
        raise RuntimeError("local_row_to_tail has an invalid shape")

    local_sources = aligned_untile_source_rows.view(sp_world_size, local_rows).to(torch.int64)
    for owner in range(sp_world_size):
        count = int(plan.owner_tile_counts[owner])
        if not 0 <= count <= kmax:
            raise RuntimeError("owner_tile_counts contains an out-of-range count")

        tile_ids = plan.owner_tile_ids[owner, :count].to(torch.int64)
        padding = plan.owner_tile_ids[owner, count:]
        if count and (
            int(tile_ids.min()) < 0
            or int(tile_ids.max()) >= plan.logical_block_count
            or torch.unique(tile_ids).numel() != count
            or (count > 1 and not torch.all(tile_ids[1:] > tile_ids[:-1]))
        ):
            raise RuntimeError("an owner's active tile IDs must be sorted, unique, and in range")
        if torch.any(padding != -1):
            raise RuntimeError("owner_tile_ids padding must be exactly -1")

        source = local_sources[owner]
        valid_mask = source >= 0
        slots = plan.local_row_to_tail[owner].to(torch.int64)
        if torch.any(slots[~valid_mask] != -1):
            raise RuntimeError("aligned padding rows must map to -1 tail slots")
        if torch.any(slots[valid_mask] < 0) or torch.any(slots[valid_mask] >= count):
            raise RuntimeError("a valid local row maps outside its owner's active tail")
        if valid_mask.any():
            expected_tiles = torch.div(source[valid_mask], H3_VSA_TILE_ROWS, rounding_mode="floor")
            routed_tiles = tile_ids.index_select(0, slots[valid_mask])
            if not torch.equal(routed_tiles, expected_tiles):
                raise RuntimeError("local_row_to_tail does not route to the row's logical tile")


def build_h3_vsa_owner_route_plan(
    aligned_untile_source_rows: torch.Tensor,
    *,
    sp_world_size: int,
    global_rows: int,
    local_rows: int,
    logical_block_count: int,
) -> H3VSAOwnerRoutePlan:
    """Build a compact per-owner coarse-tile route for reverse-O piggybacking.

    Args:
        aligned_untile_source_rows: Validated aligned output-row to tile-major
            source-row map.  It must be contiguous CPU int32 with only a
            trailing ``-1`` padding suffix.
        sp_world_size: Number of contiguous Ulysses sequence owners.
        global_rows: Aligned global sequence length.
        local_rows: Aligned rows received by each owner.
        logical_block_count: Number of tile-64 coarse blocks.
    """
    _validate_positive_int("sp_world_size", sp_world_size)
    _validate_positive_int("global_rows", global_rows)
    _validate_positive_int("local_rows", local_rows)
    _validate_positive_int("logical_block_count", logical_block_count)
    if global_rows != sp_world_size * local_rows:
        raise ValueError(
            f"global_rows must equal sp_world_size * local_rows, got {global_rows} != {sp_world_size} * {local_rows}"
        )
    if global_rows > _INT32_MAX or logical_block_count * H3_VSA_TILE_ROWS > _INT32_MAX:
        raise ValueError("route geometry exceeds the int32 indexing contract")

    _validate_aligned_source_rows(
        aligned_untile_source_rows,
        global_rows=global_rows,
        logical_block_count=logical_block_count,
    )

    # These three persistent maps participate in the trusted-plan registry,
    # whose mutation check relies on their tensor version counters.  Model
    # execution constructs this plan under torch.inference_mode(), where new
    # tensors ordinarily have no version counter.  Build the maps with
    # inference mode locally disabled, matching the trusted H3 layout-map
    # builders, so production and test execution retain the same fail-closed
    # mutation contract.
    with torch.inference_mode(False):
        local_sources = aligned_untile_source_rows.view(sp_world_size, local_rows).to(torch.int64).clone()
        owner_tiles: list[torch.Tensor] = []
        counts: list[int] = []
        for owner in range(sp_world_size):
            valid_sources = local_sources[owner][local_sources[owner] >= 0]
            tile_ids = torch.unique(
                torch.div(valid_sources, H3_VSA_TILE_ROWS, rounding_mode="floor"),
                sorted=True,
            )
            owner_tiles.append(tile_ids)
            counts.append(tile_ids.numel())

        kmax = max(counts)
        owner_tile_ids = torch.full(
            (sp_world_size, kmax),
            -1,
            dtype=torch.int32,
        )
        owner_tile_counts = torch.tensor(counts, dtype=torch.int32)
        local_row_to_tail = torch.full(
            (sp_world_size, local_rows),
            -1,
            dtype=torch.int32,
        )

        for owner, tile_ids in enumerate(owner_tiles):
            count = tile_ids.numel()
            owner_tile_ids[owner, :count] = tile_ids.to(torch.int32)

            source = local_sources[owner]
            valid_mask = source >= 0
            local_tiles = torch.div(source[valid_mask], H3_VSA_TILE_ROWS, rounding_mode="floor")
            local_row_to_tail[owner, valid_mask] = torch.searchsorted(
                tile_ids,
                local_tiles,
            ).to(torch.int32)

    plan = H3VSAOwnerRoutePlan(
        owner_tile_ids=owner_tile_ids,
        owner_tile_counts=owner_tile_counts,
        local_row_to_tail=local_row_to_tail,
        logical_block_count=logical_block_count,
    )
    _validate_owner_route_plan(plan, aligned_untile_source_rows)
    return _register_owner_route_plan(plan)


__all__ = [
    "H3_VSA_TILE_ROWS",
    "H3VSAOwnerRoutePlan",
    "build_h3_vsa_owner_route_plan",
    "h3_vsa_owner_route_plan_is_trusted",
]
