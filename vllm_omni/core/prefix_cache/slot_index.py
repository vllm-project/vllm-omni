# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Maps each KV slot to the saved CPU row holding its stage outputs."""

from typing import NamedTuple, TypeAlias

import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError, TensorName

# The outputs one save copied to the CPU, by tensor name.
SavedOutputs: TypeAlias = dict[TensorName, torch.Tensor]


class SavedSlice(NamedTuple):
    """A cache hit subspan (first row idx & num rows in the subspan) with the corresponding
    SavedOutputs. Usually the tokens one request computed in one engine step.
    """

    outputs: SavedOutputs | None  # None for slots no save has written
    first_row: int
    num_rows: int


class PrefixHit:
    """Wraps a request's prefix cache hit stored as its subspans (SavedSlices) in slot order.

    NOTE: This holds the SavedOutputs its subspans read from, so later saves that overwrite the
    hit's slots don't change it. This generally shouldn't happen, but could occur in some cases
    with async outputs under cache pressure.
    """

    def __init__(self, slices: list[SavedSlice]) -> None:
        self._slices = slices

    def all_slots_saved(self, key: TensorName) -> bool:
        """Whether every slot of the hit has a row saved for this key."""
        return all(s.outputs is not None and key in s.outputs for s in self._slices)

    def cat(self, key: TensorName, new_rows: torch.Tensor) -> torch.Tensor:
        """The hit's rows for this key in slot order, followed by new_rows.

        Every slot needs a row saved for the key; check all_slots_saved first.
        """
        rows = [s.outputs[key][s.first_row : s.first_row + s.num_rows] for s in self._slices]
        return torch.cat([*rows, new_rows])


class SlotIndex:
    """Used only on the engine thread, though the hits it returns can be read from any thread.

    A save stays alive while any slot or hit still refers to it. Note that a save's ID is just
    a monotonically increasing ID that we can delete to drop tensors.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self._block_size = block_size
        # Each slot holds the id of the save that last wrote it (-1 for none) and its row in that save.
        self._save_of_slot = torch.full((num_blocks * block_size,), -1, dtype=torch.long)
        self._row_of_slot = torch.zeros(num_blocks * block_size, dtype=torch.long)
        self._outputs_of_save: dict[int, SavedOutputs] = {}
        # How many slots hold each save. One no slot holds is dropped; hits keep their own reference.
        self._num_slots_of_save: dict[int, int] = {}
        self._next_save_id = 0

    def _release_stale_saves(self, slots: torch.Tensor) -> None:
        """Decrement reference counters for saves due to reused blocks, and delete our refs to the
        output tensors for anything with no remaining references.

        NOTE (Alex): Since outputs refs to pinned CPU tensors to avoid any copies, and we don't
        delete partial saves, it is possible to have a pinned tensor holding a long suffix that
        is no longer used if it's shared across requests. If this becomes an issue, the easiest
        way to handle it is likely compacting & copying tensors with (now unused) suffixes so that
        we can repoint the outputs to release the longer pinned tensors, but we first need
        to verify if it's a real problem.
        """
        old_ids, counts = self._save_of_slot[slots].unique(return_counts=True)
        for old_id, count in zip(old_ids.tolist(), counts.tolist()):
            if old_id < 0:  # -1 indicates this slot hasn't been written to yet
                continue
            self._num_slots_of_save[old_id] -= count
            # If the reused slot causes the save reference count to hit 0, free its tensors
            if self._num_slots_of_save[old_id] == 0:
                del self._outputs_of_save[old_id], self._num_slots_of_save[old_id]

    def record(self, outputs: SavedOutputs, slots: torch.Tensor) -> None:
        """Record that slots[i]'s outputs are row i of every tensor in outputs.

        Raises if slots repeat or are negative, or if a tensor doesn't have one row per slot.
        """
        # Example: Say we have block_size 2 and this step computed:
        #   - request A's tokens into slots 8-10
        #   - request B's into slots 4-5
        # Then, we would have slots = [8, 9, 10, 4, 5] and every tensor has 5 rows (since we
        # are mapping each token -> tensors, i.e., mapping 2D -> 3D).
        if not len(slots):
            return
        if slots.min() < 0 or len(slots.unique()) != len(slots):
            raise OmniPrefixCacheUnmatchError("record SlotIndex received repeated or negative KV slots")
        # tensor to be fetched & slots need to be the same length for all tensors in prefix cache
        mismatched = [key for key, tensor in outputs.items() if tensor.shape[0] != len(slots)]
        if mismatched:
            raise OmniPrefixCacheUnmatchError(f"save has {len(slots)} slots but another row count for {mismatched}")

        # Update saves for blocks in the batch that are being reused.
        # e.g., if slots 8-10 were never written and slots 4-5 previously were referencing 12:
        #   old_ids = [-1, 12], counts = [3, 2]
        # This means that 4,5 will now refer to something else and as a result,
        # save 12 needs to have its reference count reduced by 2.
        self._release_stale_saves(slots)

        # Point the i-th slot at row i of this save, e.g. if the current save ID is 17:
        #   slot   8   9  10   4   5
        #   save  17  17  17  17  17
        #   row    0   1   2   3   4
        save_id = self._next_save_id
        self._next_save_id += 1  # update the next save ID
        self._outputs_of_save[save_id] = outputs
        self._num_slots_of_save[save_id] = len(slots)
        self._save_of_slot[slots] = save_id
        self._row_of_slot[slots] = torch.arange(len(slots))

    def lookup(self, block_ids: list[int]) -> PrefixHit:
        """Given a set of block IDs, retrieve the corresponding PrefixHit."""
        # Example used below, continuing record's: a later step, save 20, wrote only slots 6-7,
        # with the rows 0-1, and now a request hits blocks [2, 3].
        if not len(block_ids):
            return PrefixHit([])

        # Given the block indices, lookup the KV slots that are owned by our block IDs.
        # e.g., blocks [2, 3] -> slots [4, 5, 6, 7],
        # since a given block b owns slots  [b * block_size, b * block_size + block_size - 1]
        blocks = torch.tensor(block_ids, dtype=torch.long)
        slots = (blocks[:, None] * self._block_size + torch.arange(self._block_size)).flatten()

        # Retrieve the save index where each slot maps to, as well as the row per slot
        # e.g. save_ids = [17, 17, 20, 20], rows = [3, 4, 0, 1]
        save_ids = self._save_of_slot[slots]
        rows = self._row_of_slot[slots]

        # Slot i starts a new slice unless it is the next row of the same save as slot i - 1.
        # e.g., new_slice = [True, False, True, False], so starts = [0, 2] and ends = [2, 4]
        new_slice = torch.ones(len(slots), dtype=torch.bool)
        # NOTE: We need to look for jumps in saves and rows, because we can have row jumps within
        # a save if we have successive requests, where the longer request is adding onto an already
        # cached prefix.
        new_slice[1:] = (save_ids[1:] != save_ids[:-1]) | (rows[1:] != rows[:-1] + 1)
        first_slots = new_slice.nonzero().flatten()
        starts = first_slots.tolist()
        ends = [*starts[1:], len(slots)]

        # Each slice reads from its first slot's save and mapped rows, e.g. save 17 rows 3-4 and save 20 rows 0-1.
        slices = zip(save_ids[first_slots].tolist(), rows[first_slots].tolist(), starts, ends)
        return PrefixHit(
            [SavedSlice(self._outputs_of_save.get(save_id), row, end - start) for save_id, row, start, end in slices]
        )
