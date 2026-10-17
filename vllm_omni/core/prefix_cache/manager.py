# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reuses stage outputs (hidden states and per-token multimodal outputs) on prefix cache hits.

On a hit, vLLM doesn't run the model on the cached tokens, so any hidden states or output tensors
that need to be part of the stage's output payload corresponding to the cached tokens need to be
separately retrieved for Omni models.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheEventKind, PrefixCacheStep, PrefixCacheWriteLayout
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheBatchId,
    PrefixCacheConfig,
    ReqId,
    StageCacheOutputs,
    TensorName,
)
from vllm_omni.core.prefix_cache.slot_index import PrefixHit, SlotIndex


def _is_per_token_tensor(val: Any, n: int, padded: int) -> bool:
    """2D+ tensor whose first dim is this batch's token count (``n`` or padded).

    True means callers may take ``val[:n]``. Leftover tensors (``codes.ref``),
    lists, and other shapes are False.
    """
    return isinstance(val, torch.Tensor) and val.ndim >= 2 and int(val.shape[0]) in (n, padded)


def _on_device(value: Any) -> bool:
    """Whether any tensor in a (nested) payload is outside CPU memory."""
    if isinstance(value, torch.Tensor):
        return value.device.type != "cpu"
    if isinstance(value, dict):
        return any(_on_device(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_on_device(item) for item in value)
    return False


@dataclass
class _PrefixCacheBatch:
    """What save_outputs keeps for merge_batch."""

    spans: dict[ReqId, tuple[int, int]]  # each request's rows in this batch
    outputs: dict[TensorName, torch.Tensor]  # per-token outputs, one row per scheduled token
    other_mm: dict[TensorName, Any]  # multimodal outputs that aren't per token, passed through
    hits: dict[ReqId, PrefixHit] = field(default_factory=dict)


class OmniPrefixCacheManager:
    def __init__(self, config: PrefixCacheConfig) -> None:
        self._config = config
        self._policy = ModelCachePolicy()
        self._index = SlotIndex(config.num_blocks, config.block_size)
        # This batch's hits, as each starting request's hit blocks. They are looked up in save,
        # after this batch's own outputs are indexed, so a hit on blocks computed in the same
        # batch finds them.
        self._hit_blocks: dict[ReqId, list[int]] = {}
        self._cur_num_scheduled: dict[ReqId, int] = {}
        # Saved batches waiting for merge_batch or skip_merge, which the async output thread
        # may call while the engine thread saves the next batch.
        self._batches: dict[PrefixCacheBatchId, _PrefixCacheBatch] = {}
        self._batches_lock = threading.Lock()
        self._next_batch_id: PrefixCacheBatchId = 1

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy

    def new_step_starts(self, step: PrefixCacheStep) -> None:
        """Handle one immutable adapter-produced step.

        Engine thread only; exactly once per real step. Registers new-request prefix
        hits (copying their block tables).
        """
        if not isinstance(step, PrefixCacheStep):
            raise TypeError("new_step_starts requires a PrefixCacheStep")
        events = step.events
        # Copy this arrival's prefix-hit block ids. The adapter captures
        # them before _update_states; after that point they sit
        # on the live request and grow as decode allocates more blocks.
        # save_outputs must not reread that live table.
        self._hit_blocks.clear()
        for event in events:
            if event.kind not in (PrefixCacheEventKind.STARTED,):
                continue
            req_id = event.req_id
            # The adapter alone classifies starts versus continuation or resume.
            num_computed = int(event.hit_end)
            if num_computed > 0:
                # block_ids is per-kv-group; group 0 only.
                block_groups = event.block_ids
                if not block_groups or not block_groups[0]:
                    # Fail at the cause: a hit we cannot snapshot now would
                    # crash at save_outputs time with less context (save_outputs is
                    # forbidden from reading the live batch).
                    raise OmniPrefixCacheUnmatchError(
                        f"prefix hit for req {req_id} ({num_computed} tokens) carries no block_ids"
                    )
                bs = self._config.block_size
                if num_computed % bs != 0:
                    raise OmniPrefixCacheUnmatchError(
                        f"prefix hit not block aligned (req={req_id}, hit_upto={num_computed}, block_size={bs})"
                    )
                needed = num_computed // bs
                if len(block_groups[0]) < needed:
                    raise OmniPrefixCacheUnmatchError(
                        f"prefix hit for req {req_id} ({num_computed} tokens) has "
                        f"{len(block_groups[0])} group-0 blocks, need {needed}"
                    )
                self._hit_blocks[req_id] = list(block_groups[0][:needed])
        self._cur_num_scheduled = dict(step.scheduled_tokens)

    def save_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any],
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        write_layout: PrefixCacheWriteLayout,
    ) -> PrefixCacheBatchId:
        """Index this batch's outputs by KV slot and look up its hits; returns the batch id.

        Engine thread only, after the forward. With async output, the CPU tensors may still be
        filling from the async output copy; nothing here reads their values, and the output builder
        waits for the copy before calling merge_batch.

        Pass the batch id to merge_batch, or to skip_merge when nothing reads this batch's outputs.
        Either call releases the batch, which is held until then; a second call raises.
        """
        # Rows are kept by reference: a device tensor would hold GPU memory, and CUDA graph outputs are overwritten.
        if _on_device(hidden_states) or _on_device(mm_outputs):
            raise OmniPrefixCacheUnmatchError("save_outputs takes CPU tensors; copy device outputs to the CPU first")
        n = num_tokens_unpadded
        if len(write_layout.slots_cpu) != n:
            raise OmniPrefixCacheUnmatchError(f"write layout has {len(write_layout.slots_cpu)} slots, need {n}")
        spans = {write.req_id: (write.row_start, write.row_end) for write in write_layout.writes}
        for req_id in self._hit_blocks:
            start, end = spans.get(req_id, (0, 0))
            if end - start != self._cur_num_scheduled.get(req_id, 0):
                raise OmniPrefixCacheUnmatchError(
                    f"scheduled token count mismatch for req {req_id}: "
                    f"step={self._cur_num_scheduled.get(req_id, 0)}, layout={end - start}"
                )

        outputs: dict[TensorName, torch.Tensor] = {}
        other_mm: dict[TensorName, Any] = {}
        if self._policy.needs_full_hidden_states and n:
            rows = 0 if hidden_states is None or hidden_states.ndim < 2 else int(hidden_states.shape[0])
            if rows < n:
                raise OmniPrefixCacheUnmatchError(f"hidden_states has {rows} rows, need {n}")
            outputs[HIDDEN_KEY] = hidden_states[:n]
        for key, val in mm_outputs.items():
            if n and _is_per_token_tensor(val, n, num_tokens_padded):
                outputs[key] = val[:n]
            else:
                other_mm[key] = val
        if n:
            self._index.record(outputs, write_layout.slots_cpu)

        hits = {}
        for req_id, blocks in self._hit_blocks.items():
            hit = self._index.lookup(blocks)
            # merge_batch puts the hit's rows in front of this batch's for every key the batch saved.
            for key in outputs:
                if not hit.all_slots_saved(key):
                    raise OmniPrefixCacheUnmatchError(
                        f"hit span for req {req_id} key={key} is not readable (slots with no saved rows)"
                    )
            hits[req_id] = hit
        self._hit_blocks.clear()

        with self._batches_lock:
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            self._batches[batch_id] = _PrefixCacheBatch(spans, outputs, other_mm, hits)
        return batch_id

    def merge_batch(self, batch_id: PrefixCacheBatchId, req_ids: list[ReqId]) -> StageCacheOutputs:
        """Per-request merged outputs for the batch saved as `batch_id`.

        Any thread, once the batch's CPU outputs have finished copying.
        `req_ids` must be (a subset of) the save-time snapshot;
        an outside id means the caller is reading the live batch.
        A request without a hit is a plain miss and gets exactly
        this batch's rows — normal path, nothing logged.
        """
        batch = self._take(batch_id)
        unknown = set(req_ids) - set(batch.spans)
        if unknown:
            raise OmniPrefixCacheUnmatchError(
                f"merge_batch({batch_id}) got req ids outside the save snapshot: {sorted(unknown)[:8]}"
            )

        def merged(key: TensorName) -> dict[ReqId, torch.Tensor]:
            per_req = {}
            for req_id in req_ids:
                start, end = batch.spans[req_id]
                rows = batch.outputs[key][start:end]
                hit = batch.hits.get(req_id)
                per_req[req_id] = rows if hit is None else hit.cat(key, rows)
            return per_req

        hidden = merged(HIDDEN_KEY) if HIDDEN_KEY in batch.outputs else None
        mm = {key: merged(key) for key in batch.outputs if key != HIDDEN_KEY}
        mm.update(self._split_other_mm(batch, req_ids))
        return StageCacheOutputs(hidden_states=hidden, mm_outputs=mm)

    def skip_merge(self, batch_id: PrefixCacheBatchId) -> None:
        """Release a saved batch nothing will merge. Its outputs stay cached for later hits.

        Any thread; same exactly-once contract as merge_batch (unknown or
        duplicate id raises).
        """
        self._take(batch_id)

    def _take(self, batch_id: PrefixCacheBatchId) -> _PrefixCacheBatch:
        with self._batches_lock:
            batch = self._batches.pop(batch_id, None)
        if batch is None:
            raise OmniPrefixCacheUnmatchError(f"batch {batch_id} missing; already consumed or never saved")
        return batch

    @staticmethod
    def _split_other_mm(batch: _PrefixCacheBatch, req_ids: list[ReqId]) -> dict[TensorName, dict[ReqId, Any]]:
        """Each request's part of the multimodal outputs that aren't per token."""
        if not batch.other_mm:
            return {}
        # Imports vllm, which the rest of this package avoids at import time.
        from vllm_omni.utils.mm_outputs import to_payload_element

        order = list(batch.spans)
        total = sum(end - start for start, end in batch.spans.values())
        return {
            key: {
                req_id: to_payload_element(
                    val,
                    order.index(req_id),
                    start=batch.spans[req_id][0],
                    end=batch.spans[req_id][1],
                    pass_lists_through=True,
                    seq_len=total,
                )
                for req_id in req_ids
            }
            for key, val in batch.other_mm.items()
        }
