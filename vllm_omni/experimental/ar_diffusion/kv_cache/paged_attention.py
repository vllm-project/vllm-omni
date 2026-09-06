# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paged self-attention helpers for AR-Diffusion KV reuse."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping

if TYPE_CHECKING:
    from vllm.v1.attention.backend import MultipleOf

_LAYER_IDX_TENSORS: dict[int, torch.Tensor] = {}


def _layer_idx_tensor(layer_idx: int) -> torch.Tensor:
    t = _LAYER_IDX_TENSORS.get(layer_idx)
    if t is None:
        t = torch.tensor(layer_idx, dtype=torch.int64)
        _LAYER_IDX_TENSORS[layer_idx] = t
    return t


class ARDiffusionPagedLayerInputs(NamedTuple):
    """Compiled-region payload for one layer's paged self-attention.

    A NamedTuple of plain tensors + ints so ``torch.compile`` treats every field
    as a pytree graph input (no object-attribute guards, no recompiles when only
    tensor *values* change). All layers of one KV branch forward share the same
    metadata tensor objects, built once by ``prepare()``.

    ``layer_idx`` is a 0-dim CPU tensor, NOT a python int: all 40 DiT blocks
    share one compiled code object, and an int here becomes a per-layer dynamo
    value guard (``layer_idx == k``) — 40 cache variants that blow the
    recompile limit. A tensor input guards on shape/dtype only, so one graph
    serves every layer.
    """

    layer_idx: torch.Tensor
    key_pool: torch.Tensor
    value_pool: torch.Tensor
    block_size: int
    seq_len: int
    video_slots: torch.Tensor
    action_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class ARDiffusionPagedForwardContext:
    """Mutable KV-branch state shared by all layer contexts in one forward."""

    kv_cache: Any
    adapter: Any
    kv_branch: str
    history_block_ids: list[int]
    seq_len: int
    commit_current: bool
    max_video_tokens: int
    # Which scratch region this session's uncommitted chunk writes to. Sessions
    # that may be coalesced into one forward must differ here or they overwrite
    # each other's current K/V.
    scratch_slot: int = 0
    current_video_block_ids: list[int] = field(default_factory=list)
    current_video_slot_mapping: torch.Tensor | None = None
    action_scratch_block_ids: list[int] = field(default_factory=list)
    action_slot_mapping: torch.Tensor | None = None
    query_len: int = 0
    kv_len: int = 0
    # Tokens already committed when this forward began. Recorded rather than
    # derived from the block count: the last block of the history is only
    # partly written when a chunk is not a whole number of blocks, so
    # blocks * block_size overstates it and would make attention read past
    # what was written.
    _history_tokens: int = 0
    # Scratch blocks the current video tokens occupy. Not the same as
    # len(current_video_block_ids): when the history ends mid-block, the first
    # entry there is the history's own tail block, which is managed, not
    # scratch. Action K/V is placed after this, so it must not count it.
    _scratch_blocks_used: int = 0
    _allocated_video: bool = False
    _committed: bool = False
    _action_len: int = 0
    # Set once by prepare(); shared by all layers of the KV branch forward.
    block_table: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    _prepared: bool = False

    @property
    def block_size(self) -> int:
        return int(self.kv_cache.block_size)

    @property
    def chunk_size(self) -> int:
        """Tokens in one chunk -- one latent frame. The eviction unit."""
        return int(self.kv_cache.spec.chunk_size)

    @property
    def start_offset(self) -> int:
        """Where in its first block this forward's tokens begin.

        Non-zero exactly when the committed history does not end on a block
        boundary, which is the normal case once a frame is not a whole number
        of blocks: 1560 tokens per frame against 16-token blocks leaves the
        history 8 slots into its last block on every odd-numbered chunk.
        """
        return int(self.adapter.num_computed_tokens) % self.block_size

    @property
    def max_video_blocks(self) -> int:
        """Blocks the visible window spans, rounded up.

        Rounding up is what keeps the block table a fixed shape: the window is
        a token count, and flooring it would understate the capacity whenever
        it is not a whole number of blocks.
        """
        return -(-self.max_video_tokens // self.block_size)

    @property
    def num_current_video_blocks(self) -> int:
        """Blocks this forward's tokens occupy, counted from where they start.

        A chunk need not be a whole number of blocks, so the count depends on
        the offset it begins at, not only on its length: thirty tokens
        starting at position thirty span three sixteen-token blocks, not two.
        Rounding up leaves the tail of the last block unwritten, which is only
        safe because nothing reads past ``kv_len`` -- see
        :meth:`video_block_table`.

        Both paths count the same way. The committing path writes straight
        after the history; the scratch path is made to line up with it by
        :meth:`ensure_video_slots`.
        """
        return -(-(self.start_offset + self.seq_len) // self.block_size)

    def ensure_video_slots(self, device: torch.device) -> None:
        """Allocate/write targets for the current video tokens, once per KV branch."""
        if self._allocated_video:
            return

        n_blocks = self.num_current_video_blocks
        self._history_tokens = int(self.adapter.num_computed_tokens)
        if self.commit_current:
            start = int(self.adapter.num_computed_tokens)
            self.kv_cache.allocate_token_slots(self.adapter, self.seq_len)
            table = self.kv_cache.block_table(self.adapter)
            start_block = start // self.block_size
            self.current_video_block_ids = [int(b) for b in table[start_block : start_block + n_blocks]]
            positions = torch.arange(start, start + self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = compute_slot_mapping(table, positions, self.block_size).to(device=device)
        else:
            # The scratch region cannot simply start at slot zero. The kernel
            # reads the history and this chunk as one contiguous run, so if the
            # history stopped mid-block, starting here at zero would leave the
            # rest of that block unwritten and the kernel would read those dead
            # slots as if they were tokens, shifting the whole sequence.
            #
            # Instead the chunk begins where the history left off, spilling its
            # first few tokens into the free tail of the history's own last
            # block. That block is already this session's -- blocks are
            # allocated whole -- and those slots hold nothing yet; the
            # committing forward overwrites the same slots with the clean K/V
            # later, so the committed state is unchanged either way.
            # A history whose tail block is no longer resident has nothing to
            # spill into; the window then starts at this chunk and zero is right.
            offset = self.start_offset if self.history_block_ids else 0
            self._scratch_blocks_used = n_blocks - (1 if offset else 0)
            scratch_ids = self.kv_cache.scratch_block_ids(
                self.kv_branch, 0, self._scratch_blocks_used, slot=self.scratch_slot
            )
            tail_block = [self.history_block_ids[-1]] if offset else []
            self.current_video_block_ids = tail_block + scratch_ids
            positions = torch.arange(offset, offset + self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = compute_slot_mapping(
                self.current_video_block_ids,
                positions,
                self.block_size,
            ).to(device=device)
        self._allocated_video = True

    def ensure_action_slots(self, action_len: int, device: torch.device) -> None:
        """Reserve scratch slots for action/state K/V, if present."""
        if action_len <= 0:
            self.action_scratch_block_ids = []
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
            self._action_len = 0
            return

        self.ensure_video_slots(device)
        if self.action_slot_mapping is not None and self._action_len == action_len:
            return

        action_blocks = (action_len + self.block_size - 1) // self.block_size
        # Count scratch blocks, not entries: the video list may lead with the
        # history's tail block, which lives in the managed pool.
        scratch_offset = 0 if self.commit_current else self._scratch_blocks_used
        self.action_scratch_block_ids = self.kv_cache.scratch_block_ids(
            self.kv_branch,
            scratch_offset,
            action_blocks,
            slot=self.scratch_slot,
        )
        positions = torch.arange(action_len, dtype=torch.long)
        self.action_slot_mapping = compute_slot_mapping(
            self.action_scratch_block_ids,
            positions,
            self.block_size,
        ).to(device=device)
        self._action_len = action_len

    def video_block_table(self, device: torch.device) -> tuple[list[int], int]:
        """Blocks the attention may read, and how many of their tokens are live.

        The window is expressed in tokens and converted to blocks here, rather
        than assuming one block per frame. Both boundaries round *up*: a sink
        of nine frames whose tokens do not fill a whole number of blocks keeps
        the block that straddles the boundary, so the window can retain up to
        ``block_size - 1`` tokens more than it strictly needs. That is
        deliberate -- rounding down would drop tokens the window is supposed to
        keep, and a slightly wider window is a far smaller error than a
        truncated one.

        The returned length is what the kernel treats as this sequence's KV
        length, so it must never exceed the tokens actually written. The tail
        of the final block is unwritten whenever a chunk is not a whole number
        of blocks, and reading it would mix uninitialised memory into the
        attention.
        """
        self.ensure_video_slots(device)
        # Order-preserving union, not concatenation. When a chunk is not a
        # whole number of blocks, the block holding the end of the history also
        # holds the start of this chunk, so it appears in both lists. Reading
        # it twice would feed those tokens to attention twice and drop the same
        # number from the end -- silently, since the shapes stay right.
        all_video_blocks = list(dict.fromkeys(self.history_block_ids + self.current_video_block_ids))
        max_video_blocks = self.max_video_blocks
        sink_blocks = -(-(int(self.kv_cache.spec.sink_chunks) * self.chunk_size) // self.block_size)

        # What was actually written: committed history, capped by whatever the
        # window still holds, plus this forward's own tokens.
        resident_history = min(self._history_tokens, len(self.history_block_ids) * self.block_size)
        if len(all_video_blocks) <= max_video_blocks:
            visible_video_blocks = all_video_blocks
            live_tokens = resident_history + self.seq_len
        else:
            tail_blocks = max(max_video_blocks - sink_blocks, 0)
            visible_video_blocks = all_video_blocks[:sink_blocks]
            if tail_blocks:
                visible_video_blocks += all_video_blocks[-tail_blocks:]
            live_tokens = min(resident_history + self.seq_len, len(visible_video_blocks) * self.block_size)

        video_len = min(live_tokens, len(visible_video_blocks) * self.block_size)
        return visible_video_blocks, video_len

    def build_block_table(
        self,
        *,
        action_len: int,
        query_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build FlashAttention block-table metadata for one self-attn call.

        The block table is tail-padded to a fixed width and ``max_seq_len`` is a
        constant upper bound, so across window growth only tensor *values* change
        — tensor shapes and the int consts stay stable for ``torch.compile``
        (and, later, CUDA-graph capture). The kernel only dereferences the first
        ``ceil(seq_lens/block_size)`` entries, so padding is never read.
        """
        video_blocks, video_len = self.video_block_table(device)
        self.ensure_action_slots(action_len, device)
        action_blocks = self.action_scratch_block_ids if action_len > 0 else []
        block_ids = video_blocks + action_blocks
        if not block_ids:
            raise RuntimeError("AR-Diffusion paged attention needs at least current video KV blocks")

        # Fixed capacity: full visible video window + one action-capacity block.
        action_capacity_blocks = max(1, (action_len + self.block_size - 1) // self.block_size)
        # Both of these count in blocks. Deriving them from the token count by
        # flooring would understate the capacity whenever the window is not a
        # whole number of blocks: the width would fall back on len(block_ids)
        # and start tracking how full the window is -- the exact shape churn
        # the fixed width exists to prevent -- and max_seq_len could come out
        # below the kv_len actually being passed, by up to block_size - 1.
        capacity_blocks = self.max_video_blocks + action_capacity_blocks
        width = max(capacity_blocks, len(block_ids))
        padded = block_ids + [0] * (width - len(block_ids))

        self.query_len = int(query_len)
        self.kv_len = int(video_len + action_len)
        max_seq_len = int(capacity_blocks * self.block_size)
        block_table = torch.tensor([padded], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, self.query_len], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([self.kv_len], dtype=torch.int32, device=device)
        return block_table, query_start_loc, seq_lens, self.query_len, max_seq_len

    def prepare(self, device: torch.device, action_len: int, query_len: int) -> None:
        """Host-side, once-per-KV-branch setup (called OUTSIDE torch.compile).

        Allocates the current video/action slots (still lazy: only the KV branch a
        CFG-parallel rank actually runs reaches its ``_forward_blocks``), builds
        the padded block-table metadata ONCE for all layers, and publishes the
        pool registry for the fused custom op. The compiled per-layer code then
        only consumes prebuilt tensors via ``ARDiffusionPagedLayerInputs``.
        """
        if getattr(self, "_prepared", False):
            return
        self.ensure_video_slots(device)
        (
            self.block_table,
            self.query_start_loc,
            self.seq_lens,
            self.max_query_len,
            self.max_seq_len,
        ) = self.build_block_table(action_len=action_len, query_len=query_len, device=device)
        if self.action_slot_mapping is None:
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
        self._prepared = True

    def layer_inputs(self, layer_idx: int) -> ARDiffusionPagedLayerInputs:
        if not getattr(self, "_prepared", False):
            raise RuntimeError("ARDiffusionPagedForwardContext.layer_inputs() before prepare()")
        return ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(layer_idx),
            key_pool=self.kv_cache._k_pools[layer_idx],
            value_pool=self.kv_cache._v_pools[layer_idx],
            block_size=int(self.kv_cache.block_size),
            seq_len=int(self.seq_len),
            video_slots=self.current_video_slot_mapping,
            action_slots=self.action_slot_mapping,
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=int(self.max_query_len),
            max_seq_len=int(self.max_seq_len),
        )

    def mark_committed(self) -> None:
        self._committed = True


@dataclass
class ARDiffusionPagedLayerContext:
    """Layer-specific handle passed through a model's ``kv_cache`` slot."""

    is_ar_diffusion_paged_context: ClassVar[bool] = True
    layer_idx: int
    forward_ctx: ARDiffusionPagedForwardContext

    @property
    def kv_cache(self):
        return self.forward_ctx.kv_cache

    @property
    def adapter(self):
        return self.forward_ctx.adapter

    @property
    def kv_branch(self) -> str:
        return self.forward_ctx.kv_branch

    @property
    def history_block_ids(self) -> list[int]:
        return self.forward_ctx.history_block_ids

    @property
    def current_video_block_ids(self) -> list[int]:
        return self.forward_ctx.current_video_block_ids

    @property
    def current_video_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.current_video_slot_mapping

    @property
    def action_scratch_block_ids(self) -> list[int]:
        return self.forward_ctx.action_scratch_block_ids

    @property
    def action_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.action_slot_mapping

    @property
    def seq_len(self) -> int:
        return self.forward_ctx.seq_len

    @property
    def query_len(self) -> int:
        return self.forward_ctx.query_len

    @property
    def kv_len(self) -> int:
        return self.forward_ctx.kv_len

    @property
    def commit_current(self) -> bool:
        return self.forward_ctx.commit_current

    def to_layer_inputs(self) -> ARDiffusionPagedLayerInputs:
        """Compiled-region payload; requires ``forward_ctx.prepare()`` first."""
        return self.forward_ctx.layer_inputs(self.layer_idx)


@dataclass(frozen=True)
class PagedSequenceMetadata:
    """One sequence's contribution to a coalesced forward.

    ``block_ids`` are the physical blocks holding its KV in visit order,
    ``kv_len`` how many of those tokens are live, and ``query_len`` how many
    query tokens it contributes to the batch.
    """

    block_ids: tuple[int, ...]
    kv_len: int
    query_len: int

    def __post_init__(self) -> None:
        if not self.block_ids:
            raise ValueError("A paged sequence needs at least one block.")
        if self.kv_len <= 0 or self.query_len <= 0:
            raise ValueError("kv_len and query_len must be positive.")


def batch_paged_metadata(
    sequences: Sequence[PagedSequenceMetadata],
    *,
    block_size: int,
    device: torch.device,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Merge per-sequence tables into one varlen batch.

    Returns the same five values ``build_block_table`` returns for a single
    sequence, so a coalesced forward reaches the attention kernel through the
    identical path -- the kernel already consumes ``block_table`` per row and
    ``seq_lens`` per sequence.

    Every row is padded to the widest member. Padding is never read: the
    kernel dereferences only ``ceil(seq_lens / block_size)`` entries, which is
    what keeps a short session from mixing in a neighbour's blocks. That is
    asserted in ``tests/diffusion/ar_diffusion/test_paged_attention.py`` rather
    than assumed, because the failure it prevents -- one session reading
    another's KV -- produces a plausible but wrong result that no timing metric
    can detect.
    """
    if not sequences:
        raise ValueError("A coalesced batch needs at least one sequence.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    for sequence in sequences:
        needed = -(-sequence.kv_len // block_size)
        if needed > len(sequence.block_ids):
            raise ValueError(
                f"kv_len={sequence.kv_len} needs {needed} blocks but only {len(sequence.block_ids)} were supplied."
            )

    resolved_width = max(len(sequence.block_ids) for sequence in sequences)
    if width is not None:
        if width < resolved_width:
            raise ValueError(f"width={width} is narrower than the widest sequence ({resolved_width}).")
        resolved_width = width

    table = torch.tensor(
        [list(s.block_ids) + [0] * (resolved_width - len(s.block_ids)) for s in sequences],
        dtype=torch.int32,
        device=device,
    )
    starts = [0]
    for sequence in sequences:
        starts.append(starts[-1] + sequence.query_len)
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([s.kv_len for s in sequences], dtype=torch.int32, device=device)
    max_query_len = max(s.query_len for s in sequences)
    return table, query_start_loc, seq_lens, max_query_len, resolved_width * block_size


def _reject_overlapping_writes(video_slots: torch.Tensor, action_slots: torch.Tensor, num_sessions: int) -> None:
    """Refuse a batch whose sessions would write to the same pool slots.

    Scratch blocks are handed out per KV branch, not per session
    (``ARDiffusionKVCache.scratch_block_ids`` offsets by branch index only), so
    two sessions preparing an uncommitted chunk on the same branch receive the
    *same* scratch blocks. Coalescing them would make each overwrite the
    other's current K/V and then attend over the survivor.

    That failure is invisible downstream -- the shapes are right, the kernel
    succeeds, and the output is a plausible tensor -- so it is checked here
    rather than left to whoever reads the frames. Session-indexed scratch is
    what makes such a batch legal; until then, refuse it.
    """
    if num_sessions < 2:
        return
    slots = torch.cat((video_slots, action_slots), dim=0)
    if slots.numel() and torch.unique(slots).numel() != slots.numel():
        raise ValueError(
            "Coalesced sessions write to overlapping KV slots. Sessions preparing an uncommitted "
            "chunk share their KV branch's scratch blocks, so they cannot be batched until scratch "
            "is indexed per session as well as per branch."
        )


@dataclass(frozen=True)
class CoalescedPagedForward:
    """Several sessions' prepared contexts merged into one forward's metadata.

    Mirrors :class:`ARDiffusionPagedForwardContext`: :meth:`merge` does the
    host-side work once per forward and :meth:`layer_inputs` hands each layer a
    payload that only swaps in that layer's pools. Merging inside
    ``layer_inputs`` instead would repeat the concatenations and the overlap
    check once per DiT block.

    Sessions are laid out along the *sequence* dimension, not the batch
    dimension, so a coalesced forward still runs at ``batch_size=1`` with
    ``query_start_loc`` marking the boundaries. That is forced rather than
    chosen: the rotary layer documents that "all batch elements share the same
    rotary position encoding", so sessions sitting at different chunk indices
    cannot be stacked along the batch dimension without changing it, while
    concatenating along the sequence dimension needs no change at all -- each
    session's own cos/sin table simply concatenates too.
    """

    kv_cache: Any
    block_size: int
    seq_len: int
    video_slots: torch.Tensor
    action_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int

    @classmethod
    def merge(cls, contexts: Sequence[ARDiffusionPagedForwardContext]) -> CoalescedPagedForward:
        """Merge prepared contexts in batch order.

        Slot mappings concatenate in that same order, which must be the order
        the caller concatenates K/V rows in -- the fused write op indexes the
        pool with one flat slot tensor.

        The already-built per-session tables are merged rather than rebuilt
        from block ids through :func:`batch_paged_metadata`. Rebuilding would
        re-derive the padded width from the *live* blocks, so the table would
        grow as each session's window fills and the compiled graph would see
        changing shapes. Every context has already padded itself to a fixed
        capacity; taking the widest of those keeps the shape stable for exactly
        as long as the single-session path does.
        """
        if not contexts:
            raise ValueError("A coalesced forward needs at least one session context.")
        tables: list[torch.Tensor] = []
        starts: list[torch.Tensor] = []
        for context in contexts:
            table, start = context.block_table, context.query_start_loc
            if not getattr(context, "_prepared", False) or table is None or start is None:
                raise RuntimeError("CoalescedPagedForward.merge() before prepare() on every context")
            tables.append(table)
            starts.append(start)

        first = contexts[0]
        block_size = int(first.kv_cache.block_size)
        for context in contexts[1:]:
            if context.kv_cache is not first.kv_cache:
                raise ValueError("Coalesced sessions must share one KV pool; got two different allocations.")
            if int(context.kv_cache.block_size) != block_size:
                raise ValueError("Coalesced sessions must share one block_size.")

        width = max(int(table.shape[1]) for table in tables)
        block_table = torch.cat(
            [torch.nn.functional.pad(table, (0, width - int(table.shape[1]))) for table in tables],
            dim=0,
        )

        query_lens = [int(context.query_len) for context in contexts]
        query_start_loc = torch.tensor(
            [0, *itertools.accumulate(query_lens)],
            dtype=starts[0].dtype,
            device=starts[0].device,
        )
        max_seq_len = max(int(context.max_seq_len) for context in contexts)
        if width * block_size < max_seq_len:
            raise AssertionError(
                f"Coalesced block table holds {width * block_size} tokens but max_seq_len is {max_seq_len}."
            )

        video_slots = torch.cat([context.current_video_slot_mapping for context in contexts], dim=0)
        action_slots = torch.cat([context.action_slot_mapping for context in contexts], dim=0)
        _reject_overlapping_writes(video_slots, action_slots, len(contexts))

        return cls(
            kv_cache=first.kv_cache,
            block_size=block_size,
            seq_len=sum(int(context.seq_len) for context in contexts),
            video_slots=video_slots,
            action_slots=action_slots,
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=torch.cat([context.seq_lens for context in contexts], dim=0),
            max_query_len=max(query_lens),
            max_seq_len=max_seq_len,
        )

    def layer_inputs(self, layer_idx: int) -> ARDiffusionPagedLayerInputs:
        """Compiled-region payload for one layer of the coalesced forward."""
        return ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(layer_idx),
            key_pool=self.kv_cache._k_pools[layer_idx],
            value_pool=self.kv_cache._v_pools[layer_idx],
            block_size=self.block_size,
            seq_len=self.seq_len,
            video_slots=self.video_slots,
            action_slots=self.action_slots,
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
        )


def is_ar_diffusion_paged_context(value: object) -> bool:
    return isinstance(value, ARDiffusionPagedLayerContext)


def _reference_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    if causal:
        raise NotImplementedError("AR-Diffusion paged self-attention uses causal=False")
    outs: list[torch.Tensor] = []
    block_size = key_cache.shape[1]
    for i in range(seq_lens.shape[0]):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        q = query[q_start:q_end]
        positions = torch.arange(kv_len, device=query.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[i, logical_blocks].long()
        k = key_cache[physical_blocks, offsets]
        v = value_cache[physical_blocks, offsets]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * float(softmax_scale)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outs, dim=0)


_FA_VERSION_BY_HEAD_SIZE: dict[int, int] = {}


def _resolve_fa_version(head_size: int) -> int:
    # get_flash_attn_version -> current_platform.get_device_capability() is not
    # dynamo-traceable, and the answer is fixed per head size for the process.
    version = _FA_VERSION_BY_HEAD_SIZE.get(head_size)
    if version is None:
        try:
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

            version = int(get_flash_attn_version(requires_alibi=False, head_size=head_size) or 2)
        except Exception:
            version = 2
        _FA_VERSION_BY_HEAD_SIZE[head_size] = version
    return version


def supported_kernel_block_sizes() -> list[int | MultipleOf]:
    """Block sizes the kernel this module dispatches to will accept.

    Same shape as vLLM's ``AttentionBackend.get_supported_kernel_block_sizes``
    -- a plain int is that exact size, ``MultipleOf(b)`` is any positive
    multiple of ``b`` -- but answered here rather than read off a backend,
    because AR-Diffusion does not go through backend selection. It calls
    ``flash_attn_varlen_func`` itself, choosing between vLLM's CUDA build and
    ROCm's AITER a few lines below, and only this module knows which.

    It lives next to that choice so there is one place to change. The
    constraint is a property of the kernel, not of the card, and the kernels
    reachable from here agree on 16 today: vLLM's CUDA FlashAttention, ROCm
    AITER, and upstream ``flash_attn`` all advertise ``MultipleOf(16)``. Other
    backends in the same tree do not -- ``hpc_attn`` accepts only 64, and
    FlashInfer advertises pages of 128 or more solely on Blackwell -- so a
    caller must treat this as data to be queried, never as the number 16.
    """
    from vllm.v1.attention.backend import MultipleOf

    return [MultipleOf(16)]


def _rocm_flash_attn_varlen_func():
    """Resolve a ROCm varlen kernel, preferring AITER when available.

    The caller gathers paged KV into packed tensors before invoking this
    function, avoiding AITER releases whose ``block_table`` kernel is broken.
    """
    try:
        from aiter import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func


def ar_diffusion_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """Run non-causal paged attention over a vLLM block table.

    ``query`` may be ``(B, L, H, D)`` or already flattened as ``(T, H, D)``.
    ``key_cache`` / ``value_cache`` are ``(num_blocks, block_size, H, D)``.
    """
    batched = query.dim() == 4
    if batched:
        batch, q_len = query.shape[:2]
        query_flat = query.reshape(batch * q_len, *query.shape[2:])
    else:
        query_flat = query

    if not query_flat.is_cuda:
        out = _reference_paged_attention(
            query_flat,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            softmax_scale,
            causal=causal,
        )
    elif torch.version.hip is not None:
        # vllm.vllm_flash_attn contains CUDA-only extensions. ROCm's AITER and
        # upstream flash-attn expose the standard cu_seqlens_k API instead of
        # vLLM's seqused_k/fa_version API. The ROCm flash-attn paged kernel also
        # requires 128-token blocks, while AR-Diffusion uses frame-aligned
        # 16-token blocks, so gather the visible blocks on-device first.
        flash_attn_varlen_func = _rocm_flash_attn_varlen_func()
        cu_seqlens_k = torch.cat([seq_lens.new_zeros(1), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)])
        positions = torch.arange(int(max_seq_len), device=query_flat.device)
        logical_blocks = torch.div(positions, key_cache.shape[1], rounding_mode="floor")
        offsets = positions % key_cache.shape[1]
        physical_blocks = block_table[:, logical_blocks].long()
        gathered_k = key_cache[physical_blocks, offsets]
        gathered_v = value_cache[physical_blocks, offsets]
        valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        packed_k = gathered_k[valid]
        packed_v = gathered_v[valid]
        out = flash_attn_varlen_func(
            q=query_flat,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_query_len),
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    else:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        fa_version = _resolve_fa_version(query_flat.shape[-1])

        out = torch.empty_like(query_flat)
        flash_attn_varlen_func(
            q=query_flat,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=int(max_query_len),
            seqused_k=seq_lens,
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
            block_table=block_table,
            fa_version=fa_version,
        )

    if batched:
        return out.reshape(query.shape)
    return out


# ── Fused write+attend custom op (torch.compile-safe) ──────────────────────
#
# One opaque op per layer keeps the compiled DiT block fullgraph: dynamo treats
# it as a single graph node (no eager island, no graph breaks), and the K/V slot
# writes happen inside the op so write→read ordering with the block-table kernel
# is internal. The flat pools are explicit mutable inputs: Inductor/CUDA Graph
# must track their storage lifetime instead of observing an undeclared mutation
# through a process-global registry.
def _paged_write_attn_impl(
    query: torch.Tensor,
    k_curr: torch.Tensor,
    v_curr: torch.Tensor,
    k_act: torch.Tensor | None,
    v_act: torch.Tensor | None,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_size: int,
    video_slots: torch.Tensor,
    action_slots: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    key_pool[video_slots] = k_curr.to(key_pool.dtype)
    value_pool[video_slots] = v_curr.to(value_pool.dtype)
    if k_act is not None and v_act is not None and k_act.shape[0] > 0:
        key_pool[action_slots] = k_act.to(key_pool.dtype)
        value_pool[action_slots] = v_act.to(value_pool.dtype)
    key_cache = key_pool.unflatten(0, (-1, block_size))
    value_cache = value_pool.unflatten(0, (-1, block_size))
    return ar_diffusion_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        causal=False,
    )


# hasattr guard keeps registration idempotent across test re-imports that pop
# the module from sys.modules (same as sage_attn3.py).
if not hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn"):

    @torch.library.custom_op(
        "vllm_omni::ar_diffusion_paged_write_attn",
        mutates_args=("key_pool", "value_pool"),
    )
    def _paged_write_attn_op(
        query: torch.Tensor,
        k_curr: torch.Tensor,
        v_curr: torch.Tensor,
        k_act: torch.Tensor | None,
        v_act: torch.Tensor | None,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        block_size: int,
        video_slots: torch.Tensor,
        action_slots: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        return _paged_write_attn_impl(
            query,
            k_curr,
            v_curr,
            k_act,
            v_act,
            key_pool,
            value_pool,
            block_size,
            video_slots,
            action_slots,
            block_table,
            query_start_loc,
            seq_lens,
            max_query_len,
            max_seq_len,
            softmax_scale,
        )

    @_paged_write_attn_op.register_fake
    def _(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        key_pool,
        value_pool,
        block_size,
        video_slots,
        action_slots,
        block_table,
        query_start_loc,
        seq_lens,
        max_query_len,
        max_seq_len,
        softmax_scale,
    ):
        return torch.empty_like(query)


def paged_write_attn(
    inputs: ARDiffusionPagedLayerInputs, query, k_curr, v_curr, k_act, v_act, softmax_scale: float
) -> torch.Tensor:
    """Model-facing entry: routes through the custom op (traceable in fullgraph)."""
    return torch.ops.vllm_omni.ar_diffusion_paged_write_attn(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        inputs.key_pool,
        inputs.value_pool,
        inputs.block_size,
        inputs.video_slots,
        inputs.action_slots,
        inputs.block_table,
        inputs.query_start_loc,
        inputs.seq_lens,
        inputs.max_query_len,
        inputs.max_seq_len,
        softmax_scale,
    )
