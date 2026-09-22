"""Captured graphs for the Talker's prefill, bucketed by token count.

One Talker prefill forward is ~18.4 ms of host dispatch wrapped around 1-3 ms
of device work (three back-to-back forwards never accumulate more than 0.07 ms
of drain), and a request pays roughly one of them -- all but a couple of
milliseconds of it removable. The decode side of this problem was solved by
`fixed_kv_decode`; this module is the same idea for the prefill shape.

## Why a padded prefill replays exactly

The Talker prefill this targets is a single fresh request of 9-25 tokens, so
`{16, 32}` covers it. A graph captured at bucket ``B`` with
``actual_seq_lengths = [B]`` replays a real ``n <= B`` prefill bit-exactly for
the rows that matter:

* **Attention.** The FIA call is causal (``sparse_mode=3`` against the
  singleton 2048x2048 mask), so query row ``i`` attends to rows ``0..i`` only.
  Real rows ``0..n-1`` therefore never see a padding row; padding rows produce
  garbage that nothing reads (`make_omni_output` samples row ``n-1``).
* **KV cache.** ``reshape_and_cache`` writes all ``B`` rows through the
  persistent slot buffer. A fresh prefill's slots are contiguous from the
  start of its first block, and ``B <= 32 < block_size``, so rows ``n..B-1``
  land on the *same request's own future slots* -- positions the sequence has
  not reached. Attention never reads past ``seq_len``, and the first decode
  steps overwrite those slots with the real KV as the sequence grows.
* **Inputs.** The graph bakes the runner's own persistent ``inputs_embeds``
  and ``positions`` buffers, exactly as the decode graphs do. The runner
  writes rows ``0..n-1`` every step; rows ``n..B-1`` hold stale values whose
  garbage stays behind the causal mask.

The eligibility gate is deliberately narrow -- one request, ``PrefillNoCache``,
offset 0, ``n`` inside a captured bucket -- and everything else takes today's
eager path unchanged.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# The Talker's prefill is 9-25 tokens; a stage that prefills longer selects its
# own set through ``configure``.
TALKER_BUCKETS: tuple[int, ...] = (16, 32)
BUCKETS: tuple[int, ...] = TALKER_BUCKETS


def configure(buckets: tuple[int, ...]) -> None:
    global BUCKETS
    BUCKETS = tuple(sorted(buckets))


_LOGGED_REPLAY = False


class _State:
    """Per-process buffers and the captured-bucket registry."""

    def __init__(self) -> None:
        self.slot_buffers: dict[int, torch.Tensor] = {}
        self.aranges: dict[int, torch.Tensor] = {}
        self.workspaces: dict[int, torch.Tensor] = {}
        self.lse: dict[int, torch.Tensor] = {}
        self.captured: set[int] = set()
        # bucket -> (graph, hidden_out): graphs this module owns outright.
        self.graphs: dict[int, tuple[Any, torch.Tensor]] = {}


_state = _State()


def prewarm(device: torch.device) -> None:
    """Allocate every runtime buffer before any capture can start.

    An allocation made while a graph is capturing comes out of that graph's
    private pool and is stranded there, so the slot buffers and aranges exist
    before any capture runs.
    """
    for bucket in BUCKETS:
        if bucket not in _state.slot_buffers:
            _state.slot_buffers[bucket] = torch.zeros(bucket, dtype=torch.int32, device=device)
            _state.aranges[bucket] = torch.arange(bucket, dtype=torch.int32, device=device)


def slot_buffer(bucket: int) -> torch.Tensor:
    return _state.slot_buffers[bucket]


def workspace(bucket: int, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
    """FIA workspace for this bucket, computed outside any capture."""
    ws = _state.workspaces.get(bucket)
    if ws is None:
        ws = factory()
        _state.workspaces[bucket] = ws
    return ws


def lse_buffer(bucket: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    buf = _state.lse.get(bucket)
    if buf is None:
        buf = torch.empty(1, dtype=dtype, device=device)
        _state.lse[bucket] = buf
    return buf


def store_graph(bucket: int, graph: Any, hidden_out: torch.Tensor) -> None:
    """Register a bucket's captured graph and its baked output tensor.

    These are raw ``NPUGraph`` objects this module owns, the way the codec
    step graph owns its own -- deliberately NOT entries in the ACL graph
    wrapper. A wrapper-FULL prefill replay poisons the request that follows
    it: every codec-step replay afterwards blocks the host for the queued
    backbone work, and neither the capture order nor the pre-replay barrier
    explains it. The
    codec graph itself proves raw replays coexist cleanly with everything,
    so the prefill graphs live the same way.
    """
    _state.graphs[bucket] = (graph, hidden_out)
    _state.captured.add(bucket)


def mark_captured(bucket: int) -> None:
    _state.captured.add(bucket)


def bucket_for(n: int) -> int | None:
    """Smallest captured bucket that holds an ``n``-token prefill."""
    for bucket in BUCKETS:
        if n <= bucket and bucket in _state.captured:
            return bucket
    return None


def maybe_mark_step(attn_metadata: Any, prefill_state: Any, block_size_hint: int = 0) -> None:
    """Builder hook: route an eligible prefill step onto its captured graph.

    Cheap, and refuses everything unusual: called once per step from
    ``OmniFixedKVMetadataBuilder.build``. When it marks the step, it also
    fills the persistent slot buffer -- real slots for the real rows, the
    request's own next slots for the padding, one device add with no sync.
    """
    if not _state.captured:
        return
    if getattr(attn_metadata, "attn_state", None) != prefill_state:
        return
    if not getattr(attn_metadata, "causal", False):
        return
    lengths = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if not lengths or len(lengths) != 1:
        return
    n = int(lengths[0])
    seq_lens_list = getattr(attn_metadata, "seq_lens_list", None)
    if not seq_lens_list or int(seq_lens_list[0]) != n:
        # A chunked continuation (offset > 0) needs the earlier KV, which the
        # captured PrefillNoCache graph does not read.
        return
    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    if slot_mapping is None or slot_mapping.shape[0] < n:
        return
    bucket = bucket_for(n)
    if bucket is None:
        return
    block_size = int(block_size_hint or 0)
    if block_size <= 0:
        return
    if (n - 1) // block_size != (bucket - 1) // block_size:
        # The padding rows must land on the request's own allocated blocks.
        # Slots are contiguous only within a block, and a request's block
        # table covers ceil(n / block_size) blocks -- a bucket that needs one
        # more block than the prompt does would write into somebody else's.
        # (A prompt that spans two blocks hits exactly this: a slot0+arange fill
        # pushes rows past the first block's 127 slots into the physical block
        # after it -- runaway text instead of a captured prefill.)
        return
    block_table = getattr(attn_metadata, "block_tables", None)
    if bucket <= block_size:
        # One block: slots run contiguously from slot0.
        torch.add(_state.aranges[bucket], slot_mapping[0], out=_state.slot_buffers[bucket])
    else:
        if block_table is None or block_table.shape[0] < 1:
            return
        # Per-row: slot(i) = block_table[i // bs] * bs + i % bs. Three device
        # ops, no sync; every index stays inside blocks the prompt owns
        # because of the same-block-count gate above.
        arange = _state.aranges[bucket]
        blocks = block_table[0].index_select(0, (arange // block_size).to(torch.int64))
        torch.add(arange % block_size, blocks * block_size, out=_state.slot_buffers[bucket])
    attn_metadata.fixed_kv_prefill_bucket = bucket


def replay(attn_metadata: Any) -> torch.Tensor | None:
    """Replay the marked bucket's graph and hand back its baked hidden.

    The caller skips the model call entirely: the graph read the runner's
    persistent input buffers when it was captured, the builder refreshed the
    persistent slot buffer when it marked the step, and the output tensor is
    the same storage every time.
    """
    if isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    bucket = getattr(attn_metadata, "fixed_kv_prefill_bucket", None)
    if bucket is None:
        return None
    entry = _state.graphs.get(bucket)
    if entry is None:
        return None
    graph, hidden_out = entry
    graph.replay()
    global _LOGGED_REPLAY
    if not _LOGGED_REPLAY:
        _LOGGED_REPLAY = True
        logger.info("[minicpmo] prefill replays its own captured %d-token graph", bucket)
    return hidden_out
