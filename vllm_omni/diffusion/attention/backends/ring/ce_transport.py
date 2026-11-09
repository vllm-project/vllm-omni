# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Copy-engine K/V transport for ring attention (``VLLM_OMNI_RING_CE_TRANSPORT=1``).

``RingComm`` moves each ring step's K/V with NCCL send/recv kernels. Those kernels run on SMs
that the overlapped flash-attention kernel needs, so attention slows down for as long as the
transfer lasts. This transport writes the next block straight into the successor's GPU memory
with ``cudaMemcpyAsync`` over NVLink. The copy runs on the copy engine, a DMA unit, so the only
SM work left is a few one-warp signal kernels.

Big picture, a ring of W ranks. At every step each rank copies one K/V block into the next
rank's buffer, so every rank writes its successor's slots and reads its own::

     rank r-1 (pred)                 rank r (me)                  rank r+1 (succ)
    +---------------------+         +---------------------+       +---------------------+
    | slot 0: K, V        |         | slot 0: K, V        |       | slot 0: K, V        |
    | slot 1: K, V        |  =====> | slot 1: K, V        | ====> | slot 1: K, V        |
    | signal pad          |  copy   | signal pad          | copy  | signal pad          |
    +---------------------+         +---------------------+       +---------------------+

    copy = copy engine over NVLink, written straight into the next rank's slot

Each slot has one writer, the rank before its owner, and one reader, its owner. Two flags make
them take turns. For my slot j, written by pred and read by me:

  1. I set FREE[j]: "pred may write slot j". I do this at setup and each time I finish slot j.
  2. pred waits for FREE[j], then copies K and V into my slot j.
  3. pred sets READY[j]: "slot j is full".
  4. I wait for READY[j], then attend to slot j and forward it to succ. Back to 1.

At the same time I am the writer of succ's slots, and run steps 2 and 3 against succ.

Slots. Each rank owns one symmetric buffer of shape [2 slots][K, V][B, S_local, H, D], which is
4 x B*S*H*D elements: 387 MB per GPU at Wan2.2 720p in bf16. It stays allocated across calls
(the workspace only grows). Attention alternates between the two slots, and each block is
forwarded out of the slot it arrived in, so nothing is staged:

    step s | attention reads    | commit(s) copies that block into succ's
    -------+--------------------+----------------------------------------
    0      | own K/V (no slot)  | slot 1
    1      | slot 1             | slot 0
    2      | slot 0             | slot 1
    s      | slot s % 2         | slot (s + 1) % 2
    W-1    | slot (W - 1) % 2   | nothing: the last step has no commit

The two slots are the double buffering: while attention reads slot s % 2, the predecessor
fills slot (s + 1) % 2. With a single slot the next copy could only start after attention.

Streams. commit() queues the transfer on a side copy stream, and wait() makes the compute stream
wait for it. The CPU never blocks; it only queues work. For one rank:

    compute: [attn 0: own K/V]-wait-[attn 1: slot 1]-wait-[attn 2: slot 0]-wait- ...
    copy:    [commit 0: own K/V -> succ slot 1]
                             [commit 1: slot 1 -> succ slot 0]
                                                  [commit 2: slot 0 -> succ slot 1]

commit(s) first waits for everything already queued on compute: attention s-1 and wait(s-1).
So it forwards a slot only after that slot has arrived and this rank has read it, and its copy
overlaps attention s.

Signal pad. Every symmetric buffer comes with a small array of 32-bit flag words that is mapped
into every peer, like the data. Word [channel][src] of rank X's pad means "flag ``channel``,
sent to X by rank ``src``". The two calls used here launch one-warp kernels that spin, that is,
busy-wait on an atomic compare-and-swap:

- ``put_signal(dst, ch)`` waits until dst's word [ch][me] is 0, then sets it to 1. A pending
  message is never overwritten.
- ``wait_signal(src, ch)`` waits until my word [ch][src] is 1, then clears it, which consumes
  the message.

A spinning kernel holds back everything queued after it on its stream. That is how one GPU
waits for another without involving the CPU.

Flags. A flag is a word in the signal pad of the rank that waits for it, and the other rank sets
it remotely. READY[j] for my slot j is in my pad: pred sets it and I wait for it. FREE[j] for my
slot j is in pred's pad: I set it and pred waits for it. I set FREE[j] only after I have both
read slot j (attention) and forwarded it, because an earlier overwrite would corrupt my
attention or the block I forward.

Credits. A set FREE flag is a credit: permission to write that slot once. For one pair A -> B,
where A writes B's slots, in a ring of 4:

    setup         B grants A one credit per slot              A holds: slot 0, slot 1
    A step 0      A uses its slot-1 credit and fills slot 1    A holds: slot 0
    A step 1      A uses its slot-0 credit and fills slot 0    A holds: none
    A step 2      A needs slot 1 again and waits ...          A holds: none
    B step 2      ... until B returns the slot-1 credit       A holds: slot 1
    A step 2      A uses it and fills slot 1                  A holds: none
    B finish()    B returns the credits for steps 2 and 3     A holds: slot 0, slot 1

Two credits let the writer run at most two blocks ahead of the reader. A call returns every
credit it used, so the next call (the next layer) starts from the setup state. Setup therefore
runs once per buffer (see ``_CREDITED``). Running it again would hang, because ``put_signal``
waits for a flag that is still set.

Callers. ``Attention._run_ring_attention`` -> ``RingParallelAttention.run_attention`` ->
``ring_flash_attn_func`` -> ``RingFlashAttnFunc.forward`` -> ``ring_flash_attn_forward``. Its
ring loop calls ``send_recv(k)``, ``send_recv(v)``, ``commit()``, the step's attention and
``wait()`` at every step but the last, then ``finish()`` once after the loop.

The loop order, kernels and merge are those of ``RingComm``; only the path the bytes take
changes, so the output is bit-identical. Without symmetric memory or P2P (multi-node, PCIe
without P2P, ROCm, a non-CUDA symmetric-memory backend, tracing or graph capture) the caller
falls back to ``RingComm``. That includes runs with ``ulysses_a2a_permute`` on, which switches
the process-wide symmetric-memory backend to NCCL.
"""

from __future__ import annotations

import functools

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.a2a_permute import _SYMM_WORKSPACES, _get_symm_buffer

logger = init_logger(__name__)

# Signal-pad channels, one per kind of flag and per slot. Four are needed:
# - One per slot: a flag holds one pending message, and double buffering keeps two in flight in
#   each direction. Setup grants both FREE credits back to back. With a single FREE channel the
#   second put_signal would wait on a predecessor that is stuck the same way, and the whole ring
#   deadlocks.
# - READY and FREE apart: my pad holds the flags I wait for. READY words are set by pred, at
#   [channel][pred]; FREE words are set by succ, at [channel][succ]. With W = 2, pred and succ are
#   the same rank, so if READY and FREE shared channel numbers they would share a word.
_READY = (0, 1)
_FREE = (2, 3)

# Per (device, group): the workspace handle whose signal pad already holds the setup credits.
# Credits stay in the pad between calls, so they are granted once per buffer. Growing the
# workspace for a bigger shape creates a new buffer with a zeroed pad, which needs them again.
_CREDITED: dict[tuple[torch.device, str], object] = {}


@functools.cache
def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    # One stream per device for every copy and signal kernel. commit() relies on its FIFO order:
    # a slot's credit goes back only after that slot's forward copy, queued earlier here.
    return torch.cuda.Stream(device)


def _unavailable_reason(process_group: dist.ProcessGroup, k: torch.Tensor, v: torch.Tensor) -> str | None:
    if torch.version.hip is not None:
        return "ROCm"
    if not k.is_cuda:
        return "not a CUDA tensor"
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return "tracing or CUDA graph capture"
    # K and V share one buffer layout.
    if k.shape != v.shape or k.dtype != v.dtype:
        return "K and V differ in shape or dtype"
    if not dist.is_initialized():
        return "torch.distributed is not initialized"
    backend = symm_mem.get_backend(k.device)
    if backend != "CUDA":
        return f"symmetric-memory backend is {backend}, not CUDA"
    # A ring of one transfers nothing, so it should not hold a symmetric buffer.
    if dist.get_world_size(process_group) < 2:
        return "ring world size < 2"
    return None


def maybe_ce_ring_comm(process_group: dist.ProcessGroup, k: torch.Tensor, v: torch.Tensor) -> CeRingComm | None:
    """Return a copy-engine transport for this ring call, or None to use ``RingComm``.

    ``k`` and ``v`` must be contiguous. Buffer setup is collective, so a rank that picked this
    transport while another picked ``RingComm`` would hang. The checks therefore read only state
    that is identical across the ring group (backend, shape, dtype, mode).
    """
    comm = None
    try:
        reason = _unavailable_reason(process_group, k, v)
        if reason is None:
            comm = CeRingComm(process_group, k)
    except Exception as e:  # rendezvous or peer mapping failed, e.g. across nodes
        reason = f"{type(e).__name__}: {e}"
    if comm is None:
        logger.warning_once(
            "VLLM_OMNI_RING_CE_TRANSPORT=1 but the copy-engine ring transport is unavailable (%s); using RingComm",
            reason,
        )
    else:
        logger.info_once("Copy-engine ring transport active (ring world size %d)", comm.world_size)
    return comm


class CeRingComm:
    """Drop-in for ``RingComm`` in ``ring_flash_attn_forward``, plus ``finish()``.

    Per ring step the loop calls ``send_recv(k)``, ``send_recv(v)``, ``commit()``, then runs
    the step's attention, then ``wait()``. After the loop it calls ``finish()``. The module
    docstring has the protocol.
    """

    def __init__(self, process_group: dist.ProcessGroup, k: torch.Tensor):
        group = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self._pred = (self.rank - 1) % self.world_size
        self._succ = (self.rank + 1) % self.world_size
        shape = (2, 2, *k.shape)  # [slot][K, V][B, S, H, D] = 4 x B*S*H*D elements
        # Collective on first use and on growth; every rank reaches it at step 0.
        self._slots = _get_symm_buffer(shape, k.dtype, k.device, group.group_name)
        key = (k.device, group.group_name)
        self._handle = _SYMM_WORKSPACES[key].handle
        if _CREDITED.get(key) is not self._handle:
            # Setup: both of my slots are empty, so the predecessor may fill either.
            for channel in _FREE:
                self._handle.put_signal(self._pred, channel)
            _CREDITED[key] = self._handle
        # The successor's buffer, mapped into this process: copying into it writes the successor's
        # GPU memory over NVLink. Building the view is host-only work of a few microseconds.
        self._peer = self._handle.get_buffer(self._succ, shape, k.dtype, 0)
        self._copy = _copy_stream(k.device)
        self._compute = torch.cuda.current_stream(k.device)
        self._ready = torch.cuda.Event()
        self._pending: list[torch.Tensor] = []
        self._step = 0

    def send_recv(self, to_send: torch.Tensor) -> torch.Tensor:
        """Queue K (first call) or V (second) for the successor; return the slot that receives it here."""
        # No data moves yet. The loop makes the returned slot view its next k / v, so at the next
        # step commit() forwards straight out of that slot.
        self._pending.append(to_send)
        return self._slots[(self._step + 1) % 2, len(self._pending) - 1]

    def commit(self) -> None:
        s, slot = self._step, (self._step + 1) % 2
        h = self._handle
        with torch.cuda.stream(self._copy):
            # Everything queued on compute so far: attention s-1, which read slot (s-1) % 2, and
            # wait(s-1), which means the block being forwarded now has arrived. At step 0 that is
            # just the production of the rank's own K/V.
            self._copy.wait_stream(self._compute)
            # Hand back the credit for the slot read at step s-1. Its forward copy ran in
            # commit(s-1), earlier on this stream.
            self._release(s - 1)
            # Take a credit: wait until the successor has finished with its slot `slot`.
            h.wait_signal(self._succ, _FREE[slot])
            # Copy engine over NVLink, straight into the successor's slot.
            for index, src in enumerate(self._pending):
                self._peer[slot, index].copy_(src, non_blocking=True)
            # Stream order starts this kernel only after the copies above have finished.
            h.put_signal(self._succ, _READY[slot])
            # Wait until the predecessor has filled my own slot `slot`.
            h.wait_signal(self._pred, _READY[slot])
            self._ready.record(self._copy)
        self._pending = []

    def wait(self) -> None:
        # Called by the ring loop after the step's attention. The next attention reads my slot
        # (s + 1) % 2, so compute waits until the READY wait in commit() has passed. Reusing one
        # event is safe because this wait is queued before the next commit() records it again.
        self._compute.wait_event(self._ready)
        self._step += 1

    def finish(self) -> None:
        """Return the credits for the last two steps and rejoin the compute stream."""
        with torch.cuda.stream(self._copy):
            # Attention for the last two steps must finish before their slots are handed back.
            self._copy.wait_stream(self._compute)
            self._release(self.world_size - 2)
            self._release(self.world_size - 1)
        # Every wait() already joined the copies, so memory safety does not depend on this. It also
        # orders the two credit returns before later compute work.
        self._compute.wait_stream(self._copy)

    def _release(self, step: int) -> None:
        """Return the FREE credit for the slot this rank attended to at ``step`` (copy stream)."""
        # Step 0 attends to the rank's own K/V, which is not in a slot.
        if step >= 1:
            self._handle.put_signal(self._pred, _FREE[step % 2])
