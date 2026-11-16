# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend one-query attention views for Talker multi-frame replay."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.interfaces import supports_multi_frame_decode

logger = init_logger(__name__)

_LOGGED_NARROW = False
_LOGGED_NARROW_BLOCKS: set[str] = set()

# SoC families whose one-query capture is known to fault. The 910_93 part
# trips a vector-core error (acl 507035) inside the packaged codec operator
# while that graph is being captured, so the narrow path stays off there
# unless an operator opts in explicitly -- see narrow_replay_enabled.
# "ascend910c" guards the device-name fallback: on some A3 containers the
# driver reports the part as "Ascend910C" rather than its 910_93 SoC code,
# and an unblocked unknown would let the capture brick the deployment.
# "ascend910b" is fail-safe rather than observed: no K-step attempt there
# ever reached a capture (all 8 combos faulted earlier, inside the reject
# kernel), so the one-query capture is unverified on that family.
_NARROW_BLOCKED_SOC_PREFIXES = ("ascend910_93", "ascend910c", "ascend910b")
_NARROW_SOC_LOGGED: str | None = None


def _narrow_soc_allows() -> bool:
    """Whether this part may capture the one-query graph by default.

    Judged from the part: the CANN-exported SoC code first, the device name as
    the fallback for a worker launched without them.
    """
    global _NARROW_SOC_LOGGED

    name = ""
    for variable in ("SOC_VERSION", "ASCEND_SOC_VERSION"):
        value = os.environ.get(variable, "").strip().lower()
        if value:
            name = value
            break
    if not name:
        try:
            import torch_npu

            name = str(torch_npu.npu.get_device_name(torch_npu.npu.current_device())).lower()
        except Exception:
            name = ""
    allows = True if not name else not name.startswith(_NARROW_BLOCKED_SOC_PREFIXES)
    if _NARROW_SOC_LOGGED != name:
        _NARROW_SOC_LOGGED = name
        logger.info(
            "[minicpmo] narrow replay SoC gate: name='%s' -> %s",
            name or "<unidentified>",
            "allowed" if allows else "blocked",
        )
    return allows


@dataclass
class NarrowStep:
    """What one codec frame has to change before the one-query graph replays.

    A K-query graph replay costs 1.33-1.47 ms of device time against the
    one-query graph's 0.83 ms, and it is a step change rather than a per-row
    cost -- a two-query graph costs the same as a four-query one. The loop only
    ever reads one row of the wide graph, so it replays the narrow one instead
    and moves the five things a frame changes:

    * ``inputs_embeds[r]`` -- request r's embedding for this frame
    * ``positions[r]``     -- its position, +frame
    * ``slot_mapping[r]``  -- the KV slot vLLM reserved for this frame
    * ``seq_lens[r]``      -- its length, +frame+1, in the buffer the graph reads
    * the block table      -- unchanged for the whole step

    The step's own K-wide layouts are cloned at ``begin`` because the narrow
    views alias them: ``slot_mapping[:rows]`` is the first row of
    ``slot_mapping.view(rows, K)``.
    """

    rows: int
    frames: int
    inputs_embeds: Any
    positions: Any
    slot_mapping: Any
    seq_lens: Any
    step_slots: Any
    step_positions: Any
    step_seq_lens: Any
    spans: list[tuple[int, int]]
    saved_descriptor: Any = None
    saved_runtime_mode: Any = None

    def select(self, frame: int) -> None:
        """Put frame ``frame``'s inputs where the captured replay reads them.

        Three copies and nothing else: everything a frame needs was computed
        once, at the start of the step, because these run between two graph
        replays and an allocation there costs more than the values do.
        """
        rows = self.rows
        self.slot_mapping[:rows].copy_(self.step_slots[:, frame])
        self.positions[:rows].copy_(self.step_positions[:, frame])
        self.seq_lens[:rows].copy_(self.step_seq_lens[:, frame])


def narrow_replay_enabled(runner: Any) -> bool:
    """Whether this worker should capture and replay the one-query graph.

    Off on a part whose one-query capture is known to fault
    (``_NARROW_BLOCKED_SOC_PREFIXES``), so a capture can never brick a
    deployment; those parts keep the K-query captures and the wide replay.
    """
    if not _narrow_soc_allows():
        return False
    model = getattr(runner, "model", None)
    if not supports_multi_frame_decode(model):
        return False
    return int(getattr(runner, "num_spec_tokens", 0) or 0) > 0


def begin_narrow_step(
    *,
    runner: Any,
    forward_context: Any,
    frames: int,
    positions: Any,
    inputs_embeds: Any,
    spans: list[tuple[int, int]],
) -> NarrowStep | None:
    """Point the forward context at the one-query graph, or decline.

    Declines -- and the caller falls back to the wide replay -- whenever the
    narrow graph for this row count is not captured, which is what makes this
    safe to land before every path that reaches it has been exercised.
    """
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    # The KV geometry required by narrow replay lives in
    # platforms/npu/attention/static_shape_decode, which may be absent. Import
    # defensively: when missing, decline as documented and let the caller fall
    # back to wide replay; multi-frame decode and the stop chain are unaffected.
    try:
        from vllm_omni.platforms.npu.attention import static_shape_decode
    except Exception:
        static_shape_decode = None  # type: ignore[assignment]

    if static_shape_decode is None:
        return _decline("the static-shape decode backend is not present in this build")
    if not narrow_replay_enabled(runner) or inputs_embeds is None or positions is None:
        return None
    rows = len(spans)
    if rows <= 0:
        return None
    seq_lens = static_shape_decode.captured_seq_lens(rows)
    if seq_lens is None:
        return _decline(f"no one-query graph captured for {rows} rows")
    metadata = getattr(forward_context, "attn_metadata", None)
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values()), None)
    step_slot_mapping = getattr(metadata, "slot_mapping", None)
    live_seq_lens = getattr(metadata, "seq_lens_device", None)
    graph_slot_mapping = static_shape_decode.captured_slot_mapping(rows)
    if step_slot_mapping is None or live_seq_lens is None or graph_slot_mapping is None:
        return _decline("this step's attention metadata carries no slot mapping")
    if step_slot_mapping.shape[0] < rows * frames:
        return _decline("slot mapping is shorter than the step's own frames")

    narrow = NarrowStep(
        rows=rows,
        frames=frames,
        inputs_embeds=inputs_embeds,
        positions=positions,
        slot_mapping=graph_slot_mapping,
        seq_lens=seq_lens,
        step_slots=step_slot_mapping[: rows * frames].view(rows, frames).clone(),
        step_positions=positions[: rows * frames].view(rows, frames).clone(),
        # `seq_lens` already counts all K of this step's frames, and by frame k
        # only k + 1 have been written -- so row r, column k is
        # `live - frames + k + 1`.
        step_seq_lens=(
            live_seq_lens[:rows, None]
            - frames
            + torch.arange(1, frames + 1, device=live_seq_lens.device, dtype=live_seq_lens.dtype)
        ),
        spans=[(index, index + 1) for index in range(rows)],
    )
    descriptor = getattr(forward_context, "batch_descriptor", None)
    narrow.saved_descriptor = descriptor
    if descriptor is not None:
        # `uniform` too: a failed dispatch clears it on the descriptor it hands
        # back, and the captured key has it set. What this step is has not
        # changed -- it is one token per request against a graph captured for
        # exactly that -- only which graph replays it.
        forward_context.batch_descriptor = dataclasses.replace(descriptor, num_tokens=rows, num_reqs=rows, uniform=True)
    else:
        forward_context.batch_descriptor = BatchDescriptor(num_tokens=rows, num_reqs=rows, uniform=True)
    # The dispatcher builds its lookup descriptor from its own query length --
    # 4 at runtime, which is right for the step -- so it finds no key for the
    # narrow capture and hands back NONE, and the wrapper then runs the model
    # uncaptured without saying so. This step *is* a full-graph uniform decode;
    # only which graph it replays changed, which is what this function decides.
    narrow.saved_runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    global _LOGGED_NARROW
    if not _LOGGED_NARROW:
        _LOGGED_NARROW = True
        logger.info(
            "[minicpmo] multi-frame Talker decode replays the one-query graph: "
            "%d rows, descriptor %s, dispatched mode was %s",
            rows,
            forward_context.batch_descriptor,
            narrow.saved_runtime_mode,
        )
    return narrow


def end_narrow_step(forward_context: Any, narrow: NarrowStep | None) -> None:
    if narrow is None:
        return
    forward_context.batch_descriptor = narrow.saved_descriptor
    if narrow.saved_runtime_mode is not None:
        forward_context.cudagraph_runtime_mode = narrow.saved_runtime_mode


def _decline(reason: str) -> NarrowStep | None:
    """Say each distinct decline reason once; silence is the worse failure.

    A set, not a single slot: the narrow path is re-decided every step, and one
    of its reasons carries the step's own row count, so comparing against only
    the last reason would log on every step whose row count moved.
    """
    if reason in _LOGGED_NARROW_BLOCKS:
        return None
    _LOGGED_NARROW_BLOCKS.add(reason)
    logger.info("[minicpmo] one-query Talker replay not engaged: %s", reason)
    return None
