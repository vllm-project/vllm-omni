# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Talker codec-EOS plan for the CUDA multi-frame decode.

Before each frame the single-frame Talker (``_make_omni_output_single_frame``)
decides which rows must sample the codec EOS (a finished segment, the chunk or
context budget reached) and which must not (``min_tokens``, the duplex
turn-end drain window), and scores its 16-frame repetition window. All of that
is a function of the request's frame counter and codec history. Within one
K-frame step the counter advances by exactly one per forwarded frame -- a frame
that samples EOS ends the request, so no later frame of that request is
accepted -- which makes the EOS routing of frames 1..K-1 plannable up front on
the host, while the history grows by ids that stay on the device.

``commit_codec_frames`` then applies the frames the scheduler accepts to the
host state, leaving it exactly where that many single-frame steps would have.
Frame 0 of every step is the unmodified single-frame path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _CODEC_PENALTY_WINDOW,
    _turn_end_boundary_eos_masked,
)

CODEC_PENALTY_WINDOW = _CODEC_PENALTY_WINDOW


@dataclass
class CodecFramePlan:
    """Per-frame codec-EOS routing for one K-frame step (host lists).

    ``force_eos[k][i]`` / ``mask_eos[k][i]`` are what the single-frame path
    would decide for request ``i`` at frame ``k``; row 0 is unused (frame 0 ran
    the single-frame path itself). ``recent_codes[i]`` is the repetition
    window after frame 0's bookkeeping.
    """

    request_ids: list[str]
    eos_token_id: int
    force_eos: list[list[bool]]
    mask_eos: list[list[bool]]
    recent_codes: list[list[int]]


def _request_id(info: Any, index: int) -> str:
    return str(info.get("request_id", index)) if isinstance(info, dict) else str(index)


def plan_codec_frames(talker: Any, infos: list[Any], frames: int) -> CodecFramePlan:
    """Plan frames 1..frames-1 from the state frame 0's bookkeeping left."""
    if int(getattr(talker, "_k_step_frames", 0) or 0) > 0:
        raise RuntimeError("the CUDA multi-frame decode needs the single-frame codec head (NPU K-step is armed)")
    num_reqs = len(infos)
    force = [[False] * num_reqs for _ in range(frames)]
    mask = [[False] * num_reqs for _ in range(frames)]
    request_ids: list[str] = []
    recent_codes: list[list[int]] = []
    states = talker._request_audio_states
    for index, info in enumerate(infos):
        request_id = _request_id(info, index)
        request_ids.append(request_id)
        state = states.get(request_id)
        state = state if isinstance(state, dict) else {}
        finished = bool(state.get("finished"))
        step0 = int(state.get("step", 0))
        max_tokens = state.get("max_tokens")
        min_tokens = state.get("min_tokens")
        drain = bool(state.get("turn_end_drain"))
        for k in range(1, frames):
            # Frame k forwards the id sampled on frame k-1, so its make_omni_output
            # has advanced the counter k times past frame 0's.
            step = step0 + k
            chunk_done = finished or (max_tokens is not None and step >= int(max_tokens) - 1)
            force[k][index] = chunk_done
            mask[k][index] = not chunk_done and (
                (min_tokens is not None and step < int(min_tokens)) or (drain and _turn_end_boundary_eos_masked(step))
            )
        codes = state.get("recent_codes")
        recent_codes.append([int(c) for c in codes][-CODEC_PENALTY_WINDOW:] if isinstance(codes, list) else [])
    return CodecFramePlan(
        request_ids=request_ids,
        eos_token_id=int(talker._codec_eos_id),
        force_eos=force,
        mask_eos=mask,
        recent_codes=recent_codes,
    )


def commit_codec_frames(talker: Any, plan: CodecFramePlan, forwarded: list[list[int]]) -> list[bool]:
    """Apply frames 1..n-1 of each request; return their terminal flags.

    ``forwarded[i]`` holds the ids frames 1..n_i-1 forwarded (the ids sampled
    on frames 0..n_i-2). Each one is one single-frame ``make_omni_output``
    call: the counter advances, the id joins the repetition window, and the
    frame whose budget forced EOS marks the chunk finished.
    """
    flags: list[bool] = []
    for index, (request_id, codes) in enumerate(zip(plan.request_ids, forwarded)):
        state = talker._request_audio_states.get(request_id)
        if not codes or not isinstance(state, dict):
            flags.append(False)
            continue
        state["step"] = int(state.get("step", 0)) + len(codes)
        recent = state.get("recent_codes")
        state["recent_codes"] = ((recent if isinstance(recent, list) else []) + [int(c) for c in codes])[
            -CODEC_PENALTY_WINDOW:
        ]
        done = plan.force_eos[len(codes)][index]
        if done:
            state["finished"] = True
        flags.append(done)
    return flags
