# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Talker codec-EOS plan for the CUDA multi-frame decode.

The single-frame EOS force/mask routing (``_make_omni_output_single_frame``)
depends only on the request's frame counter, which advances by one per
forwarded frame, so frames 1..K-1 can be planned on the host up front.
``commit_codec_frames`` then leaves the host state where that many
single-frame steps would have.
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
    """Per-frame codec-EOS routing (``[frame][request]``; row 0 unused) and the
    repetition window after frame 0's bookkeeping."""

    request_ids: list[str]
    eos_token_id: int
    force_eos: list[list[bool]]
    mask_eos: list[list[bool]]
    recent_codes: list[list[int]]


def plan_codec_frames(talker: Any, infos: list[Any], frames: int) -> CodecFramePlan:
    """Plan frames 1..frames-1 from the state frame 0's bookkeeping left."""
    num_reqs = len(infos)
    force = [[False] * num_reqs for _ in range(frames)]
    mask = [[False] * num_reqs for _ in range(frames)]
    request_ids: list[str] = []
    recent_codes: list[list[int]] = []
    states = talker._request_audio_states
    for index, info in enumerate(infos):
        request_id = str(info.get("request_id", index)) if isinstance(info, dict) else str(index)
        request_ids.append(request_id)
        state = states.get(request_id)
        state = state if isinstance(state, dict) else {}
        finished = bool(state.get("finished"))
        step0 = int(state.get("step", 0))
        max_tokens = state.get("max_tokens")
        min_tokens = state.get("min_tokens")
        drain = bool(state.get("turn_end_drain"))
        for k in range(1, frames):
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
    """Apply each request's forwarded ids (sampled on frames 0..n-2) as that many
    single-frame steps; return their terminal flags."""
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
