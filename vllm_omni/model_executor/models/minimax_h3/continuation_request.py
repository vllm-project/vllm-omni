# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request contract for continuing an earlier H3 video across requests.

A request naming ``extra_args["continue_from"]`` is generated as a window whose
first ``continue_overlap_frames`` frames repeat the end of that video. The
overlap is hidden: every frame-addressed input of the request -- timeline
guides, FL2VA keyframes, latent-edit masks and media, driving audio -- refers
to the returned clip, whose frame 0 is the first new frame. These helpers map
those output coordinates onto the window the model samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm_omni.errors import OmniClientError

CONTINUE_FROM = "continue_from"
CONTINUE_OVERLAP_FRAMES = "continue_overlap_frames"
DEFAULT_CONTINUE_OVERLAP_FRAMES = 22
# The hidden overlap sits on the 17n+5 grid, so the new frames after it come in
# whole 17-frame groups of five latents each.
_NEW_FRAME_GROUP = 17


@dataclass(frozen=True)
class ContinuationRequest:
    source: str
    overlap_frames: int


def resolve_continuation_request(extra: Mapping[str, Any]) -> ContinuationRequest | None:
    source = extra.get(CONTINUE_FROM)
    if source is None:
        return None
    if not isinstance(source, str) or not source:
        raise OmniClientError("MiniMax H3 continue_from must be a non-empty video id")
    overlap = extra.get(CONTINUE_OVERLAP_FRAMES, DEFAULT_CONTINUE_OVERLAP_FRAMES)
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 5 or overlap % 17 != 5:
        raise OmniClientError("MiniMax H3 continue_overlap_frames must be an integer on the 17n+5 frame grid")
    return ContinuationRequest(source, overlap)


def continuation_output_frames(requested_frames: int) -> int:
    """The returned clip length nearest to the request: whole groups after the hidden overlap."""
    return _NEW_FRAME_GROUP * max(1, round(requested_frames / _NEW_FRAME_GROUP))


def window_frame_index(index: int, overlap_frames: int) -> int:
    """Map an output frame index to the window; negative indices already count from the shared end."""
    return index + overlap_frames if index >= 0 else index
