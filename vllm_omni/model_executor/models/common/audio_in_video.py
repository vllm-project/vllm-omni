# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio-in-video interleave detection for the Qwen-Omni thinkers."""

import torch


def check_interleaved_audio_video(
    is_video: torch.Tensor,
    is_audio: torch.Tensor,
    num_video: int,
    num_audio: int,
) -> bool:
    """Whether any contiguous video/audio placeholder span interleaves the two.

    Same result as vLLM's ``qwen2_5_omni_thinker.check_interleaved_audio_video``
    (a span is a run of consecutive video or audio positions; it interleaves
    when its first video position precedes its last audio position and its
    first audio position precedes its last video position), computed with
    tensor reductions and two host reads in total.
    """
    if num_video == 0 or num_audio == 0:
        return False

    positions = (is_video | is_audio).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return False

    span_starts = torch.ones_like(positions, dtype=torch.bool)
    span_starts[1:] = positions[1:] != positions[:-1] + 1
    span_ids = span_starts.cumsum(0) - 1

    video = is_video[positions]
    audio = is_audio[positions]
    # One slot per position (at least one per span), so the span count is never read back. Unused slots
    # and spans without video (or audio) keep the past-the-end and -1 fills, which never compare as interleaved.
    past_end = torch.full_like(positions, is_video.numel())

    def span_bounds(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = past_end.scatter_reduce(0, span_ids, torch.where(mask, positions, past_end), "amin")
        last = torch.full_like(positions, -1).scatter_reduce(0, span_ids, torch.where(mask, positions, -1), "amax")
        return first, last

    video_first, video_last = span_bounds(video)
    audio_first, audio_last = span_bounds(audio)
    interleaved = (video_first < audio_last) & (audio_first < video_last)
    return bool(interleaved.any())
