# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded H3 windows with synchronized latent-tail guides and global AV RoPE positions.

The guide/discard/append algorithm follows ComfyUI-Minimax-H3-Continuation:
https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.errors import OmniClientError

from .packed_tokens import minimax_h3_pack_audio_latent, minimax_h3_patchify_video_latent

logger = init_logger(__name__)


def _video_t(frames: int) -> int:
    return (frames - 5) // 17 * 5 + 2


def _audio_t(frame_boundary: int) -> int:
    return round(frame_boundary * 5 / 3)


@dataclass(frozen=True)
class ContinuationWindow:
    start: int
    end: int
    overlap: int

    @property
    def audio_start(self) -> int:
        return _audio_t(self.start)

    @property
    def audio_end(self) -> int:
        return _audio_t(self.end)

    @property
    def overlap_audio_t(self) -> int:
        return _audio_t(self.start + self.overlap) - self.audio_start


def resolve_continuation(extra: Mapping[str, Any], *, task: str, step_execution: bool) -> tuple[int, int] | None:
    default_mode = "continuation" if extra.get("long_video") and task == "ref2va" else "full"
    mode = extra.get("long_video_mode", default_mode)
    if mode == "full":
        return None
    if mode != "continuation":
        raise OmniClientError("MiniMax H3 long_video_mode must be full or continuation")
    if task != "ref2va" or step_execution:
        raise OmniClientError("MiniMax H3 continuation requires Ref2VA request execution (not step execution)")
    window = extra.get("continuation_window_frames", 277)
    overlap = extra.get("continuation_overlap_frames", 22)
    for name, value in (("window", window), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 5 or value % 17 != 5:
            raise OmniClientError(f"MiniMax H3 continuation {name} must be an integer on the 17n+5 frame grid")
    if not 107 <= window <= 345 or overlap >= window:
        raise OmniClientError("MiniMax H3 continuation window must be 107..345 frames and exceed its overlap")
    return window, overlap


def plan_continuation_windows(total_frames: int, window_frames: int, overlap_frames: int) -> list[ContinuationWindow]:
    if total_frames < 5 or total_frames % 17 != 5:
        raise ValueError("total_frames must satisfy 17n+5")
    if not 5 <= overlap_frames < window_frames or window_frames % 17 != 5 or overlap_frames % 17 != 5:
        raise ValueError("window and overlap must satisfy 17n+5, with 5 <= overlap < window")
    end = min(total_frames, window_frames)
    windows = [ContinuationWindow(0, end, 0)]
    while end < total_frames:
        start = end - overlap_frames
        end = min(total_frames, start + window_frames)
        windows.append(ContinuationWindow(start, end, overlap_frames))
    return windows


def diffuse_continuation(
    diffuse: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    kwargs: dict[str, Any],
    *,
    window_frames: int,
    overlap_frames: int,
    text_conditioning: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise fresh windows; retain old latents and append only new suffixes.

    Guides are extra condition rows sharing the new target's temporal origin,
    not a masked target prefix. Audio boundaries refer to the cumulative frame
    timeline, preventing per-window rounding from accumulating A/V drift.
    Each window shifts temporal media positions onto that same global timeline
    before RoPE is evaluated; text and static image references remain fixed.
    """
    windows = plan_continuation_windows(kwargs["num_frames"], window_frames, overlap_frames)
    if text_conditioning is not None and len(text_conditioning) != len(windows):
        raise OmniClientError("MiniMax H3 requires one text conditioning per continuation window")
    # Different prompts have different prefix lengths. Keep the media clock
    # anchored after the longest prefix so text length cannot shift AV time.
    media_time_origin = max(item[0].shape[0] for item in text_conditioning) if text_conditioning else None
    source_rows = kwargs.get("locked_audio_rows")
    source = None if source_rows is None else source_rows.reshape(2, kwargs["audio_t"], 32)
    video = audio = None
    for index, window in enumerate(windows):
        frames = window.end - window.start
        args = {
            **kwargs,
            "num_frames": frames,
            "latent_t": _video_t(frames),
            "audio_t": window.audio_end - window.audio_start,
            # Keep fractional RoPE units; audio slice indices alone are rounded.
            "temporal_offset": window.start * (40.0 / 24.0),
        }
        if text_conditioning is not None:
            args["text_embeddings"], args["text_tags"] = text_conditioning[index]
            args["media_time_origin"] = media_time_origin
        if source is not None:
            args["locked_audio_rows"] = source[:, window.audio_start : window.audio_end].reshape(-1, 32)
        overlap_v = _video_t(window.overlap) if window.overlap else 0
        if video is not None and audio is not None:
            guide_video = minimax_h3_patchify_video_latent(video[:, :, -overlap_v:], patch_size=(1, 2, 2))
            guide_audio = minimax_h3_pack_audio_latent(audio[..., -window.overlap_audio_t :])
            for name, guide in (("visual_condition", guide_video), ("audio_condition", guide_audio)):
                existing = kwargs.get(name)
                args[name] = guide if existing is None else torch.cat((existing, guide), dim=0)
            args["visual_condition_shapes"] = [
                *(kwargs.get("visual_condition_shapes") or []),
                (overlap_v, kwargs["latent_h"], kwargs["latent_w"]),
            ]
            args["audio_condition_lengths"] = [*(kwargs.get("audio_condition_lengths") or []), window.overlap_audio_t]
            args["ref_blocks"] = [
                *(kwargs.get("ref_blocks") or []),
                {
                    "kind": "latent_guide",
                    "latent_t": overlap_v,
                    "latent_h": kwargs["latent_h"],
                    "latent_w": kwargs["latent_w"],
                    "ref_audio_t": window.overlap_audio_t,
                },
            ]
        logger.info(
            "MiniMax H3 continuation window %d/%d: frames [%d, %d), hidden overlap %d, temporal offset %.6f",
            index + 1,
            len(windows),
            window.start,
            window.end,
            window.overlap,
            args["temporal_offset"],
        )
        sampled_video, sampled_audio = diffuse(**args)
        video = sampled_video if video is None else torch.cat((video, sampled_video[:, :, overlap_v:]), dim=2)
        audio = (
            sampled_audio if audio is None else torch.cat((audio, sampled_audio[..., window.overlap_audio_t :]), dim=-1)
        )
    assert video is not None and audio is not None
    if video.shape[2] != kwargs["latent_t"] or audio.shape[-1] != kwargs["audio_t"]:
        raise RuntimeError("MiniMax H3 continuation produced inconsistent cumulative AV lengths")
    return video, audio
