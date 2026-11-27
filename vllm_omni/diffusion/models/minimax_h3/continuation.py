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
from vllm_omni.model_executor.models.minimax_h3.long_video import resolve_continuation as _resolve_continuation

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


@dataclass(frozen=True)
class LatentPrefix:
    """Clean latents of an earlier clip that the request's first frames repeat.

    ``start_frame`` places the request on that clip's global frame timeline;
    ``video``/``audio`` are its last ``overlap_frames`` frames of latents. They
    ride as a frame-0 AV timeline guide for every task: it anchors the window to
    the clip far more tightly than a Ref2VA ``latent_guide`` reference.
    """

    video: torch.Tensor  # [1, C, _video_t(overlap_frames), H, W]
    audio: torch.Tensor  # [2, C, overlap audio latents]
    start_frame: int
    overlap_frames: int


def resolve_continuation(extra: Mapping[str, Any], *, task: str, step_execution: bool) -> tuple[int, int] | None:
    return _resolve_continuation(extra, task=task, step_execution=step_execution)


def plan_continuation_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    *,
    first_overlap: int = 0,
    origin: int = 0,
) -> list[ContinuationWindow]:
    """Windows over ``total_frames``, offset onto a global timeline by ``origin``.

    ``first_overlap`` hides the first frames of the first window behind a
    :class:`LatentPrefix`; later windows overlap their predecessor.
    """
    if total_frames < 5 or total_frames % 17 != 5:
        raise ValueError("total_frames must satisfy 17n+5")
    if not 5 <= overlap_frames < window_frames or window_frames % 17 != 5 or overlap_frames % 17 != 5:
        raise ValueError("window and overlap must satisfy 17n+5, with 5 <= overlap < window")
    end = min(total_frames, window_frames)
    if first_overlap and (first_overlap % 17 != 5 or first_overlap >= end):
        raise ValueError("first_overlap must satisfy 17n+5 and leave new frames in the first window")
    windows = [ContinuationWindow(origin, origin + end, first_overlap)]
    while end < total_frames:
        start = end - overlap_frames
        end = min(total_frames, start + window_frames)
        windows.append(ContinuationWindow(origin + start, origin + end, overlap_frames))
    return windows


def latent_guide_args(
    kwargs: Mapping[str, Any],
    video_tail: torch.Tensor,
    audio_tail: torch.Tensor,
    *,
    timeline_guide: bool = False,
) -> dict[str, Any]:
    """Add clean video/audio tail latents covering the window's first overlap frames.

    ``video_tail`` is ``[1, C, overlap_v, H, W]`` on the target canvas and
    ``audio_tail`` is ``[2, C, overlap_audio_t]``. Ref2VA appends them as the
    final ``latent_guide`` reference block; other tasks prepend a frame-0 AV
    timeline guide, which shares that origin and packs ahead of references.
    """
    overlap_v, overlap_audio_t = int(video_tail.shape[2]), int(audio_tail.shape[-1])
    if tuple(video_tail.shape[-2:]) != (kwargs["latent_h"], kwargs["latent_w"]):
        raise OmniClientError("MiniMax H3 latent guide must match the target canvas")
    guide_video = minimax_h3_patchify_video_latent(video_tail, patch_size=(1, 2, 2))
    guide_audio = minimax_h3_pack_audio_latent(audio_tail)
    shape = (overlap_v, kwargs["latent_h"], kwargs["latent_w"])
    if timeline_guide:
        block = {"kind": "video_audio", "frame_index": 0, "ref_audio_t": overlap_audio_t}
        return {
            "guide_blocks": [
                {**block, "latent_t": overlap_v, "latent_h": shape[1], "latent_w": shape[2]},
                *(kwargs.get("guide_blocks") or []),
            ],
            "visual_condition": _join(guide_video, kwargs.get("visual_condition"), first=True),
            "audio_condition": _join(guide_audio, kwargs.get("audio_condition"), first=True),
            "visual_condition_shapes": [shape, *(kwargs.get("visual_condition_shapes") or [])],
            "audio_condition_lengths": [overlap_audio_t, *(kwargs.get("audio_condition_lengths") or [])],
        }
    block = {"kind": "latent_guide", "ref_audio_t": overlap_audio_t}
    return {
        "ref_blocks": [
            *(kwargs.get("ref_blocks") or []),
            {**block, "latent_t": overlap_v, "latent_h": shape[1], "latent_w": shape[2]},
        ],
        "visual_condition": _join(guide_video, kwargs.get("visual_condition"), first=False),
        "audio_condition": _join(guide_audio, kwargs.get("audio_condition"), first=False),
        "visual_condition_shapes": [*(kwargs.get("visual_condition_shapes") or []), shape],
        "audio_condition_lengths": [*(kwargs.get("audio_condition_lengths") or []), overlap_audio_t],
    }


def _join(guide: torch.Tensor, existing: torch.Tensor | None, *, first: bool) -> torch.Tensor:
    if existing is None:
        return guide
    return torch.cat((guide, existing) if first else (existing, guide), dim=0)


def window_args(
    kwargs: Mapping[str, Any],
    window: ContinuationWindow,
    *,
    video_tail: torch.Tensor | None = None,
    audio_tail: torch.Tensor | None = None,
    timeline_guide: bool = False,
    audio_origin: int = 0,
) -> dict[str, Any]:
    """Denoise arguments for one window: global AV clock, window lengths and any tail guide."""
    frames = window.end - window.start
    args = {
        **kwargs,
        "num_frames": frames,
        "latent_t": _video_t(frames),
        "audio_t": window.audio_end - window.audio_start,
        # Keep fractional RoPE units; audio slice indices alone are rounded.
        "temporal_offset": window.start * (40.0 / 24.0),
    }
    source = kwargs.get("locked_audio_rows")
    if source is not None:
        source = source.reshape(2, kwargs["audio_t"], 32)
        args["locked_audio_rows"] = source[
            :, window.audio_start - audio_origin : window.audio_end - audio_origin
        ].reshape(-1, 32)
    if video_tail is not None and audio_tail is not None:
        args.update(latent_guide_args(kwargs, video_tail, audio_tail, timeline_guide=timeline_guide))
    return args


def diffuse_continuation(
    diffuse: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    kwargs: dict[str, Any],
    *,
    window_frames: int,
    overlap_frames: int,
    text_conditioning: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    prefix: LatentPrefix | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise fresh windows; retain old latents and append only new suffixes.

    Guides are extra condition rows sharing the new target's temporal origin,
    not a masked target prefix. Audio boundaries refer to the cumulative frame
    timeline, preventing per-window rounding from accumulating A/V drift.
    Each window shifts temporal media positions onto that same global timeline
    before RoPE is evaluated; text and static image references remain fixed.

    A ``prefix`` seeds the cumulative latents with an earlier clip's tail: the
    first window repeats it behind a hidden overlap, and the result starts with
    that tail, so callers trim ``prefix.overlap_frames`` after decoding.
    """
    origin = prefix.start_frame if prefix is not None else 0
    windows = plan_continuation_windows(
        kwargs["num_frames"],
        window_frames,
        overlap_frames,
        first_overlap=prefix.overlap_frames if prefix is not None else 0,
        origin=origin,
    )
    if text_conditioning is not None and len(text_conditioning) != len(windows):
        raise OmniClientError("MiniMax H3 requires one text conditioning per continuation window")
    # Different prompts have different prefix lengths. Keep the media clock
    # anchored after the longest prefix so text length cannot shift AV time.
    media_time_origin = max(item[0].shape[0] for item in text_conditioning) if text_conditioning else None
    video, audio = (prefix.video, prefix.audio) if prefix is not None else (None, None)
    for index, window in enumerate(windows):
        overlap_v = _video_t(window.overlap) if window.overlap else 0
        guided = video is not None and audio is not None
        args = window_args(
            kwargs,
            window,
            video_tail=video[:, :, -overlap_v:] if guided else None,
            audio_tail=audio[..., -window.overlap_audio_t :] if guided else None,
            timeline_guide=prefix is not None and index == 0,
            audio_origin=_audio_t(origin),
        )
        if text_conditioning is not None:
            args["text_embeddings"], args["text_tags"] = text_conditioning[index]
            args["media_time_origin"] = media_time_origin
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
        if video is None or audio is None:
            video, audio = sampled_video, sampled_audio
        else:
            video = torch.cat((video, sampled_video[:, :, overlap_v:].to(video.device)), dim=2)
            audio = torch.cat((audio, sampled_audio[..., window.overlap_audio_t :].to(audio.device)), dim=-1)
    assert video is not None and audio is not None
    if video.shape[2] != kwargs["latent_t"] or audio.shape[-1] != kwargs["audio_t"]:
        raise RuntimeError("MiniMax H3 continuation produced inconsistent cumulative AV lengths")
    return video, audio
