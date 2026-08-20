"""Rule-based vocal segmentation from F0 contour (no neural net)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportsComponentDiscovery

logger = init_logger(__name__)


@dataclass(frozen=True)
class VocalSegmentConfig:
    hop_ms: int = 20
    smooth_ms: int = 200
    start_ms: int = 120
    end_ms: int = 200
    prepad_ms: int = 80
    postpad_ms: int = 120
    min_len_ms: int = 1000
    max_len_ms: int = 20000
    short_seg_merge_gap_ms: int = 8000
    small_gap_ms: int = 500
    lookback_ms: int = 200
    lookahead_ms: int = 100


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def _merge_short_segments(
    segments_ms: list[list[int]],
    *,
    min_len_ms: int,
    max_len_ms: int,
    short_seg_merge_gap_ms: int,
    small_gap_ms: int,
) -> list[list[int]]:
    if not segments_ms:
        return []

    merged: list[list[int]] = []
    cur_start, cur_end = segments_ms[0]
    for next_start, next_end in segments_ms[1:]:
        cur_len = cur_end - cur_start
        gap_ms = next_start - cur_end
        merged_len = next_end - cur_start
        should_merge = (cur_len < min_len_ms and gap_ms < short_seg_merge_gap_ms) or (
            gap_ms < small_gap_ms and merged_len < max_len_ms
        )
        if should_merge:
            cur_end = next_end
            continue
        if (cur_end - cur_start) >= min_len_ms:
            merged.append([cur_start, cur_end])
        cur_start, cur_end = next_start, next_end

    if (cur_end - cur_start) >= min_len_ms:
        merged.append([cur_start, cur_end])

    if not merged:
        return segments_ms

    return merged


def _voiced_to_segments(
    voiced: np.ndarray,
    *,
    hop_ms: int,
    smooth_ms: int,
    start_ms: int,
    end_ms: int,
    prepad_ms: int,
    postpad_ms: int,
    max_len_ms: int,
) -> list[list[int]]:
    smooth_frames = max(1, int(round(smooth_ms / hop_ms)))
    active = _moving_average(voiced.astype(np.float32), smooth_frames) >= 0.5

    segments: list[list[int]] = []
    start_idx = None
    start_frames = max(1, int(round(start_ms / hop_ms)))
    end_frames = max(1, int(round(end_ms / hop_ms)))
    prepad_frames = max(0, int(round(prepad_ms / hop_ms)))
    postpad_frames = max(0, int(round(postpad_ms / hop_ms)))
    active_count = 0
    inactive_count = 0

    for i, flag in enumerate(active):
        if flag:
            active_count += 1
            inactive_count = 0
        else:
            inactive_count += 1
            active_count = 0

        if start_idx is None:
            if active_count >= start_frames:
                start_idx = max(0, i - start_frames + 1 - prepad_frames)
        else:
            if inactive_count >= end_frames:
                end_idx = min(len(active) - 1, i - end_frames + 1 + postpad_frames)
                start_ms_val = start_idx * hop_ms
                end_ms_val = end_idx * hop_ms + hop_ms
                if end_ms_val > start_ms_val:
                    segments.append([int(start_ms_val), int(end_ms_val)])
                start_idx = None

    if start_idx is not None:
        start_ms_val = start_idx * hop_ms
        end_idx = min(len(active) - 1, len(active) - 1 + postpad_frames)
        end_ms_val = end_idx * hop_ms + hop_ms
        if end_ms_val > start_ms_val:
            segments.append([int(start_ms_val), int(end_ms_val)])

    def _split_segment(seg: list[int]) -> list[list[int]]:
        start_ms_val, end_ms_val = seg
        start_frame = int(start_ms_val // hop_ms)
        end_frame = int((end_ms_val - 1) // hop_ms)
        end_frame = max(start_frame, min(end_frame, len(active) - 1))

        best_start = None
        best_len = 0
        cur_start = None
        cur_len = 0
        for idx in range(start_frame, end_frame + 1):
            if not active[idx]:
                if cur_start is None:
                    cur_start = idx
                    cur_len = 1
                else:
                    cur_len += 1
            else:
                if cur_start is not None and cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_start = None
                cur_len = 0
        if cur_start is not None and cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        if best_start is None:
            split_frame = (start_frame + end_frame) // 2
        else:
            split_frame = best_start + best_len // 2

        split_ms = split_frame * hop_ms
        if split_ms <= start_ms_val:
            split_ms = start_ms_val + hop_ms
        if split_ms >= end_ms_val:
            split_ms = end_ms_val - hop_ms
        if split_ms <= start_ms_val or split_ms >= end_ms_val:
            return [seg]
        return [[start_ms_val, int(split_ms)], [int(split_ms), end_ms_val]]

    queue = segments[:]
    segments = []
    while queue:
        seg = queue.pop(0)
        if (seg[1] - seg[0]) <= max_len_ms:
            segments.append(seg)
            continue
        parts = _split_segment(seg)
        if len(parts) == 1:
            segments.append(seg)
        else:
            queue = parts + queue
    return segments


class VocalSegmenter(nn.Module, SupportAudioInput, SupportsComponentDiscovery):
    support_audio_input: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = ["."]
    _layerwise_offload_blocks_attrs: ClassVar[list[str]] = []

    def __init__(self, config: VocalSegmentConfig | None = None, *, verbose: bool = False):
        super().__init__()
        self.config = config or VocalSegmentConfig()
        self.verbose = verbose

    def forward(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0: np.ndarray,
        *,
        base_name: str = "vocal",
        origin_wav_fn: str = "",
        verbose: bool | None = None,
    ) -> list[dict]:
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            logger.info("[vocal detection] segmenting %s", base_name)
            t0 = time.time()

        cfg = self.config
        y = np.asarray(audio, dtype=np.float32).squeeze()
        sr = int(sample_rate)

        segments_ms = _voiced_to_segments(
            f0 > 0,
            hop_ms=cfg.hop_ms,
            smooth_ms=cfg.smooth_ms,
            start_ms=cfg.start_ms,
            end_ms=cfg.end_ms,
            prepad_ms=cfg.prepad_ms,
            postpad_ms=cfg.postpad_ms,
            max_len_ms=cfg.max_len_ms,
        )
        segments_ms = _merge_short_segments(
            segments_ms,
            min_len_ms=cfg.min_len_ms,
            max_len_ms=cfg.max_len_ms,
            short_seg_merge_gap_ms=cfg.short_seg_merge_gap_ms,
            small_gap_ms=cfg.small_gap_ms,
        )

        adjusted_segments: list[list[int]] = []
        prev_end = 0
        for start_ms, end_ms in segments_ms:
            start_ms = max(0, start_ms - cfg.lookback_ms)
            end_ms = min(end_ms + cfg.lookahead_ms, int(y.shape[0] / sr * 1000))
            if start_ms < prev_end and adjusted_segments:
                adjusted_segments[-1][1] = start_ms
            adjusted_segments.append([start_ms, end_ms])
            prev_end = end_ms

        segment_infos = []
        for idx, (start_ms, end_ms) in enumerate(adjusted_segments):
            if end_ms - start_ms > cfg.max_len_ms:
                start_ms = end_ms - cfg.max_len_ms
            key = f"{base_name}_{idx}"
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000)
            segment_infos.append(
                {
                    "item_name": key,
                    "wav": y[start_sample:end_sample],
                    "sample_rate": sr,
                    "wav_fn": key,
                    "start_time_ms": int(start_sample * 1000 / sr),
                    "end_time_ms": int(end_sample * 1000 / sr),
                    "origin_wav_fn": origin_wav_fn,
                    "duration": int((end_sample - start_sample) * 1000 / sr),
                }
            )

        if verbose:
            logger.info(
                "[vocal detection] done n_segments=%s time=%.3fs",
                len(segment_infos),
                time.time() - t0,
            )
        return segment_infos
