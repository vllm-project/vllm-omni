# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas@16a4c228 (Copyright 2026 Tencent).

from __future__ import annotations

import math

import torch


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    if window < 1 or stride < 1 or stride > window:
        raise ValueError(f"expected 1 <= stride <= window, got stride={stride}, window={window}")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    tail = length - window
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _slice_along(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]


def _merge_overlapping(parts: list[torch.Tensor], starts: list[int], *, dim: int) -> torch.Tensor:
    if len(parts) != len(starts) or not parts:
        raise ValueError("parts and starts must be non-empty and have equal length")
    result = parts[0]
    result_end = starts[0] + parts[0].shape[dim]
    for part, start in zip(parts[1:], starts[1:]):
        part_end = start + part.shape[dim]
        overlap = result_end - start
        if overlap <= 0 or part_end <= result_end:
            raise ValueError("overlap windows must be ordered and extend the range")
        overlap = min(overlap, result.shape[dim], part.shape[dim])
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
        shape = [1] * part.ndim
        shape[dim] = overlap
        weight = weight.view(shape).to(dtype=part.dtype)
        result = torch.cat(
            [
                _slice_along(result, dim, 0, result.shape[dim] - overlap),
                _slice_along(result, dim, result.shape[dim] - overlap, result.shape[dim]) * (1.0 - weight)
                + _slice_along(part, dim, 0, overlap) * weight,
                _slice_along(part, dim, overlap, part.shape[dim]),
            ],
            dim=dim,
        )
        result_end = part_end
    return result


class PixelUMMStreamingDecodeSession:
    """Publish stable pixels while retaining only overlapping decode windows.

    Each segment may omit up to three trailing decoder pixels. Absolute latent
    positions remain unchanged; the visible timeline excludes that padding.
    The pipeline owns decoder residency for each push and final flush.
    """

    final_segment = True

    def __init__(self, model, *, to_cpu: bool = True):
        self.model = model
        self.to_cpu = to_cpu
        self.buffer_latent_start = 0
        self.segment_raw_start = 0
        self.segment_visible_start = 0
        self.window = model.temporal_window_latents
        self.stride = self.window - model.temporal_overlap_latents
        self.temporal_ratio = model.time_compression_ratio
        self.buffer: torch.Tensor | None = None
        self.available_latent_frames = 0
        self.available_pixel_frames = 0
        self.next_window_start = 0
        self.pending: torch.Tensor | None = None
        self.pending_pixel_start = 0
        self.committed_raw_pixel_frames = 0
        self.committed_pixel_frames = 0
        self.push_latent_frames: list[int] = []
        self.push_pixel_frames: list[int] = []
        self.finished = False

    def _validate_parallel_state(self, operation: int, pixel_frames: int) -> None:
        if self.model._decode_group is None:
            return
        with self.model._decode_guard("stream state preparation"):
            shape = () if self.buffer is None else tuple(self.buffer.shape)
            values = (
                operation,
                pixel_frames,
                int(self.final_segment),
                int(self.finished),
                int(self.to_cpu),
                self.model._decode_output_rank,
                self.window,
                self.stride,
                self.temporal_ratio,
                self.available_latent_frames,
                self.available_pixel_frames,
                self.next_window_start,
                self.committed_raw_pixel_frames,
                self.committed_pixel_frames,
                self.buffer_latent_start,
                self.segment_raw_start,
                self.segment_visible_start,
                len(shape),
                len(self.push_latent_frames),
                len(self.push_pixel_frames),
                *shape,
                *self.push_latent_frames,
                *self.push_pixel_frames,
            )
        self.model._validate_decode_metadata(values)

    def _decode_windows(self, starts: list[int]) -> torch.Tensor | None:
        with self.model._decode_guard("stream window input"):
            assert self.buffer is not None
            noise = self.model._noise_tile(self.buffer, frame_offset=self.buffer_latent_start)
            relative_starts = [start - self.buffer_latent_start for start in starts]
        try:
            video = self.model._predict_x0(
                self.buffer,
                noise,
                t_starts=relative_starts,
                pad_short=bool(self.model.tail_hold_latents) and not self.final_segment,
            )
        finally:
            self.model._noise_cache = None
        with self.model._decode_guard("stream window output"):
            if video is not None and self.to_cpu:
                video = video.cpu()
        return video

    def _append_decoded_windows(self, video: torch.Tensor, *, first_window_start: int) -> None:
        clip_start = first_window_start * self.temporal_ratio
        if self.pending is None:
            self.pending = video
            self.pending_pixel_start = clip_start
        else:
            self.pending = _merge_overlapping([self.pending, video], [0, clip_start - self.pending_pixel_start], dim=2)

    @property
    def available_raw_pixel_frames(self) -> int:
        return self.available_latent_frames * self.temporal_ratio

    def _segment_raw_visible_spans(self):
        raw_start = self.segment_raw_start
        visible_start = self.segment_visible_start
        for latent_frames, pixel_frames in zip(self.push_latent_frames, self.push_pixel_frames):
            raw_end = raw_start + latent_frames * self.temporal_ratio
            visible_end = visible_start + pixel_frames
            yield raw_start, raw_start + pixel_frames, raw_end, visible_start, visible_end
            raw_start, visible_start = raw_end, visible_end

    def _visible_frames_before_raw(self, raw_end: int) -> int:
        raw_end = max(0, min(int(raw_end), self.available_raw_pixel_frames))
        visible = self.segment_visible_start
        for raw_start, raw_visible_end, _, _, visible_end in self._segment_raw_visible_spans():
            if raw_end <= raw_start:
                break
            visible += max(0, min(raw_end, raw_visible_end) - raw_start)
            if raw_end < raw_visible_end:
                break
            visible = visible_end
        return visible

    def _visible_slice_from_raw(self, video: torch.Tensor, *, raw_start: int, raw_end: int) -> torch.Tensor:
        """Drop per-segment decoder padding from one raw-pixel tensor."""
        expected = raw_end - raw_start
        if expected < 0 or int(video.shape[2]) != expected:
            raise RuntimeError(f"PixelUMM raw range mismatch: range=[{raw_start}, {raw_end}), video={video.shape[2]}")
        parts = []
        for segment_raw_start, segment_visible_raw_end, _, _, _ in self._segment_raw_visible_spans():
            start, end = max(raw_start, segment_raw_start), min(raw_end, segment_visible_raw_end)
            if end > start:
                parts.append(video[:, :, start - raw_start : end - raw_start])
        return torch.cat(parts, dim=2) if parts else video[:, :, :0]

    def _provisional_window_starts(self) -> list[int]:
        if self.available_latent_frames > self.window:
            first = max(0, (self.committed_raw_pixel_frames // self.temporal_ratio - self.window) // self.stride + 1)
            tail = self.available_latent_frames - self.window
            starts = list(range(first * self.stride, tail + 1, self.stride))
            if not starts or starts[-1] != tail:
                starts.append(tail)
            return starts
        return [0]

    def _trim_completed_prefix(self) -> None:
        assert self.buffer is not None
        stable_latent = self.committed_raw_pixel_frames // self.temporal_ratio
        keep = max(0, (stable_latent - self.window) // self.stride + 1) * self.stride
        keep = min(keep, self.next_window_start)
        if keep > self.buffer_latent_start:
            self.buffer = self.buffer[:, :, keep - self.buffer_latent_start :].clone()
            self.buffer_latent_start = keep
        count = 0
        for latent_frames, pixel_frames in zip(self.push_latent_frames, self.push_pixel_frames):
            raw_end = self.segment_raw_start + latent_frames * self.temporal_ratio
            if raw_end > self.committed_raw_pixel_frames:
                break
            self.segment_raw_start = raw_end
            self.segment_visible_start += pixel_frames
            count += 1
        if count:
            del self.push_latent_frames[:count]
            del self.push_pixel_frames[:count]

    def _decode_provisional_raw(self) -> torch.Tensor | None:
        """Decode the exact current tail from the raw stable boundary onward."""
        with self.model._decode_guard("stream tail planning"):
            assert self.buffer is not None
            raw_start, raw_end = self.committed_raw_pixel_frames, self.available_raw_pixel_frames
            selected_starts = None
            result = None
            regular_end = self.next_window_start - self.stride + self.window if self.next_window_start > 0 else 0
            if raw_end <= raw_start:
                if self.model._is_output_rank():
                    result = self.pending[:, :, :0] if self.pending is not None else self.buffer.new_empty((0,))
            elif raw_end <= regular_end * self.temporal_ratio:
                if self.model._is_output_rank():
                    if self.pending is None:
                        raise RuntimeError("PixelUMM is missing pending regular-window output")
                    pending_end = self.pending_pixel_start + int(self.pending.shape[2])
                    if not self.pending_pixel_start <= raw_start <= raw_end <= pending_end:
                        raise RuntimeError("PixelUMM pending output does not cover the decoded range")
                    result = self.pending[
                        :, :, raw_start - self.pending_pixel_start : raw_end - self.pending_pixel_start
                    ].contiguous()
            else:
                starts = self._provisional_window_starts()
                committed_latent = raw_start // self.temporal_ratio
                first_index = next(
                    (index for index, start in enumerate(starts) if start + self.window > committed_latent),
                    len(starts) - 1,
                )
                selected_starts = starts[first_index:]
        if selected_starts is None:
            return result
        decoded = self._decode_windows(selected_starts)
        with self.model._decode_guard("stream tail slicing"):
            if self.model._is_output_rank():
                assert decoded is not None
                decoded_raw_start = selected_starts[0] * self.temporal_ratio
                local_start, local_end = raw_start - decoded_raw_start, raw_end - decoded_raw_start
                if not 0 <= local_start <= local_end <= decoded.shape[2]:
                    raise RuntimeError("PixelUMM tail decode does not cover the requested raw range")
                result = decoded[:, :, local_start:local_end].contiguous()
        return result

    def _commit_until(self, stable_raw_until: int) -> torch.Tensor | None:
        if stable_raw_until <= self.committed_raw_pixel_frames:
            return None
        if not self.model._is_output_rank():
            self.committed_raw_pixel_frames = stable_raw_until
            self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
            return None
        if self.pending is None:
            raise RuntimeError("PixelUMM has no decoded pixels to commit")
        stable_local_end = stable_raw_until - self.pending_pixel_start
        if not 0 < stable_local_end <= self.pending.shape[2]:
            raise RuntimeError("Invalid PixelUMM stable-prefix boundary")
        stable = self._visible_slice_from_raw(
            self.pending[:, :, :stable_local_end].contiguous(),
            raw_start=self.pending_pixel_start,
            raw_end=stable_raw_until,
        )
        if self.to_cpu:
            stable = stable.cpu()
        self.pending = self.pending[:, :, stable_local_end:].contiguous()
        self.pending_pixel_start = stable_raw_until
        self.committed_raw_pixel_frames = stable_raw_until
        self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
        return stable

    def _decode_ready_regular_windows(self) -> torch.Tensor | None:
        with self.model._decode_guard("stream window planning"):
            starts = []
            while self.next_window_start + self.window <= self.available_latent_frames:
                starts.append(self.next_window_start)
                self.next_window_start += self.stride
        if not starts:
            return None
        decoded = self._decode_windows(starts)
        with self.model._decode_guard("stream stable-prefix commit"):
            if self.model._is_output_rank():
                assert decoded is not None
                self._append_decoded_windows(decoded, first_window_start=starts[0])
            tail_start = self.available_latent_frames - self.window
            stable_latent_frames = (
                tail_start + 1 if tail_start >= 0 and tail_start % self.stride == 0 else max(0, tail_start)
            )
            stable = self._commit_until(stable_latent_frames * self.temporal_ratio)
        return stable

    @property
    def stable_pixel_frames(self) -> int:
        return self.committed_pixel_frames

    @property
    def withheld_pixel_frames(self) -> int:
        return self.available_pixel_frames - self.committed_pixel_frames

    @torch.no_grad()
    def push(self, latent: torch.Tensor, *, pixel_frames: int | None = None) -> torch.Tensor:
        """Append one segment and return newly stable pixels."""
        with self.model._decode_guard("stream input validation"):
            if self.finished:
                raise RuntimeError("Cannot push PixelUMM latents after finish().")
            if latent.ndim != 5 or latent.shape[2] <= 0:
                raise ValueError(f"expected non-empty [B,C,T,H,W], got {tuple(latent.shape)}")
            latent_frames = int(latent.shape[2])
            segment_pixel_frames = latent_frames * self.temporal_ratio if pixel_frames is None else int(pixel_frames)
            min_pixel_frames = (latent_frames - 1) * self.temporal_ratio + 1
            max_pixel_frames = latent_frames * self.temporal_ratio
            if not min_pixel_frames <= segment_pixel_frames <= max_pixel_frames:
                raise ValueError(
                    f"PixelUMM pixel_frames={segment_pixel_frames} is inconsistent with latent_frames={latent_frames}; "
                    f"expected [{min_pixel_frames}, {max_pixel_frames}]"
                )
        self._validate_parallel_state(0, segment_pixel_frames)
        self.model._validate_decode_latent(latent)
        with self.model._decode_guard("stream buffer append"):
            previous_stable = self.committed_pixel_frames
            self.buffer = latent.clone() if self.buffer is None else torch.cat([self.buffer, latent], dim=2)
            self.available_latent_frames += latent_frames
            self.available_pixel_frames += segment_pixel_frames
            self.push_latent_frames.append(latent_frames)
            self.push_pixel_frames.append(segment_pixel_frames)
        stable = self._decode_ready_regular_windows()
        with self.model._decode_guard("stream push finalization"):
            self._trim_completed_prefix()
            if stable is None or self.committed_pixel_frames <= previous_stable:
                stable = self.buffer.new_empty((0,))
        return stable

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        """Publish the withheld tail and release the rolling buffers."""
        with self.model._decode_guard("stream finish validation"):
            if not self.final_segment:
                raise ValueError("PixelUMM finish requires final_segment=True on the closing segment")
            if self.finished:
                raise RuntimeError("PixelUMM streaming decode is already finished.")
            if self.buffer is None:
                raise RuntimeError("Cannot finish an empty PixelUMM streaming session.")
        self._validate_parallel_state(1, -1)
        previous_stable = self.committed_pixel_frames
        provisional = self._decode_provisional_raw()
        with self.model._decode_guard("stream finish commit"):
            stable = None
            if self.model._is_output_rank():
                assert provisional is not None
                stable = self._visible_slice_from_raw(
                    provisional,
                    raw_start=self.committed_raw_pixel_frames,
                    raw_end=self.available_raw_pixel_frames,
                )
                if self.to_cpu:
                    stable = stable.cpu()
            self.pending_pixel_start = self.available_raw_pixel_frames
            self.committed_raw_pixel_frames = self.available_raw_pixel_frames
            self.committed_pixel_frames = self.available_pixel_frames
            self.finished = True
            self.buffer = self.buffer[:, :, :0].clone()
            self.pending = None
            self.push_latent_frames.clear()
            self.push_pixel_frames.clear()
            if stable is None or self.committed_pixel_frames <= previous_stable:
                stable = self.buffer.new_empty((0,))
        return stable
