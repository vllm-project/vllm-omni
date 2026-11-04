# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas@16a4c228 (Copyright 2026 Tencent).

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import wraps

import torch
import torch.distributed as dist


@dataclass
class StreamingPreviewUpdate:
    """One complete segment preview plus any corrected preceding tail."""

    video: torch.Tensor
    pixel_start: int
    pixel_end: int
    replace_pixel_frames: int
    stable_pixel_frames: int
    provisional_pixel_frames: int


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


def _merge_overlapping(
    parts: list[torch.Tensor],
    starts: list[int],
    *,
    dim: int,
) -> torch.Tensor:
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
        position = torch.arange(
            1,
            overlap + 1,
            device=part.device,
            dtype=torch.float32,
        )
        weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
        shape = [1] * part.ndim
        shape[dim] = overlap
        weight = weight.view(shape).to(dtype=part.dtype)
        result_head = _slice_along(result, dim, 0, result.shape[dim] - overlap)
        result_tail = _slice_along(result, dim, result.shape[dim] - overlap, result.shape[dim])
        part_head = _slice_along(part, dim, 0, overlap)
        part_tail = _slice_along(part, dim, overlap, part.shape[dim])
        result = torch.cat(
            [
                result_head,
                result_tail * (1.0 - weight) + part_head * weight,
                part_tail,
            ],
            dim=dim,
        )
        result_end = part_end
    return result


def _streaming_cpu_offloaded(method):
    """Keep a complete streaming operation and its nested calls on device."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        controller = getattr(self.model, "_model_cpu_offload", None)
        if controller is None:
            return method(self, *args, **kwargs)
        self.model._last_inference_mode = torch.is_inference_mode_enabled()
        outermost = self._offload_depth == 0
        self._offload_depth += 1
        try:
            with controller.on_device():
                try:
                    if outermost:
                        self._move_decode_state(controller.device)
                    return method(self, *args, **kwargs)
                finally:
                    if outermost:
                        self._move_decode_state("cpu")
        finally:
            self._offload_depth -= 1

    return wrapped


class WindowStreamingDecodeSession:
    """Decode an append-only PixelUMM latent timeline without changing its result.

    PixelUMM attention is bidirectional inside each temporal window. Therefore the
    last overlap of the currently available timeline is not future-stable.
    ``push_preview`` still publishes that complete provisional tail immediately
    and replaces it after later pushes provide the missing future context.

    Segment boundaries may occur at any latent or pixel-frame position. A
    segment with ``L`` latent frames may request any pixel count in
    ``((L - 1) * temporal_ratio, L * temporal_ratio]``; unused decoder pixels at
    that segment's end are omitted from the visible timeline without changing
    the append-only latent timeline.

    With model CPU offload enabled, each push, preview, finish, or rollback
    restores the decoder and working state for the complete operation. The
    model, latent buffer, and pending pixels return to CPU between calls;
    returned pixels and retained output history still follow ``to_cpu``.

    ``rolling=True`` bounds stable-prefix sessions by window/chunk geometry.
    It requires ``retain_history=False`` and supports ``push``/``finish``;
    replace-tail previews and rollback require the original retained timeline.
    """

    final_segment = True

    def __init__(
        self,
        model,
        *,
        to_cpu: bool = True,
        retain_history: bool = True,
        rolling: bool = False,
    ):
        self.model = model
        self._offload_depth = 0
        self.to_cpu = bool(to_cpu)
        self.retain_history = bool(retain_history)
        if rolling and retain_history:
            raise ValueError("rolling decoding requires retain_history=False")
        self.rolling = bool(rolling)
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
        # ``pending`` is indexed on the decoder's raw pixel timeline. The raw
        # timeline has ``latent_frames * temporal_ratio`` slots per pushed
        # segment, while the visible timeline may omit up to
        # ``temporal_ratio - 1`` padded pixels at every segment boundary.
        self.pending_pixel_start = 0
        self.committed: list[tuple[int, torch.Tensor]] = []
        self.committed_raw_pixel_frames = 0
        self.committed_pixel_frames = 0
        self.push_latent_frames: list[int] = []
        self.push_pixel_frames: list[int] = []
        self.push_stable_pixel_frames: list[int] = []
        self.finished = False

    def _move_decode_state(self, device: torch.device | str) -> None:
        target = torch.device(device)
        for name in ("buffer", "pending"):
            tensor = getattr(self, name)
            # Window-parallel decode can assemble pixels on CPU; other modes
            # merge on the execution device even when outputs go to CPU.
            tensor_target = (
                torch.device("cpu")
                if name == "pending" and self.to_cpu and self.model._distributed_decode_mode == "window"
                else target
            )
            if tensor is not None and tensor.device != tensor_target:
                setattr(self, name, tensor.to(device=tensor_target, non_blocking=False))

    def _is_output_rank(self) -> bool:
        if self.model._distributed_decode_mode == "none" or not dist.is_available() or not dist.is_initialized():
            return True
        group = self.model._distributed_decode_group
        return dist.get_rank(group) == self.model._distributed_decode_output_rank

    def _decode_windows(self, starts: list[int]) -> torch.Tensor | None:
        raise NotImplementedError

    def _append_decoded_windows(
        self,
        video: torch.Tensor,
        *,
        first_window_start: int,
    ) -> None:
        clip_start = first_window_start * self.temporal_ratio
        if self.pending is None:
            self.pending = video
            self.pending_pixel_start = clip_start
            return
        relative_start = clip_start - self.pending_pixel_start
        self.pending = _merge_overlapping(
            [self.pending, video],
            [0, relative_start],
            dim=2,
        )

    @property
    def available_raw_pixel_frames(self) -> int:
        return self.available_latent_frames * self.temporal_ratio

    def _segment_raw_visible_spans(self):
        raw_start = self.segment_raw_start
        visible_start = self.segment_visible_start
        for latent_frames, pixel_frames in zip(
            self.push_latent_frames,
            self.push_pixel_frames,
        ):
            raw_end = raw_start + latent_frames * self.temporal_ratio
            visible_end = visible_start + pixel_frames
            yield (
                raw_start,
                raw_start + pixel_frames,
                raw_end,
                visible_start,
                visible_end,
            )
            raw_start = raw_end
            visible_start = visible_end

    def _visible_frames_before_raw(self, raw_end: int) -> int:
        raw_end = max(0, min(int(raw_end), self.available_raw_pixel_frames))
        visible = self.segment_visible_start
        for (
            raw_start,
            raw_visible_end,
            _raw_segment_end,
            _visible_start,
            visible_end,
        ) in self._segment_raw_visible_spans():
            if raw_end <= raw_start:
                break
            visible += max(
                0,
                min(raw_end, raw_visible_end) - raw_start,
            )
            if raw_end < raw_visible_end:
                break
            if raw_end >= raw_visible_end:
                visible = visible_end
        return visible

    def _visible_slice_from_raw(
        self,
        video: torch.Tensor,
        *,
        raw_start: int,
        raw_end: int,
    ) -> torch.Tensor:
        """Drop per-segment decoder padding from one raw-pixel tensor."""
        raw_start = int(raw_start)
        raw_end = int(raw_end)
        expected = raw_end - raw_start
        if expected < 0 or int(video.shape[2]) != expected:
            raise RuntimeError(
                f"PixelUMM raw preview range mismatch: range=[{raw_start}, {raw_end}), video={video.shape[2]}"
            )
        parts = []
        for (
            segment_raw_start,
            segment_visible_raw_end,
            _segment_raw_end,
            _segment_visible_start,
            _segment_visible_end,
        ) in self._segment_raw_visible_spans():
            overlap_start = max(raw_start, segment_raw_start)
            overlap_end = min(raw_end, segment_visible_raw_end)
            if overlap_end <= overlap_start:
                continue
            parts.append(
                video[
                    :,
                    :,
                    overlap_start - raw_start : overlap_end - raw_start,
                ]
            )
        if parts:
            return torch.cat(parts, dim=2)
        return video[:, :, :0]

    def _provisional_window_starts(self) -> list[int]:
        if self.rolling and self.available_latent_frames > self.window:
            first = max(0, (self.committed_raw_pixel_frames // self.temporal_ratio - self.window) // self.stride + 1)
            tail = self.available_latent_frames - self.window
            starts = list(range(first * self.stride, tail + 1, self.stride))
            if not starts or starts[-1] != tail:
                starts.append(tail)
            return starts
        return _window_starts(self.available_latent_frames, self.window, self.stride)

    def _trim_completed_prefix(self) -> None:
        if not self.rolling:
            return
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
            del self.push_stable_pixel_frames[:count]

    def _decode_provisional_raw(self) -> torch.Tensor | None:
        """Decode the exact current tail from the raw stable boundary onward."""
        raw_start = self.committed_raw_pixel_frames
        raw_end = self.available_raw_pixel_frames
        if raw_end <= raw_start:
            if not self._is_output_rank():
                return None
            assert self.buffer is not None
            if self.pending is not None:
                return self.pending[:, :, :0]
            return self.buffer.new_empty((0,))

        # All ranks can derive how far the regular-window decoder has covered.
        # When that already spans the requested tail, the output rank can reuse
        # ``pending`` and non-output ranks can skip a redundant collective call.
        regular_decoded_latent_end = (
            self.next_window_start - self.stride + self.window if self.next_window_start > 0 else 0
        )
        regular_decoded_raw_end = regular_decoded_latent_end * self.temporal_ratio
        if raw_end <= regular_decoded_raw_end:
            if not self._is_output_rank():
                return None
            if self.pending is not None:
                pending_end = self.pending_pixel_start + int(self.pending.shape[2])
                if not (self.pending_pixel_start <= raw_start and pending_end >= raw_end):
                    raise RuntimeError(
                        "PixelUMM pending output does not cover the regular decoded "
                        f"range [{raw_start}, {raw_end}); pending="
                        f"[{self.pending_pixel_start}, {pending_end})"
                    )
                return self.pending[
                    :,
                    :,
                    raw_start - self.pending_pixel_start : raw_end - self.pending_pixel_start,
                ].contiguous()
            raise RuntimeError("PixelUMM output rank is missing pending regular-window output")

        assert self.buffer is not None
        starts = self._provisional_window_starts()
        committed_latent = raw_start // self.temporal_ratio
        first_index = next(
            (index for index, start in enumerate(starts) if start + self.window > committed_latent),
            len(starts) - 1,
        )
        selected_starts = starts[first_index:]
        decoded = self._decode_windows(selected_starts)
        if not self._is_output_rank():
            return None
        assert decoded is not None
        decoded_raw_start = selected_starts[0] * self.temporal_ratio
        local_start = raw_start - decoded_raw_start
        local_end = raw_end - decoded_raw_start
        if not 0 <= local_start <= local_end <= decoded.shape[2]:
            raise RuntimeError(
                "PixelUMM provisional decode does not cover the requested raw range: "
                f"decoded_start={decoded_raw_start}, "
                f"requested=[{raw_start}, {raw_end}), "
                f"decoded_frames={decoded.shape[2]}"
            )
        return decoded[:, :, local_start:local_end].contiguous()

    def _commit_until(self, stable_raw_until: int) -> torch.Tensor | None:
        stable_raw_until = int(stable_raw_until)
        if stable_raw_until <= self.committed_raw_pixel_frames:
            return None
        if not self._is_output_rank():
            self.committed_raw_pixel_frames = stable_raw_until
            self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
            return None
        if self.pending is None:
            raise RuntimeError("PixelUMM output rank has no decoded pixels to commit")
        visible_start = self.committed_pixel_frames
        stable_local_end = stable_raw_until - self.pending_pixel_start
        if not 0 < stable_local_end <= self.pending.shape[2]:
            raise RuntimeError(
                "Invalid PixelUMM stable-prefix boundary: "
                f"local_end={stable_local_end}, pending={self.pending.shape[2]}"
            )
        stable_raw = self.pending[:, :, :stable_local_end].contiguous()
        stable = self._visible_slice_from_raw(
            stable_raw,
            raw_start=self.pending_pixel_start,
            raw_end=stable_raw_until,
        )
        if self.to_cpu:
            stable = stable.cpu()
        if self.retain_history and stable.shape[2] > 0:
            self.committed.append((visible_start, stable))
        self.pending = self.pending[:, :, stable_local_end:].contiguous()
        self.pending_pixel_start = stable_raw_until
        self.committed_raw_pixel_frames = stable_raw_until
        self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
        return stable

    def _decode_ready_regular_windows(self) -> torch.Tensor | None:
        starts = []
        while self.next_window_start + self.window <= self.available_latent_frames:
            starts.append(self.next_window_start)
            self.next_window_start += self.stride
        if not starts:
            return None
        decoded = self._decode_windows(starts)
        if self._is_output_rank():
            assert decoded is not None
            self._append_decoded_windows(
                decoded,
                first_window_start=starts[0],
            )
        # A non-stride-aligned end preview uses a window anchored at the current
        # tail. A one-latent append can shift that window one frame to the right,
        # so only frames strictly before that possible shifted window are stable.
        tail_start = self.available_latent_frames - self.window
        stable_latent_frames = (
            tail_start + 1 if tail_start >= 0 and tail_start % self.stride == 0 else max(0, tail_start)
        )
        return self._commit_until(stable_latent_frames * self.temporal_ratio)

    @property
    def stable_pixel_frames(self) -> int:
        return self.committed_pixel_frames

    @property
    def withheld_pixel_frames(self) -> int:
        return self.available_pixel_frames - self.committed_pixel_frames

    def slice_pixels(self, start: int, end: int) -> torch.Tensor:
        """Read a future-stable pixel range on the configured output rank."""
        if not self._is_output_rank():
            assert self.buffer is not None
            return self.buffer.new_empty((0,))
        if not self.retain_history:
            raise RuntimeError("PixelUMM streaming history retention is disabled for this session.")
        if not 0 <= start < end <= self.committed_pixel_frames:
            raise ValueError(
                "Requested PixelUMM pixels are not stable yet: "
                f"range=[{start}, {end}), "
                f"stable={self.committed_pixel_frames}."
            )
        selected = []
        cursor = start
        for part_start, part in self.committed:
            part_end = part_start + int(part.shape[2])
            overlap_start = max(start, part_start)
            overlap_end = min(end, part_end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start != cursor:
                raise RuntimeError(f"PixelUMM streaming output has a pixel gap: cursor={cursor}, next={overlap_start}.")
            selected.append(
                part[
                    :,
                    :,
                    overlap_start - part_start : overlap_end - part_start,
                ]
            )
            cursor = overlap_end
        if cursor != end or not selected:
            raise RuntimeError(
                f"PixelUMM streaming output does not cover requested range: [{start}, {end}), covered_to={cursor}."
            )
        return torch.cat(selected, dim=2)

    @torch.no_grad()
    @_streaming_cpu_offloaded
    def push(
        self,
        latent: torch.Tensor,
        *,
        pixel_frames: int | None = None,
    ) -> torch.Tensor:
        """Append one arbitrary-length segment and return newly stable pixels."""
        if self.finished:
            raise RuntimeError("Cannot push PixelUMM latents after finish().")
        if latent.ndim != 5 or latent.shape[2] <= 0:
            raise ValueError(f"expected non-empty [B,C,T,H,W], got {tuple(latent.shape)}")
        latent_frames = int(latent.shape[2])
        segment_pixel_frames = latent_frames * self.temporal_ratio if pixel_frames is None else int(pixel_frames)
        if segment_pixel_frames <= 0:
            raise ValueError("pixel_frames must be positive")
        min_pixel_frames = (latent_frames - 1) * self.temporal_ratio + 1
        max_pixel_frames = latent_frames * self.temporal_ratio
        if not min_pixel_frames <= segment_pixel_frames <= max_pixel_frames:
            raise ValueError(
                "PixelUMM streaming segment pixel length is inconsistent with its "
                f"latent length: latent_frames={latent_frames}, "
                f"pixel_frames={segment_pixel_frames}, "
                f"expected_range=[{min_pixel_frames}, {max_pixel_frames}]."
            )
        controller = getattr(self.model, "_model_cpu_offload", None)
        if controller is not None:
            latent = latent.to(device=controller.device, non_blocking=False)
        if self.model._distributed_decode_mode != "none":
            self.model._validate_distributed_input(latent)

        previous_stable = self.committed_pixel_frames
        self.buffer = (
            (latent.clone() if self.rolling else latent.contiguous())
            if self.buffer is None
            else torch.cat([self.buffer, latent], dim=2)
        )
        self.available_latent_frames += latent_frames
        self.available_pixel_frames += segment_pixel_frames
        self.push_latent_frames.append(latent_frames)
        self.push_pixel_frames.append(segment_pixel_frames)
        stable = self._decode_ready_regular_windows()
        self.push_stable_pixel_frames.append(self.committed_pixel_frames)
        self._trim_completed_prefix()

        if stable is None or self.committed_pixel_frames <= previous_stable:
            assert self.buffer is not None
            return self.buffer.new_empty((0,))
        return stable

    @torch.no_grad()
    @_streaming_cpu_offloaded
    def push_preview(
        self,
        latent: torch.Tensor,
        *,
        pixel_frames: int | None = None,
    ) -> StreamingPreviewUpdate:
        """Append a segment and publish its complete provisional preview.

        ``push()`` publishes only future-stable pixels.  Interactive web
        playback instead needs the complete segment immediately.  This method
        additionally exposes the current provisional overlap.  On the next
        push, the preceding provisional tail is merged with the newly decoded
        cross-boundary PixelUMM window and returned again for in-place replacement.

        Segment boundaries do not need to align to the PixelUMM stride. Callers copy
        the returned tensor over the preceding provisional tail, then append
        the remaining new frames.
        """
        if self.rolling:
            raise ValueError("rolling decoding supports stable push(), without replace-tail previews")
        previous_available = int(self.available_pixel_frames)
        previous_stable = int(self.committed_pixel_frames)
        stable = self.push(latent, pixel_frames=pixel_frames)

        replace_pixel_frames = previous_available - previous_stable
        pixel_start = previous_stable
        pixel_end = int(self.available_pixel_frames)
        provisional_pixel_frames = pixel_end - int(self.committed_pixel_frames)
        provisional_raw = None
        if provisional_pixel_frames > 0:
            # ``_decode_provisional_raw`` may enter the distributed window
            # decoder.  Every rank must therefore call it, even though only the
            # configured output rank keeps the returned pixels.
            provisional_raw = self._decode_provisional_raw()

        if not self._is_output_rank():
            assert self.buffer is not None
            video = self.buffer.new_empty((0,))
        else:
            parts = []
            if stable.numel() > 0:
                parts.append(stable)
            if provisional_pixel_frames > 0:
                assert provisional_raw is not None
                pending = self._visible_slice_from_raw(
                    provisional_raw,
                    raw_start=self.committed_raw_pixel_frames,
                    raw_end=self.available_raw_pixel_frames,
                )
                if self.to_cpu:
                    pending = pending.cpu()
                parts.append(pending)
            if not parts:
                assert self.buffer is not None
                video = self.buffer.new_empty((0,))
            elif len(parts) == 1:
                video = parts[0]
            else:
                video = torch.cat(parts, dim=2)
            expected = pixel_end - pixel_start
            if int(video.shape[2]) != expected:
                raise RuntimeError(
                    "PixelUMM preview update has the wrong pixel range: "
                    f"start={pixel_start}, end={pixel_end}, "
                    f"video={video.shape[2]}"
                )

        return StreamingPreviewUpdate(
            video=video,
            pixel_start=pixel_start,
            pixel_end=pixel_end,
            replace_pixel_frames=replace_pixel_frames,
            stable_pixel_frames=int(self.committed_pixel_frames),
            provisional_pixel_frames=provisional_pixel_frames,
        )

    @torch.no_grad()
    @_streaming_cpu_offloaded
    def finish(self) -> torch.Tensor:
        """Publish the withheld tail and close the append-only timeline."""
        if self.rolling and not self.final_segment:
            raise ValueError("rolling finish requires final_segment=True on the closing segment")
        if self.finished:
            raise RuntimeError("PixelUMM streaming decode is already finished.")
        if self.buffer is None:
            raise RuntimeError("Cannot finish an empty PixelUMM streaming session.")
        previous_stable = self.committed_pixel_frames
        provisional_raw = self._decode_provisional_raw()
        if self._is_output_rank():
            assert provisional_raw is not None
            stable = self._visible_slice_from_raw(
                provisional_raw,
                raw_start=self.committed_raw_pixel_frames,
                raw_end=self.available_raw_pixel_frames,
            )
            if self.to_cpu:
                stable = stable.cpu()
            if self.retain_history and stable.shape[2] > 0:
                self.committed.append((previous_stable, stable))
            if self.pending is not None:
                self.pending = self.pending[:, :, :0]
            self.pending_pixel_start = self.available_raw_pixel_frames
        else:
            stable = None
        self.committed_raw_pixel_frames = self.available_raw_pixel_frames
        self.committed_pixel_frames = self.available_pixel_frames
        self.finished = True
        if self.rolling:
            self.buffer = self.buffer[:, :, :0].clone()
            self.pending = None
            self.push_latent_frames.clear()
            self.push_pixel_frames.clear()
            self.push_stable_pixel_frames.clear()
        if stable is None or self.committed_pixel_frames <= previous_stable:
            assert self.buffer is not None
            return self.buffer.new_empty((0,))
        return stable

    @torch.no_grad()
    @_streaming_cpu_offloaded
    def rollback_last_push(self) -> None:
        """Discard the latest segment and rebuild the exact stable prefix."""
        if self.rolling:
            raise ValueError("rolling decoding does not retain rollback history")
        if self.finished:
            raise RuntimeError("Cannot roll back a finalized PixelUMM session.")
        if not self.push_latent_frames:
            raise RuntimeError("There is no PixelUMM streaming segment to roll back.")
        latent_frames = self.push_latent_frames.pop()
        self.push_pixel_frames.pop()
        self.push_stable_pixel_frames.pop()
        assert self.buffer is not None
        keep = self.available_latent_frames - latent_frames
        retained = self.buffer[:, :, :keep].contiguous() if keep > 0 else None
        retained_latent_frames = list(self.push_latent_frames)
        retained_pixel_frames = list(self.push_pixel_frames)
        retained_stable_pixel_frames = list(self.push_stable_pixel_frames)

        self.buffer = retained
        self.available_latent_frames = keep
        self.available_pixel_frames = sum(retained_pixel_frames)
        self.next_window_start = 0
        self.pending = None
        self.pending_pixel_start = 0
        self.committed = []
        self.committed_raw_pixel_frames = 0
        self.committed_pixel_frames = 0
        self.push_latent_frames = retained_latent_frames
        self.push_pixel_frames = retained_pixel_frames
        self.push_stable_pixel_frames = retained_stable_pixel_frames
        if retained is not None:
            self._decode_ready_regular_windows()
        if self.push_stable_pixel_frames and (self.push_stable_pixel_frames[-1] != self.committed_pixel_frames):
            raise RuntimeError(
                "Rebuilt PixelUMM stable prefix differs after rollback: "
                f"expected={self.push_stable_pixel_frames[-1]}, "
                f"rebuilt={self.committed_pixel_frames}."
            )

    def full_video(self) -> torch.Tensor:
        """Return the finalized full video on the configured output rank."""
        if not self.finished:
            raise RuntimeError("PixelUMM streaming decode is not finished")
        if not self.retain_history:
            raise RuntimeError("PixelUMM streaming history retention is disabled for this session.")
        return self.slice_pixels(0, self.available_pixel_frames)
