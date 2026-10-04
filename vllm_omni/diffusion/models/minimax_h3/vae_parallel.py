# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 VAE gather, stream lifetime, failure agreement, and temporal output assembly."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

DecodedChunkCallback = Callable[[torch.Tensor, int, int], None]


def _gather_stack_to_rank_zero(segment: torch.Tensor, group: Any) -> torch.Tensor | None:
    """Destination-only gather without a second full-size ``torch.cat``.

    ``dist.gather`` writes directly into contiguous views of one leader-owned
    ``[world_size, *segment.shape]`` allocation.  Non-leaders allocate no
    receive tensor.
    """
    segment = segment.contiguous()
    output = segment.new_empty((group.world_size, *segment.shape)) if group.rank_in_group == 0 else None
    gather_list = list(output.unbind(0)) if output is not None else None
    dist.gather(segment, gather_list=gather_list, dst=group.ranks[0], group=group.device_group)
    return output


class _LeaderAssembler:
    """Serially join gathered segments while later windows decode."""

    def __init__(
        self,
        model: Any,
        callback: DecodedChunkCallback,
        *,
        main_frames: int,
        overlap_frames: int,
        output_frames: int,
        device: torch.device,
    ) -> None:
        self._model = model
        self._callback = callback
        self._main_frames = main_frames
        self._overlap_frames = overlap_frames
        self._output_frames = output_frames
        self._previous_overlap: torch.Tensor | None = None
        self._write_pos = 0
        self._error: BaseException | None = None
        self._stream = torch.get_device_module().Stream(device=device) if device.type == "cuda" else None
        self._round_done: torch.cuda.Event | None = None

    def _remember_error(self, error: BaseException) -> None:
        if self._error is None:
            error.__traceback__ = None
            self._error = error

    def push(self, segment: torch.Tensor) -> None:
        if self._error is not None:
            return
        try:
            if self._stream is None:
                self._push(segment)
                return
            self._stream.wait_stream(torch.get_device_module().current_stream(segment.device))
            segment.record_stream(self._stream)
            with torch.get_device_module().stream(self._stream):
                self._push(segment)
        except BaseException as error:
            self._remember_error(error)

    def _push(self, segment: torch.Tensor) -> None:
        main = segment[:, :, : self._main_frames]
        overlap = segment[:, :, self._main_frames :]
        if self._previous_overlap is not None:
            main = self._model.blend(self._previous_overlap, main, self._overlap_frames, dim=-3)
        frame_count = min(int(main.shape[2]), self._output_frames - self._write_pos)
        if frame_count > 0:
            self._callback(main[:, :, :frame_count], self._write_pos, self._output_frames)
            self._write_pos += frame_count
        self._previous_overlap = overlap.clone()

    def finish_round(self) -> None:
        if self._stream is None or self._error is not None:
            return
        try:
            self._round_done = torch.get_device_module().Event()
            self._round_done.record(self._stream)
        except BaseException as error:
            self._remember_error(error)

    def drain_previous_round(self) -> None:
        if self._round_done is not None:
            try:
                self._round_done.synchronize()
            except BaseException as error:
                self._remember_error(error)
            finally:
                self._round_done = None

    def finalize(self) -> None:
        if self._error is not None:
            return
        try:
            if self._previous_overlap is None:
                raise RuntimeError("MiniMax H3 temporal chunk parallel produced no final overlap")
            remaining = self._output_frames - self._write_pos
            tail = self._previous_overlap[:, :, :remaining]
            if self._stream is None:
                self._emit_tail(tail)
            else:
                with torch.get_device_module().stream(self._stream):
                    self._emit_tail(tail)
        except BaseException as error:
            self._remember_error(error)

    def _emit_tail(self, tail: torch.Tensor) -> None:
        if int(tail.shape[2]) > 0:
            self._callback(tail, self._write_pos, self._output_frames)
            self._write_pos += int(tail.shape[2])
        if self._write_pos != self._output_frames:
            raise RuntimeError(
                f"MiniMax H3 temporal chunk parallel output length mismatch: "
                f"frames={self._write_pos}/{self._output_frames}"
            )

    def synchronize(self) -> None:
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as error:
                self._remember_error(error)

    @property
    def error(self) -> BaseException | None:
        return self._error


def _agree_on_failure(group: Any, device: torch.device, failed: bool) -> bool:
    flag = torch.tensor([int(failed)], dtype=torch.int32, device=device)
    result = group.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(result.item())


GATHER_OVERLAP_ENV = "VLLM_OMNI_H3_VAE_GATHER_OVERLAP"


def gather_overlap_enabled(*, paired: bool, mixed: bool) -> bool:
    value = os.environ.get(GATHER_OVERLAP_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{GATHER_OVERLAP_ENV} must be 0 or 1")
    if value == "1" and not (paired and mixed):
        raise ValueError(f"{GATHER_OVERLAP_ENV}=1 requires paired full-H3 VAE and mixed B2/B1")
    return value == "1"


def prepare_gather_stream(device, *, paired: bool, mixed: bool, owner):
    """Run during existing all-rank sink/setup admission, before any gather."""
    if not gather_overlap_enabled(paired=paired, mixed=mixed):
        return None
    if device.type != "cuda":
        raise ValueError(f"{GATHER_OVERLAP_ENV}=1 requires CUDA")
    for name in ("TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT"):
        if os.environ.get(name, "0") != "0":
            raise ValueError(f"{GATHER_OVERLAP_ENV}=1 requires {name} unset or 0")
    # This VAE already requires serial decode requests (mixed_decode temporarily
    # wraps model.decode). Reuse one stream per instance/device after the prior
    # request's unconditional drain, so allocator blocks retain a stable owner.
    index = device.index if device.index is not None else torch.accelerator.current_device_index()
    streams = getattr(owner, "_h3_vae_gather_streams", None)
    if streams is None:
        streams = {}
        owner._h3_vae_gather_streams = streams
    if index not in streams:
        streams[index] = torch.get_device_module().Stream(device=device)
    return streams[index]


def gather_on_stream(packed, group, stream):
    """Keep the NCCL completion dependency off the decoder caller stream.

    The existing synchronous Python gather waits for NCCL on its current CUDA
    stream. Here that current stream is dedicated to gather, so the subsequent
    decoder round may be submitted on the original caller stream. The leader
    must wait on this gather stream before consuming the returned tensor.
    """

    caller = torch.get_device_module().current_stream(packed.device)
    if stream == caller:
        raise RuntimeError("VAE gather overlap requires a distinct stream")
    stream.wait_stream(caller)
    # `packed` was allocated/written on caller. Its Python lifetime ends after
    # this round is submitted, potentially before the gather stream finishes.
    packed.record_stream(stream)
    with torch.get_device_module().stream(stream):
        return _gather_stack_to_rank_zero(packed, group)
