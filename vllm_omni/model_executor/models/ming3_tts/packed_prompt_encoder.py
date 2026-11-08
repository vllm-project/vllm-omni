# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused-kernel forward for the Ming3 prompt-audio encoder.

The AudioVAE prompt encoder is a 24-layer sliding-window Qwen2 over 50 Hz
frames followed by a 4-layer Qwen2 aggregator; both run through
:class:`PackedQwen2`. Results match transformers up to the attention
kernel's accumulation order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.utils.device_copy import to_device_nonblocking

from .packed_qwen2 import PackedQwen2

# Graphs of the packed multi-clip pass: frames of all clips in one pass, and
# the clips a pass may hold (unused sequences are empty).
PACKED_GRAPH_FRAMES = (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
PACKED_GRAPH_CLIPS = 16


def _copy_in(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy ``source`` into a static device buffer without waiting for queued device work."""
    if source.device.type == "cpu" and target.device.type != "cpu":
        source = source.pin_memory()
    target.copy_(source, non_blocking=True)


class PackedPromptEncoder:
    """``Encoder.forward(waveform)[0]`` for one clip, with fused kernels.

    The clip is ``[1, samples]`` with ``samples`` a multiple of the frame hop
    (the prompt graphs pad whole latent groups); the result is
    ``[1, latent groups, 2 * latent_dim]``.
    """

    def __init__(self, encoder: nn.Module, max_frames: int):
        if encoder.patch_size == -1 or encoder.input_dim != encoder.hop_size:
            raise ValueError("Packed prompt encoder expects non-overlapping frames and a patch aggregator")
        self.encoder = encoder
        self.max_frames = max_frames
        self.patch = encoder.patch_size
        max_groups = -(-max_frames // self.patch)
        self.backbone = PackedQwen2(encoder.encoder, max_frames)
        self.aggregator = PackedQwen2(encoder.aggregator, max_groups * (self.patch + 1))
        device = encoder.fc1.weight.device
        self.max_tokens = max_groups * (self.patch + 1)
        self._positions = torch.arange(max(max_frames, self.max_tokens), device=device)
        self._bounds = torch.tensor([0, 1], device=device, dtype=torch.int32)
        # frames -> (graph, static frames, static long metadata, static int32 metadata, output)
        self._packed_graphs: dict[int, tuple] = {}

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        if waveform.shape[0] != 1 or waveform.shape[-1] % encoder.hop_size:
            raise ValueError(f"Packed prompt encoder expects one hop-aligned clip, got {tuple(waveform.shape)}")
        frames = waveform.reshape(-1, encoder.input_dim)
        x = encoder.fc2(encoder.fc1(frames))
        count = x.shape[0]
        x = self.backbone(x, self._positions[:count], self._bounds * count, count)

        # One CLS slot after every patch of frames, as Encoder.pad_patch_insert_cls.
        groups = -(-count // self.patch)
        x = F.pad(x, (0, 0, 0, groups * self.patch - count)).view(groups, self.patch, -1)
        x = torch.cat((x, encoder.cls_embed[0].expand(groups, 1, -1)), dim=1).view(groups * (self.patch + 1), -1)
        length = x.shape[0]
        x = self.aggregator(x, self._positions[:length], self._bounds * length, length)
        x = x.view(groups, self.patch + 1, -1)[:, -1]
        return encoder.fc3(x).unsqueeze(0)

    def _sizes(self, frame_rows: int, clips: int) -> tuple[int, int, int]:
        """(aggregator rows, CLS rows, cu_seqlens length) of a pass over ``frame_rows`` frames of ``clips`` clips."""
        groups = frame_rows // self.patch + clips
        return groups * (self.patch + 1), groups, clips + 1

    def _metadata(self, counts: np.ndarray, frame_rows: int, clips: int) -> tuple[np.ndarray, np.ndarray]:
        """Index metadata of a packed pass, padded to ``frame_rows`` frames and ``clips`` sequences.

        The int64 part holds the frame positions, the aggregator's source rows
        (each patch's frames, zero rows past a clip's end, then a CLS row) and
        positions, and the CLS rows to read out; the int32 part the frame and
        token ``cu_seqlens``. Padding rows read the zero row and padding
        sequences are empty.
        """
        token_rows, group_rows, bounds = self._sizes(frame_rows, clips)
        patch = self.patch
        groups = -(-counts // patch)
        lengths = groups * (patch + 1)
        frame_starts = np.concatenate(([0], np.cumsum(counts)))
        token_starts = np.concatenate(([0], np.cumsum(lengths)))
        pad_row, cls_row = frame_rows, frame_rows + 1
        frame_positions = np.zeros(frame_rows, dtype=np.int64)
        sources = np.full(token_rows, pad_row, dtype=np.int64)
        token_positions = np.zeros(token_rows, dtype=np.int64)
        cls_tokens = np.zeros(group_rows, dtype=np.int64)
        group_start = 0
        for count, group_count, frame_start, token_start in zip(counts, groups, frame_starts, token_starts):
            frame_positions[frame_start : frame_start + count] = np.arange(count)
            frame = np.arange(group_count)[:, None] * patch + np.arange(patch)[None, :]
            rows = np.where(frame < count, frame_start + frame, pad_row)
            rows = np.concatenate((rows, np.full((group_count, 1), cls_row)), axis=1).reshape(-1)
            sources[token_start : token_start + rows.size] = rows
            token_positions[token_start : token_start + rows.size] = np.arange(rows.size)
            cls_tokens[group_start : group_start + group_count] = (
                token_start + np.arange(group_count) * (patch + 1) + patch
            )
            group_start += group_count
        cu_seqlens = np.concatenate(
            (
                np.pad(frame_starts, (0, bounds - frame_starts.size), mode="edge"),
                np.pad(token_starts, (0, bounds - token_starts.size), mode="edge"),
            )
        ).astype(np.int32)
        return np.concatenate((frame_positions, sources, token_positions, cls_tokens)), cu_seqlens

    def _packed_forward(self, frames: torch.Tensor, long_meta: torch.Tensor, int_meta: torch.Tensor, clips: int):
        encoder = self.encoder
        token_rows, group_rows, bounds = self._sizes(frames.shape[0], clips)
        frame_positions, sources, token_positions, cls_tokens = long_meta.split(
            [frames.shape[0], token_rows, token_rows, group_rows]
        )
        frame_bounds, token_bounds = int_meta.split([bounds, bounds])
        x = encoder.fc2(encoder.fc1(frames))
        x = self.backbone(x, frame_positions, frame_bounds, self.max_frames)
        x = torch.cat((x, x.new_zeros(1, x.shape[-1]), encoder.cls_embed[0])).index_select(0, sources)
        x = self.aggregator(x, token_positions, token_bounds, self.max_tokens)
        return encoder.fc3(x.index_select(0, cls_tokens))

    def capture_graphs(self, pool=None) -> None:
        """Capture the packed pass for each ``PACKED_GRAPH_FRAMES`` bucket (largest first, sharing ``pool``)."""
        encoder = self.encoder
        device, dtype = encoder.fc1.weight.device, encoder.fc1.weight.dtype
        for frame_rows in sorted(PACKED_GRAPH_FRAMES, reverse=True):
            frames = torch.zeros((frame_rows, encoder.input_dim), device=device, dtype=dtype)
            long_meta, int_meta = (
                torch.from_numpy(array).to(device)
                for array in self._metadata(
                    np.array([min(frame_rows, self.max_frames)]), frame_rows, PACKED_GRAPH_CLIPS
                )
            )
            stream = torch.cuda.Stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self._packed_forward(frames, long_meta, int_meta, PACKED_GRAPH_CLIPS)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                output = self._packed_forward(frames, long_meta, int_meta, PACKED_GRAPH_CLIPS)
            self._packed_graphs[frame_rows] = (graph, frames, long_meta, int_meta, output)

    def encode_clips(self, waveforms: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Encode several hop-aligned ``[1, samples]`` clips in one packed pass.

        Each clip is its own sequence in the attention, so the results equal
        per-clip ``__call__`` up to GEMM accumulation order. The clips may be
        on the host (in the encoder's dtype): they are uploaded in one copy.
        """
        encoder = self.encoder
        if any(waveform.shape[0] != 1 or waveform.shape[-1] % encoder.hop_size for waveform in waveforms):
            raise ValueError("Packed prompt encoder expects hop-aligned single clips")
        if any(waveform.shape[-1] // encoder.hop_size > self.max_frames for waveform in waveforms):
            raise ValueError("Packed prompt encoder clip exceeds max_frames")
        counts = np.array([waveform.shape[-1] // encoder.hop_size for waveform in waveforms])
        total, clips = int(counts.sum()), len(waveforms)
        frames = torch.cat([waveform.reshape(-1, encoder.input_dim) for waveform in waveforms])
        device = self._positions.device
        bucket = next((rows for rows in PACKED_GRAPH_FRAMES if rows >= total), None)
        entry = self._packed_graphs.get(bucket) if clips <= PACKED_GRAPH_CLIPS else None
        if entry is None:
            long_meta, int_meta = (
                to_device_nonblocking(torch.from_numpy(array), device) for array in self._metadata(counts, total, clips)
            )
            output = self._packed_forward(to_device_nonblocking(frames, device), long_meta, int_meta, clips)
        else:
            graph, static_frames, static_long, static_int, output = entry
            long_meta, int_meta = self._metadata(counts, bucket, PACKED_GRAPH_CLIPS)
            _copy_in(static_frames[:total], frames)
            _copy_in(static_long, torch.from_numpy(long_meta))
            _copy_in(static_int, torch.from_numpy(int_meta))
            graph.replay()
        groups = -(-counts // self.patch)
        # Graph outputs are static buffers; the results outlive the next replay.
        return [latents.unsqueeze(0) for latents in output[: int(groups.sum())].clone().split(groups.tolist())]


__all__ = ["PackedPromptEncoder"]
