# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-step AudioVAE decode for single-stage Ming3 serving.

Every engine step adds at most one latent per stream. Here each stream's
newest latent is decoded in the step that produced it, all streams in one
pass, with every per-stream state on the device:

- The frame-holdback upsampler emits a latent's frames once the next latent
  arrives. With linear interpolation those frames are fixed combinations of
  the previous latent (left), the held latent and the new one, so a stream's
  first decode takes ``left = held`` and its final flush ``new = held``, as the
  streaming upsampler's edge handling does.
- The decoder's sliding-window Qwen2 runs packed over ``scale`` tokens per
  stream; each stream's keys sit in a ring of paged KV blocks.
- The ISTFT head overlap-adds each stream's frames onto its carried buffers.
  A step returns the whole overlap-add span per stream; the stream's audio is
  ``span[start:end]``, where ``start`` skips the "same" padding on a stream's
  first output and ``end`` keeps the tail on its final flush.

Whether a stream flushes is decided on the device (the stop head), so the
flush flag, the output end and the state updates are device tensors and a
step never synchronizes with the host. The pass replays from CUDA graphs
captured per row-count bucket; padding rows use a scratch slot and change no
stream's state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.utils.cuda_stream import create_cuda_stream

from .packed_qwen2 import PackedQwen2, PagedKV

# Decoder tokens one stream may reach (``scale`` per latent).
MAX_STREAM_POSITIONS = 16384
# Rows (streams decoded in one step) per captured graph.
GRAPH_ROWS = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)


def _interpolation_weights(scale: int) -> torch.Tensor:
    """[scale, 3] weights of (left, held, new) for the held latent's frames."""
    basis = torch.eye(3, dtype=torch.float64).unsqueeze(0)  # [1, channel=3 sources, time=3]
    frames = F.interpolate(basis, scale_factor=scale, mode="linear", align_corners=False)[0]
    return frames[:, scale : 2 * scale].T.contiguous().float()


class StepVAEDecoder:
    def __init__(self, decoder: nn.Module, *, max_slots: int, kv_block_size: int = 64, ring_blocks: int = 4):
        qwen = decoder.decoder
        config = qwen.config
        if decoder.patch_size == -1 or not getattr(decoder.upsampling, "frame_holdback", False):
            raise ValueError("Per-step VAE decode expects the frame-holdback upsampler")
        self.decoder = decoder
        self.scale = int(decoder.upsampling.scale_factor)
        self.window = int(config.sliding_window)
        if ring_blocks * kv_block_size < self.window - 1 + self.scale:
            raise ValueError("The KV ring must hold one window plus a step's tokens")
        self.kv_block = kv_block_size
        self.packed = PackedQwen2(qwen, MAX_STREAM_POSITIONS)
        param = next(decoder.parameters())
        device, dtype = param.device, param.dtype
        # Slot ``max_slots`` takes the padding rows of a graph replay; it is never a stream's.
        self.max_slots = max_slots
        slots = max_slots + 1
        self.keys, self.values = self.packed.new_paged_caches(slots * ring_blocks, kv_block_size)
        logical = torch.arange(-(-MAX_STREAM_POSITIONS // kv_block_size), device=device)
        first = torch.arange(slots, device=device)[:, None] * ring_blocks
        self.block_table = (first + logical[None, :] % ring_blocks).to(torch.int32)

        head = decoder.head
        self.head_out = head.out
        self.hop = int(head.istft.hop_length)
        self.n_fft = int(head.istft.n_fft)
        self.buffer_len = int(head.istft.win_length) - self.hop
        self.pad = self.buffer_len // 2
        self.span = (self.scale - 1) * self.hop + int(head.istft.win_length)
        window = head.istft.window.to(device=device, dtype=torch.float32)
        self.window_fn = window
        # The window envelope of one step's frames is the same for every stream.
        square = window.square()[None, :, None].expand(1, -1, self.scale)
        self.envelope = F.fold(square, (1, self.span), (1, window.numel()), stride=(1, self.hop))[0, 0, 0]
        self.weights = _interpolation_weights(self.scale).to(device=device, dtype=torch.float32)

        hidden = config.hidden_size
        self.held = torch.zeros((slots, hidden), device=device, dtype=dtype)
        self.left = torch.zeros((slots, hidden), device=device, dtype=dtype)
        self.has_left = torch.zeros(slots, device=device, dtype=torch.bool)
        self.audio_buffer = torch.zeros((slots, self.buffer_len), device=device, dtype=torch.float32)
        self.window_buffer = torch.zeros((slots, self.buffer_len), device=device, dtype=torch.float32)
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]] = {}
        self._static: tuple[torch.Tensor, ...] | None = None
        # record_stream may retain this handle after the decoder is released;
        # use PyTorch's process-lived pool for execution and an owned stream for capture.
        # With overlap enabled, callers must run prime()/step() under this stream.
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" and torch.version.hip is None else None

    @torch.inference_mode()
    def capture_graphs(self, rows: tuple[int, ...] = GRAPH_ROWS) -> None:
        """Capture the step for each row bucket; larger steps run eagerly."""
        device = self.held.device
        if device.type != "cuda":
            return
        rows = tuple(sorted({count for count in rows if count <= self.max_slots}))
        largest = rows[-1]
        latent_dim = self.decoder.fc1.in_features
        # Inputs: slots, positions, first, latents, last, valid. Padding rows
        # are invalid scratch-slot rows.
        self._static = (
            torch.full((largest,), self.max_slots, device=device, dtype=torch.long),
            torch.zeros(largest, device=device, dtype=torch.long),
            torch.ones(largest, device=device, dtype=torch.bool),
            torch.zeros((largest, latent_dim), device=device, dtype=self.held.dtype),
            torch.zeros(largest, device=device, dtype=torch.bool),
            torch.zeros(largest, device=device, dtype=torch.bool),
        )
        pool = torch.cuda.graph_pool_handle()
        stream = create_cuda_stream(device) if torch.version.hip is None else torch.cuda.Stream(device=device)
        for count in reversed(rows):
            inputs = tuple(tensor[:count] for tensor in self._static)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._step(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                outputs = self._step(*inputs)
            graph._capture_stream_owner = stream
            self._graphs[count] = (graph, outputs)

    @torch.inference_mode()
    def prime(self, slots: torch.Tensor, latents: torch.Tensor) -> None:
        """Hold each stream's first latent; streams start with no left frame and empty buffers."""
        self.held.index_copy_(0, slots, self.decoder.fc1(latents.to(self.held.dtype)))
        self.has_left.index_fill_(0, slots, False)
        self.audio_buffer.index_fill_(0, slots, 0.0)
        self.window_buffer.index_fill_(0, slots, 0.0)

    @torch.inference_mode()
    def step(
        self,
        slots: torch.Tensor,
        positions: torch.Tensor,
        first: torch.Tensor,
        latents: torch.Tensor,
        last: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = slots.shape[0]
        bucket = next((count for count in sorted(self._graphs) if count >= rows), None)
        if bucket is None:
            return self._step(slots, positions, first, latents, last, valid)
        assert self._static is not None
        for static, value in zip(self._static, (slots, positions, first, latents, last, valid), strict=True):
            static[:rows].copy_(value)
        if bucket > rows:
            # Padding rows: invalid, on the scratch slot.
            self._static[0][rows:bucket].fill_(self.max_slots)
            self._static[1][rows:bucket].zero_()
            self._static[5][rows:bucket].fill_(False)
        graph, outputs = self._graphs[bucket]
        graph.replay()
        # The next replay rewrites the outputs while the host copy may be pending.
        return tuple(output[:rows].clone() for output in outputs)

    def _step(
        self,
        slots: torch.Tensor,
        positions: torch.Tensor,
        first: torch.Tensor,
        latents: torch.Tensor,
        last: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the held latent of each row's stream.

        ``slots``/``positions`` ([rows], long) give each row's stream and its
        decoder tokens so far; ``first`` marks a stream's first output, ``last``
        its final flush (the held latent is the stream's last) and ``valid``
        rows whose state advances. ``latents`` [rows, latent_dim] are this
        step's new latents. Returns the overlap-add span [rows, span] (float32)
        and each row's audio bounds ``start``/``end`` within it.
        """
        rows = slots.shape[0]
        held = self.held.index_select(0, slots)
        new = self.decoder.fc1(latents.to(held.dtype))
        left = torch.where(self.has_left.index_select(0, slots)[:, None], self.left.index_select(0, slots), held)
        lookahead = torch.where(last[:, None], held, new)
        # [rows, scale, hidden]: the held latent's frames, in float32 like nn.Upsample's arithmetic.
        sources = torch.stack((left, held, lookahead), dim=1).float()
        frames = torch.einsum("sk,rkh->rsh", self.weights, sources).to(held.dtype).reshape(rows * self.scale, -1)

        offsets = torch.arange(self.scale, device=slots.device)
        token_positions = (positions[:, None] + offsets[None, :]).reshape(-1)
        blocks = self.block_table[slots[:, None], token_positions.view(rows, -1) // self.kv_block]
        cache_slots = (blocks.long() * self.kv_block + token_positions.view(rows, -1) % self.kv_block).reshape(-1)
        # Rows that do not advance keep their cache untouched.
        cache_slots = torch.where(valid.repeat_interleave(self.scale), cache_slots, -1)
        cu_seqlens = torch.arange(rows + 1, device=slots.device, dtype=torch.int32) * self.scale
        kv = PagedKV(
            self.keys,
            self.values,
            slot_mapping=cache_slots,
            seqused_k=(positions + valid * self.scale).to(torch.int32),
            block_table=self.block_table.index_select(0, slots),
            max_seqlen_k=MAX_STREAM_POSITIONS,
        )
        hidden = self.packed(frames, token_positions, cu_seqlens, self.scale, kv)

        # ISTFTHead: log-magnitude and phase per frequency bin, then overlap-add.
        prediction = self.head_out(hidden).view(rows, self.scale, -1).transpose(1, 2)
        magnitude, phase = prediction.chunk(2, dim=1)
        # Rounded like ISTFTHead: exp, clip, cos and sin in the model dtype, the product in float32.
        magnitude = torch.exp(magnitude).clip(max=1e2).float()
        spectrum = torch.complex(magnitude * torch.cos(phase).float(), magnitude * torch.sin(phase).float())
        segments = torch.fft.irfft(spectrum, self.n_fft, dim=1, norm="backward") * self.window_fn[None, :, None]
        audio = F.fold(segments, (1, self.span), (1, self.n_fft), stride=(1, self.hop))[:, 0, 0]
        envelope = self.envelope.expand(rows, -1).clone()
        carried = (~first)[:, None]
        audio[:, : self.buffer_len] += torch.where(carried, self.audio_buffer.index_select(0, slots), 0.0)
        envelope[:, : self.buffer_len] += torch.where(carried, self.window_buffer.index_select(0, slots), 0.0)

        advance = valid[:, None]
        self.audio_buffer.index_copy_(
            0, slots, torch.where(advance, audio[:, -self.buffer_len :], self.audio_buffer.index_select(0, slots))
        )
        self.window_buffer.index_copy_(
            0, slots, torch.where(advance, envelope[:, -self.buffer_len :], self.window_buffer.index_select(0, slots))
        )
        self.left.index_copy_(0, slots, torch.where(advance, held, self.left.index_select(0, slots)))
        self.held.index_copy_(0, slots, torch.where(advance, new, held))
        self.has_left.index_copy_(0, slots, valid | self.has_left.index_select(0, slots))

        start = torch.where(first, self.pad, 0)
        end = torch.where(last, self.span - self.pad, self.span - self.buffer_len)
        end = torch.where(valid, end, start)
        # Positions past a stream's output end may divide by a vanishing envelope; they are never read.
        return audio / envelope.clamp_min(1e-11), start, end


__all__ = ["StepVAEDecoder"]
