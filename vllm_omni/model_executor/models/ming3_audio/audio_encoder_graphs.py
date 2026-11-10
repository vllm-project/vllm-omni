# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for the Ming-3 Thinker audio encoder and projector.

Encoding one clip launches several hundred kernels (two convolutions, 32
Whisper blocks and the projector), about 7 ms of host time in the Thinker
step that carries its prefill. A clip that fits one Whisper chunk is instead
zero-padded to a captured length and replayed. The padding leaves the clip's
embeddings as the unpadded pass computes them: every convolution output past
its valid length is zeroed, as the convolution's own padding would read it,
and attention covers only the clip's frames through FlashAttention varlen
boundaries, so the unmasked encoder still sees exactly its own frames.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ming.audio_lengths import conv_output_length, whisper_output_length

# Mel frames (10 ms each) of the captured encoder passes.
_MEL_BUCKETS = (100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1800, 2100, 2400, 2700, 3000)


class Ming3AudioEncoderGraphs:
    """Replay the encoder and projector of one single-chunk clip from a graph."""

    def __init__(self, encoder, projector, *, max_frames: int, normalize: bool):
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

        self.encoder = encoder
        self.normalize = normalize
        self._attn = flash_attn_varlen_func
        self._fa_version = 3 if is_fa_version_supported(3) else 2
        self._conv_layers = [
            (module, projector.proj[index + 1])
            for index, module in enumerate(projector.proj)
            if isinstance(module, torch.nn.Conv1d)
        ]
        self._linear = projector.proj[-1]
        self.buckets = [b for b in _MEL_BUCKETS if b <= max_frames]
        param = encoder.conv1.weight
        self.device, self.dtype = param.device, param.dtype
        frames = max(self.buckets, default=0)
        self._mel = torch.zeros(frames, encoder.n_mels, device=self.device, dtype=self.dtype)
        # Valid lengths of: conv1 output, encoder output, each projector layer.
        self._lengths = torch.zeros(2 + len(self._conv_layers), device=self.device, dtype=torch.long)
        self._cu_seqlens = torch.zeros(2, device=self.device, dtype=torch.int32)
        self._positions = torch.arange(frames, device=self.device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}

    def lengths(self, frames: int) -> list[int]:
        """Valid lengths after conv1, the encoder and each projector layer."""
        lengths = [frames, whisper_output_length(frames)]
        for conv, _ in self._conv_layers:
            lengths.append(conv_output_length(lengths[-1], conv.kernel_size[0], conv.stride[0], conv.padding[0]))
        return lengths

    def _mask(self, x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        # [channels, time]: zero time steps at or past ``length`` (NaN-safe).
        return torch.where(self._positions[: x.shape[-1]] < length, x, 0)

    def _body(self, frames: int) -> torch.Tensor:
        encoder, lengths = self.encoder, self._lengths
        x = F.gelu(encoder.conv1(self._mel[:frames].T.unsqueeze(0)))
        x = F.gelu(encoder.conv2(self._mask(x, lengths[0])))
        x = encoder._add_positions(x.transpose(1, 2))[0]
        tokens, width = x.shape
        for block in encoder.blocks:
            attn = block.attn
            h = block.attn_ln(x)
            shape = (tokens, attn.n_head, width // attn.n_head)
            out = self._attn(
                attn.query(h).view(shape),
                attn.key(h).view(shape),
                attn.value(h).view(shape),
                tokens,
                self._cu_seqlens,
                tokens,
                cu_seqlens_k=self._cu_seqlens,
                causal=False,
                fa_version=self._fa_version,
            )
            x = x + attn.out(out.reshape(tokens, width))
            x = x + block.mlp(block.mlp_ln(x))
        y = self._mask(encoder.ln_post(x).T, lengths[1]).unsqueeze(0)
        for index, (conv, activation) in enumerate(self._conv_layers):
            y = self._mask(activation(conv(y)), lengths[2 + index])
        y = self._linear(y.transpose(1, 2))[0]
        return F.normalize(y, dim=-1) if self.normalize else y

    def _stage(self, frames: int) -> None:
        lengths = self.lengths(frames)
        # From pageable memory: staged before the call returns, no wait.
        self._lengths.copy_(torch.tensor(lengths), non_blocking=True)
        self._cu_seqlens[1].fill_(lengths[1])

    @torch.inference_mode()
    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            # Largest first so smaller buckets reuse its pool blocks.
            for frames in reversed(self.buckets):
                self._stage(frames)
                for _ in range(2):
                    self._body(frames)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
                    self._outputs[frames] = self._body(frames)
                self.graphs[frames] = graph
        torch.cuda.current_stream(self.device).wait_stream(stream)

    def encode(self, features: torch.Tensor) -> torch.Tensor | None:
        """Embeddings [tokens, llm_dim] of one clip's [frames, n_mels], or None."""
        frames = features.shape[0]
        bucket = next((b for b in self.buckets if b >= frames), None)
        if bucket is None:
            return None
        self._stage(frames)
        self._mel[:frames].copy_(features)
        self._mel[frames:bucket].zero_()
        self.graphs[bucket].replay()
        return self._outputs[bucket][: self.lengths(frames)[-1]].clone()
