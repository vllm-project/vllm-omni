# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/codec_triton/stream.py @ d3612b0.
# The codec implements the VibeVoice acoustic tokenizer architecture (Copyright (c) 2025 Microsoft, MIT License).
"""Frame-by-frame VibeVoice codec on Triton kernels, one CUDA graph replay per frame.

``TritonCodecStream`` mirrors ``vllm_omni.model_executor.models.haru.vibevoice_codec.CodecStream``. Each
direction keeps its streaming state in static tensors and captures the
fixed-shape one-frame step for the stream's batch size in CUDA graphs, so a
live frame costs one input copy, one graph replay and one output copy.

State layout (private to this backend): one tensor ``[B, context, C_in]`` per
layer holding the last ``context`` input frames of its convolution (kernel -
stride frames for a conv, 6 for a block's depthwise conv, 1 for a transposed
conv). The state is double-buffered: a step reads one buffer set and writes
the other, and one graph is captured per direction of that swap.
"""

from __future__ import annotations

from collections.abc import Callable
from weakref import WeakKeyDictionary

import torch
from torch import nn

from vllm_omni.model_executor.models.haru.codec_triton import kernels
from vllm_omni.model_executor.models.haru.vibevoice_codec import (
    Block,
    CausalConv1d,
    CausalConvTranspose1d,
    VibeVoiceCodec,
)

Layer = tuple[Callable, kernels.Params, tuple[int, int]]  # kernel function, parameters, state shape [context, C_in]


def _layers(modules: list[nn.Module]) -> list[Layer]:
    """Kernel layers of an encoder or decoder ``sequence()``."""
    layers: list[Layer] = []
    for m in modules:
        if isinstance(m, Block):
            layers.append((kernels.block, kernels.block_params(m), (m.conv.context, m.conv.in_channels)))
        elif isinstance(m, CausalConv1d):
            layers.append((kernels.conv, kernels.conv_params(m), (m.context, m.in_channels)))
        elif isinstance(m, CausalConvTranspose1d):
            layers.append((kernels.conv_transpose, kernels.conv_transpose_params(m), (1, m.weight.shape[0])))
        else:
            raise TypeError(f"unknown codec layer {type(m).__name__}")
    return layers


def _run(
    layers: list[Layer], x: torch.Tensor, state: list[torch.Tensor], new_state: list[torch.Tensor]
) -> torch.Tensor:
    """Run ``layers`` on x ``[B, T, C]``, reading ``state`` and writing every tensor of ``new_state``."""
    normed = None  # first RMSNorm of the next block, when the previous kernel already produced it
    for i, (function, params, _) in enumerate(layers):
        following = layers[i + 1] if i + 1 < len(layers) else None
        next_block = following[1] if following is not None and following[0] is kernels.block else None
        next_norm = None if next_block is None else next_block["norm_w"]
        if function is kernels.block:
            x, normed = function(params, x, state[i], new_state[i], normed, next_norm)
        else:
            eps = 0.0 if next_block is None else next_block["eps"]
            x, normed = function(params, x, state[i], new_state[i], next_norm, eps)
    return x


class _FrameGraph:
    """The one-frame step of one codec direction for a fixed batch, as CUDA graphs over static buffers."""

    def __init__(
        self,
        layers: list[Layer],
        batch: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, ...],
        device: torch.device,
        pool: tuple[int, int],
    ):
        self.layers, self.batch, self.out_shape, self.pool = layers, batch, out_shape, pool
        self.input = torch.zeros(batch, *in_shape, device=device, dtype=torch.bfloat16)
        self.states = [
            [torch.zeros(batch, *shape, device=device, dtype=torch.bfloat16) for *_, shape in layers] for _ in range(2)
        ]
        self.current = 0  # index of the buffer set holding the live state
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.outputs: list[torch.Tensor] = []

    @torch.no_grad()
    def build(self) -> None:
        """Compile the kernels with one eager step, then capture a graph per state-swap direction."""
        if self.graphs:
            return
        spare = 1 - self.current
        _run(self.layers, self.input, self.states[self.current], self.states[spare])  # only overwrites `spare`
        graphs, outputs = [], []
        for source in (0, 1):
            graph = torch.cuda.CUDAGraph()
            # thread_local: other threads of the process may use CUDA during the capture
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                outputs.append(_run(self.layers, self.input, self.states[source], self.states[1 - source]))
            graphs.append(graph)
        self.graphs, self.outputs = graphs, outputs

    def step(self, frame: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Advance every row by one frame; returns a fresh tensor of ``out_shape`` in ``dtype``."""
        self.input.copy_(frame.reshape(self.input.shape))
        self.graphs[self.current].replay()
        output = self.outputs[self.current]
        self.current = 1 - self.current
        return output.view(self.out_shape).to(dtype, copy=True)

    def reset(self, rows: torch.Tensor | list[int] | None) -> None:
        for state in self.states[self.current]:
            if rows is None:
                state.zero_()
            else:
                state[rows] = 0


_LAYERS: WeakKeyDictionary[VibeVoiceCodec, tuple[list[Layer], list[Layer]]] = WeakKeyDictionary()


def _codec_layers(codec: VibeVoiceCodec) -> tuple[list[Layer], list[Layer]]:
    """Encoder and decoder kernel layers of ``codec``, built once and shared by all its streams."""
    if codec not in _LAYERS:
        _LAYERS[codec] = (_layers(codec.encoder.sequence()), _layers(codec.decoder.sequence()))
    return _LAYERS[codec]


class TritonCodecStream:
    """Frame-by-frame encode and decode for ``batch`` parallel conversations, on Triton kernels.

    Same interface and results (up to bfloat16 rounding) as ``CodecStream``. The backend always
    computes in bfloat16, whatever the codec's dtype, and returns tensors in the codec's dtype.

    The kernel parameters are derived from the codec's weights once per codec and shared by all
    its streams; they are not refreshed if the weights change afterwards.

    The CUDA graphs are built on the first ``encode`` / ``decode`` (or by ``warmup()``). The very
    first build on a machine also compiles the kernels (about 15 s); Triton caches the binaries on
    disk, so later processes take about 1 s for their first stream and tens of milliseconds for
    each further one.

    Threading: a stream may be built in one thread and used from another, but calls on one stream
    must not overlap, and it cannot be called while its thread is capturing another CUDA graph. The
    codec's device must be the calling thread's current CUDA device.
    """

    def __init__(self, codec: VibeVoiceCodec, batch: int = 1):
        if codec.device.type != "cuda":
            raise ValueError("TritonCodecStream needs the codec on a CUDA device")
        self.codec = codec
        self.batch = batch
        self.frame_samples = codec.config.frame_samples
        self.latent_dim = codec.config.latent_dim
        encoder, decoder = _codec_layers(codec)
        pool = torch.cuda.graph_pool_handle()
        self._encoder = _FrameGraph(
            encoder, batch, (self.frame_samples, 1), (batch, 1, self.latent_dim), codec.device, pool
        )
        self._decoder = _FrameGraph(
            decoder, batch, (1, self.latent_dim), (batch, self.frame_samples), codec.device, pool
        )

    def warmup(self) -> None:
        """Compile the kernels and capture both graphs now instead of on the first frame."""
        self._encoder.build()
        self._decoder.build()

    def reset(self, rows: torch.Tensor | list[int] | None = None) -> None:
        """Clear all rows, or only the conversations at ``rows``."""
        self._encoder.reset(rows)
        self._decoder.reset(rows)

    def _stream(self, graph: _FrameGraph, frames: torch.Tensor) -> torch.Tensor:
        """Step ``graph`` over frames ``[B, n, ...]`` and concatenate the outputs along dim 1."""
        if frames.shape[0] != self.batch:
            raise ValueError(f"expected {self.batch} rows, got {frames.shape[0]}")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TritonCodecStream cannot run inside another CUDA graph capture")
        graph.build()
        outputs = [graph.step(frames[:, i], self.codec.dtype) for i in range(frames.shape[1])]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """``[B, n * 3200]`` new audio -> latent means ``[B, n, D]``."""
        if audio.shape[-1] == 0 or audio.shape[-1] % self.frame_samples:
            raise ValueError(f"audio length must be a positive multiple of {self.frame_samples} samples")
        return self._stream(self._encoder, audio.reshape(audio.shape[0], -1, self.frame_samples))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, n, D]`` new latents -> audio ``[B, n * 3200]``."""
        if latents.shape[1] == 0:
            raise ValueError("latents must hold at least one frame")
        return self._stream(self._decoder, latents)
