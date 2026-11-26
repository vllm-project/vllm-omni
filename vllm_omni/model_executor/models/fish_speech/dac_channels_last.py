# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Channels-last decoding for the Fish S2 Pro / Drama 3 DAC decoder.

The NCHW decoder spends most of its GPU time outside the arithmetic: cuDNN
reformats every activation to NHWC and back, falls back to slow generic
engines for some dilated shapes, and the scripted Snake activation runs far
from memory bandwidth on the large high-rate tensors. Here activations stay
``(batch, time, channels)``: every conv is a channels-last ``conv2d`` view
that cuDNN consumes without reformatting, a dilated conv is computed as an
undilated conv over ``time / dilation`` (one phase per column), and Snake is
one Triton kernel. The approach follows SGLang-Omni's Drama 3 vocoder.

The quantizer half runs channels-last too: each codebook's embedding and
``out_proj`` fold into one table at load, so the latent is one
``embedding_bag`` sum; the window-limited transformer uses causal SDPA (a
device mask past the window); each upsample stage is one GEMM (stride ==
kernel transposed conv) and a ConvNeXt block on the channels-last conv path.

``DACDecodeGraphs`` captures the whole decode for ``(batch, frames)``
buckets: the decoder is causal, so a request right-padded to its bucket
decodes its own frames exactly. With ``DACStreamState`` a stream keeps each
conv's history and the window transformer's keys / values between windows,
so a window decodes only its new frames.

Requires weight norm to be baked into the convs (``FishSpeechDACDecoder``
does so at load).
"""

from __future__ import annotations

import bisect
import math
import time
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from vllm_omni.model_executor.models.fish_speech.dac_modules.codec import DAC, DecoderBlock, ResidualUnit
from vllm_omni.model_executor.models.fish_speech.dac_modules.conv import (
    CausalConvNet,
    CausalTransConvNet,
    ConvNeXtBlock,
)
from vllm_omni.model_executor.models.fish_speech.dac_modules.layers import Snake1d
from vllm_omni.model_executor.models.fish_speech.dac_modules.transformer import (
    WindowLimitedTransformer,
    apply_rotary_emb,
)

logger = init_logger(__name__)

_SNAKE_ROWS = 32
# PTX sin.approx (abs error ~1e-6 on the codec's ranges, far below the fp16
# activations) instead of the libdevice sin that made Snake compute bound.
_FAST_SIN = torch.version.hip is None


@triton.jit
def _snake_kernel(
    x_ptr,
    x_batch_stride,
    alpha_ptr,
    inv_alpha_ptr,
    bias_ptr,
    y_ptr,
    out_rows,
    frames,
    out_frames,
    pad_left,
    channels,
    history_ptr,
    slots_ptr,
    fresh_ptr,
    HAS_BIAS: tl.constexpr,
    HAS_HISTORY: tl.constexpr,
    FAST_SIN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """``y[b, pad_left + t] = snake(x[b, t] + bias)`` for ``t < frames``; zeros elsewhere in ``out_frames``.

    With ``HAS_HISTORY`` the left padding holds each row's saved samples
    instead, ``history[slots[b], :pad_left]`` (zeros for ``fresh`` rows).
    """
    row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channel < channels
    batch = row // out_frames
    t = row % out_frames - pad_left
    valid = (row < out_rows) & (t >= 0) & (t < frames)
    load_mask = valid[:, None] & channel_mask[None, :]
    x = tl.load(
        x_ptr + batch[:, None] * x_batch_stride + t[:, None] * channels + channel[None, :], mask=load_mask, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        x = x + tl.load(bias_ptr + channel, mask=channel_mask, other=0.0)[None, :]
    alpha = tl.load(alpha_ptr + channel, mask=channel_mask, other=1.0)
    inv_alpha = tl.load(inv_alpha_ptr + channel, mask=channel_mask, other=1.0)
    arg = alpha[None, :] * x
    if FAST_SIN:
        sine = tl.inline_asm_elementwise(
            "sin.approx.f32 $0, $1;", "=f,f", [arg], dtype=tl.float32, is_pure=True, pack=1
        )
    else:
        sine = tl.sin(arg)
    y = tl.where(load_mask, x + inv_alpha[None, :] * sine * sine, 0.0)
    if HAS_HISTORY:
        history_rows = (row < out_rows) & (t < 0)
        slot = tl.load(slots_ptr + batch, mask=history_rows, other=0)
        fresh = tl.load(fresh_ptr + batch, mask=history_rows, other=1)
        history_mask = (history_rows & (fresh == 0))[:, None] & channel_mask[None, :]
        saved = tl.load(
            history_ptr + (slot[:, None] * pad_left + (t + pad_left)[:, None]) * channels + channel[None, :],
            mask=history_mask,
            other=0.0,
        ).to(tl.float32)
        y = tl.where(history_rows[:, None], saved, y)
    store_mask = (row < out_rows)[:, None] & channel_mask[None, :]
    tl.store(y_ptr + row[:, None] * channels + channel[None, :], y.to(y_ptr.dtype.element_ty), mask=store_mask)


def _channels_last_weight(conv: nn.Conv1d | nn.ConvTranspose1d) -> torch.Tensor:
    """The conv weight with channels-last strides (what cuDNN takes without a reformat)."""
    return conv.weight.detach().transpose(1, 2).contiguous().transpose(1, 2)


def _rotary_index(positions: torch.Tensor, transformer: WindowLimitedTransformer) -> torch.Tensor:
    """``positions`` as rows of the rotary table. The decoder keeps every real frame inside the
    table; only padding frames of a graph bucket can run past its end, and their outputs are
    dropped, so they reuse the last row instead of indexing out of bounds."""
    return positions.clamp(max=transformer.freqs_cis.shape[0] - 1)


def _rotary_rows(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_emb`` with per-row positions: ``x (B, T, H, D)``, ``freqs (B, T, D / 2, 2)``."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = freqs[:, :, None].float()
    out = torch.stack(
        (pairs[..., 0] * f[..., 0] - pairs[..., 1] * f[..., 1], pairs[..., 1] * f[..., 0] + pairs[..., 0] * f[..., 1]),
        -1,
    )
    return out.flatten(3).type_as(x)


_HISTORY_BLOCK = 1024


@triton.jit
def _history_fill_kernel(dst_ptr, dst_batch_stride, history_ptr, slots_ptr, fresh_ptr, numel, BLOCK: tl.constexpr):
    """``dst[b, :pad] = history[slots[b]]`` (zeros for fresh rows); ``numel`` = pad * channels."""
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    slot = tl.load(slots_ptr + b)
    fresh = tl.load(fresh_ptr + b)
    saved = tl.load(history_ptr + slot * numel + offs, mask=mask & (fresh == 0), other=0.0)
    tl.store(dst_ptr + b.to(tl.int64) * dst_batch_stride + offs, saved, mask=mask)


@triton.jit
def _history_save_kernel(
    src_ptr, src_batch_stride, history_ptr, slots_ptr, frames_ptr, rate, channels, numel, BLOCK: tl.constexpr
):
    """``history[slots[b]] = src[b, frames[b] * rate : + pad]``; ``src`` holds the history first."""
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    slot = tl.load(slots_ptr + b)
    start = tl.load(frames_ptr + b) * rate * channels
    value = tl.load(src_ptr + b.to(tl.int64) * src_batch_stride + start + offs, mask=mask)
    tl.store(history_ptr + slot * numel + offs, value, mask=mask)


class DACStreamState:
    """Each streaming request's decode history, kept in slots: every convolution's left
    context and the window transformer's keys / values.

    A window then decodes only its new frames, as the whole utterance would:
    each causal conv reads its ``(kernel - 1) * dilation`` samples of history
    (each transposed conv its one previous input sample) from the request's
    slot and writes back the samples ending at the row's last new frame, and
    the transformer attends over the slot's keys / values. Row ``num_slots``
    is scratch for batch padding. Per call, ``bind`` sets the rows' ``slots``,
    whether each starts ``fresh`` (zero history), their new ``frames`` and
    ``positions`` (frames decoded before this window) -- one ``(4, batch)``
    int64 device tensor, read inside the decode, so a captured graph sees new
    rows on replay -- and the frames every row of the batch spans.
    """

    def __init__(self, num_slots: int, max_window_frames: int = 64) -> None:
        self.num_slots = int(num_slots)
        # Most new frames one window brings: the key / value rings hold
        # ``window + max_window_frames`` entries, so writing a window's keys
        # before attending never overwrites a key one of its queries reads.
        self.max_window_frames = int(max_window_frames)
        self.tensors: dict[int, torch.Tensor] = {}
        self.slots: torch.Tensor | None = None
        self.fresh: torch.Tensor | None = None
        self.frames: torch.Tensor | None = None
        # Frames each row decoded before this window.
        self.positions: torch.Tensor | None = None
        self.span = 0
        # Window transformer keys / values per (layer, k|v): (num_slots + 1, window, heads, head_dim).
        self.kv_tensors: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def scratch(self) -> int:
        return self.num_slots

    def bind(self, rows: torch.Tensor, span: int) -> None:
        """``rows``: ``(4, batch)`` int64 -- slot, fresh (0 / 1), new frames and the frames each
        request decoded before this window."""
        if rows.shape[0] != 4:
            raise ValueError(f"stream rows must be (4, batch), got {tuple(rows.shape)}")
        self.slots, self.fresh, self.frames, self.positions = rows.unbind(0)
        self.span = int(span)

    def unbind(self) -> None:
        self.slots = self.fresh = self.frames = self.positions = None
        self.span = 0

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.tensors.values(), *self.kv_tensors.values()))

    def ring(self, layer: int, window: int, heads: int, head_dim: int, like: torch.Tensor) -> torch.Tensor:
        """One transformer layer's keys / values, ``(2, num_slots + 1, heads, window + max_window_frames,
        head_dim)`` (the slot-ring attention layout; allocated on first use, in warm-up)."""
        key = (layer, -1)
        cache = self.kv_tensors.get(key)
        if cache is None:
            capacity = window + self.max_window_frames
            cache = torch.zeros(2, self.num_slots + 1, heads, capacity, head_dim, dtype=like.dtype, device=like.device)
            self.kv_tensors[key] = cache
        return cache

    def kv(
        self, layer: int, window: int, heads: int, head_dim: int, like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The keys / values ring of one transformer layer (allocated on first use, in warm-up)."""
        caches = []
        for which in (0, 1):
            cache = self.kv_tensors.get((layer, which))
            if cache is None:
                cache = torch.zeros(self.num_slots + 1, window, heads, head_dim, dtype=like.dtype, device=like.device)
                self.kv_tensors[(layer, which)] = cache
            caches.append(cache)
        return caches[0], caches[1]

    def history(self, key: int, pad: int, like: torch.Tensor) -> torch.Tensor:
        """The ``(num_slots + 1, pad, channels)`` history of conv ``key`` (allocated on first use)."""
        state = self.tensors.get(key)
        if state is None:
            # First use happens in an eager warm-up, outside any graph pool.
            state = torch.zeros(self.num_slots + 1, pad, like.shape[2], dtype=like.dtype, device=like.device)
            self.tensors[key] = state
        return state

    def fill(self, key: int, padded: torch.Tensor, pad: int) -> None:
        """``padded[:, :pad]`` = each row's saved history (zeros for fresh rows), one kernel."""
        state = self.history(key, pad, padded)
        numel = pad * padded.shape[2]
        _history_fill_kernel[(padded.shape[0], triton.cdiv(numel, _HISTORY_BLOCK))](
            padded, padded.stride(0), state, self.slots, self.fresh, numel, BLOCK=_HISTORY_BLOCK
        )

    def save(self, key: int, padded: torch.Tensor, pad: int, length: int) -> None:
        """Keep the ``pad`` input samples ending at each row's last new frame (``padded`` holds the history first).

        Runs after the reads of the same history (fill or the Snake that wrote
        ``padded``) on the stream, so a row's old samples are read before
        they are replaced. Padding rows all write the scratch slot.
        """
        state = self.tensors[key]
        numel = pad * padded.shape[2]
        _history_save_kernel[(padded.shape[0], triton.cdiv(numel, _HISTORY_BLOCK))](
            padded,
            padded.stride(0),
            state,
            self.slots,
            self.frames,
            length // self.span,
            padded.shape[2],
            numel,
            BLOCK=_HISTORY_BLOCK,
        )


class _ChannelsLastCodec:
    """Channels-last ``(batch, time, channels)`` versions of the codec's ops.

    ``conv_dtype`` sets the dtype of the convolutions (weights are copied in
    channels-last strides at that dtype); transformers, ConvNeXt blocks and
    linear layers run in their modules' own dtype.
    """

    def __init__(self, modules: Iterable[nn.Module], conv_dtype: torch.dtype) -> None:
        self.conv_dtype = conv_dtype
        self.snake: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.weights: dict[int, torch.Tensor] = {}
        self.biases: dict[int, torch.Tensor | None] = {}
        self.biases_fp32: dict[int, torch.Tensor | None] = {}
        self.pointwise: dict[int, torch.Tensor] = {}
        for module in modules:
            if hasattr(module, "parametrizations") or hasattr(module, "weight_g"):
                raise ValueError("bake weight norm (dac_utils.bake_weight_norm) before building a channels-last codec")
            if isinstance(module, Snake1d):
                alpha = module.alpha.detach().float().reshape(-1).contiguous()
                self.snake[id(module)] = (alpha, (alpha + 1e-9).reciprocal())
            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                self.weights[id(module)] = _channels_last_weight(module).to(conv_dtype)
                bias = module.bias
                self.biases[id(module)] = None if bias is None else bias.detach().to(conv_dtype)
                # Biases deferred to the next Snake (see _run) stay in fp32.
                self.biases_fp32[id(module)] = None if bias is None else bias.detach().float()
                if isinstance(module, nn.Conv1d) and module.kernel_size[0] == 1 and module.groups == 1:
                    self.pointwise[id(module)] = module.weight.detach().squeeze(-1).t().contiguous().to(conv_dtype)
        self._window_masks: dict[tuple[int, int], torch.Tensor] = {}
        # Bound while a streaming decode reads / writes per-request conv history.
        self.stream: DACStreamState | None = None

    def _run(self, modules: Iterable[nn.Module], hidden: torch.Tensor) -> torch.Tensor:
        hidden, pending = self._stream(modules, hidden, None)
        return hidden if pending is None else hidden + pending.to(hidden.dtype)

    def _stream(
        self, modules: Iterable[nn.Module], hidden: torch.Tensor, pending: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run ``modules`` on a residual stream whose true value is ``hidden + pending``.

        Every reader of the stream in these codecs is a Snake (which adds the
        per-channel ``pending`` bias as it loads) or a residual unit (whose
        1x1 conv adds the stream as its GEMM's C operand), so conv biases are
        deferred instead of costing a pass over the high-rate activations.
        """
        modules = list(modules)
        history_row = False
        # Streaming, a Snake writes the next conv's padded input with its
        # history: one sample for a transposed conv, (kernel - 1) * dilation
        # for a causal conv (whose real length is then prepadded_frames).
        prepadded_frames: int | None = None
        for index, module in enumerate(modules):
            following = modules[index + 1] if index + 1 < len(modules) else None
            if isinstance(module, ResidualUnit):
                hidden, pending = self._residual_unit(module, hidden, pending)
            elif isinstance(module, Snake1d):
                pad, key = (0, 0), None
                if self.stream is not None and isinstance(following, CausalTransConvNet):
                    pad, key = (1, 0), id(following.conv)
                elif isinstance(following, CausalConvNet) and following.padding:
                    # The Snake writes the conv's zero-padded input directly
                    # (no separate pad copy), with history when streaming.
                    frames = hidden.shape[1]
                    pad = self._causal_padding(following, frames)
                    prepadded_frames = frames
                    if self.stream is not None:
                        key = id(following.conv)
                history_row = isinstance(following, CausalTransConvNet) and key is not None
                hidden = self._snake(module, hidden, bias=pending, pad=pad, history_key=key)
                pending = None
            elif isinstance(module, (DecoderBlock, nn.Sequential)):
                hidden, pending = self._stream(module.block if hasattr(module, "block") else module, hidden, pending)
            else:
                if pending is not None:
                    hidden = hidden + pending.to(hidden.dtype)
                    pending = None
                if isinstance(module, (CausalConvNet, CausalTransConvNet)):
                    # The bias waits for the next reader.
                    conv = module.conv
                    if isinstance(module, CausalConvNet):
                        hidden = self._causal_conv(module, hidden, bias=False, prepadded_frames=prepadded_frames)
                    else:
                        hidden = self._transposed_conv(module, hidden, bias=False, history_row=history_row)
                    pending = self.biases_fp32[id(conv)]
                elif isinstance(module, nn.Tanh):
                    hidden = torch.tanh(hidden)
                else:
                    raise TypeError(f"no channels-last path for {type(module).__name__}")
            if not isinstance(module, Snake1d):
                history_row = False
                prepadded_frames = None
        return hidden, pending

    def _residual_unit(
        self, unit: ResidualUnit, x: torch.Tensor, pending: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``x + conv1(snake(conv7(snake(x))))`` with the stream bias deferred and the residual in the GEMM."""
        snake1, conv7, snake2, conv1 = unit.block
        batch, frames, channels = x.shape
        dilation = conv7.conv.dilation[0]
        align = (-(conv7.padding + frames)) % dilation
        # Snake writes straight into the conv's zero-padded input.
        padded = self._snake(snake1, x, bias=pending, pad=(conv7.padding, align), history_key=id(conv7.conv))
        h = self._padded_conv(conv7, padded, frames)
        if self.stream is not None:
            self.stream.save(id(conv7.conv), padded, conv7.padding, frames)
        h = self._snake(snake2, h, bias=self.biases_fp32[id(conv7.conv)])
        # In place: the stream's old value has no other reader, and C += A @ B
        # costs no extra pass (an out-of-place addmm first copies C).
        residual = x.to(self.conv_dtype).contiguous().view(batch * frames, channels)
        out = residual.addmm_(h.view(-1, channels), self.pointwise[id(conv1.conv)])
        bias = self.biases_fp32[id(conv1.conv)]
        if bias is not None:
            pending = bias if pending is None else pending + bias
        return out.view(batch, frames, channels), pending

    def _padded_conv(self, module: CausalConvNet, padded: torch.Tensor, frames: int) -> torch.Tensor:
        """A stride-1 causal conv (no bias) on an input already holding its left padding and dilation alignment."""
        conv = module.conv
        dilation = conv.dilation[0]
        weight = self.weights[id(conv)]
        if dilation == 1:
            out = F.conv2d(padded.transpose(1, 2).unsqueeze(2), weight.unsqueeze(2), None, groups=conv.groups)
            return out.squeeze(2).transpose(1, 2)
        batch, _, channels = padded.shape
        out = F.conv2d(
            padded.view(batch, -1, dilation, channels).permute(0, 3, 1, 2),
            weight.unsqueeze(3),
            None,
            groups=conv.groups,
        )
        # A view: the next Snake reads it with its batch stride.
        return out.permute(0, 2, 3, 1).reshape(batch, -1, out.shape[1])[:, :frames]

    @staticmethod
    def _causal_padding(module: CausalConvNet, frames: int) -> tuple[int, int]:
        """(left, right) zeros of a stride-1 causal conv's input: the left context, then whole
        dilation phases."""
        return module.padding, (-(module.padding + frames)) % module.conv.dilation[0]

    def _causal_conv(
        self, module: CausalConvNet, hidden: torch.Tensor, bias: bool = True, prepadded_frames: int | None = None
    ) -> torch.Tensor:
        """Causal conv; with ``prepadded_frames``, ``hidden`` already holds its padding (and history, streaming)."""
        conv = module.conv
        frames = hidden.shape[1] if prepadded_frames is None else prepadded_frames
        dilation = conv.dilation[0]
        weight = self.weights[id(conv)]
        bias = self.biases[id(conv)] if bias else None
        hidden = hidden.to(self.conv_dtype)
        # Left context, plus right zeros so a dilated length divides evenly.
        align = (-(module.padding + frames)) % dilation
        if prepadded_frames is not None:
            padded = hidden
        else:
            padded = F.pad(hidden, (0, 0, module.padding, align))
            if self.stream is not None and module.padding:
                self.stream.fill(id(conv), padded, module.padding)
        if self.stream is not None and module.padding:
            self.stream.save(id(conv), padded, module.padding, frames)
        if dilation == 1:
            out = F.conv2d(padded.transpose(1, 2).unsqueeze(2), weight.unsqueeze(2), bias, groups=conv.groups)
            return out.squeeze(2).transpose(1, 2)
        # (B, L, C) viewed as (B, C, L / d, d): undilated along L / d, one phase per column.
        batch, _, channels = padded.shape
        out = F.conv2d(
            padded.view(batch, -1, dilation, channels).permute(0, 3, 1, 2),
            weight.unsqueeze(3),
            bias,
            groups=conv.groups,
        )
        return out.permute(0, 2, 3, 1).reshape(batch, -1, out.shape[1])[:, :frames].contiguous()

    def _transposed_conv(
        self, module: CausalTransConvNet, hidden: torch.Tensor, bias: bool = True, history_row: bool = False
    ) -> torch.Tensor:
        conv = module.conv
        stream = self.stream
        if stream is not None:
            # Output sample t also reads input t // stride - 1: put the previous
            # window's last input first, then drop the outputs it alone produced.
            if not history_row:
                hidden = F.pad(hidden, (0, 0, 1, 0))
                stream.fill(id(conv), hidden, 1)
            frames = hidden.shape[1] - 1
            stream.save(id(conv), hidden, 1, frames)
        out = (
            F.conv_transpose2d(
                hidden.to(self.conv_dtype).transpose(1, 2).unsqueeze(2),
                self.weights[id(conv)].unsqueeze(2),
                self.biases[id(conv)] if bias else None,
                stride=(1, conv.stride[0]),
                padding=(0, conv.padding[0]),
                output_padding=(0, conv.output_padding[0]),
                groups=conv.groups,
                dilation=(1, conv.dilation[0]),
            )
            .squeeze(2)
            .transpose(1, 2)
        )
        # Causal trim: drop the right overhang (kernel - stride samples).
        out = out[:, : out.shape[1] - (module.kernel_size - module.stride)]
        return out[:, module.stride :] if stream is not None else out

    def _snake(
        self,
        module: Snake1d,
        hidden: torch.Tensor,
        bias: torch.Tensor | None = None,
        pad: tuple[int, int] = (0, 0),
        history_key: int | None = None,
    ) -> torch.Tensor:
        """Snake of ``hidden + bias`` into a fresh ``(B, pad[0] + T + pad[1], C)`` tensor, zero padded.

        With ``history_key`` (streaming), the left padding holds that conv's
        saved history for each row instead of zeros.
        """
        if hidden.stride(2) != 1 or hidden.stride(1) != hidden.shape[2]:
            hidden = hidden.contiguous()
        alpha, inv_alpha = self.snake[id(module)]
        batch, frames, channels = hidden.shape
        out_frames = pad[0] + frames + pad[1]
        out = torch.empty((batch, out_frames, channels), dtype=hidden.dtype, device=hidden.device)
        rows = batch * out_frames
        block_c = 64 if channels <= 192 else 128
        stream = self.stream if history_key is not None and pad[0] else None
        history = stream.history(history_key, pad[0], out) if stream is not None else alpha
        if rows:
            _snake_kernel[(triton.cdiv(rows, _SNAKE_ROWS), triton.cdiv(channels, block_c))](
                hidden,
                hidden.stride(0),
                alpha,
                inv_alpha,
                bias if bias is not None else alpha,
                out,
                rows,
                frames,
                out_frames,
                pad[0],
                channels,
                history,
                stream.slots if stream is not None else alpha,
                stream.fresh if stream is not None else alpha,
                HAS_BIAS=bias is not None,
                HAS_HISTORY=stream is not None,
                FAST_SIN=_FAST_SIN,
                BLOCK_R=_SNAKE_ROWS,
                BLOCK_C=block_c,
            )
        return out

    def _window_transformer(self, transformer: WindowLimitedTransformer, x: torch.Tensor) -> torch.Tensor:
        """``WindowLimitedTransformer`` on ``(B, T, D)``: causal SDPA, a device mask past the window."""
        if self.stream is not None:
            attention = transformer.layers[0].attention
            if (
                x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16)
                and attention.n_head == attention.n_local_heads
                and attention.head_dim in (32, 64, 128)
            ):
                return self._window_transformer_ring(transformer, x)
            return self._window_transformer_kv(transformer, x)
        frames = x.shape[1]
        freqs = transformer.freqs_cis[:frames]
        window = transformer.window_size
        mask = None
        if window is not None and frames > window:
            mask = self._window_masks.get((frames, window))
            if mask is None:
                rows = torch.arange(frames, device=x.device)
                mask = (rows[None] <= rows[:, None]) & (rows[None] > rows[:, None] - window)
                self._window_masks[(frames, window)] = mask
        return self._transformer_layers(transformer, x, freqs, mask)

    def _transformer_layers(
        self, transformer: WindowLimitedTransformer, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = transformer.look_ahead_conv(transformer.input_proj(x))
        for layer in transformer.layers:
            h = x + layer.attention_layer_scale(self._attention(layer.attention, layer.attention_norm(x), freqs, mask))
            x = h + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(h)))
        return transformer.output_proj(transformer.norm(x))

    def _window_transformer_ring(self, transformer: WindowLimitedTransformer, x: torch.Tensor) -> torch.Tensor:
        """Streaming with keys / values in slot rings, attended in place (fp16 / bf16).

        Each layer's ``slot_ring_attention_rows`` writes the window's new keys /
        values into the request's ring, then attends each new frame over the
        ring entries of positions ``(p - window, p]`` -- no gather or concat of
        the history. A ring holds ``window + max_window_frames`` entries, so the
        window's own writes never overwrite a key that one of its queries still
        needs. Padding rows (scratch slot) get zero length: no write, no read.
        Reused slots need no clearing: entries outside ``[0, end)`` are masked.
        """
        stream = self.stream
        window = transformer.window_size
        batch, frames, _ = x.shape
        if frames > stream.max_window_frames:
            raise ValueError(f"a streamed window carries {frames} frames; the rings allow {stream.max_window_frames}")
        positions = stream.positions[:, None] + torch.arange(frames, device=x.device)[None]
        freqs = transformer.freqs_cis[_rotary_index(positions, transformer)]
        lengths = torch.where(stream.slots == stream.scratch, torch.zeros_like(stream.frames), stream.frames)
        attention = transformer.layers[0].attention
        rings = [
            stream.ring(index, window, attention.n_local_heads, attention.head_dim, x)
            for index in range(len(transformer.layers))
        ]
        return self._transformer_layers_ring(
            transformer, x, freqs, rings, stream.positions, stream.slots, lengths, window
        )

    def _transformer_layers_ring(
        self,
        transformer: WindowLimitedTransformer,
        x: torch.Tensor,
        freqs: torch.Tensor,
        rings: list[torch.Tensor],
        starts: torch.Tensor,
        slots: torch.Tensor,
        lengths: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        from vllm_omni.model_executor.models.moss_tts.slot_attention import slot_ring_attention_rows

        x = transformer.look_ahead_conv(transformer.input_proj(x))
        for layer, ring in zip(transformer.layers, rings, strict=True):
            attn = layer.attention
            h = layer.attention_norm(x)
            batch, frames, _ = h.shape
            size = attn.n_head * attn.head_dim
            q, k, v = attn.wqkv(h).split([size, size, size], dim=-1)
            q = _rotary_rows(q.view(batch, frames, attn.n_head, attn.head_dim), freqs).transpose(1, 2)
            k = _rotary_rows(k.view(batch, frames, attn.n_head, attn.head_dim), freqs).transpose(1, 2)
            v = v.view(batch, frames, attn.n_head, attn.head_dim).transpose(1, 2)
            y = slot_ring_attention_rows(q, k, v, ring, starts, slots, lengths, window)
            h = x + layer.attention_layer_scale(attn.wo(y.transpose(1, 2).reshape(batch, frames, size)))
            x = h + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(h)))
        return transformer.output_proj(transformer.norm(x))

    def _window_transformer_kv(self, transformer: WindowLimitedTransformer, x: torch.Tensor) -> torch.Tensor:
        """Streaming: ``x`` holds each row's new frames only; every layer attends to the keys / values
        of the row's previous ``window - 1`` frames kept in its slot, so the output equals the
        whole-utterance transformer's for these frames.

        A ring of ``window`` entries holds position ``p`` at ``p % window``. Each layer attends to
        [old ring, new frames] and only then writes the new frames into the ring, so a query never
        loses a key it still sees. A row starting at position ``e`` reads entry ``j`` as position
        ``e - 1 - ((e - 1 - j) mod window)``, valid when ``>= 0`` and inside the query's window:
        a reused slot needs no clearing.
        """
        stream = self.stream
        window = transformer.window_size
        batch, frames, _ = x.shape
        device = x.device
        offsets = torch.arange(frames, device=device)
        start = stream.positions[:, None]
        positions = start + offsets[None]
        freqs = transformer.freqs_cis[_rotary_index(positions, transformer)]
        ring = torch.arange(window, device=device)[None]
        cached_positions = start - 1 - torch.remainder(start - 1 - ring, window)
        cache_ok = (cached_positions[:, None, :] >= 0) & (cached_positions[:, None, :] > positions[:, :, None] - window)
        # Causal and inside the window: a window longer than ``window`` frames
        # must not let its later frames see its first ones.
        new_ok = (offsets[None, :] <= offsets[:, None]) & (offsets[None, :] > offsets[:, None] - window)
        new_ok = new_ok[None].expand(batch, -1, -1)
        mask = torch.cat((cache_ok, new_ok), dim=-1)[:, None]
        attention = transformer.layers[0].attention
        caches = [
            stream.kv(index, window, attention.n_local_heads, attention.head_dim, x)
            for index in range(len(transformer.layers))
        ]
        cached_k = [k.index_select(0, stream.slots) for k, _ in caches]
        cached_v = [v.index_select(0, stream.slots) for _, v in caches]
        x, new_kv = self._transformer_layers_kv(transformer, x, freqs, mask, cached_k, cached_v)
        # Only the rows' real new frames enter their rings, and of a window longer than
        # the ring only its last ``window`` frames (two frames on one entry would race);
        # padding goes to scratch.
        valid = (offsets[None, :] < stream.frames[:, None]) & (offsets[None, :] >= stream.frames[:, None] - window)
        slot_index = torch.where(valid, stream.slots[:, None], stream.scratch)
        ring_index = torch.remainder(positions, window)
        for (cache_k, cache_v), (k, v) in zip(caches, new_kv, strict=True):
            cache_k.index_put_((slot_index, ring_index), k.to(cache_k.dtype))
            cache_v.index_put_((slot_index, ring_index), v.to(cache_v.dtype))
        return x

    def _transformer_layers_kv(
        self,
        transformer: WindowLimitedTransformer,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor,
        cached_k: list[torch.Tensor],
        cached_v: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = transformer.look_ahead_conv(transformer.input_proj(x))
        new_kv = []
        for layer, keys, values in zip(transformer.layers, cached_k, cached_v, strict=True):
            out, k, v = self._attention_kv(layer.attention, layer.attention_norm(x), freqs, mask, keys, values)
            h = x + layer.attention_layer_scale(out)
            x = h + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(h)))
            new_kv.append((k, v))
        return transformer.output_proj(transformer.norm(x)), new_kv

    @staticmethod
    def _attention_kv(
        attn: nn.Module,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention of new frames over [cached keys, new keys]; returns the new (rotated) keys and values."""
        batch, frames, _ = x.shape
        q_size, kv_size = attn.n_head * attn.head_dim, attn.n_local_heads * attn.head_dim
        q, k, v = attn.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = _rotary_rows(q.view(batch, frames, attn.n_head, attn.head_dim), freqs)
        k = _rotary_rows(k.view(batch, frames, attn.n_local_heads, attn.head_dim), freqs)
        v = v.view(batch, frames, attn.n_local_heads, attn.head_dim)
        keys = torch.cat((cached_k.to(k.dtype), k), dim=1).transpose(1, 2)
        values = torch.cat((cached_v.to(v.dtype), v), dim=1).transpose(1, 2)
        if attn.n_head != attn.n_local_heads:
            keys = keys.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
            values = values.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), keys, values, attn_mask=mask)
        return attn.wo(y.transpose(1, 2).reshape(batch, frames, q_size)), k, v

    def compile_transformer(self) -> None:
        """Run the transformer layers through ``torch.compile`` (one dynamic-shape graph for every window).

        Eager, each layer is ~50 small kernels (rotary, RMSNorm, layer scale,
        residual adds, casts) around four GEMMs and the attention; Inductor
        fuses the elementwise work, which is most of a small window's time.
        """
        self._transformer_layers = torch.compile(self._transformer_layers, dynamic=True, fullgraph=True)
        self._transformer_layers_kv = torch.compile(self._transformer_layers_kv, dynamic=True, fullgraph=True)
        self._transformer_layers_ring = torch.compile(self._transformer_layers_ring, dynamic=True, fullgraph=True)

    @staticmethod
    def _attention(attn: nn.Module, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch, frames, _ = x.shape
        q_size, kv_size = attn.n_head * attn.head_dim, attn.n_local_heads * attn.head_dim
        q, k, v = attn.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = apply_rotary_emb(q.view(batch, frames, attn.n_head, attn.head_dim), freqs).transpose(1, 2)
        k = apply_rotary_emb(k.view(batch, frames, attn.n_local_heads, attn.head_dim), freqs).transpose(1, 2)
        v = v.view(batch, frames, attn.n_local_heads, attn.head_dim).transpose(1, 2)
        if attn.n_head != attn.n_local_heads:
            k = k.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
            v = v.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return attn.wo(y.transpose(1, 2).reshape(batch, frames, q_size))

    def _convnext(self, block: ConvNeXtBlock, x: torch.Tensor) -> torch.Tensor:
        dtype = block.pwconv1.weight.dtype
        y = self._causal_conv(block.dwconv, x).to(dtype)
        y = block.pwconv2(block.act(block.pwconv1(block.norm(y))))
        if block.gamma is not None:
            y = block.gamma * y
        return x.to(dtype) + y


def _receptive_field(modules: Iterable[nn.Module], rate: int) -> tuple[float, int]:
    """Left receptive field (in input frames) of a causal conv stack at ``rate`` samples per frame."""
    field = 0.0
    for module in modules:
        if isinstance(module, CausalConvNet):
            field += (module.kernel_size - 1) / rate
            rate //= module.stride
        elif isinstance(module, CausalTransConvNet):
            # Causal trim: an output depends on its own input sample and earlier overlaps.
            field += (math.ceil(module.kernel_size / module.stride) - 1) / rate
            rate *= module.stride
        elif isinstance(module, (DecoderBlock, ResidualUnit, nn.Sequential)):
            inner, rate = _receptive_field(module.block if hasattr(module, "block") else module, rate)
            field += inner
    return field, rate


def _code_tables(quantizer: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``out_proj(codebook)`` of every codebook (semantic first) in fp32, with offsets and limits."""
    tables, offsets, limits = [], [], []
    with torch.no_grad():
        for rvq in (quantizer.semantic_quantizer, quantizer.quantizer):
            for vq in rvq.quantizers:
                proj = vq.out_proj
                table = vq.codebook.weight.float() @ proj.weight.float().squeeze(-1).t() + proj.bias.float()
                offsets.append(sum(t.shape[0] for t in tables))
                limits.append(table.shape[0] - 1)
                tables.append(table)
    device = tables[0].device
    return (
        torch.cat(tables).contiguous(),
        torch.tensor(offsets, dtype=torch.long, device=device),
        torch.tensor(limits, dtype=torch.long, device=device),
    )


class ChannelsLastDACDecoder(_ChannelsLastCodec):
    """``codes [B, N, F] -> audio [B, 1, F * hop]`` through the codec's decoder in channels-last."""

    def __init__(self, codec: DAC) -> None:
        decoder = codec.decoder
        quantizer = codec.quantizer
        self.dtype = next(decoder.parameters()).dtype
        super().__init__(
            (*decoder.modules(), *quantizer.upsample.modules(), *quantizer.post_module.modules()), self.dtype
        )
        self.codec = codec
        for unit in codec.decoder.model.modules():
            if isinstance(unit, ResidualUnit):
                _, conv7, _, conv1 = unit.block
                if not (isinstance(conv7, CausalConvNet) and conv7.stride == 1 and id(conv1.conv) in self.pointwise):
                    raise ValueError("channels-last DAC residual units need a stride-1 causal conv and a 1x1 conv")
        self.hop_length = int(codec.frame_length)
        table, offsets, limits = _code_tables(quantizer)
        self.code_table = table
        self.code_offsets = offsets.view(1, -1, 1)
        self.code_limits = limits.view(1, -1, 1)
        self.transformer = quantizer.post_module
        self.upsample: list[tuple[torch.Tensor, torch.Tensor, int, int, ConvNeXtBlock]] = []
        for stage in quantizer.upsample:
            trans, block = stage[0], stage[1]
            conv = trans.conv
            kernel, stride = conv.kernel_size[0], conv.stride[0]
            if not isinstance(trans, CausalTransConvNet) or not isinstance(block, ConvNeXtBlock) or kernel != stride:
                raise TypeError(
                    "channels-last decoding expects (stride == kernel transposed conv, ConvNeXt) upsampling"
                )
            weight = conv.weight.detach()  # (in, out, kernel)
            # y[t * k + j, o] = sum_i x[t, i] W[i, o, j] + b[o]: one GEMM per stage.
            gemm = weight.permute(0, 2, 1).reshape(weight.shape[0], kernel * weight.shape[1]).contiguous()
            self.upsample.append((gemm, conv.bias.detach().repeat(kernel), kernel, weight.shape[1], block))
        self._receptive_frames: int | None = None

    @torch.inference_mode()
    def __call__(self, indices: torch.Tensor, trim: int = 0) -> torch.Tensor:
        """``codes [B, N, F] -> audio [B, 1, (F - trim) * hop]``.

        The transformer attends over all ``F`` frames; the convolutions run on
        the last ``F - trim`` only. With at least :attr:`receptive_frames`
        frames of context left before a frame, its samples equal those of the
        untrimmed decode.
        """
        return self._run(self.codec.decoder.model, self.latent(indices, trim)).transpose(1, 2)

    @property
    def receptive_frames(self) -> int:
        """Codec frames of left context the convolutions after the transformer can see (rounded up)."""
        if self._receptive_frames is None:
            rate, field = 1, 0.0
            for _, _, kernel, _, block in self.upsample:
                rate *= kernel
                field += (block.dwconv.kernel_size - 1) / rate
            self._receptive_frames = math.ceil(field + _receptive_field(self.codec.decoder.model, rate)[0])
        return self._receptive_frames

    def latent(self, indices: torch.Tensor, trim: int = 0) -> torch.Tensor:
        """``codes [B, N, F] -> (B, (F - trim) * upsample, D)``, the quantizer decode in channels-last."""
        batch, books, frames = indices.shape
        rows = torch.minimum(indices.clamp(min=0), self.code_limits[:, :books]) + self.code_offsets[:, :books]
        rows = rows.transpose(1, 2).reshape(batch * frames, books)
        hidden = F.embedding_bag(rows, self.code_table, mode="sum").view(batch, frames, -1).to(self.dtype)
        hidden = self._window_transformer(self.transformer, hidden)
        if trim:
            hidden = hidden[:, trim:].contiguous()
        for weight, bias, kernel, channels, block in self.upsample:
            batch, frames, width = hidden.shape
            hidden = torch.addmm(bias, hidden.reshape(batch * frames, width), weight).view(
                batch, frames * kernel, channels
            )
            hidden = self._convnext(block, hidden)
        return hidden


class DACDecodeGraphs:
    """CUDA graphs of a :class:`ChannelsLastDACDecoder` for ``(batch, frames, trim)`` buckets.

    A streaming window carries ``context`` frames before its new ones. The
    transformer needs all of them, but the convolutions only the last
    ``receptive_frames``: a window decodes its convolutions with
    ``trim = context - min(context, receptive_frames)`` frames dropped, which
    leaves its new samples exact and cuts the re-decoded context.

    ``plan`` groups a step's requests by (frame bucket, trim) so each group
    replays one captured shape; a group's output is copied out of the graph
    pool before the next replay. Shapes past the buckets decode eagerly.
    """

    def __init__(
        self,
        decoder: ChannelsLastDACDecoder,
        num_codebooks: int,
        frame_buckets: Iterable[int],
        max_padded_frames: int,
        max_batch: int,
        trims: dict[int, Iterable[int]] | None = None,
        receptive_frames: int | None = None,
        stream: DACStreamState | None = None,
    ) -> None:
        self.decoder = decoder
        self.receptive_frames = decoder.receptive_frames if receptive_frames is None else int(receptive_frames)
        self.num_codebooks = num_codebooks
        self.frame_buckets = sorted({int(f) for f in frame_buckets if int(f) > 0})
        self.trims = {
            f: sorted({int(t) for t in (trims or {}).get(f, (0,)) if 0 <= int(t) < f}) for f in self.frame_buckets
        }
        self.max_batch_for: dict[int, int] = {}
        self.batch_buckets: dict[int, list[int]] = {}
        for frames in self.frame_buckets:
            limit = max(1, min(max_batch, max_padded_frames // frames))
            sizes = sorted({b for b in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128) if b <= limit} | {limit})
            self.max_batch_for[frames] = limit
            self.batch_buckets[frames] = sizes
        self.graphs: dict[tuple[int, int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        # A shape's requests split into exact batch buckets (see split) rather
        # than padding up to the next one.
        self._splits: dict[tuple[int, int, int], list[int]] = {}
        # GPU ms of one replay per captured shape, measured right after capture.
        self.cost_ms: dict[tuple[int, int, int], float] = {}
        # Streaming with per-request history (receptive_frames 0: a window's
        # convolutions run on its new frames only). Each graph reads its rows
        # from a static (4, batch) tensor; padding rows use scratch.
        self.stream = stream
        self.stream_rows: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def frame_bucket(self, frames: int) -> int | None:
        index = bisect.bisect_left(self.frame_buckets, frames)
        return self.frame_buckets[index] if index < len(self.frame_buckets) else None

    def context_trim(self, context: int) -> int:
        return max(0, int(context) - min(int(context), self.receptive_frames))

    @torch.inference_mode()
    def capture(self, device: torch.device) -> None:
        shapes = sorted(
            ((b, f, t) for f in self.frame_buckets for t in self.trims[f] for b in self.batch_buckets[f]),
            key=lambda s: (-s[0] * (s[1] - s[2]), -s[1]),
        )
        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        for batch, frames, trim in shapes:
            static = torch.zeros((batch, self.num_codebooks, frames), dtype=torch.long, device=device)
            if self.stream is not None:
                idle = torch.tensor([[self.stream.scratch], [1], [frames - trim], [0]], dtype=torch.long, device=device)
                idle = idle.expand(4, batch).contiguous()
                rows = idle.clone()
                self.stream_rows[(batch, frames, trim)] = (rows, idle)
                self._bind(rows, frames - trim)
            for _ in range(2):
                self.decoder(static, trim)
            torch.accelerator.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self.decoder(static, trim)
            self._unbind()
            self.graphs[(batch, frames, trim)] = (graph, static, out)
            self.cost_ms[(batch, frames, trim)] = self._time_replay(graph)
        torch.accelerator.synchronize(device)
        logger.info(
            "Fish DAC decode CUDA graphs: %d shapes (frames %s, conv context %d frames) captured in %.1fs; "
            "replay ms (batch 1 / largest batch) %s",
            len(self.graphs),
            {f: self.trims[f] for f in self.frame_buckets},
            self.receptive_frames,
            time.perf_counter() - start,
            {
                (f, t): (round(self.cost_ms[(1, f, t)], 2), round(self.cost_ms[(self.batch_buckets[f][-1], f, t)], 2))
                for f in self.frame_buckets
                for t in self.trims[f]
                if (1, f, t) in self.cost_ms
            },
        )

    @staticmethod
    def _time_replay(graph: torch.cuda.CUDAGraph, reps: int = 3) -> float:
        """GPU ms of one replay (streaming graphs touch only the scratch slot here)."""
        graph.replay()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / reps

    def split(self, frames: int, count: int, trim: int = 0) -> list[int]:
        """Replay sizes for ``count`` requests of one (frame bucket, trim) shape with the least decode time.

        A step's windows all leave when its decode ends, so the plan
        minimizes the summed replay time, from the times measured at capture
        (a replay's fixed part -- the whole decoder at its smallest -- against
        the per-row cost of padding up to a larger batch graph).
        """
        sizes = self.batch_buckets[frames]
        key = (frames, trim, count)
        cached = self._splits.get(key)
        if cached is not None:
            return cached

        def cost(size: int) -> float:
            return self.cost_ms[(size, frames, trim)]

        # best[n] = (decode time, sizes) for n requests.
        best: list[tuple[float, list[int]]] = [(0.0, [])]
        for n in range(1, count + 1):
            options = []
            for size in sizes:
                if size >= n:
                    options.append((cost(size), [size]))
                else:
                    total, rest = best[n - size]
                    options.append((total + cost(size), [size, *rest]))
            best.append(min(options, key=lambda option: (option[0], len(option[1]))))
        self._splits[key] = best[count][1]
        return best[count][1]

    def plan(self, frames: Sequence[int], contexts: Sequence[int] | None = None) -> list[tuple[list[int], int]]:
        """``(indices, trim)`` groups of the requests, each fitting one captured shape (or decoding eagerly)."""
        contexts = contexts if contexts is not None else [0] * len(frames)
        by_shape: dict[tuple[int | None, int], list[int]] = {}
        for index, (count, context) in enumerate(zip(frames, contexts, strict=True)):
            bucket, trim = self.frame_bucket(count), self.context_trim(context)
            if bucket is not None and trim not in self.trims[bucket]:
                bucket = None
            by_shape.setdefault((bucket, trim if bucket is not None else 0), []).append(index)
        groups: list[tuple[list[int], int]] = []
        for (bucket, trim), members in by_shape.items():
            if bucket is None:
                groups.extend(([member], trim) for member in members)
                continue
            step = self.max_batch_for[bucket]
            for i in range(0, len(members), step):
                chunk = members[i : i + step]
                start = 0
                for size in self.split(bucket, len(chunk), trim):
                    groups.append((chunk[start : start + size], trim))
                    start += size
        return groups

    def _bind(self, rows: torch.Tensor, span: int) -> None:
        self.stream.bind(rows, span)
        self.decoder.stream = self.stream

    def _unbind(self) -> None:
        if self.stream is not None:
            self.stream.unbind()
        self.decoder.stream = None

    @torch.inference_mode()
    def __call__(self, codes: torch.Tensor, trim: int = 0, rows: torch.Tensor | None = None) -> torch.Tensor:
        """``codes [B, N, F] -> float32 audio [B, 1, (F - trim) * hop]`` in freshly allocated storage.

        Streaming graphs need ``rows`` (``(4, B)`` int64, see ``DACStreamState.bind``);
        each row's history is read from and written back to its slot.
        """
        batch, _, frames = codes.shape
        if (self.stream is not None) != (rows is not None):
            raise ValueError("streaming DAC graphs take per-row slots (and only they do)")
        bucket = self.frame_bucket(frames)
        if bucket is not None and trim in self.trims[bucket]:
            index = bisect.bisect_left(self.batch_buckets[bucket], batch)
            if index < len(self.batch_buckets[bucket]):
                key = (self.batch_buckets[bucket][index], bucket, trim)
                graph, static, out = self.graphs[key]
                # Padding rows / frames keep earlier valid codes; the codec is
                # causal, so they never reach this group's samples.
                static[:batch, :, :frames].copy_(codes)
                if rows is not None:
                    # Padding rows read and write the scratch slot only.
                    static_rows, idle = self.stream_rows[key]
                    static_rows.copy_(idle)
                    static_rows[:, :batch].copy_(rows)
                graph.replay()
                return out[:batch, :, : (frames - trim) * self.decoder.hop_length].to(torch.float32, copy=True)
        if rows is None:
            return self.decoder(codes, trim).to(torch.float32)
        self._bind(rows, frames - trim)
        try:
            return self.decoder(codes, trim).to(torch.float32)
        finally:
            self._unbind()


def streaming_decode_shapes(
    ramp: Sequence[int], chunk: int, left_context: int, receptive_frames: int
) -> tuple[list[int], dict[int, set[int]]]:
    """Frame buckets of the streaming chunker's windows (context included) and each bucket's conv trims.

    Every (frames, context) a window can carry -- a full window or a shorter
    final one -- maps to its bucket and to ``context - min(context, receptive)``.
    """
    pairs, windows, emitted = set(), set(), 0
    for window in (*ramp, chunk):
        context = min(emitted, left_context)
        windows.add(window + context)
        pairs.update((new + context, context) for new in range(1, window + 1))
        emitted += window
    largest = chunk + left_context
    buckets = sorted(windows | {b for b in (8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256) if b < largest})
    trims: dict[int, set[int]] = {}
    for frames, context in pairs:
        bucket = buckets[bisect.bisect_left(buckets, frames)]
        trims.setdefault(bucket, set()).add(context - min(context, receptive_frames))
    return buckets, trims


__all__ = [
    "DACStreamState",
    "ChannelsLastDACDecoder",
    "DACDecodeGraphs",
    "streaming_decode_shapes",
]
