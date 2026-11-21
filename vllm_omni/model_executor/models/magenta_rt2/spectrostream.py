# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""48 kHz stereo causal streaming SpectroStream codec decoder for Magenta RealTime 2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.transformers_utils.configs.magenta_rt2 import MagentaRT2Config


def _unflatten_nested(flat_weights: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for flat_key, value in flat_weights.items():
        node = root
        parts = flat_key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return root


def _effective_kernel(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation + 1


def _hann_window(frame_length: int) -> torch.Tensor:
    """Periodic Hann window matching ``sequence_layers.mlx.signal.hann_window``."""
    return torch.hann_window(frame_length, periodic=True, dtype=torch.float32)


def _inverse_stft_window(frame_step: int, frame_length: int) -> torch.Tensor:
    """Griffin-Lim / SequenceLayers dual synthesis window for overlap-add ISTFT."""
    win = _hann_window(frame_length)
    denom = torch.zeros(frame_length, dtype=torch.float32)
    win_sq = win.square()
    # Compute frame_step-periodic sum of squared windows (exact COLA denominator).
    num_shifts = math.ceil(frame_length / frame_step)
    phase_sum = torch.zeros(frame_step, dtype=torch.float32)
    for k in range(num_shifts):
        start = k * frame_step
        end = min(start + frame_length, (k + 1) * frame_step)
        chunk = win_sq[start:end]
        phase_sum[: chunk.shape[0]] += chunk
    # For 960 / 480 periodic Hann window, overlap-add of two shifted squares is 1.0 everywhere.
    reps = math.ceil(frame_length / frame_step)
    denom = phase_sum.repeat(reps)[:frame_length]
    return win / denom.clamp(min=1e-12)


def _load_conv_weights(module: CausalConv2d | CausalConvTranspose2d, jax_params: dict[str, Any]) -> None:
    if "conv" in jax_params:
        jax_params = jax_params["conv"]
    kernel = torch.as_tensor(jax_params["kernel"])
    if isinstance(module, CausalConvTranspose2d):
        # JAX conv_transpose implicitly flips the kernel spatially; PyTorch does not.
        kernel = kernel.flip(0, 1)
        # JAX [kH, kW, Cin, Cout] -> PyTorch ConvTranspose2d [Cin, Cout, kH, kW]
        weight = kernel.permute(2, 3, 0, 1).contiguous()
    else:
        # JAX [kH, kW, Cin, Cout] -> PyTorch Conv2d [Cout, Cin, kH, kW]
        weight = kernel.permute(3, 2, 0, 1).contiguous()
    module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    if module.bias is not None and "bias" in jax_params:
        module.bias.data.copy_(
            torch.as_tensor(jax_params["bias"], dtype=module.bias.dtype).to(device=module.bias.device)
        )


class CausalConv2d(nn.Module):
    """Semicausal / causal time + symmetric spatial 2D convolution supporting batch and step modes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.strides = (int(strides[0]), int(strides[1]))
        self.dilation = (int(dilation[0]), int(dilation[1]))

        ek_t = _effective_kernel(self.kernel_size[0], self.dilation[0])
        self.time_pad_left = max(ek_t - self.strides[0], 0)
        self.time_pad_right = ek_t - 1 - self.time_pad_left
        self.buf_width = self.time_pad_left

        ek_f = _effective_kernel(self.kernel_size[1], self.dilation[1])
        pad_freq = max(ek_f - self.strides[1], 0)
        self.freq_pad_left = pad_freq // 2
        self.freq_pad_right = pad_freq - self.freq_pad_left

        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def init_state(
        self,
        batch_size: int,
        freq_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.buf_width == 0:
            return None
        return torch.zeros((batch_size, self.in_channels, self.buf_width, freq_dim), dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full sequence forward on ``[B, C, T, F]``."""
        x_pad = F.pad(x, (self.freq_pad_left, self.freq_pad_right, self.time_pad_left, self.time_pad_right))
        return F.conv2d(
            x_pad,
            self.weight,
            self.bias,
            stride=self.strides,
            dilation=self.dilation,
        )

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Streaming causal step on ``[B, C, T, F]`` updating fixed-size ``state`` in place."""
        if self.buf_width > 0:
            assert state is not None
            seq = torch.cat([state, x], dim=2)
            if self.time_pad_right > 0:
                seq_for_conv = F.pad(seq, (self.freq_pad_left, self.freq_pad_right, 0, self.time_pad_right))
            else:
                seq_for_conv = F.pad(seq, (self.freq_pad_left, self.freq_pad_right, 0, 0))
            new_state = seq[:, :, -self.buf_width :, :]
            if active_mask is not None:
                state.copy_(torch.where(active_mask.view(-1, 1, 1, 1), new_state, state))
            else:
                state.copy_(new_state)
        else:
            seq_for_conv = F.pad(x, (self.freq_pad_left, self.freq_pad_right, 0, self.time_pad_right))
        out = F.conv2d(
            seq_for_conv,
            self.weight,
            self.bias,
            stride=self.strides,
            dilation=self.dilation,
        )
        return out, state


class CausalConvTranspose2d(nn.Module):
    """Causal time + same spatial 2D transposed convolution with overlap-add step state."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.strides = (int(strides[0]), int(strides[1]))
        self.dilation = (int(dilation[0]), int(dilation[1]))

        ek_t = _effective_kernel(self.kernel_size[0], self.dilation[0])
        self.ola_buf = max(0, ek_t - self.strides[0])
        self.time_trim_right = self.ola_buf

        ek_f = _effective_kernel(self.kernel_size[1], self.dilation[1])
        sp_trim = max(0, ek_f - self.strides[1])
        self.freq_trim_left = sp_trim // 2
        self.freq_trim_right = sp_trim - self.freq_trim_left

        self.weight = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def init_state(
        self,
        batch_size: int,
        out_freq: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.ola_buf == 0:
            return None
        return torch.zeros((batch_size, self.out_channels, self.ola_buf, out_freq), dtype=dtype, device=device)

    def _conv_raw(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose2d(
            x,
            self.weight,
            bias=None,
            stride=self.strides,
            padding=0,
            dilation=self.dilation,
        )
        if self.freq_trim_left > 0 or self.freq_trim_right > 0:
            end_f = y.shape[3] - self.freq_trim_right if self.freq_trim_right > 0 else y.shape[3]
            y = y[:, :, :, self.freq_trim_left : end_f]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._conv_raw(x)
        if self.time_trim_right > 0:
            y = y[:, :, : -self.time_trim_right, :]
        return y + self.bias.view(1, -1, 1, 1)

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw = self._conv_raw(x)
        out_time = x.shape[2] * self.strides[0]
        if self.ola_buf > 0:
            assert state is not None
            head = raw[:, :, : self.ola_buf, :] + state
            if raw.shape[2] > self.ola_buf:
                merged = torch.cat([head, raw[:, :, self.ola_buf :, :]], dim=2)
            else:
                merged = head
            out = merged[:, :, :out_time, :]
            tail = merged[:, :, out_time : out_time + self.ola_buf, :]
            if tail.shape[2] < self.ola_buf:
                tail = F.pad(tail, (0, 0, 0, self.ola_buf - tail.shape[2]))
            if active_mask is not None:
                state.copy_(torch.where(active_mask.view(-1, 1, 1, 1), tail, state))
            else:
                state.copy_(tail)
        else:
            out = raw[:, :, :out_time, :]
        out = out + self.bias.view(1, -1, 1, 1)
        return out, state


class SpectroStreamInputLayer(nn.Module):
    """Input projection residual block: 1x1 conv with 2-layer ELU shortcut -> [B, C=512, T, F=5]."""

    def __init__(self, in_dim: int = 256, input_bins: int = 5, channels: int = 512) -> None:
        super().__init__()
        self.input_bins = int(input_bins)
        self.channels = int(channels)
        proj_dim = self.input_bins * self.channels
        self.conv1x1_first = CausalConv2d(in_dim, proj_dim, (1, 1))
        self.conv1x1_b1 = CausalConv2d(in_dim, proj_dim, (1, 1))
        self.conv1x1_b2 = CausalConv2d(proj_dim, proj_dim, (1, 1))

    def forward_common(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``[B, in_dim, T]``."""
        x2d = x.unsqueeze(-1)  # [B, 256, T, 1]
        bsz, _, t, _ = x2d.shape
        body = self.conv1x1_first(x2d)
        sc = self.conv1x1_b2(F.elu(self.conv1x1_b1(x2d)))
        out = body + sc
        # Reshape [B, F * C, T, 1] matching SequenceLayers Reshape([5, 512]) from flat [2560]:
        # In JAX/MLX NHWC `[B, T, 1, F*C]` -> `[B, T, F, C]` -> transpose to `[B, C, T, F]`.
        return (
            out.squeeze(-1)
            .permute(0, 2, 1)
            .reshape(bsz, t, self.input_bins, self.channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class SpectroStreamResidualBlock(nn.Module):
    """SpectroStream pre-activated ELU Conv2D / Conv2DTranspose residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.strides = (int(strides[0]), int(strides[1]))
        resample_k = (max(3, 2 * self.strides[0]), max(3, 2 * self.strides[1]))

        self.has_upsample = self.strides != (1, 1)
        if self.has_upsample:
            self.resample = CausalConvTranspose2d(in_channels, out_channels, resample_k, self.strides)
        else:
            self.resample = CausalConv2d(in_channels, out_channels, (3, 3), strides=(1, 1), dilation=(1, 1))
        self.conv3x3 = CausalConv2d(out_channels, out_channels, (3, 3), strides=(1, 1), dilation=dilation)
        self.shortcut_conv = CausalConv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None

    def init_state(
        self,
        batch_size: int,
        out_freq: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        in_freq = out_freq // self.strides[1]
        r_state = (
            self.resample.init_state(batch_size, out_freq, device=device, dtype=dtype)
            if self.has_upsample
            else self.resample.init_state(batch_size, in_freq, device=device, dtype=dtype)
        )
        c_state = self.conv3x3.init_state(batch_size, out_freq, device=device, dtype=dtype)
        return (r_state, c_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.resample(F.elu(x))
        h = self.conv3x3(F.elu(h))
        sc = self.shortcut_conv(x) if self.shortcut_conv is not None else x
        if self.has_upsample:
            sc = sc.repeat_interleave(self.strides[0], dim=2).repeat_interleave(self.strides[1], dim=3)
        return sc + h

    def step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None],
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        r_state, c_state = state
        h, r_state = self.resample.step(F.elu(x), r_state, active_mask=active_mask)
        h, c_state = self.conv3x3.step(F.elu(h), c_state, active_mask=active_mask)
        sc = self.shortcut_conv(x) if self.shortcut_conv is not None else x
        if self.has_upsample:
            sc = sc.repeat_interleave(self.strides[0], dim=2).repeat_interleave(self.strides[1], dim=3)
        return sc + h, (r_state, c_state)


@dataclass
class SpectroStreamRowState:
    """Allocated fixed-address GPU state buffers for a batch of streaming SpectroStream slots."""

    input_res_state: tuple[torch.Tensor | None, torch.Tensor | None]
    dec0_state: tuple[torch.Tensor | None, torch.Tensor | None]
    group_states: list[tuple[torch.Tensor | None, torch.Tensor | None]] = field(default_factory=list)
    out_conv_state: torch.Tensor | None = None
    lookahead_countdown: torch.Tensor | None = None  # [B] int32 (initialized to 5)
    ola_buf: torch.Tensor | None = None  # [B, 480, 2]

    def reset_rows(self, row_indices: torch.Tensor | list[int] | int) -> None:
        if isinstance(row_indices, int):
            batch_idx: Any = [row_indices]
            group_idx: Any = [2 * row_indices, 2 * row_indices + 1]
        elif isinstance(row_indices, torch.Tensor):
            if row_indices.numel() == 0:
                return
            batch_idx = row_indices
            group_idx = torch.stack([row_indices * 2, row_indices * 2 + 1], dim=-1).reshape(-1)
        else:
            rows = list(row_indices)
            if not rows:
                return
            batch_idx = rows
            group_idx = [c for r in rows for c in (2 * r, 2 * r + 1)]

        # Use __setitem__ (= 0) rather than t[list_idx].zero_(), since list/tensor
        # fancy indexing creates a copy whose .zero_() would not mutate state in place.
        for pair in (self.input_res_state, self.dec0_state):
            for t in pair:
                if t is not None:
                    t[batch_idx] = 0
        for pair in self.group_states:
            for t in pair:
                if t is not None:
                    t[group_idx] = 0
        if self.out_conv_state is not None:
            self.out_conv_state[group_idx] = 0
        if self.lookahead_countdown is not None:
            self.lookahead_countdown[batch_idx] = 5
        if self.ola_buf is not None:
            self.ola_buf[batch_idx] = 0


class SpectroStreamModel(nn.Module):
    """48 kHz stereo causal SpectroStream quantizer + STFT decoder + InverseSTFT."""

    def __init__(self, config: MagentaRT2Config | None = None) -> None:
        super().__init__()
        self.config = config or MagentaRT2Config()
        self.num_quantizers = int(self.config.ss_num_quantizers)
        self.active_quantizers = int(self.config.ss_active_quantizers)
        self.codebook_size = int(self.config.ss_codebook_size)
        self.num_reserved_tokens = int(self.config.num_extra_tokens)
        self.embed_dim = int(self.config.ss_input_dim)
        self.sample_rate = int(self.config.sample_rate)
        self.stft_frame_length = 960
        self.stft_frame_step = 480
        self.stft_fft_length = 960
        self.lookahead_stft_steps = 4
        self.samples_per_frame = int(self.config.samples_per_frame)

        # RVQ table: [64, 1024, 256]
        self.quantizer_embedding = nn.Parameter(torch.zeros(self.num_quantizers, self.codebook_size, self.embed_dim))
        # Precomputed codebook offsets for fast single gather
        self.register_buffer(
            "codebook_flat_offsets",
            torch.arange(self.active_quantizers, dtype=torch.long) * self.codebook_size,
            persistent=False,
        )

        # Decoder topology matching stft_spectrostream_40ms_generic_48khz_stereo_config:
        # ratios = ((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1))
        # mults  = (2, 1, 2, 1, 1, 2, 1), channel_splits = 2, channel_recombo_block = -2
        self.input_layer = SpectroStreamInputLayer(in_dim=self.embed_dim, input_bins=5, channels=512)
        self.input_layers_residual_unit = SpectroStreamResidualBlock(512, 512, strides=(1, 1))
        self.decoder_0 = SpectroStreamResidualBlock(512, 1024, strides=(2, 1))

        # Split into 2 groups of 512 channels processed via batched weights (B * 2)
        self.group_blocks = nn.ModuleList(
            [
                SpectroStreamResidualBlock(512, 256, strides=(2, 2)),  # decoder_1: T 2->4, F 5->10
                SpectroStreamResidualBlock(256, 256, strides=(1, 2)),  # decoder_2: T 4, F 10->20
                SpectroStreamResidualBlock(256, 256, strides=(1, 2)),  # decoder_3: T 4, F 20->40
                SpectroStreamResidualBlock(256, 128, strides=(1, 3)),  # decoder_4: T 4, F 40->120
                SpectroStreamResidualBlock(128, 128, strides=(1, 2)),  # decoder_5: T 4, F 120->240
                SpectroStreamResidualBlock(128, 64, strides=(1, 2)),  # decoder_6: T 4, F 240->480
            ]
        )
        self._group_out_freqs = (10, 20, 40, 120, 240, 480)
        self.output_conv = CausalConv2d(64, 2, (7, 7), strides=(1, 1))
        self.register_buffer(
            "istft_window",
            _inverse_stft_window(self.stft_frame_step, self.stft_frame_length),
            persistent=False,
        )

    def codes_to_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Dequantize ``[..., Q]`` tokens (unique Stage-0 IDs >= 6 or raw 0..1023) -> ``[..., 256]``."""
        codes_l = codes.to(dtype=torch.long)
        # Convert Depthformer unique tokens (>= num_reserved_tokens) to 0..1023 per codebook.
        local = torch.where(
            codes_l >= self.num_reserved_tokens,
            (codes_l - self.num_reserved_tokens) % self.codebook_size,
            codes_l.clamp(0, self.codebook_size - 1),
        )
        flat_table = self.quantizer_embedding[: self.active_quantizers].reshape(
            self.active_quantizers * self.codebook_size, self.embed_dim
        )
        flat_idx = local + self.codebook_flat_offsets
        return F.embedding(flat_idx, flat_table).float().sum(dim=-2).to(dtype=self.quantizer_embedding.dtype)

    def _spec_to_istft_frames(self, spec_bcft: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, 4, T_stft, 480]`` real/imag stereo spec -> ``[B, T_stft, 960, 2]`` windowed time frames."""
        # spec_bcft is [B, 4, T_stft, 480] where channels 0,1,2,3 correspond to
        # [L_real, L_imag, R_real, R_imag] (because 2 ParallelChannels groups each output 2 channels).
        bsz, _, t_stft, bins = spec_bcft.shape
        x = spec_bcft.float().permute(0, 2, 3, 1)  # [B, T, 480, 4]
        x = F.pad(x, (0, 0, 0, 1))  # [B, T, 481, 4] (pad Nyquist bin with 0)
        complex_spec = torch.complex(x[..., 0::2], x[..., 1::2]).permute(0, 1, 3, 2)  # [B, T, 2, 481]
        frames = torch.fft.irfft(complex_spec, n=self.stft_fft_length, dim=-1)  # [B, T, 2, 960]
        frames = frames * self.istft_window.to(device=frames.device)
        return frames.permute(0, 1, 3, 2).contiguous()  # [B, T, 960, 2]

    @property
    def conv_dtype(self) -> torch.dtype:
        return self.decoder_0.resample.weight.dtype

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> SpectroStreamRowState:
        state_dtype = dtype or self.conv_dtype
        with torch.inference_mode(False):
            return SpectroStreamRowState(
                input_res_state=self.input_layers_residual_unit.init_state(
                    batch_size, 5, device=device, dtype=state_dtype
                ),
                dec0_state=self.decoder_0.init_state(batch_size, 5, device=device, dtype=state_dtype),
                group_states=[
                    blk.init_state(batch_size * 2, freq, device=device, dtype=state_dtype)
                    for blk, freq in zip(self.group_blocks, self._group_out_freqs, strict=True)
                ],
                out_conv_state=self.output_conv.init_state(batch_size * 2, 480, device=device, dtype=state_dtype),
                lookahead_countdown=torch.full(
                    (batch_size,), self.lookahead_stft_steps + 1, dtype=torch.int32, device=device
                ),
                ola_buf=torch.zeros((batch_size, self.stft_frame_step, 2), dtype=torch.float32, device=device),
            )

    @torch.inference_mode()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Batch (non-streaming) decode of ``[B, T, 12]`` tokens -> ``[B, T*1920, 2]`` stereo PCM."""
        bsz, num_frames, _ = codes.shape
        if num_frames == 0:
            return torch.zeros((bsz, 0, 2), dtype=torch.float32, device=codes.device)
        emb = self.codes_to_embeddings(codes).permute(0, 2, 1)  # [B, 256, T]
        h = self.input_layer.forward_common(emb)
        h = self.input_layers_residual_unit(h)
        h = self.decoder_0(h)  # [B, 1024, 2*T, 5]

        # Fold ParallelChannels(num_groups=2) into batch dimension: [B*2, 512, 2*T, 5]
        bsz_u, ch_u, t_u, f_u = h.shape
        h_g = h.view(bsz_u, 2, ch_u // 2, t_u, f_u).reshape(bsz_u * 2, ch_u // 2, t_u, f_u)
        for blk in self.group_blocks:
            h_g = blk(h_g)
        h_g = self.output_conv(F.elu(h_g))
        # Unfold back: [B, 4, 4*T, 480]
        _, ch_o, t_o, f_o = h_g.shape
        spec = h_g.view(bsz_u, 2 * ch_o, t_o, f_o)

        # Lookahead(4) in batch layer mode drops the first 4 STFT steps (and truncates ISTFT causally).
        if spec.shape[2] <= self.lookahead_stft_steps:
            return torch.zeros((bsz, 0, 2), dtype=torch.float32, device=codes.device)
        spec = spec[:, :, self.lookahead_stft_steps :, :]
        frames = self._spec_to_istft_frames(spec)  # [B, T_stft', 960, 2]

        # Batch overlap-and-add with causal trim = 480
        t_valid = frames.shape[1]
        out_samples = t_valid * self.stft_frame_step + (self.stft_frame_length - self.stft_frame_step)
        buf = torch.zeros((bsz, out_samples, 2), dtype=torch.float32, device=codes.device)
        for step_idx in range(t_valid):
            s = step_idx * self.stft_frame_step
            buf[:, s : s + self.stft_frame_length, :] += frames[:, step_idx]
        wav = buf[:, : t_valid * self.stft_frame_step, :]
        return (0.5 * wav).clamp(-1.0, 1.0)

    @torch.inference_mode()
    def step(
        self,
        codes: torch.Tensor,
        state: SpectroStreamRowState,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Streaming decode of ``[B, 1, 12]`` (or ``[B, T, 12]``) tokens updating ``state`` in place.

        Returns ``[B, T * 1920, 2]`` float32 stereo PCM (normalized to ``[-1, 1]``).
        First frame produces silent lookahead warmup samples matching JAX/MLX streaming mode.
        """
        bsz, num_frames, _ = codes.shape
        if num_frames == 0:
            return torch.zeros((bsz, 0, 2), dtype=torch.float32, device=codes.device)
        active_batch_mask = active_mask.to(device=codes.device, dtype=torch.bool) if active_mask is not None else None
        active_group_mask = active_batch_mask.repeat_interleave(2) if active_batch_mask is not None else None

        emb = self.codes_to_embeddings(codes).permute(0, 2, 1)  # [B, 256, T]
        h = self.input_layer.forward_common(emb)
        h, state.input_res_state = self.input_layers_residual_unit.step(
            h, state.input_res_state, active_mask=active_batch_mask
        )
        h, state.dec0_state = self.decoder_0.step(h, state.dec0_state, active_mask=active_batch_mask)

        bsz_u, ch_u, t_u, f_u = h.shape
        h_g = h.view(bsz_u, 2, ch_u // 2, t_u, f_u).reshape(bsz_u * 2, ch_u // 2, t_u, f_u)
        for idx, blk in enumerate(self.group_blocks):
            h_g, state.group_states[idx] = blk.step(h_g, state.group_states[idx], active_mask=active_group_mask)
        h_g, state.out_conv_state = self.output_conv.step(
            F.elu(h_g), state.out_conv_state, active_mask=active_group_mask
        )
        _, ch_o, t_o, f_o = h_g.shape
        spec = h_g.view(bsz_u, 2 * ch_o, t_o, f_o)  # [B, 4, T_stft, 480]

        # Streaming Lookahead(4): mask STFT timesteps while countdown > 0.
        assert state.lookahead_countdown is not None and state.ola_buf is not None
        # Each STFT step decrements countdown by 1; mask_valid is True once countdown hits 0.
        steps = torch.arange(1, t_o + 1, dtype=torch.int32, device=spec.device).unsqueeze(0)
        countdown = (state.lookahead_countdown.unsqueeze(1) - steps).clamp(min=0)
        valid_mask = (countdown == 0).to(dtype=spec.dtype)  # [B, T_stft]
        if active_batch_mask is not None:
            state.lookahead_countdown.copy_(torch.where(active_batch_mask, countdown[:, -1], state.lookahead_countdown))
        else:
            state.lookahead_countdown.copy_(countdown[:, -1])
        spec = spec * valid_mask[:, None, :, None]

        frames = self._spec_to_istft_frames(spec)  # [B, T_stft, 960, 2]

        # Streaming overlap-and-add per STFT step (T_stft = 4 per 40ms codec frame).
        out_chunks: list[torch.Tensor] = []
        ola = state.ola_buf  # [B, 480, 2]
        prev_ola = ola.clone() if active_batch_mask is not None else None
        step_size = self.stft_frame_step
        for idx in range(t_o):
            cur = frames[:, idx, :, :]  # [B, 960, 2]
            out_step = cur[:, :step_size, :] + ola
            ola.copy_(cur[:, step_size:, :])
            out_chunks.append(out_step)
        if prev_ola is not None and active_batch_mask is not None:
            ola.copy_(torch.where(active_batch_mask[:, None, None], ola, prev_ola))
        wav = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]
        return (0.5 * wav).clamp(-1.0, 1.0)

    def load_flax_params(self, jax_soundstream: dict[str, Any]) -> None:
        """Load ``params/soundstream`` sub-tree from a Magenta RT 2 Flax safetensors checkpoint."""
        q_emb = torch.as_tensor(jax_soundstream["quantizer"]["embedding"], dtype=self.quantizer_embedding.dtype)
        self.quantizer_embedding.data.copy_(q_emb.to(device=self.quantizer_embedding.device))
        jax_dec = jax_soundstream["decoder"]

        # input_layer
        inp = jax_dec["input_layer"]
        _load_conv_weights(self.input_layer.conv1x1_first, inp["conv1x1_first"])
        _load_conv_weights(self.input_layer.conv1x1_b1, inp["shortcut_layer"]["conv1x1_b1"])
        _load_conv_weights(self.input_layer.conv1x1_b2, inp["shortcut_layer"]["conv1x1_b2"])

        # input_layers_residual_unit
        ru = jax_dec["input_layers_residual_unit"]
        _load_conv_weights(self.input_layers_residual_unit.resample, ru["conv2d_3x3_a"])
        _load_conv_weights(self.input_layers_residual_unit.conv3x3, ru["conv2d_3x3"])

        def _load_dec_block(blk: SpectroStreamResidualBlock, jblk: dict[str, Any]) -> None:
            t_key = next(k for k in jblk if "conv2dtranspose" in k)
            c_key = next(k for k in jblk if k.startswith("conv2d_"))
            _load_conv_weights(blk.resample, jblk[t_key])
            _load_conv_weights(blk.conv3x3, jblk[c_key])
            if blk.shortcut_conv is not None and "shortcut_layer" in jblk:
                sc_key = next(k for k in jblk["shortcut_layer"] if "conv1x1" in k)
                _load_conv_weights(blk.shortcut_conv, jblk["shortcut_layer"][sc_key])

        _load_dec_block(self.decoder_0, jax_dec["decoder_0"])
        for idx, blk in enumerate(self.group_blocks):
            _load_dec_block(blk, jax_dec[f"decoder_{idx + 1}"])
        _load_conv_weights(self.output_conv, jax_dec["output_layer"]["base_conv_last"])


__all__ = [
    "CausalConv2d",
    "CausalConvTranspose2d",
    "SpectroStreamInputLayer",
    "SpectroStreamModel",
    "SpectroStreamResidualBlock",
    "SpectroStreamRowState",
    "_unflatten_nested",
]
