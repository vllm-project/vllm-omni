# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Post-FFT ISTFT fusion for Ming's four-way overlapping audio frames.

Each lane owns one output sample and gathers its contributing frames. Audio
and window tails remain unnormalized so the next streaming chunk can add them.
The low-precision window rounding follows PyTorch's square, fold, and add
boundaries; the inverse FFT and the envelope-underflow check remain unchanged.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _istft_kernel(
        frames,
        window,
        previous_audio,
        previous_window,
        output,
        envelope_output,
        next_audio,
        next_window,
        frame_stride_b,
        frame_stride_w,
        frame_stride_t,
        audio_stride_b,
        audio_stride_t,
        window_stride_t,
        num_frames,
        raw_size,
        output_start,
        output_size,
        hop_size: tl.constexpr,
        streaming: tl.constexpr,
        has_buffer: tl.constexpr,
        block_size: tl.constexpr,
    ):
        batch = tl.program_id(0)
        sample = tl.program_id(1) * block_size + tl.arange(0, block_size)
        valid = sample < raw_size
        last_frame = sample // hop_size
        audio = tl.full((block_size,), 0, tl.float32)
        envelope = tl.full((block_size,), 0, tl.float32)
        window_dtype = window.dtype.element_ty

        # Match CUDA col2im's ascending frame order. Disable FMA at launch to
        # preserve the separately rounded window multiplication.
        for overlap in tl.static_range(3, -1, -1):
            frame = last_frame - overlap
            window_index = sample % hop_size + overlap * hop_size
            mask = valid & (frame >= 0) & (frame < num_frames)
            weight = tl.load(window + window_index, mask=mask, other=0).to(tl.float32)
            value = tl.load(
                frames + batch * frame_stride_b + window_index * frame_stride_w + frame * frame_stride_t,
                mask=mask,
                other=0,
            )
            audio += value * weight
            envelope += (weight * weight).to(window_dtype).to(tl.float32)

        # F.fold accumulates in FP32, then stores in the window's dtype.
        envelope = envelope.to(window_dtype).to(tl.float32)
        tail_size: tl.constexpr = 3 * hop_size
        if has_buffer:
            carry_mask = valid & (sample < tail_size)
            carry_audio = tl.load(
                previous_audio + batch * audio_stride_b + sample * audio_stride_t, mask=carry_mask, other=0
            )
            carry_window = tl.load(previous_window + sample * window_stride_t, mask=carry_mask, other=0)
            audio += carry_audio
            envelope = (envelope + carry_window.to(tl.float32)).to(window_dtype).to(tl.float32)

        output_index = sample - output_start
        emit = valid & (output_index >= 0) & (output_index < output_size)
        tl.store(output + batch * output_size + output_index, audio / envelope, mask=emit)
        if batch == 0:
            tl.store(envelope_output + output_index, envelope, mask=emit)

        if streaming:
            tail_index = sample - (raw_size - tail_size)
            tail_mask = valid & (tail_index >= 0)
            tl.store(next_audio + batch * tail_size + tail_index, audio, mask=tail_mask)
            if batch == 0:
                tl.store(next_window + tail_index, envelope, mask=tail_mask)


def fused_istft_supported(
    spec: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop: int,
    audio_buffer: torch.Tensor | None,
    window_buffer: torch.Tensor | None,
    streaming: bool,
) -> bool:
    if not (
        HAS_TRITON
        and spec.is_cuda
        and torch.version.hip is None
        and spec.dtype == torch.complex64
        and spec.ndim == 3
        and spec.shape[0] > 0
        and spec.shape[1] == n_fft // 2 + 1
        and spec.shape[2] >= 2
        and hop > 0
        and n_fft == 4 * hop
        and window.shape == (n_fft,)
        and window.is_contiguous()
        and window.device == spec.device
        and window.dtype in (torch.float32, torch.bfloat16, torch.float16)
    ):
        return False
    if not streaming or (audio_buffer is None and window_buffer is None):
        return True
    return (
        audio_buffer is not None
        and window_buffer is not None
        and audio_buffer.shape == (spec.shape[0], 3 * hop)
        and window_buffer.shape == (1, 3 * hop)
        and audio_buffer.device == window_buffer.device == spec.device
        and audio_buffer.dtype == torch.float32
        and window_buffer.dtype == window.dtype
    )


def fused_istft(
    frames: torch.Tensor,
    window: torch.Tensor,
    hop: int,
    audio_buffer: torch.Tensor | None,
    window_buffer: torch.Tensor | None,
    streaming: bool,
    last_chunk: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fuse supported post-FFT inputs; dispatch is checked by ``ISTFT``."""
    batch, win_length, num_frames = frames.shape
    tail_size = win_length - hop
    pad = tail_size // 2
    raw_size = (num_frames - 1) * hop + win_length
    has_buffer = streaming and audio_buffer is not None
    start = 0 if has_buffer else pad
    end = raw_size - (tail_size if streaming and not last_chunk else pad)
    output_size = end - start
    output = frames.new_empty((batch, output_size))
    envelope = window.new_empty((output_size,))
    next_audio = frames.new_empty((batch, tail_size)) if streaming else audio_buffer
    next_window = window.new_empty((1, tail_size)) if streaming else window_buffer

    _istft_kernel[(batch, triton.cdiv(raw_size, 256))](
        frames,
        window,
        audio_buffer,
        window_buffer,
        output,
        envelope,
        next_audio,
        next_window,
        *frames.stride(),
        audio_buffer.stride(0) if streaming and audio_buffer is not None else 0,
        audio_buffer.stride(1) if streaming and audio_buffer is not None else 0,
        window_buffer.stride(1) if streaming and window_buffer is not None else 0,
        num_frames,
        raw_size,
        start,
        output_size,
        hop_size=hop,
        streaming=streaming,
        has_buffer=has_buffer,
        block_size=256,
        enable_fp_fusion=False,
    )
    if not (envelope > 1e-11).all():
        raise RuntimeError("ISTFT window envelope underflowed; invalid overlap-add state.")
    return output, next_audio, next_window
