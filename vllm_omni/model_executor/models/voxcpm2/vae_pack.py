# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pack a same-shape VAE cohort directly from persistent request slots."""

# ruff: noqa: N803

from vllm.triton_utils import tl, triton


@triton.jit
def write_vae_outputs(
    decoded,
    inputs,
    audio,
    pads,
    slots,
    pad_lengths,
    starts,
    DECODED_STRIDE: tl.constexpr,
    SAMPLE_STRIDE: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    AUDIO_STRIDE: tl.constexpr,
    PAD_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    FRAMES: tl.constexpr,
    DCS: tl.constexpr,
    TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    pad_frames = tl.load(pad_lengths + row)
    start = tl.load(starts + row)
    length = (FRAMES - pad_frames) * DCS
    pcm = tl.load(
        decoded + row * DECODED_STRIDE + (pad_frames * DCS + offsets) * SAMPLE_STRIDE,
        mask=offsets < length,
        other=0,
    )
    tl.store(audio + slot * AUDIO_STRIDE + start + offsets, pcm, mask=offsets < length)
    # Save the final latent frames in the slot's frame-major context layout.
    frame = offsets // DIM
    channel = offsets % DIM
    context = tl.load(
        inputs + row * INPUT_STRIDE + channel * CHANNEL_STRIDE + FRAMES - TAIL + frame,
        mask=offsets < TAIL * DIM,
        other=0,
    )
    tl.store(pads + slot * PAD_STRIDE + offsets, context, mask=offsets < TAIL * DIM)


@triton.jit
def append_pending_patches(
    prefix,
    pending,
    slots,
    counts,
    PREFIX_STRIDE: tl.constexpr,
    PENDING_STRIDE: tl.constexpr,
    PATCH_STRIDE: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    count = tl.load(counts + row)
    values = tl.load(prefix + slot * PREFIX_STRIDE + offsets, mask=offsets < ELEMENTS, other=0)
    tl.store(pending + slot * PENDING_STRIDE + count * PATCH_STRIDE + offsets, values, mask=offsets < ELEMENTS)


@triton.jit
def pack_vae_inputs(  # noqa: N803
    pads,
    pending,
    prefix,
    slots,
    pad_lengths,
    output,
    PAD_STRIDE: tl.constexpr,
    PENDING_STRIDE: tl.constexpr,
    PREFIX_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    OUT_CHANNEL_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    FRAMES: tl.constexpr,
    USE_PENDING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    frames = FRAMES
    pad_frames = tl.load(pad_lengths + row)
    channel = offsets // frames
    frame = offsets % frames
    mask = offsets < DIM * frames
    slot = tl.load(slots + row)
    pad = tl.load(pads + slot * PAD_STRIDE + frame * DIM + channel, mask=mask & (frame < pad_frames), other=0)
    if USE_PENDING:
        source = pending + slot * PENDING_STRIDE
    else:
        source = prefix + slot * PREFIX_STRIDE
    new = tl.load(source + (frame - pad_frames) * DIM + channel, mask=mask & (frame >= pad_frames), other=0)
    tl.store(
        output + row * OUT_STRIDE + channel * OUT_CHANNEL_STRIDE + frame,
        tl.where(frame < pad_frames, pad, new),
        mask=mask,
    )
