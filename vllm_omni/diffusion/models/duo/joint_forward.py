# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer-wise clean/next-noisy joint compute with isolated KV staging.

The caller owns page reservations, acknowledgement and publication. This module
never publishes a cache. Refiner domains and time projection geometry stay
separate; only transformer projections/MLP and output projections are batched.
"""

import torch

from .contracts import dense_mask
from .kernels import gated_residual, modulate
from .transformer import embed_sequence, rms, rotary, split_qkv


def prepare_hidden(model, s):
    hidden = embed_sequence(
        model, s, mask_fn=lambda seq: dense_mask(seq.refiner_groups, seq.refiner_groups, seq.device, refiner=True)
    )
    frequencies = model.rope(s.positions[None])
    return hidden, frequencies


def joint_forward(
    model,
    clean,
    noisy,
    pair,
    clean_conditioning,
    noisy_conditioning,
    joint_conditioning=None,
    separate_projections=False,
):
    """Compute clean b and first noisy b+1; pair enforces the layer dependency."""
    a, fa = prepare_hidden(model, clean)
    b, fb = prepare_hidden(model, noisy)
    hidden = torch.cat((a, b))
    frequencies = torch.cat((fa, fb))
    rotary_coefficients = (frequencies.cos().to(hidden.dtype)[:, None], frequencies.sin().to(hidden.dtype)[:, None])
    # clean conditioning has exactly one time and three modality rows.
    indices = torch.cat((clean.time_ids * 3 + clean.tags, noisy.time_ids * 3 + noisy.tags + 3))
    for layer, block in enumerate(model.blocks):
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
            joint_conditioning[layer]
            if joint_conditioning is not None
            else tuple(
                torch.cat((ca, cb)) for ca, cb in zip(clean_conditioning[layer], noisy_conditioning[layer], strict=True)
            )
        )
        norm = modulate(rms(block.norm1, hidden), scale, shift, indices)
        packed = project(block.attn.qkv_proj, norm, clean.length, separate_projections)
        q, k, v = split_qkv(block.attn, packed)
        q = rotary(q, frequencies, rotary_coefficients)
        k = rotary(k, frequencies, rotary_coefficients)
        out = pair.attend(q, k, v, layer, block.attn.softmax_scale)
        out = project(block.attn.out_proj, out.flatten(1).to(hidden.dtype), clean.length, separate_projections)
        hidden = gated_residual(hidden, gate, out, indices)
        norm = modulate(rms(block.norm2, hidden), ff_scale, ff_shift, indices)
        hidden = gated_residual(hidden, ff_gate, project(block.mlp, norm, clean.length, separate_projections), indices)
    shift, scale = (
        joint_conditioning[-1]
        if joint_conditioning is not None
        else tuple(torch.cat((ca, cb)) for ca, cb in zip(clean_conditioning[-1], noisy_conditioning[-1], strict=True))
    )
    hidden = modulate(
        rms(model.final_layer.norm, hidden), scale, shift, torch.cat((clean.time_ids, noisy.time_ids + 1))
    )
    vi = torch.cat((clean.video_indices, noisy.video_indices + clean.length))
    ai = torch.cat((clean.audio_indices, noisy.audio_indices + clean.length))
    video = project(
        model.final_layer.video_out,
        hidden[vi].to(model.final_layer.video_out.weight.dtype),
        len(clean.video_indices),
        separate_projections,
    )
    audio = project(
        model.final_layer.audio_out,
        hidden[ai].to(model.final_layer.audio_out.weight.dtype),
        len(clean.audio_indices),
        separate_projections,
    )
    return (
        (video[: len(clean.video_indices)][None], audio[: len(clean.audio_indices)][None]),
        (video[len(clean.video_indices) :][None], audio[len(clean.audio_indices) :][None]),
    )


def project(module, value, cut, split):
    def apply(x):
        y = module(x)
        return y[0] if isinstance(y, tuple) else y

    return torch.cat((apply(value[:cut]), apply(value[cut:]))) if split else apply(value)
