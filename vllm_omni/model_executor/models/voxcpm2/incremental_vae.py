# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-slot feature caches for the native causal AudioVAE decoder."""

from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def incremental_vae_context(vae, caches, slots, valid, num_slots):
    """Replace causal padding with cached inputs, computing only new outputs.

    Cache storage stays fixed across graph replays. Overrides are installed only
    while decoding/capturing; ordinary full-sequence decode remains available.
    """
    from voxcpm.modules.audiovae.audio_vae_v2 import CausalConv1d, CausalTransposeConv1d, NoiseBlock

    originals = []

    def cached_input(name, x, history):
        cache = caches.get(name)
        if cache is None:
            cache = x.new_zeros(num_slots, x.shape[1], history)
            caches[name] = cache
        old = cache.index_select(0, slots)
        joined = torch.cat([old, x], dim=-1)
        cache.index_copy_(0, slots, torch.where(valid[:, None, None], joined[..., -history:], old))
        return joined

    try:
        for name, module in vae.decoder.named_modules():
            if isinstance(module, NoiseBlock):
                raise ValueError("Incremental VoxCPM2 VAE does not support stochastic NoiseBlock")
            if isinstance(module, nn.ConvTranspose1d):
                if not isinstance(module, CausalTransposeConv1d):
                    raise ValueError("Incremental VoxCPM2 VAE requires causal transpose convolutions")
                if module.dilation[0] != 1 or module.padding[0] != 0 or module.output_padding[0] != 0:
                    raise ValueError("Incremental VoxCPM2 VAE requires unpadded, dilation-one transpose convolutions")
                trim = module.kernel_size[0] - module.stride[0]
                history = (module.kernel_size[0] - 1) // module.stride[0]
                if trim < 0:
                    raise ValueError("Incremental VoxCPM2 VAE requires transpose kernel size >= stride")
                if not history:
                    continue

                def forward(x, conv=module, cache_key=name, history_frames=history, right_trim=trim):
                    joined = cached_input(cache_key, x, history_frames)
                    out = nn.ConvTranspose1d.forward(conv, joined)
                    # Drop outputs for cached input positions and the causal right trim.
                    left = history_frames * conv.stride[0]
                    return out[..., left:-right_trim] if right_trim else out[..., left:]

            elif isinstance(module, nn.Conv1d):
                if not isinstance(module, CausalConv1d):
                    raise ValueError("Incremental VoxCPM2 VAE requires causal convolutions")
                if module.stride[0] != 1:
                    raise ValueError("Incremental VoxCPM2 VAE requires stride-one decoder convolutions")
                if module.padding[0] != 0:
                    raise ValueError("Incremental VoxCPM2 VAE requires externally padded causal convolutions")
                history = (module.kernel_size[0] - 1) * module.dilation[0]
                if not history:
                    continue

                def forward(x, conv=module, cache_key=name, history_frames=history):
                    return nn.Conv1d.forward(conv, cached_input(cache_key, x, history_frames))

            else:
                continue
            originals.append((module, module.forward))
            module.forward = forward
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original


def incremental_vae_decode(vae, patch, caches, slots, valid, num_slots, sr_cond):
    # Bypass an existing full-sequence compile wrapper: this path is captured by
    # the model-owned graph, including all per-layer feature-cache operations.
    decode = getattr(vae.decode, "_torchdynamo_orig_callable", vae.decode)
    with incremental_vae_context(vae, caches, slots, valid, num_slots):
        return decode(patch, sr_cond=sr_cond)
