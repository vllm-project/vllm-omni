# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Wan causal-convolution spatial padding with shape-specific fallback.

Inspired by the Wan VAE optimization in PR #8694. Temporal/cache assembly
retains diffusers' semantics; no normalization or precision changes.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def spatial_padding_candidate(conv, x, cache_x=None):
    p = list(conv._padding)
    if p[0] != p[1] or p[2] != p[3] or p[5] != 0:
        raise ValueError("Wan spatial padding requires symmetric spatial and causal temporal padding")
    if cache_x is not None and p[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat((cache_x, x), dim=2)
        p[4] -= cache_x.shape[2]
    if p[4]:
        x = F.pad(x, (0, 0, 0, 0, p[4], 0))
    return F.conv3d(x, conv.weight, conv.bias, conv.stride, (0, p[2], p[0]), conv.dilation, conv.groups)


def verified_spatial_forward(conv, x, cache_x=None):
    key = (
        tuple(x.shape),
        x.dtype,
        tuple(x.stride()),
        None if cache_x is None else (tuple(cache_x.shape), cache_x.dtype),
    )
    verdicts = conv._lingbot_padding_verdicts
    if verdicts.get(key) is False:
        return conv._lingbot_original_forward(x, cache_x)
    try:
        candidate = spatial_padding_candidate(conv, x, cache_x)
    except (RuntimeError, ValueError):
        verdicts[key] = False
        return conv._lingbot_original_forward(x, cache_x)
    if key not in verdicts:
        reference = conv._lingbot_original_forward(x, cache_x)
        verdict = torch.equal(candidate, reference)
        if len(verdicts) >= 128:
            verdicts.pop(next(iter(verdicts)))
        verdicts[key] = verdict
        return candidate if verdict else reference
    return candidate


def _forward(self, x, cache_x=None):
    if x.device.type not in ("npu", "privateuseone") or torch.is_grad_enabled():
        return self._lingbot_original_forward(x, cache_x)
    return verified_spatial_forward(self, x, cache_x)


def install_spatial_padding(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    if getattr(vae, "_vllm_omni_wan_spatial_shard_installed", False):
        raise ValueError("LingBot NPU spatial padding currently requires unsharded VAE")
    count = 0
    for conv in vae.decoder.modules():
        if type(conv) is WanCausalConv3d and not hasattr(conv, "_lingbot_original_forward"):
            conv._lingbot_original_forward = conv.forward
            conv._lingbot_padding_verdicts = {}
            conv.forward = MethodType(_forward, conv)
            count += 1
    return count
