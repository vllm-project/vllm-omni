# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Make diffusers Wan VAE feat_cache compile-friendly.

diffusers ``WanResample`` stores the string sentinel ``"Rep"`` in feat_cache slots.
``tensor == "Rep"`` returns a non-Tensor and breaks ``torch.compile(fullgraph=True)``.
We replace it with a scalar int8 tensor marker (value 0).
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _is_feat_rep(entry: Any) -> bool:
    if not isinstance(entry, torch.Tensor):
        return False
    return entry.numel() == 1 and entry.dtype == torch.int8


def _make_feat_rep(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, device=x.device, dtype=torch.int8)


def _wan_resample_forward_tensorized(self, x, feat_cache=None, feat_idx=[0]):
    """``WanResample.forward`` with tensor feat_cache sentinel (no ``"Rep"`` string)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T

    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = _make_feat_rep(x)
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and not _is_feat_rep(feat_cache[idx]):
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and _is_feat_rep(feat_cache[idx]):
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                if _is_feat_rep(feat_cache[idx]):
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.resample(x)
    x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    if self.mode == "downsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x


def apply_wan_vae_feat_cache_tensor_patch() -> None:
    """Monkeypatch diffusers WanResample once per process."""
    global _PATCHED
    if _PATCHED:
        return

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample

    WanResample.forward = _wan_resample_forward_tensorized
    _PATCHED = True
    logger.info("DreamZero: applied Wan VAE feat_cache tensor sentinel patch.")
