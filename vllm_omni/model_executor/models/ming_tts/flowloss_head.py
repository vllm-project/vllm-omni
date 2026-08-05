# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/flowloss.py

import torch
import torch.nn as nn

from vllm_omni.model_executor.models.common.ming.fm import Solver, build_timesteps

from .fm.dit import DiT


class FlowLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super().__init__()
        self.z_channels = z_channels
        # Preserve checkpoint parameter names under flowloss.cfm.model.* while
        # keeping the inference-only CFM sampling logic local to FlowLoss.
        self.cfm = nn.Module()
        self.cfm.add_module("model", DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs))

    @torch.no_grad()
    def sample(self, z, latent_history, cfg=2.0, patch_size=1, sigma=0.25, temperature=0):
        # z: [Batch, Time=1, Dimension]
        # latent_history: [Batch, History_Time, Dimension]
        if z.ndim != 3:
            raise ValueError(f"Expected conditioning rank-3 [Batch, Time, Dimension], got {tuple(z.shape)}")
        if latent_history.ndim != 3:
            raise ValueError(
                f"Expected latent_history rank-3 [Batch, Time, Dimension], got {tuple(latent_history.shape)}"
            )
        if z.shape[0] != latent_history.shape[0]:
            raise ValueError(
                f"Batch mismatch across conditioning and latent_history: {z.shape[0]}, {latent_history.shape[0]}"
            )
        dtype = z.dtype
        noise = torch.randn(z.shape[0], self.z_channels, patch_size, device=z.device, dtype=torch.float32)

        def fn(t, x):
            x = x.to(dtype)
            t = t.to(dtype)
            if cfg < 1e-5:
                if t.ndim == 0:
                    t = t.repeat(x.shape[0])
                pred = self.cfm.model(x=x, t=t, c=torch.zeros_like(z), latent_history=latent_history)
                return pred[:, -patch_size:, :].float()

            pred_cfg = self.cfm.model.forward_with_cfg(
                x=x, t=t, c=z, latent_history=latent_history, cfg_scale=cfg, patch_size=patch_size
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return (pred + (pred - null_pred) * cfg).float()

        t = build_timesteps(
            10,
            device=noise.device,
            dtype=noise.dtype,
            use_epss=True,
            sway_sampling_coef=-1.0,
        )
        return Solver(fn, noise.transpose(1, 2), sigma=sigma, temperature=temperature).integrate(t)[-1].to(dtype)
