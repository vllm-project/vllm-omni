# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""S3Gen's flow-matching solvers: guided Euler for the standard checkpoint, plain Euler for meanflow."""

import torch

# Upstream's classifier-free guidance strength (``CFM_PARAMS``), used by the
# standard checkpoint; the distilled meanflow checkpoint needs none.
INFERENCE_CFG_RATE = 0.7


def cast_all(*args, dtype):
    """Cast every floating-point tensor to ``dtype``; leave the others as they are."""
    return [a if (not a.dtype.is_floating_point) or a.dtype == dtype else a.to(dtype) for a in args]


class CausalConditionalCFM(torch.nn.Module):
    """S3Gen's flow-matching solver around the U-Net estimator.

    Args:
        estimator (torch.nn.Module): The velocity estimator, ``ConditionalDecoder``.
    """

    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, noised_mels, spks=None, cond=None, meanflow=False):
        """Integrate the flow from noise to a mel.

        Args:
            mu (torch.Tensor): The encoder's output, shape (batch_size, n_feats, mel_timesteps).
            mask (torch.Tensor): Frame mask, shape (batch_size, 1, mel_timesteps).
            n_timesteps (int): Euler steps.
            noised_mels (torch.Tensor): The noise every frame starts from, shape (batch_size, n_feats, mel_timesteps).
            spks (torch.Tensor, optional): Speaker embedding, shape (batch_size, spk_emb_dim).
            cond (torch.Tensor, optional): Conditioning mel, the reference prompt's frames.
            meanflow (bool, optional): Whether the estimator is a distilled meanflow model, which
                is stepped without guidance. Defaults to False.

        Returns:
            output (tuple[torch.Tensor, None]): The generated mel-spectrogram, shape
                (batch_size, n_feats, mel_timesteps), and None.
        """
        z = noised_mels.to(mu.dtype)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if meanflow:
            return self.basic_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

        # The standard checkpoint steps on a cosine schedule.
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Euler steps with classifier-free guidance, the conditional and unconditional rows in one batch.

        Args:
            x (torch.Tensor): The starting noise, shape (B, 80, T).
            t_span (torch.Tensor): The step times, shape (n_timesteps + 1,).
            mu (torch.Tensor): The encoder's output, shape (B, 80, T).
            mask (torch.Tensor): Frame mask, shape (B, 1, T).
            spks (torch.Tensor): Speaker embedding, shape (B, 80).
            cond (torch.Tensor): Conditioning mel, shape (B, 80, T).

        Returns:
            torch.Tensor: The mel, shape (B, 80, T), in ``x``'s dtype.
        """
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype)

        # Rows [:B] are conditional; rows [B:] keep zero mu, speaker and condition.
        B, T = mu.size(0), x.size(2)
        x_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2 * B, 1, T], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2 * B], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2 * B, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)

        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(dim=0)
            r = r.unsqueeze(dim=0)
            x_in[:B] = x_in[B:] = x
            mask_in[:B] = mask_in[B:] = mask
            mu_in[:B] = mu
            t_in[:B] = t_in[B:] = t
            spks_in[:B] = spks
            cond_in[:B] = cond
            dxdt = self.estimator.forward(
                x=x_in,
                mask=mask_in,
                mu=mu_in,
                t=t_in,
                spks=spks_in,
                cond=cond_in,
            )
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = (1.0 + INFERENCE_CFG_RATE) * dxdt - INFERENCE_CFG_RATE * cfg_dxdt
            dt = r - t
            x = x + dt * dxdt

        return x.to(in_dtype)

    def basic_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Plain Euler steps, each also giving the estimator its end time, as meanflow needs.

        Args:
            x (torch.Tensor): The starting noise, shape (B, 80, T).
            t_span (torch.Tensor): The step times, shape (n_timesteps + 1,).
            mu (torch.Tensor): The encoder's output, shape (B, 80, T).
            mask (torch.Tensor): Frame mask, shape (B, 1, T).
            spks (torch.Tensor): Speaker embedding, shape (B, 80).
            cond (torch.Tensor): Conditioning mel, shape (B, 80, T).

        Returns:
            torch.Tensor: The mel, shape (B, 80, T), in ``x``'s dtype.
        """
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype)

        for t, r in zip(t_span[..., :-1], t_span[..., 1:]):
            t, r = t[None], r[None]
            dxdt = self.estimator.forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
            dt = r - t
            x = x + dt * dxdt

        return x.to(in_dtype)
