# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM-family action inference (``FastWAM.infer_action``, ``first_frame`` mode).

Per call: encode the current frame with the Wan2.2 VAE, run the video expert once
over it at t=0 and cache every layer's keys/values, then integrate the action
flow with Euler steps, each step one action-expert pass attending to the cached
video keys/values. CAE-FastWAM finally decodes the latent tokens with its action
codec. No classifier-free guidance; no future video is generated.

Numerics follow the reference eager path: the schedule is computed in float32 on
the model device and rounded to the model dtype, the Euler update runs in the
model dtype, the initial noise is drawn in float32 on a CPU generator and then
cast, and the action codec runs in float32 with its output rounded to the model
dtype before it is returned as float32.
"""

# Adapted from FastWAM (https://github.com/yuantianyuan01/FastWAM @7faa711,
# src/fastwam/models/wan22/fastwam.py, mot.py). The original portions carry the following MIT license:
# Copyright (c) 2026 The FastWAM Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.diffusion.models.fastwam.config import FastWAMConfig
from vllm_omni.diffusion.models.fastwam.modeling.action_codec import ActionLatentDecoder
from vllm_omni.diffusion.models.fastwam.modeling.umt5_encoder import WanUMT5Encoder
from vllm_omni.diffusion.models.fastwam.modeling.wan_dit import ActionExpert, FastWAMMoT, VideoExpert
from vllm_omni.diffusion.models.fastwam.modeling.wan_vae_encoder import WanVAEFirstFrameEncoder


def build_action_schedule(
    num_steps: int, shift: float, num_train_timesteps: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shifted flow-matching schedule: timesteps and Euler deltas, rounded to ``dtype``."""
    if num_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_steps}.")
    u = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    sigma = shift * u / (1.0 + (shift - 1.0) * u)
    timesteps = sigma[:-1] * float(num_train_timesteps)
    deltas = sigma[1:] - sigma[:-1]
    return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)


def build_mot_attention_masks(
    video_seq_len: int, action_seq_len: int, tokens_per_frame: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint first-frame-causal MoT mask, split into the video [Sv, Sv] and action [Sa, Sv+Sa] rows.

    With the single conditioning frame of action inference every entry is True;
    the masks are still passed to SDPA so it picks the same kernel as the reference.
    """
    total = video_seq_len + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    first = min(tokens_per_frame, video_seq_len)
    video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
    video_mask[:first, first:] = False
    mask[:video_seq_len, :video_seq_len] = video_mask
    mask[video_seq_len:, video_seq_len:] = True
    mask[video_seq_len:, :first] = True
    return mask[:video_seq_len, :video_seq_len], mask[video_seq_len:, :]


class FastWAMModel(nn.Module):
    """All neural components of a FastWAM-family policy."""

    # Unit of vLLM-Omni's regional torch.compile (``enforce_eager: false``): each
    # transformer layer of both experts is compiled as one region.
    _repeated_blocks = ["FastWAMBlock"]

    def __init__(
        self,
        cfg: FastWAMConfig,
        *,
        vae_encoder: WanVAEFirstFrameEncoder | None = None,
        text_encoder: WanUMT5Encoder | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        video = VideoExpert(
            hidden_dim=cfg.video_hidden_dim,
            in_dim=cfg.video_in_dim,
            out_dim=cfg.video_out_dim,
            ffn_dim=cfg.video_ffn_dim,
            text_dim=cfg.text_dim,
            freq_dim=cfg.freq_dim,
            eps=cfg.eps,
            patch_size=cfg.patch_size,
            num_heads=cfg.num_heads,
            attn_head_dim=cfg.attn_head_dim,
            num_layers=cfg.num_layers,
        )
        action = ActionExpert(
            hidden_dim=cfg.action_hidden_dim,
            action_dim=cfg.action_token_dim,
            ffn_dim=cfg.action_ffn_dim,
            text_dim=cfg.text_dim,
            freq_dim=cfg.freq_dim,
            eps=cfg.eps,
            num_heads=cfg.num_heads,
            attn_head_dim=cfg.attn_head_dim,
            num_layers=cfg.num_layers,
        )
        self.mot = FastWAMMoT(video, action)
        self.proprio_encoder = nn.Linear(cfg.proprio_dim, cfg.text_dim)
        self.action_codec = ActionLatentDecoder(cfg.action_codec) if cfg.action_codec is not None else None
        self.vae_encoder = vae_encoder if vae_encoder is not None else WanVAEFirstFrameEncoder()
        self.text_encoder = text_encoder if text_encoder is not None else WanUMT5Encoder()

    @property
    def device(self) -> torch.device:
        return self.proprio_encoder.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.proprio_encoder.weight.dtype

    @torch.inference_mode()
    def encode_prompt(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """umT5 context [1, L, text_dim]; padded rows are zeroed and the mask becomes
        all-True, as the reference does (its cross-attention sees the zero rows)."""
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, length in enumerate(seq_lens):
            context[i, length:] = 0
        return context, torch.ones_like(mask)

    @torch.inference_mode()
    def infer_action(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        *,
        num_steps: int,
        generator: torch.Generator | None,
        suite_id: int | None,
    ) -> torch.Tensor:
        """Predict one normalised action chunk.

        Args:
            image: uint8 [1, 3, H, W] (cameras already concatenated side by side).
            context, context_mask: prompt context from :meth:`encode_prompt`.
            proprio: normalised proprioception [1, proprio_dim] (float32).
            generator: CPU generator for the initial noise (``None``: global RNG).
            suite_id: task-suite index for a suite-conditioned action codec.

        Returns:
            float32 CPU tensor [action_horizon, action_dim] in normalised action space.
        """
        cfg = self.cfg
        device, dtype = self.device, self.dtype
        if image.dim() != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"image must be [1, 3, H, W], got {tuple(image.shape)}.")
        if image.shape[2] % 16 or image.shape[3] % 16:
            raise ValueError(f"image H and W must be multiples of 16, got {tuple(image.shape[2:])}.")

        latents = torch.randn(
            (1, cfg.num_action_tokens, cfg.action_token_dim), generator=generator, device="cpu", dtype=torch.float32
        ).to(device=device, dtype=dtype)

        pixels = image.to(device=device, dtype=dtype) * (2.0 / 255.0) - 1.0
        first_frame_latents = self.vae_encoder(pixels[0].unsqueeze(1).unsqueeze(0))

        context = context.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        proprio_token = self.proprio_encoder(proprio.to(device=device, dtype=dtype).unsqueeze(1))
        context = torch.cat([context, proprio_token.to(dtype=context.dtype)], dim=1)
        context_mask = torch.cat(
            [context_mask, torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=device)], dim=1
        )

        timestep_video = torch.zeros((1,), dtype=first_frame_latents.dtype, device=device)
        video = self.mot.video
        tokens, t_mod, video_context, video_context_mask, freqs, tokens_per_frame = video.prepare(
            first_frame_latents, timestep_video, context, context_mask
        )
        video_mask, action_mask = build_mot_attention_masks(
            tokens.shape[1], latents.shape[1], tokens_per_frame, tokens.device
        )
        cache_k, cache_v = self.mot.prefill_video_cache(
            tokens, freqs, t_mod, video_context, video_context_mask, video_mask
        )

        timesteps, deltas = build_action_schedule(
            num_steps, cfg.action_shift, cfg.num_train_timesteps, device, latents.dtype
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(dtype=latents.dtype, device=device)
            velocity = self.mot.denoise_action(latents, timestep, context, context_mask, cache_k, cache_v, action_mask)
            latents = latents + velocity * step_delta.to(device, dtype=latents.dtype)

        if self.action_codec is not None:
            codec_dtype = self.action_codec.input_proj.weight.dtype
            suite_ids = None
            if suite_id is not None:
                suite_ids = torch.full((latents.shape[0],), int(suite_id), dtype=torch.long, device=device)
            actions = self.action_codec(latents.to(dtype=codec_dtype), suite_ids).to(dtype=dtype)
        else:
            actions = latents
        return actions[0].detach().to(device="cpu", dtype=torch.float32)
