# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reference:
#   - PixDiT_T2I: pid/_src/networks/pixeldit_official.py
#   - LQ projection: pid/_src/networks/lq_projection_2d.py

import torch

from .lq_projection_2d import LQProjection2D, _build_gate
from .pixeldit import PixDiT_T2I


class PidNet(PixDiT_T2I):
    """PixDiT T2I with LQ condition injection for super-resolution.

    Inherits all PixDiT_T2I functionality (MMDiT patch blocks, PiT pixel blocks,
    text conditioning, RoPE, REPA). Adds LQ projection module and controlnet-style
    gated injection logic.

    Args (in addition to PixDiT_T2I args):
        lq_inject_mode: kept as a parameter for config compatibility — only
            "controlnet" is supported in this inference subset.
        lq_in_channels: LQ image channels (3 for RGB, 0 to disable image branch).
        lq_latent_channels: LQ latent channels (e.g. 16 for Wan VAE, 0 to disable).
        lq_hidden_dim: internal projection hidden dimension.
        lq_num_res_blocks: ResBlocks per LQ-projection branch.
        lq_latent_unpatchify_factor: optional unpatchify factor for patchified
            normalized latents in LQProjection2D. Flux2 uses 2.
        lq_conv_padding_mode: padding mode for all Conv2d layers in LQ projection.
        lq_gate_type: "sigma_aware_per_token" | "sigma_aware_per_token_per_dim".
        lq_interval: inject every N blocks.
        zero_init_lq: zero-init all LQ projections for safe pretrained start.
        sr_scale: super-resolution scale factor (default 4).
        latent_spatial_down_factor: VAE spatial downscale factor (default 8).
        pit_lq_inject: inject LQ features into PiT pixel blocks via a dedicated
            output head from the same LQ projection CNN backbone. Uses the same
            gate type as lq_gate_type.
    """

    def __init__(
        self,
        # --- PixDiT_T2I base args ---
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        rope_mode: str = "ntk_aware",
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
        repa_encoder_index: int = -1,
        # --- SR-specific args ---
        lq_inject_mode: str = "controlnet",
        lq_in_channels: int = 3,
        lq_latent_channels: int = 0,
        lq_hidden_dim: int = 512,
        lq_num_res_blocks: int = 4,
        # --- SR-specific args used in PiD v1.5 ---
        lq_latent_unpatchify_factor: int = 1,
        lq_conv_padding_mode: str = "zeros",
        lq_gate_type: str = "sigma_aware_per_token_per_dim",
        lq_interval: int = 1,
        zero_init_lq: bool = True,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        # --- PiT LQ injection args ---
        pit_lq_inject: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            num_groups=num_groups,
            hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size,
            pixel_attn_hidden_size=pixel_attn_hidden_size,
            pixel_num_groups=pixel_num_groups,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_text_blocks=num_text_blocks,
            patch_size=patch_size,
            txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length,
            use_text_rope=use_text_rope,
            text_rope_theta=text_rope_theta,
            rope_mode=rope_mode,
            rope_ref_h=rope_ref_h,
            rope_ref_w=rope_ref_w,
            repa_encoder_index=repa_encoder_index,
        )

        assert lq_inject_mode == "controlnet", (
            f"Only lq_inject_mode='controlnet' is supported in this inference subset, got '{lq_inject_mode}'"
        )
        self.lq_inject_mode = lq_inject_mode
        self.sr_scale = sr_scale
        self.lq_conv_padding_mode = lq_conv_padding_mode

        num_lq_outputs = (patch_depth + lq_interval - 1) // lq_interval
        self.num_lq_outputs = num_lq_outputs

        self.pit_lq_inject = pit_lq_inject

        self.lq_proj = LQProjection2D(
            in_channels=lq_in_channels,
            latent_channels=lq_latent_channels,
            hidden_dim=lq_hidden_dim,
            out_dim=hidden_size,
            patch_size=patch_size,
            sr_scale=sr_scale,
            latent_spatial_down_factor=latent_spatial_down_factor,
            latent_unpatchify_factor=lq_latent_unpatchify_factor,
            num_res_blocks=lq_num_res_blocks,
            num_outputs=num_lq_outputs,
            gate_type=lq_gate_type,
            interval=lq_interval,
            zero_init=zero_init_lq,
            conv_padding_mode=lq_conv_padding_mode,
            pit_output=pit_lq_inject,
        )

        # PiT LQ gate (applied to s_cond before pixel blocks)
        if pit_lq_inject:
            self.pit_lq_gate = _build_gate(lq_gate_type, hidden_size, zero_init=zero_init_lq)
        else:
            self.pit_lq_gate = None

    def _split_lq_outputs(self, lq_outputs):
        """Split LQ projection outputs into (lq_features, pit_lq_feature)."""
        lq_features = lq_outputs[: self.num_lq_outputs]
        cursor = self.num_lq_outputs

        pit_lq_feature = None
        if self.pit_lq_inject:
            if cursor >= len(lq_outputs):
                raise RuntimeError("pit_lq_inject=True but LQ projection did not return a PiT LQ feature.")
            pit_lq_feature = lq_outputs[cursor]
            cursor += 1

        if cursor != len(lq_outputs):
            raise RuntimeError(f"LQ projection returned {len(lq_outputs)} outputs, but consumed {cursor}.")
        return lq_features, pit_lq_feature

    def _compute_lq_features(self, lq_video_or_image, lq_latent, lq_mask, hs, ws):
        lq_kwargs = dict(
            lq_video_or_image=lq_video_or_image,
            lq_latent=lq_latent,
            target_ph=hs,
            target_pw=ws,
        )
        lq_features = self.lq_proj(**lq_kwargs)
        if lq_mask is not None:
            lq_features = [f * lq_mask.view(-1, 1, 1) for f in lq_features]
        return lq_features

    def _run_patch_blocks(
        self,
        s_main,
        y_emb,
        condition,
        pos,
        pos_txt,
        attn_mask_joint,
        lq_features,
        degrade_sigma=None,
    ):
        """Run patch_blocks loop with controlnet-style LQ injection."""
        has_lq = lq_features is not None

        for i in range(self.patch_depth):
            if has_lq and self.lq_proj.is_gate_active(i):
                out_idx = self.lq_proj._get_output_index(i)
                if out_idx < len(lq_features):
                    s_main = self.lq_proj.gate(s_main, lq_features[out_idx], sigma=degrade_sigma, out_idx=out_idx)

            s_main, y_emb = self.patch_blocks[i](
                s_main,
                y_emb,
                condition,
                pos,
                pos_txt,
                attn_mask_joint,
            )

            if 0 < self.repa_encoder_index == (i + 1):
                self.last_repa_tokens = s_main

        return s_main, y_emb

    def forward(
        self,
        x,
        t,
        y,
        s=None,
        mask=None,
        lq_video_or_image=None,
        lq_latent=None,
        lq_mask=None,
        degrade_sigma=None,
    ):
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        # Compute LQ features (patch-grid token features).
        has_lq = lq_video_or_image is not None or lq_latent is not None
        lq_features = None
        pit_lq_feature = None
        if has_lq:
            lq_outputs = self._compute_lq_features(lq_video_or_image, lq_latent, lq_mask, Hs, Ws)
            lq_features, pit_lq_feature = self._split_lq_outputs(lq_outputs)

        # Patch tokens
        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        # Text tokens
        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        # Condition signal: silu(t_emb), [B, 1, D]
        condition = torch.nn.functional.silu(t_emb)

        # Mask
        pad = None
        pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
        if mask is not None and isinstance(mask, torch.Tensor):
            m = mask
            while m.dim() > 2 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 3 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 2:
                pad = m == 0

        if s is None:
            s0 = self.s_embedder(x_patches)
            self.last_repa_tokens = None
            s_main = s0
            attn_mask_joint = None
            if pad is not None:
                pad_img = torch.zeros((B, L), dtype=torch.bool, device=x.device)
                pad_txt = (
                    pad[:, :Ltxt]
                    if pad.size(1) >= Ltxt
                    else torch.nn.functional.pad(pad, (0, Ltxt - pad.size(1)), value=True)
                )
                attn_mask_joint = torch.cat([pad_txt, pad_img], dim=1).view(B, 1, 1, Ltxt + L)

            s_main, y_emb = self._run_patch_blocks(
                s_main,
                y_emb,
                condition,
                pos,
                pos_txt,
                attn_mask_joint,
                lq_features,
                degrade_sigma=degrade_sigma,
            )

            s = torch.nn.functional.silu(t_emb + s_main)

        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        # Ensure patch token length matches the spatial grid L
        batch_size, length, _ = s.shape
        if length != L:
            if length > L:
                s = s[:, :L, :]
            else:
                pad_len = L - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)

        # Pixel pathway with optional PiT LQ injection
        s_cond_tokens = s
        if self.pit_lq_inject and pit_lq_feature is not None:
            s_cond_tokens = self.pit_lq_gate(s_cond_tokens, pit_lq_feature, sigma=degrade_sigma)
        s_cond = s_cond_tokens.reshape(B * L, self.hidden_size)

        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)  # [B*L, P², C_out]
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        output = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return output
