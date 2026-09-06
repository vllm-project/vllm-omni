# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_CONV_PADDING_MODES = {"zeros", "reflect", "replicate"}


def _validate_conv_padding_mode(conv_padding_mode: str) -> None:
    if conv_padding_mode not in _VALID_CONV_PADDING_MODES:
        raise ValueError(
            f"conv_padding_mode must be one of {sorted(_VALID_CONV_PADDING_MODES)}, got {conv_padding_mode!r}"
        )


# ---------------------------------------------------------------------------
# Gate modules
# ---------------------------------------------------------------------------


class SigmaAwarePerTokenGate(nn.Module):
    """Per-token scalar sigma-aware gate. Used in PiD v1.5.

    Init: content_proj.bias=2.0, log_alpha=log(5) ->
          gate ~= sigmoid(2.0 - 5*sigma): ~0.88 at sigma=0, ~0.5 at sigma=0.4, ~0.05 at sigma=1.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.content_proj = nn.Linear(dim * 2, 1)
        nn.init.trunc_normal_(self.content_proj.weight, std=0.01)
        nn.init.constant_(self.content_proj.bias, 2.0)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5.0)))

    def compute_gate_scalar(self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        assert sigma is not None, "SigmaAwarePerTokenGate requires degrade_sigma input"
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))  # (B, N, 1)
        sigma_offset = -self.log_alpha.exp() * sigma.float().view(-1, 1, 1)  # (B, 1, 1)
        return torch.sigmoid(content_logit + sigma_offset)  # (B, N, 1)

    def forward(self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.compute_gate_scalar(x, lq, sigma) * lq


class SigmaAwarePerTokenAndDimGate(nn.Module):
    """Per-token per-dim sigma-aware gate. Used in PiD v1.

    Content branch projects to dim instead of 1, so the gate is independent per
    (token, channel) instead of shared across channels. Sigma branch stays scalar
    per sample and broadcasts (B, 1, 1) -> (B, N, D).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.content_proj = nn.Linear(dim * 2, dim)
        nn.init.trunc_normal_(self.content_proj.weight, std=0.01)
        nn.init.constant_(self.content_proj.bias, 2.0)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5.0)))

    def compute_gate_scalar(self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        assert sigma is not None, "SigmaAwarePerTokenAndDimGate requires degrade_sigma input"
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))  # (B, N, D)
        sigma_offset = -self.log_alpha.exp() * sigma.float().view(-1, 1, 1)  # (B, 1, 1)
        return torch.sigmoid(content_logit + sigma_offset)  # (B, N, D)

    def forward(self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.compute_gate_scalar(x, lq, sigma) * lq


def _build_gate(gate_type: str, dim: int, zero_init: bool = True) -> nn.Module:
    # zero_init is intentionally not forwarded: gate zero-init is redundant when
    # output_heads is zero-init (lq=0 already kills the injection term).
    if gate_type == "sigma_aware_per_token":
        return SigmaAwarePerTokenGate(dim)
    elif gate_type == "sigma_aware_per_token_per_dim":
        return SigmaAwarePerTokenAndDimGate(dim)
    else:
        raise ValueError(
            f"Unknown gate_type: {gate_type!r}. Must be 'sigma_aware_per_token' or 'sigma_aware_per_token_per_dim'."
        )


# ---------------------------------------------------------------------------
# Pre-activation residual block (used by image / latent encoders below).
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Pre-activation residual block: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + skip."""

    def __init__(self, channels: int, num_groups: int = 4, conv_padding_mode: str = "zeros"):
        super().__init__()
        _validate_conv_padding_mode(conv_padding_mode)
        self.conv_padding_mode = conv_padding_mode
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# LQ Projection 2D
# ---------------------------------------------------------------------------


class LQProjection2D(nn.Module):
    """2D LQ projection for image super-resolution in pixel space.

    Spatial alignment strategy (lossless, no bilinear interpolation):

    Image branch:
      LQ image is at H_lq = H_hq / sr_scale. Patch grid is ph = H_hq / patch_size.
      Ratio = H_lq / ph = patch_size / sr_scale.
      - If ratio >= 1 (LQ res >= patch grid): PixelUnshuffle(ratio) to fold spatial
        dims into channels.
      - If ratio < 1 (LQ res < patch grid): Conv2d with PixelShuffle to upsample.

    Latent branch:
      LQ latent is at zH = H_lq / lsdf. Patch grid is ph = H_hq / patch_size.
      Optional latent_unpatchify_factor moves patchified latent channels back to
      spatial dims first. For Flux2 normalized latents, factor=2 converts
      [B, 128, H/16, W/16] -> [B, 32, H/8, W/8] without BN inverse normalization.
      z_patch_ratio = ph / zH = (sr_scale * effective_lsdf) / patch_size,
      where effective_lsdf = latent_spatial_down_factor / latent_unpatchify_factor.
      - If z_patch_ratio <= 1 (latent res >= patch grid): fold.
      - If z_patch_ratio > 1 (latent res < patch grid): nearest interpolate.

    Args:
        in_channels: LQ image channels (3 for RGB, 0 to disable image branch).
        latent_channels: LQ latent channels (e.g. 16 for Wan VAE, 0 to disable).
        hidden_dim: internal feature dimension for conv processing.
        out_dim: output dimension (must match transformer hidden_size).
        patch_size: spatial patch size of the transformer (e.g. 16).
        sr_scale: super-resolution scale factor (LQ is sr_scale times smaller).
        latent_spatial_down_factor: VAE spatial downscale factor (default 8).
        latent_unpatchify_factor: optional spatial unpatchify factor for patchified
            latents. 1 disables it. Flux2 normalized latents should use 2.
        num_res_blocks: number of ResBlocks after initial conv pair in each branch.
        num_outputs: number of output feature sets (one per transformer block).
        gate_type: "sigma_aware_per_token" | "sigma_aware_per_token_per_dim".
        interval: inject every N blocks (only relevant when num_outputs > 1).
        zero_init: if True, zero-init all output projections for safe pretrained start.
        conv_padding_mode: padding mode for all Conv2d layers.
        pit_output: if True, add a dedicated output head for PiT block injection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 0,
        hidden_dim: int = 512,
        out_dim: int = 1536,
        patch_size: int = 16,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        latent_unpatchify_factor: int = 1,
        num_res_blocks: int = 4,
        num_outputs: int = 1,
        gate_type: str = "sigma_aware_per_token_per_dim",
        interval: int = 1,
        zero_init: bool = True,
        conv_padding_mode: str = "zeros",
        pit_output: bool = False,
    ):
        super().__init__()
        assert in_channels > 0 or latent_channels > 0, "At least one of in_channels or latent_channels must be > 0"

        _validate_conv_padding_mode(conv_padding_mode)

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.sr_scale = sr_scale
        self.latent_spatial_down_factor = latent_spatial_down_factor
        self.latent_unpatchify_factor = latent_unpatchify_factor
        self.num_res_blocks = num_res_blocks
        if latent_unpatchify_factor > 1 and latent_spatial_down_factor % latent_unpatchify_factor != 0:
            raise ValueError(
                "latent_spatial_down_factor must be divisible by latent_unpatchify_factor, got "
                f"{latent_spatial_down_factor} and {latent_unpatchify_factor}."
            )
        self.effective_latent_spatial_down_factor = latent_spatial_down_factor // latent_unpatchify_factor
        self.num_outputs = num_outputs
        self.interval = interval
        self.zero_init = zero_init
        self.conv_padding_mode = conv_padding_mode
        self.pit_output = pit_output

        # --- Image branch ---
        if in_channels > 0:
            assert patch_size >= sr_scale and patch_size % sr_scale == 0, (
                f"patch_size ({patch_size}) must be >= sr_scale ({sr_scale}) and divisible"
            )
            self.image_unshuffle_factor = patch_size // sr_scale
            unshuffle_ch = in_channels * self.image_unshuffle_factor**2
            layers = [
                nn.Conv2d(unshuffle_ch, hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode=conv_padding_mode),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode=conv_padding_mode),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
            self.image_conv = nn.Sequential(*layers)
        else:
            self.image_conv = None
            self.image_unshuffle_factor = 0

        # --- Latent branch ---
        if latent_channels > 0:
            if latent_unpatchify_factor > 1 and latent_channels % (latent_unpatchify_factor**2) != 0:
                raise ValueError(
                    "latent_channels must be divisible by latent_unpatchify_factor**2, got "
                    f"{latent_channels} and {latent_unpatchify_factor}."
                )
            effective_latent_channels = latent_channels // (latent_unpatchify_factor**2)
            z_to_patch_ratio = (sr_scale * self.effective_latent_spatial_down_factor) / patch_size
            self.z_to_patch_ratio = z_to_patch_ratio

            if z_to_patch_ratio > 1:
                self.latent_upsampler = None
                self.latent_upsample_ratio = int(z_to_patch_ratio)
                latent_proj_in_ch = effective_latent_channels
            elif z_to_patch_ratio == 1:
                self.latent_upsampler = None
                latent_proj_in_ch = effective_latent_channels
            else:
                fold_factor = int(1 / z_to_patch_ratio)
                assert fold_factor * z_to_patch_ratio == 1.0, (
                    f"fold_factor {fold_factor} * z_to_patch_ratio {z_to_patch_ratio} != 1"
                )
                self.latent_upsampler = None
                self.latent_fold_factor = fold_factor
                latent_proj_in_ch = effective_latent_channels * fold_factor**2

            layers = [
                nn.Conv2d(
                    latent_proj_in_ch, hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode=conv_padding_mode
                ),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode=conv_padding_mode),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
            self.latent_proj = nn.Sequential(*layers)
        else:
            self.latent_proj = None
            self.z_to_patch_ratio = 0
            self.latent_upsampler = None

        # --- Merge + shared ResBlocks (if both branches active) ---
        if in_channels > 0 and latent_channels > 0:
            layers = [
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, padding_mode=conv_padding_mode),
                nn.SiLU(),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
            self.merge = nn.Sequential(*layers)
        else:
            self.merge = None

        # --- Output heads ---
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num_outputs)])

        # --- Dedicated PiT output head (separate from DiT heads) ---
        if pit_output:
            self.pit_head = nn.Linear(hidden_dim, out_dim)
        else:
            self.pit_head = None

        # --- Gate modules (one per injection point) ---
        self.gate_modules = nn.ModuleList(
            [_build_gate(gate_type, out_dim, zero_init=zero_init) for _ in range(num_outputs)]
        )

    def is_gate_active(self, block_idx: int) -> bool:
        """Whether gate() should be called for this block index."""
        if self.interval > 1:
            return block_idx % self.interval == 0
        return True

    def _get_output_index(self, block_idx: int) -> int:
        """Map block_idx to output head index, respecting interval."""
        if self.interval > 1:
            return block_idx // self.interval
        return block_idx

    def gate(
        self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor | None = None, out_idx: int = 0
    ) -> torch.Tensor:
        """Apply gating: inject lq features into transformer hidden state x."""
        return self.gate_modules[out_idx](x, lq, sigma=sigma)

    def _align_image_to_patch_grid(
        self, lq_video_or_image: torch.Tensor, target_ph: int, target_pw: int
    ) -> torch.Tensor:
        """Align LQ image to patch grid via PixelUnshuffle."""
        f = self.image_unshuffle_factor
        B, C, H_lq, W_lq = lq_video_or_image.shape
        target_H_lq = target_ph * f
        target_W_lq = target_pw * f

        if H_lq != target_H_lq or W_lq != target_W_lq:
            lq_video_or_image = F.interpolate(
                lq_video_or_image, size=(target_H_lq, target_W_lq), mode="bilinear", align_corners=False
            )

        x = F.pixel_unshuffle(lq_video_or_image, f)  # [B, C*f*f, target_ph, target_pw]
        return self.image_conv(x)  # [B, hidden_dim, target_ph, target_pw]

    def _align_latent_spatial_to_patch_grid(self, lq_latent: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        """Align LQ latent to patch grid via nearest interpolate or fold.

        Returns [B, effective_latent_channels_or_folded, ph, pw].
        """
        # Fold patchified latent channels back to spatial dims first (e.g.
        # Flux2 BN-normalized latents: [B, 128, H/16, W/16] -> [B, 32, H/8,
        # W/8], no BN inverse normalization).  Mirrors Flux2Pipeline
        # ._unpatchify_latents and must match the conv in-channels derived in
        # __init__ from latent_unpatchify_factor.
        if self.latent_unpatchify_factor > 1:
            f = self.latent_unpatchify_factor
            B_, C_patch, H, W = lq_latent.shape
            lq_latent = (
                lq_latent.reshape(B_, C_patch // (f * f), f, f, H, W)
                .permute(0, 1, 4, 2, 5, 3)
                .reshape(B_, C_patch // (f * f), H * f, W * f)
            )

        B, z_dim = lq_latent.shape[:2]

        if self.z_to_patch_ratio > 1:
            z_aligned = F.interpolate(lq_latent, size=(ph, pw), mode="nearest")
        elif self.z_to_patch_ratio == 1:
            z_aligned = lq_latent
            if z_aligned.shape[2] != ph or z_aligned.shape[3] != pw:
                z_aligned = F.interpolate(z_aligned, size=(ph, pw), mode="bilinear", align_corners=False)
        else:
            f = self.latent_fold_factor
            zH_expected, zW_expected = ph * f, pw * f
            if lq_latent.shape[2] != zH_expected or lq_latent.shape[3] != zW_expected:
                lq_latent = F.interpolate(
                    lq_latent, size=(zH_expected, zW_expected), mode="bilinear", align_corners=False
                )
            z_aligned = lq_latent.reshape(B, z_dim, ph, f, pw, f)
            z_aligned = z_aligned.permute(0, 1, 3, 5, 2, 4)
            z_aligned = z_aligned.reshape(B, z_dim * f * f, ph, pw)

        return z_aligned

    def _align_latent_to_patch_grid(self, lq_latent: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        """Align LQ latent to patch grid and project to [B, hidden_dim, ph, pw]."""
        z_aligned = self._align_latent_spatial_to_patch_grid(lq_latent, ph, pw)
        return self.latent_proj(z_aligned)

    def forward(
        self,
        lq_video_or_image: torch.Tensor | None = None,
        lq_latent: torch.Tensor | None = None,
        target_ph: int = 0,
        target_pw: int = 0,
    ) -> list[torch.Tensor]:
        """Project LQ inputs to patch-aligned token features.

        Returns:
            List of [B, N, out_dim] tensors where N = target_ph * target_pw.
            Length = num_outputs (+ 1 if pit_output=True).
        """
        assert target_ph > 0 and target_pw > 0, "Must provide target_ph and target_pw"
        features = []

        # Image branch
        if self.image_conv is not None and lq_video_or_image is not None:
            features.append(self._align_image_to_patch_grid(lq_video_or_image, target_ph, target_pw))

        # Latent branch
        if self.latent_proj is not None and lq_latent is not None:
            features.append(self._align_latent_to_patch_grid(lq_latent, target_ph, target_pw))

        # Merge or select single branch
        if len(features) == 2 and self.merge is not None:
            merged = self.merge(torch.cat(features, dim=1))
        elif len(features) == 1:
            merged = features[0]
        else:
            ref = lq_video_or_image if lq_video_or_image is not None else lq_latent
            if ref is None:
                raise ValueError("LQProjection2D requires at least one LQ input or a reference tensor.")
            B, device, dtype = ref.shape[0], ref.device, ref.dtype
            N = target_ph * target_pw
            num_total = self.num_outputs + (1 if self.pit_output else 0)
            return [torch.zeros(B, N, self.out_dim, device=device, dtype=dtype) for _ in range(num_total)]

        # Flatten to tokens: [B, hidden_dim, ph, pw] -> [B, N, hidden_dim]
        tokens = merged.flatten(2).transpose(1, 2)

        # Project through output heads
        outputs = [head(tokens) for head in self.output_heads]

        # Append dedicated PiT head output as last element
        if self.pit_head is not None:
            outputs.append(self.pit_head(tokens))
        return outputs
