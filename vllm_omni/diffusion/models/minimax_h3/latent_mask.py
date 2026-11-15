# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 latent-edit mask parsing and sampler math."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0

_MASK_LEVELS = 256.0


def _quantize(mask: torch.Tensor) -> torch.Tensor:
    """Match ComfyUI's token-grid mask used by H3's timestep embedding."""
    return (torch.ceil(mask * _MASK_LEVELS) / _MASK_LEVELS).clamp_(0, 1)


@dataclass(frozen=True)
class MiniMaxH3ParsedMask:
    """Token mask for the model and raw mask for the final x0 restore."""

    model_mask_rows: torch.Tensor
    restore_mask_rows: torch.Tensor


def minimax_h3_video_edit_masks(
    mask: torch.Tensor,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
) -> MiniMaxH3ParsedMask:
    """Map the canonical full-grid video mask to model and restore rows.

    Full-grid masks are max-pooled over each 2x2 DiT token. Their raw four
    cells are repeated in the packed 24-channel feature order for restoration.
    """
    latent_t, latent_h, latent_w = int(latent_t), int(latent_h), int(latent_w)
    if latent_t <= 0 or latent_h <= 0 or latent_w <= 0 or latent_h % 2 or latent_w % 2:
        raise ValueError("video latent dimensions must be positive and spatially even")

    token_shape = (latent_t, latent_h // 2, latent_w // 2)
    full_shape = (latent_t, latent_h, latent_w)
    row_count = math.prod(token_shape)
    if mask.dtype != torch.float32 or tuple(mask.shape) != full_shape or not mask.is_contiguous():
        raise ValueError(f"canonical video mask must be contiguous FP32 with shape {full_shape}")
    cells = mask.reshape(latent_t, latent_h // 2, 2, latent_w // 2, 2).permute(0, 1, 3, 2, 4).contiguous()
    model = _quantize(cells.amax(dim=(3, 4)).reshape(-1))
    restore = cells.unsqueeze(3).expand(*cells.shape[:3], 24, 2, 2).reshape(row_count, 96).clone()
    return MiniMaxH3ParsedMask(model, restore)


def minimax_h3_audio_edit_masks(mask: torch.Tensor, *, audio_t: int) -> MiniMaxH3ParsedMask:
    """Flatten the canonical channel-major audio mask to H3 row order."""
    audio_t = int(audio_t)
    if audio_t <= 0:
        raise ValueError("audio_t must be positive")
    expected_shape = (2, audio_t)
    if mask.dtype != torch.float32 or tuple(mask.shape) != expected_shape or not mask.is_contiguous():
        raise ValueError(f"canonical audio mask must be contiguous FP32 with shape {expected_shape}")
    rows = mask.reshape(-1).clone()
    return MiniMaxH3ParsedMask(_quantize(rows), rows)


@dataclass(frozen=True)
class MiniMaxH3LatentEdit:
    """Clean source, model anchor, and masks for one packed latent stream."""

    clean_rows: torch.Tensor
    anchor_rows: torch.Tensor
    mask_rows: torch.Tensor
    restore_mask_rows: torch.Tensor

    @classmethod
    def from_rows(
        cls,
        clean_rows: torch.Tensor,
        anchor_rows: torch.Tensor,
        mask_rows: torch.Tensor,
        restore_mask_rows: torch.Tensor,
    ) -> MiniMaxH3LatentEdit:
        """Create an edit from already-validated encoder outputs.

        Value validation and all-generate elision belong at the encoder input
        boundary. This constructor only normalizes devices/dtypes and checks
        the cheap row-shape invariants needed by the sampler math.
        """
        if clean_rows.ndim != 2 or anchor_rows.shape != clean_rows.shape:
            raise ValueError("clean_rows and anchor_rows must have matching [rows, width] shape")
        raw_mask = torch.as_tensor(mask_rows, dtype=torch.float32, device=clean_rows.device)
        if raw_mask.ndim != 1 or raw_mask.numel() != clean_rows.shape[0]:
            raise ValueError(f"mask_rows must have shape ({clean_rows.shape[0]},)")
        restore = torch.as_tensor(restore_mask_rows, dtype=torch.float32, device=clean_rows.device)
        valid_restore = restore.ndim == 1 and restore.shape == raw_mask.shape
        valid_restore |= restore.ndim == 2 and tuple(restore.shape) in {
            (clean_rows.shape[0], 1),
            tuple(clean_rows.shape),
        }
        if not valid_restore:
            raise ValueError("restore_mask_rows must have one scalar or one feature row per clean row")
        return cls(clean_rows, anchor_rows, raw_mask, restore)

    def to(self, *, device: torch.device, dtype: torch.dtype = torch.float32) -> MiniMaxH3LatentEdit:
        return MiniMaxH3LatentEdit(
            self.clean_rows.to(device=device, dtype=dtype),
            self.anchor_rows.to(device=device, dtype=dtype),
            self.mask_rows.to(device=device),
            self.restore_mask_rows.to(device=device),
        )

    def model_rows(self, state_rows: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.anchor_rows, state_rows, self.mask_rows.to(state_rows.dtype).unsqueeze(-1))

    def target_timesteps(
        self,
        timestep: float,
        condition_timestep: float,
        *,
        sigma: float | None = None,
    ) -> torch.Tensor:
        sigma = 1.0 - timestep if sigma is None else float(sigma)
        return torch.clamp(1.0 - self.mask_rows * sigma, max=condition_timestep)

    def prepare(
        self,
        state_rows: torch.Tensor,
        timestep: float,
        condition_timestep: float,
        *,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the model rows and per-row timesteps for one denoise step."""
        return (
            self.model_rows(state_rows),
            self.target_timesteps(timestep, condition_timestep, sigma=sigma),
        )

    def x0(self, model_rows: torch.Tensor, velocity: torch.Tensor, timestep: float) -> torch.Tensor:
        weight = self.mask_rows.to(model_rows.dtype).unsqueeze(-1)
        predicted = minimax_h3_rf_v_to_x0(
            model_rows,
            velocity.to(model_rows.dtype) * weight,
            torch.tensor(float(timestep), dtype=model_rows.dtype, device=model_rows.device),
        )
        restore = self.restore_mask_rows.to(model_rows.dtype)
        if restore.ndim == 1:
            restore = restore.unsqueeze(-1)
        return torch.lerp(self.clean_rows, predicted, restore)


def minimax_h3_prepare_edit_rows(
    rows: torch.Tensor,
    update_mask: torch.Tensor,
    edit: MiniMaxH3LatentEdit | None,
    timestep: float,
    condition_timestep: float,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build one model input view without mutating the persistent sampler rows."""
    if edit is None:
        return rows, None
    model_rows = rows.clone()
    model_rows[update_mask], target_timesteps = edit.prepare(
        rows[update_mask],
        timestep,
        condition_timestep,
        sigma=sigma,
    )
    return model_rows, target_timesteps


__all__ = [
    "MiniMaxH3LatentEdit",
    "MiniMaxH3ParsedMask",
    "minimax_h3_audio_edit_masks",
    "minimax_h3_prepare_edit_rows",
    "minimax_h3_video_edit_masks",
]
