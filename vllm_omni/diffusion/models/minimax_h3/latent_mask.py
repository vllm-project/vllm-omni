# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-token latent editing helpers for MiniMax H3.

The denoise mask follows ComfyUI's convention: zero preserves the source and
one regenerates the row.  MiniMax H3 consumes one timestep per packed token,
so masks are reduced to the video 2x2 patch grid and the stereo audio row
order before they reach the denoise loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0

_MASK_LEVELS = 256.0


def _as_float_mask(mask: Any, *, name: str) -> torch.Tensor:
    try:
        value = torch.as_tensor(mask, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a numeric scalar, sequence, or tensor") from exc
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} values must be in [0, 1]")
    return value


def _leading_singleton_candidates(value: torch.Tensor):
    """Yield ``value`` followed by forms with leading singleton axes removed."""
    yield value
    while value.ndim > 0 and value.shape[0] == 1:
        value = value.squeeze(0)
        yield value


def _require_positive_dimension(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _quantize_model_mask(mask: torch.Tensor) -> torch.Tensor:
    # Bounding masks to 8-bit levels prevents anti-aliased inputs from creating
    # an unbounded number of distinct AdaLN timestep embeddings.
    return (torch.ceil(mask * _MASK_LEVELS) / _MASK_LEVELS).clamp_(0.0, 1.0)


@dataclass(frozen=True)
class MiniMaxH3ParsedMask:
    """Model-token mask plus the unquantized mask used for final x0 restore."""

    model_mask_rows: torch.Tensor
    restore_mask_rows: torch.Tensor

    def __post_init__(self) -> None:
        model_mask = _as_float_mask(self.model_mask_rows, name="model_mask_rows")
        restore_mask = _as_float_mask(self.restore_mask_rows, name="restore_mask_rows")
        if model_mask.ndim != 1:
            raise ValueError(f"model_mask_rows must be one-dimensional, got {tuple(model_mask.shape)}")
        if restore_mask.ndim not in (1, 2) or int(restore_mask.shape[0]) != int(model_mask.shape[0]):
            raise ValueError(
                "restore_mask_rows must have one row per model mask row, got "
                f"{tuple(restore_mask.shape)} for {int(model_mask.shape[0])} rows"
            )
        object.__setattr__(self, "model_mask_rows", _quantize_model_mask(model_mask))
        object.__setattr__(self, "restore_mask_rows", restore_mask)

    @property
    def all_generate(self) -> bool:
        return bool((self.model_mask_rows == 1.0).all().item()) and bool((self.restore_mask_rows == 1.0).all().item())


def minimax_h3_video_edit_masks(
    mask: Any,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
) -> MiniMaxH3ParsedMask:
    """Parse model-token and final-restore video masks.

    Accepted shapes are a scalar, a flat target-row vector, the token grid
    ``[T, H / 2, W / 2]``, or the full latent grid ``[T, H, W]``.  Leading
    singleton axes are ignored.  A full-grid mask is conservatively pooled by
    2x2 spatial maximum for the model mask. Its unquantized cell mask is also
    expanded to the packed ``[channel=24, patch_h=2, patch_w=2]`` feature order
    so final x0 restoration can preserve individual cells within that token.
    """
    latent_t = _require_positive_dimension(latent_t, name="latent_t")
    latent_h = _require_positive_dimension(latent_h, name="latent_h")
    latent_w = _require_positive_dimension(latent_w, name="latent_w")
    if latent_h % 2 or latent_w % 2:
        raise ValueError(
            "MiniMax H3 video masks require even latent_h and latent_w for the 2x2 token grid, "
            f"got {(latent_h, latent_w)}"
        )

    value = _as_float_mask(mask, name="video mask")
    token_shape = (latent_t, latent_h // 2, latent_w // 2)
    full_shape = (latent_t, latent_h, latent_w)
    row_count = math.prod(token_shape)

    for candidate in _leading_singleton_candidates(value):
        if candidate.ndim == 0:
            rows = candidate.expand(row_count).clone()
            return MiniMaxH3ParsedMask(rows, rows.clone())
        if tuple(candidate.shape) == token_shape:
            rows = candidate.reshape(-1).clone()
            return MiniMaxH3ParsedMask(rows, rows.clone())
        if tuple(candidate.shape) == full_shape:
            patch_cells = (
                candidate.reshape(
                    latent_t,
                    latent_h // 2,
                    2,
                    latent_w // 2,
                    2,
                )
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
            pooled = patch_cells.amax(dim=(3, 4)).reshape(-1)
            # H3's video row width is 24 channels * one temporal cell * 2 * 2.
            # Each channel repeats the same four spatial mask cells, in c,h,w
            # order, matching minimax_h3_patchify_video_latent.
            restore = (
                patch_cells.unsqueeze(3)
                .expand(
                    latent_t,
                    latent_h // 2,
                    latent_w // 2,
                    24,
                    2,
                    2,
                )
                .reshape(row_count, 96)
                .clone()
            )
            return MiniMaxH3ParsedMask(pooled, restore)
        if candidate.ndim == 1 and candidate.numel() == row_count:
            rows = candidate.clone()
            return MiniMaxH3ParsedMask(rows, rows.clone())

    raise ValueError(
        "video mask shape must be scalar, flat target rows, token grid "
        f"{token_shape}, or full latent grid {full_shape}; got {tuple(value.shape)}"
    )


def minimax_h3_video_mask_rows(
    mask: Any,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Return the quantized model-token rows for a video mask.

    Use :func:`minimax_h3_video_edit_masks` when final cell-level restoration
    is required as well.
    """
    return minimax_h3_video_edit_masks(
        mask,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
    ).model_mask_rows


def minimax_h3_audio_edit_masks(mask: Any, *, audio_t: int) -> MiniMaxH3ParsedMask:
    """Parse quantized model and raw restore masks in audio row order.

    Accepted shapes are a scalar, one temporal value per latent frame
    ``[T]``, explicit stereo values ``[2, T]``, or a flat ``[2 * T]`` row
    vector.  A temporal mask is repeated as ``[ch0 t0.., ch1 t0..]`` rather
    than interleaved by frame.
    """
    audio_t = _require_positive_dimension(audio_t, name="audio_t")
    value = _as_float_mask(mask, name="audio mask")
    row_count = audio_t * 2

    for candidate in _leading_singleton_candidates(value):
        rows = None
        if candidate.ndim == 0:
            rows = candidate.expand(row_count).clone()
        elif tuple(candidate.shape) == (2, audio_t):
            rows = candidate.reshape(-1).clone()
        elif candidate.ndim == 1 and candidate.numel() == audio_t:
            rows = candidate.repeat(2)
        elif candidate.ndim == 1 and candidate.numel() == row_count:
            rows = candidate.clone()
        if rows is not None:
            return MiniMaxH3ParsedMask(rows, rows.clone())

    raise ValueError(
        "audio mask shape must be scalar, one value per latent frame "
        f"({audio_t},), stereo grid (2, {audio_t}), or flat rows ({row_count},); got {tuple(value.shape)}"
    )


def minimax_h3_audio_mask_rows(mask: Any, *, audio_t: int) -> torch.Tensor:
    """Return the quantized model-token rows for an audio mask.

    Use :func:`minimax_h3_audio_edit_masks` to retain the raw final-restore
    mask as well.
    """
    return minimax_h3_audio_edit_masks(mask, audio_t=audio_t).model_mask_rows


def _require_finite_rows(rows: torch.Tensor, *, name: str) -> None:
    if not isinstance(rows, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if rows.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, width], got {tuple(rows.shape)}")
    if not torch.is_floating_point(rows):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(rows).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _model_mask_rows(mask_rows: Any, *, row_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _as_float_mask(mask_rows, name="mask_rows")
    if mask.ndim != 1 or int(mask.numel()) != row_count:
        raise ValueError(f"mask_rows must have shape ({row_count},), got {tuple(mask.shape)}")
    raw_mask = mask.to(device=device)
    return raw_mask, _quantize_model_mask(raw_mask)


def _restore_mask_rows(
    mask_rows: Any,
    *,
    row_count: int,
    row_width: int,
    device: torch.device,
) -> torch.Tensor:
    mask = _as_float_mask(mask_rows, name="restore_mask_rows")
    valid = (mask.ndim == 1 and tuple(mask.shape) == (row_count,)) or (
        mask.ndim == 2 and tuple(mask.shape) in ((row_count, 1), (row_count, row_width))
    )
    if not valid:
        raise ValueError(
            "restore_mask_rows must have shape "
            f"({row_count},), ({row_count}, 1), or ({row_count}, {row_width}); got {tuple(mask.shape)}"
        )
    return mask.to(device=device)


def _unit_interval_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return out


@dataclass(frozen=True)
class MiniMaxH3LatentEdit:
    """Clean source plus separate model-token and final-restore masks."""

    clean_rows: torch.Tensor
    anchor_rows: torch.Tensor
    mask_rows: torch.Tensor
    restore_mask_rows: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_finite_rows(self.clean_rows, name="clean_rows")
        _require_finite_rows(self.anchor_rows, name="anchor_rows")
        if self.clean_rows.shape != self.anchor_rows.shape:
            raise ValueError(
                f"clean_rows and anchor_rows shapes must match, got {tuple(self.clean_rows.shape)} "
                f"and {tuple(self.anchor_rows.shape)}"
            )
        if self.clean_rows.device != self.anchor_rows.device:
            raise ValueError("clean_rows and anchor_rows must be on the same device")
        if self.clean_rows.dtype != self.anchor_rows.dtype:
            raise ValueError("clean_rows and anchor_rows must have the same dtype")
        raw_mask, model_mask = _model_mask_rows(
            self.mask_rows,
            row_count=int(self.clean_rows.shape[0]),
            device=self.clean_rows.device,
        )
        restore_mask = _restore_mask_rows(
            raw_mask if self.restore_mask_rows is None else self.restore_mask_rows,
            row_count=int(self.clean_rows.shape[0]),
            row_width=int(self.clean_rows.shape[1]),
            device=self.clean_rows.device,
        )
        object.__setattr__(self, "mask_rows", model_mask)
        object.__setattr__(self, "restore_mask_rows", restore_mask)

    @classmethod
    def from_rows(
        cls,
        clean_rows: torch.Tensor,
        anchor_rows: torch.Tensor,
        mask_rows: Any,
        restore_mask_rows: Any | None = None,
    ) -> MiniMaxH3LatentEdit | None:
        """Validate/canonicalize an edit, returning ``None`` for all-generate."""
        edit = cls(
            clean_rows=clean_rows,
            anchor_rows=anchor_rows,
            mask_rows=mask_rows,
            restore_mask_rows=restore_mask_rows,
        )
        if edit.all_generate:
            return None
        return edit

    @property
    def all_generate(self) -> bool:
        restore_mask = self.restore_mask_rows
        assert restore_mask is not None
        return bool((self.mask_rows == 1.0).all().item()) and bool((restore_mask == 1.0).all().item())

    def to(self, *, device: torch.device, dtype: torch.dtype = torch.float32) -> MiniMaxH3LatentEdit:
        """Move the immutable request edit alongside its sampler rows."""
        device = torch.device(device)
        if (
            self.clean_rows.device == device
            and self.anchor_rows.device == device
            and self.mask_rows.device == device
            and self.restore_mask_rows is not None
            and self.restore_mask_rows.device == device
            and self.clean_rows.dtype == dtype
            and self.anchor_rows.dtype == dtype
        ):
            return self
        return MiniMaxH3LatentEdit(
            clean_rows=self.clean_rows.to(device=device, dtype=dtype),
            anchor_rows=self.anchor_rows.to(device=device, dtype=dtype),
            mask_rows=self.mask_rows.to(device=device),
            restore_mask_rows=(None if self.restore_mask_rows is None else self.restore_mask_rows.to(device=device)),
        )

    def validate_rows(self, rows: torch.Tensor, *, name: str = "rows") -> None:
        """Validate row layout/device without synchronizing values per step."""
        if not isinstance(rows, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if rows.ndim != 2 or not torch.is_floating_point(rows):
            raise ValueError(f"{name} must be floating point with shape [rows, width]")
        if rows.shape != self.clean_rows.shape:
            raise ValueError(f"{name} shape {tuple(rows.shape)} != edit rows {tuple(self.clean_rows.shape)}")
        if rows.device != self.clean_rows.device:
            raise ValueError(f"{name} and edit rows must be on the same device")

    def model_rows(self, state_rows: torch.Tensor) -> torch.Tensor:
        """Blend the persistent sampler state with the fixed source anchor."""
        self.validate_rows(state_rows, name="state_rows")
        if state_rows.dtype != self.anchor_rows.dtype:
            raise ValueError("state_rows and edit rows must have the same dtype")
        weight = self.mask_rows.to(dtype=state_rows.dtype).unsqueeze(-1)
        return torch.lerp(self.anchor_rows, state_rows, weight)

    def target_timesteps(
        self,
        timestep: float,
        condition_timestep: float,
        *,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """Return ``1 - mask * sigma`` capped at the modality's cond pin."""
        timestep = _unit_interval_float(timestep, name="timestep")
        condition_timestep = _unit_interval_float(condition_timestep, name="condition_timestep")
        if condition_timestep < timestep:
            raise ValueError("condition_timestep must be greater than or equal to timestep")
        expected_sigma = 1.0 - timestep
        if sigma is None:
            sigma = expected_sigma
        sigma = _unit_interval_float(sigma, name="sigma")
        if not math.isclose(sigma, expected_sigma, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("sigma must equal 1 - timestep")
        return torch.clamp(1.0 - self.mask_rows * sigma, max=condition_timestep)

    def x0(
        self,
        model_rows: torch.Tensor,
        velocity: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Convert masked velocity to x0 and restore the clean source fraction."""
        self.validate_rows(model_rows, name="model_rows")
        self.validate_rows(velocity, name="velocity")
        if model_rows.dtype != self.clean_rows.dtype:
            raise ValueError("model_rows and edit rows must have the same dtype")
        timestep = _unit_interval_float(timestep, name="timestep")
        model_weight = self.mask_rows.to(dtype=model_rows.dtype).unsqueeze(-1)
        effective_velocity = velocity.to(dtype=model_rows.dtype) * model_weight
        predicted_x0 = minimax_h3_rf_v_to_x0(
            model_rows,
            effective_velocity,
            torch.tensor(timestep, dtype=model_rows.dtype, device=model_rows.device),
        )
        restore_mask = self.restore_mask_rows
        assert restore_mask is not None
        restore_weight = restore_mask.to(dtype=model_rows.dtype)
        if restore_weight.ndim == 1:
            restore_weight = restore_weight.unsqueeze(-1)
        return torch.lerp(self.clean_rows, predicted_x0, restore_weight)


__all__ = [
    "MiniMaxH3LatentEdit",
    "MiniMaxH3ParsedMask",
    "minimax_h3_audio_edit_masks",
    "minimax_h3_audio_mask_rows",
    "minimax_h3_video_edit_masks",
    "minimax_h3_video_mask_rows",
]
