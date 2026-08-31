# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized latent-form adaptation table (Plan B).

Maps each pipeline family to (PiD backbone, ``to_x0`` pure function) that
converts whatever the pipeline's existing ``output_type == "latent"`` branch
returns into PiD's expected ``x_0`` form plus the LDM pixel size
``(pid_h, pid_w)``.

Key principle: pipelines are never modified for PiD. Each ``to_x0`` receives
the latent-branch output as-is; all conversion happens here.

Contract::

    to_x0(latent, height, width, vae_scale_factor, *, pipeline=None)
        -> (x0, pid_h, pid_w)

- ``latent``: the pipeline latent-branch output, as-is.
- ``height``/``width``: request target pixel size (may be ``None``).
- ``pipeline``: optional context, used only to read model-side constants
  (e.g. Flux2's ``vae.bn`` running stats); ignored by other families.
- ``x0``: ``[B, C, zH, zW]`` grid with ``C == lq_latent_channels`` of the
  backbone net config.
- Pure tensor ops, directly unit-testable; raises ``ValueError`` on
  non-canonical shapes (Runner is fail-loud).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["LatentForm", "LATENT_FORMS", "lookup_latent_form"]


@dataclass(frozen=True)
class LatentForm:
    """Latent form of one pipeline family.

    Attributes:
        backbone: PiD net config / checkpoint registry key (e.g. ``"flux"``).
        to_x0: pure conversion function, see module docstring.
    """

    backbone: str
    to_x0: Callable[..., tuple[torch.Tensor, int, int]]


def _unpack_packed_2x2(
    latent: torch.Tensor,
    height: int | None,
    width: int | None,
    vae_scale_factor: int,
    *,
    pipeline: object | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Flux / QwenImage family: packed tokens [B, T, 4C] -> grid [B, C, zH, zW].

    Inverse of ``FluxPipeline._unpack_latents`` /
    ``QwenImagePipeline._unpack_latents`` (row-major canonical order).
    """
    del pipeline
    if latent.dim() == 5:
        latent = latent.squeeze(2)
    if latent.dim() == 4:
        _, _, z_h, z_w = latent.shape
        return latent, z_h * vae_scale_factor, z_w * vae_scale_factor
    if latent.dim() != 3:
        raise ValueError(f"packed-2x2 latent must be 3D token form, got shape {tuple(latent.shape)}")
    b, t, c = latent.shape
    if c % 4 != 0:
        raise ValueError(f"packed-2x2 latent channels {c} not divisible by 4")
    if height is None or width is None:
        raise ValueError("packed-2x2 latent requires non-None height/width")
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))
    if h < 2 or w < 2:
        raise ValueError(f"target size {height}x{width} too small for vae_scale_factor={vae_scale_factor}")
    if t != (h // 2) * (w // 2):
        raise ValueError(
            f"packed-2x2 token count {t} != grid {(h // 2) * (w // 2)} "
            f"(height={height}, width={width}, vae_scale_factor={vae_scale_factor}); "
            "non-canonical token grids (edit/img2img latents) are not supported by PiD yet"
        )
    latent = latent.view(b, h // 2, w // 2, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c // 4, h, w)
    return latent, h * vae_scale_factor, w * vae_scale_factor


def _identity(
    latent: torch.Tensor,
    height: int | None,
    width: int | None,
    vae_scale_factor: int,
    *,
    pipeline: object | None = None,
) -> tuple[torch.Tensor, int, int]:
    """ZImage / SD3 / SDXL family: native 4D grid is already x_0."""
    del pipeline
    if latent.dim() == 5:
        latent = latent.squeeze(2)
    if latent.dim() != 4:
        raise ValueError(f"identity latent must be 4D grid form, got shape {tuple(latent.shape)}")
    _, _, z_h, z_w = latent.shape
    return latent, z_h * vae_scale_factor, z_w * vae_scale_factor


def _patchify_and_normalize(
    latent: torch.Tensor,
    height: int | None,
    width: int | None,
    vae_scale_factor: int,
    *,
    pipeline: object | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Flux2 / Flux2Klein family: VAE-ready grid -> BN-normalized 2x2 patchified grid.

    The pipeline's latent branch returns the VAE-ready grid [B, 32, zH, zW]
    (BN denorm + unpatchify already applied — original behavior, preserved).
    The official PiD flux2 checkpoint expects the BN-normalized 2x2-patchified
    grid [B, 128, zH/2, zW/2] (the transformer loop's native form; see upstream
    ``pid/_src/networks/lq_projection_2d.py``: "For Flux2 normalized latents,
    factor=2 converts [B, 128, H/16, W/16] -> [B, 32, H/8, W/8] without BN
    inverse normalization").

    The two forms are related by a pure invertible tensor transform:
    2x2 patchify (exact inverse of ``_unpatchify_latents``, channel order
    c*4 + ph*2 + pw) + BN re-normalization from ``pipeline.vae.bn`` running
    stats. Normalization is computed in fp32 to avoid a second bf16 rounding
    amplifying the pipeline-side denorm rounding.
    """
    if latent.dim() != 4:
        raise ValueError(f"flux2 latent must be 4D VAE-ready grid form, got shape {tuple(latent.shape)}")
    b, c, z_h, z_w = latent.shape
    if c % 4 != 0:
        raise ValueError(f"flux2 latent channels {c} not divisible by 4 (2x2 patchify)")
    if z_h % 2 != 0 or z_w % 2 != 0:
        raise ValueError(f"flux2 latent spatial dims must be even for 2x2 patchify, got {z_h}x{z_w}")
    bn = getattr(getattr(pipeline, "vae", None), "bn", None)
    if bn is None or not hasattr(bn, "running_mean"):
        raise ValueError(
            "flux2 latent form requires the pipeline (for vae.bn running stats) to "
            "re-normalize the VAE-ready latent into PiD's BN-normalized latent space"
        )
    x = latent.view(b, c, z_h // 2, 2, z_w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, z_h // 2, z_w // 2)
    eps = float(getattr(getattr(pipeline.vae, "config", None), "batch_norm_eps", 1e-4))
    mean = bn.running_mean.detach().view(1, -1, 1, 1)
    std = torch.sqrt(bn.running_var.detach().view(1, -1, 1, 1) + eps)
    x = (x.float() - mean.float()) / std.float()
    return x.to(latent.dtype), z_h * vae_scale_factor, z_w * vae_scale_factor


# key = pipeline class name; subclasses resolve via __mro__ fallback.
LATENT_FORMS: dict[str, LatentForm] = {
    "FluxPipeline": LatentForm("flux", _unpack_packed_2x2),
    "QwenImagePipeline": LatentForm("qwenimage", _unpack_packed_2x2),
    "ZImagePipeline": LatentForm("flux", _identity),
    "Flux2Pipeline": LatentForm("flux2", _patchify_and_normalize),
    "Flux2KleinPipeline": LatentForm("flux2", _patchify_and_normalize),
    "StableDiffusion3Pipeline": LatentForm("sd3", _identity),
    "StableDiffusionXLPipeline": LatentForm("sdxl", _identity),
}


def lookup_latent_form(pipeline: object) -> LatentForm | None:
    for cls in type(pipeline).__mro__:
        form = LATENT_FORMS.get(cls.__name__)
        if form is not None:
            return form
    return None
