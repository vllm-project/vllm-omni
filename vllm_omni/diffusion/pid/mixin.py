# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PidDecodeMixin -- drop-in PiD super-resolution support for LDM pipelines.

A pipeline gains PiD support by:

1.  Inheriting ``PidDecodeMixin``.
2.  Setting ``PID_BACKBONE`` to the backbone name registered in
    :mod:`vllm_omni.diffusion.pid.config`
    (e.g. ``"qwenimage"``, ``"flux"``, ``"sd3"``, ``"sdxl"``, ``"flux2"``).
3.  Calling ``self._init_pid_decoder(od_config)`` in ``__init__``.
4.  Inserting one hook in ``_decode_latents`` right after latent unpack::

        pid_out = self.maybe_pid_decode(
            latents_4d, height, width, caption=..., pid_override=...
        )
        if pid_out is not None:
            return DiffusionOutput(output=pid_out, ...)

No per-model wrapper file is needed.  Override resolution, rank gating,
and caption fallback are handled here.
"""

from __future__ import annotations

import logging
from dataclasses import replace as _dc_replace
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from vllm_omni.diffusion.pid.decoder import PidDecodeConfig, PidDecoder

logger = logging.getLogger(__name__)


class PidDecodeMixin:
    """Mixin that adds PiD super-resolution to an LDM pipeline.

    Subclasses set :attr:`PID_BACKBONE` and call :meth:`_init_pid_decoder`
    during construction.  At decode time, call :meth:`maybe_pid_decode`;
    it returns a decoded image tensor (``[-1, 1]``) when PiD is active and
    ``None`` otherwise (so the caller falls back to its VAE path).
    """

    PID_BACKBONE: ClassVar[str] = ""

    # -- initialisation -----------------------------------------------------

    def _init_pid_decoder(
        self,
        od_config: Any,
    ) -> None:
        """Initialise PiD decoder from ``od_config.pid_decode`` (if any).

        When PiD is enabled, weights are loaded **eagerly** here (not
        deferred to the first request/warmup).  The decoder is registered
        as an ``nn.Module`` submodule and declared as a *resident* module
        so the offloader keeps it on GPU (PiD must not be CPU-offloaded).
        """
        self._pid_config: PidDecodeConfig | None = self._resolve_pid_config(od_config)
        if self._pid_config is None or not self._pid_config.enabled:
            self._pid_decoder: PidDecoder | None = None
            return

        if not self.PID_BACKBONE:
            raise RuntimeError(f"{type(self).__name__} must set PID_BACKBONE to use PiD.")

        decoder = PidDecoder(
            config=self._pid_config,
            backbone=self.PID_BACKBONE,
            enforce_eager=bool(getattr(od_config, "enforce_eager", False)),
        )
        # Eager load: weights are resident before the first request.
        decoder.load_weights()
        # Register as submodule (nn.Module.__setattr__ handles this, but
        # we also declare it as resident so the offloader does not
        # CPU-offload PiD weights).
        self._pid_decoder = decoder
        self._declare_pid_resident()

    def _declare_pid_resident(self) -> None:
        existing = list(getattr(self, "_resident_modules", []))
        if "_pid_decoder" not in existing:
            existing.append("_pid_decoder")
        self._resident_modules = existing

    @staticmethod
    def _resolve_pid_config(od_config: Any) -> PidDecodeConfig | None:
        """Normalize ``od_config.pid_decode`` to a ``PidDecodeConfig`` or None."""
        raw = getattr(od_config, "pid_decode", None)
        if raw is None:
            return None
        if isinstance(raw, PidDecodeConfig):
            return raw
        if isinstance(raw, dict):
            return PidDecodeConfig(**raw)
        raise TypeError(f"pid_decode must be PidDecodeConfig, dict, or None, got {type(raw)!r}")

    # -- decode hook -------------------------------------------------------

    def maybe_pid_decode(
        self,
        latents_4d: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        """Return a PiD-decoded image tensor, or ``None`` to fall back to VAE.

        Args:
            latents_4d: Unpacked latent ``(B, C, 1, zH, zW)`` (or
                ``(B, C, zH, zW)``; the singleton spatial dim is squeezed).
            height: *LDM* output height (pre-super-resolution).
            width: *LDM* output width (pre-super-resolution).

        """
        caption = getattr(self, "_pid_caption", None)
        pid_override = getattr(self, "_pid_override", None)
        pid_decoder = self._pid_decoder
        pid_config = self._pid_config

        # Resolve per-request override.
        if pid_override is not None:
            ov_enabled = pid_override.get("enabled")
            if ov_enabled is False:
                return None
            if ov_enabled is True and pid_decoder is None:
                raise RuntimeError(
                    "PiD decode was requested per-request (pid_decode.enabled=True) "
                    "but the pipeline was not configured with --pid-enable at startup. "
                    "PiD weights are not lazily loaded on request; restart the service "
                    "with --pid-enable to enable this feature."
                )
            if pid_decoder is not None and pid_config is not None:
                overrides = {
                    k: pid_override[k] for k in ("scale", "num_steps", "seed", "degrade_sigma") if k in pid_override
                }
                if overrides:
                    pid_config = _dc_replace(pid_config, **overrides)

        # Not enabled.
        if pid_decoder is None or pid_config is None:
            return None

        # Rank gate: only rank 0 runs PiD decode.
        if dist.is_initialized() and dist.get_rank() != 0:
            return None

        if caption is None:
            logger.warning("PiD decode is enabled but no caption was provided; falling back to an empty prompt.")
            caption = ""

        lq_latent = latents_4d.squeeze(2) if latents_4d.dim() == 5 else latents_4d
        return pid_decoder.decode(
            lq_latent=lq_latent,
            caption=caption,
            output_size=(
                int(height * pid_config.scale),
                int(width * pid_config.scale),
            ),
            degrade_sigma=pid_config.degrade_sigma,
            num_steps=pid_config.num_steps,
            seed=pid_config.seed,
        )
