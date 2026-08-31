# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic PiD (Pixel Diffusion) decoder shared by all LDM backbones.

Backbone-specific differences (VAE latent channels / spatial compression)
are captured by the ``backbone`` parameter, which selects the matching net
config from :mod:`vllm_omni.diffusion.pid.config`.  No per-model wrapper
file is needed -- a pipeline just declares its backbone name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.pid.checkpoint import load_pid_checkpoint
from vllm_omni.diffusion.pid.config import (
    PID_SAMPLING_CONFIG,
    get_pid_net_config,
)
from vllm_omni.diffusion.pid.pid_model import PidInferenceModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PidDecodeConfig:
    enabled: bool = False
    # Path to the PiD distilled checkpoint (.pth).
    checkpoint_path: str = ""
    # Local directory containing gemma-2-2b-it weights (required).
    gemma_model: str = ""
    # Super-resolution factor applied to the LDM output resolution.
    scale: int = 4
    # Number of distilled SDE sampling steps (4 for the distilled checkpoint).
    num_steps: int = 4
    # Base RNG seed. The pipeline may override this per request.
    seed: int = 0
    # Noise level injected into the LQ latent. 0.0 means the clean x_0 latent.
    degrade_sigma: float = 0.0
    # Compute precision preset: "bfloat16" (default, matches distilled
    # checkpoint training), "float16" (fp16 autocast), or "float32" (pure
    # fp32 forward, disables autocast). The tensor container is always
    # float32; non-float32 values enable autocast for matmuls only.
    precision: str = "bfloat16"


class PidDecoder(nn.Module):
    """Decode an LDM ``x_0`` latent into a high-resolution RGB image via PiD.

    Inherits :class:`torch.nn.Module` so that the wrapped
    :class:`PidInferenceModel` is visible in the pipeline's module tree
    (``named_modules()``, component discovery, CPU-offload classification).

    The PiD model (PidNet + Gemma text encoder) is loaded eagerly at
    construction time (via :meth:`load_weights`) and stays resident in GPU
    memory for the lifetime of this object.

    Args:
        config: PiD decode configuration.
        backbone: Backbone name (e.g. ``"qwenimage"``, ``"flux"``) used to
            select the matching net config from the registry.
    """

    def __init__(
        self,
        config: PidDecodeConfig,
        backbone: str,
        enforce_eager: bool = False,
        od_config: Any = None,
    ):
        super().__init__()
        self.device = get_local_device()
        self._config = config
        self._backbone = backbone
        self._enforce_eager = enforce_eager
        self._od_config = od_config
        self._model: PidInferenceModel | None = None

    # -- weight loading ----------------------------------------------------

    def load_weights(self) -> None:
        if self._model is not None:
            return

        cfg = self._config
        net_kwargs = get_pid_net_config(self._backbone)
        logger.info(
            "Loading PiD model (backbone=%s) from %s ...",
            self._backbone,
            cfg.checkpoint_path,
        )

        model = PidInferenceModel(
            net_kwargs=net_kwargs,
            gemma_model_id=cfg.gemma_model,
            sampling_overrides=dict(PID_SAMPLING_CONFIG),
            precision=cfg.precision,
            enforce_eager=self._enforce_eager,
        )
        load_pid_checkpoint(model, cfg.checkpoint_path, backbone=self._backbone)
        model.eval()
        model.to(self.device)

        # Register as submodule so pipeline.named_modules() discovers it.
        self._model = model
        logger.info(
            "PiD model loaded (backbone=%s, PidNet + Gemma) resident on %s.",
            self._backbone,
            self.device,
        )

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        lq_latent: torch.Tensor,
        caption: str | list[str],
        output_size: tuple[int, int],
        degrade_sigma: float | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Run PiD decoding.

        Args:
            lq_latent: LDM ``x_0`` latent, shape ``(B, C, zH, zW)``.
            caption: Original text prompt. A single ``str`` is broadcast to
                the whole batch; a ``list[str]`` must match ``lq_latent``'s
                batch size.
            output_size: Target pixel resolution ``(H_pixel, W_pixel)``.
            degrade_sigma: Noise level injected into the LQ latent.
            num_steps: Number of distilled SDE sampling steps.
            seed: RNG seed.

        Returns:
            Tensor of shape ``(B, 3, H_pixel, W_pixel)`` in ``[-1, 1]``.
        """
        self.load_weights()

        return self._model.generate_samples_from_batch(
            lq_latent=lq_latent,
            caption=caption,
            output_size=output_size,
            degrade_sigma=(degrade_sigma if degrade_sigma is not None else self._config.degrade_sigma),
            num_steps=num_steps or self._config.num_steps,
            seed=seed if seed is not None else self._config.seed,
        )
