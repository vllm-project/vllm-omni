# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.schedulers import DMD2EulerScheduler
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


class DMD2PipelineMixin:
    """Mixin for FastGen DMD2-distilled models. Must appear before the base pipeline in MRO."""

    def __init_dmd2__(self) -> None:
        """Call after super().__init__() to apply DMD2 scheduler and read model_index."""
        local_files_only = os.path.exists(self.od_config.model)
        try:
            model_index = _load_json(self.od_config.model, "model_index.json", local_files_only)
        except Exception:
            model_index = {}

        dmd2_timesteps = model_index.get("dmd2_denoising_timesteps", [999, 937, 833, 624])
        self.num_inference_steps = model_index.get("dmd2_num_inference_steps", 4)
        shift = model_index.get("dmd2_scheduler_shift", 1.0)
        self.dmd2_guidance_scale = model_index.get("dmd2_guidance_scale", 1.0)

        self.scheduler = DMD2EulerScheduler(
            num_train_timesteps=1000,
            shift=shift,
            dmd2_timesteps=dmd2_timesteps,
        )

    def _sanitize_dmd2_request(self, req: OmniDiffusionRequest) -> None:
        """Sanitize CFG-related fields in-place. Mutates req.sampling_params and req.prompts."""
        sp = req.sampling_params

        if sp.num_inference_steps and sp.num_inference_steps != self.num_inference_steps:
            logger.warning(
                "DMD2: ignoring num_inference_steps=%d, forcing %d.",
                sp.num_inference_steps,
                self.num_inference_steps,
            )
        sp.num_inference_steps = self.num_inference_steps

        if sp.guidance_scale_provided and sp.guidance_scale != self.dmd2_guidance_scale:
            logger.warning(
                "DMD2: ignoring guidance_scale=%.2f, forcing %.2f.",
                sp.guidance_scale,
                self.dmd2_guidance_scale,
            )
        sp.guidance_scale = self.dmd2_guidance_scale
        sp.guidance_scale_provided = False

        if sp.guidance_scale_2 is not None:
            logger.warning("DMD2: ignoring guidance_scale_2.")
            sp.guidance_scale_2 = None

        if sp.true_cfg_scale is not None:
            logger.warning("DMD2: ignoring true_cfg_scale.")
            sp.true_cfg_scale = None

        sp.do_classifier_free_guidance = False
        sp.is_cfg_negative = False

        fixed = []
        for p in req.prompts:
            if isinstance(p, dict) and "negative_prompt" in p:
                logger.warning("DMD2: ignoring negative_prompt.")
                p = {k: v for k, v in p.items() if k != "negative_prompt"}
            fixed.append(p)
        req.prompts = fixed

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._sanitize_dmd2_request(req)
        kwargs.pop("guidance_scale", None)
        kwargs.pop("num_inference_steps", None)
        return super().forward(
            req,
            guidance_scale=self.dmd2_guidance_scale,
            num_inference_steps=self.num_inference_steps,
            **kwargs,
        )
