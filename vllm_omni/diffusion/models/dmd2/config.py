# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DMD2Config:
    """Inference-time contract for a FastGen DMD2-distilled checkpoint."""

    num_inference_steps: int = 4
    denoising_timesteps: list[int] | None = None
    solver: str = "ode"
    guidance_scale: float = 1.0

    @classmethod
    def from_model_index(cls, model_index: dict) -> DMD2Config:
        """Read the `dmd2_config` block from a model_index.json dict. Missing block → defaults."""
        block = model_index.get("dmd2_config", {})
        return cls(
            num_inference_steps=block.get("num_inference_steps", cls.num_inference_steps),
            denoising_timesteps=block.get("denoising_timesteps"),
            solver=block.get("solver", cls.solver),
            guidance_scale=block.get("guidance_scale", cls.guidance_scale),
        )

    def resolve_timesteps(self) -> list[int]:
        if self.denoising_timesteps is not None:
            return list(self.denoising_timesteps)
        # Get uniformly spaced timesteps from 999 to 0.
        ts = torch.linspace(999, 0, self.num_inference_steps + 1)[:-1]
        return ts.to(torch.int32).tolist()
