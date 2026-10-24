# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in vision precision experiments for Pi0.5, fixed before graph capture."""

from dataclasses import dataclass

import torch
from torch import nn

VISION_PRECISIONS = ("fp32", "bf16_autocast", "selective_bf16")


@dataclass(frozen=True)
class Pi05VisionOptions:
    precision: str = "fp32"
    batch_cameras: bool = False

    def __post_init__(self):
        if self.precision not in VISION_PRECISIONS:
            raise ValueError(f"pi05_vision_precision must be one of {VISION_PRECISIONS}, got {self.precision!r}.")
        if not isinstance(self.batch_cameras, bool):
            raise ValueError(f"pi05_batch_cameras must be a boolean, got {self.batch_cameras!r}.")

    @classmethod
    def from_pipeline_args(cls, args: dict | None) -> "Pi05VisionOptions":
        args = args or {}
        return cls(
            precision=args.get("pi05_vision_precision", "fp32"),
            batch_cameras=args.get("pi05_batch_cameras", False),
        )


def _cast_vision_embeddings(_module, _inputs, output):
    # OpenPI's patched SigLIP makes this cast immediately before its encoder.
    # A hook preserves the stock module tree and checkpoint parameter names.
    # The patch embedding transpose otherwise makes each encoder LayerNorm
    # copy its input. Materialize the token-major layout once at this boundary.
    return output.to(torch.bfloat16).contiguous()


def configure_vision_precision(paligemma: nn.Module, precision: str) -> None:
    """Set only vision/projector dtypes, before loading checkpoint tensors.

    The caller must invoke this once per model. FP32 retained weights must never
    be loaded into BF16 parameters and subsequently upcast.
    """
    tower = paligemma.model.vision_tower
    projector = paligemma.model.multi_modal_projector
    if precision != "selective_bf16":
        tower.to(dtype=torch.float32)
        projector.to(dtype=torch.float32)
        return

    # transformers <=5.3 wraps the vision transformer; >=5.4 can flatten it.
    vision = getattr(tower, "vision_model", tower)
    embeddings = vision.embeddings
    tower.to(dtype=torch.bfloat16)
    projector.to(dtype=torch.bfloat16)
    embeddings.patch_embedding.to(dtype=torch.float32)
    embeddings.position_embedding.to(dtype=torch.float32)
    embeddings.register_forward_hook(_cast_vision_embeddings)
