# SPDX-License-Identifier: Apache-2.0
"""Target-specific Cosmos3-Nano-Sim-Bimanual action normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import (
    RANGE_FLOOR,
    ActionNormalizerContract,
    GlobalAsinhNormalizerContract,
    QuantileRotNormalizerContract,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionAffineNormalizer:
    """Unclamped affine transform, optionally followed by exported asinh compression."""

    offset: tuple[float, ...]
    scale: tuple[float, ...]
    transform_sha256: str
    asinh_unit: float | None = None

    @classmethod
    def from_contract(
        cls,
        contract: ActionNormalizerContract,
    ) -> ActionAffineNormalizer:
        if isinstance(contract, QuantileRotNormalizerContract):
            suspicious = [index for index, value in enumerate(contract.transform.scale) if value <= 100.0 * RANGE_FLOOR]
            if suspicious:
                logger.warning(
                    "Cosmos3-Nano-Sim-Bimanual normalizer %s has scales close to range_floor at channels %s.",
                    contract.transform_sha256,
                    suspicious,
                )
        return cls(
            offset=contract.transform.offset,
            scale=contract.transform.scale,
            transform_sha256=contract.transform_sha256,
            asinh_unit=contract.transform.unit if isinstance(contract, GlobalAsinhNormalizerContract) else None,
        )

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize in float32 without clamping out-of-range actions."""

        if action.shape[-1] != len(self.offset):
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual raw action dimension does not match the action contract: "
                f"{action.shape[-1]} != {len(self.offset)}."
            )
        action_f32 = action.to(dtype=torch.float32)
        if not torch.isfinite(action_f32).all():
            raise ValueError("Cosmos3-Nano-Sim-Bimanual raw actions must contain only finite values.")
        offset = action_f32.new_tensor(self.offset)
        scale = action_f32.new_tensor(self.scale)
        normalized = (action_f32 - offset) / scale
        if self.asinh_unit is not None:
            normalized = torch.asinh(normalized) / normalized.new_tensor(self.asinh_unit)
        if not torch.isfinite(normalized).all():
            raise ValueError("Cosmos3-Nano-Sim-Bimanual normalized actions must contain only finite values.")
        return normalized
