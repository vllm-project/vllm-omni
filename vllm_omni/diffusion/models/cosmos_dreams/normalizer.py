# SPDX-License-Identifier: Apache-2.0
"""Target-specific Cosmos-Dreams action normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import (
    RANGE_FLOOR,
    ActionNormalizerContract,
    QuantileRotNormalizerContract,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionAffineNormalizer:
    """Unclamped affine transform from a validated action contract."""

    offset: tuple[float, ...]
    scale: tuple[float, ...]
    transform_sha256: str

    @classmethod
    def from_contract(
        cls,
        contract: ActionNormalizerContract,
    ) -> ActionAffineNormalizer:
        if isinstance(contract, QuantileRotNormalizerContract):
            suspicious = [index for index, value in enumerate(contract.transform.scale) if value <= 100.0 * RANGE_FLOOR]
            if suspicious:
                logger.warning(
                    "Cosmos-Dreams normalizer %s has scales close to range_floor at channels %s.",
                    contract.transform_sha256,
                    suspicious,
                )
        return cls(
            offset=contract.transform.offset,
            scale=contract.transform.scale,
            transform_sha256=contract.transform_sha256,
        )

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize in float32 without clamping out-of-range actions."""

        if action.shape[-1] != len(self.offset):
            raise ValueError(
                "Cosmos-Dreams raw action dimension does not match the action contract: "
                f"{action.shape[-1]} != {len(self.offset)}."
            )
        action_f32 = action.to(dtype=torch.float32)
        if not torch.isfinite(action_f32).all():
            raise ValueError("Cosmos-Dreams raw actions must contain only finite values.")
        offset = action_f32.new_tensor(self.offset)
        scale = action_f32.new_tensor(self.scale)
        normalized = (action_f32 - offset) / scale
        if not torch.isfinite(normalized).all():
            raise ValueError("Cosmos-Dreams normalized actions must contain only finite values.")
        return normalized
