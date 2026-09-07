# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned declaration for complete final-layout tensor restoration."""

from __future__ import annotations

from dataclasses import dataclass

FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA = "vllm-omni.diffusion.final-layout-tensors-v1"


@dataclass(frozen=True)
class FinalLayoutModelContract:
    """Versioned model ABI for reconstruction from exact final-layout tensors."""

    implementation_id: str
    version: str
    schema: str = FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("schema", self.schema),
            ("implementation_id", self.implementation_id),
            ("version", self.version),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"final-layout model contract {name} must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "implementation_id": self.implementation_id,
            "version": self.version,
        }


__all__ = [
    "FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA",
    "FinalLayoutModelContract",
]
