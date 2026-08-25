# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration parsing and denoising-step policy for mixed precision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

ReasonerPolicy = Literal["native", "a16"]

_REASONER_POLICIES = frozenset({"native", "a16"})
_CONFIG_FIELDS = frozenset({"first_steps", "last_steps", "reasoner"})


@dataclass(frozen=True)
class Cosmos3MixedPrecisionConfig:
    """Validated precision policy shared by every quantization strategy."""

    first_steps: int = 3
    last_steps: int = 3
    reasoner: ReasonerPolicy = "a16"

    @classmethod
    def from_additional_config(
        cls,
        additional_config: Mapping[str, object] | None,
    ) -> Cosmos3MixedPrecisionConfig | None:
        """Parse Cosmos3 fields from vLLM-Omni's additional configuration."""
        values = additional_config or {}
        if "cosmos3_mixed_precision" not in values:
            return None
        raw_config = values["cosmos3_mixed_precision"]
        if not isinstance(raw_config, Mapping):
            raise TypeError("cosmos3_mixed_precision must be a mapping")
        unknown = set(raw_config) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"Unknown cosmos3_mixed_precision fields: {sorted(unknown)}")

        first_steps = _non_negative_int(
            raw_config.get("first_steps", 3),
            "cosmos3_mixed_precision.first_steps",
        )
        last_steps = _non_negative_int(
            raw_config.get("last_steps", 3),
            "cosmos3_mixed_precision.last_steps",
        )

        reasoner = str(raw_config.get("reasoner", "a16")).lower()
        if reasoner not in _REASONER_POLICIES:
            raise ValueError(
                "cosmos3_mixed_precision.reasoner must be one of "
                f"{sorted(_REASONER_POLICIES)}, got {reasoner!r}"
            )
        config = cls(
            first_steps=first_steps,
            last_steps=last_steps,
            reasoner=cast(ReasonerPolicy, reasoner),
        )
        if first_steps == 0 and last_steps == 0 and reasoner == "native":
            return None
        return config

    def use_high_precision(self, step_index: int, num_steps: int) -> bool:
        """Select the high path for an actual scheduler-step index."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        return step_index < self.first_steps or step_index >= num_steps - self.last_steps


def _non_negative_int(value: object, name: str) -> int:
    """Validate an integer configuration field without accepting booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value
