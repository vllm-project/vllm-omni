# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent, initialization-only Pi0.5 runtime optimization switches."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pi05RuntimeOptions:
    precompute_adarms: bool = False
    hoist_denoise_invariants: bool = False
    prefix_kv_only: bool = False

    def __post_init__(self):
        for name in ("precompute_adarms", "hoist_denoise_invariants", "prefix_kv_only"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"pi05_{name} must be a boolean, got {value!r}.")

    @classmethod
    def from_pipeline_args(cls, args: dict | None) -> "Pi05RuntimeOptions":
        args = args or {}
        return cls(
            precompute_adarms=args.get("pi05_precompute_adarms", False),
            hoist_denoise_invariants=args.get("pi05_hoist_denoise_invariants", False),
            prefix_kv_only=args.get("pi05_prefix_kv_only", False),
        )
