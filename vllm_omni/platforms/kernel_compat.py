# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

try:
    from vllm.config.kernel import IrOpPriorityConfig
except ImportError:

    @dataclass
    class IrOpPriorityConfig:  # type: ignore[no-redef]
        """Compatibility fallback for vLLM versions without IrOpPriorityConfig."""

        default: list[str] = field(default_factory=list)
        rms_norm: list[str] | None = None
        fused_add_rms_norm: list[str] | None = None
        extra: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def with_default(cls, default: list[str], **kwargs: Any) -> "IrOpPriorityConfig":
            known = {
                "rms_norm": kwargs.pop("rms_norm", None),
                "fused_add_rms_norm": kwargs.pop("fused_add_rms_norm", None),
            }
            return cls(default=list(default), extra=dict(kwargs), **known)
