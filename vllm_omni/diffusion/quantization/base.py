# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for diffusion model quantization configurations."""

from abc import ABC
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )


class DiffusionQuantizationConfig(ABC):
    """Base class for diffusion model quantization configurations.

    This provides a thin wrapper over vLLM's quantization configs,
    allowing diffusion-model-specific defaults and future extensibility.

    Subclasses should:
        - Set quant_config_cls to the vLLM QuantizationConfig class
        - Call super().__init__() after creating self._vllm_config
        - Optionally override get_name() and get_min_capability() if needed
    """

    # Subclasses should set this to the vLLM QuantizationConfig class
    quant_config_cls: ClassVar[type["QuantizationConfig"] | None] = None

    # The underlying vLLM config instance
    _vllm_config: "QuantizationConfig | None" = None

    @classmethod
    def get_name(cls) -> str:
        """Return the quantization method name (e.g., 'fp8', 'int8').

        By default, delegates to the underlying vLLM config class.
        """
        if cls.quant_config_cls is not None:
            return cls.quant_config_cls.get_name()
        raise NotImplementedError("Subclass must set quant_config_cls or override get_name()")

    def get_vllm_quant_config(self) -> "QuantizationConfig | None":
        """Return the underlying vLLM QuantizationConfig for linear layers."""
        return self._vllm_config

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """Return supported activation dtypes."""
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU compute capability required.

        By default, delegates to the underlying vLLM config class.
        """
        if cls.quant_config_cls is not None:
            return cls.quant_config_cls.get_min_capability()
        return 80  # Ampere default
