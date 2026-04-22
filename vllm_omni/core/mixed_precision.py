"""Automatic mixed precision for optimized inference."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class PrecisionType(Enum):
    """Precision types."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    AUTO = "auto"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision."""

    precision: PrecisionType = PrecisionType.AUTO
    enable_casting: bool = True
    layer_wise_precision: bool = False
    prefer_bf16: bool = True


class MixedPrecisionManager:
    """Manage automatic mixed precision for inference."""

    def __init__(self, config: MixedPrecisionConfig | None = None):
        self._config = config or MixedPrecisionConfig()
        self._current_precision: PrecisionType | None = None

    @property
    def config(self) -> MixedPrecisionConfig:
        return self._config

    @property
    def current_precision(self) -> PrecisionType | None:
        return self._current_precision

    def get_precision_for_layer(self, layer_name: str, default_precision: PrecisionType | None = None) -> PrecisionType:
        """Get precision for a specific layer."""
        if self._config.precision != PrecisionType.AUTO:
            return self._config.precision

        if self._config.layer_wise_precision:
            return self._determine_layer_precision(layer_name)

        return default_precision or self._get_default_precision()

    def _determine_layer_precision(self, layer_name: str) -> PrecisionType:
        """Determine best precision for layer type."""
        layer_lower = layer_name.lower()

        if "attention" in layer_lower or "mlp" in layer_lower:
            return PrecisionType.BF16 if self._config.prefer_bf16 else PrecisionType.FP16

        if "embedding" in layer_lower or "layernorm" in layer_lower:
            return PrecisionType.FP32

        if "head" in layer_lower or "output" in layer_lower:
            return PrecisionType.FP32

        return self._get_default_precision()

    def _get_default_precision(self) -> PrecisionType:
        """Get default precision based on config."""
        if self._config.prefer_bf16:
            return PrecisionType.BF16
        return PrecisionType.FP16

    def should_cast_input(self, from_precision: PrecisionType, to_precision: PrecisionType) -> bool:
        """Determine if casting is needed."""
        if not self._config.enable_casting:
            return False
        return from_precision != to_precision

    def get_precision_info(self) -> dict[str, Any]:
        """Get current precision information."""
        return {
            "configured_precision": self._config.precision.value,
            "current_precision": self._current_precision.value if self._current_precision else None,
            "layer_wise": self._config.layer_wise_precision,
            "prefer_bf16": self._config.prefer_bf16,
        }

    def set_precision(self, precision: PrecisionType) -> None:
        """Set current precision explicitly."""
        self._current_precision = precision
        logger.info(f"Precision set to: {precision.value}")
