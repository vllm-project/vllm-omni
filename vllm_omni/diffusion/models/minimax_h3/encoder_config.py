# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deployment-owned MiniMax-H3 encoder weight-transfer policy."""

from collections.abc import Mapping
from enum import Enum


class EncoderWeightTransfer(str, Enum):
    """Optimization modes for the manually offloaded NPU encoder."""

    OFF = "off"
    PACKED = "packed"
    PACKED_PREFETCH = "packed_prefetch"


def encoder_weight_transfer(additional_config: Mapping[str, object] | None) -> EncoderWeightTransfer:
    """Resolve the deployment mode once; requests do not change weight storage."""
    value = (additional_config or {}).get("encoder_weight_transfer", EncoderWeightTransfer.OFF)
    if not isinstance(value, str):
        raise TypeError("additional_config.encoder_weight_transfer must be a string")
    try:
        return EncoderWeightTransfer(value)
    except ValueError as exc:
        raise ValueError("additional_config.encoder_weight_transfer must be off, packed, or packed_prefetch") from exc
