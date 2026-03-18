"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
)

logger = init_logger(__name__)


def dtype_to_name(dtype: torch.dtype) -> str:
    """Convert torch dtype to a stable string name for serialization."""
    mapping = {
        torch.float32: "float32",
        torch.float: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.double: "float64",
        torch.int64: "int64",
        torch.long: "int64",
        torch.int32: "int32",
        torch.int: "int32",
        torch.int16: "int16",
        torch.short: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def serialize_additional_information(
    raw_info: dict[str, Any] | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    """Serialize omni request metadata for EngineCore transport."""
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info

    entries: dict[str, AdditionalInformationEntry] = {}
    for key, value in raw_info.items():
        if isinstance(value, torch.Tensor):
            value_cpu = value.detach().to("cpu").contiguous()
            entries[key] = AdditionalInformationEntry(
                tensor_data=value_cpu.numpy().tobytes(),
                tensor_shape=list(value_cpu.shape),
                tensor_dtype=dtype_to_name(value_cpu.dtype),
            )
            continue

        if isinstance(value, list):
            entries[key] = AdditionalInformationEntry(list_data=value)
            continue

        if log_prefix is None:
            logger.warning(
                "Dropping unsupported additional_information key=%s type=%s",
                key,
                type(value).__name__,
            )
        else:
            logger.warning(
                "[%s] Dropping unsupported additional_information key=%s type=%s",
                log_prefix,
                key,
                type(value).__name__,
            )

    return AdditionalInformationPayload(entries=entries) if entries else None
