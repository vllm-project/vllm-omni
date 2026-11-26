# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validated, opt-in MiniMax-H3 reference-KV serving configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceKVConfig:
    mode: str
    kv_refresh_interval: int = 2
    kv_host_quantization: str = "none"


def parse_reference_kv_config(value: dict[str, Any] | None) -> ReferenceKVConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kv_offload_config must be a JSON object")
    allowed = {"mode", "kv_refresh_interval", "kv_host_quantization"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown kv_offload_config keys: {sorted(unknown)}")
    mode = value.get("mode")
    if mode != "on":
        raise ValueError("kv_offload_config.mode must be on")
    interval = value.get("kv_refresh_interval", 2)
    if type(interval) is not int or interval < 0:
        raise ValueError("kv_refresh_interval must be an integer >= 0")
    quantization = value.get("kv_host_quantization", "none")
    if quantization not in ("none", "fp8", "int8"):
        raise ValueError("kv_host_quantization must be none, fp8, or int8")
    return ReferenceKVConfig(mode, interval, quantization)


REFERENCE_ISOLATION = "minimax_h3_reference_isolation"
REFERENCE_ISOLATION_METRICS = "minimax_h3_isolation_metrics"
REFERENCE_ISOLATION_STATE = "reference_isolation"


def validate_kv_isolate(value: object) -> None:
    """Reject truthy strings or integers before serialized stage construction."""
    if value is not None and type(value) is not bool:
        raise ValueError("kv_isolate must be a boolean")
