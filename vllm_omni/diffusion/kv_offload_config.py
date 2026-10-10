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
    skip_reference_projection: bool = False


def parse_reference_kv_config(value: dict[str, Any] | None) -> ReferenceKVConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kv_offload_config must be a JSON object")
    allowed = {"mode", "kv_refresh_interval", "kv_host_quantization", "skip_reference_projection"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown kv_offload_config keys: {sorted(unknown)}")
    mode = value.get("mode")
    if mode not in ("tier1", "tier2"):
        raise ValueError("kv_offload_config.mode must be tier1 or tier2")
    interval = value.get("kv_refresh_interval", 2)
    if type(interval) is not int or interval < 0:
        raise ValueError("kv_refresh_interval must be an integer >= 0")
    quantization = value.get("kv_host_quantization", "none")
    if quantization not in ("none", "fp8", "int8"):
        raise ValueError("kv_host_quantization must be none, fp8, or int8")
    if mode == "tier1" and quantization != "none":
        raise ValueError("kv_host_quantization requires mode=tier2")
    skip_projection = value.get("skip_reference_projection", False)
    if type(skip_projection) is not bool:
        raise ValueError("skip_reference_projection must be a JSON boolean")
    return ReferenceKVConfig(mode, interval, quantization, skip_projection)
