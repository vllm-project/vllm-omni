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
    kv_cache_location: str = "host"
    kv_cache_phase_contexts: bool = False


def parse_reference_kv_config(value: dict[str, Any] | None) -> ReferenceKVConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kv_offload_config must be a JSON object")
    allowed = {"mode", "kv_refresh_interval", "kv_host_quantization", "kv_cache_location", "kv_cache_phase_contexts"}
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
    location = value.get("kv_cache_location", "host")
    if location not in ("host", "device"):
        raise ValueError("kv_cache_location must be host or device")
    phase_contexts = value.get("kv_cache_phase_contexts", False)
    if type(phase_contexts) is not bool:
        raise ValueError("kv_cache_phase_contexts must be a boolean")
    return ReferenceKVConfig(mode, interval, quantization, location, phase_contexts)
