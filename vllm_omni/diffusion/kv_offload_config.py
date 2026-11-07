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
    cache_dit_policy: str = "refresh"
    max_refresh_interval: int | None = None


def parse_reference_kv_config(value: dict[str, Any] | None) -> ReferenceKVConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kv_offload_config must be a JSON object")
    allowed = {"mode", "kv_refresh_interval", "kv_host_quantization", "cache_dit_policy", "max_refresh_interval"}
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
    policy = value.get("cache_dit_policy", "refresh")
    if policy not in ("refresh", "hit"):
        raise ValueError("cache_dit_policy must be refresh or hit")
    maximum = value.get("max_refresh_interval")
    if policy == "hit":
        if interval == 0:
            raise ValueError("cache_dit_policy=hit requires kv_refresh_interval > 0")
        maximum = 2 * interval if maximum is None else maximum
        if type(maximum) is not int or maximum < interval:
            raise ValueError("max_refresh_interval must be an integer >= kv_refresh_interval")
    elif maximum is not None:
        raise ValueError("max_refresh_interval requires cache_dit_policy=hit")
    return ReferenceKVConfig(mode, interval, quantization, policy, maximum)
