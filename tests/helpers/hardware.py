# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime device labels shared by accuracy and perf result files.

Kept out of ``tests.dfx.conftest`` so a client script can record hardware
without importing pytest or the DFX server fixtures.
"""

from __future__ import annotations

import re
from typing import Any

# Longer / more specific tokens must appear before shorter substrings
# (e.g. ``H200`` before ``H20``, ``910B4`` before ``910``).
_RUNTIME_DEVICE_ALIASES: tuple[str, ...] = (
    "H100",
    "H800",
    "H200",
    "H20",
    "B200",
    "GB200",
    "L40S",
    "L40",
    "L4",
    "A100",
    "A800",
    "A10G",
    "A10",
    "A30",
    "MI325",
    "MI300",
    "MI250",
    "B60",
    "S5000",
    "910B4",
    "910B",
    "910",
    "310P",
    "A2",
    "A3",
)
_RUNTIME_RESOURCE_LABEL: str | None = None


def _safe_filename_token(value: Any | None, *, default: str = "na") -> str:
    """Make a single path segment safe for result filenames on common filesystems."""
    if value is None:
        return default
    text = str(value).strip()
    for bad in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        text = text.replace(bad, "_")
    return text if text else default


def _normalize_runtime_device_label(raw: str) -> str:
    """Map a platform device name to a short filename-safe resource token."""
    if not raw or not str(raw).strip():
        return "na"
    upper = str(raw).upper()
    for token in _RUNTIME_DEVICE_ALIASES:
        if token.upper() in upper:
            return _safe_filename_token(token)
    compact = re.sub(r"[^a-zA-Z0-9]+", "", str(raw))
    for prefix in ("NVIDIA", "AMD", "ASCEND", "HUAWEI"):
        if compact.upper().startswith(prefix):
            compact = compact[len(prefix) :]
            break
    return _safe_filename_token(compact[:48]) if compact else "na"


def _read_runtime_device_name(*, device_id: int = 0) -> str | None:
    """Device name from the active Omni platform."""
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.device_count() <= device_id:
        return None
    get_name = getattr(current_omni_platform, "get_device_name", None)
    if not callable(get_name):
        return None
    raw = get_name(device_id)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def get_runtime_resource_label(*, device_id: int = 0, refresh: bool = False) -> str:
    """Return a filename-safe hardware label detected on the running machine."""
    global _RUNTIME_RESOURCE_LABEL
    if not refresh and _RUNTIME_RESOURCE_LABEL is not None:
        return _RUNTIME_RESOURCE_LABEL
    raw = _read_runtime_device_name(device_id=device_id)
    label = _normalize_runtime_device_label(raw) if raw else "na"
    if not refresh:
        _RUNTIME_RESOURCE_LABEL = label
    return label


def hardware_json_value(resource_label: str | None) -> str:
    """Hardware token stored in result JSON (empty when unknown)."""
    token = _safe_filename_token(resource_label)
    return "" if token == "na" else token
