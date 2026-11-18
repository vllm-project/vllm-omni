# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serialize per-request duplex capture payloads onto the engine's bytes carrier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.protocol.duplex.action_records import encode_unit_payload


def encode_action_capture_entries(entries: Mapping[str, Any]) -> dict[str, bytes]:
    """Serialize per-request capture payloads for the bytes carrier."""
    payload: dict[str, bytes] = {}
    for req_id, unit_payload in entries.items():
        if isinstance(req_id, str) and isinstance(unit_payload, Mapping):
            payload[req_id] = encode_unit_payload(dict(unit_payload))
    return payload


__all__ = ["encode_action_capture_entries"]
