# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-info helpers shared by the thinker and talker AR stages."""

from __future__ import annotations

from typing import Any


def merge_runtime_info(info_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``additional_information`` into the info dict (top-level wins)."""
    additional = info_dict.get("additional_information")
    if isinstance(additional, dict):
        merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
        for k, v in additional.items():
            merged.setdefault(k, v)
        return merged
    return info_dict


def require_request_id(info: dict[str, Any], stage: str) -> str:
    """Resolve the request id from the runtime info (or its meta), or fail."""
    request_id = info.get("request_id")
    if request_id is None:
        meta = info.get("meta")
        if isinstance(meta, dict):
            request_id = meta.get("request_id")
    if request_id is None:
        raise RuntimeError(f"NemotronVoiceChat {stage} requires a request_id in the runtime info.")
    return str(request_id)
