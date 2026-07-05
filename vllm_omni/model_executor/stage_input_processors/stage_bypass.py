# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic optional stage-skip metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

OMNI_SKIP_STAGES_KEY = "omni_skip_stages"


def _parse_skip_stage_ids(value: Any) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset([value])
    if isinstance(value, list):
        stage_ids: list[int] = []
        for item in value:
            if isinstance(item, int):
                stage_ids.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                stage_ids.append(int(item.strip()))
        return frozenset(stage_ids)
    return frozenset()


def should_skip_stage(prompt: Any, stage_id: int) -> bool:
    """Return True when ``additional_information.omni_skip_stages`` includes ``stage_id``."""
    if not isinstance(prompt, dict):
        return False
    additional_info = prompt.get("additional_information")
    return should_skip_stage_from_info(additional_info, stage_id)


def should_skip_stage_from_info(additional_info: Any, stage_id: int) -> bool:
    """Return True when a per-request ``additional_information`` dict requests stage skip."""
    if not isinstance(additional_info, dict):
        return False
    return stage_id in _parse_skip_stage_ids(additional_info.get(OMNI_SKIP_STAGES_KEY))


def make_mock_text_stage_output(request_id: str, text: str = "", *, finished: bool = True) -> Any:
    """Synthetic text-stage output used when an upstream stage is bypassed."""
    output = SimpleNamespace(
        text=text,
        cumulative_text=text,
        cumulative_token_ids=[],
        multimodal_output={},
        finished=finished,
    )
    return SimpleNamespace(
        request_id=request_id,
        outputs=[output],
        finished=finished,
    )
