# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI LoRA request helpers.

Use this module for HTTP-facing LoRA parsing shared by OpenAI-compatible
endpoint families."""

import json
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException

from vllm_omni.entrypoints.openai.utils import parse_lora_request


def _get_lora_from_json_str(lora_body):
    if lora_body is None:
        return None
    try:
        lora_dict = json.loads(lora_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid LoRA JSON string")

    if not isinstance(lora_dict, dict):
        raise HTTPException(status_code=400, detail="LoRA must be a JSON object")

    return lora_dict


def _parse_lora_request(lora_body: dict[str, Any]):
    try:
        return parse_lora_request(lora_body)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(e),
        ) from e
