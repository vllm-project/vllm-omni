# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI LoRA request helpers.

Use this module for HTTP-facing LoRA parsing shared by OpenAI-compatible
endpoint families."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from vllm_omni.entrypoints.openai.utils import parse_lora_request

if TYPE_CHECKING:
    from vllm.entrypoints.openai.models.protocol import LoRAModulePath


def build_diffusion_lora_registry(lora_modules: Sequence[LoRAModulePath] | None) -> dict[str, str]:
    """Register --lora-modules names without loading or activating adapters."""
    registry: dict[str, str] = {}
    for module in lora_modules or ():
        if not module.name or not module.path:
            raise ValueError("Diffusion LoRA registration requires a name and path.")
        if module.name in registry:
            raise ValueError(f"Duplicate diffusion LoRA name: '{module.name}'.")
        registry[module.name] = module.path
    return registry


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


def _parse_lora_request(lora_body: dict[str, Any] | None, lora_modules: Mapping[str, str] | None = None):
    try:
        return parse_lora_request(lora_body, lora_modules)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(e),
        ) from e
