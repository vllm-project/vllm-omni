# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def parse_lora_request(lora_body: Any) -> tuple[LoRARequest | None, float | None]:
    """Parse a request-level LoRA object into a LoRARequest and optional scale.

    Raises:
        ValueError: If the object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None

    if not isinstance(lora_body, dict):
        raise ValueError("Invalid lora field: expected an object.")

    lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
    lora_path = (
        lora_body.get("local_path")
        or lora_body.get("path")
        or lora_body.get("lora_path")
        or lora_body.get("lora_local_path")
    )
    lora_scale = lora_body.get("scale")
    if lora_scale is None:
        lora_scale = lora_body.get("lora_scale")
    lora_int_id = lora_body.get("int_id")
    if lora_int_id is None:
        lora_int_id = lora_body.get("lora_int_id")
    if not lora_name or not lora_path:
        raise ValueError("Invalid lora object: both name and path are required.")

    if lora_int_id is None:
        lora_int_id = stable_lora_int_id(str(lora_path))

    scale = float(lora_scale) if lora_scale is not None else None
    return LoRARequest(str(lora_name), int(lora_int_id), str(lora_path)), scale


def parse_lora_requests(
    lora_body: Any,
) -> tuple[LoRARequest | list[LoRARequest] | None, float | list[float] | None]:
    """Parse a request-level LoRA object or array of objects.

    Accepts the legacy single-object shape (see ``parse_lora_request``) or a
    list of such objects to stack multiple adapters on one request. An empty
    list explicitly disables deployment-default request adapters.
    A single object (or a one-element array) keeps the legacy scalar
    ``(LoRARequest, float)`` shape; only a genuine stack of 2+ adapters
    produces lists. Per-adapter scales default to 1.0 when omitted.

    Raises:
        ValueError: If any object shape is invalid, required fields are missing,
            or the same adapter appears more than once in a stack.
    """
    if lora_body is None:
        return None, None

    items = lora_body if isinstance(lora_body, list) else [lora_body]
    if not items:
        return [], []

    lora_requests: list[LoRARequest] = []
    lora_scales: list[float] = []
    seen: set[int] = set()
    for item in items:
        lora_request, lora_scale = parse_lora_request(item)
        if lora_request is None:
            raise ValueError("Invalid lora field: array items must be objects.")
        if lora_request.lora_int_id in seen:
            raise ValueError(
                f"Duplicate LoRA adapter in stack: id={lora_request.lora_int_id}, name={lora_request.lora_name}"
            )
        seen.add(lora_request.lora_int_id)
        lora_requests.append(lora_request)
        lora_scales.append(lora_scale if lora_scale is not None else 1.0)
    if len(lora_requests) == 1:
        return lora_requests[0], lora_scales[0]
    return lora_requests, lora_scales
