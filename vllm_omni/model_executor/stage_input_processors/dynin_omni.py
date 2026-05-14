from __future__ import annotations

import json
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def _to_prompt_dict(prompt_item: OmniTokensPrompt | TextPrompt | str | None) -> dict[str, Any]:
    if isinstance(prompt_item, dict):
        return prompt_item
    return {}


def _to_token_id_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu")
        if value.ndim == 0:
            return [int(value.item())]
        if value.ndim > 1:
            value = value[0]
        return [int(x) for x in value.tolist()]
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            return [int(x) for x in value[0]]
        return [int(x) for x in value]
    if value is None:
        return []
    return [int(value)]


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.view(-1)[0].item())
    if isinstance(value, list):
        if not value:
            return default
        return int(value[0])
    if value is None:
        return default
    return int(value)


def _normalize_additional_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, val in value.items():
        if isinstance(val, list):
            normalized[key] = val
        else:
            normalized[key] = [val]
    return normalized


def _decode_runtime_bridge_info(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").reshape(-1).to(torch.uint8)
        raw = bytes(tensor.tolist())
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, list):
        try:
            raw = bytes(int(item) for item in value)
        except Exception:
            return {}
    elif value is None:
        return {}
    else:
        return value if isinstance(value, dict) else {}

    if not raw:
        return {}

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _bridge_tokens(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    next_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]

    prompt_meta_by_reqid = {src_out.request_id: _to_prompt_dict(p) for src_out, p in zip(source_outputs, prompt)}

    for source_output in source_outputs:
        output = source_output.outputs[0]
        mm_out = getattr(output, "multimodal_output", None) or {}

        token_ids = _to_token_id_list(mm_out.get("token_ids"))
        if not token_ids:
            token_ids = _to_token_id_list(mm_out.get("text_tokens"))
        if not token_ids:
            token_ids = list(output.cumulative_token_ids or [])
        if not token_ids:
            raise RuntimeError(f"Stage output for request {source_output.request_id} has no token_ids")

        detok_id = _to_int(mm_out.get("detok_id"), default=0)
        src_prompt = prompt_meta_by_reqid.get(source_output.request_id, {})
        src_additional_info = src_prompt.get("additional_information", {}) or {}
        runtime_bridge_info = _decode_runtime_bridge_info(mm_out.get("runtime_info_json"))
        if not runtime_bridge_info:
            runtime_bridge_info = mm_out.get("runtime_info", {}) or {}

        additional_information: dict[str, Any] = _normalize_additional_info(src_additional_info)
        additional_information.update(_normalize_additional_info(runtime_bridge_info))
        additional_information["detok_id"] = [detok_id]

        next_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                additional_information=additional_information,
                multi_modal_data=(src_prompt.get("multi_modal_data") if requires_multimodal_data else None),
                mm_processor_kwargs=None,
            )
        )

    return next_inputs


def token2text_to_token2image(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    return _bridge_tokens(source_outputs, prompt, requires_multimodal_data)


def token2image_to_token2audio(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    return _bridge_tokens(source_outputs, prompt, requires_multimodal_data)
