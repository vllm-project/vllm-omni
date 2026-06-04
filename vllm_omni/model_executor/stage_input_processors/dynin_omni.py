from __future__ import annotations

import json
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


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


# ============================================================================
# Worker-connector data plane (non-async-chunk path).
# ============================================================================

# Per-model REPLACE-keys for the full-payload accumulator.  dynin_omni's
# producer model emits new chunks per step (token_ids / runtime_info_json),
# all of which use the default CONCAT/replace semantics — no model_outputs
# entry needs explicit REPLACE.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _build_full_payload(pooling_output: dict[str, Any] | None, request: Any) -> dict[str, Any] | None:
    """Producer-side payload builder: assemble dynin_omni connector payload.

    Reads token_ids from ``pooling_output["token_ids"]`` (preferred) or
    ``request.output_token_ids`` (fallback).  Reads structured non-tensor
    metadata from ``pooling_output["runtime_info_json"]`` (JSON-in-uint8)
    if present, falling back to ``pooling_output["runtime_info"]`` dict.
    Carries forward ``request.additional_information`` so prompt-side
    metadata (speaker / language / detok_id) survives the IPC boundary.
    """
    if not isinstance(pooling_output, dict):
        pooling_output = {}

    token_ids = _to_token_id_list(pooling_output.get("token_ids"))
    if not token_ids:
        token_ids = _to_token_id_list(pooling_output.get("text_tokens"))
    if not token_ids and request is not None:
        token_ids = _to_token_id_list(getattr(request, "output_token_ids", None))
    if not token_ids:
        logger.warning(
            "dynin_omni._build_full_payload: no token_ids found in pooling_output "
            "(keys=%s) or request.output_token_ids for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            getattr(request, "request_id", "?"),
        )
        return None

    src_additional_info = getattr(request, "additional_information", {}) if request is not None else {}
    if not isinstance(src_additional_info, dict):
        src_additional_info = {}

    runtime_bridge_info = _decode_runtime_bridge_info(pooling_output.get("runtime_info_json"))
    if not runtime_bridge_info:
        runtime_bridge_info = pooling_output.get("runtime_info", {}) or {}

    payload = _normalize_additional_info(src_additional_info)
    payload.update(_normalize_additional_info(runtime_bridge_info))
    payload["detok_id"] = [_to_int(pooling_output.get("detok_id"), default=_to_int(payload.get("detok_id"), default=0))]
    # Use nested OmniPayload shape so the scheduling-metadata extractor in
    # OmniConnectorModelRunnerMixin reads codes.audio and meta.finished
    # (flat keys at the top level are silently dropped with a warning).
    payload["codes"] = {"audio": token_ids}
    payload["meta"] = {"finished": torch.tensor(True, dtype=torch.bool)}
    return payload


def token2text_to_token2image_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Producer-side payload builder for the Stage-0 → Stage-1 (text → image) transition."""
    del transfer_manager
    return _build_full_payload(pooling_output, request)


def token2image_to_token2audio_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Producer-side payload builder for the Stage-1 → Stage-2 (image → audio) transition."""
    del transfer_manager
    return _build_full_payload(pooling_output, request)


def _token_only_from_source(source_outputs: list[Any]) -> list[OmniTokensPrompt]:
    """Length-only placeholder list mirroring ``_bridge_tokens`` token counts."""
    inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        mm_out = getattr(output, "multimodal_output", None) or {}
        token_ids = _to_token_id_list(mm_out.get("token_ids"))
        if not token_ids:
            token_ids = _to_token_id_list(mm_out.get("text_tokens"))
        if not token_ids:
            token_ids = list(getattr(output, "token_ids", []) or [])
        if not token_ids:
            token_ids = [0]
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * len(token_ids),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def token2text_to_token2image_token_only(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for Stage-1 input (token2image)."""
    source_stage_id = engine_input_source[0] if engine_input_source else 0
    source_outputs = stage_list[source_stage_id].engine_outputs
    return _token_only_from_source(source_outputs)


def token2image_to_token2audio_token_only(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for Stage-2 input (token2audio)."""
    source_stage_id = engine_input_source[0] if engine_input_source else 0
    source_outputs = stage_list[source_stage_id].engine_outputs
    return _token_only_from_source(source_outputs)
