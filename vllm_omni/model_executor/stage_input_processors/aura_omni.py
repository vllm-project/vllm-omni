# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage processors for the AURA Omni pipeline."""

from __future__ import annotations

from typing import Any

from vllm_omni.inputs.data import OmniTokensPrompt

DEFAULT_AURA_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
)

SILENT_TEXT = "<|silent|>"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_prompt_dict(prompt_item: Any) -> dict[str, Any]:
    return prompt_item if isinstance(prompt_item, dict) else {}


def _first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _extract_output(source_output: Any) -> Any:
    outputs = getattr(source_output, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    return source_output


def _extract_text(source_output: Any) -> str:
    output = _extract_output(source_output)
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    mm = getattr(output, "multimodal_output", None)
    if isinstance(mm, dict):
        for key in ("text", "transcript", "asr_text"):
            value = mm.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    return ""


def _source_prompt_by_request_id(source_outputs: list[Any], prompt: Any) -> dict[str, dict[str, Any]]:
    prompts = _as_list(prompt)
    return {
        str(getattr(source_output, "request_id", idx)): _as_prompt_dict(prompt_item)
        for idx, (source_output, prompt_item) in enumerate(zip(source_outputs, prompts))
    }


def _vision_placeholder(multi_modal_data: dict[str, Any]) -> str:
    if "video" in multi_modal_data:
        return "<|vision_start|><|video_pad|><|vision_end|>"
    if "image" in multi_modal_data:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return ""


def _aura_prompt(system_prompt: str, transcript: str, multi_modal_data: dict[str, Any]) -> str:
    vision = _vision_placeholder(multi_modal_data)
    query = transcript.strip()
    user_body = f"{vision}{query}" if query else vision
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def asr2aura(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build AURA Qwen3-VL prompts from ASR transcripts and original video payloads."""
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []
    for idx, source_output in enumerate(source_outputs):
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        system_prompt = _first_value(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT)
        transcript = _extract_text(source_output)
        multi_modal_data = src_prompt.get("multi_modal_data") or {}
        if not isinstance(multi_modal_data, dict):
            multi_modal_data = {}

        next_input: dict[str, Any] = {
            "prompt": _aura_prompt(str(system_prompt), transcript, multi_modal_data),
        }
        if requires_multimodal_data:
            next_input["multi_modal_data"] = multi_modal_data
        if src_prompt.get("mm_processor_kwargs") is not None:
            next_input["mm_processor_kwargs"] = src_prompt.get("mm_processor_kwargs")
        next_inputs.append(next_input)
    return next_inputs


def _estimate_tts_prompt_len(text: str) -> int:
    # Qwen3-TTS replaces placeholder token embeddings during prefill.  A safe
    # overestimate avoids underfilling the prefill span; the model pads surplus
    # rows with its TTS pad embedding.
    return max(32, min(4096, len(text) + 64))


def aura2tts(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert AURA text output into Qwen3-TTS Talker requests."""
    del requires_multimodal_data
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[OmniTokensPrompt] = []
    for idx, source_output in enumerate(source_outputs):
        text = _extract_text(source_output).strip()
        if not text or text == SILENT_TEXT:
            continue

        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        tts_info = {
            "task_type": [_first_value(additional_info.get("tts_task_type"), "CustomVoice")],
            "text": [text],
            "language": [_first_value(additional_info.get("tts_language"), "Chinese")],
            "speaker": [_first_value(additional_info.get("tts_speaker"), "Vivian")],
            "instruct": [_first_value(additional_info.get("tts_instruct"), "")],
            "max_new_tokens": [int(_first_value(additional_info.get("tts_max_new_tokens"), 2048))],
        }
        next_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * _estimate_tts_prompt_len(text),
                additional_information=tts_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return next_inputs
