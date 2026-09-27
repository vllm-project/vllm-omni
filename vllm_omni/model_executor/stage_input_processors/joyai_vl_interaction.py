# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processors of the native JoyAI-VL-Interaction pipelines.

``asr_to_joyai`` builds JoyAI inputs from Qwen3-ASR transcripts (audio-input
profile); ``joyai_action_to_tts`` builds Qwen3-TTS Talker inputs from
completed JoyAI actions.
"""

from typing import Any

from vllm.outputs import RequestOutput
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import (
    parse_action,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_QUERY_HEADER,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
    first_value,
)

# Match the standalone Qwen3-TTS default attributes
_DEFAULT_TTS_TASK_TYPE = "CustomVoice"
_DEFAULT_TTS_LANGUAGE = "Auto"
_DEFAULT_TTS_SPEAKER = "Vivian"

# Qwen3-ASR emits ``language <Lang><asr_text><transcript>``.
_ASR_TEXT_TAG = "<asr_text>"
# JoyAI uses the Qwen3-VL chat template: one placeholder per visual input.
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _extract_completed_text(stage_output: RequestOutput) -> str:
    """Return a completed stage's text, using cumulative text when available."""
    completion = stage_output.outputs[0]
    text = getattr(completion, "cumulative_text", None) or completion.text
    return text if isinstance(text, str) else ""


def _clean_qwen3_asr_transcript(raw_text: str) -> str:
    """Drop the Qwen3-ASR language prefix and normalize whitespace."""
    return " ".join(raw_text.rsplit(_ASR_TEXT_TAG, 1)[-1].replace("</asr_text>", "").split())


def _visual_inputs(request_prompt: dict[str, Any]) -> dict[str, list[Any]]:
    """Return the image/video inputs the frontend deferred past the ASR stage."""
    additional_info = request_prompt.get("additional_information") or {}
    media = dict(request_prompt.get("multi_modal_data") or {})
    media.update(additional_info.get("deferred_multi_modal_data") or {})
    return {
        modality: items if isinstance(items, list) else [items]
        for modality in ("image", "video")
        if (items := media.get(modality)) is not None
    }


def asr_to_joyai(
    source_outputs: list[RequestOutput],
    prompt: object | None = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build JoyAI inputs from Qwen3-ASR transcripts and the original visual inputs.

    The transcript becomes JoyAI's user query (laid out as the Day-0 interaction
    server does), followed by the deferred image/video inputs. The system prompt
    is JoyAI's unless ``additional_information["joyai_system_prompt"]`` is set.
    """
    request_prompts = prompt if isinstance(prompt, list) else [prompt] * len(source_outputs)
    joyai_inputs: list[dict[str, Any]] = []
    for request_index, asr_output in enumerate(source_outputs):
        request_prompt = request_prompts[request_index] if request_index < len(request_prompts) else None
        request_prompt = request_prompt if isinstance(request_prompt, dict) else {}
        additional_info = request_prompt.get("additional_information") or {}
        system_prompt = str(first_value(additional_info.get("joyai_system_prompt"), DEFAULT_SYSTEM_PROMPT))
        transcript = _clean_qwen3_asr_transcript(_extract_completed_text(asr_output))
        visual_inputs = _visual_inputs(request_prompt)
        user_turn = f"{USER_QUERY_HEADER}\n{transcript}" if transcript else ""
        user_turn += _IMAGE_PLACEHOLDER * len(visual_inputs.get("image", ()))
        user_turn += _VIDEO_PLACEHOLDER * len(visual_inputs.get("video", ()))
        joyai_input: dict[str, Any] = {
            "prompt": (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_turn}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        }
        if requires_multimodal_data and visual_inputs:
            joyai_input["multi_modal_data"] = visual_inputs
        if request_prompt.get("mm_processor_kwargs") is not None:
            joyai_input["mm_processor_kwargs"] = request_prompt["mm_processor_kwargs"]
        joyai_inputs.append(joyai_input)
    return joyai_inputs


def _build_tts_metadata(request_prompt: object, spoken_text: str) -> dict[str, list[str]]:
    """Build Qwen3-TTS metadata for the text JoyAI chose to speak."""
    prompt_dict = request_prompt if isinstance(request_prompt, dict) else {}
    raw_additional_info = prompt_dict.get("additional_information")
    additional_info = raw_additional_info if isinstance(raw_additional_info, dict) else {}

    # The native pipeline loads CustomVoice checkpoints for both TTS stages.
    # Keep client metadata from selecting an incompatible Qwen3-TTS task.
    task_type = _DEFAULT_TTS_TASK_TYPE
    request_language = first_value(additional_info.get("language"), _DEFAULT_TTS_LANGUAGE)
    language = first_value(additional_info.get("tts_language"), request_language)
    request_speaker = first_value(additional_info.get("speaker"), _DEFAULT_TTS_SPEAKER)
    speaker = first_value(additional_info.get("tts_speaker"), request_speaker)
    instruction = first_value(additional_info.get("tts_instruct"), "")

    return {
        "task_type": [str(task_type)],
        "language": [str(language)],
        "speaker": [str(speaker)],
        "instruct": [str(instruction)],
        "text": [spoken_text],
    }


def _validate_talker_speaker(tts_metadata: dict[str, list[str]], talker_model_config: Any) -> None:
    """Validate the selected voice using the destination Talker's speaker table."""
    talker_config = getattr(talker_model_config.hf_config, "talker_config", None)
    if talker_config is None:
        raise ValueError("The target stage is not a Qwen3-TTS Talker.")

    # Match CustomVoice prompt construction, including case and whitespace.
    speaker = str(first_value(tts_metadata.get("speaker"), "")).lower().strip()
    speakers = {name.lower() for name in (getattr(talker_config, "spk_id", None) or {})}
    if not speaker or speaker not in speakers:
        raise OmniClientError(f"Unsupported speaker: {speaker!r}")


def _compute_talker_prompt_length(
    tts_metadata: dict[str, list[str]],
    talker_model_config: Any,
) -> int:
    """Compute the exact number of prompt positions used by the Talker."""
    task_type = str(first_value(tts_metadata.get("task_type"), _DEFAULT_TTS_TASK_TYPE))
    if task_type != _DEFAULT_TTS_TASK_TYPE:
        raise ValueError("JoyAI native TTS supports only CustomVoice.")

    talker_tokenizer = cached_tokenizer_from_config(talker_model_config)
    if talker_tokenizer is None:
        raise ValueError("The Qwen3-TTS Talker must have an initialized tokenizer.")

    talker_config = getattr(talker_model_config.hf_config, "talker_config", None)
    if talker_config is None:
        raise ValueError("The target stage is not a Qwen3-TTS Talker.")

    def tokenize_prompt(text: str) -> list[int]:
        return talker_tokenizer.encode(text, add_special_tokens=True)

    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=tts_metadata,
        task_type=task_type,
        tokenize_prompt=tokenize_prompt,
        codec_language_id=getattr(talker_config, "codec_language_id", None),
        spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
    )


def joyai_action_to_tts(
    source_outputs: list[RequestOutput],
    prompt: object | None = None,
    requires_multimodal_data: bool = False,
    *,
    target_model_config: Any,
) -> list[OmniTokensPrompt]:
    """Build Talker inputs for JoyAI actions that contain text to speak."""
    del requires_multimodal_data

    request_prompts = prompt if isinstance(prompt, list) else [prompt] * len(source_outputs)
    talker_inputs: list[OmniTokensPrompt] = []
    for request_index, joyai_output in enumerate(source_outputs):
        parsed_action = parse_action(_extract_completed_text(joyai_output))
        if not parsed_action.spoke or not parsed_action.text:
            continue

        request_prompt = request_prompts[request_index] if request_index < len(request_prompts) else None
        tts_metadata = _build_tts_metadata(request_prompt, parsed_action.text)
        _validate_talker_speaker(tts_metadata, target_model_config)
        talker_prompt_length = _compute_talker_prompt_length(tts_metadata, target_model_config)
        talker_inputs.append(
            OmniTokensPrompt(
                # OmniTokensPrompt reserves vllm scheduler positions with zero IDs;
                # the Talker builds the actual prompt from tts_metadata.
                prompt_token_ids=[0] * talker_prompt_length,
                additional_information=tts_metadata,
            )
        )

    return talker_inputs
