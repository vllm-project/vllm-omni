# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build Qwen3-TTS Talker inputs from completed JoyAI actions."""

from typing import Any

from vllm.outputs import RequestOutput
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import (
    parse_action,
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


def _extract_completed_action_text(joyai_output: RequestOutput) -> str:
    """Return the final JoyAI action text, using cumulative text when available."""
    completion = joyai_output.outputs[0]
    action_text = getattr(completion, "cumulative_text", None) or completion.text
    return action_text if isinstance(action_text, str) else ""


def _build_tts_metadata(request_prompt: object, spoken_text: str) -> dict[str, list[str]]:
    """Build Qwen3-TTS metadata for the text JoyAI chose to speak."""
    prompt_dict = request_prompt if isinstance(request_prompt, dict) else {}
    raw_additional_info = prompt_dict.get("additional_information")
    additional_info = raw_additional_info if isinstance(raw_additional_info, dict) else {}

    task_type = first_value(additional_info.get("tts_task_type"), _DEFAULT_TTS_TASK_TYPE)
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
        parsed_action = parse_action(_extract_completed_action_text(joyai_output))
        if not parsed_action.spoke or not parsed_action.text:
            continue

        request_prompt = request_prompts[request_index] if request_index < len(request_prompts) else None
        tts_metadata = _build_tts_metadata(request_prompt, parsed_action.text)
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
