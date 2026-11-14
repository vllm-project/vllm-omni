# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build JoyAI inputs from Qwen3-ASR transcripts and Qwen3-TTS Talker inputs from JoyAI actions."""

from collections.abc import Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import (
    parse_action,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    QA_HISTORY_HEADER,
    USER_QUERY_HEADER,
    VIDEO_HISTORY_HEADER,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.sentence_tts import (
    JOYAI_TTS_CLOSE_ONLY_KEY,
    JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY,
    JOYAI_TTS_PARTIAL_KEY,
    sanitize_spoken_text,
)
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
    first_value,
)

logger = init_logger(__name__)

# Match the standalone Qwen3-TTS default attributes
_DEFAULT_TTS_TASK_TYPE = "CustomVoice"
_DEFAULT_TTS_LANGUAGE = "Auto"
_DEFAULT_TTS_SPEAKER = "Vivian"

# Qwen3-ASR emits ``language <Lang><asr_text><transcript>``.
_ASR_TEXT_TAG = "<asr_text>"
# JoyAI uses the Qwen3-VL chat template: one placeholder per visual input.
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _extract_completed_action_text(joyai_output: RequestOutput) -> str:
    """Return the final JoyAI action text, using cumulative text when available."""
    completion = joyai_output.outputs[0]
    action_text = getattr(completion, "cumulative_text", None) or completion.text
    return action_text if isinstance(action_text, str) else ""


def _clean_qwen3_asr_transcript(raw_text: str) -> str:
    """Drop the Qwen3-ASR language prefix and normalize whitespace."""
    return " ".join(raw_text.rsplit(_ASR_TEXT_TAG, 1)[-1].replace("</asr_text>", "").split())


def render_joyai_memory_prefix(
    summary: str | None,
    visual_summaries: Sequence[str],
    *,
    archived_qa: str = "",
    active_query: str = "",
) -> str:
    """Render model-owned memory with the same headings for text and speech."""
    prefix = []
    if visual_summaries:
        prefix.append(VIDEO_HISTORY_HEADER + "\n\n".join(visual_summaries))
    qa_memory = [text for text in (summary, archived_qa) if text]
    if qa_memory:
        prefix.append(QA_HISTORY_HEADER + "\n\n".join(qa_memory))
    if active_query:
        prefix.append(USER_QUERY_HEADER + "\n" + active_query)
    return "\n\n".join(prefix)


def render_joyai_session_prompt(
    processor: Any,
    *,
    system: str,
    history: list[dict[str, Any]],
    query: str,
    images: list[Any],
    image_token_counts: list[int],
    budget: int,
    instruction: str = "",
    summary: str = "",
    visual_summaries: Sequence[str] = (),
    context_blocks: Sequence[str] = (),
) -> tuple[str, list[Any]]:
    """Fit ASR's query, preserving the standing instruction and current input.

    Evict history, then the text summary, then oldest visual blocks, as on the
    text path. The standing instruction, actual transcript and delegated
    results stay pinned.
    """
    blocks = [block for block in context_blocks if block]
    history, images, image_token_counts = list(history), list(images), list(image_token_counts)
    recent_history = history.copy()
    visual_summaries = list(visual_summaries)
    rebudget_history = False
    while True:
        content = [{"type": "image"} for _ in images]
        content.extend({"type": "text", "text": block} for block in blocks)
        content.append(
            {
                "type": "text",
                "text": f"{USER_QUERY_HEADER}\n{query}"
                if query
                else "Observe the current scene and choose whether to speak or stay silent.",
            }
        )
        memory = render_joyai_memory_prefix(summary, visual_summaries)
        prefix = [{"role": "user", "content": value} for value in (memory, instruction) if value]
        messages = [{"role": "system", "content": system}, *prefix, *history, {"role": "user", "content": content}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_tokens = processor.tokenizer(text=[rendered])["input_ids"][0]
        if len(text_tokens) + sum(image_token_counts) - len(images) <= budget:
            if rebudget_history:
                history = recent_history.copy()
                rebudget_history = False
                continue
            return rendered, images
        if history:
            history.pop(0)
            while history and history[0]["role"] == "assistant":
                history.pop(0)
        elif summary:
            summary = ""
            rebudget_history = True
        elif visual_summaries:
            visual_summaries.pop(0)
            rebudget_history = True
        elif len(images) > 1:
            images.pop(0)
            image_token_counts.pop(0)
        else:
            raise OmniClientError("JoyAI instruction and current observation exceed the context budget")


def asr_to_joyai(
    source_outputs: list[RequestOutput],
    prompt: object | None = None,
    requires_multimodal_data: bool = True,
    *,
    target_model_config: Any = None,
) -> list[dict[str, Any]]:
    """Build JoyAI prompts from Qwen3-ASR transcripts and the deferred image/video inputs;
    ``additional_information["joyai_system_prompt"]`` overrides the default system prompt."""
    request_prompts = prompt if isinstance(prompt, list) else [prompt] * len(source_outputs)
    joyai_inputs: list[dict[str, Any]] = []
    for request_index, asr_output in enumerate(source_outputs):
        request_prompt = request_prompts[request_index] if request_index < len(request_prompts) else None
        request_prompt = request_prompt if isinstance(request_prompt, dict) else {}
        additional_info = request_prompt.get("additional_information") or {}
        system_prompt = str(first_value(additional_info.get("joyai_system_prompt"), DEFAULT_SYSTEM_PROMPT))
        delegation_context = additional_info.get("joyai_delegation_context")
        if delegation_context is not None and not isinstance(delegation_context, str):
            raise OmniClientError("joyai_delegation_context must be a string")
        transcript = _clean_qwen3_asr_transcript(_extract_completed_action_text(asr_output))
        deferred = additional_info.get("deferred_multi_modal_data") or {}
        visual_inputs = {modality: deferred[modality] for modality in ("image", "video") if deferred.get(modality)}
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
        context = additional_info.get("joyai_prompt_context")
        if "joyai_prompt_context" in additional_info:
            # Chat clients can also supply this metadata. Reject malformed
            # session context as a request error before invoking the processor.
            if (
                not isinstance(context, dict)
                or not isinstance(context.get("system"), str)
                or not isinstance(context.get("history"), list)
                or any(not isinstance(context.get(key, ""), str) for key in ("instruction", "summary"))
                or not isinstance(context.get("visual_summaries", []), list)
                or any(not isinstance(text, str) for text in context.get("visual_summaries", []))
                or any(
                    not isinstance(message, dict)
                    or message.get("role") not in ("user", "assistant")
                    or not isinstance(message.get("content"), str)
                    for message in context["history"]
                )
            ):
                raise OmniClientError(
                    "joyai_prompt_context requires a string system and a history of user/assistant text messages"
                )
            budget = additional_info.get("joyai_prompt_budget")
            if type(budget) is not int or budget <= 0:
                raise OmniClientError("joyai_prompt_budget must be a positive integer")
            image_token_counts = additional_info.get("joyai_image_token_counts")
            images = visual_inputs.get("image", [])
            if (
                not isinstance(image_token_counts, list)
                or not isinstance(images, list)
                or len(image_token_counts) != len(images)
                or any(type(count) is not int or count < 0 for count in image_token_counts)
            ):
                raise OmniClientError("joyai_image_token_counts must contain one non-negative integer per image")
            # ASR determines the query length. Apply the same history/visual
            # eviction policy as direct observations only once it is known.
            joyai_input["prompt"], images = render_joyai_session_prompt(
                cached_processor_from_config(target_model_config),
                system=context["system"],
                history=context["history"],
                query=transcript,
                images=images,
                image_token_counts=image_token_counts,
                budget=budget,
                instruction=context.get("instruction", ""),
                summary=context.get("summary", ""),
                visual_summaries=context.get("visual_summaries", []),
                context_blocks=(delegation_context,) if delegation_context else (),
            )
            visual_inputs = {"image": images} if images else {}
        if requires_multimodal_data and visual_inputs:
            joyai_input["multi_modal_data"] = visual_inputs
        if request_prompt.get("mm_processor_kwargs") is not None:
            joyai_input["mm_processor_kwargs"] = request_prompt["mm_processor_kwargs"]
        if request_prompt.get("cache_salt") is not None:
            joyai_input["cache_salt"] = request_prompt["cache_salt"]
        joyai_inputs.append(joyai_input)
    return joyai_inputs


def _talker_handoff_flags(prompt: object) -> tuple[bool, bool, bool]:
    """Read the sentence-TTS flags stamped by ``mark_talker_prompt_flags``."""
    if not isinstance(prompt, dict):
        return False, False, False
    raw_info = prompt.get("additional_information")
    if not isinstance(raw_info, dict):
        return False, False, False
    return (
        bool(raw_info.get(JOYAI_TTS_PARTIAL_KEY)),
        bool(raw_info.get(JOYAI_TTS_CLOSE_ONLY_KEY)),
        bool(raw_info.get(JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY)),
    )


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
        request_prompt = request_prompts[request_index] if request_index < len(request_prompts) else None
        partial, close_only, finish_on_next_stop = _talker_handoff_flags(request_prompt)
        if close_only:
            spoken_text = ""
        elif partial:
            # SentenceTtsOutput already holds the spoken sentence; do not
            # re-parse the JoyAI action or the remainder is collapsed back
            # into one Talker request.
            spoken_text = _extract_completed_action_text(joyai_output).strip()
            if not spoken_text:
                continue
        else:
            parsed_action = parse_action(_extract_completed_action_text(joyai_output))
            if not parsed_action.spoke or not parsed_action.text:
                continue
            # One-shot Talker (async_chunk=false) never goes through
            # spoken_payload. Strip the same prompt scaffolding so a leaked
            # USER_QUERY_HEADER / duration / trailing </silence> is not spoken.
            spoken_text = sanitize_spoken_text(parsed_action.text)
            if not spoken_text:
                continue

        tts_metadata = _build_tts_metadata(request_prompt, spoken_text)
        _validate_talker_speaker(tts_metadata, target_model_config)
        talker_prompt_length = 1 if close_only else _compute_talker_prompt_length(tts_metadata, target_model_config)
        # Sentence 2+ is submit_update on the same Talker id after 2150. The AR
        # scheduler's default is streaming-input append (keep old codec KV).
        # Each spoken sentence is a new utterance, so tell it to replace.
        if partial and not close_only:
            # Qwen3-TTS Talker does not set replace_runtime_additional_information
            # (Omni/standalone TTS still append). The GPU runner therefore merges
            # this payload onto sentence-1's buffer and would keep embed.prefill.
            # Empty list is not a 2-D tensor, so preprocess rebuilds embeds from
            # this sentence's text instead of slicing the previous utterance.
            tts_metadata["meta"] = {
                "replace_streaming_prompt": True,
                "replace_runtime_additional_information": True,
                "talker_prefill_offset": 0,
                "talker_text_offset": 0,
            }
            if finish_on_next_stop:
                tts_metadata["meta"][JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY] = True
            tts_metadata["embed"] = {"prefill": []}
            tts_metadata["hidden_states"] = {"trailing_text": []}
        logger.info(
            "[JoyAI] talker submit close_only=%s finish_on_next_stop=%s text_len=%d prompt_len=%d",
            close_only,
            finish_on_next_stop,
            len(spoken_text),
            talker_prompt_length,
        )
        talker_inputs.append(
            OmniTokensPrompt(
                # OmniTokensPrompt reserves vllm scheduler positions with zero IDs;
                # the Talker builds the actual prompt from tts_metadata.
                prompt_token_ids=[0] * talker_prompt_length,
                additional_information=tts_metadata,
                model_intermediate_buffer=dict(tts_metadata),
            )
        )

    return talker_inputs
