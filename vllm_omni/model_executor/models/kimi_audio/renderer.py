# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render Kimi's model task independently of HTTP output-stage selection."""

from functools import cached_property

import numpy as np
from vllm.entrypoints.chat_utils import parse_chat_messages, parse_chat_messages_async
from vllm.exceptions import VLLMClientError
from vllm.multimodal.audio import resample_audio_pyav
from vllm.renderers import BaseRenderer
from vllm.utils.async_utils import make_async

from vllm_omni.errors import OmniClientError

from .audio_processing import SAMPLE_RATE, prepare_kimi_audio_inputs
from .prompt import KimiAudioPromptBuilder


class KimiAudioRenderer(BaseRenderer):
    @cached_property
    def prompt_builder(self):
        return KimiAudioPromptBuilder.from_tokenizer(self.get_tokenizer(), self.model_config.hf_config)

    def _validate_messages(self, messages, params):
        kwargs = params.chat_template_kwargs
        if "modalities" in kwargs:
            raise ValueError(
                "Kimi-Audio uses chat_template_kwargs.output_type ('text' or 'both') for the model task; "
                "modalities belongs at the request's top level for output selection"
            )
        output_type = kwargs.get("output_type", "both")
        if output_type not in ("text", "both"):
            raise ValueError("Kimi-Audio chat_template_kwargs.output_type must be 'text' or 'both'")
        supported_kwargs = {
            "output_type",
            "add_generation_prompt",
            "continue_final_message",
            "add_special_tokens",
            "tokenize",
            "tools",
            "documents",
            "reasoning_effort",
            "enable_thinking",
        }
        if unknown := kwargs.keys() - supported_kwargs:
            raise ValueError(f"Unsupported Kimi-Audio chat_template_kwargs: {', '.join(sorted(unknown))}")
        if params.chat_template is not None:
            raise ValueError("Kimi-Audio uses its native prompt format; custom chat templates are not supported")
        if params.mm_processor_kwargs:
            raise ValueError("Kimi-Audio does not support multimodal processor overrides")
        if kwargs.get("reasoning_effort") not in (None, "none") or kwargs.get("enable_thinking"):
            raise ValueError("Kimi-Audio does not support a separate reasoning mode")
        response_format = params.response_format
        if response_format is not None:
            format_type = (
                response_format.get("type")
                if isinstance(response_format, dict)
                else getattr(response_format, "type", None)
            )
            if format_type != "text":
                raise ValueError("Kimi-Audio does not support structured response formats")
        if (
            kwargs.get("tools")
            or kwargs.get("documents")
            or kwargs.get("continue_final_message")
            or kwargs.get("add_special_tokens")
            or not kwargs.get("add_generation_prompt", True)
            or params.return_assistant_tokens_mask
            or params.tool_choice not in (None, "none")
        ):
            raise ValueError("Kimi-Audio requires an ordinary conversation followed by a new assistant response")
        if not messages:
            raise ValueError("Kimi-Audio requires nonempty messages")
        for message in messages:
            if message["role"] not in ("user", "assistant"):
                raise ValueError("Kimi-Audio supports user and assistant messages")
            if any(message.get(key) for key in ("tool_calls", "audio", "reasoning", "reasoning_content")):
                raise ValueError(
                    "Kimi-Audio history requires explicit text/audio content, not tool or audio references"
                )
            content = message.get("content")
            if isinstance(content, str) and content:
                continue
            if not isinstance(content, list) or not content:
                raise ValueError("Kimi-Audio requires nonempty text or audio message content")
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in ("text", "input_audio", "audio_url"):
                    raise ValueError("Kimi-Audio accepts only text, input_audio and audio_url content parts")
                if part["type"] == "text" and not isinstance(part.get("text"), str):
                    raise ValueError("Kimi-Audio text content must be a string")
                if part.get("uuid") is not None:
                    raise ValueError(
                        "Kimi-Audio uses audio content for caching; supply audio data without uuid overrides"
                    )
        return output_type

    def _build_prompt(self, conversation, mm_data, output_type):
        audio_items = iter((mm_data or {}).get("audio", []))
        messages, audio_inputs = [], {}
        for message in conversation:
            role, content = message["role"], message["content"]
            if role == "assistant" and any(part["type"] == "audio" for part in content):
                # Official audio-text history aligns the transcript with delayed
                # GLM codes; it must not be treated as a new audio-understanding turn.
                if sum(part["type"] == "audio" for part in content) != 1:
                    raise ValueError("Kimi-Audio assistant audio history requires one recording and its transcript")
                transcript = "".join(part["text"] for part in content if part["type"] == "text")
                if not transcript:
                    raise ValueError("Kimi-Audio assistant audio history requires its transcript")
                parts = [("audio-text", (None, transcript))]
            else:
                # Content-part boundaries are not separate Kimi messages.
                # Join adjacent text before BPE; assistant text gets one EOS.
                parts = []
                for part in content:
                    if part["type"] == "text":
                        text = part["text"]
                        if not text:
                            continue
                        if parts and parts[-1][0] == "text":
                            parts[-1] = ("text", parts[-1][1] + text)
                        else:
                            parts.append(("text", text))
                    else:
                        parts.append(("audio", None))
            if not parts:
                raise ValueError("Kimi-Audio requires nonempty text or audio in each message")
            for kind, text in parts:
                if kind in ("audio", "audio-text"):
                    audio = next(audio_items, None)
                    if audio is None:
                        raise ValueError("Kimi-Audio requires audio data, not a cache-only audio UUID")
                    waveform, sample_rate = audio
                    waveform = np.asarray(waveform, dtype=np.float32)
                    if waveform.ndim != 1:
                        raise ValueError("Kimi-Audio requires mono audio")
                    if sample_rate != SAMPLE_RATE:
                        waveform = resample_audio_pyav(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
                    audio_inputs[len(messages)] = waveform
                messages.append({"role": role, "message_type": kind, "content": text})
        if next(audio_items, None) is not None:
            raise ValueError("Kimi-Audio audio data does not match the conversation's content parts")
        prompt = prepare_kimi_audio_inputs(
            messages,
            self.prompt_builder,
            audio_inputs=audio_inputs,
            output_type=output_type,
        )
        # The shared builder supplies an offline output-route default. HTTP
        # serving passes output_modalities separately to AsyncOmni.generate.
        prompt.pop("modalities")
        return conversation, prompt

    def render_messages(self, messages, params):
        output_type = self._validate_messages(messages, params)
        try:
            conversation, mm_data, _ = parse_chat_messages(
                messages, self.model_config, content_format="openai", media_io_kwargs=params.media_io_kwargs
            )
        except VLLMClientError as exc:
            raise OmniClientError(str(exc)) from None
        return self._build_prompt(conversation, mm_data, output_type)

    async def render_messages_async(self, messages, params):
        output_type = self._validate_messages(messages, params)
        try:
            conversation, mm_data, _ = await parse_chat_messages_async(
                messages, self.model_config, content_format="openai", media_io_kwargs=params.media_io_kwargs
            )
        except VLLMClientError as exc:
            raise OmniClientError(str(exc)) from None
        return await make_async(self._build_prompt, executor=self._executor)(conversation, mm_data, output_type)

    def tokenize_prompts(self, prompts, params):
        if params.truncate_prompt_tokens is not None or params.pad_prompt_tokens is not None:
            raise ValueError("Kimi-Audio cannot truncate or pad one stream independently of the other")
        if params.max_output_tokens_param == "max_completion_tokens":
            raise ValueError(
                "Kimi-Audio chat uses max_tokens; the current Omni stage override does not read max_completion_tokens"
            )
        if params.needs_detokenization or params.return_token_offsets:
            raise ValueError("Kimi-Audio's scheduler IDs do not support prompt echo or character offsets")
        try:
            return super().tokenize_prompts(prompts, params)
        except VLLMClientError as exc:
            # Keep native length validation while using Omni's client-error
            # type at the model adapter boundary, without changing serving.
            raise OmniClientError(str(exc)) from None

    async def tokenize_prompts_async(self, prompts, params):
        # Already tokenized by the native Kimi builder; retain vLLM's context
        # length validation without changing the aligned input layout.
        return self.tokenize_prompts(prompts, params)

    def process_for_engine(self, prompt, arrival_time, *, skip_mm_cache=False):
        try:
            result = super().process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
        except VLLMClientError as exc:
            raise OmniClientError(str(exc)) from None
        result["model_intermediate_buffer"] = prompt["model_intermediate_buffer"]
        return result

    async def process_for_engine_async(self, prompt, arrival_time, *, skip_mm_cache=False):
        try:
            result = await super().process_for_engine_async(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
        except VLLMClientError as exc:
            raise OmniClientError(str(exc)) from None
        result["model_intermediate_buffer"] = prompt["model_intermediate_buffer"]
        return result
