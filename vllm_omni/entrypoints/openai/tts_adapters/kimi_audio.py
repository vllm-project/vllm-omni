# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kimi-Audio instructed reading through the existing speech serving adapter.

The model generates both text and audio; reading is prompted, not forced text
conditioning. Verbatim fidelity requires validation with pretrained weights.
Multimodal chat requests belong to a separate serving boundary.
"""

from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.model_executor.models.kimi_audio.audio_processing import (
    SAMPLE_RATE,
    SAMPLES_PER_TOKEN,
    prepare_kimi_audio_inputs,
)
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioPromptBuilder
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class KimiAudioAdapter(ARTTSAdapter):
    name = "kimi_audio"
    stage_keys = frozenset({"kimi_audio_ar"})
    supported_output_sample_rates = frozenset({24000})

    @classmethod
    def stage_serves_speech(cls, model_stage: str | None, all_stage_keys: frozenset[str]) -> bool:
        return model_stage == "kimi_audio_ar" and "kimi_audio_decoder" in all_stage_keys

    def _engine_client(self) -> Any:
        client = self.ctx.engine_client
        if client is None:
            raise ValueError("Kimi-Audio speech serving requires an AR engine client")
        return client

    def _load_codec_frame_rate(self) -> float:
        # Kimi uses the GLM frame rate, not a speech_tokenizer/config.json.
        return SAMPLE_RATE / SAMPLES_PER_TOKEN

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input.strip():
            return "Kimi-Audio requires nonempty input text"
        if request.is_streaming() and not getattr(self._engine_client().model_config, "async_chunk", False):
            return "Kimi-Audio streaming speech requires the kimi_audio_async_chunk.yaml deployment"
        if request.voice not in (None, "default"):
            return "Kimi-Audio has no selectable voices; omit voice or use 'default'"
        unsupported = [
            name
            for name in (
                "instructions",
                "task_type",
                "language",
                "ref_audio",
                "ref_text",
                "ref_audio_2",
                "ambient_sound",
                "duration_seconds",
                "x_vector_only_mode",
                "speaker_embedding",
                "initial_codec_chunk_frames",
                "non_streaming_mode",
            )
            if getattr(request, name) is not None
        ]
        if request.word_timestamps:
            unsupported.append("word_timestamps")
        if unsupported:
            return f"Kimi-Audio speech does not support: {', '.join(unsupported)}"
        extra = request.extra_params or {}
        if unknown := extra.keys() - {"temperature", "top_k", "top_p", "kimi_audio"}:
            return f"Unsupported Kimi-Audio extra_params: {', '.join(sorted(unknown))}"
        if extra.get("top_p", 1.0) != 1.0:
            return "Kimi-Audio sampling requires top_p=1.0"
        overrides = extra.get("kimi_audio", {})
        if not isinstance(overrides, dict):
            return "extra_params.kimi_audio must be a parameter dictionary"
        try:
            KimiAudioSamplingParams(**overrides)
        except (TypeError, ValueError) as exc:
            return str(exc)
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        engine_client = self._engine_client()
        model_config = engine_client.model_config
        tokenizer = engine_client.renderer.get_tokenizer()
        # Like Step-Audio2, explicitly request reading rather than presenting
        # the input as a question for a conversational model to answer.
        messages = [
            {
                "role": "user",
                "message_type": "text",
                "content": "Read the following text aloud exactly as written, without adding other words:\n"
                + request.input,
            }
        ]
        prompt = prepare_kimi_audio_inputs(
            messages,
            KimiAudioPromptBuilder.from_tokenizer(tokenizer, model_config.hf_config),
            output_type="both",
        )
        max_tokens = request.max_new_tokens or sampling_params_list[0].max_tokens
        if len(prompt["prompt_token_ids"]) + (max_tokens or 0) > model_config.max_model_len:
            raise ValueError("Kimi-Audio prompt and requested output exceed the model's maximum context length")
        prompt["modalities"] = ["audio"]
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
