# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native VibeVoice speech serving; references are optional, as in Transformers."""

import time
from typing import Any
from urllib.parse import urlparse

import numpy as np
from vllm.exceptions import VLLMClientError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.model_executor.models.vibevoice.prompt import parse_vibevoice_script, render_script_prompt

logger = init_logger(__name__)

# These fields belong to other TTS models. Reject even explicit False/empty values
# instead of silently promising conditioning that VibeVoice does not consume.
_UNSUPPORTED_FIELDS = (
    "instructions",
    "language",
    "ref_text",
    "ref_audio_2",
    "task_type",
    "ambient_sound",
    "duration_seconds",
    "speaker_embedding",
    "ref_audio_codes",
    "x_vector_only_mode",
    "initial_codec_chunk_frames",
    "non_streaming_mode",
    "seed",
)


@register_tts_adapter
class VibeVoiceTTSAdapter(ARTTSAdapter):
    name = "vibevoice"
    stage_keys = frozenset({"vibevoice"})
    model_archs = frozenset({"VibeVoiceForConditionalGeneration"})
    supported_output_sample_rates = frozenset({24000})
    max_new_tokens_max = 16384
    #: The offline processor allows 60 s references; keep the shared online
    #: resolver from rejecting 30–60 s clips before the adapter runs.
    max_reference_duration = 60.0

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        for field in _UNSUPPORTED_FIELDS:
            if getattr(request, field) is not None:
                return f"VibeVoice does not support '{field}'."
        if request.word_timestamps:
            return "VibeVoice does not support word_timestamps."
        if request.speed not in (None, 1.0):
            return "VibeVoice only supports speed=1.0."
        if request.sample_rate not in (None, 24000):
            return "VibeVoice only supports sample_rate=24000."
        if request.extra_params:
            return "VibeVoice does not support extra_params; CFG=1.3 and diffusion steps=10 are fixed."
        try:
            _, speakers = parse_vibevoice_script(request.input)
        except ValueError as exc:
            return str(exc)
        server = self.ctx.server
        if request.voice is not None:
            if request.ref_audio is not None:
                return "VibeVoice accepts either voice or ref_audio, not both."
            if len(speakers) != 1:
                return "Uploaded voice is only supported for single-speaker VibeVoice scripts."
            info = server.uploaded_speakers.get(request.voice.lower())
            if info is None:
                return f"Unknown VibeVoice voice '{request.voice}'; upload audio via POST /v1/audio/voices."
            if info.get("embedding_source") == "direct":
                return "VibeVoice requires an uploaded audio voice, not a speaker embedding."
        if request.ref_audio is not None:
            refs = [request.ref_audio] if isinstance(request.ref_audio, str) else request.ref_audio
            if not refs or len(refs) > len(speakers):
                return "VibeVoice ref_audio must contain 1 to the number of distinct speakers references."
            for source in refs:
                if error := server._validate_ref_audio_format(source):
                    return error
                # MediaConnector raises RuntimeError for this configuration;
                # reject the client request before it can become an HTTP 500.
                if urlparse(source).scheme == "file" and not server.model_config.allowed_local_media_path:
                    return "VibeVoice file references require --allowed-local-media-path."
        return self._validate_max_new_tokens(request)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        lines, speakers = parse_vibevoice_script(request.input)
        source = request.ref_audio
        if request.voice is not None:
            # The shared resolver may attach a stored transcript. Use a copy:
            # VibeVoice consumes only the audio, and must not mutate the caller.
            resolved = request.model_copy()
            if error := self.ctx.server._apply_uploaded_speaker(resolved):
                raise ValueError(error)
            source = resolved.ref_audio
            if source is None:
                raise ValueError(f"Audio for uploaded voice '{request.voice}' is unavailable.")
        refs = [] if source is None else ([source] if isinstance(source, str) else source)
        audio = []
        for ref in refs:
            try:
                wav, sr, _ = await self.ctx.server._resolve_ref_audio_array(ref)
            except VLLMClientError as exc:
                # The shared speech endpoint maps ValueError to a client error.
                # Do not recategorize transient fetch or internal runtime errors.
                raise ValueError("Invalid VibeVoice reference audio; check its URL, format and permissions.") from exc
            # Fail before scheduling a malformed reference on the GPU. The
            # shared resolver already normalizes channels; never mutate its cache.
            wav = np.asarray(wav, dtype=np.float32)
            if sr <= 0 or wav.ndim != 1 or not 0 < wav.size <= 60 * sr or not np.isfinite(wav).all():
                raise ValueError("VibeVoice reference audio must be finite, non-empty and at most 60 seconds.")
            audio.append((wav, sr))
        # Positional API: K references condition the first K distinct speakers.
        # This is an API convention; HF itself attaches each audio to a role.
        prompt: dict[str, Any] = {"prompt": render_script_prompt(lines, speakers[: len(audio)])}
        if audio:
            prompt["multi_modal_data"] = {"audio": audio}
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        # Greedy gate / allowed tokens / stops remain owned by the pipeline.
        return apply_max_new_tokens(sampling_params_list, request)

    def _load_supported_speakers(self) -> list[str]:
        return []

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset()  # language is inferred from text, not selected by API

    def _load_codec_frame_rate(self) -> float:
        return 7.5  # one 3200-sample audio token at 24 kHz

    async def warmup(self) -> None:
        request = OpenAICreateSpeechRequest(input="Hello.", response_format="wav", max_new_tokens=8)
        start = time.perf_counter()
        try:
            await self.ctx.server._generate_audio_bytes(request, request_id="speech-warmup-vibevoice")
        except Exception:
            logger.warning("VibeVoice speech warmup failed", exc_info=True)
            return
        logger.info("VibeVoice speech warmup complete in %.1fs", time.perf_counter() - start)
