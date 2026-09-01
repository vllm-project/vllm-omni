# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS serving adapter."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    DEFAULT_TTS_LANGUAGES,
    ARTTSAdapter,
    PreparedRequest,
    TTSGenerationError,
)
from vllm_omni.entrypoints.openai.tts_adapters.capabilities import load_precomputed_speakers
from vllm_omni.utils.speaker_cache import validate_qwen3_tts_profile

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY = "_qwen3_tts_effective_max_tokens"
_MIN_CODEC_FRAMES = 192
_MAX_CODEC_FRAMES_PER_TEXT_TOKEN = 12


class Qwen3TTSCodecLimitError(TTSGenerationError):
    """Qwen3-TTS Base exhausted its codec budget without emitting EOS."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


@register_tts_adapter
class Qwen3TTSAdapter(ARTTSAdapter):
    """Adapter for Qwen3-TTS (AR ``engine_client`` backend)."""

    validates_generation = True
    stage_keys = frozenset({"qwen3_tts"})
    name = "qwen3_tts"

    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        """Qwen3-TTS normalization (Base-task inference, voice lowercasing) is
        performed inside ``validate`` today; kept fused for a strict behaviour
        match."""

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Qwen TTS request parameters. Returns error message or None."""
        # Infer Base task when ref_audio or ref_text is provided without explicit task_type.
        server = self.ctx.server
        if request.task_type is None and (request.ref_audio is not None or request.ref_text is not None):
            request.task_type = "Base"

        # Normalize voice to lowercase for case-insensitive matching
        if request.voice is not None:
            request.voice = request.voice.lower()
            if request.task_type is None and request.voice in self.capabilities.precomputed_speakers:
                request.task_type = "Base"
        task_type = request.task_type or "CustomVoice"

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language (case-insensitive; normalized to the title-cased config form)
        if request.language is not None:
            request.language = request.language.title()
            if request.language not in self.capabilities.supported_languages:
                return (
                    f"Invalid language '{request.language}'. Supported: "
                    f"{', '.join(sorted(self.capabilities.supported_languages))}"
                )

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice":
            available_speakers = server._get_available_speakers()
            if not available_speakers:
                return (
                    "This model does not support CustomVoice task (no speakers configured). "
                    "Use task_type='Base' with ref_audio/ref_text for voice cloning, "
                    "or use a CustomVoice model."
                )
            if request.voice is not None and request.voice not in available_speakers:
                return f"Invalid voice '{request.voice}'. Supported: {', '.join(sorted(available_speakers))}"

        # Validate speaker_embedding constraints
        if request.speaker_embedding is not None:
            if task_type != "Base":
                return "'speaker_embedding' is only valid for Base task"
            if not request.speaker_embedding:
                return "'speaker_embedding' must be a non-empty list of floats"
            # speaker_embedding implies x_vector_only_mode — set it before
            # Base task validation so callers don't need to pass it explicitly.
            request.x_vector_only_mode = True
            emb_len = len(request.speaker_embedding)
            dim_err = self.validate_tts_embedding_dim(emb_len)
            if dim_err is not None:
                return dim_err
        # Validate Base task requirements
        if task_type == "Base":
            if request.voice is None:
                # 1. Ensure a voice source is provided
                if request.ref_audio is None and getattr(request, "speaker_embedding", None) is None:
                    return "Base task requires 'ref_audio' or 'speaker_embedding' for voice cloning"
                # 2. Validate ref_audio format if it exists (using the helper from main)
                if request.ref_audio is not None:
                    fmt_err = server._validate_ref_audio_format(request.ref_audio)
                    if fmt_err:
                        return fmt_err
                # 3. Validate text requirements based on the mode
                if not getattr(request, "x_vector_only_mode", False):
                    if not request.ref_text or not request.ref_text.strip():
                        return (
                            "Base task requires non-empty 'ref_text' (transcript of "
                            "the reference audio) unless 'x_vector_only_mode' is enabled"
                        )
            else:
                voice_lower = request.voice.lower()
                if voice_lower in server.uploaded_speakers:
                    # Check if data file exists for uploaded speaker
                    speaker_info = server.uploaded_speakers[voice_lower]
                    file_path = Path(speaker_info["file_path"])
                    if not file_path.exists():
                        return f"Data file for uploaded speaker '{request.voice}' not found on disk"
                elif voice_lower in self.capabilities.precomputed_speakers:
                    profile = self.capabilities.precomputed_speakers[voice_lower]
                    mode = str(profile.get("mode") or "xvec").lower()
                    ref_text = request.ref_text or profile.get("ref_text")
                    if mode == "icl" and (not isinstance(ref_text, str) or not ref_text.strip()):
                        return (
                            f"Precomputed voice '{request.voice}' uses ICL mode but has no ref_text in "
                            "the request or custom voice manifest"
                        )
                else:
                    # need ref_audio for built-in speaker
                    if request.ref_audio is None:
                        return (
                            f"Base task with built-in speaker '{request.voice}' requires 'ref_audio' for voice cloning"
                        )
                    fmt_err = server._validate_ref_audio_format(request.ref_audio)
                    if fmt_err:
                        return fmt_err
                    if not getattr(request, "x_vector_only_mode", False) and (
                        not request.ref_text or not request.ref_text.strip()
                    ):
                        return (
                            "Base task requires non-empty 'ref_text' (transcript of "
                            "the reference audio) unless 'x_vector_only_mode' is enabled"
                        )

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length (using cached value from initialization)
        if request.instructions and len(request.instructions) > server._max_instructions_length:
            return f"Instructions too long (max {server._max_instructions_length} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt, tts_params, warmup_key = await self.ctx.server._build_qwen3_tts_request(request)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=tts_params.get("task_type", ["unknown"])[0],
            warmup_artifact_key=warmup_key,
        )

    def _get_expected_speaker_embedding_dim(self) -> int:
        """Return the loaded Qwen3-TTS speaker embedding dim, if known.

        The user-provided speaker embedding is concatenated directly with
        talker codec embeddings, so the real compatibility requirement is the
        talker hidden size.
        """
        hf_config = self.ctx.engine_client.model_config.hf_config
        talker_config = hf_config.talker_config
        return int(talker_config.hidden_size)

    def _load_precomputed_speakers(self) -> dict[str, dict]:
        return load_precomputed_speakers(
            self.ctx.engine_client,
            expected_model_type=self.name,
            validate_profile=lambda profile, tensors: validate_qwen3_tts_profile(
                profile,
                tensors,
                expected_embedding_dim=self._get_expected_speaker_embedding_dim(),
            ),
        )

    def _load_supported_languages(self) -> frozenset[str]:
        try:
            config = self.ctx.engine_client.model_config.hf_config.talker_config

            if isinstance(config, dict):
                codec_language_id = config.get("codec_language_id")
            else:
                codec_language_id = getattr(config, "codec_language_id", None)

            if codec_language_id and isinstance(codec_language_id, Mapping):
                return frozenset(str(language).title() for language in codec_language_id) | {"Auto"}

            logger.warning("No codec_language_id found in talker_config; falling back to default languages")
        except Exception as exc:
            logger.warning("Could not load languages from model config: %s", exc)
        return DEFAULT_TTS_LANGUAGES

    def validate_tts_embedding_dim(self, emb_dim: int) -> str | None:
        expected_dim = self._get_expected_speaker_embedding_dim()
        if emb_dim != expected_dim:
            return f"speaker_embedding has {emb_dim} dimensions; expected {expected_dim} for the loaded Qwen3-TTS model"
        return None

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        """Apply a text-scaled safety ceiling to Base codec generation.

        Qwen3-TTS can rarely enter a repetitive state in which codec EOS is no
        longer reachable through top-k sampling. A fixed 4096-frame ceiling
        turns that into several minutes of unusable audio. Bound the default
        Base-task budget by text length, while preserving an explicit caller
        ``max_new_tokens`` override and the configured budget for other tasks.
        """
        import copy

        del request_id
        server = self.ctx.server
        # Only scalar fields on stage 0 are changed below. Shallow-copy each
        # stage so the shared defaults stay immutable without deep-copying the
        # complete sampling configuration on every request.
        sampling_params_list = [copy.copy(params) for params in sampling_params_list]
        configured_cap = getattr(sampling_params_list[0], "max_tokens", None)
        task_type = request.task_type or "CustomVoice"
        text_tokens = None
        dynamic_cap = None
        effective_cap: int | None

        if request.max_new_tokens is not None:
            # An explicit request budget is an opt-out from the automatic
            # ceiling. It remains an upper bound, and a length finish is still
            # surfaced as an incomplete generation rather than valid audio.
            effective_cap = int(request.max_new_tokens)
        elif task_type == "Base":
            counted_text_tokens = server._count_usage_text_tokens(request.input)
            if counted_text_tokens > 0:
                text_tokens = counted_text_tokens
                dynamic_cap = max(_MIN_CODEC_FRAMES, text_tokens * _MAX_CODEC_FRAMES_PER_TEXT_TOKEN)
                effective_cap = min(dynamic_cap, int(configured_cap)) if configured_cap is not None else dynamic_cap
            else:
                # Token counting is best-effort. If the tokenizer is missing or
                # rejects the input, preserve the configured budget instead of
                # truncating an otherwise valid request at the minimum ceiling.
                effective_cap = int(configured_cap) if configured_cap is not None else None
        else:
            effective_cap = int(configured_cap) if configured_cap is not None else None

        if effective_cap is not None:
            effective_cap = max(1, effective_cap)
            sampling_params_list[0].max_tokens = effective_cap
            sampling_params_list[0].min_tokens = min(
                int(getattr(sampling_params_list[0], "min_tokens", 0) or 0),
                effective_cap,
            )

        if isinstance(prompt, dict):
            additional_information = prompt.get("additional_information")
            if isinstance(additional_information, dict) and effective_cap is not None:
                additional_information[QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY] = [effective_cap]

        logger.debug(
            "Qwen3-TTS codec budget: task_type=%s text_tokens=%s dynamic_cap=%s "
            "configured_cap=%s request_cap=%s effective_cap=%s",
            task_type,
            text_tokens,
            dynamic_cap,
            configured_cap,
            request.max_new_tokens,
            effective_cap,
        )
        return sampling_params_list

    def validate_generation(
        self,
        tts_params: Mapping[str, object],
        *,
        stage0_finish_reason: str | None,
        output_tokens: int,
    ) -> None:
        if QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY not in tts_params:
            return
        task_type = tts_params.get("task_type")
        if isinstance(task_type, (list, tuple)):
            task_type = task_type[0] if task_type else None
        if task_type != "Base" or stage0_finish_reason != "length":
            return

        raw_limit = tts_params.get(QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY)
        if isinstance(raw_limit, (list, tuple)):
            raw_limit = raw_limit[0] if raw_limit else None
        try:
            limit = int(raw_limit) if isinstance(raw_limit, (str, bytes, bytearray, int, float)) else 0
        except (TypeError, ValueError):
            limit = 0
        raise Qwen3TTSCodecLimitError(
            "Qwen3-TTS Base did not emit codec EOS before its token budget "
            f"({output_tokens}/{limit} codec tokens); the generated audio is incomplete."
        )
