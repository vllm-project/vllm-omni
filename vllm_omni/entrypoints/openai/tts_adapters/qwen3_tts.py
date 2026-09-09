# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS serving adapter."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    DEFAULT_TTS_LANGUAGES,
    ARTTSAdapter,
    PreparedRequest,
    TTSGenerationError,
    conditioning_cache_salt,
)
from vllm_omni.entrypoints.openai.tts_adapters.capabilities import load_precomputed_speakers
from vllm_omni.utils.speaker_cache import validate_qwen3_tts_profile

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)
_REF_AUDIO_CACHE_KEY = "_qwen3_tts_ref_audio_cache_key"

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
    supported_output_sample_rates = frozenset({8000, 24000})

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._estimate_prompt_len_async = make_async(
            self._estimate_prompt_len, executor=getattr(ctx.server, "_tts_executor", None)
        )

    def _estimate_ref_code_len(self, ref_audio: object) -> int | None:
        codec_frame_rate = self.capabilities.codec_frame_rate
        if codec_frame_rate is None:
            return None
        try:
            item = ref_audio
            while isinstance(item, list) and item:
                if len(item) == 2 and isinstance(item[1], (int, float)):
                    break
                item = item[0]
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None
            wav, sr = item
            sr = int(sr)
            n_samples = len(wav) if hasattr(wav, "__len__") else wav.shape[-1]
            if sr <= 0 or n_samples <= 0:
                return None
            return math.ceil(n_samples / sr * codec_frame_rate)
        except Exception:
            return None

    def _get_model_variant(self) -> str | None:
        """Return the task supported by the loaded Qwen3-TTS checkpoint.

        Checkpoint metadata is authoritative. Metadata-less re-exports may be
        served from dated snapshot directories, so only when metadata is
        absent or unrecognized do we inspect path components from the leaf
        upwards. A marker must end a component to avoid inferring ``Base`` from
        names such as ``base_models`` or ``database``.
        """
        model_config = self.ctx.server.engine_client.model_config
        hf_config = getattr(model_config, "hf_config", None)
        configured_variant = getattr(hf_config, "tts_model_type", None)
        variants = {
            "customvoice": "CustomVoice",
            "voicedesign": "VoiceDesign",
            "base": "Base",
        }

        if isinstance(configured_variant, str):
            normalized = re.sub(r"[^a-z]", "", configured_variant.lower())
            if (variant := variants.get(normalized)) is not None:
                return variant

        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str):
            return None
        for component in reversed(re.split(r"[\\/]+", model_path.rstrip("/\\"))):
            match = re.search(r"(?:^|[-_.])(custom[-_.]?voice|voice[-_.]?design|base)$", component.lower())
            if match is not None:
                return variants[re.sub(r"[-_.]", "", match.group(1))]
        return None

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
            stored_voice = (
                request.voice in server.uploaded_speakers or request.voice in self.capabilities.precomputed_speakers
            )
            # _build_tts_params dispatches stored voices as Base. Normalize to
            # that effective task before model-variant validation so the
            # admission gate and builder cannot disagree.
            if stored_voice:
                request.task_type = "Base"
        task_type = request.task_type or "CustomVoice"

        model_variant = self._get_model_variant()
        if model_variant is not None and task_type != model_variant:
            return (
                f"Qwen3-TTS {model_variant} checkpoint does not support task_type='{task_type}'. "
                f"Use task_type='{model_variant}' or load the matching {task_type} checkpoint."
            )

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

    def _build_tts_params(self, request: "OpenAICreateSpeechRequest") -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        server = self.ctx.server

        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            voice_lower = request.voice.lower()
            precomputed_speakers = self.capabilities.precomputed_speakers
            params["speaker"] = [request.voice]
            params["voice_created_at"] = [server._voice_created_at(voice_lower)]

            # Uploaded voices use task_type="Base" (CustomVoice requires built-in spk_id).
            # If ref_text was provided at upload time, use in-context cloning; otherwise x_vector only.
            if voice_lower in server.uploaded_speakers and request.ref_audio is None:
                speaker_info = server.uploaded_speakers[voice_lower]

                # Check if this voice was uploaded with a pre-computed embedding.
                # Populate request.speaker_embedding so the existing code path
                # (below) handles voice_clone_prompt and x_vector_only_mode.
                embedding = server._get_uploaded_speaker_embedding(request.voice)
                if embedding is not None:
                    request.speaker_embedding = embedding
                    params["speaker"] = [voice_lower]
                    params["task_type"] = ["Base"]
                    logger.info("Auto-set speaker_embedding for uploaded voice: %s", request.voice)
                else:
                    audio_data = server._get_uploaded_audio_data(request.voice)
                    if not audio_data:
                        raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing or corrupted")
                    stored_ref_text = speaker_info.get("ref_text")
                    params["speaker"] = [voice_lower]
                    params["ref_audio"] = [audio_data]
                    params["task_type"] = ["Base"]
                    if stored_ref_text:
                        params["ref_text"] = [stored_ref_text]
                        params["x_vector_only_mode"] = [False]
                    else:
                        params["x_vector_only_mode"] = [True]
                    logger.info(
                        "Auto-set ref_audio for uploaded voice: %s (icl=%s)", request.voice, bool(stored_ref_text)
                    )
            elif voice_lower in precomputed_speakers and request.ref_audio is None:
                profile = precomputed_speakers[voice_lower]
                mode = str(profile.get("mode") or "xvec").lower()
                params["speaker"] = [voice_lower]
                params["task_type"] = ["Base"]
                params["x_vector_only_mode"] = [mode != "icl"]
                ref_text = request.ref_text or profile.get("ref_text")
                if isinstance(ref_text, str) and ref_text.strip():
                    params["ref_text"] = [ref_text]
                ref_code_length = profile.get("ref_code_length")
                if mode == "icl" and ref_code_length:
                    params["ref_code_length"] = [int(ref_code_length)]
                logger.info("Using precomputed Qwen3-TTS custom voice profile: %s (mode=%s)", voice_lower, mode)

        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone: ref_audio resolved in create_speech(), not here.
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.speaker_embedding is not None:
            # Store as plain float list (not tensor) so it survives msgspec
            # serialization through the EngineCore IPC boundary.  The talker's
            # _build_prompt_embeds converts it back to a tensor on the GPU.
            params["voice_clone_prompt"] = [
                {
                    "ref_spk_embedding": list(request.speaker_embedding),
                }
            ]
            # speaker_embedding implies x_vector_only_mode
            params["x_vector_only_mode"] = [True]
        elif request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

        if request.initial_codec_chunk_frames is not None:
            params["initial_codec_chunk_frames"] = [request.initial_codec_chunk_frames]

        if request.non_streaming_mode is not None:
            params["non_streaming_mode"] = [request.non_streaming_mode]
        # Preserve the legacy VoiceDesign fallback when the request omits an
        # explicit override. CustomVoice and Base rely on model defaults
        # (True and False respectively).
        elif params["task_type"][0] == "VoiceDesign":
            params["non_streaming_mode"] = [True]

        return params

    def _estimate_prompt_len(self, tts_params: dict[str, Any]) -> int:
        """Estimate prompt length so the placeholder matches model-side embeddings."""
        try:
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                Qwen3TTSPromptEmbedsBuilder,
            )

            server = self.ctx.server
            if server._tts_tokenizer is None:
                from transformers import AutoTokenizer

                model_name = self.ctx.engine_client.model_config.model
                server._tts_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="left",
                )
            hf_config = self.ctx.engine_client.model_config.hf_config
            talker_config = hf_config.talker_config
            task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]
            return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=tts_params,
                task_type=task_type,
                tokenize_prompt=lambda t: server._tts_tokenizer(t, padding=False)["input_ids"],
                codec_language_id=getattr(talker_config, "codec_language_id", None),
                spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
                estimate_ref_code_len=self._estimate_ref_code_len,
            )
        except Exception as e:
            logger.warning("Failed to estimate TTS prompt length, using fallback 2048: %s", e)
            return 2048

    def _qwen3_tts_can_use_ref_audio_artifact_only(self, tts_params: dict[str, Any], artifact_key: str | None) -> bool:
        server = self.ctx.server
        x_vector_only = server._tts_x_vector_only(tts_params)
        if not artifact_key or (artifact_key, x_vector_only) not in server._ref_audio_model_artifact_ready:
            return False
        return (tts_params.get("task_type") or ["CustomVoice"])[0] == "Base"

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build prompt + tts_params for Qwen3-TTS.

        Called from ``Qwen3TTSAdapter.build``. Returns
        ``(prompt, tts_params, warmup_artifact_key)`` where the warmup key is the
        Qwen3-TTS ref-audio artifact tracked after ``generate()``.
        """
        server = self.ctx.server
        qwen3_ref_audio_warmup_artifact_key: str | None = None
        tts_params = self._build_tts_params(request)
        # Resolve ref_audio (explicit or auto-set for uploaded voices)
        # to [[wav_list, sr]] so the model doesn't re-decode base64.
        ref_audio_source = request.ref_audio
        if ref_audio_source is None and isinstance(tts_params.get("ref_audio"), list):
            # Uploaded voice: ref_audio was auto-set as [base64_data_url]
            ref_audio_source = tts_params["ref_audio"][0]
        if ref_audio_source is not None and isinstance(ref_audio_source, str):
            wav_list, sr, cache_key = await server._resolve_ref_audio(ref_audio_source)
            tts_params["ref_audio_cache_key"] = cache_key
            artifact_key = server._get_resolved_ref_audio_artifact_key(cache_key)
            if artifact_key:
                tts_params[_REF_AUDIO_CACHE_KEY] = [artifact_key]
            ref_code_length = self._estimate_ref_code_len([wav_list, sr])
            if ref_code_length is not None:
                tts_params["ref_code_length"] = [int(ref_code_length)]
            if self._qwen3_tts_can_use_ref_audio_artifact_only(tts_params, artifact_key):
                logger.debug("Using Qwen3-TTS ref_audio artifact-only path: %s", artifact_key)
            else:
                tts_params["ref_audio"] = [[wav_list, sr]]
                qwen3_ref_audio_warmup_artifact_key = artifact_key

        ph_len = await self._estimate_prompt_len_async(tts_params)
        prompt = tokens_input(prompt_token_ids=[1] * ph_len)
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=tts_params.get("task_type", ["unknown"])[0],
            warmup_artifact_key=qwen3_ref_audio_warmup_artifact_key,
        )

    def _get_expected_speaker_embedding_dim(self) -> int:
        """Return the loaded Qwen3-TTS speaker embedding dim, if known.

        The user-provided speaker embedding is concatenated directly with
        talker codec embeddings, so the real compatibility requirement is the
        talker hidden size.
        """
        hf_config = self.ctx.server.engine_client.model_config.hf_config
        talker_config = hf_config.talker_config
        return int(talker_config.hidden_size)

    def _load_precomputed_speakers(self) -> dict[str, dict]:
        return load_precomputed_speakers(
            self.ctx.server.engine_client,
            expected_model_type=self.name,
            validate_profile=lambda profile, tensors: validate_qwen3_tts_profile(
                profile,
                tensors,
                expected_embedding_dim=self._get_expected_speaker_embedding_dim(),
            ),
        )

    def _load_supported_languages(self) -> frozenset[str]:
        try:
            config = self.ctx.server.engine_client.model_config.hf_config.talker_config

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

        # Propagate the deploy/YAML stage seed to residual MTP sampling. The
        # generic serving layer applies an explicit request.seed afterwards,
        # so the request value still has higher precedence.
        stage0_params = sampling_params_list[0]
        default_seed = getattr(stage0_params, "seed", None)
        if default_seed is not None:
            if stage0_params.extra_args is None:
                stage0_params.extra_args = {}
            stage0_params.extra_args.setdefault("tts_local_seed", int(default_seed))

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
