# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS serving adapter."""

from pathlib import Path
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class Qwen3TTSAdapter(ARTTSAdapter):
    """Adapter for Qwen3-TTS (AR ``engine_client`` backend)."""

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
            if request.task_type is None and request.voice in server.precomputed_speakers:
                request.task_type = "Base"
        task_type = request.task_type or "CustomVoice"

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language (case-insensitive; normalized to the title-cased config form)
        if request.language is not None:
            request.language = request.language.title()
            if request.language not in server.supported_languages:
                return (
                    f"Invalid language '{request.language}'. Supported: {', '.join(sorted(server.supported_languages))}"
                )

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice":
            if not server.supported_speakers:
                return (
                    "This model does not support CustomVoice task (no speakers configured). "
                    "Use task_type='Base' with ref_audio/ref_text for voice cloning, "
                    "or use a CustomVoice model."
                )
            if request.voice is not None and request.voice not in server.supported_speakers:
                return f"Invalid voice '{request.voice}'. Supported: {', '.join(sorted(server.supported_speakers))}"

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
            dim_err = server._validate_qwen_tts_speaker_embedding_dim(emb_len)
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
                elif voice_lower in server.precomputed_speakers:
                    profile = server.precomputed_speakers[voice_lower]
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
