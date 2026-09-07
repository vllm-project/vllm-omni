# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 serving adapter (AR base-LM + diffusion side-computation)."""

import time
from typing import Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.entrypoints.openai.tts_adapters.capabilities import load_precomputed_speakers
from vllm_omni.utils.speaker_cache import validate_voxcpm2_profile

logger = init_logger(__name__)


@register_tts_adapter
class VoxCPM2Adapter(ARTTSAdapter):
    """VoxCPM2 shares ``latent_generator`` with VoxCPM; selected when no ``vae``
    stage is present (and/or via ``model_arch``)."""

    stage_keys = frozenset({"latent_generator"})
    model_archs = frozenset({"VoxCPM2TalkerForConditionalGeneration"})
    name = "voxcpm2"
    # The talker architecture is authoritative and is tested ahead of every
    # stage key, so a VoxCPM2 talker deployed under a stage key another model
    # claims still resolves to VoxCPM2.
    detect_priority = 10

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate VoxCPM2 request parameters. Returns error message or None."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.voice is not None:
            request.voice = request.voice.lower()
            available_voices = server._get_available_speakers()
            if request.voice not in available_voices:
                supported = ", ".join(sorted(available_voices)) or "none"
                return f"Invalid voice '{request.voice}'. Supported: {supported}"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        # VoxCPM2 needs the raw waveform tuple for prefill-length accounting, so
        # it loads uploaded audio directly rather than via _apply_uploaded_speaker.
        uploaded_ref = None
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                if server.uploaded_speakers[voice_lower].get("embedding_source") == "direct":
                    raise ValueError(
                        f"Uploaded voice '{request.voice}' uses a speaker embedding (Qwen3-only). "
                        f"Re-upload with an audio file for VoxCPM2."
                    )
                if request.ref_audio is None:
                    uploaded_ref = server._load_uploaded_audio(voice_lower)
        prompt = await server._build_voxcpm2_prompt(request, uploaded_ref=uploaded_ref)
        tts_params = {}
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers or voice_lower in self.capabilities.precomputed_speakers:
                additional = prompt.setdefault("additional_information", {})
                additional["voice_name"] = voice_lower
                additional["voice_created_at"] = server._voice_created_at(voice_lower)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="voxcpm2")

    async def warmup(self) -> None:
        """Warm up VoxCPM2 through a synthetic serving request.

        VoxCPM2 needs to warm up its PagedAttention scaffold/residual LLMs. CUDA
        Graph capture requires a vLLM ``ForwardContext`` containing attention
        metadata and slot mappings, which is only available during inference. The
        request also pays the one-time torch.compile cost for LocDiT, feat_encoder,
        AudioVAE, and projection helpers.
        """
        server = self.ctx.server
        t0 = time.time()
        logger.info("Running warmup speech request for model_type=%s", self.name)
        # VoxCPM2 has no predefined speaker presets — "default" means zero-shot
        # mode (no voice cloning).  The voice field is required by the OpenAI
        # API schema but semantically ignored by the model.
        warmup_req = OpenAICreateSpeechRequest(
            input="Warmup.",
            voice="default",
            response_format="wav",
            speed=1.0,
            stream=False,
            model=server.model_name,
        )
        try:
            _audio_bytes, _media_type = await server._generate_audio_bytes(warmup_req, request_id="speech-warmup")
        except Exception as exc:
            logger.warning("Speech warmup failed (non-fatal): %s", exc)
            return

        elapsed = time.time() - t0
        logger.info("Speech warmup complete in %.1fs", elapsed)

    def _load_precomputed_speakers(self) -> dict[str, dict]:
        return load_precomputed_speakers(
            self.ctx.engine_client,
            expected_model_type=self.name,
            validate_profile=validate_voxcpm2_profile,
        )

    def _load_supported_speakers(self) -> set[str]:
        return {"default"}

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
