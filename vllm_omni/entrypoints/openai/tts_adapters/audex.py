# SPDX-License-Identifier: Apache-2.0
"""Audex (Nemotron-Labs-Audex-2B) TTS serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


# Classifier-free guidance strength accepted on /v1/audio/speech via
# extra_params. 1.0 disables guidance (identical to omitting it); the
# official TTS quality setting is 1.5.
AUDEX_CFG_SCALE_MIN = 1.0
AUDEX_CFG_SCALE_MAX = 10.0

# Internal CFG pair-plumbing keys; injected by the serving layer once the
# request id exists, never accepted from callers.
_AUDEX_INTERNAL_CFG_KEYS = ("cfg_role", "cfg_pair_id", "cfg_null_prompt")


@register_tts_adapter
class AudexAdapter(ARTTSAdapter):
    """Plain English TTS: single built-in voice, no reference audio, optional CFG."""

    stage_keys = frozenset({"audex_thinker", "audex_omni"})
    name = "audex"

    @classmethod
    def stage_serves_speech(cls, model_stage: str | None, all_stage_keys: frozenset[str]) -> bool:
        """``audex_omni`` covers the audex_s2s deployment, whose TTS pass uses
        the same ``/v1/audio/speech`` surface. Without the speech decoder the
        same thinker is text-final and must not accept speech requests."""
        if model_stage == "audex_omni":
            return "audex_code2wav" in all_stage_keys
        return True

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Audex TTS requires non-empty input text"
        voice = (request.voice or "").strip().lower()
        if voice not in ("", "default"):
            return (
                f"Audex has a single built-in voice and no voice cloning; got voice={request.voice!r}. "
                "Omit 'voice' or pass 'default'."
            )
        if request.ref_audio is not None:
            return "Audex does not support reference audio (no voice cloning)."
        extra = request.extra_params or {}
        for key in _AUDEX_INTERNAL_CFG_KEYS:
            if key in extra:
                return f"extra_params.{key} is managed internally by the server; only cfg_scale may be set."
        cfg_scale = extra.get("cfg_scale")
        if cfg_scale is not None:
            try:
                cfg_value = float(cfg_scale)
            except (TypeError, ValueError):
                return (
                    f"extra_params.cfg_scale must be a number in "
                    f"[{AUDEX_CFG_SCALE_MIN}, {AUDEX_CFG_SCALE_MAX}]; got {cfg_scale!r}."
                )
            if not (AUDEX_CFG_SCALE_MIN <= cfg_value <= AUDEX_CFG_SCALE_MAX):
                return (
                    f"extra_params.cfg_scale must be within [{AUDEX_CFG_SCALE_MIN}, {AUDEX_CFG_SCALE_MAX}]; "
                    f"got {cfg_scale!r}. 1.0 disables guidance; 1.5 is the recommended quality setting."
                )
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = {"prompt": build_cond_prompt(request.input)}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="audex")
