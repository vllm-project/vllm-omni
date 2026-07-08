# SPDX-License-Identifier: Apache-2.0
"""Audex (Nemotron-Labs-Audex-2B) TTS serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class AudexAdapter(ARTTSAdapter):
    """Plain English TTS: single built-in voice, no reference audio, no CFG."""

    stage_keys = frozenset({"audex_thinker"})
    name = "audex"

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
        cfg_scale = extra.get("cfg_scale")
        if cfg_scale is not None and float(cfg_scale) != 1.0:
            return (
                f"Audex classifier-free guidance is not yet supported; got cfg_scale={cfg_scale}. "
                "Omit cfg_scale or pass 1.0."
            )
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = {"prompt": build_cond_prompt(request.input)}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="audex")
