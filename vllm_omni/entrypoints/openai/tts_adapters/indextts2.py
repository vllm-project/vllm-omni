# SPDX-License-Identifier: Apache-2.0
"""IndexTTS2 serving adapter."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.inputs import tokens_input
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096
_INDEXTTS2_EMOTION_KEYS = ("emo_audio", "emo_vector", "emo_alpha", "emo_text", "use_emo_text", "use_random")


def _update_conditioning_hash(h: hashlib._Hash, value: Any) -> None:
    """Hash request conditioning values without serializing huge repr strings."""
    if value is None:
        h.update(b"none")
        return
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        h.update(b"tensor")
        h.update(repr((arr.shape, str(arr.dtype))).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
        return
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        h.update(b"ndarray")
        h.update(repr((arr.shape, str(arr.dtype))).encode("utf-8"))
        h.update(arr.tobytes())
        return
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (int, np.integer)):
        samples = value[0]
        if isinstance(samples, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(samples, torch.Tensor):
                samples_arr = samples.detach().cpu().numpy()
            else:
                samples_arr = np.asarray(samples, dtype=np.float32)
            samples_arr = np.ascontiguousarray(samples_arr, dtype=np.float32)
            h.update(b"audio")
            h.update(int(value[1]).to_bytes(8, byteorder="little", signed=True))
            h.update(int(samples_arr.size).to_bytes(8, byteorder="little", signed=False))
            h.update(samples_arr.tobytes())
            return
    if isinstance(value, Mapping):
        h.update(b"mapping")
        for key in sorted(value):
            h.update(b"\x01")
            h.update(repr(key).encode("utf-8"))
            h.update(b"\x02")
            _update_conditioning_hash(h, value[key])
        return
    if isinstance(value, (list, tuple)):
        h.update(f"sequence:{len(value)}".encode())
        for item in value:
            h.update(b"\x03")
            _update_conditioning_hash(h, item)
        return
    h.update(repr(value).encode("utf-8"))


def _first_conditioning_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def indextts2_conditioning_cache_salt(
    request: OpenAICreateSpeechRequest,
    tts_params: dict[str, Any],
    *,
    request_id: str | None = None,
) -> str:
    """Stable prefix-cache salt for IndexTTS2 placeholder prompts.

    IndexTTS2 builds prefill embeddings from ``additional_information`` (speaker
    audio plus emotion controls), while the scheduler only sees dummy token ids.
    Include all supported control inputs so prefix-cache hits are only possible
    when text, speaker, and emotion conditioning are identical.
    """
    h = hashlib.sha256()
    for key in (
        "text",
        "voice",
        "voice_name",
        "voice_created_at",
        "emo_audio",
        "emo_vector",
        "emo_alpha",
        "emo_text",
        "use_emo_text",
        "use_random",
    ):
        h.update(b"\x00")
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        _update_conditioning_hash(h, tts_params.get(key))

    for key, value in (
        ("input", request.input),
        ("ref_audio", request.ref_audio),
    ):
        h.update(b"\x00")
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        _update_conditioning_hash(h, value)

    use_random = bool(_first_conditioning_value(tts_params.get("use_random")))
    if use_random:
        h.update(b"\x00indextts2-use-random-request\x00")
        h.update((request_id or random_uuid()).encode("utf-8"))
    return h.hexdigest()[:32]


@register_tts_adapter
class IndexTTS2Adapter(ARTTSAdapter):
    stage_keys = frozenset({"indextts2_talker"})
    name = "indextts2"

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        has_uploaded_voice = bool(
            voice_lower
            and voice_lower in server.uploaded_speakers
            and server._get_uploaded_audio_data(voice_lower) is not None
        )
        if request.ref_audio is None and not has_uploaded_voice:
            return "IndexTTS2 requires 'ref_audio' or an uploaded audio voice for voice cloning"
        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"
        if request.extra_params is not None:
            if not isinstance(request.extra_params, dict):
                return "extra_params must be a JSON object/dict."
            extra_err = self._validate_extra_params(request.extra_params)
            if extra_err:
                return extra_err
        return None

    def _validate_extra_params(self, extras: Mapping[str, Any]) -> str | None:
        server = self.ctx.server
        if "emo_audio" in extras:
            value = extras["emo_audio"]
            if not isinstance(value, str):
                return "extra_params.emo_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
            fmt_err = server._validate_ref_audio_format(value)
            if fmt_err:
                return fmt_err.replace("ref_audio", "extra_params.emo_audio", 1)

        if "emo_vector" in extras:
            value = extras["emo_vector"]
            if not isinstance(value, list) or len(value) != 8:
                return "extra_params.emo_vector must be a list of 8 numbers"
            for item in value:
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                    return "extra_params.emo_vector must be a list of 8 finite numbers"
                if float(item) < 0.0 or float(item) > 1.2:
                    return "extra_params.emo_vector values must be in [0, 1.2]"

        if "emo_alpha" in extras:
            value = extras["emo_alpha"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return "extra_params.emo_alpha must be a finite number"

        if "emo_text" in extras and extras["emo_text"] is not None and not isinstance(extras["emo_text"], str):
            return "extra_params.emo_text must be a string"

        for key in ("use_emo_text", "use_random"):
            if key in extras and not isinstance(extras[key], bool):
                return f"extra_params.{key} must be a boolean"
        return None

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio

        server = self.ctx.server
        tts_params = await self._build_params(request)
        from vllm_omni.model_executor.models.indextts2.prompt_utils import (
            estimate_indextts2_prefill_prompt_len,
        )

        ph_len = estimate_indextts2_prefill_prompt_len(
            server.engine_client.model_config.model,
            request.input,
        )
        prompt = tokens_input(prompt_token_ids=[1] * ph_len)
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = indextts2_conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="indextts2")

    async def _build_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        server = self.ctx.server
        params: dict[str, Any] = {"text": [request.input]}
        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        ref_audio_source = request.ref_audio
        using_uploaded_voice = False
        if ref_audio_source is None and voice_lower and voice_lower in server.uploaded_speakers:
            ref_audio_source = server._get_uploaded_audio_data(voice_lower)
            using_uploaded_voice = ref_audio_source is not None
        if ref_audio_source is not None and isinstance(ref_audio_source, str):
            wav_list, sr = await server._resolve_ref_audio(ref_audio_source)
            params["voice"] = [[wav_list, sr]]
        if using_uploaded_voice and voice_lower:
            params["voice_name"] = [voice_lower]
            params["voice_created_at"] = [server._voice_created_at(voice_lower)]

        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        for key in _INDEXTTS2_EMOTION_KEYS:
            if key not in extras:
                continue
            if key == "emo_audio":
                wav_list, sr = await server._resolve_ref_audio(extras[key])
                params[key] = [[wav_list, sr]]
            else:
                params[key] = [extras[key]]
        return params
