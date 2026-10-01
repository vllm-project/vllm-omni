# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Microsoft VibeVoice TTS serving adapter."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import regex as re
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
)
from vllm_omni.model_executor.models.vibevoice.default_voices import (
    get_default_reference_audio_path,
)
from vllm_omni.model_executor.models.vibevoice.pipeline import (
    VIBEVOICE_VALID_TOKEN_IDS,
)
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    MAX_AUDIO_SECONDS,
)
from vllm_omni.model_executor.models.vibevoice.runtime_config import (
    VIBEVOICE_RUNTIME_CONTROL_KEYS,
)
from vllm_omni.model_executor.models.vibevoice.stateful import (
    validate_guidance_scale,
    validate_num_diffusion_steps,
)
from vllm_omni.model_executor.models.vibevoice.vllm_compat import (
    get_stage0_tokenizer,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
_REFERENCE_SEGMENT = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
_OFFICIAL_MAX_SPEAKERS = 4
_UNSUPPORTED_VIBEVOICE_FIELDS: dict[str, str | None] = {
    "speaker_embedding": "use 'ref_audio' or an uploaded audio voice for VibeVoice voice cloning",
    "instructions": None,
    "language": None,
    "ref_text": None,
    "ref_audio_2": None,
    "task_type": None,
    "ambient_sound": None,
    "duration_seconds": None,
    "x_vector_only_mode": None,
    "initial_codec_chunk_frames": None,
    "non_streaming_mode": None,
    "word_timestamps": "word timestamps are not implemented for VibeVoice-1.5B",
}


@register_tts_adapter
class VibeVoiceTTSAdapter(ARTTSAdapter):
    """Build *ordered* reference-audio prompts for non-Realtime VibeVoice."""

    name = "vibevoice"
    stage_keys = frozenset({"vibevoice"})
    output_policy = OutputPolicy(expose_finish_reason=True)

    @staticmethod
    def _parse_script(text: str) -> tuple[list[tuple[int, str]], int]:
        """Return lines with speaker IDs canonicalized by first appearance."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Input text cannot be empty")

        matches = [_SPEAKER_LINE.fullmatch(line) for line in lines]
        if not any(matches):
            # Plain text is a single-speaker script. Preserve embedded newlines
            # as spaces so one input produces one deterministic speaker line.
            return [(0, " ".join(lines))], 1
        if not all(matches):
            raise ValueError(
                "VibeVoice input must be either plain text or contain only "
                "`Speaker N: text` lines; mixed formats are not supported."
            )

        speaker_map: dict[int, int] = {}
        parsed: list[tuple[int, str]] = []
        for match in matches:
            assert match is not None
            source_id = int(match.group(1))
            canonical_id = speaker_map.setdefault(source_id, len(speaker_map))
            parsed.append((canonical_id, match.group(2).strip()))
        return parsed, len(speaker_map)

    @staticmethod
    def _reference_sources(request: OpenAICreateSpeechRequest) -> list[str]:
        ref_audio = request.ref_audio
        if ref_audio is None:
            return []
        return list(ref_audio) if isinstance(ref_audio, list) else [ref_audio]

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if request.seed is not None:
            return (
                "VibeVoice does not support request-level seed determinism; "
                "omit 'seed' to preserve the official global-device RNG semantics."
            )
        for field_name, hint in _UNSUPPORTED_VIBEVOICE_FIELDS.items():
            value = getattr(request, field_name, None)
            # word_timestamps has a non-None False default. Other optional bool
            # fields use None as the default, so an explicit False is still an
            # unsupported model-specific request and must not be ignored.
            is_set = bool(value) if field_name == "word_timestamps" else value is not None
            if is_set:
                message = f"VibeVoice does not support '{field_name}'"
                return f"{message}; {hint}" if hint else message
        try:
            _, num_speakers = self._parse_script(request.input)
            if num_speakers > _OFFICIAL_MAX_SPEAKERS:
                return f"VibeVoice-1.5B supports at most {_OFFICIAL_MAX_SPEAKERS} speakers per request"
            if request.voice is not None:
                if request.ref_audio is not None:
                    return "VibeVoice accepts exactly one of 'voice' or 'ref_audio', not both"
                if num_speakers != 1:
                    return "VibeVoice uploaded 'voice' is only supported for single-speaker scripts"
                voice_lower = request.voice.lower()
                speaker_info = getattr(self.ctx.server, "uploaded_speakers", {}).get(voice_lower)
                if speaker_info is None:
                    return (
                        f"Unknown VibeVoice voice '{request.voice}'. Upload an audio voice first via "
                        "POST /v1/audio/voices, or use 'ref_audio'."
                    )
                if speaker_info.get("embedding_source") == "direct":
                    return (
                        f"Uploaded voice '{request.voice}' uses a speaker embedding, which VibeVoice "
                        "does not support; re-upload it with an audio file."
                    )
            extra_params = request.extra_params or {}
            unknown_keys = sorted(
                (key for key in extra_params if key not in VIBEVOICE_RUNTIME_CONTROL_KEYS),
                key=str,
            )
            if unknown_keys:
                return f"Unsupported VibeVoice extra_params: {unknown_keys}"
            if "guidance_scale" in extra_params:
                validate_guidance_scale(extra_params["guidance_scale"])
            if "num_diffusion_steps" in extra_params:
                validate_num_diffusion_steps(extra_params["num_diffusion_steps"])
        except ValueError as exc:
            return str(exc)

        references = self._reference_sources(request)
        if request.voice is None and references:
            if len(references) != num_speakers:
                return f"VibeVoice found {num_speakers} speakers but received {len(references)} reference audios"

            validate_format = getattr(self.ctx.server, "_validate_ref_audio_format", None)
            if callable(validate_format):
                for reference in references:
                    if error := validate_format(reference):
                        return error
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > 40500:
                return "max_new_tokens cannot exceed 40500"
        return None

    @staticmethod
    def _validate_resolved_reference(waveform: object, sample_rate: object) -> tuple[np.ndarray, int]:
        waveform_array = np.asarray(waveform, dtype=np.float32)
        if waveform_array.ndim not in (1, 2):
            raise ValueError(f"VibeVoice reference audio must be one- or two-dimensional, got {waveform_array.shape}.")
        num_samples = int(waveform_array.shape[0] if waveform_array.ndim == 1 else max(waveform_array.shape))
        sample_rate_value = int(cast(Any, sample_rate))
        if sample_rate_value <= 0:
            raise ValueError(f"VibeVoice reference audio sample rate must be positive, got {sample_rate_value}.")
        if num_samples <= 0:
            raise ValueError("VibeVoice reference audio is empty.")
        duration = num_samples / sample_rate_value
        if duration > MAX_AUDIO_SECONDS:
            raise ValueError(f"VibeVoice reference audio is {duration:.2f}s; the maximum is {MAX_AUDIO_SECONDS}s.")
        return waveform_array, sample_rate_value

    async def _resolve_reference(self, source: str) -> tuple[np.ndarray, int]:
        model_config = self.ctx.server.model_config
        connector = MediaConnector(
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        return self._validate_resolved_reference(*await connector.fetch_audio_async(source))

    async def _resolve_default_reference(self, index: int) -> tuple[np.ndarray, int]:
        path = get_default_reference_audio_path(index)
        # Framework-owned media is trusted independently of the user-facing
        # --allowed-local-media-path sandbox. Restrict this connector to the
        # package's VibeVoice asset directory rather than widening that sandbox.
        connector = MediaConnector(allowed_local_media_path=str(path.parent))
        return self._validate_resolved_reference(*await connector.fetch_audio_async(path.as_uri()))

    @staticmethod
    def _render_prompt(parsed: list[tuple[int, str]], num_speakers: int) -> str:
        voice_prompt = " Voice input:\n" + "".join(
            f" Speaker {speaker_id}:{_REFERENCE_SEGMENT}\n" for speaker_id in range(num_speakers)
        )
        text_prompt = " Text input:\n" + "".join(f" Speaker {speaker_id}: {text}\n" for speaker_id, text in parsed)
        return f"{_SYSTEM_PROMPT}{voice_prompt}{text_prompt} Speech output:\n{AUDIO_BOS_TOKEN}"

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        parsed, num_speakers = self._parse_script(request.input)
        if request.voice is not None:
            if request.ref_audio is not None:
                raise ValueError("VibeVoice accepts exactly one of 'voice' or 'ref_audio', not both")
            apply_uploaded_speaker = getattr(self.ctx.server, "_apply_uploaded_speaker", None)
            if not callable(apply_uploaded_speaker):
                raise RuntimeError("VibeVoice uploaded voice resolution is unavailable")
            if error := apply_uploaded_speaker(request):
                raise ValueError(error)
            # VibeVoice conditions only on reference audio. The generic voice
            # registry may attach a stored transcript for other TTS models;
            # clear those internal convenience fields so the resolved request
            # has one canonical source (ref_audio) and remains consistent with
            # VibeVoice's public voice/ref_text validation contract.
            request.voice = None
            request.ref_text = None
        references = self._reference_sources(request)
        # Supplying any explicit references retains the strict one-per-speaker
        # contract. Only an entirely omitted ref_audio field selects the four
        # bundled defaults, in canonical speaker first-appearance order.
        if references:
            if len(references) != num_speakers:
                raise ValueError(
                    f"VibeVoice found {num_speakers} speakers but received {len(references)} reference audios"
                )
            audio_items = [await self._resolve_reference(source) for source in references]
        else:
            audio_items = [await self._resolve_default_reference(index) for index in range(num_speakers)]
        prompt = {
            "prompt": self._render_prompt(parsed, num_speakers),
            "multi_modal_data": {"audio": audio_items},
        }
        return PreparedRequest(
            prompt=prompt,
            model_type=self.name,
        )

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        # Pinned vLLM forwards multi_modal_uuids through its token-prompt path,
        # but drops them from its text-prompt path. Supplying both text and
        # token IDs is a model-specific workaround that preserves request UUIDs
        # without patching shared vLLM/Omni runtime code.
        tokenizer = get_stage0_tokenizer(self.ctx.engine_client)
        return list(tokenizer.encode(prompt, add_special_tokens=False))

    def finalize_prepared_request(
        self,
        prepared: PreparedRequest,
        request_id: str,
    ) -> PreparedRequest:
        audio_items = prepared.prompt.get("multi_modal_data", {}).get("audio", [])
        prepared.prompt["prompt_token_ids"] = self._tokenize_prompt(prepared.prompt["prompt"])
        prepared.prompt["multi_modal_uuids"] = {
            "audio": [f"{request_id}:audio:{item_idx}" for item_idx in range(len(audio_items))]
        }
        return prepared

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        if not sampling_params_list:
            return sampling_params_list
        resolved = list(sampling_params_list)
        params = resolved[0]
        if isinstance(params, dict):
            params = params.copy()
            params.update(
                temperature=0.0,
                allowed_token_ids=list(VIBEVOICE_VALID_TOKEN_IDS),
                stop_token_ids=[151643],
                detokenize=False,
            )
            if request.max_new_tokens is not None:
                params["max_tokens"] = request.max_new_tokens
        else:
            clone = getattr(params, "clone", None)
            params = clone() if callable(clone) else copy.copy(params)
            params.temperature = 0.0
            params.allowed_token_ids = list(VIBEVOICE_VALID_TOKEN_IDS)
            params.stop_token_ids = [151643]
            params.detokenize = False
            if request.max_new_tokens is not None:
                params.max_tokens = request.max_new_tokens
            if hasattr(params, "_all_stop_token_ids"):
                # Model-specific replace semantics: caller stop IDs are not
                # retained. vLLM may add the model EOS later.
                params._all_stop_token_ids = {151643}
        resolved[0] = params
        return resolved


__all__ = ["VibeVoiceTTSAdapter"]
