# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI Speech input adapter for the native two-stage Zonos3 pipeline."""

import asyncio
import copy
from typing import TYPE_CHECKING

import torch

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    SpeechServingContext,
)
from vllm_omni.model_executor.models.zonos3.ar_config import (
    Zonos3Architecture,
)
from vllm_omni.model_executor.models.zonos3.prompt import Zonos3PromptBuilder, condition_ids

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_EXTRA_KEYS = frozenset(
    {
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repetition_penalty",
        "repetition_window",
        "repetition_codebooks",
        "min_frames",
        "eoa_policy",
        "condition_values",
        "condition_buckets",
        "loudness",
        "fmax",
        "speaking_rate",
        "silence_prefill_frames",
        "text_normalization",
        "reference_codes",
        "chunk_size",
        "chunk_ramp",
        "first_chunk_codebooks",
    }
)


@register_tts_adapter
class Zonos3TTSAdapter(ARTTSAdapter):
    name = "zonos3"
    stage_keys = frozenset({"zonos3_ar"})
    model_archs = frozenset({"Zonos3ForCausalLM"})
    supports_reference_codes = True

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        model_config = ctx.server.model_config
        hf = model_config.hf_config
        self.architecture = Zonos3Architecture.from_params(hf.zonos3_params)
        self.builder = Zonos3PromptBuilder(
            self.architecture, model_config.model, hf.zonos3_lm_subdir, model_config.revision
        )

    @classmethod
    def stage_serves_speech(cls, model_stage: str | None, all_stage_keys: frozenset[str]) -> bool:
        return "zonos3_codec" in all_stage_keys

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        uploaded_voice = request.voice is not None and request.ref_audio is None
        error = self.ctx.server._apply_uploaded_speaker(request)
        if error:
            return error
        if not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is not None:
            if not isinstance(request.ref_audio, str):
                return "Zonos3 speaker cloning requires one reference audio source"
            if self.architecture.speaker_embedding_codec != "ownmect_s2":
                return "This Zonos3 checkpoint has no supported raw-reference speaker encoder"
            if request.speaker_embedding is not None or (request.extra_params or {}).get("reference_codes") is not None:
                return "Provide one Zonos3 cloning input: ref_audio, speaker_embedding or reference_codes"
            if request.ref_audio_codes is not None:
                return "Provide one Zonos3 cloning input: ref_audio, speaker_embedding or reference codes"
            error = self.ctx.server._validate_ref_audio_format(request.ref_audio)
            if error:
                return error
        if uploaded_voice and request.ref_audio is not None and request.speaker_embedding is None:
            return (
                "Zonos3 uploaded audio voices need the unsupported reference-prefix encoder; "
                "provide ref_audio with a supported checkpoint or a speaker_embedding"
            )
        if request.speaker_embedding is not None:
            if self.architecture.speaker_embedding_codec != "ownmect_s2":
                return "This Zonos3 checkpoint has no supported speaker embedding encoder"
            speaker = request.speaker_embedding
            if len(speaker) == 1 and isinstance(speaker[0], list):
                speaker = speaker[0]
            if len(speaker) != self.architecture.speaker_embedding_dim or not all(
                torch.isfinite(torch.tensor(value)) for value in speaker
            ):
                return f"Zonos3 speaker_embedding must contain {self.architecture.speaker_embedding_dim} finite values"
        if request.ref_audio_codes is not None:
            if not self.architecture.speaker_reference_prefix_enabled:
                return "This Zonos3 checkpoint does not support reference-code prefix cloning"
            codes = request.ref_audio_codes
            if len(codes) != self.architecture.n_codebooks:
                return "Zonos3 ref_audio_codes must have shape [10, frames]"
            if any(code < 0 or code >= self.architecture.codebook_size for book in codes for code in book):
                return "Zonos3 reference code is outside the codec vocabulary"
            if request.speaker_embedding is not None or (request.extra_params or {}).get("reference_codes") is not None:
                return "Provide only one Zonos3 reference-code field"
            if len(codes[0]) > 900:
                return "Zonos3 ref_audio_codes support at most 900 frames"
        if len((request.extra_params or {}).get("reference_codes", [])) > 900:
            return "Zonos3 reference_codes support at most 900 frames"
        if request.word_timestamps:
            return "Zonos3's two-stage service does not support a forced-aligner stage"
        if request.voice and request.speaker_embedding is None and request.ref_audio is None:
            return (
                "Zonos3 has no built-in preset voice; provide speaker_embedding "
                "or omit voice for unconditioned synthesis"
            )
        extra = request.extra_params or {}
        if set(extra) - _EXTRA_KEYS:
            return f"Unsupported Zonos3 extra_params: {sorted(set(extra) - _EXTRA_KEYS)}"
        try:
            self.builder.canonical_language(request.language)
            values = dict(extra.get("condition_values") or {})
            values.update({name: extra[name] for name in self.architecture.condition_names if name in extra})
            condition_ids(self.architecture, values, extra.get("condition_buckets"))
            if not isinstance(extra.get("text_normalization", True), bool):
                raise ValueError("text_normalization must be boolean")
            frames = extra.get("silence_prefill_frames", 3)
            if not isinstance(frames, int) or not 0 <= frames <= 30:
                raise ValueError("silence_prefill_frames must be in 0..30")
            chunk_size = extra.get("chunk_size", 15)
            chunk_ramp = extra.get("chunk_ramp", (1, 2, 4, 8, 15))
            first_chunk_codebooks = extra.get("first_chunk_codebooks", 10)
            if not isinstance(chunk_size, int) or chunk_size < 1:
                raise ValueError("chunk_size must be a positive integer")
            if not isinstance(chunk_ramp, (list, tuple)) or any(
                not isinstance(size, int) or size < 1 for size in chunk_ramp
            ):
                raise ValueError("chunk_ramp must contain positive integer frame counts")
            if not isinstance(first_chunk_codebooks, int) or not 1 <= first_chunk_codebooks <= 10:
                raise ValueError("first_chunk_codebooks must be in 1..10")
        except (ValueError, TypeError) as error:
            return str(error)
        return self._validate_max_new_tokens(request)

    def _build_prompt(self, request: "OpenAICreateSpeechRequest", raw_reference: bool = False) -> dict:
        extra = request.extra_params or {}
        speaker = None
        if raw_reference:
            # The worker replaces this reserved speaker row after encoding.
            speaker = torch.zeros(self.architecture.speaker_embedding_dim, dtype=torch.float32)
        if request.speaker_embedding is not None:
            speaker = torch.tensor(request.speaker_embedding, dtype=torch.float32, device="cpu")
            if speaker.ndim == 2 and speaker.shape[0] == 1:
                speaker = speaker[0]
        reference = extra.get("reference_codes")
        if request.ref_audio_codes is not None:
            if reference is not None:
                raise ValueError("Provide reference codes in only one request field")
            reference = [list(book) for book in zip(*request.ref_audio_codes, strict=True)]
        if reference is not None:
            reference = torch.tensor(reference, device="cpu")
        values = dict(extra.get("condition_values") or {})
        values.update({name: extra[name] for name in self.architecture.condition_names if name in extra})
        prompt = self.builder.build(
            request.input,
            language=request.language,
            normalize=extra.get("text_normalization", True),
            speaker_embedding=speaker,
            reference_codes=reference,
            values=values,
            buckets=extra.get("condition_buckets"),
            silence_frames=extra.get("silence_prefill_frames", 3),
        )
        chunking = {key: extra[key] for key in ("chunk_size", "chunk_ramp", "first_chunk_codebooks") if key in extra}
        if chunking:
            prompt["additional_information"]["codec_chunking"] = chunking
        return prompt

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        waveform, sample_rate = None, None
        if request.ref_audio is not None:
            samples, sample_rate, _ = await self.ctx.server._resolve_ref_audio_array(request.ref_audio)
            waveform = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        prompt = await asyncio.to_thread(self._build_prompt, request, waveform is not None)
        if waveform is not None:
            metadata = prompt["additional_information"]
            metadata.pop("speaker_embedding")
            metadata.update(reference_waveform=waveform, reference_sample_rate=sample_rate)
        return PreparedRequest(
            prompt=prompt, model_type=self.name, output_policy=OutputPolicy(accumulate_nonstreaming=True)
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        result = [copy.copy(params) for params in sampling_params_list]
        if not prompt or not prompt.get("prompt_token_ids"):
            raise ValueError("Zonos3 sampling requires its exact logical prompt length")
        extra = request.extra_params or {}
        options = {
            key: extra[key]
            for key in ("repetition_window", "repetition_codebooks", "min_frames", "eoa_policy")
            if key in extra
        }
        left = min(self.architecture.max_seqlen, self.ctx.server.model_config.max_model_len) - len(
            prompt["prompt_token_ids"]
        )
        if left < 1:
            raise ValueError("Zonos3 prompt leaves no generation context")
        result[0].max_tokens = min(request.max_new_tokens or result[0].max_tokens or 1024, left)
        result[0].top_p = result[0].top_p or 1.0
        result[0].extra_args = options
        result[0].stop_token_ids = [2]
        result[0].detokenize = False
        return result

    def _load_supported_speakers(self) -> list[str]:
        return []

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset(self.builder.languages)

    def _load_codec_frame_rate(self) -> float:
        return 15.0
