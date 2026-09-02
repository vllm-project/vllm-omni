# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline that runs the full text-to-speech flow:
  text → tokenize → 32-step iterative unmasking → 8-codebook tokens → DAC decode → 24kHz audio

Uses request-mode execution (all steps in one forward() call).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import ClassVar

import numpy as np
import torch
import torchaudio
from tokenizers import Tokenizer as HFTokenizer
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.omnivoice.audio import (
    add_reference_punctuation,
    postprocess_generated_audio,
    prepare_reference_audio,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
from vllm_omni.utils.speaker_cache import get_speaker_cache

try:
    from transformers import HiggsAudioV2TokenizerModel
except ImportError:
    HiggsAudioV2TokenizerModel = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

logger = init_logger(__name__)

_ASR_MODEL_NAME = "openai/whisper-large-v3-turbo"
_INLINE_CACHE_MAX_ENTRIES = 8


def _parse_asr_config(additional_config: Mapping[str, object] | None) -> tuple[bool, str, str | None]:
    """Return validated OmniVoice ASR settings from the model config."""
    if additional_config is None:
        additional_config = {}
    if not isinstance(additional_config, Mapping):
        raise TypeError(f"additional_config must be a mapping or None, got {type(additional_config)!r}")

    raw_config = additional_config.get("omnivoice_asr", {})
    if raw_config is None:
        raise TypeError("additional_config['omnivoice_asr'] must be a mapping")
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"additional_config['omnivoice_asr'] must be a mapping, got {type(raw_config)!r}")

    load_asr_on_startup = raw_config.get("load_asr_on_startup", False)
    if not isinstance(load_asr_on_startup, bool):
        raise TypeError("additional_config['omnivoice_asr']['load_asr_on_startup'] must be a bool")

    model_name = raw_config.get("asr_model_name", _ASR_MODEL_NAME)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("additional_config['omnivoice_asr']['asr_model_name'] must be a non-empty string")

    asr_device = raw_config.get("asr_device")
    if asr_device is not None and (not isinstance(asr_device, str) or not asr_device.strip()):
        raise ValueError("additional_config['omnivoice_asr']['asr_device'] must be a non-empty string")

    return load_asr_on_startup, model_name.strip(), asr_device.strip() if asr_device is not None else None


def get_omnivoice_post_process_func(od_config: OmniDiffusionConfig):
    """Post-processing: convert audio tensor to numpy for WAV encoding."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "pt":
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


def _combine_text(text, ref_text: str | None = None) -> str:
    # combine with reference text if not None
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    # filter out newline / carriage-return characters
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> list[int]:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs list of length seq_len.
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer.encode(segment)
            if ids:
                parts.append(ids)
        tag_ids = tokenizer.encode(m.group())
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer.encode(segment)
        if ids:
            parts.append(ids)

    if not parts:
        return tokenizer.encode(text).ids
    else:
        combined = []
        for p in parts:
            combined.extend(p.ids)
    return combined


class OmniVoicePipeline(nn.Module, SupportAudioOutput):
    """OmniVoice text-to-speech pipeline for the diffusion engine.

    Wraps OmniVoiceGenerator (32-step iterative unmasking) and
    OmniVoiceDecoder (HiggsAudioV2 RVQ + DAC) into a single forward() call.
    """

    support_audio_output: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = od_config.model
        self._initialize_asr(getattr(od_config, "additional_config", None))
        self._inline_reference_cache: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()

        # Resolve model path (HF hub ID → local cache)
        if not os.path.isdir(self.model_path):
            from huggingface_hub import snapshot_download

            self.model_path = snapshot_download(self.model_path)

        # Load OmniVoice config
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as f:
            hf_config = json.load(f)
        self.config = OmniVoiceConfig(**hf_config)

        # Build generator and decoder
        self.generator = OmniVoiceGenerator(self.config)
        self.decoder = OmniVoiceDecoder(self.config)

        # Tokenizer (low-level, avoids HF tokenizer extra_special_tokens issue)
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)

        # Audio tokenizer for voice cloning (requires transformers>=5.3)
        if HiggsAudioV2TokenizerModel is not None:
            audio_tokenizer_path = os.path.join(self.model_path, "audio_tokenizer")
            self.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                audio_tokenizer_path, device_map=self.device
            ).eval()
            logger.info("HiggsAudioV2 tokenizer loaded for voice cloning on %s", self.device)
        else:
            self.audio_tokenizer = None
            logger.warning("Voice cloning disabled (requires transformers>=5.3.0).")

        # Duration estimator
        self.duration_estimator = RuleDurationEstimator()

        # Speaker cache for ref_audio_tokens
        self._speaker_cache = get_speaker_cache()

        # Generation parameters
        self.num_step = self.config.num_step
        self.guidance_scale = self.config.guidance_scale
        self.t_shift = self.config.t_shift
        self.layer_penalty_factor = self.config.layer_penalty_factor
        self.position_temperature = self.config.position_temperature
        self.class_temperature = self.config.class_temperature
        self.sample_rate = self.config.sample_rate

    def _initialize_asr(self, additional_config: Mapping[str, object] | None) -> None:
        self._load_asr_on_startup, self._asr_model_name, self._asr_device = _parse_asr_config(additional_config)
        self._asr_pipeline = None
        if self._load_asr_on_startup:
            self._load_asr_pipeline()

    def _load_asr_pipeline(self):
        """Load the configured reference-audio ASR pipeline."""
        if self._asr_pipeline is not None:
            return self._asr_pipeline
        asr_device = self._asr_device if self._asr_device is not None else self.device
        if hf_pipeline is None:
            raise RuntimeError(
                "OmniVoice automatic transcription requires the Hugging Face "
                f"ASR pipeline ({self._asr_model_name!r}) on device {asr_device}."
            )

        asr_dtype = torch.float16 if str(asr_device).lower().startswith(("cuda", "xpu")) else torch.float32
        logger.info(
            "Loading OmniVoice ASR model %s on %s",
            self._asr_model_name,
            asr_device,
        )
        try:
            self._asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self._asr_model_name,
                dtype=asr_dtype,
                device=asr_device,
            )
            # Transformers keeps a newly loaded pipeline model on CPU when a
            # torch.distributed process group is already initialized. vLLM
            # workers always have one, so explicitly honor the configured
            # device after construction, as other auxiliary models do here.
            target_device = torch.device(asr_device)
            self._asr_pipeline.model = self._asr_pipeline.model.to(target_device)
            self._asr_pipeline.device = target_device
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load OmniVoice ASR model {self._asr_model_name!r} on device {asr_device}: {exc}"
            ) from exc
        logger.info("OmniVoice ASR model loaded on %s", asr_device)
        return self._asr_pipeline

    @torch.inference_mode()
    def _transcribe_ref_audio(self, ref_audio) -> str:
        """Transcribe a reference waveform with the lazily loaded ASR model."""
        waveform, sr = ref_audio
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.squeeze(np.array(waveform, copy=True))
        result = self._load_asr_pipeline()(
            {
                "array": waveform,
                "sampling_rate": int(sr),
            }
        )
        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError("OmniVoice ASR returned a malformed result without a 'text' field.")
        transcript = result["text"]
        if not isinstance(transcript, str):
            raise RuntimeError("OmniVoice ASR returned a malformed result: 'text' must be a string.")
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("OmniVoice ASR returned an empty reference transcription.")
        return transcript

    def _resolve_ref_text(self, ref_audio, ref_text: str | None) -> str | None:
        """Resolve missing reference text only when reference audio is present."""
        if ref_audio is not None and (ref_text is None or not ref_text.strip()):
            logger.debug("Automatically transcribing OmniVoice reference audio")
            return self._transcribe_ref_audio(ref_audio)
        return ref_text

    def _estimate_target_len(
        self,
        text: str,
        ref_text: str | None,
        ref_audio_tokens: torch.Tensor | None,
    ) -> int:
        """Estimate target audio tokens using resolved voice-clone conditioning."""
        if ref_audio_tokens is None or not ref_text:
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25
        else:
            num_ref_audio_tokens = ref_audio_tokens.size(-1)

        target_len = self.duration_estimator.estimate_duration(
            text,
            ref_text,
            num_ref_audio_tokens,
        )
        return max(1, int(target_len))

    def _encode_ref_audio(self, audio_signal: torch.Tensor, sr: int) -> torch.Tensor:
        """Encode reference audio to 8-codebook tokens for voice cloning."""
        if self.audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not available for voice cloning")
        if audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)
        # Resample to tokenizer's expected sample rate
        target_sr = self.audio_tokenizer.config.sample_rate
        if sr != target_sr:
            audio_signal = torchaudio.functional.resample(audio_signal, sr, target_sr)
        # Ensure mono [B, 1, samples]
        if audio_signal.dim() == 2:
            audio_signal = audio_signal.unsqueeze(1)
        with torch.inference_mode():
            tokens = self.audio_tokenizer.encode(
                audio_signal.to(self.audio_tokenizer.device), return_dict=False
            )  # [B, 8, T_ref]
            tokens = tokens.squeeze(0)  # [8, T_ref]
        return tokens

    @staticmethod
    def _inline_cache_key(prepared_audio, preparation_mode: str) -> tuple[object, ...]:
        waveform = np.ascontiguousarray(prepared_audio.waveform, dtype=np.float32)
        digest = hashlib.sha256(waveform.tobytes()).digest()
        return (
            preparation_mode,
            digest,
            tuple(waveform.shape),
            int(prepared_audio.sample_rate),
            float(prepared_audio.original_rms),
        )

    def _get_inline_cache(self, key: tuple[object, ...]) -> dict[str, object] | None:
        cached = self._inline_reference_cache.pop(key, None)
        if cached is not None:
            self._inline_reference_cache[key] = cached
        return cached

    def _put_inline_cache(self, key: tuple[object, ...], artifacts: dict[str, object]) -> None:
        self._inline_reference_cache[key] = artifacts
        self._inline_reference_cache.move_to_end(key)
        while len(self._inline_reference_cache) > _INLINE_CACHE_MAX_ENTRIES:
            self._inline_reference_cache.popitem(last=False)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Generate speech audio from text, optionally with voice cloning.

        Accepts either a plain text prompt or a structured dict:
          {"text": "...", "ref_audio": (samples, sr), "ref_text": "...",
           "lang": "...", "instruct": "..."}
        """
        prompt = req.prompts[0] if req.prompts else ""
        ref_audio = None
        ref_text = None
        lang = "None"
        instruct = "None"
        extra = req.sampling_params.extra_args or {}
        seed = extra.get("seed", None)

        voice_name = None
        if isinstance(prompt, dict):
            # Top-level keys (used by serving_speech.py /v1/audio/speech path)
            text = prompt.get("input") or prompt.get("text") or prompt.get("prompt")
            ref_audio = prompt.get("ref_audio")
            ref_text = prompt.get("ref_text")
            voice_name = prompt.get("voice_name")
            lang = prompt.get("lang")
            instruct = prompt.get("instruct")
            # OmniTextPrompt format (used by offline Omni.generate path):
            # ref_audio comes via multi_modal_data["audio"] and the rest via
            # mm_processor_kwargs. Fall back to those when top-level keys are
            # absent so both invocation styles work.
            mm_data = prompt.get("multi_modal_data") or {}
            mm_kwargs = prompt.get("mm_processor_kwargs") or {}
            if ref_audio is None:
                audio_field = mm_data.get("audio")
                # Standard multimodal shape allows a list of audios; OmniVoice
                # voice cloning conditions on a single reference clip, so
                # unwrap a length-1 list and reject multi-reference prompts up
                # front (otherwise a list would later crash inside
                # ``_encode_ref_audio`` when it calls ``audio.dim()``).
                if isinstance(audio_field, list):
                    if len(audio_field) == 1:
                        audio_field = audio_field[0]
                    elif len(audio_field) > 1:
                        return DiffusionOutput(
                            error=f"OmniVoice voice cloning supports a single reference audio; got {len(audio_field)}"  # noqa: E501
                        )
                    else:
                        audio_field = None
                if audio_field is not None:
                    if isinstance(audio_field, tuple) and len(audio_field) == 2:
                        ref_audio = audio_field
                    else:
                        sr = mm_kwargs.get("sample_rate") or self.sample_rate
                        ref_audio = (audio_field, int(sr))
            if ref_text is None:
                ref_text = mm_kwargs.get("ref_text")
            if lang is None:
                lang = mm_kwargs.get("lang")
            if instruct is None:
                instruct = mm_kwargs.get("instruct")

            if isinstance(ref_text, str):
                ref_text = ref_text.strip() or None

            if not text:
                return DiffusionOutput(error="Empty text prompt")
            lang = lang or "None"
            instruct = instruct or "None"
        else:
            text = str(prompt)
            if not text:
                return DiffusionOutput(error="Empty text prompt")

        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        ref_audio_tokens = None
        reference_rms = None
        if ref_audio is not None:
            needs_asr = ref_text is None
            if self.audio_tokenizer is None:
                raise RuntimeError(
                    "Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'"
                )
            preparation_mode = "asr" if needs_asr else "explicit"
            _cache_key = None
            if voice_name:
                _cache_key = self._speaker_cache.make_cache_key(
                    voice_name,
                    model_type=f"omnivoice_{preparation_mode}",
                    created_at=int(prompt.get("voice_created_at") or 0),
                )
                cached = self._speaker_cache.get(_cache_key)
                if cached is not None and (not needs_asr or cached.get("ref_text")):
                    ref_audio_tokens = cached["ref_audio_tokens"].to(device)
                    reference_rms = cached["reference_rms"]
                    if needs_asr:
                        ref_text = cached["ref_text"]
                    _cache_key = None  # hit → don't store again
                    logger.debug("Speaker cache HIT for OmniVoice speaker '%s'", voice_name)

            if ref_audio_tokens is None:
                audio_signal, sample_rate = ref_audio
                try:
                    prepared_audio = prepare_reference_audio(
                        audio_signal,
                        int(sample_rate),
                        target_sample_rate=self.audio_tokenizer.config.sample_rate,
                        hop_length=self.audio_tokenizer.config.hop_length,
                        trim_long=needs_asr,
                    )
                except (RuntimeError, ValueError) as exc:
                    return DiffusionOutput(error=str(exc))
                reference_rms = prepared_audio.original_rms
                reference_duration = prepared_audio.waveform.shape[-1] / prepared_audio.sample_rate
                if reference_duration > 20.0:
                    logger.warning(
                        "OmniVoice reference audio is %.1fs long (>20s); this may increase memory use "
                        "and reduce cloning quality.",
                        reference_duration,
                    )

                inline_cache_key = None
                if not voice_name:
                    inline_cache_key = self._inline_cache_key(prepared_audio, preparation_mode)
                    cached = self._get_inline_cache(inline_cache_key)
                    if cached is not None:
                        ref_audio_tokens = cached["ref_audio_tokens"].to(device)
                        reference_rms = cached["reference_rms"]
                        if needs_asr:
                            ref_text = cached["ref_text"]
                        logger.debug("Inline OmniVoice reference cache HIT")

                if ref_audio_tokens is None and needs_asr:
                    try:
                        ref_text = self._resolve_ref_text(
                            (prepared_audio.waveform, prepared_audio.sample_rate),
                            ref_text,
                        )
                    except (RuntimeError, ValueError) as exc:
                        return DiffusionOutput(error=str(exc))
            if ref_audio_tokens is None:
                ref_audio_tokens = self._encode_ref_audio(
                    torch.from_numpy(prepared_audio.waveform),
                    prepared_audio.sample_rate,
                ).to(device)

                # Store named and inline entries for the matching preparation mode.
                if _cache_key is not None:
                    self._speaker_cache.put(
                        _cache_key,
                        {
                            "ref_audio_tokens": ref_audio_tokens.cpu(),
                            "ref_text": ref_text if needs_asr else None,
                            "reference_rms": reference_rms,
                        },
                    )
                    logger.debug("Speaker cache STORE for OmniVoice speaker '%s'", voice_name)
                elif not voice_name:
                    self._put_inline_cache(
                        inline_cache_key,
                        {
                            "ref_audio_tokens": ref_audio_tokens.cpu(),
                            "ref_text": ref_text if needs_asr else None,
                            "reference_rms": reference_rms,
                        },
                    )
                    logger.debug("Inline OmniVoice reference cache STORE")

            if ref_text:
                ref_text = add_reference_punctuation(ref_text)

        target_len = self._estimate_target_len(text, ref_text, ref_audio_tokens)

        # Build text prompt with control tokens
        style_text = f"<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        style_tokens = self.tokenizer.encode(style_text).ids
        text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.tokenizer)
        encoding_ids = style_tokens + text_tokens
        text_tokens = torch.tensor(encoding_ids, dtype=torch.long, device=device)
        text_len = text_tokens.shape[0]

        # Build conditional + unconditional batches [2, 8, max_len]
        text_ids = text_tokens.unsqueeze(0).repeat(num_cb, 1)
        target_ids = torch.full((num_cb, target_len), mask_id, dtype=torch.long, device=device)

        if ref_audio_tokens is not None:
            cond_ids = torch.cat([text_ids, ref_audio_tokens, target_ids], dim=1)
        else:
            cond_ids = torch.cat([text_ids, target_ids], dim=1)
        cond_len = cond_ids.shape[1]
        uncond_ids = target_ids.clone()
        uncond_len = target_len
        max_len = max(cond_len, uncond_len)
        if uncond_len < max_len:
            pad = torch.full(
                (num_cb, max_len - uncond_len),
                mask_id,
                dtype=torch.long,
                device=device,
            )
            uncond_ids = torch.cat([uncond_ids, pad], dim=1)

        batch_input_ids = torch.stack([cond_ids, uncond_ids])

        batch_audio_mask = torch.zeros(2, max_len, dtype=torch.bool, device=device)
        batch_audio_mask[0, text_len:cond_len] = True
        batch_audio_mask[1, :uncond_len] = True

        batch_attn_mask = torch.zeros(2, 1, max_len, max_len, dtype=torch.bool, device=device)
        batch_attn_mask[0, :, :cond_len, :cond_len] = True
        batch_attn_mask[1, :, :uncond_len, :uncond_len] = True

        # Run 32-step iterative unmasking
        tokens = self.generator(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attn_mask,
            target_lens=[target_len],
            num_step=self.num_step,
            guidance_scale=self.guidance_scale,
            t_shift=self.t_shift,
            layer_penalty_factor=self.layer_penalty_factor,
            position_temperature=self.position_temperature,
            class_temperature=self.class_temperature,
            seed=seed,
        )
        # Decode tokens to audio
        audio = self.decoder(tokens)  # [1, 1, samples]
        if ref_audio_tokens is None:
            return DiffusionOutput(output=audio)

        audio_np = audio.squeeze(0).detach().cpu().float().numpy()
        audio_np = postprocess_generated_audio(
            audio_np,
            sample_rate=self.sample_rate,
            reference_rms=reference_rms,
        )
        audio = torch.from_numpy(audio_np).unsqueeze(0)

        return DiffusionOutput(output=audio)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from model directory (not from the iterator).

        The diffusion model loader passes HF safetensors weights, but OmniVoice
        has custom weight names (llm.* → generator.*, audio_tokenizer.* → decoder.*).
        We load from model_path directly and return all param names to satisfy
        the loader's "all weights initialized" check.
        """
        # Consume the iterator (required by the loader contract)
        for _ in weights:
            pass

        device = self.device
        self.generator.load_weights(self.model_path, device)
        self.generator = self.generator.to(device).eval()
        self.decoder.load_weights(self.model_path, device)
        logger.info("OmniVoice pipeline loaded on %s", device)

        # Return all parameter names to indicate they're initialized
        return {name for name, _ in self.named_parameters()}
