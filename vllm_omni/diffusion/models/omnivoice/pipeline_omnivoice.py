# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline that runs the full text-to-speech flow:
  text → tokenize → 32-step iterative unmasking → 8-codebook tokens → DAC decode → 24kHz audio

Uses request-mode execution (all steps in one forward() call).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
from tokenizers import Tokenizer as HFTokenizer
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
from vllm_omni.utils.speaker_cache import get_speaker_cache

try:
    from transformers import HiggsAudioV2TokenizerModel
except ImportError:
    HiggsAudioV2TokenizerModel = None

import torchaudio

logger = init_logger(__name__)


def get_omnivoice_post_process_func(od_config: OmniDiffusionConfig):
    """Post-processing: convert audio tensor to numpy for WAV encoding."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "pt":
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


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

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
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
                            error=f"OmniVoice voice cloning supports a single reference audio; got {len(audio_field)}"
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

        # Estimate target duration
        target_len = self.duration_estimator.estimate_duration(text, "Nice to meet you.", 25)
        target_len = max(1, int(target_len))

        # Build text prompt with control tokens
        style = f"<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
        if ref_text:
            full_text = f"{ref_text} {text}"
        else:
            full_text = text
        full_prompt = f"{style}<|text_start|>{full_text}<|text_end|>"
        encoding = self.tokenizer.encode(full_prompt)
        text_tokens = torch.tensor(encoding.ids, dtype=torch.long, device=device)
        text_len = text_tokens.shape[0]

        # Encode reference audio tokens if provided (with voice caching)
        ref_audio_tokens = None
        if ref_audio is not None:
            if self.audio_tokenizer is None:
                raise RuntimeError(
                    "Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'"
                )
            # Check speaker cache first
            _cache_key = None
            if voice_name:
                _cache_key = self._speaker_cache.make_cache_key(
                    voice_name,
                    model_type="omnivoice",
                    created_at=int(prompt.get("voice_created_at") or 0),
                )
                cached = self._speaker_cache.get(_cache_key)
                if cached is not None:
                    ref_audio_tokens = cached["ref_audio_tokens"].to(device)
                    _cache_key = None  # hit → don't store again
                    logger.debug("Speaker cache HIT for OmniVoice speaker '%s'", voice_name)

            if ref_audio_tokens is None:
                audio_signal, sr = ref_audio
                if isinstance(audio_signal, np.ndarray):
                    audio_signal = torch.from_numpy(audio_signal).float()
                ref_audio_tokens = self._encode_ref_audio(audio_signal, int(sr)).to(device)

                # Store in cache for next request
                if _cache_key is not None:
                    self._speaker_cache.put(_cache_key, {"ref_audio_tokens": ref_audio_tokens.cpu()})
                    logger.debug("Speaker cache STORE for OmniVoice speaker '%s'", voice_name)

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
        )

        # Decode tokens to audio
        audio = self.decoder(tokens)  # [1, 1, samples]

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
