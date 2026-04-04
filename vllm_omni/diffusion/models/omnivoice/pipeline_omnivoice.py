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

        # Duration estimator
        self.duration_estimator = RuleDurationEstimator()

        # Generation parameters
        self.num_step = self.config.num_step
        self.guidance_scale = self.config.guidance_scale
        self.t_shift = self.config.t_shift
        self.layer_penalty_factor = self.config.layer_penalty_factor
        self.position_temperature = self.config.position_temperature
        self.class_temperature = self.config.class_temperature
        self.sample_rate = self.config.sample_rate

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Generate speech audio from text.

        Args:
            req: Diffusion request containing text prompt(s).

        Returns:
            DiffusionOutput with audio tensor in .output
        """
        # Extract text from request
        prompt = req.prompts[0] if req.prompts else ""
        if isinstance(prompt, dict):
            text = prompt.get("input", prompt.get("text", str(prompt)))
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

        # Tokenize with control tokens
        style = "<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
        full_prompt = f"{style}<|text_start|>{text}<|text_end|>"
        encoding = self.tokenizer.encode(full_prompt)
        text_tokens = torch.tensor(encoding.ids, dtype=torch.long, device=device)
        text_len = text_tokens.shape[0]

        # Build conditional + unconditional batches [2, 8, max_len]
        text_ids = text_tokens.unsqueeze(0).repeat(num_cb, 1)
        target_ids = torch.full((num_cb, target_len), mask_id, dtype=torch.long, device=device)
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
