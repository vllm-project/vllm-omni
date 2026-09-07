# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
OmniVoice TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline that runs the full text-to-speech flow:
  text → tokenize → 32-step iterative unmasking → 8-codebook tokens → DAC decode → 24kHz audio

Uses request-mode execution (all steps in one forward() call).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from tokenizers import Tokenizer as HFTokenizer
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    OmniVoiceGenerator,
    _build_cu_seqs,
    _get_time_steps,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
from vllm_omni.utils.speaker_cache import get_speaker_cache

try:
    from transformers import HiggsAudioV2TokenizerModel
except ImportError:
    HiggsAudioV2TokenizerModel = None

import torchaudio

logger = init_logger(__name__)


@dataclass
class _PreparedOmniVoiceRequest:
    input_ids: torch.Tensor
    audio_mask: torch.Tensor
    cond_len: int
    target_len: int
    seed: int | None


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
    supports_request_batch: ClassVar[bool] = True
    supports_step_execution: ClassVar[bool] = True

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
        self.generator = OmniVoiceGenerator(self.config, od_config)
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

    def _prepare_request_input(
        self,
        prompt: Any,
        extra: dict[str, Any],
    ) -> _PreparedOmniVoiceRequest | DiffusionOutput:
        """Build one request's conditional/unconditional model inputs."""
        ref_audio = None
        ref_text = None
        lang = "None"
        instruct = "None"
        voice_name = None
        seed = extra.get("seed", None)

        if isinstance(prompt, dict):
            text = prompt.get("input") or prompt.get("text") or prompt.get("prompt")
            ref_audio = prompt.get("ref_audio")
            ref_text = prompt.get("ref_text")
            voice_name = prompt.get("voice_name")
            lang = prompt.get("lang")
            instruct = prompt.get("instruct")
            mm_data = prompt.get("multi_modal_data") or {}
            mm_kwargs = prompt.get("mm_processor_kwargs") or {}
            if ref_audio is None:
                audio_field = mm_data.get("audio")
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

        target_len = self.duration_estimator.estimate_duration(text, "Nice to meet you.", 25)
        target_len = max(1, int(target_len))

        style_text = f"<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        style_tokens = self.tokenizer.encode(style_text).ids
        text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.tokenizer)
        encoding_ids = style_tokens + text_tokens
        text_tokens_tensor = torch.tensor(encoding_ids, dtype=torch.long, device=self.device)
        text_len = text_tokens_tensor.shape[0]

        ref_audio_tokens = None
        if ref_audio is not None:
            if self.audio_tokenizer is None:
                raise RuntimeError(
                    "Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'"
                )
            cache_key = None
            if voice_name:
                cache_key = self._speaker_cache.make_cache_key(
                    voice_name,
                    model_type="omnivoice",
                    created_at=int(prompt.get("voice_created_at") or 0),
                )
                cached = self._speaker_cache.get(cache_key)
                if cached is not None:
                    ref_audio_tokens = cached["ref_audio_tokens"].to(self.device)
                    cache_key = None
                    logger.debug("Speaker cache HIT for OmniVoice speaker '%s'", voice_name)

            if ref_audio_tokens is None:
                audio_signal, sr = ref_audio
                if isinstance(audio_signal, np.ndarray):
                    audio_signal = torch.from_numpy(audio_signal).float()
                ref_audio_tokens = self._encode_ref_audio(audio_signal, int(sr)).to(self.device)
                if cache_key is not None:
                    self._speaker_cache.put(cache_key, {"ref_audio_tokens": ref_audio_tokens.cpu()})
                    logger.debug("Speaker cache STORE for OmniVoice speaker '%s'", voice_name)

        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        text_ids = text_tokens_tensor.unsqueeze(0).repeat(num_cb, 1)
        target_ids = torch.full((num_cb, target_len), mask_id, dtype=torch.long, device=self.device)
        cond_ids = (
            torch.cat([text_ids, ref_audio_tokens, target_ids], dim=1)
            if ref_audio_tokens is not None
            else torch.cat([text_ids, target_ids], dim=1)
        )
        cond_len = cond_ids.shape[1]
        uncond_ids = target_ids.clone()
        input_ids = torch.cat([cond_ids, uncond_ids], dim=1).transpose(0, 1).contiguous()

        max_len = input_ids.shape[0]
        audio_mask = torch.zeros(max_len, dtype=torch.bool, device=self.device)
        audio_mask[text_len:] = True

        return _PreparedOmniVoiceRequest(
            input_ids=input_ids,
            audio_mask=audio_mask,
            cond_len=cond_len,
            target_len=target_len,
            seed=seed,
        )

    def _collate_request_inputs(
        self,
        prepared_requests: Sequence[_PreparedOmniVoiceRequest],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Pack request-major [cond, uncond] token sequences."""
        input_ids: list[torch.Tensor] = []
        audio_masks: list[torch.Tensor] = []
        cond_lens: list[int] = []

        for request in prepared_requests:
            input_id = request.input_ids
            audio_mask = request.audio_mask
            cond_len = request.cond_len
            input_ids.append(input_id)
            audio_masks.append(audio_mask)
            cond_lens.append(cond_len)

        input_ids = torch.cat(input_ids, dim=0)
        audio_masks = torch.cat(audio_masks, dim=0)

        return input_ids, audio_masks, cond_lens

    def prepare_encode(self, state: StepRequestState) -> StepRequestState:
        prompt = state.prompt if state.prompt else ""
        extra = state.sampling.extra_args or {}
        prepared = self._prepare_request_input(prompt, extra)
        if isinstance(prepared, DiffusionOutput):
            raise OmniClientError(prepared.error or "OmniVoice request preparation failed")

        prepared_request = prepared
        cond_len = prepared_request.cond_len
        target_len = prepared_request.target_len
        input_ids = prepared_request.input_ids
        audio_mask = prepared_request.audio_mask
        seed = prepared_request.seed
        device = self.device
        mask_id = self.config.audio_mask_id
        num_codebooks = self.config.num_audio_codebook
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        num_step = (
            state.sampling.num_inference_steps if state.sampling.num_inference_steps is not None else self.num_step
        )

        t_shift = self.t_shift

        # Initialize all target tokens as [MASK]
        tokens = torch.full((1, num_codebooks, target_len), mask_id, dtype=torch.long, device=device)

        timesteps = _get_time_steps(0.0, 1.0, num_step + 1, t_shift)

        # Compute unmasking schedule
        schedules = []
        total_mask = target_len * num_codebooks
        rem = total_mask
        sched = []
        for step in range(num_step):
            num = (
                rem
                if step == num_step - 1
                else min(
                    math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                    rem,
                )
            )
            sched.append(int(num))
            rem -= int(num)
        schedules = torch.tensor(sched, dtype=torch.long, device=device)

        layer_ids = torch.arange(num_codebooks, device=device).view(1, -1, 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        guidance_scale = (
            state.sampling.guidance_scale if state.sampling.guidance_scale is not None else self.guidance_scale
        )
        state.latents = input_ids
        state.timesteps = schedules
        state.guidance = guidance_scale
        state.extra["schedules"] = schedules
        state.extra["layer_ids"] = layer_ids
        state.extra["generator"] = generator
        state.extra["t_shift"] = t_shift
        state.extra["cond_len"] = cond_len
        state.extra["target_len"] = target_len
        state.extra["audio_mask"] = audio_mask
        state.extra["tokens"] = tokens
        return state

    def denoise_step(self, input_batch: InputBatch, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any):
        input_ids = input_batch.latents
        use_cuda_graph = self.generator._cuda_graph_fwd is not None and input_ids.is_cuda
        layer_ids = states[0].extra["layer_ids"]

        audio_masks: list[torch.Tensor] = []
        target_lens: list[int] = []
        batch_tokens: list[torch.Tensor] = []
        cond_lens: list[int] = []

        steps: list[int] = []
        schedules: list[torch.Tensor] = []
        generators: list[torch.Generator] = []
        guidance_scales: list[float] = []

        for state in states:
            audio_masks.append(state.extra.get("audio_mask", None))
            cond_lens.append(state.extra["cond_len"])
            target_lens.append(state.extra["target_len"])
            batch_tokens.append(state.extra["tokens"])
            guidance_scales.append(state.guidance)
            generators.append(state.extra.get("generator", None))
            schedules.append(state.extra["schedules"])
            steps.append(state.step_index)

        audio_masks = torch.cat(audio_masks, dim=0)

        B = len(target_lens)
        cu_seqs = _build_cu_seqs(cond_lens, target_lens, input_ids.device)

        position_temperature = self.position_temperature
        class_temperature = self.class_temperature
        layer_penalty_factor = self.layer_penalty_factor
        if use_cuda_graph:
            # Replay a fixed packed-token bucket with dynamic varlen metadata.
            batch_logits = self.generator._cuda_graph_fwd(input_ids, audio_masks, cu_seqs, B)
        else:
            # Run packed eager attention for the current active requests.
            inputs_embeds = self.generator._prepare_embeddings(input_ids, audio_masks)
            hidden_states = self.generator._transformer_forward(
                inputs_embeds,
                cu_seqs,
                max_seqlen=max(cond_lens),
            )
            # fp32 cast deferred to the per-item slices below.
            batch_logits = self.generator._get_logits(hidden_states)
        # batch_logits: [8, total_seq_len, 1025]

        target_offsets: list[int] = []
        target_offset = 0
        for target_len in target_lens:
            target_offsets.append(target_offset)
            target_offset += target_len

        sequence_offsets: list[int] = []
        sequence_offset = 0
        for cond_len, target_len in zip(cond_lens, target_lens):
            sequence_offsets.append(sequence_offset)
            sequence_offset += cond_len + target_len

        for i in range(B):
            k = schedules[i][steps[i]]
            if k <= 0:
                continue

            c_len = cond_lens[i]
            t_len = target_lens[i]

            # Extract logits for target region; upcast only the slices we actually consume.
            request_start = sequence_offsets[i]
            cond_end = request_start + c_len
            uncond_start = cond_end

            # Extract logits for target region; upcast only the slices we actually consume.
            c_logits = batch_logits[:, cond_end - t_len : cond_end, :].unsqueeze(0).to(torch.float32)
            u_logits = batch_logits[:, uncond_start : uncond_start + t_len, :].unsqueeze(0).to(torch.float32)
            sample = batch_tokens[i]
            sample_tokens = sample[..., :t_len]
            self.generator._unmask_one_request(
                c_logits,
                u_logits,
                sample_tokens,
                num_to_unmask=k,
                guidance_scale=guidance_scales[i],
                generator=generators[i],
                class_temperature=class_temperature,
                position_temperature=position_temperature,
                layer_penalty_factor=layer_penalty_factor,
                layer_ids=layer_ids,
            )

            # Mirror update into both cond and uncond input_ids halves for the next step.
            packed_sample_tokens = sample_tokens.squeeze(0).transpose(0, 1)
            input_ids[cond_end - t_len : cond_end] = packed_sample_tokens
            input_ids[uncond_start : uncond_start + t_len] = packed_sample_tokens
            states[i].extra["tokens"] = sample_tokens

        # InputBatch reuses its latents buffer across steps. Returning that
        # same storage would make the Runner persist per-request views into the
        # cached destination; the next make_batch() would then copy overlapping
        # source/destination slices. Break the alias at the lifecycle boundary.
        return input_ids.clone()

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any):
        state.latents = noise_pred
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any):
        tokens = state.extra["tokens"]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        audio = self.decoder(tokens)
        return DiffusionOutput(output=audio)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Generate speech audio from text, optionally with voice cloning.

        Accepts either a plain text prompt or a structured dict:
          {"text": "...", "ref_audio": (samples, sr), "ref_text": "...",
           "lang": "...", "instruct": "..."}
        """
        prepared_requests: list[_PreparedOmniVoiceRequest] = []
        outputs = [None] * len(req.requests)
        prepared_indices: list[int] = []
        for i, request in enumerate(req.requests):
            prompt = request.prompt if request.prompt else ""
            extra = request.sampling_params.extra_args or {}
            prepared = self._prepare_request_input(prompt, extra)
            if isinstance(prepared, DiffusionOutput):
                outputs[i] = prepared
                continue
            prepared_indices.append(i)
            prepared_requests.append(prepared)

        if not prepared_requests:
            return outputs

        batch_target_len = [request.target_len for request in prepared_requests]
        batch_seeds = [request.seed for request in prepared_requests]
        batch_input_ids, batch_audio_mask, batch_cond_lens = self._collate_request_inputs(prepared_requests)
        # Run 32-step iterative unmasking
        sampling = req.requests[0].sampling_params
        num_step = sampling.num_inference_steps if sampling.num_inference_steps is not None else self.num_step
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale is not None else self.guidance_scale
        tokens = self.generator(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            cond_lens=batch_cond_lens,
            target_lens=batch_target_len,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=self.t_shift,
            layer_penalty_factor=self.layer_penalty_factor,
            position_temperature=self.position_temperature,
            class_temperature=self.class_temperature,
            seed=batch_seeds,
        )

        target_offset = 0
        for i, target_len in enumerate(batch_target_len):
            request_tokens = tokens[:, :, target_offset : target_offset + target_len]
            audio = self.decoder(request_tokens)
            outputs[prepared_indices[i]] = DiffusionOutput(output=audio)
            target_offset += target_len
        return outputs

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
