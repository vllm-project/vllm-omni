# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice model for vLLM-Omni two-stage TTS pipeline.

Stage 0 (Generator): Qwen3 backbone + iterative unmasking → 8-codebook tokens
Stage 1 (Decoder): HiggsAudioV2 decoder → 24kHz waveform
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.llm import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Multimodal processing
# ---------------------------------------------------------------------------


class OmniVoiceMultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OmniVoiceConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.ctx.get_hf_config().sample_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class OmniVoiceMultiModalProcessor(BaseMultiModalProcessor[OmniVoiceMultiModalProcessingInfo]):
    """Processes text + optional reference audio for OmniVoice.

    For voice cloning: text + reference audio → tokenized reference
    For auto voice: text only
    """

    def _ensure_cached_runtime_components(self, model_dir: str, config: OmniVoiceConfig) -> None:
        cached_model_dir = getattr(self, "_cached_model_dir", None)
        if cached_model_dir == model_dir:
            return

        from transformers import AutoTokenizer

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Audio tokenizer for encoding reference audio (requires transformers>=5.3)
        audio_tokenizer_path = os.path.join(model_dir, "audio_tokenizer")
        try:
            from transformers import (
                AutoFeatureExtractor,
                HiggsAudioV2TokenizerModel,
            )

            self.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(audio_tokenizer_path, device_map="cpu")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_tokenizer_path)
            self.audio_tokenizer.eval()
        except ImportError:
            self.audio_tokenizer = None
            self.feature_extractor = None
            logger.warning("Voice cloning disabled (requires transformers>=5.3.0).")

        self._cached_model_dir = model_dir

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self._ensure_cached_runtime_components(model_dir, config)

        audio = mm_data.get("audio", None)
        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                audio = audio[0], config.sample_rate

        # Build text prompt with control tokens
        lang = mm_kwargs.get("lang", None)
        instruct = mm_kwargs.get("instruct", None)
        denoise = mm_kwargs.get("denoise", True)
        ref_text = mm_kwargs.get("ref_text", None)

        # Construct the style + text portion
        style_text = ""
        if denoise:
            style_text += "<|denoise|>"
        lang_str = lang if lang else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        # Combine ref_text and main text
        if ref_text:
            full_text = f"{ref_text} {prompt}"
        else:
            full_text = prompt

        text_prompt = f"{style_text}<|text_start|>{full_text}<|text_end|>"
        text_tokens = self.text_tokenizer(text_prompt, return_tensors="pt").input_ids.squeeze(0)  # [N_text]

        if audio is None:
            # Text-only path (auto voice mode)
            return BatchFeature(
                {
                    "input_ids": text_tokens,
                    "input_len": [len(text_tokens)],
                }
            )

        # Voice cloning: encode reference audio to tokens
        audio_signal, sr = audio
        if isinstance(audio_signal, np.ndarray):
            audio_signal = torch.from_numpy(audio_signal).float()
        if audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)

        # Resample to tokenizer sample rate if needed
        if self.feature_extractor is not None:
            target_sr = self.feature_extractor.sampling_rate
            if sr != target_sr:
                audio_signal = torchaudio.functional.resample(audio_signal, sr, target_sr)

        # Encode reference audio to 8-codebook tokens
        if self.audio_tokenizer is None:
            raise RuntimeError("Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'")

        with torch.inference_mode():
            ref_audio_tokens = self.audio_tokenizer.encode(audio_signal)  # [8, T_ref]
            if ref_audio_tokens.dim() == 3:
                ref_audio_tokens = ref_audio_tokens.squeeze(0)  # [8, T_ref]

        ft = BatchFeature(
            {
                "input_ids": text_tokens,
                "ref_audio_tokens": ref_audio_tokens,  # [8, T_ref]
                "ref_audio_len": [ref_audio_tokens.shape[1]],
            }
        )
        return ft

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "ref_audio_tokens": MultiModalFieldConfig.batched("audio"),
            "ref_audio_len": MultiModalFieldConfig.batched("audio"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def insertion_end(item_idx):
            if "audio" in out_mm_kwargs and out_mm_kwargs["audio"]:
                ref_len = out_mm_kwargs["audio"][0]["ref_audio_len"].data[0].item()
                return [1] * ref_len
            return []

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=insertion_end,
            ),
        ]


class OmniVoiceDummyInputsBuilder(BaseDummyInputsBuilder[OmniVoiceMultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the OmniVoice system."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")
        max_prompt_seconds = 10
        prompt_sample_rate = 24000
        target_audio_length = max_prompt_seconds * prompt_sample_rate

        audio_overrides = mm_options.get("audio") if mm_options else None
        mm_data = {
            "audio": (
                self._get_dummy_audios(
                    length=target_audio_length,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )[0],
                24000,
            ),
        }
        return mm_data

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {"ref_text": "Testing voice cloning."}
        return inputs


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class OmniVoiceModel(
    nn.Module,
):
    """OmniVoice model for vLLM-Omni two-stage pipeline.

    Routes to generator (Stage 0) or decoder (Stage 1) based on model_stage.
    """

    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.model_stage = vllm_config.model_config.model_stage
        self.model_dir = vllm_config.model_config.model

        if self.model_stage == "omnivoice_generator":
            from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
                OmniVoiceGenerator,
            )

            self.generator = OmniVoiceGenerator(self.config)
            self.model = self.generator
        elif self.model_stage == "omnivoice_decoder":
            from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import (
                OmniVoiceDecoder,
            )

            self.decoder = OmniVoiceDecoder(self.config)
            self.model = self.decoder
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "omnivoice_generator":
            # Generator handles its own embedding in forward()
            hidden = int(self.config.llm_hidden_size)
            return torch.zeros((input_ids.shape[0], hidden), device=input_ids.device)
        elif self.model_stage == "omnivoice_decoder":
            hidden = int(self.config.llm_hidden_size)
            return torch.zeros((input_ids.shape[0], hidden), device=input_ids.device)
        else:
            raise RuntimeError(f"embed_input_ids not valid for {self.model_stage}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == "omnivoice_generator":
            return self._forward_generator(input_ids, kwargs)
        elif self.model_stage == "omnivoice_decoder":
            return self._forward_decoder(input_ids, kwargs)
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

    def _forward_generator(self, input_ids: torch.Tensor, kwargs: dict) -> OmniOutput:
        """Run generator stage: text → 8-codebook audio tokens."""
        runtime_info = kwargs.get("runtime_additional_information", [])

        if not runtime_info:
            # Profiling / dummy run — return a plain tensor (not OmniOutput)
            # so the v1 model runner's _dummy_run can index into it.
            return torch.zeros(
                (input_ids.shape[0], self.config.llm_hidden_size),
                device=input_ids.device,
                dtype=torch.float32,
            )

        info = runtime_info[0]
        device = input_ids.device
        num_codebooks = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        # Extract text tokens from input_ids
        text_tokens = input_ids  # [N_text]
        text_len = text_tokens.shape[0]

        # Estimate target length using RuleDurationEstimator
        # (same formula as reference OmniVoice: weight * 25 / 14.1)
        from vllm_omni.model_executor.models.omnivoice.duration import (
            RuleDurationEstimator,
        )

        if not hasattr(self, "_duration_estimator"):
            self._duration_estimator = RuleDurationEstimator()
        raw_text = info.get("raw_text", "")
        if raw_text:
            target_len = self._duration_estimator.estimate_duration(raw_text, "Nice to meet you.", 25)
            target_len = max(1, int(target_len))
        else:
            # Fallback: use character weight formula on text tokens
            # approximate ~1.77 frames per text token (25/14.1)
            target_len = max(int(text_len * 1.77), 25)

        # Get reference audio tokens if available
        ref_audio_tokens = info.get("ref_audio_tokens", None)

        # Build input_ids tensor: [2*B, 8, S]
        # B=1, conditional + unconditional

        # Replicate text tokens across 8 codebooks
        text_ids = text_tokens.unsqueeze(0).repeat(num_codebooks, 1)  # [8, N_text]

        # Target: all MASK
        target_ids = torch.full((num_codebooks, target_len), mask_id, dtype=torch.long, device=device)

        # Conditional: [text] [ref_audio?] [target_mask]
        if ref_audio_tokens is not None:
            ref_tokens = ref_audio_tokens.to(device)  # [8, T_ref]
            cond_ids = torch.cat([text_ids, ref_tokens, target_ids], dim=1)
            cond_audio_start = text_ids.shape[1]
        else:
            cond_ids = torch.cat([text_ids, target_ids], dim=1)
            cond_audio_start = text_ids.shape[1]

        cond_len = cond_ids.shape[1]

        # Unconditional: [target_mask only]
        uncond_ids = target_ids.clone()
        uncond_len = target_len

        # Pad to same length
        max_len = max(cond_len, uncond_len)
        if cond_len < max_len:
            pad = torch.full(
                (num_codebooks, max_len - cond_len),
                mask_id,
                dtype=torch.long,
                device=device,
            )
            cond_ids = torch.cat([cond_ids, pad], dim=1)
        if uncond_len < max_len:
            pad = torch.full(
                (num_codebooks, max_len - uncond_len),
                mask_id,
                dtype=torch.long,
                device=device,
            )
            uncond_ids = torch.cat([uncond_ids, pad], dim=1)

        batch_input_ids = torch.stack([cond_ids, uncond_ids], dim=0)  # [2, 8, max_len]

        # Audio mask: True for audio positions
        batch_audio_mask = torch.zeros((2, max_len), dtype=torch.bool, device=device)
        batch_audio_mask[0, cond_audio_start:cond_len] = True
        batch_audio_mask[1, :uncond_len] = True

        # Attention mask: [2, 1, S, S]
        batch_attention_mask = torch.zeros((2, 1, max_len, max_len), dtype=torch.bool, device=device)
        batch_attention_mask[0, :, :cond_len, :cond_len] = True
        batch_attention_mask[1, :, :uncond_len, :uncond_len] = True

        # Run iterative generation
        tokens = self.generator(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
            target_lens=[target_len],
            num_step=self.config.num_step,
            guidance_scale=self.config.guidance_scale,
            t_shift=self.config.t_shift,
            layer_penalty_factor=self.config.layer_penalty_factor,
            position_temperature=self.config.position_temperature,
            class_temperature=self.config.class_temperature,
        )  # [1, 8, target_len]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio_tokens": tokens},
        )

    def _forward_decoder(self, input_ids: torch.Tensor, kwargs: dict) -> OmniOutput:
        """Run decoder stage: 8-codebook tokens → audio waveform."""
        runtime_info = kwargs.get("runtime_additional_information", [])

        if not runtime_info:
            # Profiling / dummy run — return plain tensor for v1 runner compat
            return torch.zeros(
                (input_ids.shape[0], self.config.llm_hidden_size),
                device=input_ids.device,
                dtype=torch.float32,
            )

        info = runtime_info[0]
        audio_tokens = info.get("audio_tokens", None)

        if audio_tokens is None:
            raise RuntimeError("No audio_tokens received from generator stage")

        if isinstance(audio_tokens, np.ndarray):
            audio_tokens = torch.from_numpy(audio_tokens)

        # audio_tokens: [B, 8, T]; buffer may be CPU — move to decoder weights
        if audio_tokens.dim() == 2:
            audio_tokens = audio_tokens.unsqueeze(0)  # Add batch dim

        dec_device = next(self.decoder.parameters()).device
        audio_tokens = audio_tokens.to(device=dec_device, dtype=torch.long)

        tts_speech = self.decoder(audio_tokens)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": tts_speech,
                "sr": self.config.sample_rate,
            },
        )

    def _resolve_model_dir(self) -> str:
        """Resolve model directory to local path (handles HF hub IDs)."""
        model_dir = self.model_dir
        if os.path.isdir(model_dir):
            return model_dir
        # HF hub model ID — resolve to local cache
        from huggingface_hub import snapshot_download

        return snapshot_download(model_dir)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = self._resolve_model_dir()

        if self.model_stage == "omnivoice_generator":
            self.generator.load_weights(model_dir, device)
        elif self.model_stage == "omnivoice_decoder":
            self.decoder.load_weights(model_dir, device)
        else:
            raise ValueError(f"{self.model_stage} not supported!")
