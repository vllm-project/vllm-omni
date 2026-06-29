# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe unified model (thinker + talker + code2wav)."""

import asyncio
from collections.abc import AsyncGenerator, Iterable
from functools import cached_property
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP, SupportsRealtime
from vllm.model_executor.models.qwen3_asr_realtime import Qwen3ASRRealtimeBuffer
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeConditionalGenerationMixin,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.data_entry_keys import Embeddings, HiddenStates, Ids, OmniPayload, OmniPayloadMeta
from vllm_omni.metrics import definitions as defs
from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights, safe_tensor_reshape
from vllm_omni.platforms import current_omni_platform

# Special token IDs for Qwen3 Omni MoE
# Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json

# Audio tokens (thinker vocabulary, for marking audio boundaries)
AUDIO_START_TOKEN_ID = 151669  # <|audio_start|> (audio_bos_token)
AUDIO_END_TOKEN_ID = 151670  # <|audio_end|> (audio_eos_token)
AUDIO_PAD_TOKEN_ID = 151675  # <|audio_pad|>

# TTS text tokens (thinker vocabulary, for text-to-speech control)
TTS_PAD_TOKEN_ID = 151671  # <tts_pad>
TTS_BOS_TOKEN_ID = 151672  # <tts_text_bos>
TTS_EOS_TOKEN_ID = 151673  # <tts_text_eod> (end of dialogue)
TTS_BOS_SINGLE_TOKEN_ID = 151674  # <tts_text_bos_single>

# Talker codec tokens (talker vocabulary, used for RVQ code generation)
TALKER_CODEC_PAD_TOKEN_ID = 4196  # Padding token
TALKER_CODEC_BOS_TOKEN_ID = 4197  # Beginning of speech
TALKER_CODEC_EOS_TOKEN_ID = 4198  # End of speech
TALKER_CODEC_NOTHINK_ID = 4203  # No-think mode
TALKER_CODEC_THINK_BOS_ID = 4204  # Think mode start
TALKER_CODEC_THINK_EOS_ID = 4205  # Think mode end

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3OmniMoeConditionalGenerationMixin,
    CustomProcessMixin,
    SupportsMRoPE,
    SupportsRealtime,
):
    """
    Unified Qwen3 Omni MoE model combining thinker, talker, and code2wav.

    Architecture:
    - Thinker: Multimodal understanding (text + audio + video) → text generation
    - Talker: Text embeddings → RVQ codec codes
    - Code2Wav: RVQ codes → audio waveform

    Usage:
        Set `model_stage` in vllm_config to one of: "thinker", "talker", "code2wav"
    """

    realtime_max_tokens = 64

    def _is_chunked_prefill_between_stage_enabled(self) -> bool:
        model_config = self.vllm_config.model_config
        return bool(getattr(model_config, "async_chunk", False)) and bool(
            getattr(model_config, "enable_chunked_prefill_between_stage", False)
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.use_async_omni_output = False
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # Keep vllm_config for later submodule init
        self.vllm_config = vllm_config
        self.config = config

        # Initialize thinker components
        thinker_config: Qwen3OmniMoeThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config
        self.multimodal_config = multimodal_config

        # Initialize talker components
        talker_config: Qwen3OmniMoeTalkerConfig = config.talker_config
        self.talker_config = talker_config

        # Initialize code2wav components
        code2wav_config: Qwen3OmniMoeCode2WavConfig = config.code2wav_config
        self.code2wav_config = code2wav_config

        # Determine model stage
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            self.use_async_omni_output = True
            # Initialize thinker model (multimodal processing + text generation)
            # Create a new vllm_config with thinker_config as the hf_config
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None
            self.tts_tokens = torch.tensor(
                [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                device=self._module_device(self.thinker),
                dtype=torch.long,
            )
        elif self.model_stage == "talker":
            multimodal_config.skip_mm_profiling = True
            self.has_preprocess = True
            self.has_postprocess = True
            # Talker only ships codec codes to code2wav; skip latent hidden D2H.
            # Build Omni output asynchronously like thinker, but run lightweight
            # postprocess eagerly so hidden_states.last stays on GPU before the
            # next decode step.
            self.use_async_omni_output = True
            self.eager_omni_postprocess_before_async_output = True
            self.omni_pooler_payload_include_hidden = False
            self.set_custom_preprocess(self.talker_preprocess)
            self.set_custom_postprocess(self.talker_postprocess)
            self.thinker = None
            # Initialize talker model (text embeddings → codec codes)
            # Create a new vllm_config with talker_config as the hf_config
            # This ensures the talker uses its own text_config (smaller vocab_size)
            talker_vllm_config = vllm_config.with_hf_config(
                talker_config, architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"]
            )
            self.talker = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"],
            )
            self.model = self.talker
            self.code2wav = None

            # for CI: Initialize special tokens embeddings early to avoid AttributeError when loading dummy weights
            self._init_special_tokens_embeddings()
            # suppress tokens by setting their probability to ~1e-9 (finite very small)
            self.suppressed_tokens = self._get_talker_suppressed_tokens()
            self.requires_raw_input_tokens = True
            # Keys that should stay on GPU in model_intermediate_buffer to avoid CPU↔GPU round-trips
            self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
                ("hidden_states", "last"),
                ("hidden_states", "trailing_text"),
                ("embed", "tts_pad_projected"),
                # talker MTP codec codes must stay on GPU to avoid a per-step D2H
                # sync stall; build_mm_cpu handles the eventual D2H at payload time.
                ("codes", "audio"),
            }

        elif self.model_stage == "code2wav":
            multimodal_config.skip_mm_profiling = True
            self.enable_update_additional_information = True
            self.thinker = None
            self.talker = None
            # Initialize code2wav (codec codes → audio waveform)
            # Create a new vllm_config with code2wav_config as the hf_config
            code2wav_vllm_config = vllm_config.with_hf_config(code2wav_config, architectures=["Qwen3OmniMoeCode2Wav"])
            self.code2wav = init_vllm_registered_model(
                vllm_config=code2wav_vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=code2wav_config,
                architectures=["Qwen3OmniMoeCode2Wav"],
            )
            self.model = self.code2wav
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be one of: 'thinker', 'talker', 'code2wav'"
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage == "thinker" else lambda: None
        )

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)

        # Use a small segment size for low-latency streaming.
        segment_duration_s = 5.0
        buffer = Qwen3ASRRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        audio_placeholder = Qwen3OmniMoeThinkerForConditionalGeneration.get_placeholder_str("audio", 0)
        prompt_template = f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n<|im_start|>assistant\n"

        prompt_token_ids = tokenizer.encode(prompt_template)

        # In non-async-chunk (full-payload) mode the engine treats each
        # streaming TokensPrompt as a fresh decode, so mid-stream segment
        # yields cause the thinker to emit duplicate responses and the
        # talker to emit duplicate audio. Defer all audio to the final
        # flush so the thinker sees one complete prompt.
        async_chunk = getattr(model_config, "async_chunk", False)

        async for audio_chunk in audio_stream:
            buffer.write_audio(audio_chunk)

            if async_chunk:
                while (segment := buffer.read_audio()) is not None:
                    yield TokensPrompt(
                        prompt_token_ids=prompt_token_ids,
                        multi_modal_data={"audio": segment},
                    )

        remaining = buffer.flush()
        if remaining is not None and len(remaining) > 0:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": remaining},
            )

    # ==================== Device utilities ====================

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    @cached_property
    def sampler(self):
        """Get sampler from active model."""
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def get_language_model(self) -> torch.nn.Module:
        """Delegate to the active stage's language model for upstream MoE resolution."""
        if hasattr(self.model, "get_language_model"):
            return self.model.get_language_model()
        return self.model

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        if self.model_stage == "talker":
            return self.model.embed_input_ids(input_ids)
        return self.model.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        """Delegate to active model for multimodal processing."""
        return self.model.embed_multimodal(**kwargs)

    # ==================== Forward Pass ====================
    def _get_talker_suppressed_tokens(self):
        """Return a boolean mask on GPU for suppressed token positions."""
        vocab_size = self.config.talker_config.text_config.vocab_size
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        start = vocab_size - 1024
        eos_id = self.config.talker_config.codec_eos_token_id
        for i in range(start, vocab_size):
            if i != eos_id:
                mask[i] = True
        # Will be moved to the correct device on first use
        return mask

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, int]:
        if self.model_stage == "thinker":
            if mm_features is None:
                msg = "Qwen3 Omni thinker get_mrope_input_positions requires mm_features"
                raise ValueError(msg)
            return self.thinker.get_mrope_input_positions(input_tokens, mm_features)
        # Talker/code2wav stages are text/codec-only and do not need
        # multimodal M-RoPE position computation. Return a cheap linear
        # position tensor to avoid unnecessary per-request M-RoPE work.
        seq_len = len(input_tokens)
        linear = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, seq_len)
        return linear, 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generate_audio: bool = True,
        voice_type: str = "ethan",
        codec: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Unified forward pass for all model stages.

        Workflow:
        1) Thinker: multimodal understanding → text hidden states
        2) Talker -> Code Predictor: text embeddings → codec codes (layer 0 + code_predictor:residual layers)
        3) Code2wav: 8-layer RVQ codes → audio waveform

        Returns:
            OmniOutput with text_hidden_states and optional audio
        """

        # ========== Stage 1: Thinker ==========
        if self.model_stage == "thinker":
            thinker_dev = self._module_device(self.thinker)

            # Move to thinker device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            # Run thinker forward
            # If talker expects a specific intermediate layer, capture it here
            accept_layer = getattr(self.talker_config, "accept_hidden_layer", None)
            capture_kwargs = {}
            if accept_layer is not None:
                capture_kwargs = {
                    "capture_layer_indices": [0, int(accept_layer)],
                    "return_hidden_states": True,
                }

            # Run thinker
            text_hidden_states, captured_layer_dict = self.thinker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **capture_kwargs,
                **kwargs,
            )
            return text_hidden_states, captured_layer_dict

        # ========== Stage 2.1: Talker ==========
        elif self.model_stage == "talker":
            if input_ids is None:
                # special case for profile run
                input_ids = torch.zeros(inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)

            # Ensure we have base embeddings when only ids are provided
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.talker.embed_input_ids(input_ids)

            # Run talker forward
            with torch.inference_mode():
                talker_hidden = self.talker.forward(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )
            return talker_hidden

        # ========== Stage 3: Code2Wav ==========
        elif self.model_stage == "code2wav":
            seq_token_counts: list[int] | None = kwargs.get("seq_token_counts")

            # Extract codec codes from input
            if input_ids.shape[0] % 16 == 0:
                if seq_token_counts is not None:
                    max_seq_len = max(seq_token_counts) // 16
                    batch_size = len(seq_token_counts)
                    split_codes = torch.split(input_ids, seq_token_counts, dim=0)
                    codes = torch.zeros((batch_size, 16, max_seq_len), device=input_ids.device, dtype=input_ids.dtype)
                    for idx, code in enumerate(split_codes):
                        seq_len = code.shape[0] // 16
                        codes[idx, :, :seq_len] = code.reshape(16, seq_len)
                else:
                    codes = input_ids.reshape(1, 16, -1)
            else:
                if seq_token_counts is None:
                    logger.debug(
                        "Code2Wav warmup input length %s is not divisible by 16; padding with zeros.",
                        input_ids.shape[0],
                    )
                else:
                    logger.warning_once(
                        "Code2Wav input length is not divisible by 16; padding with zeros. "
                        "This is expected only during cudagraph warmup."
                    )
                input_ids_flatten = input_ids.reshape(-1)
                input_ids_flatten = torch.cat(
                    [
                        input_ids_flatten,
                        torch.zeros(16 - input_ids.shape[0] % 16, dtype=torch.long, device=input_ids.device),
                    ]
                )
                codes = input_ids_flatten.reshape(1, 16, -1)

            # Generate audio from codec codes
            # Get every request's left_context_size from runtime_additional_information (passed via kwargs)
            left_context_size = []
            if runtime_additional_information is not None:
                for info in runtime_additional_information:
                    meta = info.get("meta", {})
                    if "left_context_size" in meta:
                        left_context_size.append(meta["left_context_size"])
            else:
                logger.debug("No additional_information provided to code2wav stage.")
            audio_tensors = self.generate_audio(codes, left_context_size, seq_token_counts)

            return audio_tensors

        # Fallback (shouldn't reach here)
        return OmniOutput(
            text_hidden_states=torch.zeros(
                [inputs_embeds.shape[0], self.talker.config.hidden_size],
                dtype=torch.bfloat16,
            ).to(self._module_device(self.model)),
            multimodal_outputs=None,
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs) -> OmniOutput:
        """
        Make an OmniOutput object from model outputs.
        Args:
            model_outputs: Model outputs
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if self.model_stage == "thinker":
            text_hidden_states, captured_layer_dict = model_outputs
            # Compute thinker-side TTS token embeddings for BOS/EOS/PAD and expose via multimodal outputs.
            # These will later be projected into talker text space by the talker stage.
            multimodal_outputs: OmniPayload = captured_layer_dict if captured_layer_dict is not None else {}
            try:
                thinker_tts_embeds = self.thinker.embed_input_ids(self.tts_tokens)  # [1,3,thinker_hidden]
                if (
                    isinstance(thinker_tts_embeds, torch.Tensor)
                    and thinker_tts_embeds.ndim == 3
                    and thinker_tts_embeds.shape[1] == 3
                ):
                    bos_eos_pad = thinker_tts_embeds.to(text_hidden_states.device).chunk(3, dim=1)  # 3 * [1,1,H]
                    embed = multimodal_outputs.setdefault("embed", {})
                    embed["tts_bos"] = [bos_eos_pad[0]]
                    embed["tts_eos"] = [bos_eos_pad[1]]
                    embed["tts_pad"] = [bos_eos_pad[2]]
            except Exception:
                # Best-effort; absence will be handled by talker with fallbacks
                pass

            # Return text-only output (with multimodal sidecar)
            return OmniOutput(
                text_hidden_states=(text_hidden_states.reshape(-1, text_hidden_states.shape[-1])),
                multimodal_outputs=multimodal_outputs,
            )
        elif self.model_stage == "talker":
            talker_hidden = model_outputs
            # Here is the only place to use model_intermediate_buffer. After MTP in the
            # preprocess function, the code_predictor_codes are stored in the info_dict list.
            # We need to merge the tensors from different requests into a single tensor.
            # In the future, we may allow user to custom an aggregated function.
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information")

            if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
                logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
            code_predictor_codes = [info.get("codes", {}).get("audio") for info in info_dicts]
            audio_codes = torch.cat(code_predictor_codes, dim=0)
            multimodal_outputs: OmniPayload = {"codes": {"audio": audio_codes}}
            span_len = audio_codes.shape[0]
            talker_hidden = talker_hidden[:span_len]
            return OmniOutput(text_hidden_states=talker_hidden, multimodal_outputs=multimodal_outputs)
        elif self.model_stage == "code2wav":
            audio_tensors = model_outputs
            sample_rate = defs.resolve_audio_sample_rate(self.code2wav_config)
            # `sr` is the audio sample rate metadata consumed by downstream
            # audio serving and stage-local audio metrics.
            sr_tensors = [torch.tensor(sample_rate, dtype=torch.int32) for _ in audio_tensors]
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [audio_tensor.reshape(1, -1) for audio_tensor in audio_tensors],
                    "sr": sr_tensors,
                },
            )

        return model_outputs

    # ==================== Audio Generation ====================

    def generate_audio(
        self,
        code: torch.Tensor,
        left_context_size: list[int] | None = None,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Generate audio waveform from codec codes.

        Args:
            code: [batch, num_quantizers, T] - RVQ codec codes
            left_context_size: Left context size for streaming decode
            seq_token_counts: Token count for each request in batch

        Returns:
            list of audio waveforms
        """
        code2wav_dev = self._module_device(self.code2wav)

        # Convert to tensor if needed
        if isinstance(code, torch.Tensor):
            talker_codes = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            talker_codes = torch.as_tensor(code, dtype=torch.long, device=code2wav_dev)

        # Ensure shape is [batch=1, 8, T]
        if talker_codes.ndim == 2:
            # [8, T] → [1, 8, T]
            talker_codes = talker_codes.unsqueeze(0)
        elif talker_codes.ndim == 1:
            # [T] → assume single layer, expand to 16 layers
            talker_codes = talker_codes.unsqueeze(0).unsqueeze(0)
            talker_codes = talker_codes.expand(1, 16, -1)

        if self.vllm_config.model_config.async_chunk:
            # Only use left_context_size from additional information
            audio_tensors = self.code2wav.chunked_decode_streaming(
                talker_codes,
                left_context_size=left_context_size,
                seq_token_counts=seq_token_counts,
            )
        else:
            # Use chunked decode for memory efficiency
            audio_tensors = self.code2wav.chunked_decode(
                talker_codes,
                chunk_size=300,
                left_context_size=25,
                seq_token_counts=seq_token_counts,
            )

        return audio_tensors

    # ==================== Thinker-Talker Projection ====================

    def _load_talker_embedding(self) -> torch.nn.Embedding:
        """Load talker embedding layer."""
        return self.talker.language_model.model.codec_embedding

    def _init_special_tokens_embeddings(self) -> set[str]:
        """
        Initialize special token embeddings for thinker-talker projection.

        Following Transformers implementation:
        - TTS tokens (BOS/EOS/PAD) come from thinker's embedding, projected to talker space
        - Codec tokens (BOS/EOS/PAD/NOTHINK/THINK_*) come from talker's embedding
        - Speaker tokens are also from talker's embedding

        Note on projections:
        - text_projection: Used here for text token embeddings (thinker → talker dimension)
        - hidden_projection: Used at runtime for multimodal hidden states (audio/image/video)
          from thinker's last layer, not needed for special token initialization
        """
        self.talker_embedding = self._load_talker_embedding()

        # Get configuration
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        codec_special_tokens = torch.tensor(
            [
                [
                    talker_hf_config.codec_nothink_id,
                    talker_hf_config.codec_think_bos_id,
                    talker_hf_config.codec_think_eos_id,
                    talker_hf_config.codec_pad_id,
                    talker_hf_config.codec_bos_id,
                    talker_hf_config.codec_eos_token_id,
                ]
            ],
            device=self._module_device(self.talker),
            dtype=torch.long,
        )
        codec_embeds = self.talker_embedding(codec_special_tokens)  # [1, 6, talker_hidden]
        (
            self.embed_codec_nothink_token,
            self.embed_codec_think_bos_token,
            self.embed_codec_think_eos_token,
            self.embed_codec_pad_token,
            self.embed_codec_bos_token,
            self.embed_codec_eos_token,
        ) = codec_embeds.chunk(6, dim=1)

        # Speaker token IDs (for voice selection)
        # In Qwen3, speaker_id mapping is in talker_config.speaker_id
        # Keys are lowercased for case-insensitive matching with serving layer.
        if hasattr(talker_hf_config, "speaker_id") and talker_hf_config.speaker_id:
            self.tts_text_spk_token_ids = {k.lower(): v for k, v in talker_hf_config.speaker_id.items()}
        else:
            # Default to audio_start_token_id if no speaker mapping
            self.tts_text_spk_token_ids = {
                "default": talker_hf_config.audio_start_token_id,
                "ethan": talker_hf_config.audio_start_token_id,
                "prefix_caching": talker_hf_config.audio_start_token_id,
            }

        self.default_tts_text_spk_type = list(self.tts_text_spk_token_ids.keys())[0]

        return set(["thinker_embedding.weight", "talker_embedding.weight"])

    def _get_text_spk_token_id(self, voice_type: str) -> int:
        """Get speaker token ID for voice type."""
        if voice_type not in self.tts_text_spk_token_ids:
            return self.tts_text_spk_token_ids[self.default_tts_text_spk_type]
        return self.tts_text_spk_token_ids[voice_type]

    def talker_postprocess(self, hidden_states: torch.Tensor, **info_dict: object):
        """
        Postprocess the talker hidden states.
        """
        return {"hidden_states": {"last": hidden_states[-1, :].detach()}}

    def talker_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict):
        """
        Preprocess talker embeds. Noted that we set the MTP here.
        """
        payload: OmniPayload = info_dict
        meta = payload.setdefault("meta", {})

        # Ensure we have base embeddings when only ids are provided
        if input_embeds is None and input_ids is not None:
            input_embeds = self.talker.embed_input_ids(input_ids)

        span_len = input_ids.shape[0]
        update_dict: OmniPayload = {}
        # Prefix caching can reduce a new request's remaining prefill span to a
        # single token. Use the runner-provided phase flag instead of span_len.
        is_prefill = bool(payload.get("_omni_is_prefill", span_len > 1))
        if is_prefill:
            num_computed_tokens = payload.get("_omni_num_computed_tokens")
            request_resumable = meta.get("resumable", False)
            if num_computed_tokens is not None and not request_resumable:
                meta["num_processed_tokens"] = int(num_computed_tokens)
            if self._is_chunked_prefill_between_stage_enabled():
                input_ids, input_embeds, update_dict = self.talker_preprocess_prefill_chunked_prefill(
                    input_ids, input_embeds, payload
                )
            else:
                input_ids, input_embeds, update_dict = self.talker_preprocess_prefill(input_ids, input_embeds, payload)
            code_predictor_codes = torch.zeros(
                (span_len, self.talker.num_code_groups),
                device=self._module_device(self.talker),
                dtype=torch.long,
            )
            update_dict.setdefault("codes", {})["audio"] = code_predictor_codes
        else:
            # decode
            if not meta.get("decode_flag", False):
                # Prefill already consumed the first text token via the
                # assistant bootstrap path, so decode starts from the
                # remaining-text boundary rather than cumulative index 0.
                prefill_consumed_text_tokens = meta.get("prefill_consumed_text_tokens")
                if prefill_consumed_text_tokens is None:
                    raise RuntimeError("Missing prefill_consumed_text_tokens for talker decode handoff.")
                meta["num_processed_tokens"] = prefill_consumed_text_tokens
                update_dict.setdefault("meta", {})["decode_flag"] = True

            last_talker_hidden, text_step, update_dict = self.talker_preprocess_decode(
                input_ids, input_embeds, update_dict, payload
            )
            update_dict["mtp_inputs"] = last_talker_hidden, text_step

        update_dict.setdefault("meta", {})["num_processed_tokens"] = meta.get("num_processed_tokens", 0) + span_len
        return input_ids, input_embeds, update_dict

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ):
        # TODO(Peiqi): not support intermediate_tensors now
        input_ids = safe_tensor_reshape(input_ids, (input_ids.shape[0], -1))
        inputs_embeds = safe_tensor_reshape(input_embeds, (-1, self.talker_config.text_config.hidden_size))
        text_step = safe_tensor_reshape(text_step, (-1, self.talker_config.text_config.hidden_size))
        last_talker_hidden = safe_tensor_reshape(
            last_talker_hidden, (-1, 1, self.talker_config.text_config.hidden_size)
        )
        # for profiling
        if inputs_embeds.shape[-1] == 2048:
            inputs_embeds = self.text_projection(inputs_embeds)
        code_predictor_codes, summed_embeddings = self.talker.code_predictor_forward(
            input_ids, inputs_embeds, last_talker_hidden=last_talker_hidden
        )
        # summed_embeddings is [B, seq_len, H] (3D) while text_step is [B, H] (2D).
        # Flatten to 2D first to avoid wrong broadcasting: [B,1,H]+[B,H] → [B,B,H]
        inputs_embeds = summed_embeddings.reshape(-1, self.talker_config.text_config.hidden_size)
        inputs_embeds = (inputs_embeds + text_step).reshape(-1, self.talker_config.text_config.hidden_size)
        return inputs_embeds, code_predictor_codes.squeeze(-1)

    def _get_tts_embed(self, thinker_embed, tts_bos_thinker, tts_eos_thinker, tts_pad_thinker):
        """Project thinker-side TTS embeddings into talker text space."""
        module_device = self._module_device(self.talker)

        def _ensure_1x1(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                return x[0, -1:, :]
            if x.ndim == 2:
                return x[-1]
            return x.view(1, 1, -1)

        def _proj_from_thinker(x_opt: torch.Tensor | None) -> torch.Tensor:
            if isinstance(x_opt, torch.Tensor) and x_opt.numel() > 0:
                xin = _ensure_1x1(x_opt).to(module_device)
            else:
                xin = torch.zeros(
                    (1, thinker_embed.shape[-1]),
                    device=module_device,
                    dtype=thinker_embed.dtype,
                )
            return self.talker.text_projection(xin).to(module_device)

        self.tts_bos_embed = _proj_from_thinker(tts_bos_thinker)
        self.tts_eos_embed = _proj_from_thinker(tts_eos_thinker)
        self.tts_pad_embed = _proj_from_thinker(tts_pad_thinker)
        return self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed

    def _prepare_talker_prefill_inputs(self, payload: OmniPayload):
        hs: HiddenStates = payload.get("hidden_states", {})
        embed: Embeddings = payload.get("embed", {})
        ids: Ids = payload.get("ids", {})

        voice_type = payload.get("speaker")
        if voice_type is not None and isinstance(voice_type, (list, tuple)) and len(voice_type) > 0:
            voice_type = voice_type[0]
        if not isinstance(voice_type, str) or not voice_type.strip():
            voice_type = self.default_tts_text_spk_type
        else:
            voice_type = str(voice_type).lower().strip()

        talker_device = self._module_device(self.talker)
        # Read thinker outputs for prefill
        thinker_sequence_embeds = embed["prefill"].to(device=talker_device, dtype=torch.bfloat16)  # Tensor [P,H]
        thinker_hidden_states = hs["output"].to(device=talker_device, dtype=torch.bfloat16)  # Tensor [K,H]
        thinker_sequences = (
            ids.get("all") if ids.get("all") is None else torch.as_tensor(ids["all"], device=talker_device)
        )
        thinker_chatml_ids = (
            ids.get("prompt") if ids.get("prompt") is None else torch.as_tensor(ids["prompt"], device=talker_device)
        )

        tts_bos_thinker = embed["tts_bos"].to(device=talker_device, dtype=torch.bfloat16)
        tts_eos_thinker = embed["tts_eos"].to(device=talker_device, dtype=torch.bfloat16)
        tts_pad_thinker = embed["tts_pad"].to(device=talker_device, dtype=torch.bfloat16)

        if thinker_sequence_embeds is None or thinker_hidden_states is None:
            raise ValueError(
                "additional_information_by_req_id must include "
                "'embed.prefill' and 'hidden_states.output' for talker prefill."
            )

        # Normalize to tensors
        if not isinstance(thinker_sequence_embeds, torch.Tensor):
            thinker_sequence_embeds = torch.as_tensor(thinker_sequence_embeds, device=talker_device)
        if not isinstance(thinker_hidden_states, torch.Tensor):
            thinker_hidden_states = torch.as_tensor(thinker_hidden_states, device=talker_device)

        if isinstance(thinker_chatml_ids, torch.Tensor) or isinstance(thinker_chatml_ids, list):
            ids_chatml = (
                thinker_chatml_ids
                if isinstance(thinker_chatml_ids, torch.Tensor)
                else torch.as_tensor(thinker_chatml_ids, device=talker_device)
            )
            if ids_chatml.ndim == 1:
                ids_chatml = ids_chatml.unsqueeze(0)
        else:
            # Fallback: create dummy ids if not provided
            ids_chatml = torch.zeros(
                (1, thinker_sequence_embeds.shape[1]),
                dtype=torch.long,
                device=talker_device,
            )
            thinker_sequences = ids_chatml

        speaker_id = self._get_text_spk_token_id(voice_type)

        return (
            talker_device,
            speaker_id,
            thinker_sequence_embeds,
            thinker_hidden_states,
            thinker_sequences,
            ids_chatml,
            tts_bos_thinker,
            tts_eos_thinker,
            tts_pad_thinker,
        )

    def talker_preprocess_prefill(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, payload: OmniPayload):
        embed: Embeddings = payload.get("embed", {})
        meta: OmniPayloadMeta = payload.get("meta", {})

        # Containers to return per-request updates (e.g., code_predictor_hidden_per_request)
        update_dict: OmniPayload = {}

        start_index = meta.get("num_processed_tokens", 0)
        end_index = start_index + input_embeds.shape[0]
        (
            talker_device,
            speaker_id,
            thinker_sequence_embeds,
            thinker_hidden_states,
            thinker_sequences,
            ids_chatml,
            tts_bos_thinker,
            tts_eos_thinker,
            tts_pad_thinker,
        ) = self._prepare_talker_prefill_inputs(payload)
        req_input_ids, req_embeds, trailing_text_hidden = self._thinker_to_talker_prefill(
            thinker_embed=thinker_sequence_embeds.to(talker_device),
            thinker_hidden=thinker_hidden_states.to(talker_device),
            multimodal_mask=None,
            input_ids=ids_chatml.to(talker_device),
            thinker_result_ids=thinker_sequences.to(talker_device),
            speaker_id=speaker_id,
            tts_bos_thinker=tts_bos_thinker,
            tts_eos_thinker=tts_eos_thinker,
            tts_pad_thinker=tts_pad_thinker,
        )

        # Queue trailing_text_hidden for decode (drop first for next steps),
        try:
            if isinstance(trailing_text_hidden, torch.Tensor) and trailing_text_hidden.numel() > 0:
                if trailing_text_hidden.ndim == 2:
                    rem_tail = trailing_text_hidden
                elif trailing_text_hidden.ndim == 1:
                    rem_tail = torch.zeros(
                        0,
                        trailing_text_hidden.shape[0],
                        dtype=trailing_text_hidden.dtype,
                        device=trailing_text_hidden.device,
                    )
                else:
                    # compatible with old shape [1,S,D]
                    rem_tail = trailing_text_hidden.squeeze(0)
                if rem_tail.shape[0] > 0:
                    update_dict.setdefault("hidden_states", {})["trailing_text"] = rem_tail.detach()
            # Also persist projected tts_pad for decode fallback if needed
            if isinstance(tts_pad_thinker, torch.Tensor):
                pad_in = tts_pad_thinker
                if pad_in.ndim == 2:
                    pad_in = pad_in.unsqueeze(0)
                if pad_in.ndim == 1:
                    pad_in = pad_in.view(1, 1, -1)
                pad_proj = self.talker.text_projection(pad_in.to(talker_device))
                update_dict.setdefault("embed", {})["tts_pad_projected"] = pad_proj.detach()
        except Exception:
            pass
        update_dict.setdefault("meta", {})["prefill_consumed_text_tokens"] = 1
        self._talker_cache_thinker_decode_embeds(embed, update_dict)

        return req_input_ids[start_index:end_index], req_embeds[start_index:end_index], update_dict

    def talker_preprocess_prefill_chunked_prefill(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, payload: OmniPayload
    ):
        update_dict: OmniPayload = {}
        (
            talker_device,
            speaker_id,
            thinker_sequence_embeds,
            thinker_hidden_states,
            thinker_sequences,
            ids_chatml,
            tts_bos_thinker,
            tts_eos_thinker,
            tts_pad_thinker,
        ) = self._prepare_talker_prefill_inputs(payload)

        num_processed_thinker_tokens = payload.get("meta", {}).get("num_processed_thinker_tokens", 0)
        if thinker_sequences is None:
            thinker_sequences = ids_chatml
        if self.vllm_config.model_config.async_chunk:
            chunk_size = input_ids.shape[0]
        else:
            chunk_size = thinker_sequence_embeds.shape[0]
        assert chunk_size > 0, f"Prefill chunk can not be 0. Received chunk_size={chunk_size}."
        chunk_offset = num_processed_thinker_tokens

        decode_assistant_fill = payload.get("embed", {}).get("decode")
        if isinstance(decode_assistant_fill, torch.Tensor):
            if decode_assistant_fill.numel() == 0:
                decode_assistant_fill = None
            elif decode_assistant_fill.ndim == 1:
                decode_assistant_fill = decode_assistant_fill.unsqueeze(0)
        else:
            decode_assistant_fill = None

        is_final_chunk = bool(payload.get("meta", {}).get("finished", False))
        if is_final_chunk:
            thinker_sequence_embed_chunk = thinker_sequence_embeds[chunk_offset:]
            thinker_hidden_chunk = thinker_hidden_states[chunk_offset:]
        else:
            thinker_sequence_embed_chunk = thinker_sequence_embeds[chunk_offset : chunk_offset + chunk_size]
            thinker_hidden_chunk = thinker_hidden_states[chunk_offset : chunk_offset + chunk_size]

        actual_embed_size = thinker_sequence_embed_chunk.shape[0]
        thinker_sequences_chunk = thinker_sequences[chunk_offset : chunk_offset + actual_embed_size]
        # Between-stage chunked prefill: the trailing assistant bootstrap is a
        # fixed 9-token block expanded from the 3 assistant ChatML rows. If a
        # scheduler chunk boundary lands inside it, the block must be emitted as
        # a contiguous slice rather than rebuilt from partial rows. Compute the
        # slice from authoritative talker positions and pass the full assistant
        # rows so the bootstrap is built once and split correctly. Guarded to the
        # single-turn 1:1 user mapping (thinker prompt == talker prompt minus the
        # +6 bootstrap expansion); otherwise fall back to the legacy path.
        assistant_full_embed = None
        bootstrap_emit_offset = 0
        bootstrap_emit_count = None
        _BOOTSTRAP_LEN = 9
        talker_prompt_len = int(payload.get("_omni_prompt_len") or 0)
        talker_num_computed = int(payload.get("_omni_num_computed_tokens") or 0)
        total_thinker_rows = int(thinker_sequence_embeds.shape[0])
        span = int(input_ids.shape[0])
        bootstrap_text_emitted = False
        if (
            self.vllm_config.model_config.async_chunk
            and talker_prompt_len >= _BOOTSTRAP_LEN
            and total_thinker_rows == talker_prompt_len - (_BOOTSTRAP_LEN - 3)
        ):
            bootstrap_region_start = talker_prompt_len - _BOOTSTRAP_LEN
            emit_start = max(talker_num_computed, bootstrap_region_start)
            emit_end = min(talker_num_computed + span, talker_prompt_len)
            emit_count = emit_end - emit_start
            if emit_count > 0 and bootstrap_region_start + 3 <= total_thinker_rows:
                assistant_full_embed = thinker_sequence_embeds[bootstrap_region_start : bootstrap_region_start + 3].to(
                    talker_device
                )
                bootstrap_emit_offset = emit_start - bootstrap_region_start
                bootstrap_emit_count = int(emit_count)
                bootstrap_text_emitted = (
                    bootstrap_emit_offset <= (_BOOTSTRAP_LEN - 1) < bootstrap_emit_offset + bootstrap_emit_count
                )

        req_input_ids, req_embeds, trailing_text_hidden = self._thinker_to_talker_prefill_chunked_prefill(
            thinker_embed=thinker_sequence_embed_chunk.to(talker_device),
            thinker_hidden=thinker_hidden_chunk.to(talker_device),
            input_ids=ids_chatml.to(talker_device),
            thinker_result_ids=thinker_sequences_chunk.to(talker_device),
            speaker_id=speaker_id,
            tts_bos_thinker=tts_bos_thinker,
            tts_eos_thinker=tts_eos_thinker,
            tts_pad_thinker=tts_pad_thinker,
            chunk_offset=chunk_offset,
            decode_assistant_fill=decode_assistant_fill
            if assistant_full_embed is None or bootstrap_text_emitted
            else None,
            assistant_full_embed=assistant_full_embed,
            bootstrap_emit_offset=bootstrap_emit_offset,
            bootstrap_emit_count=bootstrap_emit_count,
        )

        new_thinker_cursor = chunk_offset + len(thinker_sequences_chunk)
        if assistant_full_embed is not None:
            # While the fixed 9-token bootstrap is being emitted (it spans more
            # talker positions than the 3 assistant thinker rows), keep the
            # thinker cursor pinned at the user-region end so the next chunk's
            # slice still contains the assistant rows and re-enters the bootstrap
            # branch. Only advance past them once the whole bootstrap is emitted.
            user_thinker_len = total_thinker_rows - 3
            bootstrap_fully_emitted = (talker_num_computed + span) >= talker_prompt_len
            new_thinker_cursor = (
                total_thinker_rows if bootstrap_fully_emitted else min(new_thinker_cursor, user_thinker_len)
            )
        update_dict.setdefault("meta", {})["num_processed_thinker_tokens"] = new_thinker_cursor

        if isinstance(trailing_text_hidden, torch.Tensor) and trailing_text_hidden.ndim == 2:
            update_dict.setdefault("hidden_states", {})["trailing_text"] = trailing_text_hidden.detach()
        meta_out = update_dict.setdefault("meta", {})
        text_token_consumed = decode_assistant_fill is not None and (
            assistant_full_embed is None or bootstrap_text_emitted
        )
        meta_out["prefill_consumed_text_tokens"] = 1 if text_token_consumed else 0
        self._talker_cache_thinker_decode_embeds(payload.get("embed", {}), update_dict)
        update_dict.setdefault("embed", {})["decode"] = None

        return req_input_ids, req_embeds, update_dict

    def _talker_cache_thinker_decode_embeds(
        self,
        embed: Embeddings,
        update_dict: OmniPayload,
    ) -> None:
        """
        Cache thinker embeds for decode stage.
        """
        thinker_decode_embeds = embed.get("decode", None)
        if thinker_decode_embeds is not None:
            cached_thinker_decode_embeds = embed.get("cached_decode", None)
            if cached_thinker_decode_embeds is None:
                update_dict.setdefault("embed", {})["cached_decode"] = thinker_decode_embeds
            else:
                cached_thinker_decode_embeds = cached_thinker_decode_embeds.to(
                    device=self._module_device(self.talker), dtype=torch.bfloat16
                )
                thinker_decode_embeds = thinker_decode_embeds.to(
                    device=self._module_device(self.talker), dtype=torch.bfloat16
                )
                update_dict.setdefault("embed", {})["cached_decode"] = torch.cat(
                    [cached_thinker_decode_embeds, thinker_decode_embeds], dim=0
                )
        update_dict.setdefault("embed", {})["decode"] = None

    def _thinker_to_talker_prefill(
        self,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        thinker_result_ids: torch.Tensor,
        speaker_id,
        tts_bos_thinker: torch.Tensor | None = None,
        tts_eos_thinker: torch.Tensor | None = None,
        tts_pad_thinker: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Project thinker outputs to talker inputs during prefill stage.

        Returns:
            (input_ids, input_embeds) for talker
        """
        target_len = thinker_result_ids.shape[-1]
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([target_len], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result_ids == self.thinker_config.audio_token_id) |
            (thinker_result_ids == self.thinker_config.image_token_id) |
            (thinker_result_ids == self.thinker_config.video_token_id)
        ).to(input_ids.device)  # [t] # fmt: skip

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_embed(
            thinker_embed, tts_bos_thinker, tts_eos_thinker, tts_pad_thinker
        )

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        trailing_text_hidden_all: torch.Tensor | None = None
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if (role_token == self.config.system_token_id).item():
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif (role_token == self.config.user_token_id).item():
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result_ids[im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif (role_token == self.config.assistant_token_id).item() and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
                # capture trailing text hidden for decode steps
                try:
                    if isinstance(trailing_text_hidden, torch.Tensor):
                        trailing_text_hidden_all = trailing_text_hidden
                except Exception:
                    pass
            # History assistant output (ignore for now)
            elif (role_token == self.config.assistant_token_id).item() and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(input_ids.device) for embed in talker_input_embeds], dim=0)
        talker_input_id = torch.cat([embed.to(input_ids.device) for embed in talker_input_ids], dim=0)

        return talker_input_id, talker_input_embed, trailing_text_hidden_all

    def _thinker_to_talker_prefill_chunked_prefill(
        self,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        thinker_result_ids: torch.Tensor,
        speaker_id,
        tts_bos_thinker: torch.Tensor | None = None,
        tts_eos_thinker: torch.Tensor | None = None,
        tts_pad_thinker: torch.Tensor | None = None,
        chunk_offset: int = 0,
        decode_assistant_fill: torch.Tensor | None = None,
        assistant_full_embed: torch.Tensor | None = None,
        bootstrap_emit_offset: int = 0,
        bootstrap_emit_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Project thinker outputs to talker inputs during between-stage chunked prefill.

        Returns:
            (input_ids, input_embeds) for talker

        ``assistant_full_embed`` / ``bootstrap_emit_offset`` /
        ``bootstrap_emit_count`` support between-stage chunked prefill: the
        trailing assistant bootstrap is a fixed 9-token block built from the 3
        assistant ChatML rows. When a scheduler chunk boundary lands inside that
        block, this lets the caller build the bootstrap once from the full
        assistant rows and emit only the contiguous slice belonging to this
        chunk, instead of rebuilding it from partial rows (which duplicates the
        block and corrupts the prefill->decode handoff -> drifted/truncated
        audio).
        """
        num_embed_rows = int(thinker_embed.shape[0])
        thinker_result_ids = thinker_result_ids.reshape(-1)
        num_id_tokens = int(thinker_result_ids.shape[0])
        if num_embed_rows != num_id_tokens:
            # Full-payload handoff may omit the stop-token hidden row while ids
            # still include it; align before segment slicing and bootstrap.
            common_len = min(num_embed_rows, num_id_tokens)
            thinker_embed = thinker_embed[:common_len]
            thinker_hidden = thinker_hidden[:common_len]
            thinker_result_ids = thinker_result_ids[:common_len]

        target_len = chunk_offset + thinker_result_ids.shape[-1]
        input_id_view = input_ids.reshape(-1)
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).flatten(),
                torch.tensor([target_len], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )
        trailing_text_hidden = None
        multimodal_mask = (
            (thinker_result_ids == self.thinker_config.audio_token_id) |
            (thinker_result_ids == self.thinker_config.image_token_id) |
            (thinker_result_ids == self.thinker_config.video_token_id)
        ).to(input_ids.device)  # [t] # fmt: skip

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_embed(
            thinker_embed, tts_bos_thinker, tts_eos_thinker, tts_pad_thinker
        )

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []

        chunk_start_index = chunk_offset
        chunk_end_index = chunk_start_index + thinker_result_ids.shape[0]
        segment_start_tmp = chunk_start_index
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()

            if segment_end_index <= chunk_start_index or im_start_index >= chunk_end_index:
                continue

            execute_segments = []
            if im_start_index > segment_start_tmp:
                remain_segment_start = segment_start_tmp
                remain_segment_end = im_start_index
                if i > 0:
                    prev_segment_start = im_start_indexes[i - 1].item()
                    remain_segment_role_token = input_id_view[prev_segment_start + 1]
                else:
                    remain_segment_role_token = torch.tensor(
                        self.config.system_token_id,
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                execute_segments.append((remain_segment_start, remain_segment_end, remain_segment_role_token))

            segment_start = max(im_start_index, chunk_start_index)
            segment_end = min(segment_end_index, chunk_end_index)
            segment_start_tmp = segment_end
            role_token = input_id_view[im_start_index + 1]
            execute_segments.append((segment_start, segment_end, role_token))

            for segment_start, segment_end, segment_role_token in execute_segments:
                local_start = segment_start - chunk_start_index
                local_end = segment_end - chunk_start_index
                if (segment_role_token == self.config.system_token_id).item():
                    continue
                elif (segment_role_token == self.config.user_token_id).item():
                    talker_user_part = self._get_talker_user_parts(
                        local_start, local_end, multimodal_mask, thinker_hidden, thinker_embed
                    )
                    talker_input_embeds.append(talker_user_part)
                    talker_input_ids.append(thinker_result_ids[local_start:local_end])
                elif (segment_role_token == self.config.assistant_token_id).item() and i == len(im_start_indexes) - 2:
                    if assistant_full_embed is not None:
                        # Build the full bootstrap once from all assistant rows,
                        # then emit only this chunk's contiguous slice so a
                        # boundary inside the block does not rebuild it.
                        full_boot_embeds, full_boot_ids, full_boot_trailing = (
                            self._get_talker_assistant_parts_chunked_prefill(
                                0,
                                assistant_full_embed.shape[0],
                                speaker_id,
                                assistant_full_embed,
                                tts_pad_embed,
                                tts_bos_embed,
                                tts_eos_embed,
                                decode_assistant_fill=decode_assistant_fill,
                            )
                        )
                        emit_off = bootstrap_emit_offset
                        emit_cnt = (
                            bootstrap_emit_count
                            if bootstrap_emit_count is not None
                            else full_boot_embeds.shape[0] - emit_off
                        )
                        talker_assistant_embeds = full_boot_embeds[emit_off : emit_off + emit_cnt]
                        talker_assistant_ids = full_boot_ids[emit_off : emit_off + emit_cnt]
                        # trailing_text is consumed once; attach it on the first slice.
                        if emit_off == 0:
                            trailing_text_hidden = full_boot_trailing
                    else:
                        talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = (
                            self._get_talker_assistant_parts_chunked_prefill(
                                local_start,
                                local_end,
                                speaker_id,
                                thinker_embed,
                                tts_pad_embed,
                                tts_bos_embed,
                                tts_eos_embed,
                                decode_assistant_fill=decode_assistant_fill,
                            )
                        )
                    talker_input_embeds.append(talker_assistant_embeds)
                    talker_input_ids.append(talker_assistant_ids)
                elif (segment_role_token == self.config.assistant_token_id).item() and i != len(im_start_indexes) - 2:
                    continue
                else:
                    raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")

        if len(talker_input_embeds) == 0 or len(talker_input_ids) == 0:
            embed_dim = self.talker_config.text_config.hidden_size
            talker_input_embed = torch.empty(0, embed_dim, dtype=thinker_embed.dtype, device=input_ids.device)
            talker_input_id = torch.empty(0, dtype=thinker_result_ids.dtype, device=input_ids.device)
            return talker_input_id, talker_input_embed, trailing_text_hidden

        talker_input_embed = torch.cat([embed.to(input_ids.device) for embed in talker_input_embeds], dim=0)
        talker_input_id = torch.cat([embed.to(input_ids.device) for embed in talker_input_ids], dim=0)

        return talker_input_id, talker_input_embed, trailing_text_hidden

    def _thinker_decode_to_talker_decode(
        self,
        payload: OmniPayload,
        device: torch.device,
        update_dict,
    ):
        """
        Project thinker outputs to talker inputs during decode stage.
        Returns:
            text_step embedding for talker
        """
        embed = payload.get("embed", {})
        meta = payload.get("meta", {})

        cached_thinker_decode_embeds = embed.get("cached_decode", None)
        thinker_decode_embed = embed.get("decode", None)
        start_index = meta.get("num_processed_tokens", 0)

        if cached_thinker_decode_embeds is not None and start_index < cached_thinker_decode_embeds.shape[0]:
            cached_thinker_decode_embeds = cached_thinker_decode_embeds.to(device)
            thinker_embed = cached_thinker_decode_embeds[start_index]
            if thinker_decode_embed is not None:
                thinker_decode_embed = thinker_decode_embed.to(device)
                cached_thinker_decode_embeds = torch.cat([cached_thinker_decode_embeds, thinker_decode_embed], dim=0)
                update_dict.setdefault("embed", {})["cached_decode"] = cached_thinker_decode_embeds

        elif thinker_decode_embed is not None:
            thinker_embed = thinker_decode_embed
            if thinker_embed.device != device:
                thinker_embed = thinker_embed.to(device)

        else:
            if meta.get("eos_emitted", False):
                return self.tts_pad_embed.to(device)
            update_dict.setdefault("meta", {})["eos_emitted"] = True
            return self.tts_eos_embed.to(device)

        update_dict.setdefault("embed", {})["decode"] = None
        return self.talker.text_projection(thinker_embed).to(device)

    def talker_preprocess_decode(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, update_dict: OmniPayload, payload: OmniPayload
    ):
        hs = payload.get("hidden_states", {})

        last_talker_hidden = None
        text_step = None
        try:
            if self.vllm_config.model_config.async_chunk:
                text_step = self._thinker_decode_to_talker_decode(payload, input_ids.device, update_dict)
            else:
                q_tail = hs.get("trailing_text", None)
                if isinstance(q_tail, torch.Tensor) and q_tail.numel() > 0:
                    use_vec = q_tail[0:1, :]
                    new_q_tail = (
                        q_tail[1:, :].detach()
                        if q_tail.shape[0] > 1
                        else self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)
                    )
                    text_step = use_vec.to(input_embeds.device, dtype=input_embeds.dtype)
                    update_dict.setdefault("hidden_states", {})["trailing_text"] = new_q_tail
                else:
                    text_step = self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)

            last_talker_hidden_tensor = hs.get("last")
            if last_talker_hidden_tensor is not None:
                last_talker_hidden = last_talker_hidden_tensor.to(input_embeds.device, dtype=input_embeds.dtype)
                last_talker_hidden = last_talker_hidden.reshape(*last_talker_hidden.shape[-2:])  # [1, hidden_size]
            else:
                last_talker_hidden = torch.zeros(
                    (1, self.talker_config.text_config.hidden_size),
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )
        except Exception as e:
            logger.error(f"Error in decode: {e}")

        return last_talker_hidden, text_step, update_dict

    def _get_talker_user_parts(self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed):
        clamped = min(
            segment_end_index,
            multimodal_mask.shape[0],
            thinker_hidden.shape[0],
            thinker_embed.shape[0],
        )
        if clamped < segment_end_index:
            logger.warning(
                "_get_talker_user_parts: segment_end_index %d clamped to %d "
                "(embed=%d, hidden=%d, mask=%d). "
                "This usually means _merge_pd_embeddings failed to merge "
                "prefill embeddings – check PD prefill_mm keys.",
                segment_end_index,
                clamped,
                thinker_embed.shape[0],
                thinker_hidden.shape[0],
                multimodal_mask.shape[0],
            )
        segment_end_index = clamped
        seg_len = segment_end_index - im_start_index
        if seg_len <= 0:
            return torch.empty(
                (0, self.config.talker_config.text_config.hidden_size),
                device=thinker_hidden.device,
                dtype=torch.bfloat16,
            )

        user_talker_part = torch.empty(
            (seg_len, self.config.talker_config.text_config.hidden_size),
            device=thinker_hidden.device,
            dtype=torch.bfloat16,
        )

        user_mm_mask = multimodal_mask[im_start_index:segment_end_index]
        # Multimodal data exists
        if user_mm_mask.any():
            user_thinker_hidden_mm = thinker_hidden[im_start_index:segment_end_index][user_mm_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm).to(thinker_hidden.device)
            user_talker_part[user_mm_mask] = mm_hidden
        user_thinker_embed = thinker_embed[im_start_index:segment_end_index][~user_mm_mask]
        user_text_hidden = self.talker.text_projection(user_thinker_embed).to(thinker_hidden.device)
        user_talker_part[~user_mm_mask] = user_text_hidden
        return user_talker_part

    def _get_talker_assistant_parts(
        self, im_start_index, segment_end_index, speaker_id, thinker_embed, tts_pad_embed, tts_bos_embed, tts_eos_embed
    ):
        assistant_hidden = self.talker.text_projection(thinker_embed[im_start_index:segment_end_index]).to(
            tts_pad_embed.device
        )  # [t, d]
        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:3],
                tts_pad_embed.expand(4, -1),
                tts_bos_embed,
                assistant_hidden[3:4]
                if assistant_hidden.shape[0] > 3
                else torch.zeros(
                    (1, assistant_hidden.shape[1]),
                    device=assistant_hidden.device,
                    dtype=assistant_hidden.dtype,
                ),  # First text
            ),
            dim=0,
        )
        codec_special_tokens = torch.tensor(
            [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                speaker_id,
                self.config.talker_config.codec_pad_id,
                self.config.talker_config.codec_bos_id,
            ],
            device=tts_pad_embed.device,
            dtype=torch.long,
        )
        embed_input_ids = self.talker.embed_input_ids(codec_special_tokens).to(
            device=tts_pad_embed.device, dtype=torch.bfloat16
        )
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (3, self.config.talker_config.text_config.hidden_size),
                    device=tts_pad_embed.device,
                    dtype=torch.bfloat16,
                ),
                embed_input_ids,
            ),
            dim=0,
        )

        if assistant_hidden.shape[0] > 4:
            trailing_text_hidden = torch.cat(
                (assistant_hidden[4:], tts_eos_embed),
                dim=0,
            )
        else:
            trailing_text_hidden = tts_eos_embed

        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.full(
            (assistant_text_hidden.shape[0],),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )
        return input_embeds, input_ids, trailing_text_hidden

    def _get_talker_assistant_parts_chunked_prefill(
        self,
        im_start_index,
        segment_end_index,
        speaker_id,
        thinker_embed,
        tts_pad_embed,
        tts_bos_embed,
        tts_eos_embed,
        decode_assistant_fill: torch.Tensor | None = None,
    ):
        assistant_hidden = self.talker.text_projection(thinker_embed[im_start_index:segment_end_index]).to(
            tts_pad_embed.device
        )  # [t, d]
        hidden_dim = assistant_hidden.shape[-1] if assistant_hidden.shape[0] > 0 else tts_pad_embed.shape[-1]

        # [3 tokens] + [4 pad] + [1 BOS] + [1 first text] = 9 tokens
        has_first_text = isinstance(decode_assistant_fill, torch.Tensor) and decode_assistant_fill.shape[0] >= 1
        if has_first_text:
            first_text_embed_all = self.talker.text_projection(decode_assistant_fill.to(tts_pad_embed.device)).to(
                assistant_hidden.dtype
            )
            first_text_embed = first_text_embed_all[0:1]  # pos 8 = tok0 only
            extra_text_embeds = first_text_embed_all[1:]  # tok1..tok(N-1) → trailing_text
        elif assistant_hidden.shape[0] > 3:
            # Full-payload (non-async-chunk) path: assistant_hidden already contains
            # projected thinker response embeddings. Use them directly so pos-8 gets
            # the real tok0 embed and trailing_text is built from tok1..tok(N-1).
            first_text_embed = assistant_hidden[3:4]  # pos 8 = tok0 (already projected)
            extra_text_embeds = assistant_hidden[4:]  # tok1..tok(N-1) → trailing_text
        else:
            first_text_embed = torch.zeros((1, hidden_dim), device=tts_pad_embed.device, dtype=assistant_hidden.dtype)
            extra_text_embeds = torch.empty((0, hidden_dim), device=tts_pad_embed.device, dtype=assistant_hidden.dtype)
        n_proj = assistant_hidden.shape[0]
        if n_proj < 3:
            pad_rows = torch.zeros((3 - n_proj, hidden_dim), device=tts_pad_embed.device, dtype=assistant_hidden.dtype)
            assistant_proj3 = torch.cat((assistant_hidden[:n_proj], pad_rows), dim=0)
        else:
            assistant_proj3 = assistant_hidden[:3]
        assistant_text_hidden = torch.cat(
            (
                assistant_proj3,  # pos 0-2: thinker prefill projections (padded if partial)
                tts_pad_embed.expand(4, -1),  # pos 3-6
                tts_bos_embed,  # pos 7
                first_text_embed,  # pos 8 = tok0
            ),
            dim=0,
        )
        if extra_text_embeds.shape[0] > 0:
            trailing_text_hidden = torch.cat((extra_text_embeds, tts_eos_embed), dim=0)
        else:
            trailing_text_hidden = tts_eos_embed
        codec_special_tokens = torch.tensor(
            [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                speaker_id,
                self.config.talker_config.codec_pad_id,
                self.config.talker_config.codec_bos_id,
            ],
            device=tts_pad_embed.device,
            dtype=torch.long,
        )
        embed_input_ids = self.talker.embed_input_ids(codec_special_tokens).to(
            device=tts_pad_embed.device, dtype=torch.bfloat16
        )
        # Full codec channel: [zeros×3, codec_embeds×6] = 9 tokens.
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (3, self.config.talker_config.text_config.hidden_size),
                    device=tts_pad_embed.device,
                    dtype=torch.bfloat16,
                ),
                embed_input_ids,
            ),
            dim=0,
        )

        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.zeros(
            (assistant_text_hidden.shape[0],),
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )
        return input_embeds, input_ids, trailing_text_hidden

    def _talker_to_code_predictor(
        self,
        talker_hidden_states: torch.Tensor | None,
        layer0_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project talker outputs to code predictor inputs.

        Returns:
            (input_ids, input_embeds) for code predictor.
        """
        predictor = getattr(self, "code_predictor", None)
        device = (
            self._module_device(predictor)
            if predictor is not None
            else (
                talker_hidden_states.device
                if isinstance(talker_hidden_states, torch.Tensor)
                else current_omni_platform.get_torch_device()
            )
        )

        if not isinstance(talker_hidden_states, torch.Tensor):
            raise ValueError("Talker hidden states must be provided for the code predictor stage.")

        inputs_embeds = talker_hidden_states.to(device=device, dtype=torch.bfloat16)
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if not isinstance(layer0_token_ids, torch.Tensor):
            raise ValueError("Layer-0 codec token ids must accompany talker hidden states.")
        input_ids = layer0_token_ids.to(device=device, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        return input_ids, inputs_embeds

    # ==================== Logits and Sampling ====================

    def _warn_talker_sampling_temperature(self, sampling_metadata: SamplingMetadata):
        warning_parts = []
        if sampling_metadata.temperature is None:
            warning_parts.append(
                "Temperature is set to None, as all requests are greedy. "
                "This is equivalent to setting temperature to 0.0."
                "Please consider setting a higher temperature i.e. 0.4."
            )
        else:
            warning_parts.append(
                "Temperature is set to: "
                f"{sampling_metadata.temperature}, where temperature as 0.0 may "
                "cause repetitive output. Please consider setting a higher "
                "temperature i.e. 0.4."
            )
        warning_parts.append(
            "This warning will be shown only once, for the first request where "
            "temperature is 0.0. Later requests will not show this warning but "
            "still be affected by the temperature."
        )
        warning_info = "\n".join(warning_parts)
        logger.warning_once(warning_info)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata = None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if (
            getattr(self, "model_stage", None) == "talker"
            and sampling_metadata is not None
            and (sampling_metadata.temperature is None)
        ):
            self._warn_talker_sampling_temperature(sampling_metadata)

        # Use active model for logits computation
        logits = self.model.compute_logits(hidden_states)  # V, d
        # Talker: suppress tokens by setting their probability to ~1e-9 (finite very small),
        # implemented by assigning their logits to log(1e-9).

        if getattr(self, "model_stage", None) == "talker" and isinstance(logits, torch.Tensor):
            # Move mask to device once (lazy), then reuse every step
            if self.suppressed_tokens.device != logits.device:
                self.suppressed_tokens = self.suppressed_tokens.to(logits.device)
            logits.masked_fill_(self.suppressed_tokens.unsqueeze(0), -1e9)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        """Sample from logits."""
        return self.model.sample(logits, sampling_metadata)

    # ==================== Weight Loading ====================

    def _get_codec_frame_config(self) -> tuple[int, int]:
        """Extract codec_chunk_frames and codec_left_context_frames from stage connector config."""
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra = connector_cfg.get("extra", {})
        else:
            extra = getattr(connector_cfg, "extra", None) or {}
        chunk_frames = int(extra.get("codec_chunk_frames", 0) or 0)
        left_frames = int(extra.get("codec_left_context_frames", 0) or 0)
        return chunk_frames, left_frames

    def _maybe_enable_code2wav_cudagraph(self) -> None:
        """Enable the inner Code2Wav CUDA graph unless this stage runs in eager mode."""
        if not self.code2wav or not hasattr(self.code2wav, "enable_cudagraph"):
            return

        model_cfg = getattr(self.vllm_config, "model_config", None)
        if getattr(model_cfg, "enforce_eager", False):
            logger.info("Code2Wav CUDA Graph disabled because enforce_eager is set")
            return

        chunk_frames, left_frames = self._get_codec_frame_config()
        self.code2wav.enable_cudagraph(
            codec_chunk_frames=chunk_frames,
            codec_left_context_frames=left_frames,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        code2wav_weights = []

        # Separate weights by component
        for k, v in weights:
            if k.startswith("thinker."):
                thinker_weights.append((k, v))
            elif k.startswith("talker."):
                talker_weights.append((k, v))
            elif k.startswith("code2wav."):
                code2wav_weights.append((k, v))
            else:
                logger.warning(f"Unknown weight prefix: {k}")
        # Load thinker weights
        if self.thinker and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if self.talker and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)
            loaded_weights.update(self._init_special_tokens_embeddings())

        # Load code2wav weights
        if self.code2wav and code2wav_weights:
            code2wav_loaded = self.code2wav.load_weights(code2wav_weights)
            code2wav_loaded = add_prefix_to_loaded_weights(code2wav_loaded, "code2wav")
            loaded_weights.update(code2wav_loaded)

            # Precompute SnakeBeta caches; Code2Wav CUDA graph follows the stage's
            # enforce_eager setting, the same switch vLLM uses for outer graphs.
            try:
                self.code2wav.precompute_snake_caches()
                self._maybe_enable_code2wav_cudagraph()
            except Exception:
                logger.warning(
                    "Failed to enable CUDA Graph for Code2Wav; falling back to eager.",
                    exc_info=True,
                )

        # Log summary
        logger.info(
            "Loaded %d weights for Qwen3OmniMoe (stage=%s)",
            len(loaded_weights),
            self.model_stage,
        )

        return loaded_weights
