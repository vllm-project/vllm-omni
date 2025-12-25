# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Fun-Audio-Chat unified model for vLLM-Omni.

Fun-Audio-Chat is a Large Audio Language Model built for natural, low-latency voice
interactions. It features Dual-Resolution Speech Representations with 5Hz backbone
and 25Hz refined head.

Model: https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B
Paper: https://arxiv.org/abs/2506.09349

Architecture:
- Main Stage: Audio understanding (continuous + discrete) + Text LLM → Text generation
- CosyVoice Stage (optional): Speech tokens → Audio waveform via CosyVoice3

Components:
- continuous_audio_tower: Whisper-like encoder for mel spectrograms
- audio_tower: Discrete speech token encoder (FunAudioChatDiscreteEncoder)
- language_model: Qwen3 8B for text understanding and generation
- audio_invert_tower: CRQ Transformer for speech token generation (disabled in S2T mode)
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

from .audio_encoder import FunAudioChatAudioEncoder, FunAudioChatDiscreteEncoder
from .processing_fun_audio_chat import FunAudioChatProcessor

logger = init_logger(__name__)


# Token IDs from config.json
AUDIO_TOKEN_INDEX = 151669  # <|audio_start|>
AUDIO_BOS_INDEX = 151670  # <|audio_bos|>
AUDIO_EOS_INDEX = 151671  # <|audio_eos|>


# ============================================================================
# Multimodal Processor Components
# ============================================================================


class FunAudioChatProcessingInfo(BaseProcessingInfo):
    """Processing info for Fun-Audio-Chat multimodal inputs."""

    _cached_processor: FunAudioChatProcessor | None = None

    def get_hf_processor(self, **kwargs) -> FunAudioChatProcessor:
        """
        Override to return our custom FunAudioChatProcessor.
        The default implementation uses AutoProcessor which doesn't work
        because the model doesn't have proper processor registration in HF Hub.
        """
        if self._cached_processor is not None:
            return self._cached_processor

        import os

        model_path = self.ctx.model_config.model

        # Check if speech_tokenizer folder exists
        speech_tokenizer_path = os.path.join(model_path, "speech_tokenizer")
        if not os.path.exists(speech_tokenizer_path):
            raise FileNotFoundError(
                f"speech_tokenizer folder not found at {speech_tokenizer_path}. "
                f"The Fun-Audio-Chat model requires the speech_tokenizer subfolder. "
                f"Please download the complete model from HuggingFace: "
                f"huggingface-cli download FunAudioLLM/Fun-Audio-Chat-8B --local-dir {model_path}"
            )

        self._cached_processor = FunAudioChatProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )
        return self._cached_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}  # Currently support single audio input

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # Fun-Audio-Chat uses 5Hz frame rate, so audio tokens are fewer
        # Estimate: 1 second of audio ≈ 5 tokens at 5Hz
        return {"audio": 1500}  # Max ~5 minutes of audio


class FunAudioChatDummyInputsBuilder(BaseDummyInputsBuilder[FunAudioChatProcessingInfo]):
    """Build dummy inputs for Fun-Audio-Chat profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Build the text input corresponding to mm_counts.

        For audio inputs, we need to include the <|AUDIO|> placeholder token
        that the HF processor will expand to audio tokens.
        """
        num_audio = mm_counts.get("audio", 0)
        if num_audio > 0:
            # Include audio placeholder for processor to expand
            hf_processor = self.info.get_hf_processor()
            audio_token = getattr(hf_processor, "audio_token", "<|AUDIO|>")
            return audio_token * num_audio
        return "Hello, how are you?"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        """Build dummy multimodal data for profiling."""
        num_audio = mm_counts.get("audio", 0)
        if num_audio > 0:
            # Create dummy audio: 16kHz sample rate, ~5 seconds
            # This will be processed by the audio encoder
            audio_length = 16000 * 5  # 5 seconds at 16kHz
            dummy_audios = self._get_dummy_audios(length=audio_length, num_audios=num_audio)
            return {"audio": dummy_audios}
        return {}


def _funaudiochat_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    """Get field config for Fun-Audio-Chat multimodal inputs.

    Fun-Audio-Chat uses a simple batched format where each tensor has
    a batch dimension that gets squeezed for single-item inputs.

    All tensors should be 2D (no batch dim) or 3D (with batch dim) and
    properly handled by batched() config.
    """
    config = {}

    # Packed continuous audio features
    # Use flat_from_sizes for packed 2D tensors with known sizes
    if "input_audio_features" in hf_inputs:
        audio_feature_lengths = hf_inputs.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            config["input_audio_features"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_feature_lengths, dim=1
            )
        else:
            # Fallback: treat as single item
            config["input_audio_features"] = MultiModalFieldConfig.batched("audio")

    if "audio_feature_lengths" in hf_inputs:
        config["audio_feature_lengths"] = MultiModalFieldConfig.batched("audio")
    if "feature_attention_mask" in hf_inputs:
        config["feature_attention_mask"] = MultiModalFieldConfig.batched("audio")

    # Legacy format (if not packed)
    if "input_features" in hf_inputs:
        config["input_features"] = MultiModalFieldConfig.batched("audio")
    if "feature_exist_mask" in hf_inputs:
        config["feature_exist_mask"] = MultiModalFieldConfig.batched("audio")

    # Discrete speech tokens
    if "speech_ids" in hf_inputs:
        config["speech_ids"] = MultiModalFieldConfig.batched("audio")
    if "speech_attention_mask" in hf_inputs:
        config["speech_attention_mask"] = MultiModalFieldConfig.batched("audio")

    return config


class FunAudioChatMultiModalProcessor(BaseMultiModalProcessor[FunAudioChatProcessingInfo]):
    """Multimodal processor for Fun-Audio-Chat audio inputs."""

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Given the HF-processed data, output the metadata of each field."""
        return _funaudiochat_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Given the original multi-modal items and HF-processed data,
        output the updates to perform on the prompt.

        Fun-Audio-Chat uses <|AUDIO|> as a placeholder that gets expanded
        to multiple audio tokens. We need to provide this info so vLLM
        can track placeholder positions.
        """
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Get audio tokens from processor
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_bos_token = getattr(processor, "audio_bos_token", "<|audio_bos|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|audio_eos|>")

        audio_token_id = vocab.get(audio_token, AUDIO_TOKEN_INDEX)
        audio_bos_id = vocab.get(audio_bos_token, AUDIO_BOS_INDEX)
        audio_eos_id = vocab.get(audio_eos_token, AUDIO_EOS_INDEX)

        # Get audio group size from processor (default 5)
        audio_group_size = getattr(processor, "audio_group_size", 5)

        # Get processed audio data to compute token counts
        out_mm_data = out_mm_kwargs.get_data()

        # Try to get audio lengths from different sources
        audio_output_lengths = []

        # 1. First try audio_feature_lengths (packed format)
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            if isinstance(audio_feature_lengths, torch.Tensor):
                # Compute output lengths through the encoder's length calculation
                # For Whisper-like encoder: output = ceil(input / 2) convolutions twice
                feature_lengths = audio_feature_lengths
                # Similar to _get_feat_extract_output_lengths
                feat_lengths = (feature_lengths - 1) // 2 + 1
                output_lengths = (feat_lengths - 2) // 2 + 1
                # Then group by audio_group_size
                audio_output_lengths = [
                    (int(length) + (audio_group_size - 1)) // audio_group_size for length in output_lengths.tolist()
                ]

        # 2. Fall back to speech_attention_mask if available
        if not audio_output_lengths:
            speech_attention_mask = out_mm_data.get("speech_attention_mask")
            if speech_attention_mask is not None:
                if isinstance(speech_attention_mask, torch.Tensor):
                    speech_lengths = speech_attention_mask.sum(-1)
                    audio_output_lengths = [
                        (int(length) + (audio_group_size - 1)) // audio_group_size for length in speech_lengths.tolist()
                    ]

        # 3. Fall back to feature_attention_mask
        if not audio_output_lengths:
            feature_attention_mask = out_mm_data.get("feature_attention_mask")
            if feature_attention_mask is not None:
                if isinstance(feature_attention_mask, torch.Tensor):
                    feature_lengths = feature_attention_mask.sum(-1)
                    # Compute encoder output lengths
                    feat_lengths = (feature_lengths - 1) // 2 + 1
                    output_lengths = (feat_lengths - 2) // 2 + 1
                    audio_output_lengths = [
                        (int(length) + (audio_group_size - 1)) // audio_group_size for length in output_lengths.tolist()
                    ]

        def get_replacement_funaudiochat(item_idx: int):
            if audio_output_lengths and item_idx < len(audio_output_lengths):
                num_audio_tokens = audio_output_lengths[item_idx]
            else:
                # Fallback: estimate based on ~5 seconds audio at 5Hz
                num_audio_tokens = 25  # ~5 seconds

            # Audio tokens surrounded by bos/eos
            audio_tokens = [audio_token_id] * num_audio_tokens

            return PromptUpdateDetails.select_token_id(
                [audio_bos_id] + audio_tokens + [audio_eos_id],
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_funaudiochat,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Call HuggingFace processor to process inputs.

        Following Qwen2_5_Omni pattern: pack audio features into a 2D tensor
        with audio_feature_lengths to track individual lengths.
        """
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Handle audios key (vLLM uses "audios" but processor expects "audios")
        mm_data = dict(mm_data)  # Make mutable copy
        audio_data = mm_data.pop("audios", None) or mm_data.pop("audio", None)

        if audio_data is not None:
            # Convert to list format if needed
            if not isinstance(audio_data, list):
                audio_data = [audio_data]

            # FunAudioChatProcessor expects audios as a list of audio arrays
            # Process audio using HF processor
            processed = processor(
                text=prompt,
                audios=audio_data,
                return_tensors="pt",
                padding=True,
            )

            # Pack audio features similar to Qwen2_5_Omni:
            # Convert [batch, num_mel_bins, seq_len] -> [num_mel_bins, total_frames]
            input_features = processed.get("input_features")
            feature_attention_mask = processed.get("feature_attention_mask")

            if input_features is not None and feature_attention_mask is not None:
                # Pack features by removing padding
                # input_features: [batch, num_mel_bins, seq_len]
                # feature_attention_mask: [batch, seq_len]
                packed_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
                # packed_features is now [num_mel_bins, total_valid_frames]

                processed["input_audio_features"] = packed_features
                processed["audio_feature_lengths"] = feature_attention_mask.sum(-1)
            elif input_features is not None:
                # No attention mask - treat as all valid
                # Flatten batch dimension
                batch_size, num_mel_bins, seq_len = input_features.shape
                packed_features = input_features.permute(0, 2, 1).reshape(-1, num_mel_bins).permute(1, 0)
                processed["input_audio_features"] = packed_features
                processed["audio_feature_lengths"] = torch.full((batch_size,), seq_len, dtype=torch.long)

            return processed
        else:
            # Text-only input
            processed = processor(text=prompt, return_tensors="pt")
            return processed


# ============================================================================
# Main Model Class
# ============================================================================


@MULTIMODAL_REGISTRY.register_processor(
    FunAudioChatMultiModalProcessor,
    info=FunAudioChatProcessingInfo,
    dummy_inputs=FunAudioChatDummyInputsBuilder,
)
class FunAudioChatForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    Fun-Audio-Chat model for vLLM-Omni.

    This implementation focuses on S2T (Speech-to-Text) mode.
    Speech generation (S2S) requires CosyVoice3 and will be implemented separately.

    Architecture:
    - Main Stage: Audio understanding + Text generation
    - CosyVoice Stage (future): Speech tokens → Audio waveform
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "main")

        logger.info(f"Initializing FunAudioChat in '{self.model_stage}' stage")

        if self.model_stage == "main":
            self._init_main_stage(vllm_config, prefix)
        elif self.model_stage == "cosyvoice":
            self._init_cosyvoice_stage(vllm_config, prefix)
        else:
            raise ValueError(f"Invalid model_stage: {self.model_stage}. Must be one of: 'main', 'cosyvoice'")

        # Initialize sampler
        self._sampler = None

    def _init_main_stage(self, vllm_config: VllmConfig, prefix: str):
        """Initialize the main audio-language stage."""
        config = self.config
        text_config = config.text_config
        audio_config = config.audio_config

        # Store configs
        self.text_config = text_config
        self.audio_config = audio_config

        # The language model is Qwen3-based
        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size

        # Audio token configuration
        self.audio_token_index = getattr(config, "audio_token_index", AUDIO_TOKEN_INDEX)
        self.group_size = getattr(audio_config, "group_size", 5)

        # Initialize continuous audio encoder (Whisper-like)
        logger.info("Initializing continuous_audio_tower (Whisper-like encoder)")
        self.continuous_audio_tower = FunAudioChatAudioEncoder(audio_config)

        # Initialize discrete audio encoder
        logger.info("Initializing audio_tower (discrete encoder)")
        self.audio_tower = FunAudioChatDiscreteEncoder(audio_config)

        # Initialize the language model (Qwen3)
        logger.info("Initializing language_model (Qwen3)")
        text_vllm_config = vllm_config.with_hf_config(text_config, architectures=["Qwen3ForCausalLM"])
        self.language_model = init_vllm_registered_model(
            vllm_config=text_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=text_config,
            architectures=["Qwen3ForCausalLM"],
        )

        # Audio invert tower is disabled for S2T mode
        # It would be needed for S2S (speech generation)
        self.audio_invert_tower = None
        self.enable_speech_generation = False

        # Flags
        self.have_multimodal_outputs = True

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
            if hasattr(self.language_model, "make_empty_intermediate_tensors")
            else lambda: None
        )

    def _init_cosyvoice_stage(self, vllm_config: VllmConfig, prefix: str):
        """Initialize CosyVoice3 for speech synthesis."""
        logger.warning("CosyVoice stage not yet implemented. For S2S, please use the separate CosyVoice3 model.")
        self.cosyvoice = None
        self.have_multimodal_outputs = True
        self.make_empty_intermediate_tensors = lambda: None
        self.hidden_size = 3584  # Default hidden size

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    @cached_property
    def sampler(self):
        """Get sampler for token generation."""
        if hasattr(self, "language_model") and self.language_model is not None:
            if hasattr(self.language_model, "sampler"):
                return self.language_model.sampler
        return Sampler()

    # ==================== Text Generation Interface ====================

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states for text generation.

        This method is required by vLLM's VllmModelForTextGeneration interface.

        Args:
            hidden_states: Hidden states from forward pass, can be raw tensor
                or OmniOutput wrapper.

        Returns:
            Logits tensor or None if TP rank > 0.
        """
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Delegate to language model's compute_logits
        if hasattr(self, "language_model") and self.language_model is not None:
            return self.language_model.compute_logits(hidden_states)

        # Fallback for cosyvoice stage or uninitialized language model
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        """Sample tokens from logits.

        Args:
            logits: Logits from compute_logits
            sampling_metadata: Sampling parameters

        Returns:
            Sampled tokens or SamplerOutput
        """
        if hasattr(self, "language_model") and self.language_model is not None:
            return self.language_model.sample(logits, sampling_metadata)
        return self.sampler(logits, sampling_metadata)

    # ==================== Embedding Methods ====================

    def get_input_embeddings(self):
        """Get input embeddings from language model."""
        if hasattr(self.language_model, "get_input_embeddings"):
            return self.language_model.get_input_embeddings()
        return self.language_model.model.embed_tokens

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | None = None,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input token IDs and merge with audio embeddings if present."""
        # Note: handle_oov_mm_token is accepted for vLLM compatibility but we
        # handle audio token merging ourselves below.

        # Debug: log input_ids info
        # audio_token_count = (input_ids == self.audio_token_index).sum().item()
        # logger.info(f"embed_input_ids: input_ids.shape={input_ids.shape}, "
        #            f"audio_token_index={self.audio_token_index}, count={audio_token_count}")

        if self.model_stage == "cosyvoice":
            return torch.zeros(
                input_ids.shape[0],
                self.hidden_size,
                device=input_ids.device,
                dtype=torch.bfloat16,
            )

        # Get base embeddings from language model
        embeddings = self.get_input_embeddings()(input_ids)

        # Merge with multimodal embeddings if provided
        if multimodal_embeddings is not None:
            # Flatten list of per-item embeddings if needed
            if isinstance(multimodal_embeddings, (list, tuple)):
                if len(multimodal_embeddings) == 0:
                    return embeddings
                multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)

            # Match audio tokens - use is_multimodal mask if provided (more reliable)
            if is_multimodal is not None:
                audio_mask = is_multimodal.bool()
                logger.info(f"embed_input_ids: using is_multimodal mask, sum={audio_mask.sum().item()}")
            else:
                audio_mask = input_ids == self.audio_token_index
                logger.info(
                    f"embed_input_ids: using audio_token_index={self.audio_token_index}, "
                    f"mask sum={audio_mask.sum().item()}"
                )

            if audio_mask.any() and multimodal_embeddings.numel() > 0:
                logger.info(
                    f"embed_input_ids: merging {multimodal_embeddings.shape[0]} audio "
                    f"embeddings into {audio_mask.sum().item()} positions"
                )
                logger.info(
                    f"embed_input_ids: multimodal_embeddings stats: "
                    f"min={multimodal_embeddings.min():.4f}, "
                    f"max={multimodal_embeddings.max():.4f}, mean={multimodal_embeddings.mean():.4f}"
                )
                logger.info(
                    f"embed_input_ids: text embeddings stats: min={embeddings.min():.4f}, "
                    f"max={embeddings.max():.4f}, mean={embeddings.mean():.4f}"
                )
                # Flatten for masked scatter
                flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                flat_mask = audio_mask.reshape(-1)

                n_audio_tokens = flat_mask.sum().item()
                n_audio_features = multimodal_embeddings.shape[0]

                if n_audio_tokens != n_audio_features:
                    logger.warning(
                        f"Audio token count ({n_audio_tokens}) != audio features ({n_audio_features}). Adjusting..."
                    )
                    # Truncate or pad as needed
                    if n_audio_features > n_audio_tokens:
                        multimodal_embeddings = multimodal_embeddings[:n_audio_tokens]
                    else:
                        # Pad with zeros
                        padding = torch.zeros(
                            n_audio_tokens - n_audio_features,
                            multimodal_embeddings.shape[-1],
                            device=multimodal_embeddings.device,
                            dtype=multimodal_embeddings.dtype,
                        )
                        multimodal_embeddings = torch.cat([multimodal_embeddings, padding], dim=0)

                flat_embeddings[flat_mask] = multimodal_embeddings.to(flat_embeddings.device, flat_embeddings.dtype)
                embeddings = flat_embeddings.reshape(embeddings.shape)
                logger.info(
                    f"embed_input_ids: merged embeddings stats: "
                    f"min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
                    f"mean={embeddings.mean():.4f}"
                )

        return embeddings

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        speech_maxlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode continuous audio features using the audio encoder.

        Args:
            input_features: Mel features [batch, num_mel_bins, seq_len]
            feature_attention_mask: Attention mask for features
            speech_maxlen: Maximum output length

        Returns:
            Tuple of (audio_features, audio_output_lengths)
        """
        device = input_features.device

        # Compute feature lengths from attention mask
        if feature_attention_mask is not None:
            audio_feature_lengths = feature_attention_mask.sum(dim=-1)
            # Pack features (remove padding)
            packed_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = torch.tensor(
                [input_features.shape[-1]] * input_features.shape[0],
                device=device,
            )
            packed_features = input_features.permute(0, 2, 1).reshape(-1, input_features.shape[1]).permute(1, 0)

        # Get output lengths
        audio_feat_lengths, audio_output_lengths = self.continuous_audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths
        )

        # Encode through continuous audio tower
        audio_features = self.continuous_audio_tower(
            input_features=packed_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
            speech_maxlen=speech_maxlen,
        )

        return audio_features, audio_output_lengths

    def embed_multimodal(self, **kwargs) -> list[torch.Tensor] | None:
        """
        Process and embed multimodal (audio) inputs.

        This method is called by vLLM framework to compute multimodal embeddings
        before they are merged with text embeddings.

        This combines continuous and discrete audio representations
        following the dual-resolution approach in Fun-Audio-Chat.

        Supports two input formats:
        1. Packed format (Qwen2_5_Omni style):
           - input_audio_features: [num_mel_bins, total_frames]
           - audio_feature_lengths: [num_audios]
        2. Legacy batched format:
           - input_features: [batch, num_mel_bins, seq_len]
           - feature_attention_mask: [batch, seq_len]

        Returns:
            List of audio embedding tensors, one per audio item.
        """
        # Extract parameters from kwargs
        input_features = kwargs.get("input_features")
        input_audio_features = kwargs.get("input_audio_features")
        audio_feature_lengths = kwargs.get("audio_feature_lengths")
        speech_ids = kwargs.get("speech_ids")
        speech_attention_mask = kwargs.get("speech_attention_mask")
        feature_attention_mask = kwargs.get("feature_attention_mask")
        feature_exist_mask = kwargs.get("feature_exist_mask")

        # Use packed format if available, otherwise fall back to legacy
        logger.info(
            f"embed_multimodal called: input_audio_features={input_audio_features is not None}, "
            f"input_features={input_features is not None}, speech_ids={speech_ids is not None}, "
            f"audio_feature_lengths={audio_feature_lengths is not None}"
        )
        logger.info(f"embed_multimodal: feature_exist_mask={feature_exist_mask}")
        if input_audio_features is not None:
            logger.info(f"  input_audio_features.shape={input_audio_features.shape}")
        if speech_ids is not None:
            logger.info(f"  speech_ids.shape={speech_ids.shape}")
        if input_audio_features is None and input_features is not None:
            # Legacy format - use old input_features
            packed_features = input_features
            use_packed_format = False
        elif input_audio_features is not None:
            packed_features = input_audio_features
            use_packed_format = True
        else:
            packed_features = None
            use_packed_format = False

        if packed_features is None and speech_ids is None:
            return None

        device = self._module_device(self.continuous_audio_tower)
        audio_features = None

        # Process speech IDs (discrete tokens)
        if speech_ids is not None:
            speech_ids = speech_ids.to(device)

            # Handle 3D input from vLLM batching: [batch, 1, seq] -> [batch, seq]
            if speech_ids.dim() == 3 and speech_ids.shape[1] == 1:
                speech_ids = speech_ids.squeeze(1)

            # Pad to multiple of group_size
            seq_len = speech_ids.shape[-1]
            target_len = ((seq_len + self.group_size - 1) // self.group_size) * self.group_size
            if target_len > seq_len:
                pad_len = target_len - seq_len
                pad_id = getattr(self.audio_config, "pad_token_id", 0)
                speech_ids = torch.nn.functional.pad(speech_ids, (0, pad_len), value=pad_id)

            # Get output lengths
            if speech_attention_mask is not None:
                speech_attention_mask = speech_attention_mask.to(device)
                # Handle 3D attention mask from vLLM batching
                if speech_attention_mask.dim() == 3 and speech_attention_mask.shape[1] == 1:
                    speech_attention_mask = speech_attention_mask.squeeze(1)
                speech_lengths = speech_attention_mask.sum(dim=-1)
            else:
                speech_lengths = torch.full((speech_ids.shape[0],), speech_ids.shape[-1], device=device)

            _, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(speech_lengths)

            # Process continuous features if available
            continuous_audio_features = None
            continuous_audio_output_lengths = None
            if packed_features is not None:
                packed_features = packed_features.to(device)
                # logger.info(
                #     f"embed_multimodal: processing continuous features, "
                #     f"packed_features.shape={packed_features.shape}"
                # )
                if use_packed_format and audio_feature_lengths is not None:
                    audio_feature_lengths = audio_feature_lengths.to(device)
                    # logger.info(
                    #     f"embed_multimodal: using packed format, "
                    #     f"audio_feature_lengths={audio_feature_lengths}"
                    # )
                    continuous_audio_features, continuous_audio_output_lengths = self._get_audio_features_packed(
                        input_audio_features=packed_features,
                        audio_feature_lengths=audio_feature_lengths,
                        speech_maxlen=speech_ids.shape[-1],
                    )
                    # logger.info(
                    #     f"embed_multimodal: continuous_audio_features.shape="
                    #     f"{continuous_audio_features.shape}"
                    # )
                    # logger.info(
                    #     f"embed_multimodal: continuous_audio_features stats: "
                    #     f"min={continuous_audio_features.min():.4f}, "
                    #     f"max={continuous_audio_features.max():.4f}, "
                    #     f"mean={continuous_audio_features.mean():.4f}"
                    # )
                elif feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(device)
                    (
                        continuous_audio_features,
                        continuous_audio_output_lengths,
                    ) = self.get_audio_features(
                        input_features=packed_features,
                        feature_attention_mask=feature_attention_mask,
                        speech_maxlen=speech_ids.shape[-1],
                    )
                    logger.info(f"embed_multimodal: continuous_audio_features.shape={continuous_audio_features.shape}")
                else:
                    logger.warning(
                        "embed_multimodal: packed_features provided but no "
                        "audio_feature_lengths or feature_attention_mask!"
                    )
            else:
                logger.warning("embed_multimodal: no packed_features (continuous audio) available!")

            # Debug: log continuous features before combining
            if continuous_audio_features is not None:
                logger.info(
                    f"embed_multimodal: continuous_audio_features.shape={continuous_audio_features.shape}, "
                    f"stats: min={continuous_audio_features.min():.4f}, max={continuous_audio_features.max():.4f}, "
                    f"mean={continuous_audio_features.mean():.4f}"
                )
                logger.info(f"embed_multimodal: continuous_audio_output_lengths={continuous_audio_output_lengths}")

            # Debug: log speech_ids values
            logger.info(f"embed_multimodal: speech_ids unique values: {speech_ids.unique()[:10].tolist()}...")
            logger.info(f"embed_multimodal: audio_output_lengths={audio_output_lengths}")

            # Encode discrete tokens and combine with continuous features
            audio_features = self.audio_tower(
                audio_ids=speech_ids,
                continuous_audio_features=continuous_audio_features,
                continuous_audio_output_lengths=continuous_audio_output_lengths,
                feature_exist_mask=feature_exist_mask,
            )

            # Debug: log output features
            logger.info(
                f"embed_multimodal: audio_features.shape={audio_features.shape}, "
                f"stats: min={audio_features.min():.4f}, max={audio_features.max():.4f}, "
                f"mean={audio_features.mean():.4f}"
            )

            # Create mask for valid audio tokens
            max_audio_tokens = audio_features.shape[1]
            audio_features_mask = torch.arange(max_audio_tokens, device=device)[None, :]
            audio_features_mask = audio_features_mask < audio_output_lengths[:, None]

            # Pack audio features (remove padding) and split into list
            audio_features = audio_features[audio_features_mask]
            audio_features = audio_features.split(audio_output_lengths.tolist())

        elif packed_features is not None:
            # Only continuous features (no speech IDs)
            packed_features = packed_features.to(device)

            if use_packed_format and audio_feature_lengths is not None:
                audio_feature_lengths = audio_feature_lengths.to(device)
                audio_features, audio_output_lengths = self._get_audio_features_packed(
                    input_audio_features=packed_features,
                    audio_feature_lengths=audio_feature_lengths,
                )
                # Split into list by output lengths
                audio_features = audio_features.split(audio_output_lengths.tolist())
            else:
                if feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(device)
                audio_features, audio_output_lengths = self.get_audio_features(
                    input_features=packed_features,
                    feature_attention_mask=feature_attention_mask,
                )
                # Pack features and split into list
                max_len = audio_features.shape[1]
                audio_features_mask = torch.arange(max_len, device=device)[None, :]
                audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                audio_features = audio_features[audio_features_mask]
                audio_features = audio_features.split(audio_output_lengths.tolist())

        if isinstance(audio_features, (list, tuple)) and len(audio_features) == 0:
            return None

        return audio_features

    def _get_audio_features_packed(
        self,
        input_audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        speech_maxlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode packed continuous audio features using the audio encoder.

        Args:
            input_audio_features: Packed mel features [num_mel_bins, total_frames]
                                  or batched [batch, num_mel_bins, frames]
            audio_feature_lengths: Lengths per audio [num_audios] or [num_audios, 1]
            speech_maxlen: Maximum output length

        Returns:
            Tuple of (audio_features, audio_output_lengths)
        """
        # Handle audio_feature_lengths shape: squeeze extra dimension
        if audio_feature_lengths.dim() == 2 and audio_feature_lengths.shape[1] == 1:
            audio_feature_lengths = audio_feature_lengths.squeeze(1)

        # Handle 3D batched input [batch, num_mel_bins, frames] -> convert to packed [num_mel_bins, total_frames]
        if input_audio_features.dim() == 3:
            batch_size = input_audio_features.shape[0]
            logger.info(
                f"_get_audio_features_packed: converting 3D input {input_audio_features.shape} to packed format"
            )
            # Pack by concatenating along frames dimension
            # First, extract valid frames for each sample
            packed_list = []
            for i in range(batch_size):
                length = (
                    audio_feature_lengths[i].item()
                    if audio_feature_lengths.numel() > 1
                    else audio_feature_lengths.item()
                )
                packed_list.append(input_audio_features[i, :, : int(length)])
            input_audio_features = torch.cat(packed_list, dim=1)  # [num_mel_bins, total_frames]
            logger.info(f"_get_audio_features_packed: packed shape {input_audio_features.shape}")

        # Get output lengths from the continuous audio tower
        audio_feat_lengths, audio_output_lengths = self.continuous_audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths
        )

        # Cast input to model dtype (audio features from processor may be float32)
        input_audio_features = input_audio_features.to(
            dtype=self.continuous_audio_tower.conv1.weight.dtype,
            device=self.continuous_audio_tower.conv1.weight.device,
        )

        # Encode through continuous audio tower
        audio_features = self.continuous_audio_tower(
            input_features=input_audio_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
            speech_maxlen=speech_maxlen,
        )

        return audio_features, audio_output_lengths

    # ==================== Forward Pass ====================

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Unified forward pass for Fun-Audio-Chat.

        Args:
            input_ids: Token IDs
            positions: Position IDs
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed embeddings
            sampling_metadata: Sampling parameters
            additional_information: Stage-specific data
            **kwargs: Additional arguments (input_features, speech_ids, etc.)

        Returns:
            OmniOutput with text_hidden_states
        """
        if self.model_stage == "main":
            return self._forward_main(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                sampling_metadata=sampling_metadata,
                **kwargs,
            )
        elif self.model_stage == "cosyvoice":
            return self._forward_cosyvoice(
                input_ids=input_ids,
                positions=positions,
                additional_information=additional_information,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model_stage: {self.model_stage}")

    def _forward_main(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata=None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass for the main audio-language stage (S2T)."""
        # Normalize dimensions
        _added_batch_dim = False
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            _added_batch_dim = True
        if positions is not None and positions.ndim == 1:
            positions = positions.unsqueeze(0)
            _added_batch_dim = True
        if inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
            _added_batch_dim = True

        device = self._module_device(self.language_model)

        # Move to device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if positions is not None:
            positions = positions.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)

        # Note: In vLLM v1, multimodal embeddings are computed by embed_multimodal()
        # and merged by embed_input_ids() before forward() is called.
        # The inputs_embeds parameter already contains the merged embeddings.

        # Debug: check if inputs_embeds is provided
        logger.info(
            f"_forward_main: inputs_embeds provided={inputs_embeds is not None}, "
            f"input_ids.shape={input_ids.shape if input_ids is not None else None}"
        )
        if inputs_embeds is not None:
            logger.info(
                f"_forward_main: inputs_embeds.shape={inputs_embeds.shape}, "
                f"stats: min={inputs_embeds.min():.4f}, max={inputs_embeds.max():.4f}"
            )

        # Get embeddings if not already provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids=input_ids)

        # Store text embeddings for CRQ decoder (S2S mode)
        text_embeds = inputs_embeds.clone()

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=None,  # Use embeddings instead
            positions=positions[0] if positions.ndim > 1 else positions,
            inputs_embeds=inputs_embeds.reshape(-1, inputs_embeds.shape[-1]),
            intermediate_tensors=intermediate_tensors,
        )

        # Check if we're in S2S mode (engine_output_type == "latent")
        engine_output_type = getattr(self.vllm_config.model_config, "engine_output_type", "text")

        # Build multimodal outputs
        multimodal_outputs: dict[str, Any] = {}

        if engine_output_type == "latent":
            # S2S mode: output hidden states for CRQ decoder
            multimodal_outputs["latent"] = hidden_states.reshape(-1, hidden_states.shape[-1])
            multimodal_outputs["hidden_states"] = hidden_states.reshape(-1, hidden_states.shape[-1])
            multimodal_outputs["text_embeds"] = text_embeds.reshape(-1, text_embeds.shape[-1])

        return OmniOutput(
            text_hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
            multimodal_outputs=multimodal_outputs,
        )

    def _forward_cosyvoice(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        additional_information: dict[str, Any] | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass for CosyVoice speech synthesis stage."""
        logger.warning("CosyVoice forward not yet implemented")
        return OmniOutput(
            text_hidden_states=torch.zeros(1, self.hidden_size),
            multimodal_outputs={"audio": None},
        )

    # ==================== Weight Loading ====================

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights with proper prefix handling."""
        loaded_weights: set[str] = set()

        # Categorize weights by module
        language_model_weights = []
        audio_tower_weights = []
        continuous_audio_tower_weights = []
        audio_invert_tower_weights = []
        other_weights = []

        for name, weight in weights:
            if name.startswith("language_model."):
                language_model_weights.append((name, weight))
            elif name.startswith("audio_tower."):
                audio_tower_weights.append((name, weight))
            elif name.startswith("continuous_audio_tower."):
                continuous_audio_tower_weights.append((name, weight))
            elif name.startswith("audio_invert_tower."):
                audio_invert_tower_weights.append((name, weight))
            else:
                other_weights.append((name, weight))

        # Load language model weights
        if hasattr(self, "language_model") and self.language_model is not None and language_model_weights:
            logger.info(f"Loading {len(language_model_weights)} language_model weights")
            lm_weights = [(name.replace("language_model.", ""), weight) for name, weight in language_model_weights]
            lm_loaded = self.language_model.load_weights(lm_weights)
            lm_loaded = add_prefix_to_loaded_weights(lm_loaded, "language_model")
            loaded_weights.update(lm_loaded)

        # Load continuous audio tower weights
        if (
            hasattr(self, "continuous_audio_tower")
            and self.continuous_audio_tower is not None
            and continuous_audio_tower_weights
        ):
            logger.info(f"Loading {len(continuous_audio_tower_weights)} continuous_audio_tower weights")
            state_dict = self.continuous_audio_tower.state_dict()
            for name, weight in continuous_audio_tower_weights:
                param_name = name.replace("continuous_audio_tower.", "")
                if param_name in state_dict:
                    if state_dict[param_name].shape == weight.shape:
                        state_dict[param_name].copy_(weight)
                        loaded_weights.add(name)
                    else:
                        logger.warning(
                            f"Shape mismatch for {name}: expected {state_dict[param_name].shape}, got {weight.shape}"
                        )
                else:
                    logger.debug(f"Skipping weight: {name}")

        # Load discrete audio tower weights
        if hasattr(self, "audio_tower") and self.audio_tower is not None and audio_tower_weights:
            logger.info(f"Loading {len(audio_tower_weights)} audio_tower weights")
            state_dict = self.audio_tower.state_dict()
            for name, weight in audio_tower_weights:
                param_name = name.replace("audio_tower.", "")
                if param_name in state_dict:
                    if state_dict[param_name].shape == weight.shape:
                        state_dict[param_name].copy_(weight)
                        loaded_weights.add(name)
                    else:
                        logger.warning(
                            f"Shape mismatch for {name}: expected {state_dict[param_name].shape}, got {weight.shape}"
                        )
                else:
                    logger.debug(f"Skipping weight: {name}")

        # Log skipped audio_invert_tower weights (S2T mode doesn't use them)
        if audio_invert_tower_weights:
            logger.info(
                f"Skipping {len(audio_invert_tower_weights)} audio_invert_tower weights (not needed for S2T mode)"
            )

        return loaded_weights


__all__ = ["FunAudioChatForConditionalGeneration"]
