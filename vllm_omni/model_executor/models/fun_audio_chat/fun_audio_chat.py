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

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

from .audio_encoder import FunAudioChatAudioEncoder, FunAudioChatDiscreteEncoder

logger = init_logger(__name__)


# Token IDs from config.json
AUDIO_TOKEN_INDEX = 151669  # <|audio_start|>
AUDIO_BOS_INDEX = 151670  # <|audio_bos|>
AUDIO_EOS_INDEX = 151671  # <|audio_eos|>


# ============================================================================
# Multimodal Processor Components
# ============================================================================


class FunAudioChatProcessingInfo:
    """Processing info for Fun-Audio-Chat multimodal inputs."""

    def __init__(self, ctx):
        self.ctx = ctx

    def get_supported_mm_limits(self):
        return {"audio": 1}  # Currently support single audio input

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: dict):
        # Fun-Audio-Chat uses 5Hz frame rate, so audio tokens are fewer
        # Estimate: 1 second of audio ≈ 5 tokens at 5Hz
        return {"audio": 1500}  # Max ~5 minutes of audio


class FunAudioChatMultiModalProcessor:
    """Multimodal processor for Fun-Audio-Chat audio inputs."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._processor = None

    def _get_processor(self):
        """Lazy-load the HuggingFace processor."""
        if self._processor is None:
            from transformers import AutoProcessor

            model_config = self.ctx.model_config
            self._processor = AutoProcessor.from_pretrained(
                model_config.model,
                trust_remote_code=model_config.trust_remote_code,
            )
        return self._processor

    def apply(
        self,
        prompt_text: str,
        mm_data: dict,
        hf_processor_mm_kwargs: dict,
    ) -> dict:
        """Process multimodal inputs for Fun-Audio-Chat."""
        processor = self._get_processor()

        # Extract audio from multimodal data
        audio_data = mm_data.get("audio")

        if audio_data is not None:
            # Process audio using HF processor
            processed = processor(
                text=prompt_text,
                audios=audio_data,
                return_tensors="pt",
                **hf_processor_mm_kwargs,
            )
            return {
                "prompt_token_ids": processed["input_ids"][0].tolist(),
                "mm_kwargs": {
                    "input_features": processed.get("input_features"),
                    "speech_ids": processed.get("speech_ids"),
                    "speech_attention_mask": processed.get("speech_attention_mask"),
                    "feature_attention_mask": processed.get("feature_attention_mask"),
                },
            }
        else:
            # Text-only input
            processed = processor(text=prompt_text, return_tensors="pt")
            return {
                "prompt_token_ids": processed["input_ids"][0].tolist(),
                "mm_kwargs": {},
            }


class FunAudioChatDummyInputsBuilder:
    """Build dummy inputs for Fun-Audio-Chat profiling."""

    def __init__(self, ctx):
        self.ctx = ctx

    def get_dummy_processor_inputs(self, seq_len: int, mm_counts: dict) -> dict:
        """Create dummy inputs for profiling."""
        # Create dummy audio features (mel spectrogram)
        num_audio = mm_counts.get("audio", 0)
        if num_audio > 0:
            # Dummy mel features: [batch, num_mel_bins, seq_len]
            dummy_features = torch.zeros(num_audio, 128, 3000)
            return {
                "audio": dummy_features,
                "prompt_text": "Transcribe the audio.",
            }
        return {
            "prompt_text": "Hello, how are you?",
        }


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
    ) -> torch.Tensor:
        """Embed input token IDs and merge with audio embeddings if present."""
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
            audio_mask = input_ids == self.audio_token_index
            if audio_mask.any() and multimodal_embeddings.numel() > 0:
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

    def embed_multimodal(
        self,
        input_features: torch.Tensor | None = None,
        speech_ids: torch.Tensor | None = None,
        speech_attention_mask: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        feature_exist_mask: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """
        Process and embed multimodal (audio) inputs.

        This combines continuous and discrete audio representations
        following the dual-resolution approach in Fun-Audio-Chat.
        """
        if input_features is None and speech_ids is None:
            return None

        device = self._module_device(self.continuous_audio_tower)
        audio_features = None

        # Process speech IDs (discrete tokens)
        if speech_ids is not None:
            speech_ids = speech_ids.to(device)

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
                speech_lengths = speech_attention_mask.sum(dim=-1)
            else:
                speech_lengths = torch.full((speech_ids.shape[0],), speech_ids.shape[-1], device=device)

            _, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(speech_lengths)

            # Process continuous features if available
            continuous_audio_features = None
            continuous_audio_output_lengths = None
            if input_features is not None:
                input_features = input_features.to(device)
                if feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(device)

                continuous_audio_features, continuous_audio_output_lengths = self.get_audio_features(
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    speech_maxlen=speech_ids.shape[-1],
                )

            # Encode discrete tokens and combine with continuous features
            audio_features = self.audio_tower(
                audio_ids=speech_ids,
                continuous_audio_features=continuous_audio_features,
                continuous_audio_output_lengths=continuous_audio_output_lengths,
                feature_exist_mask=feature_exist_mask,
            )

            # Create mask for valid audio tokens
            max_audio_tokens = audio_features.shape[1]
            audio_features_mask = torch.arange(max_audio_tokens, device=device)[None, :]
            audio_features_mask = audio_features_mask < audio_output_lengths[:, None]

            # Pack audio features (remove padding)
            audio_features = audio_features[audio_features_mask]

        elif input_features is not None:
            # Only continuous features (no speech IDs)
            input_features = input_features.to(device)
            if feature_attention_mask is not None:
                feature_attention_mask = feature_attention_mask.to(device)

            audio_features, audio_output_lengths = self.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )

            # Pack features
            max_len = audio_features.shape[1]
            audio_features_mask = torch.arange(max_len, device=device)[None, :]
            audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
            audio_features = audio_features[audio_features_mask]

        return audio_features

    # ==================== Forward Pass ====================

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata=None,
        additional_information: dict[str, Any] | None = None,
        **kwargs,
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

        # Process audio features if present
        input_features = kwargs.get("input_features")
        speech_ids = kwargs.get("speech_ids")

        audio_features = None
        if input_features is not None or speech_ids is not None:
            audio_features = self.embed_multimodal(
                input_features=input_features,
                speech_ids=speech_ids,
                speech_attention_mask=kwargs.get("speech_attention_mask"),
                feature_attention_mask=kwargs.get("feature_attention_mask"),
                feature_exist_mask=kwargs.get("feature_exist_mask"),
            )

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(
                input_ids=input_ids,
                multimodal_embeddings=audio_features,
            )

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=None,  # Use embeddings instead
            positions=positions[0] if positions.ndim > 1 else positions,
            inputs_embeds=inputs_embeds.reshape(-1, inputs_embeds.shape[-1]),
            intermediate_tensors=intermediate_tensors,
        )

        return OmniOutput(
            text_hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
            multimodal_outputs={},
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
