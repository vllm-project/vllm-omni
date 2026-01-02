from typing import Any, List, Optional, Tuple, Union, Iterable, Set

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
    extract_layer_inputs,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.logger import init_logger
from .mamba_mia import MambaMiaCompressor

logger = init_logger(__name__)

class HyperCLOVAXSeedOmniThinkerMultiModalProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        return args[0]

@MULTIMODAL_REGISTRY.register_processor(
    HyperCLOVAXSeedOmniThinkerMultiModalProcessor,
    info=None, 
    dummy_inputs=None,
)
class HyperCLOVAXSeedOmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        
        # 1. Initialize LLM backbone (Llama-based 8B)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=self.config,
            architectures=["LlamaForCausalLM"], 
        )

        # 2. Initialize Encoders using HF AutoModel
        # Vision Encoder
        if hasattr(self.config, "vision_config"):
            try:
                self.vision_encoder = AutoModel.from_config(self.config.vision_config, trust_remote_code=True)
                logger.info("Successfully loaded Vision Encoder via AutoModel (remote code).")
            except Exception as e:
                logger.error(f"Failed to load Vision Encoder: {e}")
                raise RuntimeError("Could not load Vision Encoder.") from e
        else:
            self.vision_encoder = None

        # Audio Encoder
        if hasattr(self.config, "audio_config"):
            try:
                self.audio_encoder = AutoModel.from_config(self.config.audio_config, trust_remote_code=True)
                logger.info("Successfully loaded Audio Encoder via AutoModel (remote code).")
            except Exception as e:
                logger.error(f"Failed to load Audio Encoder: {e}")
                raise RuntimeError("Could not load Audio Encoder.") from e
        else:
            self.audio_encoder = None

        # 3. MambaMia Compression (Video)
        if getattr(self.config, "use_mamba_mia", False):
            logger.info("Initializing MambaMia Video Compressor")
            self.mamba_mia = MambaMiaCompressor(self.config)
        else:
            self.mamba_mia = None

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model.model.embed_tokens(input_ids)

    def _process_vision_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Forward pass through Vision Encoder
        if self.vision_encoder is None:
            raise ValueError("Vision inputs provided but Vision Encoder is not initialized.")
            
        vision_outputs = self.vision_encoder(pixel_values)
        
        # Handle different output formats (hidden_states or direct tensor)
        if hasattr(vision_outputs, "last_hidden_state"):
            image_features = vision_outputs.last_hidden_state
        else:
            image_features = vision_outputs
            
        # Apply MambaMia compression if enabled (typically for video)
        # Assuming pixel_values shape hints at video (Batch, Num_Frames, C, H, W) vs Image
        if self.mamba_mia is not None:
            # Check if input is video-like or if we should always apply it
            # For now, apply if MambaMia is initialized
            image_features = self.mamba_mia(image_features)
            
        return image_features

    def _process_audio_input(self, audio_values: torch.Tensor) -> torch.Tensor:
        if self.audio_encoder is None:
            raise ValueError("Audio inputs provided but Audio Encoder is not initialized.")
            
        audio_outputs = self.audio_encoder(audio_values)
        if hasattr(audio_outputs, "last_hidden_state"):
            return audio_outputs.last_hidden_state
        return audio_outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, torch.Tensor]:
        
        # 1. Process Multimodal Inputs if present in kwargs
        pixel_values = kwargs.pop("pixel_values", None)
        audio_values = kwargs.pop("audio_values", None)
        
        vision_embeddings = None
        audio_embeddings = None
        
        if pixel_values is not None:
            vision_embeddings = self._process_vision_input(pixel_values)
            
        if audio_values is not None:
            audio_embeddings = self._process_audio_input(audio_values)
            
        # 2. Embed Text Inputs if inputs_embeds not provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
            
        # 3. Merge Embeddings
        # If we have multimodal embeddings, we need to merge them into inputs_embeds
        # vLLM provides `merge_multimodal_embeddings` utility, but specific logic depends on model
        # Here we assume a simple replacement or concatenation strategy supported by vLLM
        # In a real scenario, this would use `merge_multimodal_embeddings` with the model's placeholder tokens.
        
        if vision_embeddings is not None or audio_embeddings is not None:
            # Create a dictionary of embeddings to merge
            mm_embeddings_dict = {}
            if vision_embeddings is not None:
                mm_embeddings_dict["image"] = vision_embeddings
            if audio_embeddings is not None:
                mm_embeddings_dict["audio"] = audio_embeddings
                
            # Use vLLM's utility to merge (this requires input_ids to have placeholder tokens)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, mm_embeddings_dict, self.config
            )

        # 4. Forward through LLM
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
