import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Iterable, Set, Tuple

from transformers import AutoModel, AutoConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix, add_prefix_to_loaded_weights
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.hyperclovax_seed_omni.hyperclovax_seed_omni_thinker import (
    HyperCLOVAXSeedOmniThinkerForConditionalGeneration,
    HyperCLOVAXSeedOmniThinkerMultiModalProcessor
)

logger = init_logger(__name__)

class HyperCLOVAXSeedOmniAudioDecoder(nn.Module):
    """
    Wrapper for CosyVoice2 Audio Decoder.
    Loads the actual model implementation from the Hugging Face repository using trust_remote_code.
    """
    def __init__(self, config, model_path: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # Load the actual CosyVoice2 model from the HF repository
        # This executes the remote code (cosyvoice.py) provided by the model authors.
        try:
            path_to_load = model_path if model_path else str(getattr(config, "_name_or_path", ""))
            logger.info(f"Loading CosyVoice2 Audio Decoder from {path_to_load}...")
            
            # Use AutoModel with trust_remote_code=True to load the custom model class
            self.model = AutoModel.from_pretrained(
                path_to_load, 
                config=config, 
                trust_remote_code=True
            )
            logger.info("Successfully loaded CosyVoice2 Audio Decoder.")
            
        except Exception as e:
            logger.error(f"Failed to load CosyVoice2 model via AutoModel: {e}")
            logger.warning("Falling back to placeholder for compilation checks (Runtime will fail if not fixed).")
            self.model = None

    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_tokens: (Batch, Seq_Len) tensor of discrete audio tokens
        Returns:
            waveform: (Batch, Audio_Len) tensor of generated audio waveform
        """
        if self.model is None:
            # Runtime fail-safe if model loading failed
            raise RuntimeError("CosyVoice2 model failed to load. Cannot generate audio.")
            
        # Call the actual model's inference/decode method.
        # The method name depends on the specific implementation in cosyvoice.py.
        # Common patterns: model.decode(), model.inference(), or forward()
        
        # Assuming standard forward or a specific decode method exposed by the remote code
        # We try to detect the method or default to forward
        if hasattr(self.model, "decode"):
            return self.model.decode(audio_tokens)
        elif hasattr(self.model, "inference"):
            return self.model.inference(audio_tokens)
        else:
            # Default forward pass
            return self.model(audio_tokens)

    def load_weights(self, weights):
        # AutoModel handles weight loading during init, so we might not need manual loading here
        # unless vLLM requires it. For external models loaded via AutoModel, we return empty set
        # to tell vLLM "we handled it".
        return set()

@MULTIMODAL_REGISTRY.register_processor(
    HyperCLOVAXSeedOmniThinkerMultiModalProcessor,
    info=None,
    dummy_inputs=None,
)
class HyperCLOVAXSeedOmniForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.prefix = prefix
        
        self.model_stage = vllm_config.model_config.model_stage
        self.model_path = vllm_config.model_config.model # Original model path
        
        self.thinker: Optional[HyperCLOVAXSeedOmniThinkerForConditionalGeneration] = None
        self.code2wav: Optional[HyperCLOVAXSeedOmniAudioDecoder] = None
        self.model: Optional[nn.Module] = None

        if self.model_stage == "thinker":
            # Initialize thinker model (LLM + Encoders)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=self.config,
                architectures=["HyperCLOVAXSeedOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            
        elif self.model_stage == "code2wav":
            # Initialize audio decoder (CosyVoice2 wrapper)
            # We pass the full config and the model path to allow loading the remote code
            
            # Check if there's a specific audio decoder config section
            audio_config = getattr(self.config, "audio_decoder_config", self.config)
            
            self.code2wav = HyperCLOVAXSeedOmniAudioDecoder(audio_config, model_path=self.model_path)
            self.model = self.code2wav
            
        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}. Supported: 'thinker', 'code2wav'")

        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, OmniOutput]:
        
        if self.model_stage == "thinker" and self.thinker is not None:
            hidden_states = self.thinker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs=None
            )

        elif self.model_stage == "code2wav" and self.code2wav is not None:
            # Input to Code2Wav stage are the Audio Tokens generated by Thinker
            audio_tokens = input_ids
            waveform = self.code2wav(audio_tokens)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": waveform}
            )
            
        raise RuntimeError("Model stage not initialized correctly")

    def compute_logits(self, hidden_states: Union[torch.Tensor, OmniOutput], sampling_metadata: Optional[Any] = None) -> Optional[torch.Tensor]:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "thinker" and self.thinker is not None:
            return self.thinker.compute_logits(hidden_states, sampling_metadata)
        return None

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        if self.model_stage == "thinker" and self.thinker is not None:
            return self.thinker.language_model.sample(logits, sampling_metadata)
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loaded_weights = set()
        thinker_weights = []
        
        # If we are in code2wav stage, AutoModel loaded the weights already.
        if self.model_stage == "code2wav":
            # Consume generator to prevent warnings about unconsumed weights if applicable
            for _ in weights: pass
            return set() 

        # Filter weights for Thinker stage
        for k, v in weights:
            if k.startswith("thinker.") or k.startswith("language_model.") or k.startswith("vision_encoder.") or k.startswith("audio_encoder."):
                thinker_weights.append((k, v))
                
        if self.thinker:
            loaded = self.thinker.load_weights(thinker_weights)
            loaded_weights.update(add_prefix_to_loaded_weights(loaded, "thinker"))
            
        return loaded_weights
