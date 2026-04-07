# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections.abc import Iterable
from functools import cached_property
from typing import Optional, Union

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.minicpmo.minicpmo_omni_thinker import MiniCPMOConfig
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Import processor components from thinker module
from vllm_omni.model_executor.models.minicpmo.minicpmo_omni_thinker import (
    MiniCPMOOmniThinkerDummyInputsBuilder,
    MiniCPMOOmniThinkerMultiModalProcessor,
    MiniCPMOOmniThinkerProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOOmniThinkerMultiModalProcessor,
    info=MiniCPMOOmniThinkerProcessingInfo,
    dummy_inputs=MiniCPMOOmniThinkerDummyInputsBuilder,
)
class MiniCPMOOmniForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE
):
    """MiniCPM-o 2.6 Omni model for conditional generation.
    
    This model supports multi-stage processing:
    - thinker: Image preprocessing + Vision encoder + 3D resampler
    - talker: LLM generation
    - code2wav: Speech output
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "(<image>./</image>)"
        if modality.startswith("video"):
            return "(<video>./</video>)"
        if modality.startswith("audio"):
            return "(<audio>./</audio>)"
        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config
        
        # Store configs
        self.config = config
        self.multimodal_config = multimodal_config
        
        self.model_stage = vllm_config.model_config.model_stage
        
        if self.model_stage == "thinker":
            # Initialize thinker model (image preprocessing + vision encoder + 3D resampler)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMOOmniThinkerModel"],
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None
            
        elif self.model_stage == "talker":
            self.thinker = None
            # Initialize talker model (LLM generation)
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMOOmniTalkerModel"],
            )
            # Initialize multimodal components if needed
            if hasattr(self.talker, "init_multi_modal"):
                self.talker.init_multi_modal(config)
            self.model = self.talker
            self.code2wav = None
            
        elif self.model_stage == "code2wav":
            self.thinker = None
            self.talker = None
            # Code2wav only runs Vocos (mel → waveform);
            # use tts_config if available, otherwise use the main config.
            self.code2wav_config = getattr(config, "tts_config", None) or config
            self.code2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=self.code2wav_config,
                architectures=["MiniCPMOOmniCode2WavModel"],
            )
            self.model = self.code2wav
        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}. Must be one of: 'thinker', 'talker', 'code2wav'")
        
        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors) if self.model_stage == "thinker" and self.thinker is not None else lambda: None
        )
    
    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler
        return Sampler()

    # -------------------- Device utilities --------------------
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")
    
    def move_submodules_to_devices(
        self,
        *,
        thinker_device: Optional[Union[str, torch.device]] = None,
        talker_device: Optional[Union[str, torch.device]] = None,
        code2wav_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Optionally move thinker/talker/code2wav to different devices.
        
        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                code2wav_device='cpu',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
        if code2wav_device is not None and self.code2wav is not None:
            self.code2wav.to(code2wav_device)
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            tts_cfg = getattr(self.config, "tts_config", None)
            hs = getattr(tts_cfg, "hidden_size", 768) if tts_cfg else 768
            return torch.zeros(
                input_ids.shape[0], hs,
                device=input_ids.device, dtype=torch.bfloat16,
            )
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage in ("talker", "code2wav"):
            return self.get_input_embeddings(input_ids)
        return super().embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )
    
    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to the active stage submodule when it implements MM encoding.
        mm_fn = getattr(self.model, "get_multimodal_embeddings", None)
        if mm_fn is not None:
            return mm_fn(**kwargs)
        return []

    def embed_multimodal(self, **kwargs: object):
        """vLLM V1 encoder profiling calls this; the inherited Protocol stub returns None."""
        return self.get_multimodal_embeddings(**kwargs)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        sampler=None,
        additional_information: Optional[dict[str, object]] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, OmniOutput]:
        """
        Forward pass for MiniCPM-o Omni model.
        
        Workflow:
        1) Thinker: Image preprocessing + Vision encoder + 3D resampler → hidden states
        2) Talker: LLM generation from hidden states → text tokens
        3) Code2Wav: Text tokens → speech waveform
        """
        if self.model_stage == "thinker":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
            # TODO: Remove this hack when NPU supports batched inputs properly
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            thinker_dev = self._module_device(self.thinker)
            
            # if input_ids is None, set it to a zero tensor
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev).unsqueeze(0)
                added_batch_dim = True
            
            # Ensure inputs on thinker's device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)
            
            if current_omni_platform.is_npu():
                # TODO: remove this hack when NPU supports batched inputs properly
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions.ndim > 1 else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )
            else:
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions is not None and added_batch_dim else positions
                thinker_inputs_embeds = inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
            
            # Run thinker
            thinker_output = self.thinker(
                input_ids=thinker_input_ids,
                positions=thinker_positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=thinker_inputs_embeds,
                **kwargs,
            )
            
            if isinstance(thinker_output, tuple):
                embeds, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output
            
            # Prepare hidden states for downstream stages
            # Ensure correct shape: (batch_size, seq_len, hidden_dim)
            if added_batch_dim:
                text_hidden_states = text_hidden_states.squeeze(0)
            
            # Return hidden states with latent in multimodal_outputs for stage_input_processors
            return OmniOutput(
                text_hidden_states=text_hidden_states,
                multimodal_outputs={"latent": text_hidden_states},
            )
        
        # Talker stage: runs ConditionalChatTTS + DVAE → mel_spec (+ optional Vocos → waveform)
        if self.model_stage == "talker":
            if input_ids is not None:
                num_tokens = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                num_tokens = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                num_tokens = 1
                device = torch.device("cuda")
            hidden_dim = self.config.hidden_size if hasattr(self.config, "hidden_size") else 2560

            # Profile/dummy run: both input_ids and inputs_embeds are None.
            # Note: SupportsMultiModal preprocessing converts input_ids to
            # inputs_embeds, so input_ids=None alone does NOT indicate a dummy run.
            if input_ids is None and inputs_embeds is None:
                dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

            runtime_info = kwargs.get("runtime_additional_information")
            talker_info = {}
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                talker_info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}

            with torch.inference_mode():
                talker_result = self.talker(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    additional_information=talker_info,
                )

            dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)

            # talker returns (mel_spec, waveform_or_None) tuple
            if isinstance(talker_result, tuple) and len(talker_result) == 2:
                mel_spec, waveform = talker_result
                mm_out = {}
                if mel_spec is not None:
                    mm_out["mel_spec"] = [mel_spec]
                if waveform is not None:
                    mm_out["model_outputs"] = [waveform]
                elif mel_spec is not None:
                    mm_out["model_outputs"] = [mel_spec]
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=mm_out)

            return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

        # Code2Wav stage: Vocos mel → waveform
        if self.model_stage == "code2wav":
            if input_ids is not None:
                n_tokens = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                n_tokens = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                n_tokens = 1
                device = torch.device("cuda")
            hidden_dim = self.config.hidden_size if hasattr(self.config, "hidden_size") else 2560

            # Profile/dummy run: both input_ids and inputs_embeds are None.
            if input_ids is None and inputs_embeds is None:
                dummy_hidden = torch.zeros(n_tokens, hidden_dim, device=device)
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

            runtime_info = kwargs.get("runtime_additional_information")
            code2wav_info = {}
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                code2wav_info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}

            mel_spec = code2wav_info.get("mel_spec")
            dummy_hidden = torch.zeros(n_tokens, hidden_dim, device=device)

            if mel_spec is not None and self.code2wav is not None:
                with torch.inference_mode():
                    waveform = self.code2wav(
                        input_ids=input_ids,
                        positions=positions,
                        inputs_embeds=mel_spec if isinstance(mel_spec, torch.Tensor) else None,
                        additional_information=code2wav_info,
                    )
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs={"model_outputs": [waveform]})

            logger.warning("Code2Wav: no mel_spec or code2wav model, returning empty")
            return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs={"model_outputs": [torch.zeros(0)]})
        
        raise ValueError(f"Unsupported model stage: {self.model_stage}")
    
    def compute_logits(self, hidden_states: Union[torch.Tensor, OmniOutput]) -> Optional[torch.Tensor]:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        
        # Use model for logits computation
        return self.model.compute_logits(hidden_states)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # Use model for sampling
        return self.model.sample(logits, sampling_metadata)
    
    def generate_audio(self, code: torch.Tensor, voice_type: str = "default") -> torch.Tensor:
        """
        Generate audio from code tokens using the code2wav model.
        
        Args:
            code: Code tokens from talker model
            voice_type: Voice type for speech generation (optional for MiniCPM-o)
            
        Returns:
            Audio tensor
        """
        if self.code2wav is None:
            logger.warning("Code2Wav model not initialized, cannot generate audio")
            return torch.zeros(0)
        
        code2wav_dev = self._module_device(self.code2wav)
        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            code_tensor = torch.as_tensor(code, dtype=torch.long, device=code2wav_dev)
        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)
        
        # Generate audio using code2wav model
        # TODO: Implement actual audio generation based on MiniCPM-o's code2wav implementation
        with torch.inference_mode():
            audio_tensor = self.code2wav(code_tensor)
        
        return audio_tensor
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        code2wav_weights = []
        
        # MiniCPM-o checkpoint prefixes → stage mapping:
        #   thinker: vpm, resampler, llm, apm, audio_projection_layer
        #   talker:  tts (ConditionalChatTTS)
        #   code2wav: (vocos loaded separately, not from main checkpoint)
        for k, v in weights:
            if k.startswith(("vpm.", "resampler.", "llm.", "apm.", "audio_projection_layer.")):
                thinker_weights.append((k, v))
            elif k.startswith("tts."):
                talker_weights.append((k, v))
            else:
                logger.warning("Unknown weight prefix: %s, skipping", k)

        # Load thinker weights
        if self.thinker is not None and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)
        
        # Load talker weights
        if self.talker is not None and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)
        
        # Load code2wav weights
        if self.code2wav is not None and code2wav_weights:
            code2wav_loaded = self.code2wav.load_weights(code2wav_weights)
            code2wav_loaded = add_prefix_to_loaded_weights(code2wav_loaded, "code2wav")
            loaded_weights.update(code2wav_loaded)
        
        return loaded_weights
