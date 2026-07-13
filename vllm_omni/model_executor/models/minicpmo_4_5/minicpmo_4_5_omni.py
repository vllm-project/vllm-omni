# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
#
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

from collections.abc import Iterable
from functools import cached_property

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

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMDummyInputsBuilder,
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMO45OmniLLMMultiModalProcessor,
    info=MiniCPMO45OmniLLMProcessingInfo,
    dummy_inputs=MiniCPMO45OmniLLMDummyInputsBuilder,
)
class MiniCPMO45OmniForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    """MiniCPM-o 4.5 Omni model for conditional generation.

    Two-stage pipeline:
    - thinker (model_stage="llm"): image / video / audio encoders + 3D
      resampler + the omni LLM that emits text + hidden states.
    - talker  (model_stage="tts"): MiniCPMTTS + the in-process Token2Wav
      vocoder that emits the final audio waveform directly.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
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

        if self.model_stage == "llm":
            # Initialize thinker model (image preprocessing + vision encoder + 3D resampler)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                # Registry key — must match the entries declared in
                # vllm_omni/model_executor/models/registry.py::_OMNI_MODELS.
                architectures=["MiniCPMO45OmniLLMForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None

        elif self.model_stage == "tts":
            self.thinker = None
            # Initialize talker model — runs MiniCPMTTS + the in-process
            # Token2Wav vocoder and emits the final audio waveform directly.
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=config,
                # Registry key — must match the entries declared in
                # vllm_omni/model_executor/models/registry.py::_OMNI_MODELS.
                architectures=["MiniCPMO45OmniTTSForConditionalGeneration"],
            )
            # Initialize multimodal components if needed
            if hasattr(self.talker, "init_multi_modal"):
                self.talker.init_multi_modal(config)
            self.model = self.talker

        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}. Must be one of: 'llm', 'tts'")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors)
            if self.model_stage == "llm" and self.thinker is not None
            else lambda: None
        )

        self._language_model_names = ["model"]

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
        thinker_device: str | torch.device | None = None,
        talker_device: str | torch.device | None = None,
    ) -> None:
        """Optionally move thinker / talker to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "tts":
            return self.get_input_embeddings(input_ids)
        return super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

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
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Forward pass for MiniCPM-o Omni model.

        Workflow:
        1) Thinker (model_stage="llm"): Image / video / audio encoders +
           3D resampler + omni LLM → text + hidden states.
        2) Talker (model_stage="tts"): MiniCPMTTS + the in-process
           Token2Wav vocoder → audio waveform (final pipeline output).
        """
        if self.model_stage == "llm":
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
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )

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

        # Talker stage: runs MiniCPMTTS + the in-process Token2Wav vocoder and
        # emits the final audio waveform directly.
        if self.model_stage == "tts":
            if input_ids is not None:
                num_tokens = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                num_tokens = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                num_tokens = 1
                device = current_omni_platform.get_torch_device()
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

            # Talker returns a (mel_spec, waveform_or_None) tuple. MiniCPM-o
            # 4.5 emits only the waveform (mel_spec is always None); keep the
            # 2-slot unpack so a future mel-emitting variant can plug in
            # without changing the wrapper.
            mm_out: dict = {}
            if isinstance(talker_result, tuple) and len(talker_result) == 2:
                _, waveform = talker_result
                if waveform is not None:
                    mm_out["model_outputs"] = [waveform]

            return OmniOutput(
                text_hidden_states=dummy_hidden,
                multimodal_outputs=mm_out if mm_out else None,
            )

        raise ValueError(f"Unsupported model stage: {self.model_stage}")

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use model for logits computation
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        # Use model for sampling
        return self.model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the active stage of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []

        # MiniCPM-o checkpoint prefixes → stage mapping:
        #   thinker: vpm, resampler, llm, apm, audio_projection_layer
        #   talker:  tts (MiniCPMTTS); the Token2Wav vocoder weights load
        #            separately inside the talker module from the
        #            ``assets/token2wav`` subdirectory and do not appear
        #            in this iterator.
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

        return loaded_weights
