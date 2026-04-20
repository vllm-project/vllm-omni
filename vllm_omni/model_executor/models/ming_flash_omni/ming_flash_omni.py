# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
# Adapted from Ming repository modeling_bailingmm2.py
# https://github.com/inclusionAI/Ming
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

"""Ming-flash-omni-2.0 unified model (thinker + imagegen + talker)."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config, MingFlashOmniConfig

from .ming_flash_omni_thinker import (
    MingFlashOmniThinkerDummyInputsBuilder,
    MingFlashOmniThinkerMultiModalProcessor,
    MingFlashOmniThinkerProcessingInfo,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MingFlashOmniThinkerMultiModalProcessor,
    info=MingFlashOmniThinkerProcessingInfo,
    dummy_inputs=MingFlashOmniThinkerDummyInputsBuilder,
)
class MingFlashOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """Unified Ming-flash-omni-2.0 model combining thinker, imagegen, and talker."""

    supports_multimodal = True
    requires_raw_input_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        config = vllm_config.model_config.hf_config

        self.vllm_config = vllm_config
        self.config = config

        if isinstance(config, MingFlashOmniConfig):
            thinker_config = config.thinker_config
        else:
            thinker_config = config

        self.thinker_config: BailingMM2Config = thinker_config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["MingFlashOmniThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                architectures=["MingFlashOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.imagegen = None
            self.talker = None

        elif self.model_stage == "imagegen":
            # TODO: Implement image generator stage
            raise NotImplementedError(
                "Image generation stage is not yet implemented. Please use model_stage='thinker' for now."
            )

        elif self.model_stage == "talker":
            # TODO: Implement talker (TTS) stage
            raise NotImplementedError(
                "Talker (TTS) stage is not yet implemented. Please use model_stage='thinker' for now."
            )

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be one of: 'thinker', 'imagegen', 'talker'"
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage == "thinker" else lambda: None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        return self.model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        raise NotImplementedError("sample method not available on current stage")

    def get_mrope_input_positions(self, *args, **kwargs):
        if hasattr(self.model, "get_mrope_input_positions"):
            return self.model.get_mrope_input_positions(*args, **kwargs)
        raise NotImplementedError("get_mrope_input_positions not available on current stage")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()
        thinker_weights = []
        imagegen_weights = []
        talker_weights = []

        for name, value in weights:
            if name.startswith("thinker."):
                thinker_weights.append((name, value))
            elif name.startswith("imagegen."):
                imagegen_weights.append((name, value))
            elif name.startswith("talker."):
                talker_weights.append((name, value))
            else:
                # Weights without prefix go to thinker by default
                thinker_weights.append((name, value))

        if self.model_stage == "thinker" and thinker_weights:
            # Remove "thinker." prefix before loading
            thinker_weights_stripped = [
                (name.replace("thinker.", "", 1) if name.startswith("thinker.") else name, value)
                for name, value in thinker_weights
            ]
            thinker_loaded = self.thinker.load_weights(thinker_weights_stripped)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # TODO: Load imagegen weights when implemented
        # TODO: Load talker weights when implemented

        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="thinker.language_model",
            connector=["thinker.linear_proj.", "thinker.linear_proj_audio."],
            tower_model=["thinker.vision.", "thinker.audio."],
        )

    @property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs):
        return self.model.embed_multimodal(**kwargs)
