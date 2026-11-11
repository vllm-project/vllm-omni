# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio: the LFM3-Omni temporal/depth stage with the checkpoint's speech encoder only."""

from collections.abc import Iterable
from types import ModuleType

import regex as re
import torch
from transformers import PretrainedConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.lfm3_omni.model import Lfm3OmniForConditionalGeneration

from .processing import Lfm3AudioDummyInputsBuilder, Lfm3AudioMultiModalProcessor, Lfm3AudioProcessingInfo

# Each tied depth head is also exported as `to_logits`; the reference ignores the copy.
_TIED_DEPTH_HEAD = re.compile(r"depth_embeddings\.\d+\.to_logits\.weight$")


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3AudioMultiModalProcessor, info=Lfm3AudioProcessingInfo, dummy_inputs=Lfm3AudioDummyInputsBuilder
)
class Lfm3AudioForConditionalGeneration(Lfm3OmniForConditionalGeneration):
    _reference = ("modeling_lfm3_audio", "Lfm3AudioModel")
    # The released checkpoint keeps the training export's module names; this
    # mirrors the reference's checkpoint conversion mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lfm.model.": "model.language_model.",
            "lfm.lm_head.": "lm_head.",
            "encoder.downsample.": "model.audio_encoder.downsample.",
            "encoder.lfm.lfm.": "model.audio_encoder.lfm.",
            "mlp.model.0.": "model.audio_projector.layer_norm.",
            "mlp.model.1.": "model.audio_projector.linear_1.",
            "mlp.model.3.": "model.audio_projector.linear_2.",
            "codebook_embedding.": "model.codebook_embedding.",
            "depth_linear.": "model.depth_decoder.depth_linear.",
            "mask_embedding": "model.depth_decoder.mask_embedding",
            "depthformer.model.": "model.depth_decoder.model.",
            "depth_embeddings.": "model.depth_decoder.depth_embeddings.",
        }
    )

    def _build_reference_modules(self, ref: ModuleType, c: PretrainedConfig) -> None:
        self.model.audio_encoder = ref.Lfm3AudioEncoder(c.audio_encoder_config)
        self.model.audio_projector = ref.Lfm3AudioProjector(c)
        self.model.depth_decoder = ref.Lfm3AudioDepthDecoder(c)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        return super().load_weights((name, tensor) for name, tensor in weights if not _TIED_DEPTH_HEAD.search(name))
