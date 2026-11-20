# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sesame CSM stage 0 core: Llama backbone, first-codebook head and depth decoder.

One engine token is one audio frame. The backbone hidden state gives the first
codebook through ``lm_head``, the depth decoder completes the frame, and the sum
of the frame's codebook embeddings is the backbone's next input.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaModel
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix

from vllm_omni.model_executor.models.csm.csm_depth_decoder import CsmDepthDecoder


def stop_frame_mask(codes: torch.Tensor, codebook_eos_token_id: int = 0) -> torch.Tensor:
    """Frames that end generation: every codebook except the last is the EOS id."""
    return (codes[..., :-1] == codebook_eos_token_id).all(dim=-1)


def empty_frame_mask(codes: torch.Tensor, codebook_eos_token_id: int = 0) -> torch.Tensor:
    """Frames where audio is cut: every codebook, including the last, is the EOS id."""
    return (codes == codebook_eos_token_id).all(dim=-1)


class CsmAudioEmbeddings(nn.Module):
    """Embeds an audio frame as the sum of its codebook embeddings."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_audio_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, config.hidden_size)
        self.register_buffer(
            "audio_tokens_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed ``codes`` of shape ``[..., num_codebooks]``."""
        return self.embed_audio_tokens(codes + self.audio_tokens_offsets).sum(dim=-2)


class CsmBackbone(nn.Module):
    """Llama decoder layers and final norm of the CSM backbone."""

    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.embed_tokens = CsmAudioEmbeddings(config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(vllm_config=vllm_config, prefix=prefix, config=config),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Return the hidden states after the final norm."""
        hidden_states = inputs_embeds
        residual = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone weights, stacking q/k/v and gate/up projections."""
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)


class CsmTalkerForConditionalGeneration(nn.Module):
    """CSM backbone, first-codebook head and depth decoder for ``sesame/csm-1b``."""

    # The repository also ships the same weights in the original format (model.safetensors, ckpt.pt).
    allow_patterns_overrides = ["transformers-*.safetensors"]
    fall_back_to_pt_during_load = False

    # The Mimi codec is unused in this stage.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"codec_model.": None})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not getattr(config, "tie_codebooks_embeddings", True):
            raise ValueError("CSM checkpoints with untied codebook embeddings are not supported")
        self.config = config
        self.num_codebooks = int(config.num_codebooks)
        self.codebook_vocab_size = int(config.vocab_size)
        self.codebook_eos_token_id = int(config.codebook_eos_token_id)

        self.embed_text_tokens = VocabParallelEmbedding(
            config.text_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.text_vocab_size,
            prefix=maybe_prefix(prefix, "embed_text_tokens"),
        )
        self.backbone_model = CsmBackbone(
            vllm_config=vllm_config, config=config, prefix=maybe_prefix(prefix, "backbone_model")
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.depth_decoder = CsmDepthDecoder(config.depth_decoder_config)
        # Tied in the checkpoint (tie_codebooks_embeddings).
        self.depth_decoder.model.embed_tokens.weight = self.backbone_model.embed_tokens.embed_audio_tokens.weight

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed prompt text tokens."""
        return self.embed_text_tokens(input_ids)

    def embed_audio_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed audio frames of shape ``[..., num_codebooks]`` as backbone inputs."""
        return self.backbone_model.embed_tokens(codes)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Run the backbone and return its hidden states."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.backbone_model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        """Return first-codebook logits."""
        return self.lm_head(hidden_states)

    def generate_frame(
        self,
        backbone_hidden: torch.Tensor,
        first_codes: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        uniforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete the frames started by ``first_codes`` with the depth decoder."""
        return self.depth_decoder.generate_frame(
            backbone_hidden,
            first_codes,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            uniforms=uniforms,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a transformers-format CSM checkpoint."""
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
