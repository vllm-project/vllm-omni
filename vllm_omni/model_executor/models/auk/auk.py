# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK (Tencent): instruction-driven speech generation and editing.

Stage 0 (``model_stage="encoder"``) runs the frozen Qwen2.5-Omni thinker as a
prefill-only encoder. The outputs of all decoder layers are fused with learned
softmax weights into one ``[tokens, hidden]`` text condition, emitted through
``multimodal_outputs["hidden_states"]["output"]`` for the diffusion stage
(``vllm_omni/diffusion/models/auk``). The stage samples exactly one token
(``max_tokens=1``, forced to EOS); that token is discarded downstream.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import cached_property
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

logger = init_logger(__name__)

# File next to the thinker shards in the assembled checkpoint directory that
# carries the DiT (``transformer.*``) and the two layer-fusion tensors.
AUK_WEIGHTS_FILE = "auk.safetensors"
LAYER_WEIGHTS_KEY = "layer_weights"
LAYER_SCALE_KEY = "layer_scale"
# The fused text condition travels as ``multimodal_outputs["hidden_states"]["output"]``
# (``HiddenStatesStruct.output``), the same stage-wire slot MiniMax H3 uses.
TEXT_COND_KEY = "output"
# ``F.layer_norm`` epsilon used by the upstream fusion (torch default).
_FUSION_LN_EPS = 1e-5


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class AuKForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    """Encoder stage of AuK: Qwen2.5-Omni thinker with learned layer fusion.

    The forward pass re-implements the thinker language model's layer loop so
    the per-layer outputs can be fused on the fly: vLLM's decoder layers carry
    the residual stream separately, and ``hidden + residual`` after layer ``i``
    is the HF ``output_hidden_states[i + 1]`` entry, while the last entry is
    the final-normed state. Only one accumulator lives at a time.
    """

    have_multimodal_outputs = True
    # The fused condition is the payload; the runner's default last-hidden
    # copy and prefix-cached hidden states are not needed (MiniMax H3 pattern).
    omni_pooler_payload_include_hidden = False
    requires_full_prefix_cached_hidden_states = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config

        self.model_stage = getattr(vllm_config.model_config, "model_stage", None) or "encoder"
        if self.model_stage != "encoder":
            raise ValueError(
                f"AuKForConditionalGeneration only hosts the 'encoder' stage, got model_stage={self.model_stage!r}; "
                "the DiT stage is the AuKPipeline diffusion model."
            )

        thinker_config = config.thinker_config
        self.thinker_config = thinker_config
        self.thinker = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "thinker"),
            hf_config=thinker_config,
            architectures=["Qwen2_5OmniThinkerModel"],
        )
        self.model = self.thinker
        self.make_empty_intermediate_tensors = self.thinker.make_empty_intermediate_tensors

        num_layers = thinker_config.text_config.num_hidden_layers
        # Filled by load_weights() from AUK_WEIGHTS_FILE; buffers so the vLLM
        # loader does not expect them among the thinker shards.
        self.register_buffer("layer_weights", torch.zeros(num_layers, dtype=torch.float32))
        self.register_buffer("layer_scale", torch.ones(1, dtype=torch.float32))
        self._fusion_loaded = False

    # -------------------- delegations to the thinker --------------------

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return Qwen2_5OmniThinkerForConditionalGeneration.get_placeholder_str(modality, i)

    def get_language_model(self) -> nn.Module:
        return self.thinker.get_language_model()

    @cached_property
    def sampler(self):
        if hasattr(self.thinker, "sampler"):
            return self.thinker.sampler
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.thinker.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs):
        return self.thinker.embed_multimodal(**kwargs)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        *,
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor,
        video_grid_thw: list[list[int]] | torch.Tensor,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        # The omni runner filters kwargs by this signature (the extended
        # Qwen2.5-Omni form); the unified Qwen2.5-Omni class implements the
        # audio-aware position math without touching instance state.
        return Qwen2_5OmniForConditionalGeneration.get_mrope_input_positions(
            self,
            input_tokens,
            mm_features,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            context_len=context_len,
            seq_len=seq_len,
            audio_feature_lengths=audio_feature_lengths,
            use_audio_in_video=use_audio_in_video,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, **kwargs: object) -> torch.Tensor | None:
        """Emit EOS immediately; the useful result is the prefill text condition."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        text_config = self.thinker_config.text_config
        logits = hidden_states.new_full(
            (hidden_states.shape[0], int(text_config.vocab_size)),
            torch.finfo(hidden_states.dtype).min,
        )
        logits[:, self._eos_token_id()] = 0
        return logits

    def _eos_token_id(self) -> int:
        for cfg in (self.thinker_config.text_config, self.thinker_config, self.vllm_config.model_config.hf_config):
            eos = getattr(cfg, "eos_token_id", None)
            if isinstance(eos, (list, tuple)) and eos:
                eos = eos[0]
            if isinstance(eos, int):
                return eos
        # Qwen2.5-Omni tokenizers end turns with <|im_end|>.
        return 151645

    # -------------------- forward --------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if intermediate_tensors is not None:
            raise NotImplementedError("AuK encoder does not support pipeline parallelism")
        language_model = self.thinker.language_model.model
        num_layers = len(language_model.layers)
        if language_model.start_layer != 0 or language_model.end_layer != num_layers:
            raise NotImplementedError("AuK encoder needs all decoder layers on one rank")

        if inputs_embeds is None:
            hidden_states = language_model.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        residual: torch.Tensor | None = None

        weights = torch.softmax(self.layer_weights, dim=0)
        fused: torch.Tensor | None = None
        normed_shape = (hidden_states.shape[-1],)
        layers = islice(language_model.layers, language_model.start_layer, language_model.end_layer)
        for idx, layer in enumerate(layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if idx + 1 < num_layers:
                layer_output = hidden_states + residual
            else:
                # HF's last hidden_states entry is post final norm.
                hidden_states, _ = language_model.norm(hidden_states, residual)
                layer_output = hidden_states
            normed = F.layer_norm(layer_output.float(), normed_shape, eps=_FUSION_LN_EPS)
            contribution = normed * weights[idx]
            fused = contribution if fused is None else fused + contribution

        assert fused is not None
        text_cond = (fused * self.layer_scale).to(hidden_states.dtype)
        # Stage-wire layout: [tokens, features] under hidden_states.output so
        # the runner slices it per request (and concatenates across chunked
        # prefill steps); the stage-1 processor reads the same path.
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={"hidden_states": {"output": text_cond}},
        )

    # -------------------- weights --------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        thinker_weights: list[tuple[str, torch.Tensor]] = []
        skipped = 0
        for name, tensor in weights:
            if name.startswith("thinker."):
                thinker_weights.append((name, tensor))
            elif name.startswith(("talker.", "token2wav.", "transformer.")):
                skipped += 1
            else:
                logger.warning("AuK encoder: ignoring unexpected weight %s", name)
        if skipped:
            logger.debug("AuK encoder: skipped %d non-thinker weights", skipped)

        loaded = self.thinker.load_weights(thinker_weights)
        loaded = add_prefix_to_loaded_weights(loaded, "thinker")
        self._load_fusion_params()
        return loaded

    def _load_fusion_params(self) -> None:
        model_dir = self.vllm_config.model_config.model
        path = os.path.join(model_dir, AUK_WEIGHTS_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"AuK encoder: {AUK_WEIGHTS_FILE} not found in {model_dir!r}; assemble the checkpoint with "
                "tools/prepare_auk_checkpoint.py"
            )
        with safe_open(path, framework="pt", device="cpu") as f:
            layer_weights = f.get_tensor(LAYER_WEIGHTS_KEY).float()
            layer_scale = f.get_tensor(LAYER_SCALE_KEY).float().reshape(1)
        if layer_weights.shape != self.layer_weights.shape:
            raise ValueError(
                f"AuK encoder: {LAYER_WEIGHTS_KEY} has {tuple(layer_weights.shape)} entries, "
                f"thinker has {tuple(self.layer_weights.shape)} layers"
            )
        self.layer_weights.copy_(layer_weights.to(self.layer_weights.device))
        self.layer_scale.copy_(layer_scale.to(self.layer_scale.device))
        self._fusion_loaded = True
        logger.info(
            "AuK encoder: loaded layer fusion (scale %.3f, top layer weight %.3f at layer %d)",
            float(layer_scale[0]),
            float(torch.softmax(layer_weights, 0).max()),
            int(torch.softmax(layer_weights, 0).argmax()) + 1,
        )
