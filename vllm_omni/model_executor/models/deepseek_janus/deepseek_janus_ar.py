# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus — AR image-token generation model for vLLM engine.

This model extends vLLM's ``LlamaForCausalLM``, which means it automatically
inherits ALL vLLM LLM optimisations:

  - PagedAttention (block-based KV cache with paged memory management)
  - CUDA Graph via CUDAGraphWrapper (automatic capture/replay of decode steps)
  - Two-phase prefill/decode execution (via attention metadata)
  - Chunked prefill (split long prompts for memory efficiency)
  - Prefix caching (automatic KV cache reuse for shared prefixes)
  - FP8 KV cache quantization (reduced memory footprint)
  - FlashInfer/FA3 attention backends (auto-selected best kernel)
  - Tensor parallelism (multi-GPU model distribution)
  - Continuous / asynchronous scheduling

The model adds Janus-specific components on top of the Llama backbone:
  - ``gen_head``: linear layer mapping hidden states → image token logits
  - ``gen_embed`` + ``gen_aligner``: embed image tokens for AR decode steps
  - CFG (classifier-free guidance): cond / uncond batch doubling with logit merging

For single-image generation, use the single-stage deploy config which runs the
full pipeline (AR loop + VQ decode) inside the diffusion worker.  For online
serving with continuous batching, use the two-stage config where this model
runs in the AR engine (stage 0) and a lightweight VQ decoder runs as stage 1.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from addict import Dict as AttrDict
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM

logger = init_logger(__name__)


class OmniDeepSeekJanusForConditionalGeneration(LlamaForCausalLM):
    """LLaMA decoder (+ LM head) for the Janus ``language_model`` subtree.

    Used in two-stage deployments where stage 0 generates text via the Llama
    backbone (prompt refinement / instruction following) and stage 1 handles
    image generation via the full Janus pipeline.

    For image token generation specifically, use :class:`JanusForImageGeneration`
    which adds gen_head and CFG support on top of this class.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[torch.nn.Module] = LlamaDecoderLayer,
    ) -> None:
        mm_cfg = vllm_config.model_config.hf_config
        lang = getattr(mm_cfg, "language_config", None)
        if lang is None:
            raise ValueError("DeepSeek Janus expects a MultiModality-style HF config with ``language_config`` (Llama).")
        if isinstance(lang, LlamaConfig):
            lang_cfg = lang
        elif hasattr(lang, "to_dict"):
            lang_cfg = LlamaConfig(**lang.to_dict())
        else:
            lang_cfg = LlamaConfig(**dict(lang))

        orig_hf = vllm_config.model_config.hf_config
        object.__setattr__(vllm_config.model_config, "hf_config", lang_cfg)
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        finally:
            object.__setattr__(vllm_config.model_config, "hf_config", orig_hf)

        self.janus_multi_modality_hf_config = mm_cfg

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _language_rows(it: Iterable[tuple[str, torch.Tensor]]):
            for name, tensor in it:
                if name.startswith("language_model."):
                    yield name[len("language_model.") :], tensor

        return super().load_weights(_language_rows(weights))


class JanusForImageGeneration(LlamaForCausalLM):
    """Janus AR image-token generation model with full vLLM optimisation stack.

    Inherits from ``LlamaForCausalLM`` to get PagedAttention, CUDAGraphWrapper,
    chunked prefill, prefix caching, FP8 KV cache, FlashInfer, tensor parallelism,
    and continuous batching — all automatically through vLLM's GPU model runner.

    Janus-specific additions:
      - ``gen_head``: Linear(hidden_size → gen_vocab_size) for image token logits
      - ``gen_embed``: Embedding(gen_vocab_size → gen_n_embed) for image token lookup
      - ``gen_aligner``: MLP projector mapping gen embeddings to hidden_size
      - CFG support: batch doubling with conditional/unconditional logit merging

    The model's ``compute_logits()`` method applies gen_head and optionally
    performs CFG merging if the batch contains paired (cond, uncond) sequences.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[torch.nn.Module] = LlamaDecoderLayer,
    ) -> None:
        mm_cfg = vllm_config.model_config.hf_config
        lang = getattr(mm_cfg, "language_config", None)
        if lang is None:
            raise ValueError("DeepSeek Janus expects a MultiModality-style HF config with ``language_config`` (Llama).")
        if isinstance(lang, LlamaConfig):
            lang_cfg = lang
        elif hasattr(lang, "to_dict"):
            lang_cfg = LlamaConfig(**lang.to_dict())
        else:
            lang_cfg = LlamaConfig(**dict(lang))

        orig_hf = vllm_config.model_config.hf_config
        object.__setattr__(vllm_config.model_config, "hf_config", lang_cfg)
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        finally:
            object.__setattr__(vllm_config.model_config, "hf_config", orig_hf)

        self.janus_multi_modality_hf_config = mm_cfg

        # --- Janus image generation heads ---
        gen_vision_config = mm_cfg.gen_vision_config
        gen_head_config = mm_cfg.gen_head_config
        gen_aligner_config = mm_cfg.gen_aligner_config

        self._gen_vocab_size = int(gen_vision_config.params.image_token_size)
        gen_n_embed = int(gen_vision_config.params.n_embed)

        # gen_head: maps hidden states → image token logits
        # Follows the same pattern as `vision_head` in modeling_vlm.py
        self.gen_head = torch.nn.Sequential(
            torch.nn.Linear(gen_head_config.params.n_embed, gen_head_config.params.image_token_embed),
            torch.nn.GELU(),
            torch.nn.Linear(gen_head_config.params.image_token_embed, gen_head_config.params.image_token_size),
        )

        # gen_embed: image token → embedding
        self.gen_embed = torch.nn.Embedding(self._gen_vocab_size, gen_n_embed)

        # gen_aligner: embedding → hidden_size
        from vllm_omni.diffusion.models.deepseek_janus._janus_hf_vendor.projector import MlpProjector

        self.gen_aligner = MlpProjector(AttrDict(gen_aligner_config.params))

        # CFG parameters (set at runtime)
        self._cfg_weight: float = 5.0
        self._cfg_enabled: bool = False

    def set_cfg_params(self, cfg_weight: float, enabled: bool = True) -> None:
        """Configure CFG for the next generation request."""
        self._cfg_weight = cfg_weight
        self._cfg_enabled = enabled

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute image token logits via gen_head, with optional CFG merging.

        When CFG is enabled, the hidden states are expected to be interleaved
        as [cond_0, uncond_0, cond_1, uncond_1, ...] and this method applies
        the CFG formula:  uncond + cfg_weight * (cond - uncond).
        """
        logits = self.gen_head(hidden_states)

        if self._cfg_enabled and logits.shape[0] % 2 == 0:
            cond_logits = logits[0::2]
            uncond_logits = logits[1::2]
            merged = uncond_logits + self._cfg_weight * (cond_logits - uncond_logits)
            return merged

        return logits

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor) -> torch.Tensor:
        """Convert image token IDs to input embeddings (for decode steps)."""
        return self.gen_aligner(self.gen_embed(image_ids))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from the MultiModalityCausalLM checkpoint.

        Weight paths are prefixed with ``language_model.``, ``gen_head.``,
        ``gen_embed.``, ``gen_aligner.``, etc.  We route each prefix
        to the appropriate sub-module.
        """
        from vllm_omni.diffusion.models.deepseek_janus import _load_param

        weights = list(weights)
        loaded: set[str] = set()

        # Weights for the Llama backbone (language_model.*)
        def _language_rows(it):
            for name, tensor in it:
                if name.startswith("language_model."):
                    yield name[len("language_model.") :], tensor

        loaded.update(super().load_weights(_language_rows(weights)))

        # Weights for Janus-specific heads
        for name, tensor in weights:
            if name.startswith("gen_head."):
                sub = name[len("gen_head.") :]
                # Map checkpoint keys to Sequential indices
                # Checkpoint uses: output_mlp_projector.weight/bias,
                #                 vision_head.weight/bias
                param_map = {
                    "output_mlp_projector.weight": "0.weight",
                    "output_mlp_projector.bias": "0.bias",
                    "vision_head.weight": "2.weight",
                    "vision_head.bias": "2.bias",
                }
                mapped = param_map.get(sub, sub)
                try:
                    _load_param(self.gen_head, mapped, tensor)
                    loaded.add(name)
                except Exception as e:
                    logger.warning("Failed to load gen_head weight %s: %s", name, e)

            elif name.startswith("gen_embed."):
                sub = name[len("gen_embed.") :]
                if sub == "weight":
                    try:
                        self.gen_embed.weight.data.copy_(tensor)
                        loaded.add(name)
                    except Exception as e:
                        logger.warning("Failed to load gen_embed weight: %s", e)

            elif name.startswith("gen_aligner."):
                sub = name[len("gen_aligner.") :]
                try:
                    _load_param(self.gen_aligner, sub, tensor)
                    loaded.add(name)
                except Exception as e:
                    logger.warning("Failed to load gen_aligner weight %s: %s", name, e)

        return loaded
