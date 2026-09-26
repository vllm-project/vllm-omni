# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared backbone composition helpers for Pi-family action models."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, apply_rotary_pos_emb
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from vllm_omni.diffusion.models.pi.common import attention

PrefixKV = tuple[torch.Tensor, torch.Tensor]
ModuleInputAligner = Callable[[torch.Tensor, nn.Module], torch.Tensor]


@dataclass(frozen=True)
class GemmaVariantConfig:
    """OpenPI Gemma dimensions used by the Pi-family backbones."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    """Return the OpenPI dimensions for a supported Gemma variant."""
    if variant == "gemma_2b":
        return GemmaVariantConfig(2048, 18, 16384, 8, 1, 256)
    if variant == "gemma_300m":
        return GemmaVariantConfig(1024, 18, 4096, 8, 1, 256)
    raise ValueError(f"Unknown variant: {variant}")


def build_backbones(
    vlm_config: GemmaVariantConfig,
    action_expert_config: GemmaVariantConfig,
) -> tuple[PaliGemmaForConditionalGeneration, GemmaForCausalLM]:
    """Build the common PaliGemma prefix and stock Gemma action expert.

    Variants may modify the returned expert after construction. Pi0 keeps the
    stock RMSNorms, while Pi0.5 replaces them with timestep-conditioned AdaRMS
    modules. The returned modules are assigned directly to each concrete
    model's existing ``paligemma`` and ``gemma_expert`` attributes, preserving
    checkpoint names.
    """
    vlm_config_hf = CONFIG_MAPPING["paligemma"]()
    vlm_config_hf._vocab_size = 257152
    vlm_config_hf.image_token_index = 257152
    vlm_config_hf.text_config.hidden_size = vlm_config.width
    vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
    vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
    vlm_config_hf.text_config.head_dim = vlm_config.head_dim
    vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
    vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
    vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
    # transformers >= 5 uses ``dtype``; older versions still accept
    # ``torch_dtype``. PretrainedConfig normalizes one to the other.
    vlm_config_hf.text_config.dtype = "float32"
    vlm_config_hf.text_config.vocab_size = 257152
    vlm_config_hf.vision_config.intermediate_size = 4304
    vlm_config_hf.vision_config.projection_dim = 2048
    vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
    vlm_config_hf.vision_config.dtype = "float32"

    action_expert_config_hf = CONFIG_MAPPING["gemma"](
        head_dim=action_expert_config.head_dim,
        hidden_size=action_expert_config.width,
        intermediate_size=action_expert_config.mlp_dim,
        num_attention_heads=action_expert_config.num_heads,
        num_hidden_layers=action_expert_config.depth,
        num_key_value_heads=action_expert_config.num_kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
    )

    paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
    gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
    # The action expert consumes projected state/action embeddings, not tokens.
    gemma_expert.model.embed_tokens = None
    return paligemma, gemma_expert


def embed_image(paligemma: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode images explicitly through SigLIP and the multimodal projector.

    ``PaliGemmaModel.get_image_features`` has changed scaling behavior across
    Transformers releases. Calling the two stable submodules directly keeps
    Pi0 and Pi0.5 on the same unambiguous path.
    """
    vision_tower = paligemma.model.vision_tower
    parameters = getattr(vision_tower, "parameters", None)
    if parameters is not None:
        target_dtype = next((param.dtype for param in parameters() if param.is_floating_point()), None)
        if target_dtype is not None:
            pixel_values = pixel_values.to(dtype=target_dtype)
    vision_outputs = vision_tower(pixel_values)
    return paligemma.model.multi_modal_projector(vision_outputs.last_hidden_state)


def embed_language_tokens(paligemma: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Return canonical Gemma-scaled language embeddings across releases.

    Transformers <=5.3 applies ``sqrt(hidden_size)`` inside ``GemmaModel.forward``
    (which the Pi kernels bypass). In >=5.4 the embedding module self-applies
    that scale. Detecting ``embed_scale`` prevents both missing and double scale.
    """
    embed_tokens = paligemma.model.language_model.embed_tokens
    embeddings = embed_tokens(tokens)
    if getattr(embed_tokens, "embed_scale", None) is None:
        embeddings = embeddings * math.sqrt(embeddings.shape[-1])
    return embeddings


def _keep_input_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Leave module inputs unchanged when the variant needs no dtype bridge."""
    del module
    return tensor


def embed_multimodal_prefix(
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    *,
    paligemma: nn.Module,
    expected_num_views: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ordered camera and language embeddings into one prefix.

    Cameras are embedded one slot at a time and retain their configured order.
    Each per-batch camera-validity bit is expanded across that camera's image
    tokens, so a missing camera keeps its fixed token positions while masking
    every token in the slot. Language embeddings and their padding mask follow
    the image slots.

    Language embedding scaling and the explicit SigLIP/projector path are owned
    here so variants cannot drift. All prefix block markers are false, making
    the prefix bidirectional; each variant declares its own causal boundary
    when it appends the suffix.

    Camera count remains caller-owned: variants with a fixed deployment layout
    pass ``expected_num_views``; variants without one leave it unset.
    """
    num_views = len(images)
    if len(image_masks) != num_views:
        raise ValueError(
            f"images and image_masks must contain the same number of views, got {num_views} and {len(image_masks)}."
        )
    if expected_num_views is not None and num_views != expected_num_views:
        raise ValueError(f"Expected exactly {expected_num_views} image views, got {num_views}.")

    embeddings: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []

    for image, image_mask in zip(images, image_masks):
        image_embedding = embed_image(paligemma, image)
        batch_size, num_image_tokens = image_embedding.shape[:2]
        embeddings.append(image_embedding)
        padding_masks.append(image_mask[:, None].expand(batch_size, num_image_tokens))

    language_embedding = embed_language_tokens(paligemma, lang_tokens)
    embeddings.append(language_embedding)
    padding_masks.append(lang_masks)

    prefix_embeddings = torch.cat(embeddings, dim=1)
    prefix_padding_masks = torch.cat(padding_masks, dim=1)
    prefix_attention_markers = torch.zeros(
        (prefix_padding_masks.shape[0], prefix_embeddings.shape[1]),
        dtype=torch.bool,
        device=prefix_embeddings.device,
    )
    return prefix_embeddings, prefix_padding_masks, prefix_attention_markers


def execute_prefix_layer(
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    paligemma: nn.Module,
    *,
    align_module_input: ModuleInputAligner = _keep_input_dtype,
) -> tuple[torch.Tensor, PrefixKV]:
    """Run one ordinary PaliGemma prefix layer and return its post-RoPE K/V.

    The prefix contains ordered image and language tokens and never sees the
    action expert's timestep conditioning. Both Pi variants therefore use the
    stock Gemma residual path here. ``align_module_input`` keeps dtype policy
    explicit: Pi0 currently preserves inputs, while Pi0.5 aligns inputs with
    each projection for mixed-dtype checkpoints.

    K/V are cached after RoPE with shape
    ``(batch, num_kv_heads, prefix_length, head_dim)``. The model-specific
    suffix executor later combines them with freshly projected suffix K/V.
    """
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]

    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    normalized = align_module_input(normalized, layer.self_attn.q_proj)

    hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)

    cos, sin = model.rotary_emb(value, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)

    attended = attention.eager_attention(
        query,
        key,
        value,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    attended = attended.transpose(1, 2).reshape(
        query.shape[0],
        -1,
        query.shape[1] * layer.self_attn.head_dim,
    )

    attended = align_module_input(attended, layer.self_attn.o_proj)
    hidden_states = layer.self_attn.o_proj(attended) + residual
    residual = hidden_states

    normalized = layer.post_attention_layernorm(hidden_states)
    normalized = align_module_input(normalized, layer.mlp.up_proj)
    hidden_states = layer.mlp(normalized) + residual
    return hidden_states, (key, value)


def execute_prefix(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    paligemma: nn.Module,
    *,
    align_module_input: ModuleInputAligner = _keep_input_dtype,
) -> tuple[torch.Tensor, list[PrefixKV]]:
    """Execute the complete shared PaliGemma prefix and collect its KV cache."""
    language_model = paligemma.model.language_model
    kv_cache: list[PrefixKV] = []
    for layer_idx in range(len(language_model.layers)):
        hidden_states, layer_kv = execute_prefix_layer(
            layer_idx,
            hidden_states,
            attention_mask,
            position_ids,
            paligemma,
            align_module_input=align_module_input,
        )
        kv_cache.append(layer_kv)
    return language_model.norm(hidden_states), kv_cache
