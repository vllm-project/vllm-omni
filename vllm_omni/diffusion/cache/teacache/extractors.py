# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model-specific extractors for TeaCache.

This module provides a registry of extractor functions that know how to extract
modulated inputs from different transformer architectures. Adding support for
a new model requires only adding a new extractor function to the registry.

With Option B enhancement, extractors now return a CacheContext object containing
all model-specific information needed for generic caching, including preprocessing,
transformer execution, and postprocessing logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CacheContext:
    """
    Context object containing all model-specific information for caching.

    This allows the TeaCacheHook to remain completely generic - all model-specific
    logic is encapsulated in the extractor that returns this context.

    Attributes:
        modulated_input: Tensor used for cache decision (similarity comparison).
            Must be a torch.Tensor extracted from the first transformer block,
            typically after applying normalization and modulation.

        hidden_states: Current hidden states (will be modified by caching).
            Must be a torch.Tensor representing the main image/latent states
            after preprocessing but before transformer blocks.

        encoder_hidden_states: Optional encoder states (for dual-stream models).
            Set to None for single-stream models (e.g., Flux).
            For dual-stream models (e.g., Qwen), contains text encoder outputs.

        temb: Timestep embedding tensor.
            Must be a torch.Tensor containing the timestep conditioning.

        run_transformer_blocks: Callable that executes model-specific transformer blocks.
            Signature: () -> tuple[torch.Tensor, ...]

            Returns:
                tuple containing:
                - [0]: processed hidden_states (required)
                - [1]: processed encoder_hidden_states (optional, only for dual-stream)

            Example for single-stream:
                def run_blocks():
                    h = hidden_states
                    for block in module.transformer_blocks:
                        h = block(h, temb=temb)
                    return (h,)

            Example for dual-stream:
                def run_blocks():
                    h, e = hidden_states, encoder_hidden_states
                    for block in module.transformer_blocks:
                        e, h = block(h, e, temb=temb)
                    return (h, e)

        postprocess: Callable that does model-specific output postprocessing.
            Signature: (torch.Tensor) -> Union[torch.Tensor, Transformer2DModelOutput, tuple]

            Takes the processed hidden_states and applies final transformations
            (normalization, projection) to produce the model output.

            Example:
                def postprocess(h):
                    h = module.norm_out(h, temb)
                    output = module.proj_out(h)
                    return Transformer2DModelOutput(sample=output)

        extra_states: Optional dict for additional model-specific state.
            Use this for models that need to pass additional context beyond
            the standard fields.
    """

    modulated_input: torch.Tensor
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    run_transformer_blocks: Callable[[], tuple[torch.Tensor, ...]]
    postprocess: Callable[[torch.Tensor], Any]
    extra_states: dict[str, Any] | None = None

    def validate(self) -> None:
        """
        Validate that the CacheContext contains valid data.

        Raises:
            TypeError: If fields have wrong types
            ValueError: If tensors have invalid properties
            RuntimeError: If callables fail basic invocation tests

        This method should be called after creating a CacheContext to catch
        common developer errors early with clear error messages.
        """
        # Validate tensor fields
        if not isinstance(self.modulated_input, torch.Tensor):
            raise TypeError(f"modulated_input must be torch.Tensor, got {type(self.modulated_input)}")

        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(self.hidden_states)}")

        if self.encoder_hidden_states is not None and not isinstance(self.encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"encoder_hidden_states must be torch.Tensor or None, got {type(self.encoder_hidden_states)}"
            )

        if not isinstance(self.temb, torch.Tensor):
            raise TypeError(f"temb must be torch.Tensor, got {type(self.temb)}")

        # Validate callables
        if not callable(self.run_transformer_blocks):
            raise TypeError(f"run_transformer_blocks must be callable, got {type(self.run_transformer_blocks)}")

        if not callable(self.postprocess):
            raise TypeError(f"postprocess must be callable, got {type(self.postprocess)}")

        # Validate tensor shapes are compatible
        if self.modulated_input.shape[0] != self.hidden_states.shape[0]:
            raise ValueError(
                f"Batch size mismatch: modulated_input has batch size "
                f"{self.modulated_input.shape[0]}, but hidden_states has "
                f"{self.hidden_states.shape[0]}"
            )

        # Validate devices match
        if self.modulated_input.device != self.hidden_states.device:
            raise ValueError(
                f"Device mismatch: modulated_input on {self.modulated_input.device}, "
                f"hidden_states on {self.hidden_states.device}"
            )


def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor | float | int,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    additional_t_cond: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for QwenImageTransformer2DModel.

    This is the ONLY Qwen-specific code needed for TeaCache support.
    It encapsulates preprocessing, modulated input extraction, transformer execution,
    and postprocessing logic.

    Args:
        module: QwenImageTransformer2DModel instance
        hidden_states: Input hidden states tensor
        encoder_hidden_states: Text encoder outputs
        encoder_hidden_states_mask: Mask for text encoder
        timestep: Current diffusion timestep
        img_shapes: Image shapes for position embedding
        txt_seq_lens: Text sequence lengths
        guidance: Optional guidance scale for CFG
        additional_t_cond: Optional additional timestep conditioning
        attention_kwargs: Additional attention arguments
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # ============================================================================
    # PREPROCESSING (Qwen-specific)
    # ============================================================================
    hidden_states = module.img_in(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        module.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else module.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_modulated, _ = block.img_norm1(hidden_states, img_mod1)

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Qwen-specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Qwen transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
        return (h, e)

    # ============================================================================
    # DEFINE POSTPROCESSING (Qwen-specific)
    # ============================================================================
    return_dict = kwargs.get("return_dict", True)

    def postprocess(h):
        """Apply Qwen-specific output postprocessing."""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_zimage_context(
    module: nn.Module,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    patch_size: int = 2,
    f_patch_size: int = 1,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for ZImageTransformer2DModel.

    This is the ONLY Z-Image-specific code needed for TeaCache support.
    It encapsulates preprocessing, modulated input extraction, transformer execution,
    and postprocessing logic.

    Args:
        module: ZImageTransformer2DModel instance
        x: List of image tensors per batch item
        t: Timestep tensor
        cap_feats: List of caption feature tensors per batch item
        patch_size: Patch size for patchification (default: 2)
        f_patch_size: Frame patch size (default: 1)
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    from torch.nn.utils.rnn import pad_sequence

    if not hasattr(module, "layers") or len(module.layers) == 0:
        raise ValueError("Module must have main transformer layers")

    bsz = len(x)
    device = x[0].device

    # ============================================================================
    # PREPROCESSING (Z-Image specific)
    # ============================================================================
    # Scale timestep and create timestep embedding
    t_scaled = t * module.t_scale
    adaln_input = module.t_embedder(t_scaled)

    # Patchify and embed inputs
    (
        x_patches,
        cap_feats_processed,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = module.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # Process image patches through embedder and noise refiner
    x_item_seqlens = [len(_) for _ in x_patches]
    x_max_item_seqlen = max(x_item_seqlens)

    x_embedded = torch.cat(x_patches, dim=0)
    x_embedded = module.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_embedded)

    # Match adaln_input dtype to x_embedded
    adaln_input = adaln_input.type_as(x_embedded)

    # Apply pad token
    x_embedded[torch.cat(x_inner_pad_mask)] = module.x_pad_token
    x_list = list(x_embedded.split(x_item_seqlens, dim=0))

    # Compute rope embeddings for image patches
    x_cos, x_sin = module.rope_embedder(torch.cat(x_pos_ids, dim=0))
    x_cos = list(x_cos.split(x_item_seqlens, dim=0))
    x_sin = list(x_sin.split(x_item_seqlens, dim=0))

    # Pad sequences for batch processing
    x_batched = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    x_cos_batched = pad_sequence(x_cos, batch_first=True, padding_value=0.0)
    x_sin_batched = pad_sequence(x_sin, batch_first=True, padding_value=0.0)
    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    # Run noise refiner blocks
    for layer in module.noise_refiner:
        x_batched = layer(x_batched, x_attn_mask, x_cos_batched, x_sin_batched, adaln_input)

    # Process caption features through embedder and context refiner
    cap_item_seqlens = [len(_) for _ in cap_feats_processed]
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_embedded = torch.cat(cap_feats_processed, dim=0)
    cap_embedded = module.cap_embedder(cap_embedded)
    cap_embedded[torch.cat(cap_inner_pad_mask)] = module.cap_pad_token
    cap_list = list(cap_embedded.split(cap_item_seqlens, dim=0))

    # Compute rope embeddings for caption
    cap_cos, cap_sin = module.rope_embedder(torch.cat(cap_pos_ids, dim=0))
    cap_cos = list(cap_cos.split(cap_item_seqlens, dim=0))
    cap_sin = list(cap_sin.split(cap_item_seqlens, dim=0))

    # Pad sequences for batch processing
    cap_batched = pad_sequence(cap_list, batch_first=True, padding_value=0.0)
    cap_cos_batched = pad_sequence(cap_cos, batch_first=True, padding_value=0.0)
    cap_sin_batched = pad_sequence(cap_sin, batch_first=True, padding_value=0.0)
    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    # Run context refiner blocks
    for layer in module.context_refiner:
        cap_batched = layer(cap_batched, cap_attn_mask, cap_cos_batched, cap_sin_batched)

    # Create unified sequence (image + caption)
    unified_list = []
    unified_cos_list = []
    unified_sin_list = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified_list.append(torch.cat([x_batched[i][:x_len], cap_batched[i][:cap_len]]))
        unified_cos_list.append(torch.cat([x_cos_batched[i][:x_len], cap_cos_batched[i][:cap_len]]))
        unified_sin_list.append(torch.cat([x_sin_batched[i][:x_len], cap_sin_batched[i][:cap_len]]))

    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)
    unified_cos = pad_sequence(unified_cos_list, batch_first=True, padding_value=0.0)
    unified_sin = pad_sequence(unified_sin_list, batch_first=True, padding_value=0.0)
    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    # Use the first main transformer block's modulation
    # The main layers have modulation=True and process the unified sequence
    block = module.layers[0]
    # Get modulation parameters: scale_msa, gate_msa, scale_mlp, gate_mlp
    mod_params = block.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
    scale_msa = 1.0 + mod_params[0]
    # Extract modulated input: normalized hidden states scaled by modulation
    modulated_input = block.attention_norm1(unified) * scale_msa

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Z-Image specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Z-Image main transformer blocks."""
        h = unified
        for layer in module.layers:
            h = layer(h, unified_attn_mask, unified_cos, unified_sin, adaln_input)
        return (h,)

    # ============================================================================
    # DEFINE POSTPROCESSING (Z-Image specific)
    # ============================================================================
    def postprocess(h):
        """Apply Z-Image specific output postprocessing."""
        h = module.all_final_layer[f"{patch_size}-{f_patch_size}"](h, adaln_input)
        h = list(h.unbind(dim=0))
        output = module.unpatchify(h, x_size, patch_size, f_patch_size)
        return output, {}

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=unified,
        encoder_hidden_states=None,  # Z-Image uses unified sequence, no separate encoder states
        temb=adaln_input,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
        extra_states={
            "unified_attn_mask": unified_attn_mask,
            "unified_cos": unified_cos,
            "unified_sin": unified_sin,
            "x_size": x_size,
            "x_item_seqlens": x_item_seqlens,
            "patch_size": patch_size,
            "f_patch_size": f_patch_size,
        },
    )


# Registry for model-specific extractors
# Key: Transformer class name
# Value: extractor function with signature (module, *args, **kwargs) -> CacheContext
#
# Note: Use the transformer class name as specified in pipelines as TeaCache hooks operate
# on the transformer module and multiple pipelines can share the same transformer.
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "ZImageTransformer2DModel": extract_zimage_context,
    # Future models:
    # "FluxTransformer2DModel": extract_flux_context,
    # "CogVideoXTransformer3DModel": extract_cogvideox_context,
}


def register_extractor(transformer_cls_name: str, extractor_fn: Callable) -> None:
    """
    Register a new extractor function for a model type.

    This allows extending TeaCache support to new models without modifying
    the core TeaCache code.

    Args:
        transformer_cls_name: Transformer model type identifier (class name or type string)
        extractor_fn: Function with signature (module, *args, **kwargs) -> CacheContext

    Example:
        >>> def extract_flux_context(module, hidden_states, timestep, guidance=None, **kwargs):
        ...     # Preprocessing
        ...     temb = module.time_text_embed(timestep, guidance)
        ...     # Extract modulated input
        ...     modulated = module.transformer_blocks[0].norm1(hidden_states, emb=temb)
        ...     # Define execution
        ...     def run_blocks():
        ...         h = hidden_states
        ...         for block in module.transformer_blocks:
        ...             h = block(h, temb=temb)
        ...         return (h,)
        ...     # Define postprocessing
        ...     def postprocess(h):
        ...         return module.proj_out(module.norm_out(h, temb))
        ...     # Return context
        ...     return CacheContext(modulated, hidden_states, None, temb, run_blocks, postprocess)
        >>> register_extractor("FluxTransformer2DModel", extract_flux_context)
    """
    EXTRACTOR_REGISTRY[transformer_cls_name] = extractor_fn


def get_extractor(transformer_cls_name: str) -> Callable:
    """
    Get extractor function for given transformer class.

    This function looks up the extractor based on the exact transformer_cls_name string,
    which should match the transformer type in the pipeline (i.e., pipeline.transformer.__class__.__name__).

    Args:
        transformer_cls_name: Transformer class name (e.g., "QwenImageTransformer2DModel")
                                Must exactly match a key in EXTRACTOR_REGISTRY.

    Returns:
        Extractor function with signature (module, *args, **kwargs) -> CacheContext

    Raises:
        ValueError: If model type not found in registry

    Example:
        >>> # Get extractor for QwenImageTransformer2DModel
        >>> extractor = get_extractor("QwenImageTransformer2DModel")
        >>> ctx = extractor(transformer, hidden_states, encoder_hidden_states, timestep, ...)
    """
    # Direct lookup - no substring matching
    if transformer_cls_name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[transformer_cls_name]

    # No match found
    available_types = list(EXTRACTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown model type: '{transformer_cls_name}'. "
        f"Available types: {available_types}\n"
        f"To add support for a new model, use register_extractor() or add to EXTRACTOR_REGISTRY."
    )
