# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NAVER Cloud Corp. vision-decoder-api

"""
VisionTransformer USP Wrapper for xDiT Integration.

This module provides Unified Sequence Parallelism (USP) support for the
HyperCLOVAX VisionTransformer used in vision token to image generation.

USP enables multi-GPU acceleration by:
- Splitting input sequences across GPUs (Ulysses parallelism)
- Using ring attention patterns for long sequences
- Efficiently gathering outputs after processing
"""

import functools

import torch
import torch.nn as nn
from einops import rearrange

from .layers import timestep_embedding

# xDiT imports
try:
    from xfuser.core.distributed import (
        get_sequence_parallel_rank,
        get_sequence_parallel_world_size,
        get_sp_group,
    )
    from xfuser.model_executor.layers.usp import USP

    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False


def split_sequence(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Split tensor along sequence dimension for parallel processing.

    Args:
        tensor: Input tensor to split
        dim: Dimension to split along (default: 1 for sequence dim)

    Returns:
        Local chunk of the tensor for this rank
    """
    if not XDIT_AVAILABLE or get_sequence_parallel_world_size() <= 1:
        return tensor

    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()

    chunks = torch.chunk(tensor, world_size, dim=dim)
    return chunks[rank].contiguous()


def gather_sequence(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Gather tensor from all ranks along sequence dimension.

    Args:
        tensor: Local tensor chunk
        dim: Dimension to gather along (default: 1 for sequence dim)

    Returns:
        Full tensor gathered from all ranks
    """
    if not XDIT_AVAILABLE or get_sequence_parallel_world_size() <= 1:
        return tensor

    return get_sp_group().all_gather(tensor.contiguous(), dim=dim)


def split_rope_embedding(pe: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Split RoPE position embedding for sequence parallelism.

    The VisionTransformer uses 3D position encoding with axes_dim [8, 36, 36].
    The PE tensor has shape (B, 1, L, head_dim//2, 2, 2) after EmbedAND.

    Args:
        pe: Position embedding tensor
        seq_len: Original sequence length

    Returns:
        Local chunk of position embeddings
    """
    if not XDIT_AVAILABLE or get_sequence_parallel_world_size() <= 1:
        return pe

    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()

    # PE shape: (B, 1, L, D, 2, 2) where L is sequence length
    # Split along dim 2 (sequence dimension)
    seq_dim = 2
    chunks = torch.chunk(pe, world_size, dim=seq_dim)
    return chunks[rank].contiguous()


def apply_rope_usp(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding with USP support.

    Args:
        xq: Query tensor (B, H, L, D)
        xk: Key tensor (B, H, L, D)
        freqs_cis: RoPE frequencies (B, 1, L, D//2, 2, 2)

    Returns:
        Tuple of rotated query and key tensors
    """
    # Reshape for RoPE application
    # xq: (B, H, L, D) -> (B, H, L, D//2, 1, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    # freqs_cis: (B, 1, L, D//2, 2, 2) - contains cos and sin
    # Apply rotation: x_out = x * cos + rotate(x) * sin
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    return (
        xq_out.reshape(*xq.shape).type_as(xq),
        xk_out.reshape(*xk.shape).type_as(xk),
    )


def parallelize_transformer(transformer: nn.Module) -> nn.Module:
    """
    Parallelize VisionTransformer for sequence parallelism.

    This function wraps the transformer's forward method to:
    1. Split input sequences across GPUs
    2. Replace attention with USP attention
    3. Gather outputs after processing

    Args:
        transformer: HyperCLOVAXVisionTransformer2DModel instance

    Returns:
        Modified transformer with USP support
    """
    if not XDIT_AVAILABLE:
        return transformer

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def usp_forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        USP-enabled forward pass.

        Args:
            img: Input tensor (B, L, in_channels + context_in_dim)
            img_ids: Position IDs tensor (B, L, 3)
            timesteps: Sigma/timestep tensor (B,)
            y: Vision pooler output tensor (B, vec_in_dim)

        Returns:
            Output tensor (B, L, out_channels)
        """
        sp_world_size = get_sequence_parallel_world_size()

        if sp_world_size <= 1:
            # Single GPU mode
            return original_forward(img, img_ids, timesteps, y)

        # Split input sequences across GPUs
        img_local = split_sequence(img, dim=1)
        img_ids_local = split_sequence(img_ids, dim=1)

        # Run transformer with local sequences
        output_local = _usp_transformer_forward(self, img_local, img_ids_local, timesteps, y)

        # Gather output from all ranks
        output = gather_sequence(output_local, dim=1)

        return output

    # Bind the new forward method
    usp_forward = usp_forward.__get__(transformer)
    transformer.forward = usp_forward

    # Parallelize attention in single blocks
    _parallelize_attention_blocks(transformer)

    return transformer


def _usp_transformer_forward(
    transformer: nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    timesteps: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Internal forward pass with sequence-parallel attention.

    This function reimplements the transformer forward to use USP attention
    instead of standard attention.
    """
    if img.ndim != 3:
        raise ValueError("Input img tensor must have 3 dimensions.")

    # Project input
    img = transformer.img_in(img)

    # Time and vector embedding (no splitting needed - these are per-sample)
    vec = transformer.time_in(
        timestep_embedding(timesteps, 256).to(dtype=transformer.time_in.in_layer.weight.dtype, device=img.device)
    )
    vec = vec + transformer.vector_in(y)

    # Position embedding - compute for local sequence
    pe = transformer.pe_embedder(img_ids)

    # Single stream blocks with USP attention
    for block in transformer.single_blocks:
        img = _usp_single_block_forward(block, img, vec, pe)

    # Final projection
    img = transformer.final_layer(img, vec)

    return img


def _usp_single_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    vec: torch.Tensor,
    pe: torch.Tensor,
) -> torch.Tensor:
    """
    Single block forward with USP attention.

    This replaces the standard attention with USP attention that
    handles cross-GPU communication internally.
    """
    mod, _ = block.modulation(vec)
    x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(
        block.linear1(x_mod),
        [3 * block.hidden_size, block.mlp_hidden_dim],
        dim=-1,
    )

    q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
    q, k = block.norm(q, k, v)

    # USP attention
    if XDIT_AVAILABLE and get_sequence_parallel_world_size() > 1:
        # Apply RoPE to local Q, K
        q, k = apply_rope_usp(q, k, pe)

        # Use xfuser's USP for efficient parallel attention
        # USP handles cross-GPU communication internally
        attn = USP(q, k, v, dropout_p=0.0, is_causal=False)

        attn = rearrange(attn, "B H L D -> B L (H D)")
    else:
        # Standard attention with RoPE
        from .layers import attention

        attn = attention(q, k, v, pe=pe)

    output = block.linear2(torch.cat((attn, block.mlp_act(mlp)), 2))
    return x + mod.gate * output


def _parallelize_attention_blocks(transformer: nn.Module) -> None:
    """
    Replace attention in all single blocks with USP-enabled attention.

    This modifies the blocks in-place to use USP attention.
    """
    if not hasattr(transformer, "single_blocks"):
        return

    for i, block in enumerate(transformer.single_blocks):
        # Store original parameters
        block._usp_enabled = True
        block._original_forward = block.forward

        # Create new forward that uses USP
        def make_usp_block_forward(blk):
            @functools.wraps(blk.__class__.forward)
            def usp_block_forward(self, x, vec, pe):
                return _usp_single_block_forward(self, x, vec, pe)

            return usp_block_forward

        block.forward = make_usp_block_forward(block).__get__(block)
