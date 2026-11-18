# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""cuDNN's public BF16 block-sparse API, with caller-owned tile64/128 routing."""

from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import version

import torch
from packaging.version import Version

from vllm_omni.platforms import current_omni_platform

from .block_sparse import block_map_to_indices, validate_softmax_scale, validate_sparse_inputs


@lru_cache(maxsize=32)
def require_cudnn_sparse(device: torch.device) -> Callable[..., dict[str, torch.Tensor]]:
    """Validate the optional provider before model loading or kernel launch."""
    if device.type != "cuda":
        raise ValueError("cuDNN block-sparse attention requires CUDA")
    capability = current_omni_platform.get_device_capability(device.index)
    if capability is None or capability[0] not in (9, 10, 11, 12):
        raise ValueError(f"cuDNN block-sparse attention requires SM90 through SM120 families; got {capability}")
    try:
        if Version(version("nvidia-cudnn-frontend")) < Version("1.31.0"):
            raise ImportError("cuDNN BSA requires nvidia-cudnn-frontend>=1.31.0")
        from cudnn.torch import block_sparse_attention_forward
    except ImportError as exc:
        raise ImportError("cuDNN BSA requires nvidia-cudnn-frontend>=1.31.0; install vllm-omni[cudnn-bsa]") from exc
    return block_sparse_attention_forward


@torch.library.custom_op("vllm_omni::cudnn_block_sparse_attn", mutates_args=())
def _cudnn_block_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    sizes: torch.Tensor | None,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    kernel = require_cudnn_sparse(query.device)
    return kernel(
        query,
        key,
        value,
        indices,
        q2k_block_nums=counts,
        block_sizes=sizes,
        sparse_block_size=block_size,
        allow_empty_block_nums=True,
        softmax_scale=scale,
        layout="bshd",
    )["o_tensor"].contiguous()


@_cudnn_block_sparse_attn.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    sizes: torch.Tensor | None,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def cudnn_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    block_size: int,
) -> torch.Tensor:
    """Convert the selected mask without changing top-k or its logical tiles."""
    if block_size not in (64, 128):
        raise ValueError("cuDNN sparse block_size must be 64 or 128")
    validate_softmax_scale(softmax_scale)
    if block_sizes is not None:
        if block_sizes.ndim != 1 or block_sizes.shape[0] != block_map.shape[-1]:
            raise ValueError("cuDNN VSA requires one valid-token count per KV block")
        # cuDNN forbids zero-sized blocks in the active index prefix.
        block_map = block_map & (block_sizes[None, None, None, :] > 0)
    indices, counts = block_map_to_indices(block_map)
    validate_sparse_inputs(query, key, value, indices, counts, block_sizes, block_size)
    if block_map.shape[-1] != (key.shape[1] + block_size - 1) // block_size:
        raise ValueError("block_map key blocks must match ceil(Sk/block_size)")
    return _cudnn_block_sparse_attn(query, key, value, indices, counts, block_sizes, softmax_scale, block_size)
