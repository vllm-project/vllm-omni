# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader-side adapters for proven direct-checkpoint mmap layouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch
from torch import nn


@dataclass(frozen=True)
class DirectMmapTensorPolicy:
    """Loader-owned proof for one tensor's direct mmap behavior."""

    allow_custom_loader: bool = False
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None


class DirectMmapAdapter(Protocol):
    def policy_for(
        self,
        runtime_name: str,
        target: torch.Tensor,
    ) -> DirectMmapTensorPolicy | None: ...


class _MiniMaxH3DirectMmapAdapter:
    """TP1 runtime-layout contract already enforced by MiniMax-H3's loader."""

    def __init__(self, pipeline: nn.Module) -> None:
        self.pipeline = pipeline

    def policy_for(
        self,
        runtime_name: str,
        target: torch.Tensor,
    ) -> DirectMmapTensorPolicy:
        del target
        transform = None
        suffix = ".qkv_proj.weight"
        if runtime_name.endswith(suffix):
            from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
                MiniMaxH3Attention,
                _reorder_grouped_qkv_to_qkv,
            )

            attention_path = runtime_name[: -len(suffix)]
            attention = self.pipeline.get_submodule(attention_path)
            if not isinstance(attention, MiniMaxH3Attention):
                raise ValueError(
                    f"MiniMax-H3 direct-mmap adapter expected attention at {attention_path!r}, "
                    f"got {type(attention).__name__}"
                )
            transform = partial(
                _reorder_grouped_qkv_to_qkv,
                num_query_groups=attention.total_num_heads,
                heads_per_group=1,
                head_dim=attention.head_dim,
            )
        return DirectMmapTensorPolicy(
            allow_custom_loader=True,
            transform=transform,
        )


class _Cosmos3DirectMmapAdapter:
    """Preserve Cosmos3's existing TP1 direct-mmap load contract."""

    def policy_for(
        self,
        runtime_name: str,
        target: torch.Tensor,
    ) -> DirectMmapTensorPolicy:
        del runtime_name, target
        # Cosmos3 has no packed module mapping or TP1 semantic transform after
        # its checkpoint-key remap. Its custom vLLM loaders reduce to exact
        # copies at TP1; topology and metadata are checked by plan preflight.
        return DirectMmapTensorPolicy(allow_custom_loader=True)


def get_direct_mmap_adapter(pipeline: nn.Module) -> DirectMmapAdapter | None:
    """Return a loader-owned adapter without requiring a pipeline flag."""
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    if any(isinstance(module, MiniMaxH3DiTModel) for module in pipeline.modules()):
        return _MiniMaxH3DirectMmapAdapter(pipeline)

    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        Cosmos3VFMTransformer,
    )

    if any(isinstance(module, Cosmos3VFMTransformer) for module in pipeline.modules()):
        return _Cosmos3DirectMmapAdapter()
    return None


__all__ = [
    "DirectMmapAdapter",
    "DirectMmapTensorPolicy",
    "get_direct_mmap_adapter",
]
