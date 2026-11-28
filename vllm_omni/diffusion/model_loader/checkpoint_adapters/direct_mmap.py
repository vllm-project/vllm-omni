# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Loader-side adapters for proven direct-checkpoint mmap layouts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import regex as re
import torch
from torch import nn


@dataclass(frozen=True)
class DirectMmapTensorPolicy:
    """Loader-owned proof for one tensor's direct mmap behavior.

    ``transform`` maps one checkpoint tensor to one runtime tensor.
    ``fused_sources`` instead declares that the runtime tensor is fused from
    several checkpoint slices named by these checkpoint sub-suffixes, in the
    order ``fuse`` consumes them; the two are mutually exclusive.
    """

    allow_custom_loader: bool = False
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    fused_sources: tuple[str, ...] | None = None
    fuse: Callable[[Sequence[torch.Tensor]], torch.Tensor] | None = None


class DirectMmapAdapter(Protocol):
    def policy_for(
        self,
        runtime_name: str,
        target: torch.Tensor,
    ) -> DirectMmapTensorPolicy | None: ...


class _MiniMaxH3DirectMmapAdapter:
    """TP1 runtime-layout contract already enforced by MiniMax-H3's loader."""

    # The text encoder builds ``qkv_proj``/``gate_up_proj`` by concatenating
    # checkpoint slices along dim 0, so those runtime tensors are fused from
    # several entries instead of being an mmap view of one entry.
    _ENCODER_QKV = re.compile(r"^text_encoder\.text_model\.layers\.\d+\.self_attn\.qkv_proj\.weight$")
    _ENCODER_GATE_UP = re.compile(r"^text_encoder\.text_model\.layers\.\d+\.mlp\.gate_up_proj\.weight$")

    def __init__(self, pipeline: nn.Module) -> None:
        self.pipeline = pipeline

    def policy_for(
        self,
        runtime_name: str,
        target: torch.Tensor,
    ) -> DirectMmapTensorPolicy:
        del target
        if self._ENCODER_QKV.match(runtime_name):
            return DirectMmapTensorPolicy(
                allow_custom_loader=True,
                fused_sources=("q_proj", "k_proj", "v_proj"),
                fuse=_fuse_rows,
            )
        if self._ENCODER_GATE_UP.match(runtime_name):
            return DirectMmapTensorPolicy(
                allow_custom_loader=True,
                fused_sources=("gate_proj", "up_proj"),
                fuse=_fuse_rows,
            )
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


def _fuse_rows(slices: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate checkpoint slices in projection order."""
    return torch.cat(tuple(slices), dim=0)


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


class _Magi2DirectMmapAdapter:
    """Apply MAGI-2's rank-local MoE-head slices during bounded staging."""

    def policy_for(
        self,
        runtime_name: str,
        target: torch.Tensor,
    ) -> DirectMmapTensorPolicy:
        del runtime_name
        transform = getattr(target, "mmap_weight_transform", None)
        return DirectMmapTensorPolicy(
            transform=transform if callable(transform) else None,
        )


def get_direct_mmap_adapter(pipeline: nn.Module) -> DirectMmapAdapter | None:
    """Return a loader-owned adapter without requiring a pipeline flag."""
    from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
        Magi2PreviewTransformer,
    )

    if any(isinstance(module, Magi2PreviewTransformer) for module in pipeline.modules()):
        return _Magi2DirectMmapAdapter()

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
