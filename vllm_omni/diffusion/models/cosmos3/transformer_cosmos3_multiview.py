# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 transformer variant with Multiview-AV FlexAttention."""

from __future__ import annotations

from typing import Any

import torch

from .multiview_flex_attention import (
    MultiviewAttentionContext,
    MultiviewLayout,
    padded_multiview_flex_attention,
)
from .transformer_cosmos3 import (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    Cosmos3CrossAttention,
    Cosmos3VFMTransformer,
    _tf_config_get,
)


class Cosmos3MultiviewCrossAttention(Cosmos3CrossAttention):
    """Use sparse rectangular attention when a multiview context is present."""

    def _forward_multiview(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        multiview_layout: Any,
    ) -> torch.Tensor:
        if not isinstance(multiview_layout, MultiviewAttentionContext):
            raise TypeError(
                "Cosmos3 multiview cross-attention expected MultiviewAttentionContext, "
                f"got {type(multiview_layout).__name__}."
            )
        output = padded_multiview_flex_attention(q, k, v, k_und, v_und, multiview_layout)
        return output.reshape(q.shape[0], q.shape[1], -1)


class Cosmos3MultiviewVFMTransformer(Cosmos3VFMTransformer):
    """Cosmos3 Nano weights with request-local multiview block-mask caching."""

    _cross_attention_cls = Cosmos3MultiviewCrossAttention

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        Cosmos3VFMTransformer._validate_supported_config(model_config)
        backbone_type = _tf_config_get(model_config, "backbone_type", None)
        if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
            raise ValueError(
                "Cosmos3MultiviewVFMTransformer requires transformer/config.json "
                f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
            )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._multiview_mask_cache: dict[tuple[Any, ...], Any] = {}
        # Padded q/k/v packing buffers, keyed by shape/dtype/device. Held on the
        # transformer rather than the per-forward context so the ~2.5 GiB of
        # packed tensors are zeroed once per request instead of once per layer.
        self._multiview_buffer_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def reset_cache(self) -> None:
        super().reset_cache()
        self._multiview_mask_cache.clear()
        self._multiview_buffer_cache.clear()

    def forward(
        self,
        *args,
        multiview_layout: MultiviewLayout | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if multiview_layout is None:
            return super().forward(*args, **kwargs)
        control_latents = kwargs.get("control_latents")
        if isinstance(control_latents, torch.Tensor):
            control_count = 1
        elif control_latents is None:
            control_count = 0
        else:
            control_count = len(control_latents)
        if control_count != 1:
            raise ValueError(f"Cosmos3 multiview v1 requires exactly one packed WSM control item, got {control_count}.")
        if kwargs.get("action_latents") is not None or kwargs.get("sound_latents") is not None:
            raise ValueError("Cosmos3 multiview v1 cannot be combined with action or sound streams.")
        context = MultiviewAttentionContext(
            multiview_layout,
            self._multiview_mask_cache,
            self._multiview_buffer_cache,
        )
        return super().forward(*args, multiview_layout=context, **kwargs)
