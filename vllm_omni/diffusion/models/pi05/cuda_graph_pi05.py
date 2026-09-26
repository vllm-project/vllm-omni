# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph path for π0.5 ``sample_actions`` (batch size 1).

``sample_actions`` has three regions that are captured and replayed separately:

1. the prefix embedding (``embed_prefix``);
2. the prefix forward pass that builds the per-layer KV cache
   (``paligemma_with_expert.forward`` with prefix inputs);
3. one denoising step (``denoise_step``), replayed once per step.

Each region method below has the signature of the eager call it replaces and
falls back to that call whenever it has no graph to replay. The eager path is
the baseline: ``Pi05ForActionPrediction.cuda_graphs is None``.

No region is captured yet, so every region currently runs eagerly.
``tests/diffusion/models/pi05/test_pi05_cuda_graph_parity.py`` pins this path
bit-exact against the eager baseline on the real checkpoint.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction


class Pi05CUDAGraphs:
    """Per-region CUDA Graph capture/replay for one ``Pi05ForActionPrediction``."""

    def __init__(self, model: Pi05ForActionPrediction):
        # The model owns this object through ``model.cuda_graphs``; a weak
        # reference back keeps that from forming a cycle, so dropping the model
        # frees its GPU memory immediately rather than at the next GC pass.
        self._model = weakref.proxy(model)

    # ── Region 1: prefix embedding ───────────────────────────────────
    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Eager fallback: no graph is captured for this region yet.
        return self._model.embed_prefix(images, image_masks, lang_tokens, lang_masks)

    # ── Region 2: prefix forward (prefix KV construction) ────────────
    def prefix_forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
        adarms_cond: torch.Tensor | None = None,
    ):
        # Eager fallback: no graph is captured for this region yet.
        return self._model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )

    # ── Region 3: one denoising step ─────────────────────────────────
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Eager fallback: no graph is captured for this region yet.
        return self._model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )
