# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-level AdaLN projections for MiniMax-H3.

Each block's ``adaln_proj`` reads a ~520 MB weight per denoising step for a few timestep rows. A request's
timesteps follow from its sigma schedule, so one batched call per block reads the weight once per request. The
time embeddings and tables are built inside the first planned forward (where offloaded weights are staged or
materialized) and a table is kept only if its row for that step equals the per-step result bit for bit. Lookups use
host-side step keys, so they read nothing back from the device; unplanned forwards use the per-step call.

The tables hold steps x rows x 18 * hidden BF16 values per block: ~1.4 GB per request for MiniMax-H3 at 49 steps.
Tables beyond ``MAX_TABLE_BYTES`` are not kept; ``minimax_h3_adaln_plan: false`` turns the plan off.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Hashable, Sequence

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.exact_projection_cache import ExactProjectionCache

logger = init_logger(__name__)

MAX_TABLE_BYTES = 2 * 1024**3


def projection_plan_safe(proj: nn.Module) -> bool:
    """Whether one batched call over planned rows computes what per-step calls compute."""
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    linear = proj.linear
    return (
        not isinstance(linear, BaseLayerWithLoRA)
        and type(getattr(linear, "quant_method", None)) is UnquantizedLinearMethod
        and not ExactProjectionCache._has_forward_hooks(linear)
        and get_tensor_model_parallel_world_size() == 1
    )


def adaln_step_key(branch, times: tuple[float, float], cond: tuple[float, float]) -> Hashable:
    return (id(branch), *times, *cond)


class MiniMaxH3AdalnPlan:
    """One request's AdaLN projections, looked up by a host-side step key."""

    def __init__(
        self,
        time_embedder: Callable[[torch.Tensor], torch.Tensor],
        steps: Sequence[tuple[Hashable, torch.Tensor]],
        *,
        safe: Callable[[nn.Module], bool] = projection_plan_safe,
    ) -> None:
        """``steps``: (step key, that forward's unique timesteps) for every planned forward."""
        self._safe = safe
        self._key: Hashable | None = None
        self._entry: tuple[int, int] | None = None
        self._bound: weakref.ref[torch.Tensor] | None = None
        self._index: dict[Hashable, tuple[int, int]] = {}  # step key -> (row count M, index in that M's table)
        self._tables: dict[int, dict[int, torch.Tensor]] = {}  # id(proj) -> M -> [steps, M, out]
        self._pending: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}  # table, matched flag
        self._built: set[tuple[int, int]] = set()
        self._bytes = 0
        self._time_embedder = time_embedder
        self._unique: dict[int, list[torch.Tensor]] = {}
        for key, unique in steps:
            m = unique.numel()
            # A single-row batched call takes a different GEMM path than the per-step GEMV.
            if key in self._index or m < 2:
                continue
            rows = self._unique.setdefault(m, [])
            self._index[key] = (m, len(rows))
            rows.append(unique)
        self.embed_dtype: torch.dtype | None = None
        self._inputs: dict[int, torch.Tensor] | None = None

    def _embed(self) -> None:
        # Run inside the model's forward, where a model-level CPU offload has moved the time embedder in.
        embed_dtypes = set()
        inputs = {}
        for m, uniques in self._unique.items():
            rows = []
            for unique in uniques:
                t_emb = self._time_embedder(unique)
                embed_dtypes.add(t_emb.dtype)
                rows.append(nn.functional.silu(t_emb).to(torch.bfloat16))
            inputs[m] = torch.stack(rows)
        self.embed_dtype = embed_dtypes.pop() if len(embed_dtypes) == 1 else None
        self._inputs = inputs

    def select(self, key: Hashable | None) -> None:
        """Name the next forward's step, or None for an unplanned forward."""
        self._key = key

    def bind(self, t_emb: torch.Tensor) -> None:
        """Bind lookups to this forward (consuming the selected key) and check pending tables with one readback."""
        if self._inputs is None:
            self._embed()
        if self._pending:
            pending, self._pending = self._pending, {}
            same = torch.stack([flag for _, flag in pending.values()]).tolist()
            for ((proj_id, m), (table, _)), keep in zip(pending.items(), same, strict=True):
                if keep:
                    self._tables.setdefault(proj_id, {})[m] = table
        entry = self._index.get(self._key) if self._key is not None else None
        self._key = None
        # The control branch casts the embedding before AdaLN; a different dtype or row count is unplanned.
        valid = entry is not None and t_emb.dtype == self.embed_dtype and t_emb.shape[0] == entry[0]
        self._entry = entry if valid else None
        self._bound = weakref.ref(t_emb)

    def _entry_for(self, t_emb: torch.Tensor) -> tuple[int, int] | None:
        # Only the embedding this forward bound: a forward that skipped bind() must not reuse an old step's row.
        return self._entry if self._bound is not None and self._bound() is t_emb else None

    def lookup(self, proj: nn.Module, t_emb: torch.Tensor) -> torch.Tensor | None:
        entry = self._entry_for(t_emb)
        table = self._tables.get(id(proj), {}).get(entry[0]) if entry is not None else None
        return None if table is None else table[entry[1]]

    def wants(self, proj: nn.Module, t_emb: torch.Tensor) -> bool:
        entry = self._entry_for(t_emb)
        return entry is not None and (id(proj), entry[0]) not in self._built and self._safe(proj)

    def rows(self) -> torch.Tensor:
        assert self._entry is not None
        return self._inputs[self._entry[0]]

    def record(self, proj: nn.Module, table: torch.Tensor, per_step: torch.Tensor) -> None:
        assert self._entry is not None
        m, index = self._entry
        self._built.add((id(proj), m))
        nbytes = table.numel() * table.element_size()
        if self._bytes + nbytes > MAX_TABLE_BYTES:
            logger.info_once("MiniMax H3 AdaLN plan reached %d bytes; further projections run per step", self._bytes)
            return
        self._bytes += nbytes
        self._pending[(id(proj), m)] = (table, (table[index] == per_step).all())


def build_adaln_plan(
    model: nn.Module,
    branch,
    sigmas_video: Sequence[float],
    sigmas_audio: Sequence[float],
    cond_anchors: tuple[float, float],
) -> MiniMaxH3AdalnPlan | None:
    """Install the request's plan for every step of a plain schedule: (sigma_video[i], sigma_audio[i])."""
    from .minimax_h3_blocks import MiniMaxH3AdalnProj

    cache = getattr(model, "adaln_cache", None)
    if cache is None or not cache.plan_enabled:
        return None
    if not any(isinstance(m, MiniMaxH3AdalnProj) and projection_plan_safe(m) for m in model.modules()):
        return None
    steps = []
    timesteps = torch.empty(branch.seq_len, dtype=torch.float32, device=branch.device)
    with torch.inference_mode():
        for sigma_video, sigma_audio in zip(sigmas_video[:-1], sigmas_audio[:-1], strict=True):
            # Mirrors MiniMaxH3ModelBatch, which selects the same key per forward.
            times = (1.0 - sigma_video, 1.0 - sigma_audio)
            cond = tuple(max(t, anchor) for t, anchor in zip(times, cond_anchors, strict=True))
            branch.fill_timesteps(
                timesteps,
                t_video=times[0],
                t_audio=times[1],
                imgvid_cond_timestep=cond[0],
                audio_ref_cond_timestep=cond[1],
            )
            steps.append((adaln_step_key(branch, times, cond), torch.unique(timesteps, sorted=True)))
        cache.plan = MiniMaxH3AdalnPlan(model.time_embedder, steps)
    return cache.plan


__all__ = ["MiniMaxH3AdalnPlan", "adaln_step_key", "build_adaln_plan", "projection_plan_safe"]
