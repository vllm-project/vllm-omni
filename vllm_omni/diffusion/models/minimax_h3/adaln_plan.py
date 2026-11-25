# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-level AdaLN projections for MiniMax-H3.

Each block's ``adaln_proj`` reads a ~520 MB weight per denoising step for a few timestep rows. A request's
timesteps follow from its sigma schedule, so one batched call per block reads the weight once per request. The
table is built inside the first planned forward (where an offloaded weight is staged or materialized) and kept
only if its row for that step equals the per-step result bit for bit. Lookups use host-side step keys, so they
read nothing back from the device; unplanned forwards use the per-step call.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

import torch
from torch import nn

_BF16_DTYPE = torch.bfloat16


def projection_plan_safe(proj: nn.Module) -> bool:
    """Whether one batched call over planned rows computes what per-step calls compute."""
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    linear = proj.linear
    return (
        not isinstance(linear, BaseLayerWithLoRA)
        and type(getattr(linear, "quant_method", None)) is UnquantizedLinearMethod
        and not linear._forward_hooks
        and not linear._forward_pre_hooks
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
        self._index: dict[Hashable, tuple[int, int]] = {}  # step key -> (row count M, index in that M's table)
        self._tables: dict[int, dict[int, torch.Tensor]] = {}  # id(proj) -> M -> [steps, M, out]
        self._pending: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}  # table, matched flag
        self._built: set[tuple[int, int]] = set()
        groups: dict[int, list[torch.Tensor]] = {}
        embed_dtypes = set()
        for key, unique in steps:
            m = unique.numel()
            # A single-row batched call takes a different GEMM path than the per-step GEMV.
            if key in self._index or m < 2:
                continue
            rows = groups.setdefault(m, [])
            self._index[key] = (m, len(rows))
            t_emb = time_embedder(unique)
            embed_dtypes.add(t_emb.dtype)
            rows.append(nn.functional.silu(t_emb).to(_BF16_DTYPE))
        self.embed_dtype = embed_dtypes.pop() if len(embed_dtypes) == 1 else None
        self._inputs = {m: torch.stack(rows) for m, rows in groups.items()}

    def select(self, key: Hashable | None) -> None:
        """Name the next forward's step, or None for an unplanned forward."""
        self._key = key

    def bind(self, t_emb: torch.Tensor) -> None:
        """Bind lookups to this forward (consuming the selected key) and check pending tables with one readback."""
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

    def lookup(self, proj: nn.Module) -> torch.Tensor | None:
        entry = self._entry
        table = self._tables.get(id(proj), {}).get(entry[0]) if entry is not None else None
        return None if table is None else table[entry[1]]

    def wants(self, proj: nn.Module) -> bool:
        entry = self._entry
        return entry is not None and (id(proj), entry[0]) not in self._built and self._safe(proj)

    def rows(self) -> torch.Tensor:
        assert self._entry is not None
        return self._inputs[self._entry[0]]

    def record(self, proj: nn.Module, table: torch.Tensor, per_step: torch.Tensor) -> None:
        assert self._entry is not None
        m, index = self._entry
        self._built.add((id(proj), m))
        self._pending[(id(proj), m)] = (table, (table[index] == per_step).all())


def build_adaln_plan(
    model: nn.Module,
    branch,
    sigmas_video: Sequence[float],
    sigmas_audio: Sequence[float],
    cond_anchors: tuple[float, float],
) -> MiniMaxH3AdalnPlan | None:
    """Install the request's plan for every step of a plain schedule: (sigma_video[i], sigma_audio[i])."""
    cache = getattr(model, "adaln_cache", None)
    if cache is None or not cache.plan_enabled:
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
