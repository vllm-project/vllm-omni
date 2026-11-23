# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-level AdaLN projections for MiniMax-H3.

Each block's ``adaln_proj`` applies a ~520 MB weight to a handful of timestep
rows, so computing it every denoising step reads the whole weight once per
step: from HBM when it is resident, over PCIe when it lives on the host
(``minimax_h3_adaln_offload``) or is streamed by layerwise offload. A request's
per-step timestep sets follow from its sigma schedule, so one call per block
over all steps' rows reads each weight once per request.

A projection's table is built inside the first planned forward that reaches
it, where its weight is available exactly as the per-step call sees it
(resident, staged from the host, or materialized by an offload hook). That
forward returns its own per-step result; the table row for its step is compared
with that result, and at the next forward one readback keeps the tables that
match bit for bit and drops the rest.

Results are keyed by host-side scalars the sampler already has (branch and the
step's video/audio/condition timesteps), so a lookup reads nothing back from
the device. A forward that was not planned (another solver's intermediate
point, a latent edit, a packed multi-request batch), or whose embedding was
transformed before AdaLN, falls back to the per-step call.
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
    """Host-side identity of one forward's timesteps: the branch layout and the scalars that fill them."""
    return (id(branch), *times, *cond)


class MiniMaxH3AdalnPlan:
    """One request's AdaLN projections, looked up by a host-side step key (no device readback per forward)."""

    def __init__(
        self,
        time_embedder: Callable[[torch.Tensor], torch.Tensor],
        steps: Sequence[tuple[Hashable, torch.Tensor]],
        *,
        safe: Callable[[nn.Module], bool] = projection_plan_safe,
    ) -> None:
        """``steps``: (step key, that forward's unique timesteps) for every planned forward."""
        self.hits = self.misses = 0
        self._safe = safe
        self._key: Hashable | None = None
        self._entry: tuple[int, int] | None = None
        # step key -> (row count M, index in that M's table)
        self._index: dict[Hashable, tuple[int, int]] = {}
        self._tables: dict[int, dict[int, torch.Tensor]] = {}  # id(proj) -> M -> [steps, M, out]
        # (id(proj), M) -> (table, whether its row matched that forward's per-step result), until checked
        self._pending: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
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
            # The same call as the forward's embedding, so t_emb is bitwise the same.
            t_emb = time_embedder(unique)
            embed_dtypes.add(t_emb.dtype)
            rows.append(nn.functional.silu(t_emb).to(_BF16_DTYPE))
        self.embed_dtype = embed_dtypes.pop() if len(embed_dtypes) == 1 else None
        # M -> [steps, M, t_dim] BF16 projection inputs, as the per-step call computes them
        self._inputs = {m: torch.stack(rows) for m, rows in groups.items()}

    @property
    def planned_steps(self) -> int:
        return len(self._index)

    def select(self, key: Hashable | None) -> None:
        """Name the next forward's step (from host-side scalars), or None for an unplanned forward."""
        self._key = key

    def bind(self, t_emb: torch.Tensor) -> None:
        """Bind lookups to this forward. Only host metadata is checked: the embedding's dtype (changed by the
        control branch) and row count must match the planned step. The selected key is consumed. Tables built
        by earlier forwards are checked here, with one readback."""
        if self._pending:
            self._check_pending()
        entry = self._index.get(self._key) if self._key is not None else None
        self._key = None
        valid = entry is not None and t_emb.dtype == self.embed_dtype and t_emb.shape[0] == entry[0]
        self._entry = entry if valid else None

    def lookup(self, proj: nn.Module) -> torch.Tensor | None:
        entry = self._entry
        table = self._tables.get(id(proj), {}).get(entry[0]) if entry is not None else None
        if table is None:
            self.misses += 1
            return None
        self.hits += 1
        return table[entry[1]]

    def wants(self, proj: nn.Module) -> bool:
        """Whether this forward should build ``proj``'s table for its row count (once per request)."""
        entry = self._entry
        return entry is not None and (id(proj), entry[0]) not in self._built and self._safe(proj)

    def rows(self) -> torch.Tensor:
        """Every planned step's projection inputs for this forward's row count: [steps, M, t_dim]."""
        assert self._entry is not None
        return self._inputs[self._entry[0]]

    def record(self, proj: nn.Module, table: torch.Tensor, per_step: torch.Tensor) -> None:
        """Keep ``table`` ([steps, M, out]) pending a check of this forward's row against ``per_step``."""
        assert self._entry is not None
        m, index = self._entry
        self._built.add((id(proj), m))
        self._pending[(id(proj), m)] = (table, (table[index] == per_step).all())

    def _check_pending(self) -> None:
        pending, self._pending = self._pending, {}
        same = torch.stack([flag for _, flag in pending.values()]).tolist()
        for ((proj_id, m), (table, _)), keep in zip(pending.items(), same, strict=True):
            if keep:
                self._tables.setdefault(proj_id, {})[m] = table


def build_adaln_plan(
    model: nn.Module,
    branch,
    sigmas_video: Sequence[float],
    sigmas_audio: Sequence[float],
    cond_anchors: tuple[float, float],
) -> MiniMaxH3AdalnPlan | None:
    """Plan the timestep sets a plain schedule evaluates: (sigma_video[i], sigma_audio[i]) for every step."""
    cache = getattr(model, "adaln_cache", None)
    if cache is None or not cache.plan_enabled:
        return None
    steps = []
    timesteps = torch.empty(branch.seq_len, dtype=torch.float32, device=branch.device)
    with torch.inference_mode():
        for sigma_video, sigma_audio in zip(sigmas_video[:-1], sigmas_audio[:-1], strict=True):
            # Mirrors MiniMaxH3Model.prepare and MiniMaxH3ModelBatch, which select the same key per forward.
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
        return MiniMaxH3AdalnPlan(model.time_embedder, steps)


__all__ = ["MiniMaxH3AdalnPlan", "adaln_step_key", "build_adaln_plan", "projection_plan_safe"]
