# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-level AdaLN projections for MiniMax-H3.

Each block's ``adaln_proj`` applies a ~520 MB weight to a handful of timestep
rows, so computing it every denoising step streams the whole weight from HBM
once per step. A request's per-step timestep sets follow from its sigma
schedule, so one call per block over all steps' rows reads each weight once.

Results are keyed by host-side scalars the sampler already has (branch and the
step's video/audio/condition timesteps), so a forward reads nothing back from
the device. A forward that was not planned (another solver's intermediate
point, a latent edit, a packed multi-request batch), or whose embedding was
transformed before AdaLN, falls back to the per-step call. Each projection's batched result is compared with a
per-step call when the plan is built and only bit-identical groups are kept.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence

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
        and not getattr(proj, "_offload_weights", False)
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
        projs: Iterable[nn.Module],
        steps: Sequence[tuple[Hashable, torch.Tensor]],
        *,
        safe: Callable[[nn.Module], bool] = projection_plan_safe,
    ) -> None:
        """``steps``: (step key, that forward's unique timesteps) for every planned forward."""
        self.hits = self.misses = 0
        self._key: Hashable | None = None
        self._entry: tuple[int, int] | None = None
        # step key -> (row count M, index in that M's table)
        self._index: dict[Hashable, tuple[int, int]] = {}
        self._tables: dict[int, dict[int, torch.Tensor]] = {}  # id(proj) -> M -> [steps, M, out]
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
        inputs = {m: torch.stack(rows) for m, rows in groups.items()}
        candidates = []
        for proj in projs:
            if not safe(proj):
                continue
            for m, x in inputs.items():
                table = proj.linear(x.reshape(-1, x.shape[-1]))[0].view(x.shape[0], m, -1)
                per_step = proj.linear(x[0])[0]
                candidates.append((proj, m, table, (table[0] == per_step).all()))
        # One readback per request decides which (projection, M) tables are kept.
        same = torch.stack([c[3] for c in candidates]).tolist() if candidates else []
        for (proj, m, table, _), keep in zip(candidates, same, strict=True):
            if keep:
                self._tables.setdefault(id(proj), {})[m] = table

    @property
    def planned_steps(self) -> int:
        return len(self._index)

    def select(self, key: Hashable | None) -> None:
        """Name the next forward's step (from host-side scalars), or None for an unplanned forward."""
        self._key = key

    def bind(self, t_emb: torch.Tensor) -> None:
        """Bind lookups to this forward. Only host metadata is checked: the embedding's dtype (changed by the
        control branch) and row count must match the planned step. The selected key is consumed."""
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
        projs = [block.adaln_proj for block in model.blocks]
        final_layer = getattr(model, "final_layer", None)
        if final_layer is not None:
            projs.append(final_layer.adaln_proj)
        return MiniMaxH3AdalnPlan(model.time_embedder, projs, steps)


__all__ = ["MiniMaxH3AdalnPlan", "adaln_step_key", "build_adaln_plan", "projection_plan_safe"]
