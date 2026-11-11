# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One schedule cell on Wan DiT: ``embed → block → unembed`` (+ Euler).

A tick can carry several compatible requests. They then run through one batched
tensor forward: the transformer sees a single ``B = len(members)`` activation,
while conditioning, seed, sampler, KV owner and the returned activation rows
stay per member.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import CellAdapter
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler


@dataclass(frozen=True)
class _CellRow:
    """One member's slice of a cell tick."""

    req: str
    chunk: int
    step: int
    block: int
    step_key: int
    timestep: torch.Tensor
    latent: torch.Tensor
    tokens: torch.Tensor | None
    kv_ctx: Any | None


def _kv_view(kv_contexts: Sequence[Any] | None, index: int) -> Any | None:
    """This row's KV view; a member's per-layer views collapse to its own layer."""
    if kv_contexts is None or index >= len(kv_contexts):
        return None
    view = kv_contexts[index]
    return view[0] if isinstance(view, list) else view


def _previous_row(
    previous: Mapping[tuple[str, int, int, int], dict[str, torch.Tensor]] | None,
    hidden: Any,
    *,
    req: str,
    chunk: int,
    step: int,
    block: int,
    n_tasks: int,
) -> Any:
    """Same member's previous-block activation, when this tick carries one."""
    if isinstance(hidden, torch.Tensor):
        # Batched payload: ``_resolve_latent_hidden`` slices the row itself.
        return hidden
    if isinstance(hidden, dict):
        if previous:
            return previous.get((req, chunk, step, block - 1))
        return hidden if n_tasks == 1 else None
    return None


def _stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


def _cell_rows(
    cells: Mapping[str, WanCell],
    tasks: Sequence[tuple[str, Any]],
    kv_contexts: Sequence[Any] | None,
    *,
    previous: Mapping[tuple[str, int, int, int], dict[str, torch.Tensor]] | None,
    hidden: Any,
    index_offset: int,
    n_tasks: int,
) -> list[_CellRow]:
    rows: list[_CellRow] = []
    for offset, (req, cell) in enumerate(tasks):
        chunk, step, block = cell
        member = cells[req]
        member._last_meta = (req, chunk, step)
        live = member._live.setdefault(req, {})
        latent, tokens = member._resolve_latent_hidden(
            req=req,
            chunk=chunk,
            step=step,
            index=index_offset + offset,
            n_tasks=n_tasks,
            hidden=_previous_row(previous, hidden, req=req, chunk=chunk, step=step, block=block, n_tasks=n_tasks),
            live=live,
        )
        noisy = step < member.num_denoise_steps
        rows.append(
            _CellRow(
                req=req,
                chunk=chunk,
                step=step,
                block=block,
                step_key=step if noisy else member.num_denoise_steps,
                timestep=member._timesteps[step] if noisy else member._timestep_clean,
                latent=latent,
                tokens=tokens,
                kv_ctx=_kv_view(kv_contexts, offset),
            )
        )
    return rows


def _block_input(tf: Any, rows: Sequence[_CellRow], latents: torch.Tensor) -> torch.Tensor:
    """First-stage rows embed the latent; later stages reuse the previous block's tokens."""
    if rows[0].block == tf.start_layer and tf.is_layer_group_first:
        return tf.embed(latents)
    missing = [row.req for row in rows if row.tokens is None]
    if missing:
        raise RuntimeError(f"cell {(rows[0].chunk, rows[0].step, rows[0].block)} missing hidden_states for {missing}")
    return _stack([row.tokens for row in rows if row.tokens is not None])


def _advance_rows(cells: Mapping[str, WanCell], rows: Sequence[_CellRow], pred: torch.Tensor) -> list[torch.Tensor]:
    """Advance every member with its own sampler and record its finished chunk."""
    advanced: list[torch.Tensor] = []
    for index, row in enumerate(rows):
        member = cells[row.req]
        latent = row.latent
        if row.step < member.num_denoise_steps:
            latent = member.sampler.advance(pred[index : index + 1], latent, row.step)
            if row.step == member.num_denoise_steps - 1:
                member.finished.setdefault(row.req, {})[row.chunk] = latent.detach()
                if member.on_finished is not None:
                    member.on_finished(row.chunk, latent)
        else:
            member.finished.setdefault(row.req, {})[row.chunk] = latent.detach()
        member._live[row.req][row.chunk] = latent
        advanced.append(latent)
    return advanced


def _run_rows(
    cells: Mapping[str, WanCell],
    tasks: Sequence[tuple[str, Any]],
    kv_contexts: Sequence[Any] | None,
    *,
    previous: Mapping[tuple[str, int, int, int], dict[str, torch.Tensor]] | None = None,
    hidden: Any = None,
    index_offset: int = 0,
    n_tasks: int = 1,
) -> dict[str, torch.Tensor]:
    """One tensor forward for every row, then per-member sampler bookkeeping.

    State is committed only after the forward succeeds: a failing member leaves
    every member's live latent, finished chunks and stashed activation untouched.
    """
    snapshots = {req: (dict(cell._stash), dict(cell._live)) for req, cell in cells.items()}
    try:
        rows = _cell_rows(
            cells,
            tasks,
            kv_contexts,
            previous=previous,
            hidden=hidden,
            index_offset=index_offset,
            n_tasks=n_tasks,
        )
        tf = next(iter(cells.values())).transformer
        blocks = {row.block for row in rows}
        if len(blocks) != 1:
            raise ValueError(f"one cell tick runs one block, got {sorted(blocks)}")
        members = tuple(row.req for row in rows)
        tf.prepare_cell({req: cells[req].prompt_embeds for req in members}, members=members)
        latents = _stack([row.latent for row in rows])
        tokens = _block_input(tf, rows, latents)
        if len(rows) == 1:
            row = rows[0]
            tokens = tf.block(
                row.block,
                tokens,
                chunk=row.chunk,
                step=row.step_key,
                latent=row.latent,
                timestep=row.timestep,
                kv_ctx=row.kv_ctx,
                member=row.req,
            )
        else:
            tokens = tf.block_batch(
                rows[0].block,
                tokens,
                members=members,
                chunks=tuple(row.chunk for row in rows),
                steps=tuple(row.step_key for row in rows),
                latents=tuple(row.latent for row in rows),
                timesteps=tuple(row.timestep for row in rows),
                kv_ctxs=tuple(row.kv_ctx for row in rows),
            )
        if not (rows[0].block == tf.end_layer - 1 and tf.is_layer_group_last):
            for row in rows:
                cells[row.req]._live[row.req][row.chunk] = row.latent
            return {"latent": latents, "hidden_states": tokens}
        if len(rows) == 1:
            row = rows[0]
            pred = tf.unembed(
                tokens,
                chunk=row.chunk,
                step=row.step_key,
                latent=row.latent,
                timestep=row.timestep,
                member=row.req,
            )
        else:
            pred = tf.unembed_batch(
                tokens,
                members=members,
                steps=tuple(row.step_key for row in rows),
                timesteps=tuple(row.timestep for row in rows),
                latent=latents,
            )
        return {"latent": _stack(_advance_rows(cells, rows, pred))}
    except BaseException:
        for req, (stash, live) in snapshots.items():
            cells[req]._stash = stash
            cells[req]._live = live
        raise


class WanCell(CellAdapter):
    """Run one ``(chunk, step, block)`` cell; last stage applies FlowEuler and records latents."""

    def __init__(
        self,
        transformer: StageWanTransformer,
        *,
        sampler: FlowEuler,
        prompt_embeds: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        on_finished: Callable[[int, torch.Tensor], None] | None = None,
    ) -> None:
        self.transformer = transformer
        self.sampler = sampler
        self.prompt_embeds = prompt_embeds
        self.latent_shape = latent_shape
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.on_finished = on_finished
        self.num_denoise_steps = len(sampler.timesteps)
        self.finished: dict[str, dict[int, torch.Tensor]] = {}
        self._live: dict[str, dict[int, torch.Tensor]] = {}
        self._stash: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self._last_meta: tuple[str, int, int] | None = None
        self._timesteps = [torch.tensor(float(t), dtype=torch.float32) for t in sampler.timesteps]
        self._timestep_clean = torch.zeros((), dtype=torch.float32)
        # Conditioning belongs to the request. A batched dispatcher may alternate
        # requests through this shared transformer on every executor tick.
        self.transformer.prepare_cell(self.prompt_embeds)

    def _init_noise(self, chunk: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed * 1_000_003 + chunk * 4096)
        return torch.randn(self.latent_shape, generator=gen, device=self.device, dtype=self.dtype)

    @staticmethod
    def _slice_batch(tensor: torch.Tensor, index: int, n_tasks: int) -> torch.Tensor:
        return tensor[index : index + 1] if tensor.shape[0] == n_tasks else tensor

    def stash_activation(
        self,
        payload: dict,
        *,
        handles: list | None = None,
        meta: tuple[str, int, int] | None = None,
    ) -> None:
        del handles
        tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor) and k not in ("chunk", "step")}
        key = meta or self._last_meta
        if key is None:
            raise RuntimeError("stash_activation needs meta")
        self._stash[key] = tensors

    def take_stashed_activation(self, req: str, chunk: int, step: int):
        return self._stash.pop((req, chunk, step), None) or self._stash.pop(("", chunk, step), None)

    def _resolve_latent_hidden(
        self, *, req: str, chunk: int, step: int, index: int, n_tasks: int, hidden: Any, live: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent: torch.Tensor | None = None
        tokens: torch.Tensor | None = None
        src = hidden if isinstance(hidden, dict) else None
        if src is None or "hidden_states" not in src:
            src = self.take_stashed_activation(req, chunk, step)
        if isinstance(src, dict):
            if "latent" in src:
                latent = self._slice_batch(src["latent"], index, n_tasks)
            if "hidden_states" in src:
                tokens = self._slice_batch(src["hidden_states"], index, n_tasks)
        elif isinstance(hidden, torch.Tensor):
            latent = self._slice_batch(hidden, index, n_tasks)
        if latent is None:
            stored = live.get(chunk)
            latent = self._init_noise(chunk) if stored is None else stored
        return latent, tokens

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        outs: list[dict[str, torch.Tensor]] = []
        for index, (req, cell) in enumerate(tasks):
            views = [kv_contexts[index]] if kv_contexts is not None and index < len(kv_contexts) else None
            outs.append(
                _run_rows(
                    {req: self},
                    ((req, cell),),
                    views,
                    hidden=hidden,
                    index_offset=index,
                    n_tasks=len(tasks),
                )
            )
        if len(outs) == 1:
            return outs[0]
        keys = outs[0].keys()
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}

    def pack_activation(self, output):
        if isinstance(output, dict):
            return {k: v.contiguous() for k, v in output.items() if isinstance(v, torch.Tensor)}
        if isinstance(output, IntermediateTensors):
            return {k: v.contiguous() for k, v in output.tensors.items()}
        if isinstance(output, torch.Tensor):
            return {"latent": output.contiguous()}
        raise TypeError(f"pack expects dict/tensor, got {type(output)}")

    def activation_spec(self, *, batch: int, include_hidden: bool):
        batch = max(1, int(batch))
        _b, c, t, h, w = self.latent_shape
        spec: list[tuple[str, tuple[int, ...], torch.dtype]] = [
            ("latent", (batch, int(c), int(t), int(h), int(w)), self.dtype),
        ]
        if include_hidden:
            seq = self.transformer.seq_len_for_latent((batch, int(c), int(t), int(h), int(w)))
            spec.append(("hidden_states", (batch, seq, int(self.transformer.dim)), self.dtype))
        return spec

    def unpack_activation(self, payload: dict):
        return payload


class WanCellBatch(CellAdapter):
    """Fused dispatcher: one tensor forward per tick carrying explicit member rows.

    Members keep their own conditioning, seed, sampler, KV owner and output row;
    only the transformer compute is shared.
    """

    _META = "wan_cell_rows"

    def __init__(self, cells: dict[str, WanCell]) -> None:
        if not cells:
            raise ValueError("WanCellBatch requires at least one cell")
        self.cells = cells
        self._requests = list(cells)
        self._indices = {req: index for index, req in enumerate(self._requests)}
        first = next(iter(cells.values()))
        self.device = first.device
        schema = first.activation_spec(batch=1, include_hidden=True)
        if any(cell.activation_spec(batch=1, include_hidden=True) != schema for cell in cells.values()):
            raise ValueError("incompatible WanCell activation shapes or dtypes")
        if len({id(cell.transformer) for cell in cells.values()}) != 1:
            raise ValueError("WanCellBatch members must share one transformer")

    def _rows(self, payload):
        if not isinstance(payload, dict) or self._META not in payload:
            return {}
        rows = {}
        for index, (member, chunk, step, block) in enumerate(payload[self._META].tolist()):
            req = self._requests[member]
            rows[(req, chunk, step, block)] = {
                key: value[index : index + 1] for key, value in payload.items() if key != self._META
            }
        return rows

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        members: dict[str, WanCell] = {}
        for req, _cell in tasks:
            member = self.cells.get(req)
            if member is None:
                raise KeyError(f"WanCellBatch has no member {req!r}")
            members[req] = member
        result = _run_rows(
            members,
            tasks,
            kv_contexts,
            previous=self._rows(hidden),
            hidden=hidden,
            n_tasks=len(tasks),
        )
        result[self._META] = torch.tensor(
            [(self._indices[req], *cell) for req, cell in tasks], dtype=torch.int64, device=self.device
        )
        return result

    def pack_activation(self, output):
        return {key: value.contiguous() for key, value in output.items()}

    def unpack_activation(self, payload):
        return payload

    def activation_spec(self, *, batch: int, include_hidden: bool):
        spec = next(iter(self.cells.values())).activation_spec(batch=batch, include_hidden=include_hidden)
        return [*spec, (self._META, (batch, 4), torch.int64)]

    def stash_activation(self, payload, *, handles=None, meta=None):
        del handles
        if meta is None:
            raise RuntimeError("WanCellBatch stash_activation needs metadata")
        rows = self._rows(payload)
        if not rows:
            raise RuntimeError("WanCellBatch activation is missing member metadata")
        # The executor metadata describes only the first task; route rows using
        # their own chunk/step and apply just its cross-replica step adjustment.
        first_step = int(next(iter(rows))[2])
        step_delta = int(meta[2]) - first_step
        for (req, chunk, step, _block), row in rows.items():
            self.cells[req].stash_activation(row, meta=(req, chunk, int(step) + step_delta))

    def take_stashed_activation(self, req, chunk, step):
        return self.cells[req].take_stashed_activation(req, chunk, step)
