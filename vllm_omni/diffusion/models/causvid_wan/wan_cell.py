# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One schedule cell on Wan DiT: ``embed → block → unembed`` (+ Euler)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import CellAdapter
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler


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
        tf = self.transformer
        tf.prepare_cell(self.prompt_embeds)
        for i, (req, cell) in enumerate(tasks):
            chunk, step, block = cell
            self._last_meta = (req, chunk, step)
            ctx_list = kv_contexts[i] if i < len(kv_contexts) else None
            kv_ctx = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            live = self._live.setdefault(req, {})
            latent, tokens = self._resolve_latent_hidden(
                req=req, chunk=chunk, step=step, index=i, n_tasks=len(tasks), hidden=hidden, live=live
            )
            t = self._timesteps[step] if step < self.num_denoise_steps else self._timestep_clean
            step_key = step if step < self.num_denoise_steps else self.num_denoise_steps

            if block == tf.start_layer and tf.is_layer_group_first:
                tokens = tf.embed(latent)
            elif tokens is None:
                raise RuntimeError(f"cell {(chunk, step, block)} missing hidden_states")
            tokens = tf.block(block, tokens, chunk=chunk, step=step_key, latent=latent, timestep=t, kv_ctx=kv_ctx)
            if not (block == tf.end_layer - 1 and tf.is_layer_group_last):
                live[chunk] = latent
                outs.append({"latent": latent, "hidden_states": tokens})
                continue
            pred = tf.unembed(tokens, chunk=chunk, step=step_key, latent=latent, timestep=t)
            if step < self.num_denoise_steps:
                latent = self.sampler.advance(pred, latent, step)
                if step == self.num_denoise_steps - 1:
                    self.finished.setdefault(req, {})[chunk] = latent.detach()
                    if self.on_finished is not None:
                        self.on_finished(chunk, latent)
            else:
                self.finished.setdefault(req, {})[chunk] = latent.detach()
            live[chunk] = latent
            outs.append({"latent": latent})
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
    """Request dispatcher carrying explicit member/cell metadata per row."""

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
        previous = self._rows(hidden)
        outputs = []
        for index, (req, cell) in enumerate(tasks):
            chunk, step, block = cell
            row_hidden = previous.get((req, chunk, step, block - 1))
            outputs.append(self.cells[req].forward(((req, cell),), [kv_contexts[index]], hidden=row_hidden))
        keys = outputs[0].keys()
        if any(output.keys() != keys for output in outputs):
            raise RuntimeError("batched Wan cells produced incompatible activation fields")
        result = {key: torch.cat([output[key] for output in outputs], dim=0) for key in keys}
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
