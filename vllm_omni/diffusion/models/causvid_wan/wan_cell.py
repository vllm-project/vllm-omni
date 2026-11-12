# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One schedule cell on Wan DiT: ``embed → block → unembed`` (+ Euler)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.cell_batch import (
    CellComputeBatch,
    CellMemberRef,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import CellAdapter
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Cell
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
        transformer.prepare_cell(prompt_embeds)

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
        self,
        *,
        req: str,
        chunk: int,
        step: int,
        index: int,
        n_tasks: int,
        hidden: Any,
        live: dict[int, torch.Tensor],
        stash_take: Callable[[str, int, int], Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent: torch.Tensor | None = None
        tokens: torch.Tensor | None = None
        src = hidden if isinstance(hidden, dict) else None
        take = stash_take or self.take_stashed_activation
        if src is None or "hidden_states" not in src:
            src = take(req, chunk, step)
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

    @staticmethod
    def pack_compute_batch(
        batch: CellComputeBatch,
        members: Sequence["WanCell"],
        *,
        tasks: Sequence[tuple[str, Cell]],
        hidden: Any,
        stash_take: Callable[[str, int, int], Any],
    ) -> dict[str, torch.Tensor]:
        """Wan DiT packer for one ``CellComputeBatch`` (model backend only)."""
        if len(members) != batch.size:
            raise ValueError(f"members length {len(members)} != batch size {batch.size}")
        chunk, step, block = batch.cell
        tf = members[0].transformer
        text = torch.cat([m.prompt_embeds for m in members], dim=0)
        tf.prepare_cell(text)

        latents: list[torch.Tensor] = []
        tokens_list: list[torch.Tensor | None] = []
        for local_i, member_ref in enumerate(batch.members):
            member = members[local_i]
            live = member._live.setdefault(member_ref.request_id, {})
            latent, tokens = member._resolve_latent_hidden(
                req=member_ref.request_id,
                chunk=chunk,
                step=step,
                index=member_ref.task_index,
                n_tasks=len(tasks),
                hidden=hidden,
                live=live,
                stash_take=stash_take,
            )
            latents.append(latent)
            tokens_list.append(tokens)

        latent_b = torch.cat(latents, dim=0)
        first = members[0]
        t = first._timesteps[step] if step < first.num_denoise_steps else first._timestep_clean
        step_key = step if step < first.num_denoise_steps else first.num_denoise_steps

        if block == tf.start_layer and tf.is_layer_group_first:
            tokens_b = tf.embed(latent_b)
        else:
            if any(tok is None for tok in tokens_list):
                raise RuntimeError(f"cell {(chunk, step, block)} missing hidden_states")
            tokens_b = torch.cat([tok for tok in tokens_list if tok is not None], dim=0)

        tokens_b = tf.block(
            block,
            tokens_b,
            chunk=chunk,
            step=step_key,
            latent=latent_b,
            timestep=t,
            kv_ctx=list(batch.kv_views),
        )
        if not (block == tf.end_layer - 1 and tf.is_layer_group_last):
            for local_i, member_ref in enumerate(batch.members):
                members[local_i]._live.setdefault(member_ref.request_id, {})[chunk] = latents[local_i]
            return {"latent": latent_b, "hidden_states": tokens_b}

        pred_b = tf.unembed(tokens_b, chunk=chunk, step=step_key, latent=latent_b, timestep=t)
        out_latents: list[torch.Tensor] = []
        for local_i, member_ref in enumerate(batch.members):
            member = members[local_i]
            req = member_ref.request_id
            latent = latents[local_i]
            pred = pred_b[batch.row_slice(local_i)]
            if step < member.num_denoise_steps:
                latent = member.sampler.advance(pred, latent, step)
                if step == member.num_denoise_steps - 1:
                    member.finished.setdefault(req, {})[chunk] = latent.detach()
                    if member.on_finished is not None:
                        member.on_finished(chunk, latent)
            else:
                member.finished.setdefault(req, {})[chunk] = latent.detach()
            member._live.setdefault(req, {})[chunk] = latent
            out_latents.append(latent)
        return {"latent": torch.cat(out_latents, dim=0)}

    def forward(self, tasks, kv_contexts, *, hidden):
        """Single-request CellAdapter entry; multi-request uses ``BatchedCellAdapter``."""
        if not tasks:
            return hidden
        outs: list[dict[str, torch.Tensor] | None] = [None] * len(tasks)
        for i, (req, cell) in enumerate(tasks):
            ctx_list = kv_contexts[i] if i < len(kv_contexts) else None
            kv_ctx = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            chunk, step, _block = cell
            self._last_meta = (req, chunk, step)
            batch = CellComputeBatch(
                members=(CellMemberRef(request_id=req, cell=cell, task_index=i),),
                kv_views=(kv_ctx,),
            )
            packed = self.pack_compute_batch(
                batch,
                members=[self],
                tasks=tasks,
                hidden=hidden,
                stash_take=self.take_stashed_activation,
            )
            outs[i] = {k: v[0:1] for k, v in packed.items()}
        typed_outs = [item for item in outs if item is not None]
        if len(typed_outs) == 1:
            return typed_outs[0]
        keys = typed_outs[0].keys()
        return {k: torch.cat([o[k] for o in typed_outs], dim=0) for k in keys}

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
