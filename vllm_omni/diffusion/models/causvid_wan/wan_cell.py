# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One schedule cell on Wan DiT: ``embed → block → unembed`` (+ Euler)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.experimental.ar_diffusion.cell_executor import CellAdapter
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
