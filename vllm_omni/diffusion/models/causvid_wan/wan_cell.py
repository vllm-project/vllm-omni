# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One schedule cell on Wan DiT: ``embed → block → unembed`` (+ Euler)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.batch import ComputeBatch
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import CellAdapter
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Cell
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler


class WanCell(CellAdapter):
    """Wan DiT backend for one request: compatibility key + ``forward``."""

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
        # FlowEuler shift selects each request's timestep grid; packed via per-member t.
        self.shift = float(getattr(sampler, "shift", 1.0))
        self.finished: dict[str, dict[int, torch.Tensor]] = {}
        self._live: dict[str, dict[int, torch.Tensor]] = {}
        self._timesteps = [torch.tensor(float(t), dtype=torch.float32) for t in sampler.timesteps]
        self._timestep_clean = torch.zeros((), dtype=torch.float32)
        transformer.prepare_cell(prompt_embeds)

    def compatibility_key(self, cell: Cell) -> Hashable:
        """Same DiT block + latent shape/dtype may pack; chunk/step/t are member inputs."""
        _chunk, _step, block = cell
        return (int(block), self.latent_shape, str(self.dtype))

    def _init_noise(self, chunk: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed * 1_000_003 + chunk * 4096)
        return torch.randn(self.latent_shape, generator=gen, device=self.device, dtype=self.dtype)

    def _resolve_member_input(
        self,
        *,
        req: str,
        chunk: int,
        member_input: Any,
        live: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent: torch.Tensor | None = None
        tokens: torch.Tensor | None = None
        if isinstance(member_input, dict):
            if "latent" in member_input and isinstance(member_input["latent"], torch.Tensor):
                latent = member_input["latent"]
            if "hidden_states" in member_input and isinstance(member_input["hidden_states"], torch.Tensor):
                tokens = member_input["hidden_states"]
        elif isinstance(member_input, torch.Tensor):
            latent = member_input
        if latent is None:
            stored = live.get(chunk)
            latent = self._init_noise(chunk) if stored is None else stored
        return latent, tokens

    def forward(
        self,
        batch: ComputeBatch,
        members: Sequence[CellAdapter],
        *,
        task_count: int,
        member_inputs: Sequence[Any | None],
    ) -> dict[str, torch.Tensor]:
        """Wan DiT packer for one ``ComputeBatch`` (inputs already mapped by executor)."""
        del task_count
        typed_members = [m for m in members if isinstance(m, WanCell)]
        if len(typed_members) != batch.size:
            raise ValueError(f"members length {len(typed_members)} != batch size {batch.size}")
        if len(member_inputs) != batch.size:
            raise ValueError(f"member_inputs length {len(member_inputs)} != batch size {batch.size}")
        block = batch.block
        tf = typed_members[0].transformer
        text = torch.cat([m.prompt_embeds for m in typed_members], dim=0)
        tf.prepare_cell(text)

        latents: list[torch.Tensor] = []
        tokens_list: list[torch.Tensor | None] = []
        timesteps: list[float] = []
        chunks: list[int] = []
        steps: list[int] = []
        for local_i, member_ref in enumerate(batch.members):
            member = typed_members[local_i]
            chunk, step, member_block = member_ref.chunk, member_ref.step, member_ref.block
            if int(member_block) != int(block):
                raise ValueError(f"member block {member_block} != batch.block {block}")
            live = member._live.setdefault(member_ref.request_id, {})
            latent, tokens = member._resolve_member_input(
                req=member_ref.request_id,
                chunk=chunk,
                member_input=member_inputs[local_i],
                live=live,
            )
            latents.append(latent)
            tokens_list.append(tokens)
            chunks.append(int(chunk))
            steps.append(int(step))
            if step < member.num_denoise_steps:
                timesteps.append(float(member._timesteps[step].item()))
            else:
                timesteps.append(float(member._timestep_clean.item()))

        latent_b = torch.cat(latents, dim=0)
        # One noise-level condition per member (shift / plan step may differ).
        t = torch.tensor(timesteps, dtype=torch.float32, device=latent_b.device)

        if block == tf.start_layer and tf.is_layer_group_first:
            tokens_b = tf.embed(latent_b)
        else:
            if any(tok is None for tok in tokens_list):
                raise RuntimeError(f"block {block} missing hidden_states")
            tokens_b = torch.cat([tok for tok in tokens_list if tok is not None], dim=0)

        tokens_b = tf.block(
            block,
            tokens_b,
            chunks=chunks,
            steps=steps,
            latent=latent_b,
            timestep=t,
            kv_ctx=list(batch.kv_views),
        )
        if not (block == tf.end_layer - 1 and tf.is_layer_group_last):
            for local_i, member_ref in enumerate(batch.members):
                typed_members[local_i]._live.setdefault(member_ref.request_id, {})[member_ref.chunk] = latents[local_i]
            return {"latent": latent_b, "hidden_states": tokens_b}

        pred_b = tf.unembed(
            tokens_b, chunks=chunks, steps=steps, latent=latent_b, timestep=t
        )
        out_latents: list[torch.Tensor] = []
        for local_i, member_ref in enumerate(batch.members):
            member = typed_members[local_i]
            req = member_ref.request_id
            chunk, step = member_ref.chunk, member_ref.step
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
