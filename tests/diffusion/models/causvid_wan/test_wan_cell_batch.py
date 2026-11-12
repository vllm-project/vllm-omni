# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WanCell.pack_compute_batch via framework BatchedCellAdapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.cell_batch import BatchedCellAdapter
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTransformer:
    start_layer = 0
    end_layer = 1
    is_layer_group_first = True
    is_layer_group_last = True
    dim = 4

    def __init__(self) -> None:
        self.prepare_calls: list[tuple[int, ...]] = []
        self.embed_calls = 0
        self.block_calls = 0
        self.unembed_calls = 0
        self.last_kv_ctx = None
        self.last_latent = None

    def prepare_cell(self, text: torch.Tensor, **_kwargs) -> None:
        self.prepare_calls.append(tuple(text.shape))

    def embed(self, latent: torch.Tensor) -> torch.Tensor:
        self.embed_calls += 1
        b = latent.shape[0]
        return torch.zeros(b, 2, self.dim, dtype=latent.dtype, device=latent.device)

    def block(self, index, tokens, *, chunk, step, latent, timestep, kv_ctx=None):
        del index, chunk, step, timestep
        self.block_calls += 1
        self.last_kv_ctx = kv_ctx
        self.last_latent = latent.detach().clone()
        assert tokens.shape[0] == latent.shape[0]
        return tokens + 1

    def unembed(self, tokens, *, chunk, step, latent, timestep):
        del chunk, step, timestep, tokens
        self.unembed_calls += 1
        return torch.ones_like(latent)

    def seq_len_for_latent(self, latent_shape):
        return 2


def _member(tf, *, seed: int, prompt: torch.Tensor | None = None) -> WanCell:
    sampler = FlowEuler(1, shift=1.0)
    cell = WanCell(
        tf,  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt if prompt is not None else torch.randn(1, 3, 4),
        latent_shape=(1, 4, 1, 2, 2),
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cell.sampler = SimpleNamespace(
        timesteps=[1000.0],
        advance=lambda pred, latent, step, s=seed: latent - (0.01 * s) * pred,
    )
    cell.num_denoise_steps = 1
    cell._timesteps = [torch.tensor(1000.0)]
    cell._timestep_clean = torch.zeros(())
    return cell


def _batched(members: dict[str, WanCell], *, max_batch_size: int) -> BatchedCellAdapter:
    return BatchedCellAdapter(
        members,
        max_batch_size=max_batch_size,
        compatibility_of=lambda req, cell: (
            cell,
            members[req].latent_shape,
            members[req].num_denoise_steps,
        ),
        pack_batch=WanCell.pack_compute_batch,
    )


def test_batched_adapter_fuses_same_cell_into_one_embed_block():
    tf = _FakeTransformer()
    a = _member(tf, seed=1, prompt=torch.ones(1, 3, 4))
    b = _member(tf, seed=2, prompt=torch.ones(1, 3, 4) * 2)
    batch = _batched({"A": a, "B": b}, max_batch_size=2)
    cell = (0, 0, 0)
    kv = [SimpleNamespace(name="ka"), SimpleNamespace(name="kb")]
    out = batch.forward(
        [("A", cell), ("B", cell)],
        [[kv[0]], [kv[1]]],
        hidden=None,
    )
    assert out["latent"].shape[0] == 2
    assert tf.embed_calls == 1
    assert tf.block_calls == 1
    assert tf.unembed_calls == 1
    assert batch.block_call_members == [2]
    assert len(batch.last_batches) == 1 and batch.last_batches[0].size == 2
    assert isinstance(tf.last_kv_ctx, list) and len(tf.last_kv_ctx) == 2
    assert tf.prepare_calls[-1][0] == 2
    assert set(a.finished) | set(b.finished) == {"A", "B"}
    assert 0 in a.finished["A"] and 0 in b.finished["B"]


def test_batched_adapter_falls_back_per_member_when_cells_differ():
    tf = _FakeTransformer()
    a = _member(tf, seed=1)
    b = _member(tf, seed=2)
    batch = _batched({"A": a, "B": b}, max_batch_size=2)
    out = batch.forward(
        [("A", (0, 0, 0)), ("B", (1, 0, 0))],
        [[object()], [object()]],
        hidden=None,
    )
    assert out["latent"].shape[0] == 2
    assert tf.embed_calls == 2
    assert tf.block_calls == 2
    assert batch.block_call_members == [1, 1]


def test_batched_adapter_matches_serial_within_tolerance():
    """Fused B=2 output matches independent B=1 runs (design §5.3 acceptance)."""
    cell = (0, 0, 0)
    prompts = {
        "A": torch.ones(1, 3, 4),
        "B": torch.ones(1, 3, 4) * 2,
    }
    seeds = {"A": 1, "B": 2}

    tf_serial = _FakeTransformer()
    serial_latents: dict[str, torch.Tensor] = {}
    for req in ("A", "B"):
        member = _member(tf_serial, seed=seeds[req], prompt=prompts[req].clone())
        member.forward([(req, cell)], [[SimpleNamespace(name=req)]], hidden=None)
        serial_latents[req] = member.finished[req][0].detach().clone()

    tf_fused = _FakeTransformer()
    fused_members = {
        req: _member(tf_fused, seed=seeds[req], prompt=prompts[req].clone()) for req in ("A", "B")
    }
    batch = _batched(fused_members, max_batch_size=2)
    batch.forward(
        [("A", cell), ("B", cell)],
        [[SimpleNamespace(name="A")], [SimpleNamespace(name="B")]],
        hidden=None,
    )
    assert tf_fused.block_calls == 1
    assert tf_serial.block_calls == 2
    for req in ("A", "B"):
        torch.testing.assert_close(
            fused_members[req].finished[req][0],
            serial_latents[req],
            rtol=1e-5,
            atol=1e-5,
        )


def test_batched_adapter_respects_max_batch_size_split():
    tf = _FakeTransformer()
    members = {rid: _member(tf, seed=i + 1) for i, rid in enumerate(("A", "B", "C"))}
    batch = _batched(members, max_batch_size=2)
    cell = (0, 0, 0)
    batch.forward(
        [(rid, cell) for rid in ("A", "B", "C")],
        [[object()] for _ in range(3)],
        hidden=None,
    )
    assert batch.block_call_members == [2, 1]
    assert [b.size for b in batch.last_batches] == [2, 1]
