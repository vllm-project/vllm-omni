# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WanCellBatch packs compatible same-cell members into one DiT call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell, WanCellBatch
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
        assert tokens.shape[0] == latent.shape[0]
        return tokens + 1

    def unembed(self, tokens, *, chunk, step, latent, timestep):
        del chunk, step, timestep
        self.unembed_calls += 1
        b = latent.shape[0]
        return torch.ones(b, *latent.shape[1:], dtype=latent.dtype, device=latent.device)

    def seq_len_for_latent(self, latent_shape):
        return 2


def _member(tf, *, seed: int, req: str) -> WanCell:
    del req
    sampler = FlowEuler(1, shift=1.0)
    return WanCell(
        tf,  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=torch.randn(1, 3, 4),
        latent_shape=(1, 4, 1, 2, 2),
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_wancell_batch_fuses_same_cell_into_one_embed_block():
    tf = _FakeTransformer()
    a = _member(tf, seed=1, req="A")
    b = _member(tf, seed=2, req="B")
    # Replace samplers with deterministic stubs so ctor variance cannot break the test.
    for member, seed in ((a, 1), (b, 2)):
        member.sampler = SimpleNamespace(
            timesteps=[1000.0],
            advance=lambda pred, latent, step, s=seed: latent - (0.01 * s) * pred,
        )
        member.num_denoise_steps = 1
        member._timesteps = [torch.tensor(1000.0)]
        member._timestep_clean = torch.zeros(())

    batch = WanCellBatch({"A": a, "B": b})
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
    assert isinstance(tf.last_kv_ctx, list) and len(tf.last_kv_ctx) == 2
    assert tf.prepare_calls[-1][0] == 2  # batched text
    assert set(a.finished) | set(b.finished) == {"A", "B"}
    assert 0 in a.finished["A"] and 0 in b.finished["B"]


def test_wancell_batch_falls_back_per_member_when_cells_differ():
    tf = _FakeTransformer()
    a = _member(tf, seed=1, req="A")
    b = _member(tf, seed=2, req="B")
    for member in (a, b):
        member.sampler = SimpleNamespace(
            timesteps=[1000.0],
            advance=lambda pred, latent, step: latent,
        )
        member.num_denoise_steps = 1
        member._timesteps = [torch.tensor(1000.0)]
        member._timestep_clean = torch.zeros(())

    batch = WanCellBatch({"A": a, "B": b})
    out = batch.forward(
        [("A", (0, 0, 0)), ("B", (1, 0, 0))],
        [[object()], [object()]],
        hidden=None,
    )
    assert out["latent"].shape[0] == 2
    assert tf.embed_calls == 2
    assert tf.block_calls == 2
