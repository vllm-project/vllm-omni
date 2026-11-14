# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WanCell forward: model packing, member isolation, fused vs serial parity."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.batch import (
    ComputeBatch,
    MemberRef,
    organize_batches,
)
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler


def _make_member(
    tf,
    req: str,
    *,
    seeds: dict[str, int],
    prompts: dict[str, torch.Tensor],
    grids: dict[str, list[float]] | None,
    shared_grid: list[float] | None,
) -> WanCell:
    """Build one member, taking the per-case seed/prompt/grid inputs explicitly."""
    if grids is not None:
        return _member(tf, seed=seeds[req], prompt=prompts[req].clone(), timesteps=grids[req])
    assert shared_grid is not None
    return _member(
        tf,
        seed=seeds[req],
        prompt=prompts[req].clone(),
        num_denoise_steps=len(shared_grid),
        timesteps=shared_grid,
    )


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTransformer:
    """Predictions depend on timestep, chunk RoPE id, and prepared prompt text."""

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
        self.last_timestep = None
        self.last_chunks: list[int] | None = None
        self._text: torch.Tensor | None = None

    def prepare_cell(self, text: torch.Tensor, **_kwargs) -> None:
        if not isinstance(text, torch.Tensor):
            text = torch.cat(tuple(text), dim=0)
        self.prepare_calls.append(tuple(text.shape))
        self._text = text.detach().clone()

    def embed(self, latent: torch.Tensor) -> torch.Tensor:
        self.embed_calls += 1
        b = latent.shape[0]
        return torch.zeros(b, 2, self.dim, dtype=latent.dtype, device=latent.device)

    def block(self, index, tokens, *, latent, timestep, kv_ctx=None, chunks=None, steps=None, chunk=None, step=None):
        del index, steps, step
        self.block_calls += 1
        self.last_kv_ctx = kv_ctx
        self.last_latent = latent.detach().clone()
        self.last_timestep = timestep.detach().clone() if isinstance(timestep, torch.Tensor) else timestep
        self.last_chunks = list(chunks) if chunks is not None else ([int(chunk)] if chunk is not None else None)
        assert tokens.shape[0] == latent.shape[0]
        return tokens + 1

    def unembed(self, tokens, *, latent, timestep, chunks=None, steps=None, chunk=None, step=None):
        del steps, step
        self.unembed_calls += 1
        batch = int(latent.shape[0])
        ts = timestep.to(device=latent.device, dtype=torch.float32).reshape(-1)
        if ts.numel() == 1:
            ts = ts.expand(batch)
        if chunks is not None:
            chunk_ids = [int(c) for c in chunks]
        elif chunk is not None:
            chunk_ids = [int(chunk)] * batch
        else:
            chunk_ids = [0] * batch
        if self._text is None or int(self._text.shape[0]) != batch:
            raise RuntimeError("prepare_cell text batch must match latent batch")
        text_scale = self._text.reshape(batch, -1).to(dtype=torch.float32).mean(dim=1)
        tok_scale = tokens.reshape(batch, -1).to(dtype=torch.float32).mean(dim=1)
        scale = (
            (ts / 1000.0) + 0.01 * torch.tensor(chunk_ids, dtype=torch.float32) + 0.1 * text_scale + 0.01 * tok_scale
        )
        return scale.to(dtype=latent.dtype).view(batch, *([1] * (latent.ndim - 1))).expand_as(latent).contiguous()

    def seq_len_for_latent(self, latent_shape):
        return 2


def _failing_forward(*_args, **_kwargs):
    """Transformer forward stand-in that fails the shared tick."""
    raise RuntimeError("fused forward failure")


def _member(
    tf,
    *,
    seed: int,
    prompt: torch.Tensor | None = None,
    shift: float = 1.0,
    num_denoise_steps: int = 1,
    timesteps: list[float] | None = None,
) -> WanCell:
    n = max(1, int(num_denoise_steps))
    grid = timesteps if timesteps is not None else [float(1000 - 200 * i) for i in range(n)]
    if len(grid) != n:
        raise ValueError(f"timesteps length {len(grid)} != num_denoise_steps {n}")
    sampler = FlowEuler(n, shift=shift)
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
        timesteps=list(grid),
        shift=shift,
        advance=lambda pred, latent, step, s=seed: latent - (0.01 * s) * (step + 1) * pred,
    )
    cell.num_denoise_steps = n
    cell._timesteps = [torch.tensor(float(t)) for t in grid]
    cell._timestep_clean = torch.zeros(())
    return cell


def _run_batches(
    members: dict[str, WanCell],
    tasks: list[tuple[str, tuple[int, int, int]]],
    kvs: list,
    *,
    max_batch_size: int,
    member_inputs: list | None = None,
) -> list[ComputeBatch]:
    batches = organize_batches(
        tasks,
        kvs,
        compatibility_of=lambda req, cell: members[req].compatibility_key(cell),
        max_batch_size=max_batch_size,
    )
    for batch in batches:
        adapters = [members[m.request_id] for m in batch.members]
        if member_inputs is None:
            inputs = [None] * batch.size
        else:
            inputs = [member_inputs[m.task_index] for m in batch.members]
        adapters[0].forward(
            batch,
            adapters,
            member_inputs=inputs,
        )
    return batches


def _one(
    member: WanCell,
    req: str,
    cell: tuple[int, int, int],
    kv,
    *,
    member_input=None,
) -> dict[str, torch.Tensor]:
    chunk, step, block = cell
    batch = ComputeBatch(
        members=(MemberRef(req, chunk=chunk, step=step, block=block, task_index=0),),
        kv_views=(kv,),
    )
    return member.forward(batch, [member], member_inputs=[member_input])


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "name": "same_progress",
                "grid": [1000.0],
                "cells": {"A": (0, 0, 0), "B": (0, 0, 0)},
                "expected_ts": [1000.0, 1000.0],
                "expected_chunks": [0, 0],
                "finish_chunks": {"A": 0, "B": 0},
            },
            id="same_progress",
        ),
        pytest.param(
            {
                "name": "different_chunk_step",
                "grid": [1000.0, 800.0, 600.0, 400.0, 200.0],
                "cells": {"A": (0, 2, 0), "B": (3, 4, 0)},
                "expected_ts": [600.0, 200.0],
                "expected_chunks": [0, 3],
                "finish_chunks": {"B": 3},
            },
            id="different_chunk_step",
        ),
        pytest.param(
            {
                "name": "different_shift",
                "grid": None,
                "member_timesteps": {"A": [800.0], "B": [500.0]},
                "cells": {"A": (0, 0, 0), "B": (0, 0, 0)},
                "expected_ts": [800.0, 500.0],
                "expected_chunks": [0, 0],
                "finish_chunks": {"A": 0, "B": 0},
            },
            id="different_shift",
        ),
    ],
)
def test_wan_cell_fused_matches_serial(case):
    """One DiT call for two members; latents match independent runs with input-dependent preds."""
    prompts = {"A": torch.ones(1, 3, 4), "B": torch.ones(1, 3, 4) * 2}
    seeds = {"A": 1, "B": 2}
    cells = case["cells"]
    grids = case.get("member_timesteps")
    shared_grid = case.get("grid")

    make = partial(
        _make_member,
        seeds=seeds,
        prompts=prompts,
        grids=grids,
        shared_grid=shared_grid,
    )

    tf_serial = _FakeTransformer()
    serial_live: dict[str, torch.Tensor] = {}
    for req in ("A", "B"):
        member = make(tf_serial, req)
        _one(member, req, cells[req], SimpleNamespace(name=req))
        chunk = cells[req][0]
        serial_live[req] = member._live[req][chunk].detach().clone()
    assert tf_serial.block_calls == 2
    assert tf_serial.unembed_calls == 2
    # Constant-ones preds would collapse these; input-dependent unembed keeps them apart.
    assert not torch.allclose(serial_live["A"], serial_live["B"])

    tf_fused = _FakeTransformer()
    fused = {req: make(tf_fused, req) for req in ("A", "B")}
    batches = _run_batches(
        fused,
        [("A", cells["A"]), ("B", cells["B"])],
        [[SimpleNamespace(name="A")], [SimpleNamespace(name="B")]],
        max_batch_size=2,
    )
    assert len(batches) == 1 and batches[0].size == 2
    assert tf_fused.embed_calls == 1
    assert tf_fused.block_calls == 1
    assert tf_fused.unembed_calls == 1
    assert tf_fused.prepare_calls[-1][0] == 2
    assert isinstance(tf_fused.last_kv_ctx, list) and [kv.name for kv in tf_fused.last_kv_ctx] == ["A", "B"]
    assert tf_fused.last_chunks == case["expected_chunks"]
    torch.testing.assert_close(tf_fused.last_timestep, torch.tensor(case["expected_ts"]))
    for req in ("A", "B"):
        chunk = cells[req][0]
        torch.testing.assert_close(
            fused[req]._live[req][chunk],
            serial_live[req],
            rtol=1e-5,
            atol=1e-5,
        )
    for req, chunk in case["finish_chunks"].items():
        assert chunk in fused[req].finished[req]
    for req in ("A", "B"):
        if req not in case["finish_chunks"]:
            assert cells[req][0] not in fused[req].finished.get(req, {})


def test_wan_cell_mid_group_uses_per_member_hidden_inputs():
    tf = _FakeTransformer()
    tf.start_layer = 1
    tf.is_layer_group_first = False
    a = _member(tf, seed=1, prompt=torch.ones(1, 3, 4))
    b = _member(tf, seed=2, prompt=torch.ones(1, 3, 4) * 2)
    cell = (0, 0, 1)
    inputs = [
        {
            "latent": torch.full((1, 4, 1, 2, 2), 1.0),
            "hidden_states": torch.full((1, 2, 4), 10.0),
        },
        {
            "latent": torch.full((1, 4, 1, 2, 2), 2.0),
            "hidden_states": torch.full((1, 2, 4), 20.0),
        },
    ]
    _run_batches(
        {"A": a, "B": b},
        [("A", cell), ("B", cell)],
        [[object()], [object()]],
        max_batch_size=2,
        member_inputs=inputs,
    )
    assert tf.embed_calls == 0
    assert tf.block_calls == 1
    assert tf.last_latent is not None and tf.last_latent.shape[0] == 2
    assert torch.allclose(tf.last_latent[0], torch.full((4, 1, 2, 2), 1.0))
    assert torch.allclose(tf.last_latent[1], torch.full((4, 1, 2, 2), 2.0))


@pytest.mark.parametrize("fail_attr", ["block", "unembed"])
def test_failed_fused_tick_leaves_every_member_untouched(monkeypatch, fail_attr):
    """A raising shared forward must not advance any member's state.

    The executor publishes KV only after a successful forward, so a partially
    applied tick would leave the pool claiming writes that never happened.
    """
    tf = _FakeTransformer()
    a = _member(tf, seed=1, prompt=torch.ones(1, 3, 4))
    b = _member(tf, seed=2, prompt=torch.ones(1, 3, 4) * 2)
    members = {"A": a, "B": b}
    tasks = [("A", (0, 0, 0)), ("B", (0, 0, 0))]
    kvs = [object(), object()]

    monkeypatch.setattr(tf, fail_attr, _failing_forward)
    with pytest.raises(RuntimeError, match="fused forward failure"):
        _run_batches(members, tasks, kvs, max_batch_size=2)

    for req in ("A", "B"):
        assert members[req].finished == {}, req
        # The live map may be pre-created per member, but no chunk may advance.
        assert members[req]._live.get(req, {}) == {}, req
