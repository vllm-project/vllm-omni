# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end: one outer session, multiple Noisy KV requests with advancing chunk_base."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionSamplingParams
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.cell_executor import CellAdapter, run_cell_pipeline
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    clean_history_cells,
    prior_history_in_pool,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "world-e2e"
HISTORY = 1
DENOISE = 1
CHUNK_TOKENS = 4


@dataclass
class OuterSession:
    """Serving-side session: owns identity and chunk_base only."""

    session_id: str
    chunk_base: int = 0

    def next_request(self, *, num_chunks: int, request_id: str) -> OmniDiffusionRequest:
        req = OmniDiffusionRequest(
            prompt="e2e",
            request_id=request_id,
            sampling_params=OmniDiffusionSamplingParams(
                extra_args={
                    "session_id": self.session_id,
                    "chunk_base": self.chunk_base,
                    "num_chunks": num_chunks,
                    "num_denoise_steps": DENOISE,
                    "kv_history_chunks": HISTORY,
                    "cell_schedule": "self_forcing",
                }
            ),
        )
        self.chunk_base += num_chunks
        return req


class _CountingAdapter(CellAdapter):
    def __init__(self) -> None:
        self.cells: list[tuple[int, int, int]] = []

    def forward(self, tasks, kv_contexts, *, hidden):
        for _req, cell in tasks:
            self.cells.append(cell)
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


class SessionCellPipeline:
    """Mirrors CausVid's session_id / chunk_base / retain protocol on CPU."""

    def __init__(self) -> None:
        self._chunk_ctx = None
        self.last_adapter: _CountingAdapter | None = None
        self.executed_cells: list[tuple[int, int, int]] = []

    def ar_diffusion_noisy_kv_spec(self) -> ARDiffusionNoisyKVSpec:
        return ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=4,
            block_size=4,
            max_chunk_tokens=CHUNK_TOKENS,
            max_history_chunks=HISTORY,
        )

    @contextmanager
    def bind_ar_diffusion_cell_context(self, ctx) -> Iterator[None]:
        prev = self._chunk_ctx
        self._chunk_ctx = ctx
        try:
            yield
        finally:
            self._chunk_ctx = prev

    def forward(self, req: OmniDiffusionRequest) -> list[DiffusionOutput]:
        ctx = self._chunk_ctx
        if ctx is None:
            raise RuntimeError("cell context required")
        extra = dict(req.sampling_params.extra_args or {})
        session_id = str(extra["session_id"])
        chunk_base = int(extra.get("chunk_base", 0) or 0)
        num_chunks = int(extra.get("num_chunks", 1) or 1)
        prior = prior_history_in_pool(ctx.kv.cache, session_id, chunk_base=chunk_base)
        schedule = CellSchedule(
            chunks=num_chunks,
            num_denoise_steps=DENOISE,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=HISTORY,
            chunk_base=chunk_base,
        )
        plan = build_cell_plan(schedule, prior_history=prior)
        retain = set(prior) | clean_history_cells(schedule, num_blocks=1)
        ctx.enqueue(session_id, plan, chunk_tokens=CHUNK_TOKENS, retain=retain)
        adapter = _CountingAdapter()
        run_cell_pipeline(ctx=ctx, adapter=adapter)
        kept = clean_history_cells(schedule, num_blocks=1)
        ctx.kv.end_request(session_id, retain=kept)
        self.last_adapter = adapter
        self.executed_cells.extend(adapter.cells)
        return [DiffusionOutput(output={"cells": list(adapter.cells), "chunk_base": chunk_base})]


def _runner_with_pipeline(pipeline: SessionCellPipeline) -> ARDiffusionModelRunner:
    cache = NoisyKVCache(
        pipeline.ar_diffusion_noisy_kv_spec(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=1,
    )
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(
        max_num_seqs=1,
        dtype=torch.float32,
        ar_diffusion_stage_config={"stage_parallel_size": 1, "max_batch_size": 1},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True)
    runner.kv_cache = None
    runner.noisy_kv_cache = cache
    runner._ar_diffusion_capability = None
    runner._ar_diffusion_cell_capability = pipeline
    runner._ar_diffusion_kv_cache_spec = None
    runner._sessions = OrderedDict()
    runner._session_capacity = 0
    runner._perf_e2e_times = []
    runner._stepwise_chunk_started = {}
    runner._active_stepwise_request_id = None
    return runner


def test_e2e_one_session_multiple_requests_through_runner(monkeypatch):
    """Outer session sends 3 requests; pool history and chunk_base advance together."""
    pipeline = SessionCellPipeline()
    runner = _runner_with_pipeline(pipeline)
    session = OuterSession(session_id=SESSION_ID)

    def _parent_execute(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        outs = self.pipeline.forward(req)
        return outs[0]

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", _parent_execute)

    # Request 1: chunks [0, 1)
    out1 = runner.execute_model(session.next_request(num_chunks=1, request_id="r0"))
    assert out1.output["chunk_base"] == 0
    assert {c for c, _s, _b in out1.output["cells"]} == {0}
    assert runner.noisy_kv_cache.pool.has((SESSION_ID, 0, DENOISE, 0))
    assert session.chunk_base == 1

    # Request 2: chunks [1, 3) — two chunks, must see chunk 0 as prior history
    out2 = runner.execute_model(session.next_request(num_chunks=2, request_id="r1"))
    assert out2.output["chunk_base"] == 1
    cells2 = out2.output["cells"]
    assert {c for c, _s, _b in cells2} == {1, 2}
    # Trailing window H=1 keeps only the latest clean chunk.
    assert not runner.noisy_kv_cache.pool.has((SESSION_ID, 0, DENOISE, 0))
    assert not runner.noisy_kv_cache.pool.has((SESSION_ID, 1, DENOISE, 0))
    assert runner.noisy_kv_cache.pool.has((SESSION_ID, 2, DENOISE, 0))
    assert session.chunk_base == 3

    # Request 3: chunks [3, 4)
    out3 = runner.execute_model(session.next_request(num_chunks=1, request_id="r2"))
    assert out3.output["chunk_base"] == 3
    assert {c for c, _s, _b in out3.output["cells"]} == {3}
    assert runner.noisy_kv_cache.pool.has((SESSION_ID, 3, DENOISE, 0))
    assert not runner.noisy_kv_cache.pool.has((SESSION_ID, 2, DENOISE, 0))
    assert session.chunk_base == 4

    # Outer close releases the pool key.
    runner.release_noisy_kv(SESSION_ID)
    assert not any(key[0] == SESSION_ID for key in runner.noisy_kv_cache.pool.keys)
    # Legacy paged session table never touched.
    assert runner._sessions == OrderedDict()


def test_e2e_same_session_requests_see_growing_absolute_chunks(monkeypatch):
    pipeline = SessionCellPipeline()
    runner = _runner_with_pipeline(pipeline)
    session = OuterSession(session_id=SESSION_ID)
    monkeypatch.setattr(
        DiffusionModelRunner,
        "execute_model",
        lambda self, req, kv_prefetch_job=None, diffusion_kv_metadata=None: self.pipeline.forward(req)[0],
    )

    for i, n in enumerate((1, 1, 1)):
        out = runner.execute_model(session.next_request(num_chunks=n, request_id=f"step-{i}"))
        assert out.output["chunk_base"] == i
        assert all(cell[0] == i for cell in out.output["cells"])

    assert pipeline.executed_cells
    assert {c for c, _s, _b in pipeline.executed_cells} == {0, 1, 2}
    runner.release_noisy_kv(SESSION_ID)
