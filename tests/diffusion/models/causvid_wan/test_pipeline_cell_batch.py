# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid pipeline admits compatible requests into one cell pipeline run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.cell_batch import BatchedCellAdapter
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_cell(*, seed: int) -> WanCell:
    tf = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        is_layer_group_first=True,
        is_layer_group_last=True,
        dim=4,
        prepare_cell=lambda *_a, **_k: None,
        parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
    )
    cell = WanCell.__new__(WanCell)
    cell.transformer = tf
    cell.sampler = SimpleNamespace(timesteps=[1.0], advance=lambda *a, **k: a[1])
    cell.prompt_embeds = torch.zeros(1, 2, 4)
    cell.latent_shape = (1, 4, 1, 2, 2)
    cell.seed = seed
    cell.device = torch.device("cpu")
    cell.dtype = torch.float32
    cell.on_finished = None
    cell.num_denoise_steps = 1
    cell.finished = {}
    cell._live = {}
    cell._stash = {}
    cell._last_meta = None
    cell._timesteps = [torch.tensor(1.0)]
    cell._timestep_clean = torch.zeros(())
    return cell


def test_pipeline_forward_runs_one_cell_pipeline_for_batch(monkeypatch):
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline._chunk_ctx = SimpleNamespace()
    calls: list[object] = []
    cells = {
        "A": _fake_cell(seed=1),
        "B": _fake_cell(seed=2),
    }

    def _admit(req: OmniDiffusionRequest):
        rid = req.request_id
        return cells[rid], rid, False, None

    monkeypatch.setattr(pipeline, "_execution_context", lambda: pipeline._chunk_ctx)
    monkeypatch.setattr(pipeline, "admit_cell_request", _admit)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.run_cell_pipeline",
        lambda **kwargs: calls.append(kwargs["adapter"]),
    )
    emitted: list[str] = []

    def _emit(adapter, kv_id, stream_job=None):
        del adapter, stream_job
        emitted.append(kv_id)
        return DiffusionOutput(finished=True)

    monkeypatch.setattr(pipeline, "emit_cell_output", _emit)

    batch = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(prompt="a", request_id="A", sampling_params=OmniDiffusionSamplingParams()),
            OmniDiffusionRequest(prompt="b", request_id="B", sampling_params=OmniDiffusionSamplingParams()),
        ]
    )
    outs = pipeline.forward(batch)
    assert len(calls) == 1
    assert isinstance(calls[0], BatchedCellAdapter)
    assert set(calls[0].members) == {"A", "B"}
    assert calls[0]._pack_batch is WanCell.pack_compute_batch
    assert emitted == ["A", "B"]
    assert len(outs) == 2
