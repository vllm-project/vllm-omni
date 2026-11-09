# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.ar_diffusion.test_capability_runner import make_runner
from tests.diffusion.models.hy_nomas.test_ar import PipelineWithCPUComponents, batch, execute

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("budget", [9, 18, 36, 72])
def test_recent_kv_matches_pinned_reference_eviction(budget, monkeypatch):
    from hymm.models.dit.modules.swa_cache import append_dense_record, commit_window

    pipeline = PipelineWithCPUComponents()
    pipeline.initialize_ar(
        {
            "ar_height": 32,
            "ar_width": 32,
            "ar_chunk_latent_frames": 3,
            "ar_history_mode": "recent",
            "ar_history_latent_frames": budget,
        }
    )
    monkeypatch.setattr(
        "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform", SimpleNamespace(synchronize=lambda: None)
    )
    runner = make_runner(pipeline)
    reference = [
        {
            "k": torch.zeros(1, budget, 2, 4),
            "v": torch.zeros(1, budget, 2, 4),
            "swa_capacity": budget,
        }
        for _ in range(2)
    ]
    masks = [torch.zeros(1, budget, dtype=torch.bool) for _ in range(2)]
    valid = 0
    try:
        for index in range(45):
            execute(runner, batch(index=index))
            state = runner._sessions["s"].segmented("cond")
            state.prepare_layer_views(1, 0)
            empty = torch.zeros(0, 2, 4)
            current = torch.zeros(1, 2, 4)
            for layer, (cache, mask) in enumerate(zip(reference, masks)):
                rows = torch.full((1, 7, 2, 4), float(layer + index * 3 + 1))
                append_dense_record(cache, valid, 3, 4, 1)
                commit_window(cache, mask, rows, -rows, torch.tensor([[1, 1, 1, 1, 1, 1, 0]]).bool(), valid)
                key, value = state.assemble(layer, empty, empty, current, current)
                count = cache["swa_valid_len"]
                assert state.history_tokens == count
                torch.testing.assert_close(key[:count], cache["k"][0, :count], rtol=0, atol=0)
                torch.testing.assert_close(value[:count], cache["v"][0, :count], rtol=0, atol=0)
                torch.testing.assert_close(pipeline._ar_sessions["s"].history_mask, mask[:, :count])
            valid = count
    finally:
        runner.close_session("s")
