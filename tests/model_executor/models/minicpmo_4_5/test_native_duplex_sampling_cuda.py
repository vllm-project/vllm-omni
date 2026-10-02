# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The deferred Stage-0 duplex sampler on CUDA: no host sync, and the synchronous path's tokens and generator offsets."""

import pytest
import torch

from tests.model_executor.models.minicpmo_4_5.test_native_duplex_sampling import (
    _run_two_steps,
    _snapshot,
    _stage0_scenario,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _on_cuda(build):
    def build_cuda():
        model, states, md = build()
        md.generators = {row: torch.Generator(device="cuda").manual_seed(100 + row) for row in md.generators}
        md.temperature, md.top_k, md.top_p = (t.cuda() for t in (md.temperature, md.top_k, md.top_p))
        model._minicpmo45_duplex_row_sampling_host = {
            row: (float(md.temperature[row].cpu()), int(md.top_k[row].cpu()), float(md.top_p[row].cpu()))
            for row in range(6)
        }
        return model, states, md

    return build_cuda


@requires_cuda
@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_deferred_stage0_on_cuda_matches_the_synchronous_path(seed, all_greedy, monkeypatch):
    steps, build = _stage0_scenario(seed, all_greedy=all_greedy)
    steps = [logits.cuda() for logits in steps]
    build = _on_cuda(build)

    sync_model, sync_states, sync_md = build()
    monkeypatch.setattr(sync_model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    expected = _run_two_steps(sync_model, sync_md, steps)

    model, states, md = build()
    actual = _run_two_steps(model, md, steps)

    assert actual == expected
    got_states, _ = _snapshot(states, md)
    want_states, _ = _snapshot(sync_states, sync_md)
    assert got_states == want_states
    for row in md.generators:
        assert md.generators[row].get_offset() == sync_md.generators[row].get_offset()


@requires_cuda
@pytest.mark.parametrize("all_greedy", [False, True])
def test_deferred_stage0_samples_without_a_host_sync(all_greedy):
    steps, build = _stage0_scenario(0, all_greedy=all_greedy)
    model, states, md = _on_cuda(build)()
    logits = steps[0].cuda()
    # Warm the cached device tables and the pinned allocator outside the guard.
    model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
    model._commit_minicpmo45_duplex_pending_samples()
    torch.accelerator.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        out = model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert model._minicpmo45_duplex_pending_samples is not None
    assert out.sampled_token_ids.is_cuda
