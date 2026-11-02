# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for CFG-branch-aware TeaCache coefficient collection."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    DataCollectionHook,
    calculate_relative_l1,
    estimate_teacache_coefficients,
    iter_same_branch_trajectories,
)
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TinyModule(nn.Module):
    def forward(self, *args: Any, **kwargs: Any):
        raise AssertionError("Hook should intercept forward")


def _make_context(
    *,
    branch: str,
    step: int,
    bypass: bool = False,
) -> CacheContext:
    # Distinct magnitudes so cross-branch adjacent pairs would pollute a fit.
    scale = 1.0 if branch == "positive" else 10.0
    modulated = torch.full((1, 2), float(scale * (step + 1)), dtype=torch.float32)
    hidden = modulated.clone()

    def run_blocks():
        return (hidden + 0.1,)

    def postprocess(h):
        return h

    return CacheContext(
        modulated_input=modulated,
        hidden_states=hidden,
        encoder_hidden_states=None,
        temb=torch.zeros(1),
        run_transformer_blocks=run_blocks,
        postprocess=postprocess,
        extra_states={"teacache_branch": branch, "teacache_bypass": bypass},
    )


@pytest.fixture
def branch_hook(monkeypatch):
    # Bind a real registered type, then replace the extractor with a scripted one.
    hook = DataCollectionHook("ZImageTransformer2DModel")
    script: list[CacheContext] = []

    def fake_extractor(module, *args, **kwargs):
        assert script, "unexpected extractor call"
        return script.pop(0)

    monkeypatch.setattr(hook, "extractor_fn", fake_extractor)
    return hook, script


def test_collection_groups_alternating_cfg_branches(branch_hook):
    """Alternating pos/neg forwards must not form cross-branch step pairs."""
    hook, script = branch_hook
    hook.start_collection()

    # Deterministic CFG sequence: + - + - (two steps per branch).
    for step in range(2):
        for branch in ("positive", "negative"):
            script.append(_make_context(branch=branch, step=step))
            hook.new_forward(_TinyModule())

    assert script == []
    by_branch = hook.stop_collection()
    assert set(by_branch) == {"positive", "negative"}
    assert len(by_branch["positive"]) == 2
    assert len(by_branch["negative"]) == 2

    trajectories = iter_same_branch_trajectories([by_branch])
    assert len(trajectories) == 2

    # Only one consecutive pair per branch enters the fit (step0->step1).
    pairs = []
    for traj in trajectories:
        assert len(traj) == 2
        pairs.append(calculate_relative_l1(traj[0][0], traj[1][0]))
    assert len(pairs) == 2

    # A flat alternating view would have three adjacent pairs and include
    # cross-branch jumps; the branch-aware path must not.
    flat = hook.current_trajectory
    assert len(flat) == 4
    flat_pairs = [calculate_relative_l1(flat[i][0], flat[i + 1][0]) for i in range(3)]
    assert len(flat_pairs) == 3

    coeffs = estimate_teacache_coefficients([by_branch], poly_order=1)
    assert len(coeffs) == 2


def test_collection_skips_teacache_bypass(branch_hook):
    hook, script = branch_hook
    hook.start_collection()
    script.append(_make_context(branch="positive", step=0, bypass=True))
    script.append(_make_context(branch="positive", step=1, bypass=False))
    hook.new_forward(_TinyModule())
    hook.new_forward(_TinyModule())
    by_branch = hook.stop_collection()
    assert list(by_branch) == ["positive"]
    assert len(by_branch["positive"]) == 1
