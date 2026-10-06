# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA graph tree replay at the shared sequential CFG model boundary."""

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class ReferenceModel(nn.Module):
    """Declared independent affine model, isolating graph output lifetime and CFG."""

    def forward(self, x, t, cap_feats):
        return [sample / 8 + condition.mean() + t[i] / 4 for i, (sample, condition) in enumerate(zip(x, cap_feats))], {}


class GraphBoundaryObserver(nn.Module):
    """Observe the real compiled invocation without changing its result."""

    def __init__(self, compiled, events):
        super().__init__()
        self.compiled = compiled
        self.events = events

    def forward(self, *args, **kwargs):
        self.events.append(("forward", get_forward_context().cfg_branch))
        return self.compiled(*args, **kwargs)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sequential_cfg_replay_preserves_both_branch_predictions(monkeypatch):
    """Scenario: sequential positive/negative calls reuse an actual compiled graph.

    Input source: ZImage prepare_latents and per-frame model/CFG kwargs contract.
    Why valid: one image with two caption branches and the same denoise timestep.
    Expected source: affine model equation and Z-Image positive+scale*(pos-neg).
    Bug prevented: a captured output overwritten by the next branch/replay, or
    missing CUDA graph step boundaries. The real CFG mixin and graph manager run;
    the affine model isolates graph lifetime, not real DiT numerical correctness.
    """
    pipe = object.__new__(ZImagePipeline)
    nn.Module.__init__(pipe)
    pipe.vae_scale_factor = 2
    pipe._uses_cudagraph_trees = True
    events = []
    real_marker = torch.compiler.cudagraph_mark_step_begin

    def observe_marker():
        events.append(("marker", get_forward_context().cfg_branch))
        real_marker()

    # Reviewer contract: every positive/negative invocation starts a new step.
    # Observation delegates to the actual marker; graph capture is still real.
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", observe_marker)
    compiled = torch.compile(ReferenceModel().cuda(), mode="reduce-overhead", fullgraph=True)
    pipe.transformer = GraphBoundaryObserver(compiled, events)
    latents = pipe.prepare_latents(
        1, 1, 8, 12, torch.float32, torch.device("cuda"), torch.Generator("cuda").manual_seed(73)
    )
    samples = list(latents.unsqueeze(2).unbind(0))
    timestep = torch.full((1,), 0.4, device="cuda")
    positive = {
        "x": samples,
        "t": timestep,
        "cap_feats": [torch.full((2, 2), 2.0, device="cuda")],
        "_cfg_branch": "positive",
    }
    negative = {**positive, "cap_feats": [torch.ones((2, 2), device="cuda")], "_cfg_branch": "negative"}
    # Independent equation: base + 2 + .4/4 + 2*(2-1).
    expected = latents.unsqueeze(2) / 8 + 4.1
    with set_forward_context():
        get_forward_context().cfg_branch = "outer"
        for _ in range(5):
            events.clear()
            actual = pipe.predict_noise_maybe_with_cfg(
                do_true_cfg=True,
                true_cfg_scale=2.0,
                positive_kwargs=positive,
                negative_kwargs=negative,
                cfg_normalize=False,
            )
            torch.testing.assert_close(actual, expected)
            assert get_forward_context().cfg_branch == "outer"
            assert events == [
                ("marker", "positive"),
                ("forward", "positive"),
                ("marker", "negative"),
                ("forward", "negative"),
            ]
    # Prove graph capture/replay was actually active rather than silently skipped.
    from torch._inductor.cudagraph_trees import get_manager

    manager = get_manager(0, create_if_none_exists=False)
    assert manager is not None and manager.current_node is not None
