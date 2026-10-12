# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The request's seed must reach GR00T's flow-matching noise.

The runner builds ``sampling_params.generator`` from ``sampling_params.seed``, the pipeline hands
that generator to the policy, and the action head draws its initial noise from it, so the same
seed reproduces the same action chunk and different seeds give independent ones. These tests pin
both ends of that plumbing.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from vllm_omni.diffusion.models.gr00t import pipeline_gr00t
from vllm_omni.diffusion.models.gr00t.configs.gr00t_n1d7 import Gr00tN1d7Config
from vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 import Gr00tN1d7ActionHead
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROBOT_OBS = {
    "images": {"cam": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)},
    "state": {"joint": np.zeros((1, 1, 2), dtype=np.float32)},
    "prompt": "pick the cube",
}


class _OptionsRecordingPolicy:
    """Stand-in for ``Gr00tPolicy``; the pipeline only needs ``language_key`` and ``get_action``."""

    def __init__(self, *, model_path: str, embodiment_tag: str, device: str, strict: bool) -> None:
        del model_path, embodiment_tag, device, strict  # the pipeline's constructor contract; unused here
        self.language_key = "annotation.language.language_instruction"
        self.seen_options: dict[str, object] | None = None

    def get_action(self, obs: dict[str, object], options: dict[str, object] | None = None):
        self.seen_options = options
        return {"arm": np.zeros((1, 2, 2), dtype=np.float32)}, {}


@pytest.fixture
def pipeline(monkeypatch) -> Gr00tN1d7Pipeline:
    monkeypatch.setattr(pipeline_gr00t, "Gr00tPolicy", _OptionsRecordingPolicy)
    od_config = SimpleNamespace(
        model="nvidia/GR00T-N1.7-3B",
        model_config={"embodiment_tag": "LIBERO_PANDA", "strict": False},
        custom_pipeline_args={},
    )
    return Gr00tN1d7Pipeline(od_config=od_config)


def _request(**sampling_kwargs) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="pick",
        request_id="req",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"robot_obs": ROBOT_OBS}, **sampling_kwargs),
    )


def test_runner_generator_is_passed_through_untouched(pipeline):
    """In serving the runner has already built ``sampling_params.generator``; the head must get that object."""
    generator = torch.Generator().manual_seed(99)

    assert pipeline.forward(_request(generator=generator)).error is None

    assert pipeline.policy.seen_options["generator"] is generator


def test_seed_without_generator_builds_one_on_the_pipeline_device(pipeline):
    """Callers that bypass the runner (direct use, tests) still get the seed honoured."""
    pipeline.forward(_request(seed=1234))

    generator = pipeline.policy.seen_options["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 1234
    assert generator.device.type == pipeline.device


def test_auto_assigned_seed_reaches_the_head(pipeline):
    """``OmniDiffusionRequest`` seeds every request, so the head is never left on the global RNG."""
    req = _request()
    assert req.sampling_params.seed is not None

    pipeline.forward(req)

    assert pipeline.policy.seen_options["generator"].initial_seed() == req.sampling_params.seed


def _captured_warnings(monkeypatch) -> list[tuple]:
    """vLLM loggers do not propagate to the root logger, so ``caplog`` cannot see them; record calls instead."""
    captured: list[tuple] = []
    monkeypatch.setattr(pipeline_gr00t.logger, "warning", lambda *args, **kwargs: captured.append(args))
    return captured


def test_unexpected_generator_form_warns_and_falls_back_to_the_seed(pipeline, monkeypatch):
    """GR00T draws one noise tensor per request, so a list of several generators cannot be honoured."""
    warnings = _captured_warnings(monkeypatch)
    generators = [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)]

    pipeline.forward(_request(generator=generators, seed=7))

    assert len(warnings) == 1
    assert "torch.Generator" in warnings[0][0]
    assert pipeline.policy.seen_options["generator"].initial_seed() == 7


def test_unexpected_generator_form_without_a_seed_warns_and_leaves_the_global_rng(pipeline, monkeypatch):
    warnings = _captured_warnings(monkeypatch)
    generators = [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)]

    pipeline.forward(_request(generator=generators))

    assert len(warnings) == 1
    assert pipeline.policy.seen_options is None


def _noise_only_head(action_horizon: int, action_dim: int) -> Gr00tN1d7ActionHead:
    """An action head whose single denoising step is the identity, so ``action_pred`` is the initial noise."""
    config = Gr00tN1d7Config(
        max_action_dim=action_dim,
        action_horizon=action_horizon,
        num_inference_timesteps=1,
        add_pos_embed=False,
        use_alternate_vl_dit=False,
    )
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(head)
    head.config = config
    head.action_dim = config.max_action_dim
    head.action_horizon = config.action_horizon
    head.num_inference_timesteps = config.num_inference_timesteps
    head.num_timestep_buckets = config.num_timestep_buckets
    head.action_encoder = lambda actions, timesteps, embodiment_id: actions
    head.model = lambda hidden_states, encoder_hidden_states, timestep: hidden_states
    head.action_decoder = lambda hidden_states, embodiment_id: torch.zeros_like(hidden_states)
    return head


def _predict(head: Gr00tN1d7ActionHead, options) -> torch.Tensor:
    batch, seq, dim = 2, 3, head.action_dim
    return head.get_action_with_features(
        backbone_features=torch.zeros(batch, seq, dim),
        state_features=torch.zeros(batch, 1, dim),
        embodiment_id=torch.zeros(batch, dtype=torch.long),
        backbone_output=BatchFeature(data={}),
        action_input=BatchFeature(data={}),
        options=options,
    )["action_pred"]


def test_head_draws_its_noise_from_the_request_generator():
    head = _noise_only_head(action_horizon=4, action_dim=3)
    expected = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(5))

    assert torch.equal(_predict(head, {"generator": torch.Generator().manual_seed(5)}), expected)
    assert not torch.equal(_predict(head, {"generator": torch.Generator().manual_seed(6)}), expected)


def test_head_without_generator_stays_on_the_global_rng():
    """Upstream Isaac-GR00T parity: no generator means a bare ``torch.randn`` on the global RNG."""
    head = _noise_only_head(action_horizon=4, action_dim=3)

    torch.manual_seed(5)
    expected = torch.randn(2, 4, 3)
    torch.manual_seed(5)

    assert torch.equal(_predict(head, None), expected)
