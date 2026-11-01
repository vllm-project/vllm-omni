# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Alpamayo 2 Super expert stage: action space and batched trajectory sampling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.alpamayo2_super.action_expert import (
    Alpamayo2SuperActionExpert,
    UnicycleAccelCurvatureActionSpace,
    _velocity_from_history,
)
from vllm_omni.diffusion.models.alpamayo2_super.pipeline_alpamayo2_super import (
    ACTIONS_KEY,
    Alpamayo2SuperPipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_zero_controls_keep_constant_speed_straight():
    action_space = UnicycleAccelCurvatureActionSpace(dt=0.1, n_waypoints=64)
    speed = 10.0
    history_xyz = torch.zeros(1, 16, 3)
    history_xyz[0, :, 0] = speed * 0.1 * torch.arange(-15.0, 1.0)
    history_rot = torch.eye(3).expand(1, 16, 3, 3)
    xyz, rot = action_space.action_to_traj(torch.zeros(1, 64, 2), history_xyz, history_rot)
    assert xyz.shape == (1, 64, 3) and rot.shape == (1, 64, 3, 3)
    expected_x = speed * 0.1 * torch.arange(1.0, 65.0)
    torch.testing.assert_close(xyz[0, :, 0], expected_x, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(xyz[0, :, 1], torch.zeros(64), rtol=0, atol=1e-4)


def test_velocity_solver_matches_direct_least_squares():
    # Reference formulation: minimize sum_t |v_t u_t + v_{t+1} u_{t+1} - 2 dp_t / dt|^2
    # + lam / dt^6 * |D3 v|^2 + ridge * |v|^2, solved densely.
    generator = torch.Generator().manual_seed(0)
    n, dt, lam, ridge = 15, 0.1, 1e-6, 1e-4
    theta = torch.cumsum(0.05 * torch.randn(n + 1, generator=generator, dtype=torch.float64), dim=0)
    dxy = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    a = torch.zeros(2 * n, n + 1, dtype=torch.float64)
    for t in range(n):
        for row, trig in ((2 * t, torch.cos), (2 * t + 1, torch.sin)):
            a[row, t] = trig(theta[t])
            a[row, t + 1] = trig(theta[t + 1])
    d3 = torch.zeros(n - 2, n + 1, dtype=torch.float64)
    for row in range(n - 2):
        d3[row, row : row + 4] = torch.tensor([-1.0, 3.0, -3.0, 1.0], dtype=torch.float64)
    lhs = a.T @ a + lam / dt**6 * d3.T @ d3 + ridge * torch.eye(n + 1, dtype=torch.float64)
    expected = torch.linalg.solve(lhs, a.T @ (2 / dt * dxy).flatten())
    actual = _velocity_from_history(dxy, theta, dt=dt, v_lambda=lam, v_ridge=ridge)
    torch.testing.assert_close(actual, expected)


_TINY_EXPERT = {
    "llm_config": {
        "model_type": "qwen3_vl_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 16,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"},
    },
    "action_in_proj_cfg": {"hidden_size": 16, "num_enc_layers": 2, "num_fourier_feats": 4},
    "action_space_cfg": {"n_waypoints": 64},
}


def _make_pipeline() -> Alpamayo2SuperPipeline:
    torch.manual_seed(0)
    pipeline = Alpamayo2SuperPipeline.__new__(Alpamayo2SuperPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.history_steps = 16
    pipeline.expert = Alpamayo2SuperActionExpert(_TINY_EXPERT).eval()
    return pipeline


def _expert_request(prefix_len: int, seed: int) -> OmniDiffusionRequest:
    generator = torch.Generator().manual_seed(prefix_len)
    keys = [torch.randn(prefix_len, 2, 8, generator=generator) for _ in range(2)]
    values = [torch.randn(prefix_len, 2, 8, generator=generator) for _ in range(2)]
    history_xyz = torch.zeros(16, 3)
    history_xyz[:, 0] = torch.arange(-15.0, 1.0)
    params = OmniDiffusionSamplingParams(
        # The KV manager hands received caches over in this form.
        past_key_values=SimpleNamespace(key_cache=keys, value_cache=values),
        kv_metadata={"seq_len": prefix_len, "prefix_len": prefix_len, "trigger_position": [prefix_len - 1] * 3},
        extra_args={
            "robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": [torch.eye(3).tolist()] * 16}
        },
        seed=seed,
        num_inference_steps=2,
    )
    return OmniDiffusionRequest(prompt="", sampling_params=params, request_id=f"req-{prefix_len}-{seed}")


def _run(pipeline: Alpamayo2SuperPipeline, *requests: OmniDiffusionRequest):
    return pipeline(DiffusionRequestBatch(requests=list(requests)))


def test_expert_batch_matches_requests_run_alone():
    pipeline = _make_pipeline()
    lengths_and_seeds = [(5, 1), (9, 2)]
    batched = _run(pipeline, *[_expert_request(*args) for args in lengths_and_seeds])
    for output, args in zip(batched, lengths_and_seeds):
        (alone,) = _run(pipeline, _expert_request(*args))
        assert output.output[ACTIONS_KEY].shape == (64, 3)
        torch.testing.assert_close(output.output[ACTIONS_KEY], alone.output[ACTIONS_KEY], rtol=1e-4, atol=1e-4)
    assert not torch.equal(batched[0].output[ACTIONS_KEY], batched[1].output[ACTIONS_KEY])


def test_expert_reports_missing_kv_prefix():
    pipeline = _make_pipeline()
    request = _expert_request(5, 1)
    request.sampling_params.past_key_values = None
    (output,) = _run(pipeline, request)
    assert output.error is not None and "KV prefix" in output.error


def test_expert_reports_reasoning_without_trajectory_start_and_serves_the_rest():
    pipeline = _make_pipeline()
    truncated = _expert_request(5, 1)
    truncated.sampling_params.kv_metadata = {"seq_len": 5}
    failed, served = _run(pipeline, truncated, _expert_request(9, 2))
    assert "traj_future_start" in failed.error
    assert served.error is None and served.output[ACTIONS_KEY].shape == (64, 3)


def test_expert_rejects_invalid_steps():
    pipeline = _make_pipeline()
    request = _expert_request(5, 1)
    request.sampling_params.num_inference_steps = -1
    (output,) = _run(pipeline, request)
    assert "num_inference_steps" in output.error


def test_expert_kv_length_mismatch_fails_only_that_request():
    pipeline = _make_pipeline()
    mismatched = _expert_request(5, 1)
    mismatched.sampling_params.kv_metadata["seq_len"] = 4
    failed, served = _run(pipeline, mismatched, _expert_request(9, 2))
    assert "KV transfer covers 4 tokens, expected 5" in failed.error
    assert served.error is None and served.output[ACTIONS_KEY].shape == (64, 3)


def test_expert_reports_missing_robot_obs_and_serves_the_rest():
    pipeline = _make_pipeline()
    no_history = _expert_request(5, 1)
    no_history.sampling_params.extra_args = {}
    failed, served = _run(pipeline, no_history, _expert_request(9, 2))
    assert "robot_obs" in failed.error
    assert served.error is None and served.output[ACTIONS_KEY].shape == (64, 3)


def test_expert_without_seed_or_with_generator_list():
    pipeline = _make_pipeline()
    unseeded = _expert_request(5, 1)
    unseeded.sampling_params.seed = None
    (output,) = _run(pipeline, unseeded)
    assert output.error is None and output.output[ACTIONS_KEY].isfinite().all()

    listed = _expert_request(5, 1)
    listed.sampling_params.seed = None
    listed.sampling_params.generator = [torch.Generator().manual_seed(1)]
    (from_list,) = _run(pipeline, listed)
    (seeded,) = _run(pipeline, _expert_request(5, 1))
    torch.testing.assert_close(from_list.output[ACTIONS_KEY], seeded.output[ACTIONS_KEY])


def _set_history(request: OmniDiffusionRequest, key: str, value) -> OmniDiffusionRequest:
    observation = request.sampling_params.extra_args["robot_obs"]
    if value is None:
        del observation[key]
    else:
        observation[key] = value
    return request


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("ego_history_rot", None, "missing or not numeric"),
        ("ego_history_xyz", [[0.0, 0.0, 0.0]] * 15, "must have shape (16, 3)"),
        ("ego_history_rot", [[1.0, 0.0, 0.0]] * 16, "must have shape (16, 3, 3)"),
        ("ego_history_xyz", "not a history", "missing or not numeric"),
        ("ego_history_xyz", [[float("nan"), 0.0, 0.0]] * 16, "non-finite"),
    ],
)
def test_expert_rejects_malformed_history_and_serves_the_rest(key, value, message):
    pipeline = _make_pipeline()
    malformed = _set_history(_expert_request(5, 1), key, value)
    failed, served = _run(pipeline, malformed, _expert_request(9, 2))
    assert key in failed.error and message in failed.error
    assert served.error is None and served.output[ACTIONS_KEY].shape == (64, 3)


def test_expert_accepts_history_with_a_leading_batch_dimension():
    pipeline = _make_pipeline()
    request = _expert_request(5, 1)
    observation = request.sampling_params.extra_args["robot_obs"]
    batched = _expert_request(5, 1)
    batched.sampling_params.extra_args["robot_obs"] = {key: [value] for key, value in observation.items()}
    (plain,) = _run(pipeline, request)
    (with_batch_dim,) = _run(pipeline, batched)
    torch.testing.assert_close(with_batch_dim.output[ACTIONS_KEY], plain.output[ACTIONS_KEY])


def test_pipeline_reads_config_and_weights_at_the_configured_revision(monkeypatch):
    from vllm_omni.diffusion.models.alpamayo2_super import pipeline_alpamayo2_super as module

    calls = []

    def from_pretrained(model, revision=None):
        calls.append(("config", model, revision))
        return SimpleNamespace(expert_config=_TINY_EXPERT, tokens_per_history_traj=45)

    def get_hf_file_to_dict(name, model, revision):
        calls.append(("index", model, revision))
        return {"weight_map": {"expert.a": "model-00014.safetensors", "vlm.b": "model-00001.safetensors"}}

    monkeypatch.setattr(module.Alpamayo2SuperConfig, "from_pretrained", from_pretrained)
    monkeypatch.setattr(module, "get_hf_file_to_dict", get_hf_file_to_dict)
    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))

    pipeline = Alpamayo2SuperPipeline(od_config=SimpleNamespace(model="repo", revision="pinned", dtype=torch.float32))

    assert calls == [("config", "repo", "pinned"), ("index", "repo", "pinned")]
    assert [(source.revision, source.allow_patterns_overrides) for source in pipeline.weights_sources] == [
        ("pinned", ["model-00014.safetensors"])
    ]
    assert pipeline.history_steps == 16
