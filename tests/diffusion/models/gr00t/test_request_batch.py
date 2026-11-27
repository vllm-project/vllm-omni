# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GR00T serves a scheduler wave of requests with one policy call.

Single-observation requests are concatenated along the batch axis, each request keeps
its own noise generator, and actions are split back in request order. Requests that
cannot be merged, or a batched call that fails, fall back to one call per request so a
bad request only fails itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.gr00t import pipeline_gr00t
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LANGUAGE_KEY = "annotation.language.language_instruction"


class _BatchRecordingPolicy:
    """Stand-in for ``Gr00tPolicy`` whose action echoes each sample's last state value."""

    def __init__(self, *, model_path: str, embodiment_tag: str, device: str, strict: bool) -> None:
        del model_path, embodiment_tag, device
        self.strict = strict
        self.language_key = LANGUAGE_KEY
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        self.fail_batched = False
        self.draw_noise = False
        self.noise: list[list[torch.Tensor]] = []
        self.reset_calls = 0

    def check_observation(self, obs: dict[str, Any]) -> None:
        assert obs["state"]["joint"].dtype == np.float32, "state must be float32"

    def get_action(self, obs: dict[str, Any], options: dict[str, Any] | None = None):
        self.calls.append((obs, options))
        if self.strict:
            self.check_observation(obs)
        if self.draw_noise:
            # Like the action head, draw each sample's noise before anything can fail.
            generator = (options or {}).get("generator")
            generators = generator if isinstance(generator, list) else [generator]
            self.noise.append([torch.randn(1, generator=g) for g in generators])
        state = obs["state"]["joint"]
        if self.fail_batched and state.shape[0] > 1:
            raise RuntimeError("batched call failed")
        if np.isnan(state).any():
            raise ValueError("NaN state")
        return {"arm": np.repeat(state[:, -1:, :1], 2, axis=1)}, {}

    def reset(self) -> dict[str, Any]:
        self.reset_calls += 1
        return {}


@pytest.fixture
def pipeline(monkeypatch) -> Gr00tN1d7Pipeline:
    monkeypatch.setattr(pipeline_gr00t, "Gr00tPolicy", _BatchRecordingPolicy)
    # The batching logic is device independent; keep generators on CPU so no GPU is needed.
    monkeypatch.setattr(pipeline_gr00t, "get_local_device", lambda: torch.device("cpu"))
    od_config = OmniDiffusionConfig(
        model="nvidia/GR00T-N1.7-3B",
        model_config={"embodiment_tag": "LIBERO_PANDA", "strict": False},
    )
    return Gr00tN1d7Pipeline(od_config=od_config)


def _robot_obs(value: float, *, camera: str = "cam") -> dict[str, Any]:
    return {
        "images": {camera: np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)},
        "state": {"joint": np.full((1, 1, 2), value, dtype=np.float32)},
        "prompt": f"task {value}",
    }


def _request(request_id: str, robot_obs: Any = None, *, seed: int | None = None, **extra) -> OmniDiffusionRequest:
    extra_args = dict(extra)
    if robot_obs is not None:
        extra_args["robot_obs"] = robot_obs
    return OmniDiffusionRequest(
        prompt="pick",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(extra_args=extra_args, seed=seed),
    )


def _arm(output) -> float:
    assert output.error is None, output.error
    arm = output.output["actions"]["arm"]
    assert arm.shape == (1, 2, 1)
    assert arm.dtype == np.float32
    return float(arm[0, 0, 0])


def test_pipeline_declares_request_batch_support():
    assert Gr00tN1d7Pipeline.supports_request_batch is True


def test_wave_runs_one_policy_call_and_returns_actions_in_request_order(pipeline):
    wave = DiffusionRequestBatch(requests=[_request(f"r{i}", _robot_obs(float(i))) for i in range(3)])

    outputs = pipeline.forward(wave)

    assert [_arm(output) for output in outputs] == [0.0, 1.0, 2.0]
    assert len(pipeline.policy.calls) == 1
    merged, _ = pipeline.policy.calls[0]
    assert merged["video"]["cam"].shape[0] == 3
    assert merged["state"]["joint"].shape == (3, 1, 2)
    assert merged["language"] == {LANGUAGE_KEY: [["task 0.0"], ["task 1.0"], ["task 2.0"]]}


def test_each_request_keeps_its_own_generator(pipeline):
    wave = DiffusionRequestBatch(requests=[_request("a", _robot_obs(0.0)), _request("b", _robot_obs(1.0))])
    generators = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(22)]
    for request, generator in zip(wave.requests, generators):
        request.sampling_params.generator = generator

    pipeline.forward(wave)

    _, options = pipeline.policy.calls[0]
    assert options is not None
    assert len(options["generator"]) == 2
    assert all(seen is expected for seen, expected in zip(options["generator"], generators))


def test_malformed_requests_fail_alone_and_the_rest_are_batched(pipeline):
    wave = DiffusionRequestBatch(
        requests=[
            _request("ok-1", _robot_obs(1.0)),
            _request("missing"),
            _request("not-a-dict", ["bad"]),
            _request("ok-2", _robot_obs(2.0)),
        ]
    )

    outputs = pipeline.forward(wave)

    assert _arm(outputs[0]) == 1.0
    assert "expects sampling_params.extra_args['robot_obs']" in outputs[1].error
    assert "robot_obs must be a dict" in outputs[2].error
    assert _arm(outputs[3]) == 2.0
    assert len(pipeline.policy.calls) == 1


def test_observations_that_cannot_be_merged_are_served_one_at_a_time(pipeline):
    wave = DiffusionRequestBatch(
        requests=[_request("a", _robot_obs(1.0, camera="left")), _request("b", _robot_obs(2.0, camera="right"))]
    )

    outputs = pipeline.forward(wave)

    assert [_arm(output) for output in outputs] == [1.0, 2.0]
    assert [obs["state"]["joint"].shape[0] for obs, _ in pipeline.policy.calls] == [1, 1]


def test_multi_sample_observation_is_not_merged(pipeline):
    two_samples = _robot_obs(1.0)
    two_samples["images"]["cam"] = np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
    two_samples["state"]["joint"] = np.ones((2, 1, 2), dtype=np.float32)
    wave = DiffusionRequestBatch(requests=[_request("a", two_samples), _request("b", _robot_obs(2.0))])

    outputs = pipeline.forward(wave)

    assert outputs[0].output["actions"]["arm"].shape == (2, 2, 1)
    assert _arm(outputs[1]) == 2.0
    assert len(pipeline.policy.calls) == 2


def test_failed_batched_call_falls_back_and_only_the_bad_request_errors(pipeline):
    pipeline.policy.fail_batched = True
    wave = DiffusionRequestBatch(
        requests=[_request("good", _robot_obs(1.0)), _request("nan", _robot_obs(float("nan")))]
    )

    outputs = pipeline.forward(wave)

    assert _arm(outputs[0]) == 1.0
    assert "GR00T policy inference failed: NaN state" in outputs[1].error
    assert [obs["state"]["joint"].shape[0] for obs, _ in pipeline.policy.calls] == [2, 1, 1]


@pytest.mark.parametrize("bad_value", [None, 0.5])
def test_nested_value_without_a_batch_axis_fails_alone(pipeline, bad_value):
    bad = _robot_obs(2.0)
    bad["state"]["joint"] = bad_value
    wave = DiffusionRequestBatch(
        requests=[_request("ok-1", _robot_obs(1.0)), _request("bad", bad), _request("ok-2", _robot_obs(3.0))]
    )

    outputs = pipeline.forward(wave)

    assert _arm(outputs[0]) == 1.0
    assert "GR00T policy inference failed" in outputs[1].error
    assert _arm(outputs[2]) == 3.0
    assert len(pipeline.policy.calls) == 3


def test_retry_after_a_failed_batch_draws_each_requests_solo_noise(pipeline):
    pipeline.policy.fail_batched = True
    pipeline.policy.draw_noise = True
    seeds = (11, 22)
    wave = DiffusionRequestBatch(requests=[_request("a", _robot_obs(1.0)), _request("b", _robot_obs(2.0))])
    for request, seed in zip(wave.requests, seeds):
        request.sampling_params.generator = torch.Generator().manual_seed(seed)

    outputs = pipeline.forward(wave)

    assert [_arm(output) for output in outputs] == [1.0, 2.0]
    batched, *retries = pipeline.policy.noise
    assert len(batched) == 2
    for (drawn,), seed in zip(retries, seeds):
        assert torch.equal(drawn, torch.randn(1, generator=torch.Generator().manual_seed(seed)))


def test_strict_mode_rejects_an_invalid_request_alone_and_batches_the_rest(pipeline):
    pipeline.policy.strict = True
    bad = _robot_obs(2.0)
    bad["state"]["joint"] = bad["state"]["joint"].astype(np.float64)
    wave = DiffusionRequestBatch(
        requests=[_request("ok-1", _robot_obs(1.0)), _request("bad", bad), _request("ok-2", _robot_obs(3.0))]
    )

    outputs = pipeline.forward(wave)

    assert _arm(outputs[0]) == 1.0
    assert "GR00T observation is invalid: state must be float32" in outputs[1].error
    assert _arm(outputs[2]) == 3.0
    assert [obs["state"]["joint"].shape[0] for obs, _ in pipeline.policy.calls] == [2]


def test_requests_with_different_dtypes_are_not_merged(pipeline):
    float_frames = _robot_obs(2.0)
    float_frames["images"]["cam"] = float_frames["images"]["cam"].astype(np.float32)
    wave = DiffusionRequestBatch(requests=[_request("a", _robot_obs(1.0)), _request("b", float_frames)])

    outputs = pipeline.forward(wave)

    assert [_arm(output) for output in outputs] == [1.0, 2.0]
    assert [obs["video"]["cam"].dtype for obs, _ in pipeline.policy.calls] == [np.uint8, np.float32]


def test_reset_from_any_request_resets_the_shared_policy_once_per_request(pipeline):
    wave = DiffusionRequestBatch(requests=[_request("a", _robot_obs(1.0), reset=True), _request("b", _robot_obs(2.0))])

    pipeline.forward(wave)

    assert pipeline.policy.reset_calls == 1
