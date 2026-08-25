# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.models.gr00t import pipeline_gr00t
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeGr00tPolicy:
    instances: list["FakeGr00tPolicy"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_calls = 0
        self.calls = []
        self.embodiment_tag = SimpleNamespace(value="fake_embodiment")
        self.language_key = "annotation.language.language_instruction"
        self.modality_configs = {
            "action": SimpleNamespace(
                delta_indices=[0, 1],
                modality_keys=["arm", "gripper"],
            )
        }
        self.processor = SimpleNamespace(
            state_action_processor=SimpleNamespace(
                norm_params={
                    "fake_embodiment": {
                        "action": {
                            "arm": {"dim": np.array(2)},
                            "gripper": {"dim": np.array(1)},
                        }
                    }
                }
            )
        )
        FakeGr00tPolicy.instances.append(self)

    def get_action(self, obs):
        self.calls.append(obs)
        batch_size = len(next(iter(obs["video"].values())))
        # Echo each sample's state marker into its actions so tests can check
        # that batched outputs are sliced back to the right request.
        marker = np.asarray(obs["state"]["joint"], dtype=np.float64)[:, 0, 0]
        arm = np.broadcast_to(marker[:, None, None], (batch_size, 2, 2)).copy()
        gripper = np.broadcast_to(marker[:, None, None], (batch_size, 2, 1)).tolist()
        return {"arm": arm, "gripper": gripper}, {"latency_ms": 1.0}

    def reset(self):
        self.reset_calls += 1
        return {"reset": True}


@pytest.fixture(autouse=True)
def fake_gr00t_policy(monkeypatch):
    FakeGr00tPolicy.instances.clear()
    monkeypatch.setattr(pipeline_gr00t, "Gr00tPolicy", FakeGr00tPolicy)


def _pipeline():
    od_config = SimpleNamespace(
        model="nvidia/GR00T-N1.7-3B",
        model_config={
            "embodiment_tag": "LIBERO_PANDA",
            "strict": False,
        },
        custom_pipeline_args={},
    )
    return Gr00tN1d7Pipeline(od_config=od_config)


def _robot_request(
    request_id: str, *, marker: float, prompt: str = "pick the cube", side: int = 8, reset: bool = False
):
    return OmniDiffusionRequest(
        prompt="pick",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "images": {"cam": np.full((1, 1, side, side, 3), int(marker), dtype=np.uint8)},
                    "state": {"joint": np.full((1, 1, 2), marker, dtype=np.float32)},
                    "prompt": prompt,
                    "session_id": f"session-{request_id}",
                },
                "reset": reset,
            }
        ),
    )


def test_pipeline_initializes_local_policy():
    pipeline = _pipeline()

    policy = FakeGr00tPolicy.instances[0]
    assert policy.kwargs["model_path"] == "nvidia/GR00T-N1.7-3B"
    assert policy.kwargs["embodiment_tag"] == "LIBERO_PANDA"
    assert policy.kwargs["strict"] is False
    assert pipeline.weights_sources == ()
    assert pipeline.load_weights(iter(())) == set()
    assert Gr00tN1d7Pipeline.supports_request_batch is True


def test_forward_returns_dict_actions_in_output():
    pipeline = _pipeline()
    batch = DiffusionRequestBatch(requests=[_robot_request("req", marker=1.0, reset=True)])

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    output = outputs[0]
    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"arm", "gripper"}
    assert actions["arm"].dtype == np.float32
    np.testing.assert_allclose(actions["arm"], np.ones((1, 2, 2), dtype=np.float32))
    policy = FakeGr00tPolicy.instances[0]
    seen_obs = policy.calls[0]
    assert "video" in seen_obs
    assert seen_obs["language"] == {"annotation.language.language_instruction": [["pick the cube"]]}
    assert "images" not in seen_obs
    assert "prompt" not in seen_obs
    assert "session_id" not in seen_obs
    assert policy.reset_calls == 1


def test_forward_merges_wave_into_single_policy_call():
    pipeline = _pipeline()
    batch = DiffusionRequestBatch(
        requests=[
            _robot_request("req-a", marker=1.0, prompt="pick the cube"),
            _robot_request("req-b", marker=2.0, prompt="open the drawer"),
        ]
    )

    outputs = pipeline.forward(batch)

    policy = FakeGr00tPolicy.instances[0]
    assert len(policy.calls) == 1
    merged_obs = policy.calls[0]
    assert merged_obs["video"]["cam"].shape == (2, 1, 8, 8, 3)
    assert merged_obs["state"]["joint"].shape == (2, 1, 2)
    assert merged_obs["language"] == {
        "annotation.language.language_instruction": [["pick the cube"], ["open the drawer"]]
    }

    assert [output.error for output in outputs] == [None, None]
    np.testing.assert_allclose(outputs[0].output["actions"]["arm"], np.ones((1, 2, 2), dtype=np.float32))
    np.testing.assert_allclose(outputs[1].output["actions"]["arm"], np.full((1, 2, 2), 2.0, dtype=np.float32))
    for output in outputs:
        assert output.output["actions"]["gripper"].shape == (1, 2, 1)
        assert output.output["actions"]["gripper"].dtype == np.float32


def test_forward_wave_isolates_malformed_request():
    pipeline = _pipeline()
    bad_request = OmniDiffusionRequest(
        prompt="pick",
        request_id="req-bad",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"robot_obs": "not-a-dict"}),
    )
    batch = DiffusionRequestBatch(requests=[_robot_request("req-a", marker=3.0), bad_request])

    outputs = pipeline.forward(batch)

    assert outputs[0].error is None
    np.testing.assert_allclose(outputs[0].output["actions"]["arm"], np.full((1, 2, 2), 3.0, dtype=np.float32))
    assert "robot_obs must be a dict" in outputs[1].error
    # The valid request still runs through the single-observation path.
    assert len(FakeGr00tPolicy.instances[0].calls) == 1


def test_forward_wave_serves_dummy_and_real_requests():
    pipeline = _pipeline()
    dummy = OmniDiffusionRequest(
        prompt="dummy run",
        request_id="dummy_req_id",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )
    batch = DiffusionRequestBatch(requests=[dummy, _robot_request("req-a", marker=4.0)])

    outputs = pipeline.forward(batch)

    dummy_actions = outputs[0].output["actions"]
    assert dummy_actions["arm"].shape == (1, 2, 2)
    assert not dummy_actions["arm"].any()
    np.testing.assert_allclose(outputs[1].output["actions"]["arm"], np.full((1, 2, 2), 4.0, dtype=np.float32))


def test_forward_wave_survives_garbage_modality():
    pipeline = _pipeline()
    garbage = OmniDiffusionRequest(
        prompt="pick",
        request_id="req-garbage",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "images": {"cam": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)},
                    "state": 5,
                    "prompt": "pick",
                }
            }
        ),
    )
    batch = DiffusionRequestBatch(requests=[_robot_request("req-a", marker=6.0), garbage])

    outputs = pipeline.forward(batch)

    assert outputs[0].error is None
    np.testing.assert_allclose(outputs[0].output["actions"]["arm"], np.full((1, 2, 2), 6.0, dtype=np.float32))
    assert "GR00T policy inference failed" in outputs[1].error


def test_forward_wave_falls_back_on_incompatible_observations():
    pipeline = _pipeline()
    batch = DiffusionRequestBatch(
        requests=[
            _robot_request("req-a", marker=1.0, side=8),
            _robot_request("req-b", marker=2.0, side=4),
        ]
    )

    outputs = pipeline.forward(batch)

    # Video shapes cannot be concatenated, so each request gets its own call.
    assert len(FakeGr00tPolicy.instances[0].calls) == 2
    assert [output.error for output in outputs] == [None, None]
    np.testing.assert_allclose(outputs[0].output["actions"]["arm"], np.ones((1, 2, 2), dtype=np.float32))
    np.testing.assert_allclose(outputs[1].output["actions"]["arm"], np.full((1, 2, 2), 2.0, dtype=np.float32))


def test_dummy_warmup_returns_shape_correct_zero_actions():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="dummy run",
        request_id="dummy_req_id",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    outputs = pipeline.forward(DiffusionRequestBatch(requests=[req]))

    assert len(outputs) == 1
    assert outputs[0].error is None
    actions = outputs[0].output["actions"]
    assert set(actions) == {"arm", "gripper"}
    assert actions["arm"].shape == (1, 2, 2)
    assert actions["gripper"].shape == (1, 2, 1)
    assert not actions["arm"].any()
    assert FakeGr00tPolicy.instances[0].calls == []


def test_reset_delegates_to_policy():
    pipeline = _pipeline()

    assert pipeline.reset() == {"reset": True}
    assert FakeGr00tPolicy.instances[0].reset_calls == 1
