from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.models.gr00t import pipeline_gr00t
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeGr00tPolicy:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_calls = 0
        self.seen_obs = None
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
        self.seen_obs = obs
        return {
            "arm": np.array([[[1.0, 2.0]]], dtype=np.float64),
            "gripper": [[[3.0]]],
        }, {"latency_ms": 1.0}

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


def test_pipeline_initializes_local_policy():
    pipeline = _pipeline()

    policy = FakeGr00tPolicy.instances[0]
    assert policy.kwargs["model_path"] == "nvidia/GR00T-N1.7-3B"
    assert policy.kwargs["embodiment_tag"] == "LIBERO_PANDA"
    assert policy.kwargs["strict"] is False
    assert pipeline.weights_sources == ()
    assert pipeline.load_weights(iter(())) == set()


def test_forward_returns_dict_actions_in_output():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompts=["pick"],
        request_id="req",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "images": {"cam": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)},
                    "state": {"joint": np.zeros((1, 1, 2), dtype=np.float32)},
                    "prompt": "pick the cube",
                    "session_id": "session-a",
                },
                "reset": True,
            }
        ),
    )

    output = pipeline.forward(req)

    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"arm", "gripper"}
    assert actions["arm"].dtype == np.float32
    np.testing.assert_allclose(actions["arm"], np.array([[[1.0, 2.0]]], dtype=np.float32))
    policy = FakeGr00tPolicy.instances[0]
    assert "video" in policy.seen_obs
    assert policy.seen_obs["language"] == {"annotation.language.language_instruction": [["pick the cube"]]}
    assert "images" not in policy.seen_obs
    assert "prompt" not in policy.seen_obs
    assert "session_id" not in policy.seen_obs
    assert policy.reset_calls == 1


def test_dummy_warmup_returns_shape_correct_zero_actions():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompts=["dummy run"],
        request_id="dummy_req_id",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    output = pipeline.forward(req)

    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"arm", "gripper"}
    assert actions["arm"].shape == (1, 2, 2)
    assert actions["gripper"].shape == (1, 2, 1)
    assert not actions["arm"].any()
    assert FakeGr00tPolicy.instances[0].seen_obs is None


def test_reset_delegates_to_policy():
    pipeline = _pipeline()

    assert pipeline.reset() == {"reset": True}
    assert FakeGr00tPolicy.instances[0].reset_calls == 1
