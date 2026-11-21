# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.gr00t import pipeline_gr00t
from vllm_omni.diffusion.models.gr00t.modeling.modules.dit import BasicTransformerBlock, DiT
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeGr00tPolicy:
    instances: list["FakeGr00tPolicy"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_calls = 0
        self.seen_obs: dict[str, Any] | None = None
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

    def get_action(self, obs, options=None):
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
        prompt="pick",
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
    assert policy.seen_obs is not None
    assert "video" in policy.seen_obs
    assert policy.seen_obs["language"] == {"annotation.language.language_instruction": [["pick the cube"]]}
    assert "images" not in policy.seen_obs
    assert "prompt" not in policy.seen_obs
    assert "session_id" not in policy.seen_obs
    assert policy.reset_calls == 1


def test_dummy_warmup_returns_shape_correct_zero_actions():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="dummy run",
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


class _TinyActionHead(nn.Module):
    """Real action-head modules at toy size; no checkpoint needed."""

    def __init__(self):
        super().__init__()
        self.model = DiT(num_attention_heads=2, attention_head_dim=8, output_dim=16, num_layers=2)
        self.action_encoder = nn.Linear(4, 16)
        self.action_decoder = nn.Linear(16, 4)


class _TinyPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = _TinyActionHead()


def _compile_pipeline(monkeypatch, granularity):
    od_config = OmniDiffusionConfig(
        model="nvidia/GR00T-N1.7-3B",
        model_config={"embodiment_tag": "LIBERO_PANDA", "strict": False},
        diffusion_compile_granularity=granularity,
    )
    pipeline = Gr00tN1d7Pipeline(od_config=od_config)
    pipeline.policy.model = _TinyPolicyModel()
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(pipeline_gr00t.logger, "info", lambda msg, *args: infos.append(msg % args))
    monkeypatch.setattr(pipeline_gr00t.logger, "warning", lambda msg, *args: warnings.append(msg % args))
    return pipeline, pipeline.policy.model.action_head, infos, warnings


def _is_compiled(module: nn.Module) -> bool:
    return module._compiled_call_impl is not None


def test_setup_compile_full_compiles_each_action_head_module(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "full")

    pipeline.setup_compile()

    assert all(_is_compiled(getattr(head, name)) for name in ("model", "action_encoder", "action_decoder"))
    assert warnings == []
    assert infos == [
        "GR00T: compiled action_head model, action_encoder, action_decoder (granularity=full, dynamic=True)."
    ]


def test_setup_compile_regional_compiles_only_dit_blocks(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "regional")
    blocks = [m for m in head.model.modules() if isinstance(m, BasicTransformerBlock)]
    eager_forwards = [block.forward for block in blocks]

    pipeline.setup_compile()

    assert len(blocks) == 2
    assert all(block.forward is not eager for block, eager in zip(blocks, eager_forwards))
    assert not _is_compiled(head.action_encoder)
    assert not _is_compiled(head.action_decoder)
    assert warnings == []
    assert any("model (2 blocks)" in msg for msg in infos)


def test_setup_compile_keeps_a_failing_module_eager_and_names_it(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "full")

    def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(head.action_encoder, "compile", _fail)

    pipeline.setup_compile()

    assert not _is_compiled(head.action_encoder)
    assert _is_compiled(head.model)
    assert _is_compiled(head.action_decoder)
    assert warnings == ["GR00T: torch.compile of action_head.action_encoder failed (boom); it stays eager."]
    assert infos == ["GR00T: compiled action_head model, action_decoder (granularity=full, dynamic=True)."]


def test_setup_compile_does_not_report_success_when_nothing_compiled(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "full")

    def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    for name in ("model", "action_encoder", "action_decoder"):
        monkeypatch.setattr(getattr(head, name), "compile", _fail)

    pipeline.setup_compile()

    assert infos == []
    assert warnings[-1] == "GR00T: no action-head module was compiled; the policy runs eager."


def test_compiled_action_head_matches_eager(monkeypatch):
    """Full compile must not change the action-head numerics on CPU."""
    torch.manual_seed(0)
    pipeline, head, _, _ = _compile_pipeline(monkeypatch, "full")
    head.eval()
    hidden = torch.randn(1, 6, 16)
    context = torch.randn(1, 5, 16)
    timestep = torch.tensor([3])
    with torch.inference_mode():
        eager = head.action_decoder(head.model(hidden_states=hidden, encoder_hidden_states=context, timestep=timestep))

    pipeline.setup_compile()

    with torch.inference_mode():
        compiled = head.action_decoder(
            head.model(hidden_states=hidden, encoder_hidden_states=context, timestep=timestep)
        )
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
