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
            "video": SimpleNamespace(delta_indices=[-1, 0], modality_keys=["front"]),
            "state": SimpleNamespace(delta_indices=[0], modality_keys=["eef"]),
            "action": SimpleNamespace(
                delta_indices=[0, 1],
                modality_keys=["arm", "gripper"],
            ),
        }
        self.processor = SimpleNamespace(
            state_action_processor=SimpleNamespace(
                norm_params={
                    "fake_embodiment": {
                        "state": {"eef": {"dim": np.array(3), "mean": np.array([0.5, -1.0, 2.0])}},
                        "action": {
                            "arm": {"dim": np.array(2)},
                            "gripper": {"dim": np.array(1)},
                        },
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


def _dit_inputs():
    return {
        "hidden_states": torch.randn(1, 6, 16),
        "encoder_hidden_states": torch.randn(1, 5, 16),
        "timestep": torch.tensor([3]),
    }


def _run_action_head(head, warmup_obs):
    """Stand in for Gr00tPolicy.get_action: run the action head the way a denoising step does."""

    def get_action(obs, options=None):
        warmup_obs.append(obs)
        with torch.inference_mode():
            hidden = head.action_encoder(torch.randn(1, 6, 4))
            head.action_decoder(
                head.model(
                    hidden_states=hidden, encoder_hidden_states=torch.randn(1, 5, 16), timestep=torch.tensor([3])
                )
            )
        return {}, {}

    return get_action


@pytest.fixture
def failing_inductor(monkeypatch):
    """Make the inductor backend fail the way a real compile failure does.

    torch.compile only registers the compile; dynamo traces on the first call and hands
    the graph to the backend there, which raises BackendCompilerFailed. Patching the
    backend entry point keeps that real, lazy path.
    """
    import torch._inductor.compile_fx as compile_fx

    def _fail(*args, **kwargs):
        raise RuntimeError("inductor exploded")

    monkeypatch.setattr(compile_fx, "compile_fx", _fail)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _is_compiled(module: nn.Module) -> bool:
    return "forward" in vars(module)


def test_setup_compile_full_compiles_each_module_at_startup(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "full")
    warmup_obs: list = []
    pipeline.policy.get_action = _run_action_head(head, warmup_obs)

    pipeline.setup_compile()

    assert all(_is_compiled(getattr(head, name)) for name in ("model", "action_encoder", "action_decoder"))
    assert len(warmup_obs) == 3
    assert warnings == []
    assert len(infos) == 1
    assert infos[0].startswith(
        "GR00T: compiled action_head.model, action_head.action_encoder, action_head.action_decoder at startup in "
    )
    assert infos[0].endswith(" s (granularity=full, dynamic=True).")


def test_setup_compile_regional_compiles_only_dit_blocks(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "regional")
    warmup_obs: list = []
    pipeline.policy.get_action = _run_action_head(head, warmup_obs)
    blocks = [m for m in head.model.modules() if isinstance(m, BasicTransformerBlock)]

    pipeline.setup_compile()

    assert len(blocks) == 2
    assert all(_is_compiled(block) for block in blocks)
    assert not _is_compiled(head.model)
    assert not _is_compiled(head.action_encoder)
    assert not _is_compiled(head.action_decoder)
    assert len(warmup_obs) == 1
    assert warnings == []
    assert infos[0].startswith("GR00T: compiled action_head.model blocks at startup in ")


def test_warmup_observation_follows_the_embodiment_config(monkeypatch):
    pipeline, head, _, _ = _compile_pipeline(monkeypatch, "regional")
    warmup_obs: list = []
    pipeline.policy.get_action = _run_action_head(head, warmup_obs)

    pipeline.setup_compile()

    (obs,) = warmup_obs
    assert set(obs["video"]) == {"front"}
    assert obs["video"]["front"].shape == (1, 2, 256, 256, 3)
    assert obs["video"]["front"].dtype == np.uint8
    assert obs["state"]["eef"].dtype == np.float32
    np.testing.assert_array_equal(obs["state"]["eef"], np.array([[[0.5, -1.0, 2.0]]], dtype=np.float32))
    assert obs["language"] == {"annotation.language.language_instruction": [["warm up"]]}


def test_full_compile_failure_names_the_module_and_restores_eager(monkeypatch, failing_inductor):
    pipeline, head, infos, _ = _compile_pipeline(monkeypatch, "full")
    pipeline.policy.get_action = _run_action_head(head, [])
    x = torch.randn(2, 4)
    with torch.inference_mode():
        eager = head.action_encoder(x)

    with pytest.raises(RuntimeError, match=r"^GR00T: torch.compile warm-up of action_head.model failed: "):
        pipeline.setup_compile()

    assert not any(_is_compiled(getattr(head, name)) for name in ("model", "action_encoder", "action_decoder"))
    assert infos == []
    with torch.inference_mode():
        torch.testing.assert_close(head.action_encoder(x), eager)


def test_regional_compile_failure_restores_every_block(monkeypatch, failing_inductor):
    pipeline, head, _, _ = _compile_pipeline(monkeypatch, "regional")
    pipeline.policy.get_action = _run_action_head(head, [])

    with pytest.raises(RuntimeError, match=r"^GR00T: torch.compile warm-up of action_head.model blocks failed: "):
        pipeline.setup_compile()

    assert not any(_is_compiled(m) for m in head.model.modules())


def test_setup_compile_warns_when_nothing_is_compiled(monkeypatch):
    pipeline, head, infos, warnings = _compile_pipeline(monkeypatch, "regional")
    warmup_obs: list = []
    pipeline.policy.get_action = _run_action_head(head, warmup_obs)
    head.model._repeated_blocks = ()

    pipeline.setup_compile()

    assert warmup_obs == []
    assert infos == []
    assert warnings[-1] == "GR00T: no action-head module was compiled; the policy runs eager."


def test_compiled_action_head_matches_eager(monkeypatch):
    """Full compile must not change the action-head numerics on CPU."""
    torch.manual_seed(0)
    pipeline, head, _, _ = _compile_pipeline(monkeypatch, "full")
    head.eval()
    inputs = _dit_inputs()
    with torch.inference_mode():
        eager = head.action_decoder(head.model(**inputs))
    pipeline.policy.get_action = _run_action_head(head, [])

    pipeline.setup_compile()

    with torch.inference_mode():
        compiled = head.action_decoder(head.model(**inputs))
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
