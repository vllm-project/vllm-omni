# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAMPipeline end to end on a tiny on-disk checkpoint (CPU).

The tiny assets are written in the real layouts (raw ``.pt`` policy and action
codec, per-suite stats JSON, Diffusers VAE, HF umT5, tokenizer), so this covers
loading, the OpenPI observation contract, seeding, warm-up and the handshake check.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.diffusion.models.fastwam import tiny
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.fastwam import config as fastwam_config
from vllm_omni.diffusion.models.fastwam import pipeline_fastwam
from vllm_omni.diffusion.models.fastwam.pipeline_fastwam import FastWAMPipeline
from vllm_omni.diffusion.models.fastwam.processing_fastwam import build_image, postprocess_actions
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HANDSHAKE = {
    "image_resolution": [32, 32],
    "n_external_cameras": 1,
    "needs_wrist_camera": True,
    "needs_session_id": False,
    "action_horizon": 8,
    "action_dim": 7,
}


@pytest.fixture
def tiny_assets(tmp_path, monkeypatch):
    cfg = tiny.tiny_cae_config()
    monkeypatch.setitem(fastwam_config.PRESETS, "tiny_cae", cfg)
    monkeypatch.setattr(pipeline_fastwam, "get_local_device", lambda: torch.device("cpu"))
    model = tiny.build_tiny_model(cfg, seed=0)
    model_dir, base_dir = tiny.write_tiny_assets(tmp_path, cfg, model)
    return cfg, model, model_dir, base_dir


def _pipeline(tiny_assets, **model_config) -> FastWAMPipeline:
    _, _, model_dir, base_dir = tiny_assets
    od_config = OmniDiffusionConfig(
        model=model_dir,
        model_class_name="FastWAMPipeline",
        dtype=torch.float32,
        model_config={
            "fastwam_variant": "tiny_cae",
            "base_model": base_dir,
            "task_suite": "suite_a",
            "num_inference_steps": 3,
            "policy_server_config": HANDSHAKE,
            **model_config,
        },
    )
    return FastWAMPipeline(od_config=od_config)


def _robot_obs(**overrides):
    rng = np.random.RandomState(0)
    obs = {
        "observation/image": rng.randint(0, 256, (40, 40, 3), dtype=np.uint8),
        "observation/wrist_image": rng.randint(0, 256, (40, 40, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-1, 1, 8).astype(np.float32),
        "prompt": "pick the bowl",
    }
    obs.update(overrides)
    return obs


def _request(robot_obs=None, *, seed=42, steps=None, request_id="req-0") -> DiffusionRequestBatch:
    extra_args = {} if robot_obs is None else {"robot_obs": robot_obs, "reset": False, "session_id": "s"}
    params = OmniDiffusionSamplingParams(seed=seed, num_inference_steps=steps, extra_args=extra_args)
    return DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(prompt="", sampling_params=params, request_id=request_id)]
    )


def test_loads_every_component_strictly(tiny_assets):
    _, model, _, _ = tiny_assets
    pipeline = _pipeline(tiny_assets)
    loaded = pipeline.model.state_dict()
    for name, value in model.state_dict().items():
        if name.startswith("vae_encoder.") and name.split(".")[-1] in ("mean", "inv_std"):
            continue  # rebuilt from vae/config.json
        assert torch.equal(loaded[name], value), name
    assert pipeline.model.action_codec.input_proj.weight.dtype == torch.float32
    assert pipeline.load_weights([]) == {name for name, _ in pipeline.named_parameters()}
    with pytest.raises(RuntimeError, match="weights_sources"):
        pipeline.load_weights([("x", torch.zeros(1))])


def test_forward_returns_postprocessed_chunk(tiny_assets):
    cfg, _, _, _ = tiny_assets
    pipeline = _pipeline(tiny_assets)
    robot_obs = _robot_obs(task_suite="suite_b")
    output = pipeline.forward(_request(robot_obs))
    assert output.error is None
    actions = output.output["actions"]
    assert isinstance(actions, np.ndarray) and actions.dtype == np.float32
    assert actions.shape == (cfg.action_horizon, cfg.action_dim)
    assert set(np.unique(actions[:, -1])) <= {-1.0, 0.0, 1.0}

    # Same chunk computed by hand from the loaded model.
    policy = pipeline.policy
    stats = policy.stats("suite_b")
    proprio = stats.state.normalize(torch.as_tensor(robot_obs["observation/state"]).unsqueeze(0))
    context, context_mask = policy.encode_prompt("pick the bowl")
    normalized = pipeline.model.infer_action(
        build_image((robot_obs["observation/image"], robot_obs["observation/wrist_image"]), cfg.camera_size),
        context,
        context_mask,
        proprio,
        num_steps=3,
        generator=torch.Generator(device="cpu").manual_seed(42),
        suite_id=1,
    )
    expected = postprocess_actions(stats.action.denormalize(normalized.unsqueeze(0)).numpy()[0], "libero")
    np.testing.assert_array_equal(actions, expected)


def test_seed_and_steps_control_the_chunk(tiny_assets):
    pipeline = _pipeline(tiny_assets, action_postprocess="none")
    first = pipeline.forward(_request(_robot_obs(), seed=1)).output["actions"]
    again = pipeline.forward(_request(_robot_obs(), seed=1)).output["actions"]
    other = pipeline.forward(_request(_robot_obs(), seed=2)).output["actions"]
    fewer = pipeline.forward(_request(_robot_obs(), seed=1, steps=1)).output["actions"]
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)
    assert not np.allclose(first, fewer)


def test_prompt_context_is_cached(tiny_assets):
    pipeline = _pipeline(tiny_assets)
    pipeline.forward(_request(_robot_obs()))
    pipeline.forward(_request(_robot_obs()))
    pipeline.forward(_request(_robot_obs(prompt="open the drawer")))
    assert list(pipeline.policy._prompt_cache) == ["pick the bowl", "open the drawer"]


@pytest.mark.parametrize("warmup", [True, False])
def test_dummy_request_warms_up(tiny_assets, warmup):
    cfg, _, _, _ = tiny_assets
    pipeline = _pipeline(tiny_assets, warmup=warmup)
    output = pipeline.forward(_request(None, request_id=DUMMY_DIFFUSION_REQUEST_ID))
    assert output.error is None
    actions = output.output["actions"]
    assert actions.shape == (cfg.action_horizon, cfg.action_dim)
    assert bool(np.any(actions != 0)) == warmup


@pytest.mark.parametrize(
    ("robot_obs", "match"),
    [
        (None, "robot_obs"),
        (["not", "a", "dict"], "must be a dict"),
        (_robot_obs(prompt=""), "prompt"),
        (_robot_obs(task_suite="libero_goal"), "Unknown task_suite"),
    ],
)
def test_bad_requests_return_errors(tiny_assets, robot_obs, match):
    pipeline = _pipeline(tiny_assets)
    if robot_obs is None:
        request = _request(None)
    else:
        request = _request(_robot_obs())
        request.requests[0].sampling_params.extra_args["robot_obs"] = robot_obs
    output = pipeline.forward(request)
    assert output.output is None and match in output.error


def test_bad_step_count_returns_error(tiny_assets):
    pipeline = _pipeline(tiny_assets)
    output = pipeline.forward(_request(_robot_obs(), steps=0))
    assert "num_inference_steps" in output.error


@pytest.mark.parametrize(
    ("key", "value"),
    [("action_horizon", 16), ("action_dim", 8), ("image_resolution", [224, 224]), ("needs_session_id", True)],
)
def test_handshake_must_match_checkpoint(tiny_assets, key, value):
    with pytest.raises(ValueError, match=f"policy_server_config.{key}"):
        _pipeline(tiny_assets, policy_server_config={**HANDSHAKE, key: value})


def test_preset_must_match_checkpoint(tiny_assets, monkeypatch):
    cfg, _, _, _ = tiny_assets
    wrong = replace(cfg, variant="tiny_wrong", action_hidden_dim=cfg.action_hidden_dim * 2)
    monkeypatch.setitem(fastwam_config.PRESETS, "tiny_wrong", wrong)
    with pytest.raises(ValueError, match="fastwam_variant"):
        _pipeline(tiny_assets, fastwam_variant="tiny_wrong")


def test_rejects_unsupported_dtype_and_postprocess(tiny_assets):
    _, _, model_dir, base_dir = tiny_assets
    od_config = OmniDiffusionConfig(
        model=model_dir, dtype=torch.float16, model_config={"fastwam_variant": "tiny_cae", "base_model": base_dir}
    )
    with pytest.raises(ValueError, match="dtype"):
        FastWAMPipeline(od_config=od_config)
    with pytest.raises(ValueError, match="action_postprocess"):
        _pipeline(tiny_assets, action_postprocess="bogus")
