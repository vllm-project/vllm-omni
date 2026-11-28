# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request and controller-output contracts of the Motubrain adapter."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.motubrain2 import pipeline_motubrain2
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("options", "support", "saved_sampler", "error"),
    [
        ({}, "none", "consistency", None),
        ({}, "recipe", "euler", None),
        ({}, "none", "euler", ImportError),
        ({}, "thresholds", "euler", ImportError),
        ({"dit_cache_enabled": True}, "none", "consistency", ImportError),
        ({"dit_cache_enabled": True}, "recipe", "consistency", None),
        ({"dit_cache_enabled": False, "dit_cache_threshold": 0.8}, "thresholds", "euler", None),
        ({"dit_cache_enabled": False}, "none", "euler", None),
        ({"dit_cache_enabled": "false"}, "recipe", "euler", ValueError),
        ({"dit_cache_threshold": 0.0}, "thresholds", "consistency", None),
        (
            dict(
                dit_cache_threshold=0.95,
                dit_cache_threshold_2=0.93,
                dit_cache_skip_steps=2,
                dit_cache_skip_steps_2=1,
                dit_cache_damping=0.5,
            ),
            "recipe",
            "consistency",
            None,
        ),
        ({"dit_cache_threshold": 0.0}, "none", "consistency", ImportError),
        ({"dit_cache_threshold_2": 0.93}, "none", "consistency", ImportError),
    ],
)
def test_load_policy_native_dit_cache(monkeypatch, tmp_path, options, support, saved_sampler, error):
    # Capture the namespace passed to the optional native package without loading weights.
    raw_policy = SimpleNamespace(from_options=lambda args: args)
    monkeypatch.setitem(sys.modules, "motubrain2.inference.policy", SimpleNamespace(RawPolicy=raw_policy))
    cache_module = {
        "none": None,
        "thresholds": SimpleNamespace(DiTCacheConfig=object),
        "recipe": SimpleNamespace(DiTCacheConfig=SimpleNamespace(for_recipe=object())),
    }[support]
    monkeypatch.setitem(sys.modules, "motubrain2.inference.dit_cache", cache_module)
    bundle = tmp_path / "weights.safetensors"
    bundle.touch()
    (tmp_path / "profile.json").touch()

    def saved_recipe(path):
        assert path == bundle
        return dict(sampler=saved_sampler, steps=30 if saved_sampler == "euler" else 4, video_shift=6, action_shift=1)

    monkeypatch.setitem(sys.modules, "motubrain2.inference.sampling", SimpleNamespace(sampling_options=saved_recipe))
    # Set conflicting stage overrides to verify that older native versions are checked against the saved recipe.
    config = dict(bundle=bundle, text_encoder="text", wan="wan", sampler="consistency", steps=2, **options)
    if error is not None:
        with pytest.raises(error, match="dit_cache_enabled" if error is ValueError else "native DiT cache support"):
            pipeline_motubrain2._load_policy(config, str(tmp_path), torch.device("cpu"), torch.bfloat16)
        return
    loaded = pipeline_motubrain2._load_policy(config, str(tmp_path), torch.device("cpu"), torch.bfloat16)
    defaults = dict(
        dit_cache_enabled=None,
        dit_cache_threshold=None,
        dit_cache_threshold_2=None,
        dit_cache_skip_steps=4,
        dit_cache_skip_steps_2=2,
        dit_cache_damping=0.25,
    )
    expected = defaults | options
    if support != "recipe" and expected["dit_cache_enabled"] is False:
        expected["dit_cache_threshold"] = expected["dit_cache_threshold_2"] = None
    assert {name: getattr(loaded, name) for name in defaults} == expected


@pytest.mark.parametrize(
    ("mxfp8", "files", "missing"),
    [
        (None, ["model.safetensors", "model-mxfp8.safetensors"], None),
        (False, ["model.safetensors"], None),
        (False, ["model-mxfp8.safetensors"], "model.safetensors"),
        (True, ["model.safetensors", "model-mxfp8.safetensors"], None),
        (True, ["model-mxfp8.safetensors"], None),
        (True, ["model.safetensors"], "model-mxfp8.safetensors"),
    ],
)
def test_load_policy_selects_checkpoint(monkeypatch, tmp_path, mxfp8, files, missing):
    policy = SimpleNamespace(from_options=lambda args: args)
    monkeypatch.setitem(sys.modules, "motubrain2.inference.policy", SimpleNamespace(RawPolicy=policy))
    cli = SimpleNamespace(add_policy_options=lambda p: p.add_argument("--mxfp8", action="store_true"))
    monkeypatch.setitem(sys.modules, "motubrain2.inference.__main__", cli)
    monkeypatch.setattr(pipeline_motubrain2, "_check_native_dit_cache", lambda args: None)
    for name in ["config.json", "profile.json", *files]:
        (tmp_path / name).touch()
    config = {"bundle": str(tmp_path), "text_encoder": "text", "wan": "wan"}
    if mxfp8 is not None:
        config["mxfp8"] = mxfp8
    args = (config, str(tmp_path), torch.device("cpu"), torch.bfloat16)
    if missing:
        with pytest.raises(FileNotFoundError) as err:
            pipeline_motubrain2._load_policy(*args)
        assert str(tmp_path / missing) in str(err.value)
        return
    loaded = pipeline_motubrain2._load_policy(*args)
    assert loaded.mxfp8 is bool(mxfp8)
    if mxfp8:
        # A BF16-only native parser accepts the namespace but ignores mxfp8.
        monkeypatch.setattr(cli, "add_policy_options", lambda p: None)
        with pytest.raises(ImportError, match="standalone --mxfp8"):
            pipeline_motubrain2._load_policy(*args)


class _RecordingPolicy:
    def __init__(self, field=None):
        self.model = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.generator = torch.Generator().manual_seed(31)
        self.sampling = {"sampler": "euler", "steps": 3}
        field = field or {"name": "joint", "kind": "linear", "width": 1}
        width = 7 if field["kind"].startswith("eef_") else field["width"]
        views = [{"name": "front"}]
        self.processor = SimpleNamespace(
            actions=SimpleNamespace(wire_dim=width),
            profile={"views": views, "action": {"fields": [field]}},
            views=views,
            action_steps=4,
            observed_frames=1,
            video=SimpleNamespace(temporal_stride=4),
            sparse_history=None,
        )
        self.valid = torch.ones(4, width, dtype=torch.bool)
        self.calls = []
        self.prefixes = []
        self.constraints = []

    def predict(self, observation, *, rtc):
        self.calls.append(self.sampling["steps"])
        self.prefixes.append(None if rtc is None else rtc.future)
        self.constraints.append(rtc)
        actions = torch.rand(self.valid.shape, generator=self.generator)
        return {"video": torch.zeros(2, 1), "action": actions, "decoded_action": actions, "decoded_valid": self.valid}


@pytest.fixture
def make_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_motubrain2, "get_local_device", lambda: torch.device("cpu"))

    def create(*, field=None, **model_config):
        monkeypatch.setattr(pipeline_motubrain2, "_load_policy", lambda *args, **kwargs: _RecordingPolicy(field))
        config = OmniDiffusionConfig(
            model=str(tmp_path), dtype=torch.float32, model_config={"text_cache_size": 0, **model_config}
        )
        return pipeline_motubrain2.Motubrain2Pipeline(od_config=config)

    return create


@pytest.fixture
def pipeline(make_pipeline):
    return make_pipeline()


def _request(extra_args, request_id="test", **params):
    request = OmniDiffusionRequest(
        prompt="move",
        sampling_params=OmniDiffusionSamplingParams(extra_args=extra_args, **params),
        request_id=request_id,
    )
    return DiffusionRequestBatch([request])


def _obs(**extra):
    return {"robot_obs": {"prompt": "move", "front": np.zeros((8, 8, 3), dtype=np.uint8)}, **extra}


@pytest.mark.parametrize(
    "field,bad_value", [("action_dim", 55), ("camera_sources", ["wrist"]), ("action_space", "eef")]
)
def test_handshake_must_match_profile(make_pipeline, field, bad_value):
    metadata = {"action_dim": 1, "action_horizon": 4, "camera_sources": ["front"], "action_space": "joint_position"}
    metadata[field] = bad_value
    with pytest.raises(ValueError, match=field):
        make_pipeline(motubrain2_metadata=metadata)


def test_only_dummy_observation_returns_zero_actions(pipeline):
    result = pipeline(_request({}))
    assert result.error_status_code == 400
    assert "robot_obs" in result.error
    assert result.output is None
    warmup = pipeline(_request({}, request_id=DUMMY_DIFFUSION_REQUEST_ID))
    assert warmup.error is None
    np.testing.assert_array_equal(warmup.output["actions"], np.zeros((4, 1), dtype=np.float32))
    assert not pipeline.policy.calls


def test_native_sdpa_restores_worker_flags_after_failure(pipeline, monkeypatch):
    # Dispatch flags can be inspected with a CPU test double for the CUDA policy.
    pipeline.policy.device = torch.device("cuda")

    def fail_predict(observation, *, rtc):
        assert torch.backends.cuda.cudnn_sdp_enabled()
        raise RuntimeError("predict failed")

    monkeypatch.setattr(pipeline.policy, "predict", fail_predict)
    previous = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        with pytest.raises(RuntimeError, match="predict failed"):
            pipeline(_request(_obs(), generator=torch.Generator()))
        assert not torch.backends.cuda.cudnn_sdp_enabled()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(previous)


def test_steps_and_random_generator_belong_to_each_request(pipeline):
    generator = torch.Generator().manual_seed(8)
    expected = torch.Generator().manual_seed(8)
    default_state = pipeline.policy.generator.get_state().clone()
    for _ in range(2):
        reply = pipeline(_request(_obs(), generator=generator, num_inference_steps=1))
        assert reply.error is None
        np.testing.assert_array_equal(reply.output["actions"], torch.rand(4, 1, generator=expected).numpy())
    seeded = pipeline(_request(_obs(), seed=8))
    assert seeded.error is None
    np.testing.assert_array_equal(seeded.output["actions"], torch.rand(4, 1, generator=expected.manual_seed(8)).numpy())
    assert pipeline.policy.calls == [1, 1, 3]
    assert pipeline.policy.sampling["steps"] == 3
    assert torch.equal(pipeline.policy.generator.get_state(), default_state)


def test_named_unseeded_requests_commit_and_reset_the_native_rng(pipeline):
    expected = torch.Generator().manual_seed(0)
    first = torch.rand(4, 1, generator=expected).numpy()
    second = torch.rand(4, 1, generator=expected).numpy()

    def request(session_id, requested=None, **extra):
        # Simulate the generator already materialized by the diffusion worker.
        return pipeline(
            _request(
                _obs(session_id=session_id, motubrain2_session_seed=0 if requested is None else None, **extra),
                generator=torch.Generator().manual_seed(99 if requested is None else requested),
            )
        )

    np.testing.assert_array_equal(request("a").output["actions"], first)
    np.testing.assert_array_equal(request("b").output["actions"], first)
    np.testing.assert_array_equal(request("a", requested=0).output["actions"], first)
    pipeline.policy.valid[-1, 0] = False
    assert request("a").error
    pipeline.policy.valid[-1, 0] = True
    np.testing.assert_array_equal(request("a").output["actions"], second)
    np.testing.assert_array_equal(request("a", reset=True).output["actions"], first)
    np.testing.assert_array_equal(request("b").output["actions"], second)
    pipeline.drop_session_state("b")
    np.testing.assert_array_equal(request("b").output["actions"], first)
    for _ in range(2):
        np.testing.assert_array_equal(request("default").output["actions"], first)
    assert pipeline.sessions.peek("default") is None


@pytest.mark.parametrize("steps", [0, -1, True])
def test_invalid_step_count_rejected_before_inference(pipeline, steps):
    result = pipeline(_request(_obs(), num_inference_steps=steps))
    assert result.error_status_code == 400
    assert result.error
    assert not pipeline.policy.calls


@pytest.mark.parametrize("missing", ["state", "camera_extrinsic"])
def test_camera_relative_pose_requires_state_and_extrinsics(make_pipeline, missing):
    pipeline = make_pipeline(field={"name": "left_eef", "kind": "eef_camera"})
    extra = _obs()
    extra["robot_obs"].update(state=[0, 0, 0, 0, 0, 0, 1], camera_extrinsic=np.eye(4, dtype=np.float32))
    del extra["robot_obs"][missing]
    result = pipeline(_request(extra))
    assert result.error_status_code == 400
    assert missing in result.error
    assert not pipeline.policy.calls


def test_held_joint_requires_current_position(make_pipeline):
    pipeline = make_pipeline(field={"name": "base", "kind": "linear", "width": 1, "model_used": False})
    result = pipeline(_request(_obs()))
    assert result.error_status_code == 400
    assert "state" in result.error
    assert not pipeline.policy.calls


@pytest.mark.parametrize("automatic", [False, True])
def test_request_rtc_overrides_session_prefix_and_supplies_anchor(make_pipeline, automatic):
    pytest.importorskip("motubrain2.inference")
    pipeline = make_pipeline(field={"name": "joint", "kind": "relative", "width": 1}, rtc={"enabled": automatic})
    extra = _obs(session_id="robot")
    extra["robot_obs"]["state"] = [9.0]
    assert pipeline(_request(extra, seed=12)).error is None
    del extra["robot_obs"]["state"]
    extra["robot_obs"]["rtc"] = {"anchor": [1], "future": [[2], [3], [4]], "delay": 2}
    assert pipeline(_request(extra, seed=13)).error is None
    anchor, future = pipeline.policy.constraints[-1].prefix(1, 4)
    assert anchor.tolist() == [1] and future.tolist() == [[2], [3]]
    extra["robot_obs"]["state"] = [9.0]
    extra["robot_obs"]["rtc"] = {"delay": 0}
    assert pipeline(_request(extra, seed=13)).error is None
    anchor, future = pipeline.policy.constraints[-1].prefix(1, 4)
    assert anchor is None and future.shape == (0, 1)
    extra["robot_obs"]["rtc"] = {"anchor": [1], "future": [], "delay": 0}
    assert pipeline(_request(extra, seed=13)).error_status_code == 400


def test_invalid_first_request_still_resets_previous_episode(make_pipeline):
    pipeline = make_pipeline(rtc={"enabled": True})
    reply = pipeline(_request(_obs(session_id="reused")))
    assert reply.error is None
    assert pipeline.sessions.peek("reused") is not None
    reply = pipeline(_request(_obs(session_id="reused", reset=True), num_inference_steps=0))
    assert reply.error_status_code == 400
    assert pipeline.sessions.peek("reused") is None
    reply = pipeline(_request(_obs(session_id="reused")))
    assert reply.error is None
    assert pipeline.policy.prefixes == [None, None]


def test_invalid_controller_coordinate_never_reaches_client_or_rtc(pipeline):
    pipeline.rtc_config = {"enabled": True, "execute_prefix": 1, "delay": 1}
    pipeline.policy.valid[-1, 0] = False
    result = pipeline(_request(_obs(session_id="robot-a")))
    assert result.error
    assert result.output is None
    assert pipeline.sessions.peek("robot-a") is None


def test_native_output_preserves_partial_validity_and_invalidates_rtc_state(make_pipeline):
    pipeline = make_pipeline(
        field={"name": "held", "kind": "linear", "width": 1, "model_used": False}, rtc={"enabled": True}
    )
    extra = _obs(session_id="robot-a")
    extra["robot_obs"]["return_native_outputs"] = True
    pipeline.policy.valid[-1, 0] = False
    result = pipeline(_request(extra))
    assert result.error is None
    assert result.output is None
    mask = result.custom_output["decoded_valid"]
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, pipeline.policy.valid.numpy())
    assert pipeline.sessions.peek("robot-a") is None


@pytest.mark.parametrize("session_id", [None, "", "  ", "default"])
def test_rtc_requires_a_session_key(pipeline, session_id):
    pipeline.rtc_config = {"enabled": True, "execute_prefix": 1, "delay": 1}
    result = pipeline(_request(_obs(session_id=session_id, reset=True)))
    assert result.error_status_code == 400
    assert "session_id" in result.error
    assert not pipeline.policy.calls
    assert len(pipeline.sessions) == 0


def test_stateless_inference_accepts_anonymous_connection(pipeline):
    result = pipeline(_request(_obs(session_id="default", reset=True)))
    assert result.error is None
    assert result.output["actions"].shape == (4, 1)
    assert len(pipeline.sessions) == 0


def test_rtc_state_is_scoped_to_each_session_and_dropped(pipeline):
    pytest.importorskip("motubrain2.inference")
    pipeline.rtc_config = {"enabled": True, "execute_prefix": 1, "delay": 1}
    for session_id, seed in [("robot-a", 11), ("robot-b", 12)]:
        reply = pipeline(_request(_obs(session_id=session_id), seed=seed))
        assert reply.error is None
        np.testing.assert_array_equal(pipeline.sessions.peek(session_id).last_action, reply.output["actions"])
    a = pipeline.sessions.peek("robot-a").last_action.copy()
    assert not np.array_equal(a, pipeline.sessions.peek("robot-b").last_action)
    assert pipeline.policy.prefixes == [None, None]
    reply = pipeline(_request(_obs(session_id="robot-a"), seed=13))
    assert reply.error is None
    np.testing.assert_array_equal(pipeline.policy.prefixes[-1], a[1:2])
    pipeline.drop_session_state("robot-b")
    assert pipeline.sessions.peek("robot-b") is None
    np.testing.assert_array_equal(pipeline.sessions.peek("robot-a").last_action, reply.output["actions"])
