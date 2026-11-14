# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts of the Praxis-1 worker: configuration, caches, warmup, sessions and request limits."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.diffusion.models.praxis1.test_praxis1 import _worker
from vllm_omni.diffusion.models.praxis1 import pipeline_praxis1, processing
from vllm_omni.diffusion.models.praxis1.pipeline_praxis1 import Praxis1Pipeline
from vllm_omni.diffusion.output_formatter import _build_multimodal_output, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.praxis1 import PraxisSessionOptions, SamplerSettings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(name, **extra):
    params = extra.pop("params", {})
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="",
                request_id=name,
                sampling_params=OmniDiffusionSamplingParams(**params, extra_args={"session_id": name, **extra}),
            )
        ]
    )


@pytest.mark.parametrize("entries,budget", [(2, 1024), (128, 64)])
def test_text_cache_bounds_entries_and_padded_storage(monkeypatch, entries, budget):
    monkeypatch.setattr(pipeline_praxis1, "DEFAULT_TEXT_CACHE_MAX_ENTRIES", entries)
    monkeypatch.setattr(pipeline_praxis1, "DEFAULT_TEXT_CACHE_MAX_BYTES", budget)
    worker = _worker()
    worker.text_encoder.encode = Mock(side_effect=lambda _: [torch.zeros(8)])
    first = worker.encode_text("first")
    worker.encode_text("second")
    assert worker.encode_text("first") is first
    worker.encode_text("third")
    assert list(worker._text_cache) == ["first", "third"] and worker._text_cache_bytes == 64
    worker.text_encoder.encode = Mock(return_value=[torch.zeros(budget // 4 + 1)[:1]])
    worker.encode_text("oversized")
    assert "oversized" not in worker._text_cache
    worker.reset()
    assert not worker._text_cache and worker._text_cache_bytes == 0


@pytest.mark.parametrize("entries,bytes_", [(0, 64), (4, 0)])
def test_text_cache_can_be_disabled(entries, bytes_):
    worker = _worker()
    worker.text_cache_max_entries, worker.text_cache_max_bytes = entries, bytes_
    worker.text_encoder.encode = Mock(return_value=[torch.zeros(8)])
    worker.encode_text("task")
    worker.encode_text("task")
    assert worker.text_encoder.encode.call_count == 2 and not worker._text_cache


@pytest.mark.parametrize("steps", [0, -1, True, 17, 100000000])
def test_worker_step_budget_applies_before_session_allocation(steps):
    worker = _worker()
    with pytest.raises(ValueError, match="num_inference_steps"):
        worker._forward(_request("episode", robot_obs={}, params={"num_inference_steps": steps}))
    assert not worker._sessions


@pytest.mark.parametrize("key", ["task", "prompt"])
@pytest.mark.parametrize("value", ["x" * 65537, b"x" * 65537, "é" * 32769], ids=["text", "bytes", "utf8"])
def test_prompt_byte_limit_precedes_session_allocation(monkeypatch, key, value):
    worker = _worker()
    monkeypatch.setattr(
        pipeline_praxis1, "Session", Mock(side_effect=AssertionError("oversized prompt reached allocation"))
    )
    with pytest.raises(ValueError, match="64 KiB"):
        worker._forward(_request("limit", robot_obs={key: value}))
    assert not worker._sessions


def test_idle_sessions_expire_on_the_next_request(monkeypatch):
    worker = _worker()
    worker.session_idle_timeout = 5
    old = Mock(spec=pipeline_praxis1.Session, last_used=0)
    live = Mock(spec=pipeline_praxis1.Session, last_used=9)
    live.step.return_value = Mock(commands=[{"robot_command": {"pose": np.zeros(12)}, "target_grip": 0.5}])
    worker._sessions = {"old": old, "live": live, "socket": Mock(last_used=0, transport_owned=True)}
    monkeypatch.setattr(pipeline_praxis1.time, "monotonic", lambda: 10)
    worker._forward(_request("live", robot_obs={}))
    assert set(worker._sessions) == {"live", "socket"} and live.last_used == 10
    old.step.assert_not_called()


@pytest.mark.parametrize("sampler", [PraxisSessionOptions(), PraxisSessionOptions(control_mode="none")])
@pytest.mark.parametrize("embodiment,width", [("single_arm", 13), ("bimanual", 26)])
def test_worker_reserves_capacity_without_sampling(monkeypatch, sampler, embodiment, width):
    worker = _worker()
    worker.embodiment = embodiment
    worker.sampler = sampler
    worker.max_sessions = 1
    factory = Mock(return_value=Mock(last_used=time.monotonic()))
    monkeypatch.setattr(pipeline_praxis1, "Session", factory)

    def reserve(name):
        result = worker._forward(
            _request(name, robot_obs={}, reserve_session=True, praxis_session_options={"context_mode": "stateless"})
        )
        assert result.output["payload"]["actions"].shape == (0, width)
        formatted = _build_multimodal_output(normalize_diffusion_postprocess_output(result.output), None)
        return formatted["trajectory"]["session_available"]

    assert reserve("first") and not reserve("second")
    factory.return_value.step.assert_not_called()
    assert factory.call_args.kwargs["control_mode"] == sampler.control_mode
    worker.reset("first")
    assert reserve("second")


def test_session_options_override_sampler_and_commands_skip_video(monkeypatch):
    worker = _worker()
    options = PraxisSessionOptions(num_steps=4, max_context_chunks=3, exec_horizon=12, cfg_scale=2.5)
    commands = [{"robot_command": {"pose": np.arange(12, dtype=np.float32)}, "target_grip": 0.5, "timestamp": 0.0}]
    session = Mock(spec=pipeline_praxis1.Session, state=Mock(), sampler=SamplerSettings(), last_used=time.monotonic())
    session.step.return_value = Mock(spec=pipeline_praxis1.StepResult, commands=commands)
    factory = Mock(return_value=session)
    monkeypatch.setattr(pipeline_praxis1, "Session", factory)
    output = worker._forward(
        _request(
            "a",
            robot_obs={"task": "move"},
            praxis_session_options=options.to_dict(),
            return_commands=True,
            params={"num_inference_steps": 4, "seed": 7},
        )
    )
    assert factory.call_args.kwargs["max_context_chunks"] == 0  # a short horizon is stateless
    assert factory.call_args.kwargs["exec_horizon"] == 12 and factory.call_args.kwargs["seed"] == 7
    assert session.sampler.num_steps == 4 and session.sampler.cfg_scale == 2.5
    session.state.restart.assert_called_once()
    formatted = _build_multimodal_output(normalize_diffusion_postprocess_output(output.output), None)
    assert formatted["trajectory"] == {"commands": commands}
    session.decode_video.assert_not_called()


def _warmup_worker(monkeypatch, embodiment, depth):
    worker = _worker()
    worker.startup_warmup = True
    worker.embodiment = embodiment
    worker.sampler = PraxisSessionOptions(max_context_chunks=depth)
    worker.transformer.config = {"video": {"tile_height": 224, "tile_width": 416}}
    factory = Mock()
    monkeypatch.setattr(pipeline_praxis1, "Session", factory)
    worker._sessions["stale"] = Mock()
    return worker, factory


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("embodiment,width", [("single_arm", 13), ("bimanual", 26)])
def test_startup_warmup_covers_configured_depths_and_resets(monkeypatch, depth, embodiment, width):
    worker, factory = _warmup_worker(monkeypatch, embodiment, depth)
    assert worker._warmup().output["payload"]["actions"].shape == (0, width)
    assert factory.return_value.step.call_count == depth + 1
    obs = factory.return_value.step.call_args.args[0]
    views = ("exterior", "wrist_left", "wrist_right") if width == 26 else ("exterior", "wrist")
    assert {key for key in obs if key.startswith("image.")} == {f"image.{v}" for v in views}
    assert obs["image.exterior"].shape == (25, 260 if width == 26 else 224, 416, 3)
    assert ("descriptor" in obs) == (width == 13)
    assert not worker._sessions and not worker._text_cache


def test_startup_warmup_failure_propagates_and_releases_state(monkeypatch):
    worker, factory = _warmup_worker(monkeypatch, "single_arm", 1)
    factory.return_value.step.side_effect = RuntimeError("warmup failed")
    with pytest.raises(RuntimeError, match="warmup failed"):
        worker._warmup()
    assert not worker._sessions and not worker._text_cache


@pytest.mark.parametrize("embodiment,arms", [("single_arm", 1), ("bimanual", 2)])
@pytest.mark.parametrize("dims", [None, []])
def test_session_normalization_defaults_cover_each_arm(embodiment, arms, dims):
    worker = _worker()
    worker.device = "cpu"
    worker.vae = Mock()
    worker.stats = processing.NormStats(*(np.full(10 * arms, v, np.float32) for v in (0, 1, 0, 1)))
    worker.transformer.config = {
        "embodiment": embodiment,
        "normalization": {} if dims is None else {"q99_normed_dims": dims},
        "chunking": {},
        "video": {"tile_height": 256 if arms == 2 else 224, "tile_width": 416},
    }
    session = pipeline_praxis1.Session(worker)
    _, relative = processing.decode_actions(torch.zeros(1, 6, 40 * arms), worker.stats, 0, session.state.dims)
    np.testing.assert_array_equal(relative[:, 9::10], 0.5 if dims is None else 0.0)


_GEOMETRY = {"image_resolution": [234, 416], "tile_size": [224, 416], "fps": 15, "video_antialias": True}
_CACHE_ERROR = "text_cache_max_.*nonnegative integer"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"text_cache_max_entries": 2, "text_cache_max_bytes": 1024}, None),
        ({"text_cache_max_entries": -1}, _CACHE_ERROR),
        ({"text_cache_max_bytes": -1}, _CACHE_ERROR),
        ({"text_cache_max_entries": True}, _CACHE_ERROR),
        ({"text_cache_max_bytes": 1.5}, _CACHE_ERROR),
        ({"attention_impl": "dense"}, "attention_impl"),
        ({"policy_server_config": _GEOMETRY}, "policy_server_config"),
        ({"policy_server_config": {**_GEOMETRY, "needs_session_id": True}}, "policy_server_config"),
        (
            {"policy_server_config": {**_GEOMETRY, "needs_session_id": True, "supports_close_session": 1}},
            "policy_server_config",
        ),
        ({"policy_server_config": {"policy": "praxis"}}, "policy_server_config"),
        ({"policy_server_config": {"image_resolution": [256, 416], "video_antialias": False}}, "policy_server_config"),
    ],
)
def test_model_config_is_validated_before_weight_loading(tmp_path, monkeypatch, overrides, error):
    for name in ("config.json", "Wan2.2_VAE.pth", "text_encoder/config.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        pipeline_praxis1,
        "load_config",
        lambda _: {
            "embodiment": "single_arm",
            "video": {"tile_height": 224, "tile_width": 416, "antialias": True},
            "chunking": {"grid_hz": 15},
        },
    )
    monkeypatch.setattr(pipeline_praxis1, "resolve_offload_strategy", lambda _: pipeline_praxis1.OffloadStrategy.NONE)

    class BeforeWeightsError(Exception):
        pass

    monkeypatch.setattr(pipeline_praxis1, "get_current_vllm_config", Mock(side_effect=BeforeWeightsError))
    settings = {"num_steps": 4, "cfg_scale": 1.5, "cfg_video_scale": 2.5, "t_shift": 8.0, "context_mode": "stateless"}
    od = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1),
        quantization_config=None,
        enable_sleep_mode=False,
        lora_path=None,
        step_execution=False,
        dtype=torch.bfloat16,
        model=str(tmp_path),
        model_config=settings | overrides,
    )
    worker = Praxis1Pipeline.__new__(Praxis1Pipeline)
    if error:
        with pytest.raises(ValueError, match=error):
            worker.__init__(od_config=od)
        return
    with pytest.raises(BeforeWeightsError):
        worker.__init__(od_config=od)
    assert worker.text_cache_max_entries == 2 and worker.text_cache_max_bytes == 1024
    assert worker.sampler == PraxisSessionOptions(**settings)
