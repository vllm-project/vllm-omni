# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request validation and full/stepwise SGF lifecycle through the pipeline."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig
from vllm_omni.diffusion.models.solarwm_h3.cache import SolarRawKVCache
from vllm_omni.diffusion.models.solarwm_h3.decode import SolarH3DecodeState
from vllm_omni.diffusion.models.solarwm_h3.layout import build_layout, patchify_video
from vllm_omni.diffusion.models.solarwm_h3.pipeline_solarwm_h3 import SolarWMH3Pipeline, _SolarSession
from vllm_omni.diffusion.models.solarwm_h3.request import CAMERA_SCHEMA, parse_request
from vllm_omni.diffusion.models.solarwm_h3.sampling import stage2_timesteps
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _request():
    prompt = {"prompt": "A room", "multi_modal_data": {"image": Image.new("RGB", (32, 32))}}
    sampling = OmniDiffusionSamplingParams(
        height=32,
        width=32,
        num_frames=22,
        num_inference_steps=4,
        guidance_scale=1,
        seed=12,
        output_type="latent",
        extra_args={
            "camera": {
                "schema": CAMERA_SCHEMA,
                "c2w": torch.eye(4).repeat(22, 1, 1).tolist(),
                "intrinsics": torch.eye(3).tolist(),
            }
        },
    )
    return prompt, sampling


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_outputs_per_prompt", 2),
        ("num_inference_steps", 5),
        ("timesteps", torch.tensor([1.0, 0.5])),
        ("sigmas", [1.0, 0.5]),
        ("timesteps", torch.empty(0)),
        ("sigmas", []),
        ("guidance_scale", 2),
        ("height", 64),
        ("height", 0),
        ("fps", 0),
        ("fps", float("nan")),
        ("output_type", "invalid"),
    ],
)
def test_rejects_unsupported_sampling(field, value):
    prompt, sampling = _request()
    setattr(sampling, field, value)
    with pytest.raises(ValueError, match="SolarWM-H3"):
        parse_request(prompt, sampling, height=32, width=32)


def test_request_geometry_and_camera_validation():
    prompt, sampling = _request()
    request = parse_request(prompt, sampling, height=32, width=32)
    assert request.camera.geometry.decode_latents == 7
    assert request.camera.geometry.rollout_latents == 10
    assert request.camera.viewmats.device.type == "cpu"
    sampling.extra_args["camera"]["c2w"][2][0][0] = 2
    with pytest.raises(ValueError, match="rigid transforms"):
        parse_request(prompt, sampling, height=32, width=32)


@pytest.mark.parametrize(
    "schema,data_schema",
    [
        ("solarwm.camera_trajectory.v0", None),
        ("solarwm.camera_trajectory.v0", CAMERA_SCHEMA),
        (CAMERA_SCHEMA, CAMERA_SCHEMA),
    ],
)
def test_tick_camera_schema_cannot_be_overridden(schema, data_schema):
    prompt, sampling = _request()
    camera = dict(sampling.extra_args["camera"])
    del camera["schema"]
    if data_schema is not None:
        camera["schema"] = data_schema
    tick = ARDiffusionTickRequest(
        session_id="s",
        request_id="r",
        chunk_index=0,
        controls=(ARDiffusionControlInput(track="camera", schema=schema, data=camera),),
    )
    sampling.extra_args = tick.to_extra_args()
    with pytest.raises(ValueError, match="camera control requires schema"):
        parse_request(prompt, sampling, height=32, width=32)


def test_tick_camera_schema_accepts_versioned_control():
    prompt, sampling = _request()
    camera = dict(sampling.extra_args["camera"])
    del camera["schema"]
    tick = ARDiffusionTickRequest(
        session_id="s",
        request_id="r",
        chunk_index=0,
        controls=(ARDiffusionControlInput(track="camera", schema=CAMERA_SCHEMA, data=camera),),
    )
    sampling.extra_args = tick.to_extra_args()
    request = parse_request(prompt, sampling, height=32, width=32)
    assert request.tick == tick


def test_rejects_data_parallel_replicas(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.solarwm_h3.pipeline_solarwm_h3.get_local_device", lambda: torch.device("cpu")
    )
    config = OmniDiffusionConfig(
        model="unused",
        num_gpus=2,
        parallel_config=DiffusionParallelConfig(data_parallel_size=2),
    )
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        SolarWMH3Pipeline(od_config=config)


class _Student(nn.Module):
    """Deterministic velocity boundary; real native projections have device tests."""

    arch = MiniMaxH3DiTArchConfig(num_layers=2)

    def __init__(self):
        super().__init__()
        self.calls = []

    def predict_video(self, prompt, anchor, current, audio, layout, video_t, audio_t, context):
        self.calls.append((context.chunk_index, float(video_t), context.commit_cache))
        assert current.dtype == torch.float32
        for layer in range(2):
            history = context.cache.history(layer, context.chunk_index)
            if context.commit_cache:
                key = torch.nn.functional.pad(patchify_video(current).unsqueeze(2), (0, 32))
                context.cache.stage(layer, context.chunk_index, key, key + layer)
        offset = sum(float(k.mean()) for _, k, _ in history) * 0.001
        return current * 0.1 + video_t + offset


def _pipeline_and_session():
    prompt, sampling = _request()
    inputs = parse_request(prompt, sampling, height=32, width=32)
    pipe = SolarWMH3Pipeline.__new__(SolarWMH3Pipeline)
    nn.Module.__init__(pipe)
    pipe.height = pipe.width = 32
    pipe.device = torch.device("cpu")
    pipe._rank = 0
    pipe._ar_state = None
    pipe.transformer = _Student()
    generator = torch.Generator().manual_seed(12)
    noise = torch.randn((1, 24, 10, 2, 2), generator=generator)
    session = _SolarSession(
        inputs=inputs,
        prompt=torch.zeros(2, 5120),
        anchor=torch.zeros(1, 1, 96),
        audio=torch.zeros(1, 526, 32),
        audio_t=torch.tensor(0.2),
        layout=build_layout(torch.tensor([1, 0]), latent_height=2, latent_width=2),
        generator=generator,
        initial_noise=noise,
        decode=SolarH3DecodeState("s", max_frames=22),
        cache=SolarRawKVCache(2),
    )
    pipe._sessions = {"s": session}
    return pipe, session, sampling


@torch.inference_mode()
def test_stepwise_sampler_matches_full_sampler_and_clean_commits():
    reference, rollout, _ = _pipeline_and_session()
    expected = []
    for _ in range(2):
        expected.append(reference._generate(rollout))
        rollout.next_chunk += 1
    pipe, session, sampling = _pipeline_and_session()
    state = StepRequestState("request", sampling)
    state.extra["solar_session_id"] = "s"
    state.total_chunks = 2
    state.chunk_num_steps = 4
    state.timesteps = stage2_timesteps().repeat(2)
    actual = []
    for index in range(2):
        pipe.prepare_next_chunk(state)
        for _ in range(4):
            prediction = pipe.denoise_step(None, states=[state])
            assert session.cache.chunk_index == index
            assert not session.cache.pending
            pipe.step_scheduler(state, prediction)
        assert state.chunk_denoise_completed
        actual.append(state.latents)
        result = pipe.post_decode(state)
        assert result.finished is (index == 1)
        assert result.chunk_index == index
        assert result.output["payload"]["latents"].shape[2] == (5 if index == 0 else 2)
        assert session.cache.chunk_index == index + 1
    for first, second in zip(expected, actual):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert pipe.transformer.calls == reference.transformer.calls
    assert sum(commit for _, _, commit in pipe.transformer.calls) == 2
    assert all(time == 1 for _, time, commit in pipe.transformer.calls if commit)
    pipe.close_ar_diffusion_session("s")
    assert not pipe._sessions and session.decode.closed and not session.cache.layers


def test_continuation_preserves_condition_and_committed_camera():
    pipe, session, _ = _pipeline_and_session()
    session.next_chunk = 1
    with pytest.raises(ValueError, match="settings are fixed"):
        pipe._session("s", replace(session.inputs, prompt="Changed"))
    camera = session.inputs.camera
    views = camera.viewmats.clone()
    views[:, 2, 0, 3] = 1
    with pytest.raises(ValueError, match="committed poses"):
        pipe._session("s", replace(session.inputs, camera=replace(camera, viewmats=views)))
    views = camera.viewmats.clone()
    views[:, 7, 0, 3] = 1
    updated = replace(session.inputs, camera=replace(camera, viewmats=views))
    assert pipe._session("s", updated) is session
    assert session.inputs is updated
    pipe.reset_ar_diffusion_session("s")
    assert not pipe._sessions and session.decode.closed
