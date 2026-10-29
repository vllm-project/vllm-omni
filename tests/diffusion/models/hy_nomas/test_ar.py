# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.diffusion.ar_diffusion.test_capability_runner import make_runner
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import REQUIRED_FILES, HyNomasARConfig
from vllm_omni.diffusion.models.hy_nomas.interaction import (
    HyNomasCameraInteractionHandler,
    HyNomasPromptInteractionHandler,
)
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import normalize_request, plan_chunks
from vllm_omni.diffusion.models.interface import supports_interaction_apply, supports_step_execution
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.ar_diffusion.capability import SupportsARDiffusionPipeline
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class DecoderStub:
    """A decoder dependency with a three-frame lookahead, not a PixelUMM quality test."""

    def __init__(self, hold_frames=3):
        self.hold_frames = hold_frames
        self.available_pixel_frames = 0
        self.pending = torch.empty((1, 3, 0, 32, 32))
        self.finished = False

    def push(self, latent, *, pixel_frames):
        assert not self.finished
        frames = latent[:, :1].repeat_interleave(4, 2)[:, :, :pixel_frames].expand(-1, 3, -1, 32, 32)
        self.pending = torch.cat((self.pending, frames), 2)
        self.available_pixel_frames += pixel_frames
        stable, self.pending = self.pending[:, :, : -self.hold_frames], self.pending[:, :, -self.hold_frames :]
        return stable

    def finish(self):
        assert not self.finished
        self.finished = True
        tail, self.pending = self.pending, self.pending[:, :, :0]
        return tail


class SchedulerStub:
    def __init__(self, steps):
        self.timesteps = torch.linspace(1000, 1, steps)

    def step(self, *, sample, model_output, **kwargs):
        return (sample.float() - model_output.float() / len(self.timesteps),)


class PipelineWithCPUComponents(HyNomasPipeline):
    """Learned components are substituted; the pipeline's generation and session methods are real."""

    target_dtype = torch.float32
    in_channels = 2
    s_ratio = 32
    patch_size = (1, 1, 1)
    num_double_blocks = 2
    heads_num = 2
    hidden_size = 8
    cache_noise_sigma = 0.206897

    def __init__(self):
        nn.Module.__init__(self)
        self.args = SimpleNamespace(text_len=8)
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(out_channels=2),
            camera_injection_mode="extrinsics_attn",
            forward_with_cache=self.transformer_forward,
        )
        self.vae = SimpleNamespace(create_streaming_decode_session=lambda **kwargs: DecoderStub(), decode=self.decode)
        self.text_encoder = SimpleNamespace()
        self.calls = []
        self.encoded = []
        self.camera_calls = []
        self.fail_clean = False
        self.device = torch.device("cpu")
        self.sp_size = 1
        self._cpu_offload = False

    def _encode_prompt(self, text, *args, **kwargs):
        self.encoded.append(text)
        length = 7 if text == "longer prompt" else 4
        return {"text_states": torch.ones(1, length, 2), "text_mask": torch.tensor([[1] * (length - 1) + [0]])}

    def _component_context(self, component):
        return nullcontext()

    def prepare_camera(self, transformer, config, camera, *, num_frames, **kwargs):
        self.camera_calls.append((camera, num_frames))
        return None if camera is None else torch.arange((num_frames - 1) // 4 + 1).reshape(1, -1, 1)

    def _new_scheduler(self, steps):
        return SchedulerStub(steps)

    def transformer_forward(self, **kwargs):
        hidden = kwargs["hidden_states"]
        start = kwargs["cache_valid_len"]
        image = hidden.shape[2]
        count = image + (kwargs["text_states"].shape[1] if kwargs["cache_txt"] else 0)
        self.calls.append(
            {
                "commit": kwargs["update_cache"],
                "position": kwargs["global_frame_counter"],
                "mask": kwargs["cache_mask_list"][0][:, :start].clone(),
                "text_length": kwargs["text_states"].shape[1],
                "timestep_dtype": kwargs["timestep"].dtype,
                "cache_txt": kwargs["cache_txt"],
                "history": kwargs["kv_cache_list"][0]["k"][:, :start].clone(),
            }
        )
        if kwargs["update_cache"]:
            for layer, view in enumerate(kwargs["kv_cache_list"]):
                view["k"][:, start:] = layer + kwargs["global_frame_counter"] + 1
                view["v"][:, start:] = -view["k"][:, start:]
                mask = kwargs["cache_mask_list"][layer]
                mask[:, start : start + image] = True
                if kwargs["cache_txt"]:
                    mask[:, start + image :] = kwargs["text_mask"].bool()
                if self.fail_clean and layer == 0:
                    raise RuntimeError("injected clean-forward failure")
        return torch.ones_like(hidden) * 0.25, count if kwargs["update_cache"] else 0

    def _prepare_vae_decode_latents(self, latents):
        return latents

    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :1].repeat_interleave(4, 2).expand(-1, 3, -1, 32, 32))


@pytest.fixture
def pipeline(monkeypatch):
    pipeline = PipelineWithCPUComponents()
    monkeypatch.setattr("vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas.prepare_camera", pipeline.prepare_camera)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas.action_conditioning", lambda *args: (2, None, None)
    )
    pipeline.initialize_ar(
        {
            "ar_height": 32,
            "ar_width": 32,
            "ar_chunk_latent_frames": 3,
            "ar_max_latent_frames": 12,
            "ar_max_segments": 4,
            "ar_session_capacity": 2,
        }
    )
    return pipeline


@pytest.fixture
def runner(pipeline, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform", SimpleNamespace(synchronize=lambda: None)
    )
    return make_runner(pipeline)


def sampling(*, frames=9, seed=7, extra=None):
    return OmniDiffusionSamplingParams(
        num_frames=frames,
        height=32,
        width=32,
        seed=seed,
        num_inference_steps=4,
        guidance_scale=1,
        extra_args=extra or {},
    )


def batch(*, session="s", index=0, close=False, seed=7, prompt="forest", controls=()):
    tick = ARDiffusionTickRequest(session, f"request-{index}", index, (index,), prompt, controls, close_session=close)
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=sampling(seed=seed, extra=tick.to_extra_args()),
                request_id=tick.request_id,
            )
        ]
    )


def execute(runner, request, session="s"):
    with runner._bound_ar_session(session, description="test"):
        return runner.pipeline.forward(request)


def step_chunk(pipeline, state, cached=None):
    pipeline.prepare_next_chunk(state)
    inputs = InputBatch.make_batch([state], cached_batch=cached)
    for _ in range(state.chunk_num_steps):
        prediction = pipeline.denoise_step(inputs, states=[state])
        pipeline.step_scheduler(state, prediction)
    return pipeline.post_decode(state), inputs


def test_capabilities_and_real_kv_geometry(pipeline):
    assert supports_step_execution(pipeline)
    assert supports_interaction_apply(pipeline)
    assert isinstance(pipeline, SupportsARDiffusionPipeline)
    spec = pipeline.ar_diffusion_kv_cache_spec()
    assert (spec.num_layers, spec.num_kv_heads, spec.head_size) == (2, 2, 4)
    assert spec.segmented.max_retained_tokens == 12 + 4 * 8
    pipeline.sp_size = 2
    assert pipeline.ar_diffusion_kv_cache_spec().num_kv_heads == 1


@pytest.mark.parametrize(
    "name,value",
    [
        ("ar_height", 33),
        ("ar_width", False),
        ("ar_max_segments", 0),
        ("ar_session_capacity", 1.5),
        ("ar_chunk_latent_frames", 121),
    ],
)
def test_invalid_ar_capacity_fails_before_loading(name, value):
    with pytest.raises(ValueError):
        HyNomasARConfig.from_args({name: value})


def test_chunk_plan_preserves_all_pixels_and_segment_boundaries():
    request = normalize_request("first ||| second", sampling(frames=25, extra={"segment_frame_boundaries": [0, 8, 25]}))
    chunks = plan_chunks(request, 3)
    assert [(chunk.segment, chunk.start, chunk.end, chunk.pixel_frames) for chunk in chunks] == [
        (0, 0, 2, 8),
        (1, 2, 5, 12),
        (1, 5, 7, 5),
    ]
    assert sum(chunk.pixel_frames for chunk in chunks) == 25


def test_continuation_keeps_real_kv_noise_and_decoder_timeline(runner):
    first = execute(runner, batch())
    formatted = normalize_diffusion_postprocess_output(first.output)
    assert formatted.metadata["ar_diffusion"]["session_id"] == "s"
    session = runner.pipeline._ar_sessions["s"]
    noise = session.noise.clone()
    decoder = session.decoder
    history = runner._sessions["s"].segmented("cond")
    assert history.history_tokens == 7
    assert len(first.output["payload"]["video"]) == 6
    second = execute(runner, batch(index=1, close=True, prompt="longer prompt"))
    assert session.decoder is decoder and decoder.finished
    torch.testing.assert_close(session.noise, noise)
    assert (session.latent_frames, session.tick_index, session.emitted_frames) == (6, 2, 18)
    assert history.history_tokens == 17
    assert len(second.output["payload"]["video"]) == 12
    assert second.output["metadata"]["hy_nomas"]["frame_start"] == 6
    assert second.output["metadata"]["ar_diffusion"]["applied_event_ids"] == [1]
    assert runner.pipeline._ar_state is None
    calls = runner.pipeline.calls
    assert len(calls) == 10 and sum(call["commit"] for call in calls) == 2
    assert calls[5]["position"] == 3
    assert calls[5]["mask"].tolist() == [[True, True, True, True, True, False, False]]


def test_interleaved_sessions_restore_their_own_pages(runner):
    execute(runner, batch())
    execute(runner, batch(session="other", seed=19), "other")
    execute(runner, batch(index=1))
    assert runner.pipeline._ar_sessions["s"].latent_frames == 6
    assert runner.pipeline._ar_sessions["other"].latent_frames == 3
    torch.testing.assert_close(runner.pipeline.calls[-5]["history"], torch.ones(1, 7, 2, 4))


@pytest.mark.parametrize("bad", ["seed", "index", "shape", "capacity"])
def test_invalid_continuation_fails_closed(runner, bad):
    execute(runner, batch())
    request = batch(index=2 if bad == "index" else 1, seed=8 if bad == "seed" else 7)
    if bad == "shape":
        request.sampling_params.height = 64
    if bad == "capacity":
        runner.pipeline._ar_sessions["s"].latent_frames = 12
    with pytest.raises(OmniClientError):
        execute(runner, request)
    assert "s" not in runner._sessions and "s" not in runner.pipeline._ar_sessions
    assert runner.pipeline._ar_state is None


def test_clean_forward_failure_releases_kv_and_decoder(runner):
    free = runner.kv_cache.manager.block_pool.get_num_free_blocks()
    execute(runner, batch())
    runner.pipeline.fail_clean = True
    with pytest.raises(RuntimeError, match="injected clean-forward failure"):
        execute(runner, batch(index=1))
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free
    assert not runner._sessions and not runner.pipeline._ar_sessions


def test_reset_replays_the_initial_frames_and_close_reclaims_pages(runner):
    free = runner.kv_cache.manager.block_pool.get_num_free_blocks()
    expected = execute(runner, batch()).output["payload"]["video"]
    runner.reset_session("s")
    actual = execute(runner, batch()).output["payload"]["video"]
    np.testing.assert_array_equal(actual, expected)
    runner.close_session("s")
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free
    assert not runner.pipeline._ar_sessions


def test_live_chunks_flush_tail_and_update_prompt_without_trimming(runner):
    pipeline = runner.pipeline
    state = StepRequestState(request_id="live", sampling=sampling(frames=21), prompt="forest")
    with runner._bound_ar_session("live", description="test"):
        pipeline.prepare_encode(state)
        first, inputs = step_chunk(pipeline, state)
        assert not first.finished and len(first.output["payload"]["video"]) == 9
        assert inputs.prompt_embeds.shape[1] == 4
        handler = HyNomasPromptInteractionHandler()
        handler.enqueue(
            state, event_id="change", received_at=2, payload={"prompt": "longer prompt"}, transition_chunks=0
        )
        metadata = handler.apply_at_chunk_boundary(state, boundary_at=3)
        second, inputs = step_chunk(pipeline, state, inputs)
        assert metadata.completed_event_ids == ["change"]
        assert inputs.prompt_embeds.shape[1] == 7
        assert second.finished and len(second.output["payload"]["video"]) == 12
        assert second.output["metadata"]["hy_nomas"]["withheld_frames"] == 0
        assert pipeline.calls[-5]["text_length"] == 7


def test_unchanged_prompt_is_cached_once_and_camera_is_sliced(runner):
    pipeline = runner.pipeline
    state = StepRequestState(
        request_id="live", sampling=sampling(frames=21, extra={"camera_trajectory": "pan_left"}), prompt="forest"
    )
    with runner._bound_ar_session("live", description="test"):
        pipeline.prepare_encode(state)
        step_chunk(pipeline, state)
        step_chunk(pipeline, state)
        assert pipeline.encoded == ["forest"]
        assert pipeline.camera_calls == [("pan_left", 21)]
        assert not pipeline.calls[-5]["cache_txt"]
        assert state.extra["hy_nomas_camera"].flatten().tolist() == [3, 4, 5]


def test_camera_tick_and_live_updates(runner):
    control = ARDiffusionControlInput("camera", "hy_nomas.camera.v1", {"camera_trajectory": "pan_left"})
    execute(runner, batch(controls=(control,)))
    assert runner.pipeline.camera_calls == [("pan_left", 9)]
    state = StepRequestState(request_id="live", sampling=sampling(), prompt="forest")
    handler = HyNomasCameraInteractionHandler()
    for event_id, timestamp, camera in [("old", 1, "pan_left"), ("new", 2, None)]:
        handler.enqueue(
            state, event_id=event_id, received_at=timestamp, payload={"camera_trajectory": camera}, transition_chunks=0
        )
    assert handler.apply_at_chunk_boundary(state, boundary_at=1.5) is None
    metadata = handler.apply_at_chunk_boundary(state, boundary_at=3)
    assert metadata.coalesced_event_ids == ["old"]
    assert metadata.completed_event_ids == ["new"]
    assert state.extra["hy_nomas_camera_override"] is None


def test_invalid_composite_interaction_does_not_queue_prompt():
    state = StepRequestState(request_id="live", sampling=sampling(), prompt="forest")
    coordinator = InteractionCoordinator(
        {"prompt": HyNomasPromptInteractionHandler(), "camera": HyNomasCameraInteractionHandler()}
    )
    with pytest.raises(ValueError, match="camera_trajectory"):
        coordinator.enqueue_parts(
            state,
            parts=[("prompt", {"prompt": "changed"}), ("camera", {"invalid": 1})],
            event_id="bad",
            received_at=1,
            transition_chunks=0,
        )
    assert not state.interaction_sessions


def test_independent_request_reuses_default_session_without_old_state(runner):
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                request_id="ordinary",
                prompt="forest",
                sampling_params=sampling(),
            )
        ]
    )
    first = execute(runner, request, "default")
    second = execute(runner, request, "default")
    np.testing.assert_array_equal(first.output["payload"]["video"], second.output["payload"]["video"])
    assert runner.pipeline._ar_sessions["default"].latent_frames == 3


def test_cancel_releases_step_session_without_another_execution_wave(runner):
    free = runner.kv_cache.manager.block_pool.get_num_free_blocks()
    execute(runner, batch(session="live"), "live")
    runner.release_step_requests(["live"])
    assert not runner.pipeline._ar_sessions and not runner._sessions
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free


def test_incomplete_chunk_cannot_commit(runner):
    state = StepRequestState(request_id="live", sampling=sampling(), prompt="forest")
    with runner._bound_ar_session("live", description="test"):
        runner.pipeline.prepare_encode(state)
        runner.pipeline.prepare_next_chunk(state)
        with pytest.raises(RuntimeError, match="complete every denoising step"):
            runner.pipeline.post_decode(state)
        assert runner._sessions["live"].segmented("cond").history_tokens == 0


def test_empty_decoder_chunk_preserves_tick_metadata(runner):
    runner.pipeline.vae.create_streaming_decode_session = lambda **kwargs: DecoderStub(hold_frames=100)
    request = batch()
    output = execute(runner, request)
    [formatted] = format_diffusion_outputs(
        request=request.requests[0],
        od_config=OmniDiffusionConfig(model_class_name="HyNomasPipeline"),
        diffusion_output=output,
        output_data=output.output,
        postprocess_output=normalize_diffusion_postprocess_output(output.output),
    )
    assert len(formatted.images) == 1
    assert formatted.images[0].shape == (0, 32, 32, 3)
    assert formatted.multimodal_output["metadata"]["ar_diffusion"]["chunk_index"] == 0
    assert len(execute(runner, batch(index=1, close=True)).output["payload"]["video"]) == 18


def test_public_deploy_resolution_preserves_ar_configuration(tmp_path):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"_class_name": "HyNomasModularPipeline"}) if relative == "modular_model_index.json" else "{}"
        )
    deployment = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/hy_nomas_ar.yaml"
    resolution = resolve_omni_config(
        str(tmp_path),
        trust_remote_code=None,
        deploy_config_path=str(deployment),
        cli_overrides={},
        stage_overrides=None,
        strategy_config_path=None,
    )
    [stage] = resolution.stage_configs
    assert stage.diffusion_config.engine_backend == "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
    assert stage.diffusion_config.step_execution and stage.diffusion_config.streaming_output
    assert stage.parallel_config.ulysses_degree == 2
    assert stage.diffusion_config.custom_pipeline_args["ar_max_latent_frames"] == 120


def test_offline_deploy_selects_isolated_executor(tmp_path):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"_class_name": "HyNomasModularPipeline"}) if relative == "modular_model_index.json" else "{}"
        )
    deployment = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/hy_nomas.yaml"
    resolution = resolve_omni_config(
        str(tmp_path),
        trust_remote_code=None,
        deploy_config_path=str(deployment),
        cli_overrides={},
        stage_overrides=None,
        strategy_config_path=None,
    )
    [stage] = resolution.stage_configs
    assert stage.diffusion_config.distributed_executor_backend == "mp"
    assert stage.model_config.enforce_eager
    assert not stage.diffusion_config.step_execution


@pytest.mark.parametrize("local_failure", [False, True])
def test_decode_failure_is_propagated_to_every_sp_rank(runner, monkeypatch, local_failure):
    flags = []

    def reduce_status(value):
        flags.append(value.item())
        return torch.ones_like(value)

    def fail_decode(*args, **kwargs):
        raise RuntimeError("injected decode failure")

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas.get_sp_group",
        lambda: SimpleNamespace(all_reduce=reduce_status),
    )
    state = StepRequestState(request_id="live", sampling=sampling(), prompt="forest")
    free = runner.kv_cache.manager.block_pool.get_num_free_blocks()
    match = "injected decode failure" if local_failure else "another SP rank"
    with pytest.raises(RuntimeError, match=match), runner._bound_ar_session("live", description="test"):
        pipeline = runner.pipeline
        pipeline.prepare_encode(state)
        pipeline.prepare_next_chunk(state)
        for _ in range(state.chunk_num_steps):
            pipeline.step_scheduler(state, pipeline.denoise_step(None, states=[state]))
        if local_failure:
            monkeypatch.setattr(pipeline._session().decoder, "push", fail_decode)
        pipeline.sp_size = 2
        pipeline.post_decode(state)
    assert flags == [int(local_failure)]
    assert not runner._sessions and not runner.pipeline._ar_sessions
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free


def test_cache_commit_timestep_keeps_reference_precision(runner):
    with set_default_torch_dtype(torch.bfloat16):
        execute(runner, batch())
    assert runner.pipeline.calls[-1]["commit"]
    assert runner.pipeline.calls[-1]["timestep_dtype"] == torch.float32
