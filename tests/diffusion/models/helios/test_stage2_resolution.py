# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.helios.pipeline_helios as pipeline_module
import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.models.helios.pipeline_helios import (
    HeliosPipeline,
    get_helios_post_process_func,
)
from vllm_omni.diffusion.models.helios.scheduling_helios import HeliosScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def pipeline(mocker):
    pipeline = object.__new__(HeliosPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor_spatial = 8
    pipeline.transformer = mocker.Mock()
    pipeline.transformer.config.patch_size = (1, 2, 2)
    pipeline.transformer.dtype = torch.float32
    pipeline.device = torch.device("cpu")
    pipeline.encode_prompt = mocker.Mock()
    return pipeline


def _configure_step_execution_pipeline(pipeline: HeliosPipeline, mocker) -> None:
    pipeline.vae = SimpleNamespace(
        config=SimpleNamespace(
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
            z_dim=4,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
        decode=mocker.Mock(return_value=(torch.zeros(1, 3, 2, 16, 16),)),
    )
    pipeline.vae_scale_factor_temporal = 4
    pipeline.transformer.config.in_channels = 4
    pipeline.transformer.side_effect = lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
    pipeline.scheduler = HeliosScheduler(scheduler_type="euler")
    pipeline.is_distilled = False
    pipeline._interrupt = False
    pipeline._guidance_scale = None
    pipeline.encode_prompt.return_value = (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))


@pytest.mark.parametrize(
    "height,width,stages",
    [(400, 624, 1), (416, 640, 2), (384, 640, 3), (640, 384, 3), (512, 512, 3), (512, 768, 4)],
)
def test_supported_stage2_resolution(pipeline, height, width, stages):
    pipeline._validate_stage2_resolution(height, width, stages)


@pytest.mark.parametrize("height,width", [(400, 624), (624, 400), (528, 752), (512, 288), (400, 640)])
def test_three_stage_pyramid_rejects_patch_truncation_and_extent_loss(pipeline, height, width):
    with pytest.raises(OmniClientError, match="positive multiples of 64 and 64") as exc_info:
        pipeline._validate_stage2_resolution(height, width, 3)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("height,width,stages", [(400, 624, 2), (448, 640, 4), (0, 640, 3), (384, 0, 3)])
def test_alignment_depends_on_stage_count_and_requires_positive_dimensions(pipeline, height, width, stages):
    with pytest.raises(OmniClientError, match="positive multiples") as exc_info:
        pipeline._validate_stage2_resolution(height, width, stages)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("stages", [0, -1, True, False, 2.5, "3", None])
def test_invalid_stage_count_rejected(pipeline, stages):
    with pytest.raises(OmniClientError, match="pyramid_num_stages must be a positive integer") as exc_info:
        pipeline._validate_stage2_resolution(384, 640, stages)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("stages", [6, 20000, pytest.param(10**5000, id="oversized-int")])
def test_infeasible_stage_count_rejected_before_computing_multiples(pipeline, stages):
    with pytest.raises(OmniClientError, match="exceeds the maximum 5 stage\\(s\\) supported for 384x640") as exc_info:
        pipeline._validate_stage2_resolution(384, 640, stages)
    assert exc_info.value.status_code == 400


def test_alignment_uses_actual_spatial_scale_and_asymmetric_patch(pipeline):
    pipeline.vae_scale_factor_spatial = 4
    pipeline.transformer.config.patch_size = (1, 2, 4)
    pipeline._validate_stage2_resolution(48, 64, 2)
    with pytest.raises(OmniClientError, match="positive multiples of 16 and 32"):
        pipeline._validate_stage2_resolution(48, 48, 2)


def _run_request(pipeline, execution, sampling):
    prompt = {"prompt": "A train crossing a landscape."}
    if execution == "request_batch":
        request = OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id="geometry-test")
        pipeline.forward(DiffusionRequestBatch(requests=[request]))
    else:
        pipeline.prepare_encode(StepRequestState(request_id="geometry-test", sampling=sampling, prompt=prompt))


@pytest.mark.parametrize("execution", ["request_batch", "step"])
@pytest.mark.parametrize("height,width", [(400, 624), (624, 400), (528, 752), (512, 288), (400, 640)])
def test_both_entrypoints_reject_before_prompt_encoding(pipeline, execution, height, width):
    pipeline.encode_prompt.side_effect = AssertionError("unsupported geometry reached prompt encoding")
    sampling = OmniDiffusionSamplingParams(height=height, width=width, seed=42, extra_args={"is_enable_stage2": True})
    with pytest.raises(OmniClientError, match=f"got {height}x{width} after 16-pixel alignment") as exc_info:
        _run_request(pipeline, execution, sampling)
    assert exc_info.value.status_code == 400
    pipeline.encode_prompt.assert_not_called()
    pipeline.transformer.assert_not_called()


@pytest.mark.parametrize("execution", ["request_batch", "step"])
@pytest.mark.parametrize("stages", [20000, pytest.param(10**5000, id="oversized-int")])
def test_both_entrypoints_reject_infeasible_stage_count_before_prompt_encoding(pipeline, execution, stages):
    pipeline.encode_prompt.side_effect = AssertionError("infeasible stage count reached prompt encoding")
    sampling = OmniDiffusionSamplingParams(
        height=384,
        width=640,
        seed=42,
        extra_args={"is_enable_stage2": True, "pyramid_num_stages": stages},
    )
    with pytest.raises(OmniClientError, match="exceeds the maximum 5 stage\\(s\\) supported for 384x640") as exc_info:
        _run_request(pipeline, execution, sampling)
    assert exc_info.value.status_code == 400
    pipeline.encode_prompt.assert_not_called()
    pipeline.transformer.assert_not_called()


@pytest.mark.parametrize("execution", ["request_batch", "step"])
@pytest.mark.parametrize("height,width,stages", [(400, 624, 1), (416, 640, 2), (384, 640, 3), (399, 655, 3)])
def test_both_entrypoints_accept_resolved_supported_geometry(pipeline, execution, height, width, stages):
    # Stop at encoding: these tests need no model weights or tensor denoising.
    pipeline.encode_prompt.side_effect = RuntimeError("encoding reached")
    sampling = OmniDiffusionSamplingParams(
        height=height, width=width, seed=42, extra_args={"is_enable_stage2": True, "pyramid_num_stages": stages}
    )
    with pytest.raises(RuntimeError, match="encoding reached"):
        _run_request(pipeline, execution, sampling)
    pipeline.encode_prompt.assert_called_once()


def test_step_prepare_encode_accepts_string_stage_count_and_reuses_parsed_value(pipeline, mocker):
    _configure_step_execution_pipeline(pipeline, mocker)
    sampling = OmniDiffusionSamplingParams(
        height=384,
        width=640,
        seed=42,
        extra_args={"is_enable_stage2": True, "pyramid_num_stages": "3"},
    )
    state = StepRequestState(
        request_id="step-string-stages",
        sampling=sampling,
        prompt={"prompt": "A train crossing a landscape."},
    )

    prepared = pipeline.prepare_encode(state)
    pipeline.prepare_next_chunk(prepared)

    pipeline.encode_prompt.assert_called_once()
    assert prepared.extra["pyramid_num_stages"] == 3
    assert type(prepared.extra["pyramid_num_stages"]) is int
    assert prepared.extra["stage2_height"] == 12
    assert prepared.extra["stage2_width"] == 20


def test_step_prepare_encode_rejects_incompatible_resolution_for_string_stage_count(pipeline):
    pipeline.encode_prompt.side_effect = AssertionError("unsupported geometry reached prompt encoding")
    sampling = OmniDiffusionSamplingParams(
        height=400,
        width=624,
        seed=42,
        extra_args={"is_enable_stage2": True, "pyramid_num_stages": "3"},
    )

    with pytest.raises(OmniClientError, match="positive multiples of 64 and 64") as exc_info:
        _run_request(pipeline, "step", sampling)
    assert exc_info.value.status_code == 400
    pipeline.encode_prompt.assert_not_called()


@pytest.mark.parametrize("stages", ["0", "-1", "abc", "2.5", pytest.param("9" * 5000, id="oversized-int-str")])
def test_step_prepare_encode_rejects_invalid_string_stage_count_before_encoding(pipeline, stages):
    pipeline.encode_prompt.side_effect = AssertionError("invalid stage count reached prompt encoding")
    sampling = OmniDiffusionSamplingParams(
        height=384,
        width=640,
        seed=42,
        extra_args={"is_enable_stage2": True, "pyramid_num_stages": stages},
    )

    with pytest.raises(OmniClientError, match="pyramid_num_stages must be a positive integer") as exc_info:
        _run_request(pipeline, "step", sampling)
    assert exc_info.value.status_code == 400
    pipeline.encode_prompt.assert_not_called()


@pytest.mark.parametrize("execution", ["request_batch", "step"])
def test_stage2_disabled_preserves_stage1_resolution_handling(pipeline, execution):
    pipeline.encode_prompt.side_effect = RuntimeError("encoding reached")
    sampling = OmniDiffusionSamplingParams(height=400, width=624, seed=42, extra_args={"is_enable_stage2": False})
    with pytest.raises(RuntimeError, match="encoding reached"):
        _run_request(pipeline, execution, sampling)
    pipeline.encode_prompt.assert_called_once()


def test_sync_video_step_mode_returns_400_for_unsupported_stage2_and_recovers(pipeline, monkeypatch, mocker):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_omni.entrypoints.openai.api_server import router
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

    _configure_step_execution_pipeline(pipeline, mocker)

    od_config = OmniDiffusionConfig(
        model="helios-test",
        model_class_name="HeliosPipeline",
        step_execution=True,
        streaming_output=False,
    )

    @contextlib.contextmanager
    def _noop_forward_context(*args, **kwargs):
        del args, kwargs
        yield

    @contextlib.contextmanager
    def _noop_priority(*args, **kwargs):
        del args, kwargs
        yield

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    cpu_platform = SimpleNamespace(
        is_available=lambda: False,
        is_npu=lambda: False,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )
    monkeypatch.setattr(model_runner_module, "current_omni_platform", cpu_platform)
    monkeypatch.setattr(pipeline_module, "current_omni_platform", cpu_platform)

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace(set_priority=_noop_priority)),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
    )
    runner.od_config = od_config
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.prompt_embed_cache = None
    runner.input_batch = None
    runner._interaction_coordinator = InteractionCoordinator({}, model_class_name="HeliosPipeline")
    pipeline._interaction_coordinator = runner._interaction_coordinator
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None
    )
    runner._kv_prefetch_enabled = False

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = od_config
    engine.pre_process_func = None
    engine.post_process_func = get_helios_post_process_func(od_config)
    engine._post_process_accepts_sampling_params = False
    engine.prefix_cache_func = None
    engine._diffusion_kv_profile_limits = None
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine.step_execution = True
    engine.supports_request_batch = False
    engine.scheduler = StepScheduler()
    engine.scheduler.initialize(od_config)
    engine.executor = SimpleNamespace(
        execute_step=runner.execute_stepwise,
        shutdown=lambda: None,
    )
    engine.execute_fn = engine.executor.execute_step
    engine._init_runtime_state()

    class _StepEngineClient:
        def __init__(self, diffusion_engine: DiffusionEngine):
            self.diffusion_engine = diffusion_engine
            self.stage_configs = [
                SimpleNamespace(
                    stage_type="diffusion",
                    final_output=True,
                    final_output_type="video",
                    engine_args=SimpleNamespace(model_class_name="HeliosPipeline"),
                )
            ]
            self.default_sampling_params_list = [OmniDiffusionSamplingParams(num_inference_steps=3)]

        def get_diffusion_od_config(self) -> OmniDiffusionConfig:
            return self.diffusion_engine.od_config

        async def generate(self, prompt, request_id, sampling_params_list):
            request = OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=sampling_params_list[0],
                request_id=request_id,
            )
            async for outputs in self.diffusion_engine.step_streaming(request):
                for output in outputs:
                    yield output

        async def abort(self, request_id, *, timeout=None):
            del timeout
            self.diffusion_engine.abort(request_id)

    encode_video = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-helios-mp4",
    )

    engine_client = _StepEngineClient(engine)
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine_client,
        model_name="Helios",
        stage_configs=engine_client.stage_configs,
    )

    app = FastAPI()
    app.state.api_server_count = 1
    app.include_router(router)
    app.state.openai_serving_video = handler

    try:
        with TestClient(app) as client:
            bad_response = client.post(
                "/v1/videos/sync",
                data={
                    "prompt": "A train crossing a landscape.",
                    "width": "624",
                    "height": "400",
                    "num_frames": "1",
                    "num_inference_steps": "3",
                    "extra_params": json.dumps(
                        {
                            "is_enable_stage2": True,
                            "pyramid_num_stages": 3,
                            "pyramid_num_inference_steps_list": [1, 1, 1],
                        }
                    ),
                },
            )
            assert bad_response.status_code == 400
            detail = bad_response.json()["detail"]
            assert "positive multiples of 64 and 64" in detail
            assert "400x624" in detail
            pipeline.encode_prompt.assert_not_called()
            assert runner.state_cache == {}

            good_response = client.post(
                "/v1/videos/sync",
                data={
                    "prompt": "A train crossing a landscape.",
                    "width": "640",
                    "height": "384",
                    "num_frames": "1",
                    "num_inference_steps": "3",
                    "extra_params": json.dumps(
                        {
                            "is_enable_stage2": True,
                            "pyramid_num_stages": "3",
                            "pyramid_num_inference_steps_list": [1, 1, 1],
                        }
                    ),
                },
            )
            assert good_response.status_code == 200
            assert good_response.headers["content-type"] == "video/mp4"
            assert good_response.content == b"fake-helios-mp4"
            pipeline.encode_prompt.assert_called_once()
            pipeline.vae.decode.assert_called_once()
            encode_video.assert_called_once()
            assert runner.state_cache == {}
    finally:
        engine.close()
        handler.shutdown()
