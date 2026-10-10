# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_image.engine import MingImageCompileWorkerExtension, MingImageDiffusionEngine
from vllm_omni.diffusion.models.ming_image.pipeline import MingImageDiffusionPipeline
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def engine(pipeline) -> MingImageDiffusionEngine:
    engine = MingImageDiffusionEngine.__new__(MingImageDiffusionEngine)
    engine.od_config = SimpleNamespace(additional_config={})
    engine.close = Mock()
    engine._make_dummy_request = Mock()
    engine.add_req_and_wait_for_response = Mock()
    worker = MingImageCompileWorkerExtension()
    worker.model_runner = SimpleNamespace(vllm_config=None, od_config=pipeline.od_config, pipeline=pipeline)
    # Replace transport only; run the actual worker extension and pipeline helper.
    engine.collective_rpc = Mock(side_effect=lambda method: getattr(worker, method)())
    return engine


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> MingImageDiffusionPipeline:
    # Keep Ming's request handling real; replace weight-backed computation.
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline.device = torch.device("cpu")
    pipeline.is_layer_decomposition = False
    pipeline.compile_buckets = ()
    pipeline.default_guidance_scale = 1.0
    pipeline.default_num_inference_steps = 12
    pipeline.conditioning = Mock(return_value=(torch.ones(1, 2, 4), torch.ones(1, 1, 4)))
    pipeline.transformer = Mock(in_channels=4)
    pipeline.vae_scale_factor = 8
    monkeypatch.setattr(pipeline, "_encode_reference", Mock(return_value=None))
    monkeypatch.setattr(pipeline, "_decode_latent_frames", Mock(side_effect=lambda latents: latents))
    monkeypatch.setattr(
        ZImagePipeline,
        "forward",
        Mock(side_effect=lambda req: DiffusionOutput(output=torch.zeros(pipeline._num_frames_per_prompt, 1, 1, 1))),
    )
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", Mock())
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    return pipeline


@pytest.mark.parametrize("layered", [False, True], ids=["design", "design-layer"])
def test_startup_warms_transformer_buckets_with_variant_conditioning(engine, pipeline, layered: bool) -> None:
    buckets = [(1024, 1024, 3), (768, 1024, 6)] if layered else [(1024, 1024, 1), (768, 1024, 1)]
    engine.od_config.additional_config = {
        "ming_image_compile_buckets": [
            dict(height=height, width=width, num_layers=layers) for height, width, layers in buckets
        ]
    }
    pipeline.compile_buckets = tuple(buckets)
    pipeline.is_layer_decomposition = layered
    pipeline.default_guidance_scale = 2.0 if layered else 1.0
    observed_shapes: set[tuple[int, ...]] = set()

    def transformer_forward(latents, timestep, cap_feats) -> None:
        context = get_forward_context()
        observed_shapes.add((len(latents), *latents[0].shape))
        assert len(latents) == timestep.numel() == len(cap_feats)
        torch.testing.assert_close(context.direct_condition[:1], pipeline.conditioning.return_value[1])
        torch.testing.assert_close(cap_feats[0], pipeline.conditioning.return_value[0][0])
        if layered:
            torch.testing.assert_close(context.direct_condition[1], torch.zeros_like(context.direct_condition[1]))
            torch.testing.assert_close(cap_feats[1], torch.zeros_like(cap_feats[1]))
            assert context.ref_latent.shape == (len(latents), *latents[0][:, :1].shape)
        else:
            assert context.ref_latent is None

    pipeline.transformer.side_effect = transformer_forward

    engine.run_startup_warmup()

    assert observed_shapes == {
        (
            2 if layered else 1,
            pipeline.transformer.in_channels,
            layers + 1 if layered else 1,
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor,
        )
        for height, width, layers in buckets
    }
    engine.collective_rpc.assert_called_once_with("warmup_ming_image_compile_buckets")
    engine._make_dummy_request.assert_not_called()
    engine.add_req_and_wait_for_response.assert_not_called()
    ZImagePipeline.forward.assert_not_called()
    pipeline._encode_reference.assert_not_called()
    pipeline._decode_latent_frames.assert_not_called()


def test_no_profile_keeps_generic_startup_warmup(engine, monkeypatch: pytest.MonkeyPatch) -> None:
    generic_warmup = Mock()
    monkeypatch.setattr(DiffusionEngine, "_dummy_run", generic_warmup)

    engine.run_startup_warmup()

    generic_warmup.assert_called_once_with()
    engine.collective_rpc.assert_not_called()
    engine.add_req_and_wait_for_response.assert_not_called()


def test_failed_transformer_warmup_closes_engine_and_restores_context(engine, pipeline) -> None:
    engine.od_config.additional_config = {
        "ming_image_compile_buckets": [{"height": 1024, "width": 1024, "num_layers": 1}]
    }
    pipeline.compile_buckets = ((1024, 1024, 1),)
    pipeline.transformer.side_effect = RuntimeError("compile failed")

    with set_forward_context():
        previous_context = get_forward_context()
        with pytest.raises(RuntimeError, match="compile failed"):
            engine.run_startup_warmup()
        assert get_forward_context() is previous_context

    engine.close.assert_called_once_with()


@pytest.mark.parametrize("profile_enabled", [False, True], ids=["no-profile", "profile"])
def test_forward_preserves_resolved_shape_and_layers(pipeline, profile_enabled: bool) -> None:
    pipeline.is_layer_decomposition = True
    pipeline.default_guidance_scale = 2.0
    pipeline.compile_buckets = ((768, 1024, 3),) if profile_enabled else ()
    request = OmniDiffusionRequest(
        prompt={
            "extra": {
                "query_hidden_states": torch.ones(1, 2),
                "direct_hidden_states": torch.ones(1, 2),
                "reference_image": object(),
            }
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=512,
            width=512,
            extra_args={"height": 768, "width": 1024, "num_layers": 3},
            seed=42,
        ),
        request_id="real-request",
    )

    with set_forward_context():
        output = pipeline.forward(DiffusionRequestBatch(requests=[request]))

    inner_request = ZImagePipeline.forward.call_args.args[0]
    assert (inner_request.sampling_params.height, inner_request.sampling_params.width) == (768, 1024)
    assert output.output.shape[0] == 4


@pytest.mark.parametrize("shape", [(768, 1024, 3), (1024, 1024, 6)], ids=["resolution", "layer-count"])
def test_forward_rejects_unprofiled_shape_before_denoising(pipeline, shape: tuple[int, int, int]) -> None:
    pipeline.compile_buckets = ((1024, 1024, 3),)
    pipeline.is_layer_decomposition = True
    height, width, layers = shape
    request = OmniDiffusionRequest(
        prompt={
            "extra": {
                "query_hidden_states": torch.ones(1, 2),
                "direct_hidden_states": torch.ones(1, 2),
                "reference_image": object(),
            }
        },
        sampling_params=OmniDiffusionSamplingParams(height=height, width=width, extra_args={"num_layers": layers}),
        request_id="unprofiled-request",
    )

    with pytest.raises(ValueError, match="outside ming_image_compile_buckets"):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))

    ZImagePipeline.forward.assert_not_called()
