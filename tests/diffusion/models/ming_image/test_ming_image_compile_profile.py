# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_image.compile_profile import MingImageCompileBucket
from vllm_omni.diffusion.models.ming_image.engine import MingImageCompileWorkerExtension, MingImageDiffusionEngine
from vllm_omni.diffusion.models.ming_image.pipeline import MingImageDiffusionPipeline
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def engine(pipeline, mocker) -> MingImageDiffusionEngine:
    engine = MingImageDiffusionEngine.__new__(MingImageDiffusionEngine)
    engine.od_config = SimpleNamespace(additional_config={})
    engine.close = mocker.Mock()
    worker = MingImageCompileWorkerExtension()
    worker.model_runner = SimpleNamespace(vllm_config=None, od_config=pipeline.od_config, pipeline=pipeline)
    # Replace transport only; run the actual worker extension and pipeline helper.
    engine.collective_rpc = mocker.Mock(side_effect=lambda method: getattr(worker, method)())
    return engine


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, mocker) -> MingImageDiffusionPipeline:
    # Keep Ming's request handling real; replace weight-backed computation.
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline.device = torch.device("cpu")
    pipeline.is_layer_decomposition = False
    pipeline.compile_buckets = ()
    pipeline.default_guidance_scale = 1.0
    pipeline.default_num_inference_steps = 12
    pipeline.conditioning = mocker.Mock(
        query_encoder=SimpleNamespace(thinker_hidden_size=2),
        direct_projector=[None, SimpleNamespace(in_features=6)],
        side_effect=lambda query, direct: (
            torch.ones(*query.shape[:2], 4),
            torch.ones(*direct.shape[:2], 4),
        ),
    )
    pipeline.transformer = mocker.Mock(in_channels=4)
    pipeline.vae_scale_factor = 8
    monkeypatch.setattr(pipeline, "_encode_reference", mocker.Mock(return_value=None))
    monkeypatch.setattr(pipeline, "_decode_latent_frames", mocker.Mock(side_effect=lambda latents: latents))
    monkeypatch.setattr(
        ZImagePipeline,
        "forward",
        mocker.Mock(
            side_effect=lambda req: DiffusionOutput(output=torch.zeros(pipeline._num_frames_per_prompt, 1, 1, 1))
        ),
    )
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", mocker.Mock())
    monkeypatch.setattr(torch.accelerator, "synchronize", mocker.Mock())
    return pipeline


@pytest.mark.parametrize(
    "layered,guidance,has_reference,batch",
    [(False, None, False, 1), (True, None, True, 2), (False, 2.0, True, 2), (True, 1.0, True, 1)],
    ids=["design", "design-layer", "design-edit-cfg", "layer-no-cfg"],
)
def test_startup_warms_transformer_buckets(
    engine, pipeline, layered: bool, guidance: float | None, has_reference: bool, batch: int
) -> None:
    buckets = [(1024, 1024, 3), (768, 1024, 6)] if layered else [(1024, 1024, 1), (768, 1024, 1)]
    engine.od_config.additional_config = {
        "compile_buckets": [
            dict(
                height=height,
                width=width,
                num_layers=layers,
                num_condition_tokens=tokens,
                guidance_scale=guidance,
                has_reference=has_reference and not layered,
            )
            for (height, width, layers), tokens in zip(buckets, [3, 7])
        ]
    }
    pipeline.compile_buckets = tuple(
        MingImageCompileBucket(**bucket) for bucket in engine.od_config.additional_config["compile_buckets"]
    )
    pipeline.is_layer_decomposition = layered
    pipeline.default_guidance_scale = 2.0 if layered else 1.0
    observed_shapes: set[tuple[int, ...]] = set()

    def transformer_forward(latents, timestep, cap_feats) -> None:
        context = get_forward_context()
        observed_shapes.add((len(latents), *latents[0].shape, context.direct_condition.shape[1]))
        assert len(latents) == timestep.numel() == len(cap_feats)
        assert torch.all(context.direct_condition[0] == 1)
        assert torch.all(cap_feats[0] == 1)
        if batch == 2:
            torch.testing.assert_close(context.direct_condition[1], torch.zeros_like(context.direct_condition[1]))
            torch.testing.assert_close(cap_feats[1], torch.zeros_like(cap_feats[1]))
        if has_reference:
            assert context.ref_latent.shape == (len(latents), *latents[0][:, :1].shape)
        else:
            assert context.ref_latent is None

    pipeline.transformer.side_effect = transformer_forward

    engine.run_startup_warmup()

    assert observed_shapes == {
        (
            batch,
            pipeline.transformer.in_channels,
            layers + 1 if layered else 1,
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor,
            tokens,
        )
        for (height, width, layers), tokens in zip(buckets, [3, 7])
    }
    assert [call.args[1].shape for call in pipeline.conditioning.call_args_list] == [(1, 3, 6), (1, 7, 6)]
    ZImagePipeline.forward.assert_not_called()


def test_no_profile_keeps_generic_startup_warmup(engine, monkeypatch: pytest.MonkeyPatch, mocker) -> None:
    generic_warmup = mocker.Mock()
    monkeypatch.setattr(DiffusionEngine, "_dummy_run", generic_warmup)

    engine.run_startup_warmup()

    generic_warmup.assert_called_once_with()
    engine.collective_rpc.assert_not_called()


@pytest.mark.parametrize("profile_enabled", [False, True], ids=["no-profile", "profile"])
@pytest.mark.parametrize("tokens", [3, 7], ids=["warmed-conditioning", "other-conditioning"])
def test_forward_preserves_resolved_shape_and_layers(pipeline, profile_enabled: bool, tokens: int) -> None:
    pipeline.is_layer_decomposition = True
    pipeline.default_guidance_scale = 2.0
    pipeline.compile_buckets = (MingImageCompileBucket(768, 1024, 3, 3),) if profile_enabled else ()
    request = OmniDiffusionRequest(
        prompt={
            "extra": {
                "query_hidden_states": torch.ones(1, 2),
                "direct_hidden_states": torch.ones(tokens, 6),
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
    pipeline.compile_buckets = (MingImageCompileBucket(1024, 1024, 3, 3),)
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

    with pytest.raises(ValueError, match="outside compile_buckets"):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))

    ZImagePipeline.forward.assert_not_called()


def test_compile_profile_requires_16_divisible_resolution() -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        MingImageCompileBucket(1024, 1000, 1, 7)
