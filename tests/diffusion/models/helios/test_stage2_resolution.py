# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
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


@pytest.mark.parametrize(
    "height,width,stages",
    [(400, 624, 1), (416, 640, 2), (384, 640, 3), (640, 384, 3), (512, 512, 3), (512, 768, 4)],
)
def test_supported_stage2_resolution(pipeline, height, width, stages):
    pipeline._validate_stage2_resolution(height, width, stages)


@pytest.mark.parametrize("height,width", [(400, 624), (624, 400), (528, 752), (512, 288), (400, 640)])
def test_three_stage_pyramid_rejects_patch_truncation_and_extent_loss(pipeline, height, width):
    with pytest.raises(ValueError, match="positive multiples of 64 and 64"):
        pipeline._validate_stage2_resolution(height, width, 3)


@pytest.mark.parametrize("height,width,stages", [(400, 624, 2), (448, 640, 4), (0, 640, 3), (384, 0, 3)])
def test_alignment_depends_on_stage_count_and_requires_positive_dimensions(pipeline, height, width, stages):
    with pytest.raises(ValueError, match="positive multiples"):
        pipeline._validate_stage2_resolution(height, width, stages)


@pytest.mark.parametrize("stages", [0, -1, True, False, 2.5, "3", None])
def test_invalid_stage_count_rejected(pipeline, stages):
    with pytest.raises(ValueError, match="pyramid_num_stages must be a positive integer"):
        pipeline._validate_stage2_resolution(384, 640, stages)


def test_alignment_uses_actual_spatial_scale_and_asymmetric_patch(pipeline):
    pipeline.vae_scale_factor_spatial = 4
    pipeline.transformer.config.patch_size = (1, 2, 4)
    pipeline._validate_stage2_resolution(48, 64, 2)
    with pytest.raises(ValueError, match="positive multiples of 16 and 32"):
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
    with pytest.raises(ValueError, match=f"got {height}x{width} after 16-pixel alignment"):
        _run_request(pipeline, execution, sampling)
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


@pytest.mark.parametrize("execution", ["request_batch", "step"])
def test_stage2_disabled_preserves_stage1_resolution_handling(pipeline, execution):
    pipeline.encode_prompt.side_effect = RuntimeError("encoding reached")
    sampling = OmniDiffusionSamplingParams(height=400, width=624, seed=42, extra_args={"is_enable_stage2": False})
    with pytest.raises(RuntimeError, match="encoding reached"):
        _run_request(pipeline, execution, sampling)
    pipeline.encode_prompt.assert_called_once()
