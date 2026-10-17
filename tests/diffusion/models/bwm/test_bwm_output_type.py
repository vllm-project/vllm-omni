# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 tests for BWM ``output_type`` at the DiffusionEngine boundary (CPU).

The engine builds the BWM post-process hook from the registry and passes the
request's ``output_type`` only through ``sampling_params``; these tests drive
``DiffusionEngine.postprocess_output`` with a stubbed executor.
"""

from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.bwm.pipeline_bwm import BoundlessWorldModelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_FRAMES, HEIGHT, WIDTH = 5, 32, 48


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, mocker) -> DiffusionEngine:
    od_config = SimpleNamespace(
        model_class_name="BoundlessWorldModelPipeline",
        custom_pipeline_args=None,
        streaming_output=False,
        enable_cpu_offload=False,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
        lambda *args, **kwargs: mocker.Mock(),
    )
    monkeypatch.setattr(DiffusionEngine, "_dummy_run", lambda self: None)
    return DiffusionEngine(od_config)


def _postprocess(engine: DiffusionEngine, output: torch.Tensor, output_type: str | None):
    request = OmniDiffusionRequest(
        prompt={"prompt": ""},
        sampling_params=OmniDiffusionSamplingParams(output_type=output_type),
        request_id="req-0",
    )
    results = engine.postprocess_output(request, DiffusionOutput(output=output), 0.0, 0.0, 0.0)
    assert len(results) == 1
    assert len(results[0].images) == 1
    return results[0].images[0]


def _decoded_video() -> torch.Tensor:
    # VAE decode output: (B, C, T, H, W) in [-1, 1].
    return torch.rand(1, 3, NUM_FRAMES, HEIGHT, WIDTH) * 2 - 1


def test_latent_passes_through_unchanged(engine):
    latents = torch.randn(1, 48, 2, HEIGHT // 16, WIDTH // 16)
    expected = latents.clone()
    out = _postprocess(engine, latents, "latent")
    assert isinstance(out, torch.Tensor)
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


@pytest.mark.parametrize("output_type", [None, "np"])
def test_numpy_is_default(engine, output_type):
    out = _postprocess(engine, _decoded_video(), output_type)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, NUM_FRAMES, HEIGHT, WIDTH, 3)


def test_pt_returns_tensor(engine):
    video = _decoded_video()
    out = _postprocess(engine, video, "pt")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, NUM_FRAMES, 3, HEIGHT, WIDTH)
    expected = (video.permute(0, 2, 1, 3, 4) / 2 + 0.5).clamp(0, 1)
    torch.testing.assert_close(out, expected)


def test_pil_returns_frames(engine):
    out = _postprocess(engine, _decoded_video(), "pil")
    assert len(out) == NUM_FRAMES
    assert all(isinstance(frame, PIL.Image.Image) for frame in out)
    assert out[0].size == (WIDTH, HEIGHT)


def test_unsupported_type_rejected_by_hook(engine):
    with pytest.raises(ValueError, match="output_type"):
        _postprocess(engine, _decoded_video(), "mp4")


def test_unsupported_type_rejected_at_forward_ingress():
    pipeline = object.__new__(BoundlessWorldModelPipeline)
    req = SimpleNamespace(prompts=[{}], sampling_params=OmniDiffusionSamplingParams(output_type="mp4"))
    with pytest.raises(ValueError, match="output_type"):
        pipeline.forward(req)
