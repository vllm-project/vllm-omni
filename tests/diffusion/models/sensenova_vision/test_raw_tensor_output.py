# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for ``output_type=\"raw_tensor\"`` in SenseNovaVisionPipeline.

Offline inference can opt into upstream ``decode_image(output_raw_tensor=True)``
semantics: the pipeline returns raw float32 HxWx3 VAE tensors instead of 8-bit
PIL images.  The flag is request-scoped (``OmniDiffusionSamplingParams.output_type``
with an ``extra_args[\"output_type\"]`` fallback) so the OpenAI-compatible server,
which never sets it, keeps the PIL decode bit-for-bit unchanged.

These tests are CPU-only and drive the pipeline with duck-typed stubs (no model
weights, no GPU).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.models.sensenova_vision.pipeline_sensenova_vision import (
    SenseNovaVisionPipeline,
    build_sensenova_vision_diffusion_output,
)
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _pipeline() -> SenseNovaVisionPipeline:
    """Build a SenseNovaVisionPipeline instance without loading weights."""
    pipeline = object.__new__(SenseNovaVisionPipeline)
    pipeline.bagel = SimpleNamespace(
        latent_downsample=2,
        latent_patch_size=2,
        latent_channel=3,
    )
    pipeline.vae = SimpleNamespace(
        decode=lambda latent: latent,
        parameters=lambda: iter([SimpleNamespace(dtype=torch.float32)]),
    )
    pipeline._stage_durations = None
    pipeline.scheduler = None
    pipeline.scheduler_kwargs = None
    pipeline.new_token_ids = {}
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    return pipeline


def _request(
    *,
    output_type: str | None = None,
    extra_output_type: str | None = None,
    num_views: int = 3,
) -> DiffusionRequestBatch:
    extra_args: dict[str, object] = {"sensenova_vision_mode": "recon3d", "num_views": num_views}
    if extra_output_type is not None:
        extra_args["output_type"] = extra_output_type
    params = OmniDiffusionSamplingParams(
        num_inference_steps=1,
        output_type=output_type,
        extra_args=extra_args,
        past_key_values=SimpleNamespace(
            key_cache=[torch.zeros(8, 4)],
            value_cache=[torch.zeros(8, 4)],
        ),
        kv_metadata={"ropes": [8], "image_shape": [8, 8]},
    )
    req = OmniDiffusionRequest(prompt="generate", request_id="req-raw", sampling_params=params)
    return DiffusionRequestBatch(requests=[req])


def test_should_return_raw_tensor_reads_output_type() -> None:
    """The flag is honored from ``params.output_type`` and the extra_args fallback."""
    assert SenseNovaVisionPipeline._should_return_raw_tensor(_request(output_type="raw_tensor").sampling_params)
    assert SenseNovaVisionPipeline._should_return_raw_tensor(
        _request(output_type=None, extra_output_type="raw_tensor").sampling_params
    )
    assert not SenseNovaVisionPipeline._should_return_raw_tensor(_request(output_type=None).sampling_params)
    assert not SenseNovaVisionPipeline._should_return_raw_tensor(_request(output_type="pil").sampling_params)
    assert not SenseNovaVisionPipeline._should_return_raw_tensor(
        _request(output_type="latent").sampling_params  # Wan-style output types keep PIL
    )


def test_decode_latent_raw_returns_hxwx3_float32() -> None:
    """``_decode_latent_raw`` returns the raw VAE output as an HxWx3 float32 array.

    The stub VAE echoes its input, so this also verifies the einsum unpack into
    an HxWx3 tensor in VAE-output space ([0.25, 0.25, 0.25]-style values), not
    an 8-bit clamp.
    """
    pipeline = _pipeline()
    latent = torch.full((4, 3 * 2 * 2), 0.25, dtype=torch.float32)  # (h*w, c*p*p)
    arr = pipeline._decode_latent_raw(pipeline.bagel, pipeline.vae, latent, (4, 4))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4, 3)
    assert arr.dtype == np.float32
    # VAE output is raw (≈[-1,1]): values must NOT be clamped to 8-bit 0..255.
    assert arr.min() >= -1.0 - 1e-6 and arr.max() <= 1.0 + 1e-6


def test_decode_image_from_latent_dispatch_raw() -> None:
    """``_decode_image_from_latent`` returns a raw array when the flag is set."""
    pipeline = _pipeline()
    params = _request(output_type="raw_tensor").sampling_params
    latent = torch.full((4, 3 * 2 * 2), 0.25, dtype=torch.float32)
    out = pipeline._decode_image_from_latent(pipeline.bagel, pipeline.vae, latent, (4, 4), params)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32


def test_decode_image_from_latent_keeps_pil_by_default() -> None:
    """Without the flag, ``_decode_image_from_latent`` returns a PIL image (server contract)."""
    pipeline = _pipeline()
    params = _request(output_type=None).sampling_params
    latent = torch.full((4, 3 * 2 * 2), 0.25, dtype=torch.float32)

    # The default super() path renders an 8-bit PIL image from the raw latent.
    # Stub the VAE decode with the identity so the base math stays meaningful,
    # then assert the result is a PIL Image (the server contract).
    out = pipeline._decode_image_from_latent(pipeline.bagel, pipeline.vae, latent, (4, 4), params)
    assert isinstance(out, Image.Image)
    assert out.size == (4, 4)


def test_forward_recon3d_raw_tensor_list() -> None:
    """``_forward_recon3d`` with the flag set returns a list of raw arrays.

    The stub ``generate_image`` returns one latent per view; the per-view
    decode goes through ``_decode_image_from_latent`` (this stub mirrors the
    raw branch) so the payload carries HxWx3 float32 arrays.
    """
    pipeline = _pipeline()
    pipeline._stage_durations = None

    # Mirrors the production signature: _forward_recon3d passes the request's
    # sampling params as the 5th argument so the decode can dispatch on output_type.
    def raw_decode(bagel, vae, latent, image_shape, params=None):
        H, W = image_shape
        h, w = H // bagel.latent_downsample, W // bagel.latent_downsample
        p = bagel.latent_patch_size
        c = bagel.latent_channel
        reshaped = latent.reshape(1, h, w, p, p, c)
        reshaped = torch.einsum("nhwpqc->nchpwq", reshaped).reshape(1, c, h * p, w * p)
        return np.asarray(reshaped[0].permute(1, 2, 0).float().cpu().numpy())

    pipeline._decode_image_from_latent = raw_decode
    # Per-view packed_seqlens (h*w + 2 each); ``_forward_recon3d`` collapses the
    # sum for the denoise loop and passes the per-view values as unpack_seqlens.
    pipeline.bagel.prepare_vae_latent = lambda **kw: {
        "packed_seqlens": torch.tensor([16 + 2] * len(kw.get("image_sizes", [])), dtype=torch.int),
        "image_sizes": kw.get("image_sizes", []),
    }
    pipeline.bagel.generate_image = lambda **kw: (
        [torch.full((16, 3 * 2 * 2), 0.25, dtype=torch.float32) for _ in kw["unpack_seqlens"].tolist()],
        None,
        None,
        None,
    )

    out = pipeline._forward_recon3d(_request(output_type="raw_tensor", num_views=3))
    payload = out.output["payload"]
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 3
    assert all(isinstance(img, np.ndarray) for img in payload["image"])
    assert payload["image"][0].shape == (8, 8, 3)
    assert payload["image"][0].dtype == np.float32


def test_forward_recon3d_default_keeps_pil() -> None:
    """Without the flag, ``_forward_recon3d`` keeps the 8-bit PIL per-view decode."""
    pipeline = _pipeline()
    pipeline._stage_durations = None

    def pil_decode(bagel, vae, latent, image_shape, params=None):
        return Image.new("RGB", (4, 4))

    pipeline._decode_image_from_latent = pil_decode
    pipeline.bagel.prepare_vae_latent = lambda **kw: {
        "packed_seqlens": torch.tensor([16 + 2] * len(kw.get("image_sizes", [])), dtype=torch.int),
        "image_sizes": kw.get("image_sizes", []),
    }
    pipeline.bagel.generate_image = lambda **kw: (
        [torch.full((16, 3 * 2 * 2), 0.25, dtype=torch.float32) for _ in kw["unpack_seqlens"].tolist()],
        None,
        None,
        None,
    )

    out = pipeline._forward_recon3d(_request(output_type=None, num_views=2))
    payload = out.output["payload"]
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 2
    assert all(isinstance(img, Image.Image) for img in payload["image"])


def test_raw_payload_formats_to_image_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw-array payload passes through the formatter as an ``images`` entry.

    ``output_type=raw_tensor`` is a producer-side contract: the payload item is
    an HxWx3 float32 array, and the standard diffusion formatter still places it
    under ``result.images`` (the serving layer must not request raw_tensor).
    """
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    arr = np.random.default_rng(0).uniform(-1, 1, size=(8, 8, 3)).astype(np.float32)
    diffusion_output = build_sensenova_vision_diffusion_output(image=arr)

    postprocess_output = normalize_diffusion_postprocess_output(diffusion_output.output)
    assert postprocess_output.primary_key == "image"

    req = OmniDiffusionRequest(
        prompt="generate",
        request_id="req-raw",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, output_type="raw_tensor"),
    )
    config = SimpleNamespace(model_class_name="SenseNovaVisionPipeline")
    [result] = format_diffusion_outputs(
        request=req,
        od_config=config,
        diffusion_output=diffusion_output,
        output_data=diffusion_output.output,
        postprocess_output=postprocess_output,
    )
    assert result.images == [arr]
    assert result.final_output_type == "image"
