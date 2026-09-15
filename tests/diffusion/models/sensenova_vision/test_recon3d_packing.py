# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SenseNova-Vision recon3d multi-view packing and per-task transforms.

``SenseNovaVisionPipeline._forward_recon3d`` decodes ``num_views`` square views
from a single injected AR KV context.  The packing arithmetic is ported verbatim
from upstream ``inference/inferencer.py::gen_image``:

    curr_kvlens = [kv_len] + [0] * (num_views - 1)
    curr_rope   = [rope0 + x for x in range(num_views)]
    image_sizes = [image_shape] * num_views

and the per-task transform table distinguishes the VAE and ViT target sides
(e.g. recon3d -> VAE 512 / ViT 448; camera-pose -> ViT 560).  Everything here is
pure Python + PIL (no torch, no GPU, no checkpoint download).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.sensenova_vision.pipeline_sensenova_vision import (
    SenseNovaVisionPipeline,
)
from vllm_omni.diffusion.models.sensenova_vision.transforms_sensenova_vision import (
    PER_TASK_VAE_SIDE,
    PER_TASK_VIT_SIDE,
    ResizeSpec,
    max_long_edge_resize,
    packed_seqlens,
    recon3d_packing,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_recon3d_packing_n2() -> None:
    """Two views: first continues from the AR KV, second starts empty."""
    num_views = 2
    kv_len = 137
    base_rope = 137
    kv_lens, ropes = recon3d_packing(num_views, kv_len, base_rope)
    assert kv_lens == [137, 0]
    assert ropes == [137, 138]


def test_recon3d_packing_n4() -> None:
    """Four views share the same KV prefix, ropes increment per view."""
    kv_len = 300
    kv_lens, ropes = recon3d_packing(4, kv_len, 300)
    assert kv_lens == [300, 0, 0, 0]
    assert ropes == [300, 301, 302, 303]


def test_packed_seqlens_n2() -> None:
    """Per-branch packed_seqlens = (h*w + 2) markers; latent 32x32 -> 1026."""
    seqlens = packed_seqlens(2, 32, 32)
    assert seqlens == [1026, 1026]


def test_per_task_vae_side_contract() -> None:
    """recon3d selects VAE 512; camera-pose has no VAE prefill."""
    assert PER_TASK_VAE_SIDE["recon3d"] == 512
    assert PER_TASK_VAE_SIDE["camera_pose"] is None


def test_per_task_vit_side_contract() -> None:
    """recon3d ViT 448 / camera-pose ViT 560."""
    assert PER_TASK_VIT_SIDE["recon3d"] == 448
    assert PER_TASK_VIT_SIDE["camera_pose"] == 560


def test_resize_spec_target_side() -> None:
    """Stride-aligned square target: largest stride multiple <= max_size."""
    # ImageTransform(512, 256, 16) -> 512; (448, 224, 14) -> 448; (560, 378, 14) -> 560.
    assert ResizeSpec(512, 256, 16).target_side == 512
    assert ResizeSpec(448, 224, 14).target_side == 448
    assert ResizeSpec(560, 378, 14).target_side == 560


def test_resize_spec_vae_grid() -> None:
    """Latent grid for the recon3d VAE side (downsample 8, patch 2 -> 16)."""
    grid = ResizeSpec(512, 256, 16).vae_grid(latent_downsample=16)
    assert grid == (32, 32)


def test_max_long_edge_resize_downscales_to_target() -> None:
    """A square input above the max downscales to the stride-aligned target."""
    img = Image.new("RGB", (700, 700))
    fn = max_long_edge_resize(512, 256, 16)
    out = fn(img)
    assert out.size == (512, 512)


def test_resize_does_not_upscale_below_target() -> None:
    """Inputs already within max_size are left at their native size (no upscale)."""
    img = Image.new("RGB", (256, 256))
    out = max_long_edge_resize(512, 256, 16)(img)
    assert out.size == (256, 256)


def test_max_long_edge_resize_clamps_stride() -> None:
    """Output side is a multiple of stride and never below stride."""
    img = Image.new("RGB", (1024, 1024))
    out = max_long_edge_resize(560, 378, 14)(img)
    assert out.size[0] % 14 == 0
    assert out.size[0] <= 560


def _cache(seq_len: int) -> SimpleNamespace:
    """A minimal duck-typed naive KV cache with ``key_cache[0]`` shaped rows.

    ``NaiveCache.from_object`` iterates the cache to rebuild layer-indexed
    tensors, so the mock exposes per-layer tensors as an indexable iterable.
    """
    return SimpleNamespace(
        key_cache=[torch.zeros(seq_len, 4)],
        value_cache=[torch.zeros(seq_len, 4)],
    )


def _recon3d_request(*, num_views: int = 2, mode: str = "recon3d") -> DiffusionRequestBatch:
    params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        extra_args={"sensenova_vision_mode": mode, "num_views": num_views},
        past_key_values=_cache(16),
        kv_metadata={"ropes": [16], "image_shape": [16, 16]},
    )
    req = OmniDiffusionRequest(prompt="recon3d", request_id="req-recon3d", sampling_params=params)
    return DiffusionRequestBatch(requests=[req])


def _recon3d_pipeline() -> SenseNovaVisionPipeline:
    """Build a SenseNovaVisionPipeline instance without loading weights."""
    pipeline = object.__new__(SenseNovaVisionPipeline)
    pipeline.bagel = SimpleNamespace(
        latent_downsample=8,
        max_latent_size=64,
        prepare_vae_latent=lambda **kw: {
            "packed_seqlens": [0],
            "packed_init_noises": torch.zeros(1, 1),
            "image_sizes": kw.get("image_sizes", []),
        },
        generate_image=lambda **kw: (
            [torch.zeros(1, 1)] * len(kw.get("image_sizes", [])),
            None,
            None,
            None,
        ),
    )
    pipeline.new_token_ids = {}
    pipeline.device = torch.device("cpu")
    pipeline.scheduler = None
    pipeline.scheduler_kwargs = None
    pipeline.od_config = SimpleNamespace(dtype=torch.bfloat16)
    pipeline.vae = SimpleNamespace()
    pipeline._stage_durations = None
    pipeline._decode_image_from_latent = lambda *a: Image.new("RGB", (4, 4))
    return pipeline


def test_is_recon3d_selects_mode() -> None:
    """Only the recon3d mode routes to the multi-view decode."""
    pipeline = _recon3d_pipeline()
    assert pipeline._is_recon3d(_recon3d_request(mode="recon3d")) is True
    assert pipeline._is_recon3d(_recon3d_request(mode="generate")) is False


def test_forward_recon3d_decodes_num_views_images() -> None:
    """``_forward_recon3d`` decodes one PIL image per view and packs them as a list."""
    pipeline = _recon3d_pipeline()
    out = pipeline._forward_recon3d(_recon3d_request(num_views=3))
    payload = out.output["payload"]
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 3
    assert all(isinstance(img, Image.Image) for img in payload["image"])
