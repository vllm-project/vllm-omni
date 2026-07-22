# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared stubs and dummy-input helpers for Lingbot World Fast L1 tests.

The real pipeline pulls in T5-XXL, the Wan VAE and a 5B-parameter transformer
on construction. Tests exercise only the state container, msgpack protocol and
scheduler, so these stubs replace the heavy dependencies with the smallest
implementations that match the call sites in
``vllm_omni/diffusion/models/lingbot_world_fast/pipeline_lingbot_world_fast.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch import nn

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast import (
        LingbotWorldFastPipeline,
    )


class StubT5Encoder:
    """Minimal stand-in for ``T5EncoderModel``.

    The pipeline calls ``self.text_encoder([prompt], device)`` and expects a
    list of token-embedding tensors, one per prompt.
    """

    def __init__(self, text_len: int = 512, dim: int = 32, dtype: torch.dtype = torch.float32) -> None:
        self.text_len = text_len
        self.dim = dim
        self.dtype = dtype

    def __call__(self, prompts: list[str], device: torch.device) -> list[torch.Tensor]:
        return [torch.zeros(self.text_len, self.dim, dtype=self.dtype, device=device) for _ in prompts]


class StubVAE:
    """Stand-in for ``Wan2_1_VAE``.

    ``encode([pixels])`` returns a list with one latent tensor shaped
    ``[16, F_lat, lat_h, lat_w]`` where ``F_lat = (F + 3) // 4`` so the
    pipeline's masking / slicing math is exercised normally.
    ``decode([latents])`` returns the latents unchanged (caller indexes [0]).
    """

    vae_stride = (4, 8, 8)

    def encode(self, video_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for v in video_list:
            # v: [C, F, H, W]
            _, f, h, w = v.shape
            lat_f = (f + self.vae_stride[0] - 1) // self.vae_stride[0]
            lat_h = h // self.vae_stride[1]
            lat_w = w // self.vae_stride[2]
            out.append(torch.zeros(16, lat_f, lat_h, lat_w, dtype=v.dtype, device=v.device))
        return out

    def decode(self, latents_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for latents in latents_list:
            # latents: [16, F_lat, lat_h, lat_w]; produce pixels at the inverse stride.
            _, f_lat, lat_h, lat_w = latents.shape
            f = f_lat * self.vae_stride[0]
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
            out.append(torch.zeros(3, f, h, w, dtype=latents.dtype, device=latents.device))
        return out


class StubWanModelFast(nn.Module):
    """Stand-in for ``WanModelFast``.

    Returns zeros shaped like the input latent, and bumps the local/global
    index tensors so chunk-boundary arithmetic is exercised by the pipeline.
    """

    def __init__(self, *, dim: int = 16, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=-1,
        )

    def forward(self, *, x, t, **kwargs):  # noqa: ARG002 — matches pipeline call site
        del t, kwargs
        return [torch.zeros_like(x[0])]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()


def make_dummy_camera_inputs(num_frames: int) -> dict[str, np.ndarray]:
    """Camera payload matching the shape the pipeline expects."""
    intrinsics = np.eye(4, dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    return {"intrinsics": intrinsics, "poses": poses}


def make_dummy_image(width: int = 64, height: int = 64) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def make_stubbed_pipeline(
    *,
    device: torch.device | None = None,
    dim: int = 16,
    num_heads: int = 4,
    num_layers: int = 2,
    target_dtype: torch.dtype = torch.float32,
) -> LingbotWorldFastPipeline:
    """Build a ``LingbotWorldFastPipeline`` backed by the conftest stubs.

    Skips the real ``__init__`` (which loads umt5-xxl, Wan VAE and a 5B
    transformer) via ``object.__new__`` and assigns the stubs directly,
    mirroring ``_make_i2v_pipeline`` in ``tests/diffusion/models/wan2_2``.
    The returned pipeline is suitable for driving ``.forward(req)`` end-to-end
    against ``LingbotWorldFastState`` without touching real weights.
    """
    from vllm_omni.diffusion.models.lingbot_world_fast.fm_solvers_unipc import (
        FlowUniPCMultistepScheduler,
    )
    from vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast import (
        CONFIG,
        LingbotWorldFastPipeline,
    )
    from vllm_omni.diffusion.models.lingbot_world_fast.state_lingbot_world_fast import (
        LingbotWorldFastState,
    )

    if device is None:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu", 0)

    parallel_config = SimpleNamespace(world_size=1)
    patch_size = [1, 2, 2]
    vae_stride = [4, 8, 8]

    od_config = SimpleNamespace(
        model="stub/Lingbot-World-Fast",
        parallel_config=parallel_config,
        dtype=target_dtype,
        model_config={
            "latent_frames_per_chunk": 3,
            "max_area": 64 * 64,
        },
    )

    pipeline = object.__new__(LingbotWorldFastPipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = od_config
    pipeline.parallel_config = parallel_config
    pipeline.device = device
    pipeline.target_dtype = target_dtype
    pipeline.control_type = "cam"
    pipeline.num_train_timesteps = CONFIG["num_train_timesteps"]
    pipeline.sp_size = parallel_config.world_size
    pipeline.state = LingbotWorldFastState()
    pipeline.text_encoder = StubT5Encoder(dim=dim, dtype=target_dtype)
    pipeline.vae = StubVAE()
    pipeline.vae_stride = vae_stride
    pipeline.patch_size = patch_size
    pipeline.model = StubWanModelFast(dim=dim, num_heads=num_heads, num_layers=num_layers).to(device)
    pipeline.scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=CONFIG["num_train_timesteps"],
        shift=1,
        use_dynamic_shifting=False,
    )
    return pipeline
