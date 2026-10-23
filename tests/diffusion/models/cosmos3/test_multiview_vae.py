# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner placement, and the single-rank VAE path against the original serial code."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.cosmos3.multiview_vae import stream_media_to_uint8_cthw
from vllm_omni.diffusion.models.cosmos3.transfer import media_to_uint8_cthw
from vllm_omni.diffusion.models.cosmos3.vae_owner import VaeTask, _place
from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HEIGHT, WIDTH, NUM_FRAMES, LATENT_FRAMES = 32, 48, 9, 3


def _task(key: str, cost: float) -> VaeTask:
    return VaeTask(key, cost, (1,), torch.float32, run=lambda _: torch.zeros(1))


def test_place_longest_first_and_prefers_output_rank_on_ties() -> None:
    # 2 x 2 grid, grid rank = cfg_rank * 2 + sp_rank: hops to rank 0 are 0, 1, 1, 2.
    tasks = [_task("e", 1.0), _task("b", 3.0), _task("a", 5.0), _task("c", 3.0), _task("d", 2.0)]
    order, owners = _place(tasks, sp_size=2, cfg_size=2)
    assert [task.key for task in order] == ["a", "b", "c", "d", "e"]
    assert owners == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 3}

    order, owners = _place([_task(str(i), 1.0) for i in range(6)], sp_size=4, cfg_size=1)
    assert owners == {"0": 0, "1": 1, "2": 2, "3": 3, "4": 0, "5": 1}


def _write_video(iio, path, seed: int, frames: int = 7) -> str:
    generator = np.random.default_rng(seed)
    video = [generator.integers(0, 256, (80, 128, 3), dtype=np.uint8) for _ in range(frames)]
    iio.imwrite(path, video, fps=10, codec="libx264")
    return str(path)


@pytest.mark.parametrize("max_frames", [None, 1, 5])
def test_stream_media_matches_media_to_uint8_cthw(tmp_path, max_frames) -> None:
    iio = pytest.importorskip("imageio.v3")
    path = _write_video(iio, tmp_path / "camera.mp4", seed=0, frames=9)

    expected = media_to_uint8_cthw(path, height=HEIGHT, width=WIDTH, max_frames=max_frames)
    actual = stream_media_to_uint8_cthw(path, height=HEIGHT, width=WIDTH, max_frames=max_frames)
    assert torch.equal(actual, expected)


class _LidarEncoder:
    config = {
        "fps": 10.0,
        "network_config": {"resolution": [128, 1800]},
        "spatial_compression": [16, 18],
        "latent_channels": 2,
        "streaming_chunk_frames": 2,
    }

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        latents = F.avg_pool3d(frames[:2].unsqueeze(0), (1, 16, 18))
        return latents.contiguous(memory_format=torch.channels_last_3d)


class _LidarDecoder:
    config = {
        "network_config": {"resolution": [128, 1800]},
        "range_projection": {"native_height": 4, "semantic_width": 5},
    }

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return F.interpolate(latents[:, :1].float(), size=(latents.shape[2], 4, 5)).expand(-1, 3, -1, -1, -1) * 3


def _single_rank_pipeline():
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.bfloat16
    pipeline.transformer = SimpleNamespace(latent_channel_size=3)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae = SimpleNamespace(dtype=torch.bfloat16)
    pipeline._encode_video_tensor = lambda pixels: F.avg_pool3d(pixels[:, :, ::4].float(), (1, 8, 8))
    pipeline._decode_latents = lambda latents: (F.interpolate(latents.float(), size=(9, 32, 48)) * 2).bfloat16()
    pipeline.lidar_encoder = _LidarEncoder()
    pipeline.lidar_decoder = _LidarDecoder()
    return pipeline


def _original_encode(pipeline, views, field: str) -> torch.Tensor:
    """PR 6917's serial path: all views' pixels at once, then one encode per view."""
    pixels = pipeline._prepare_camera_major_pixels(
        views, field=field, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES, keep_first=False
    )
    return torch.cat(
        [
            pipeline._encode_video_tensor(pixels[:, :, view * NUM_FRAMES : (view + 1) * NUM_FRAMES])
            for view in range(len(views))
        ],
        dim=2,
    )


def test_single_rank_encode_matches_original(tmp_path) -> None:
    iio = pytest.importorskip("imageio.v3")
    from safetensors.torch import save_file

    pipeline = _single_rank_pipeline()
    views = [
        {
            "camera_key": f"camera_{index}",
            "vision_path": _write_video(iio, tmp_path / f"vision_{index}.mp4", seed=index),
            "control_path": _write_video(iio, tmp_path / f"control_{index}.mp4", seed=10 + index),
        }
        for index in range(3)
    ]
    lidar_path = tmp_path / "lidar.safetensors"
    save_file({"frames": torch.rand(3, 4, 128, 1800)}, str(lidar_path))
    lidar_request = {"control_path": str(lidar_path), "condition_path": str(lidar_path)}

    vision, control, lidar_control, lidar_condition = pipeline._encode_inputs(
        views=views,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        frame_rate=30.0,
        vision_views=[0, 2],
        keep_first=False,
        require_complete=False,
        encode_control=True,
        lidar_request=lidar_request,
    )

    assert torch.equal(control, _original_encode(pipeline, views, "control"))
    expected_vision = _original_encode(pipeline, views, "vision")
    for view in (0, 2):
        span = slice(view * LATENT_FRAMES, (view + 1) * LATENT_FRAMES)
        assert torch.equal(vision[:, :, span], expected_vision[:, :, span])
    # The view without conditioned frames is not encoded.
    assert not vision[:, :, LATENT_FRAMES : 2 * LATENT_FRAMES].any()

    sweeps = 3  # 9 frames at 30 FPS, LiDAR at 10 FPS
    expected_lidar = pipeline.lidar_encoder(load_lidar_frames(lidar_path, num_sweeps=sweeps)).to(torch.bfloat16)
    assert torch.equal(lidar_control, expected_lidar)
    assert lidar_control.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(lidar_condition, pipeline._encode_lidar_condition(lidar_request, sweeps, lidar_control.shape))


def test_single_rank_encode_rejects_lidar_condition_without_generated_sweeps() -> None:
    pipeline = _single_rank_pipeline()
    lidar_request = {"control_path": "unused.safetensors", "condition_path": "unused.safetensors"}
    with pytest.raises(ValueError, match="must leave at least one"):
        pipeline._encode_inputs(
            views=[],
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=30.0,
            vision_views=[],
            keep_first=False,
            require_complete=False,
            encode_control=False,
            lidar_request={**lidar_request, "num_conditional_sweeps": 3},
        )


def test_single_rank_decode_matches_original() -> None:
    pipeline = _single_rank_pipeline()
    generator = torch.Generator().manual_seed(0)
    camera = torch.randn(1, 3, 2 * LATENT_FRAMES, 4, 6, generator=generator)
    lidar_latents = torch.randn(1, 2, 3, 8, 100, generator=generator)

    video, lidar = pipeline._decode_outputs(
        [camera, lidar_latents], num_views=2, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, return_lidar=True
    )

    # PR 6917's serial path: one decode per view, copied to the host, then clamped.
    expected = torch.cat(
        [pipeline._decode_latents(camera.narrow(2, view * LATENT_FRAMES, LATENT_FRAMES)) for view in range(2)], dim=2
    ).clamp_(-1, 1)
    assert video.device.type == "cpu" and video.dtype == torch.bfloat16
    assert torch.equal(video, expected)
    assert lidar.device.type == "cpu" and lidar.dtype == torch.float32
    assert torch.equal(lidar, pipeline.lidar_decoder(lidar_latents))

    video, lidar = pipeline._decode_outputs(
        [camera], num_views=2, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, return_lidar=False
    )
    assert torch.equal(video, expected) and lidar is None
