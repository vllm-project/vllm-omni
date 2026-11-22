# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.offline_inference.cosmos3_nano_sim_auto.cosmos3_nano_sim_auto import (
    _extract_payload,
    _save_camera_videos,
    _save_latents,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("joint", [False, True])
def test_offline_saves_reference_latent_format(tmp_path, joint: bool) -> None:
    latents = {"vision_latent": torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 2, 2, 2)}
    if joint:
        latents["lidar_latent"] = latents["vision_latent"] + 2
    output = [SimpleNamespace(multimodal_output={"latents": latents})]
    path = _save_latents(output, tmp_path, joint=joint)
    saved = torch.load(path, weights_only=True)
    assert len(saved) == 1 and saved[0].keys() == latents.keys()
    for name, expected in latents.items():
        torch.testing.assert_close(saved[0][name], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requested latents"):
        _save_latents({"payload": {"latents": {}}}, tmp_path, joint=joint)


def test_offline_extracts_video_and_metadata_from_request_output() -> None:
    video = torch.zeros(1, 3, 2, 16, 16)
    metadata = {"multiview": {"cameras": ["front", "rear"]}}
    output = SimpleNamespace(images=[video], multimodal_output={"metadata": metadata})
    actual_video, actual_metadata = _extract_payload([output])
    assert actual_video is video and actual_metadata == metadata


@pytest.mark.parametrize("pixel_range", ["uint8", "unit", "signed"])
def test_camera_videos_encode_in_request_order(tmp_path, pixel_range: str) -> None:
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    front = np.full((16, 16, 3), [255, 0, 0], dtype=np.uint8)
    rear = np.full((16, 16, 3), [0, 255, 0], dtype=np.uint8)
    frames = [front, front, rear, rear]
    if pixel_range != "uint8":
        frames = [frame.astype(np.float32) / 255 for frame in frames]
        if pixel_range == "signed":
            frames = [frame * 2 - 1 for frame in frames]
    originals = [frame.copy() for frame in frames]
    paths = _save_camera_videos(frames, ["front", "rear"], 2, tmp_path, 5.0)
    assert list(paths) == ["front", "rear"]
    for camera, expected in (("front", front), ("rear", rear)):
        with imageio.get_reader(paths[camera]) as reader:
            assert reader.get_meta_data()["fps"] == pytest.approx(5.0)
            decoded = list(reader.iter_data())
        assert len(decoded) == 2
        for frame in decoded:
            np.testing.assert_allclose(np.asarray(frame), expected, rtol=0, atol=4)
    for frame, original in zip(frames, originals, strict=True):
        np.testing.assert_array_equal(frame, original)
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_failed_encode_preserves_existing_camera_videos(tmp_path, monkeypatch) -> None:
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    paths = [tmp_path / f"vision_view{index:02d}_{camera}.mp4" for index, camera in enumerate(("front", "rear"))]
    for path in paths:
        path.write_bytes(b"previous video")
    get_writer = imageio.get_writer

    def fail_rear(path, **kwargs):
        if "rear" in path:
            raise RuntimeError("encoder failed")
        return get_writer(path, **kwargs)

    monkeypatch.setattr(imageio, "get_writer", fail_rear)
    with pytest.raises(RuntimeError, match="encoder failed"):
        _save_camera_videos([np.zeros((16, 16, 3), dtype=np.uint8)] * 2, ["front", "rear"], 1, tmp_path, 5.0)
    assert all(path.read_bytes() == b"previous video" for path in paths)
    assert set(tmp_path.iterdir()) == set(paths)
