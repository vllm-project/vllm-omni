# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_pipeline_import_registry_and_component_discovery():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
    )

    assert _DIFFUSION_MODELS["MiniMaxH3Pipeline"] == (
        "minimax_h3",
        "pipeline_minimax_h3",
        "MiniMaxH3Pipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["MiniMaxH3Pipeline"] == "get_minimax_h3_post_process_func"
    assert MiniMaxH3Pipeline._dit_modules == ["transformer"]
    assert MiniMaxH3Pipeline._encoder_modules == ["text_encoder"]
    assert MiniMaxH3Pipeline._vae_modules == ["video_vae", "audio_vae"]


def test_joint_postprocess_is_multiprocessing_picklable():
    from vllm_omni.diffusion.models.minimax_h3 import (
        get_minimax_h3_post_process_func,
    )

    postprocess = get_minimax_h3_post_process_func(SimpleNamespace())
    postprocess = ForkingPickler.loads(ForkingPickler.dumps(postprocess))
    video = torch.linspace(0, 1, 2 * 3 * 2 * 4 * 5).reshape(2, 3, 2, 4, 5)
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

    result = postprocess((video, audio), output_type="np")

    assert isinstance(result["video"], list)
    assert result["video"][0].shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(result["audio"], audio.numpy())
    assert result["audio_sample_rate"] == 32000
    assert result["fps"] == 24


def test_cfg_parallel_is_rejected_for_distilled_checkpoint():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cfg_parallel_size=2),
    )
    with pytest.raises(ValueError, match="CFG-distilled"):
        MiniMaxH3Pipeline(od_config=od_config)


def test_shape_contract_matches_three_reference_tasks():
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
        minimax_h3_align_frame_count,
    )

    assert minimax_h3_align_frame_count(round(8.7 * 24)) == 209
    assert MINIMAX_H3_SHAPE_PLANNER.video_latent_t(209) == 62
    assert MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(8.7) == 348
    assert minimax_h3_align_frame_count(round(5.0 * 24)) == 124
    assert MINIMAX_H3_SHAPE_PLANNER.video_latent_t(124) == 37
    assert MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(5.0) == 200
    assert MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(62) == 209


def test_shifted_sigma_schedule_matches_reference_values():
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    sigmas = minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=12.0)

    assert sigmas == pytest.approx(
        [1.0, 0.9729729891, 0.9230769277, 0.8000000119, 0.0],
        abs=1e-7,
    )


def test_reference_image_resize_contract():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _reference_image_shape,
    )

    assert _reference_image_shape(Image.new("RGB", (1080, 1440))) == (
        2048,
        2720,
    )
    with pytest.raises(ValueError, match="aspect ratio"):
        _reference_image_shape(Image.new("RGB", (100, 501)))


def test_encoder_forward_uses_hook_compatible_encode_entrypoint():
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    expected = torch.ones(2, 3)
    encoder.encode_ids = Mock(return_value=expected)
    input_ids = torch.tensor([1, 2])
    pixel_values = torch.ones(1, 4)
    image_grid_thw = torch.tensor([[1, 1, 1]])

    actual = encoder(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    assert actual is expected
    encoder.encode_ids.assert_called_once_with(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )


def test_encoder_forward_forwards_video_inputs():
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    expected = torch.ones(2, 3)
    encoder.encode_ids = Mock(return_value=expected)
    input_ids = torch.tensor([1, 2])
    pixel_values_videos = torch.ones(1, 4)
    video_grid_thw = torch.tensor([[2, 1, 1]])

    actual = encoder(
        input_ids,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )

    assert actual is expected
    encoder.encode_ids.assert_called_once_with(
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )


def test_reference_video_shape_uses_h3_adapt_shape_policy():
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        _reference_video_shape,
    )

    assert _reference_video_shape(1280, 720) == (1344, 768)
    assert _reference_video_shape(3844, 2160) == (1344, 768)


def test_text_encoder_stub_constructs_without_group_or_weights():
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    encoder = MiniMaxH3Qwen3VLEncoder(
        "/nonexistent/text_encoder",
        device=torch.device("cpu"),
        load_model=False,
        encoder_group=None,
    )
    assert not encoder.is_loaded
    assert encoder.tp_size == 1
    # The stub has no parameters, so it never contributes to the runner's
    # strict missing-parameter check on non-encoder ranks.
    assert list(encoder.named_parameters()) == []


def test_no_offload_keeps_text_encoder_resident():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected
    input_ids = torch.tensor([1, 2])

    actual = pipeline._encode_text_hidden(input_ids, {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_not_called()


def test_model_offload_uses_hooked_text_encoder_call():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
    )
    expected = torch.ones(2, 3)
    pipeline.text_encoder = Mock(return_value=expected)
    input_ids = torch.tensor([1, 2])
    vision_kwargs = {"pixel_values": torch.ones(1, 4)}

    actual = pipeline._encode_text_hidden(input_ids, vision_kwargs)

    assert actual is expected
    pipeline.text_encoder.assert_called_once_with(input_ids, **vision_kwargs)
    pipeline.text_encoder.load_to_device.assert_not_called()


def test_layerwise_offload_releases_text_encoder():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=True,
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()


def test_video_vae_keeps_reference_fp32_weights(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1).half()

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )

    video_vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cpu"),
    )

    assert next(video_vae.parameters()).dtype == torch.float32


def test_video_vae_encode_uses_configured_parallel_tiling():
    from vllm_omni.diffusion.models.minimax_h3.vae import (
        MiniMaxH3VideoVAE,
    )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.parallel_tiling = True
            self.encode_calls = []

        def encode_videos(self, frames, *, use_fp16_latent):
            assert self.parallel_tiling
            self.encode_calls.append((frames, use_fp16_latent))
            return [torch.ones(1, 1, 2, 2, 2)]

    video_vae = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(video_vae)
    video_vae.model = FakeModel()
    video_vae.config_dict = {
        "latent_channels": 1,
        "latents_mean": [0.0],
        "latents_std": [1.0],
    }

    rows, shape = video_vae.encode_video("frames")

    assert video_vae.model.parallel_tiling
    assert video_vae.model.encode_calls == [("frames", True)]
    assert rows.shape == (2, 4)
    assert shape == (2, 2, 2)


def test_distributed_video_vae_encodes_references_sequentially(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import (
        pipeline_minimax_h3 as pipeline_module,
    )

    prepared = [
        {"prepared_path": "video-1.mp4"},
        {"prepared_path": "video-2.mp4"},
    ]

    class FakeVideoVAE:
        def __init__(self):
            self.calls = []

        def is_distributed_enabled(self):
            return True

        def encode_video(self, frames):
            self.calls.append(frames)
            index = len(self.calls)
            return torch.full((1, 2), index, dtype=torch.float32), (index, 2, 3)

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = FakeVideoVAE()

    monkeypatch.setattr(
        pipeline_module,
        "_dit_rank_world",
        lambda: ("dit-group", 1, 4),
    )

    def fake_broadcast_object_list(values, *, src, group, device):
        assert values == [None]
        assert (src, group, device) == (0, "dit-group", torch.device("cpu"))
        values[0] = prepared

    monkeypatch.setattr(
        pipeline_module.dist,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_video_frames",
        lambda path: f"frames:{path}",
    )

    rows, shapes = pipeline._encode_video_conditions(None, count=2)

    assert pipeline.video_vae.calls == [
        "frames:video-1.mp4",
        "frames:video-2.mp4",
    ]
    torch.testing.assert_close(
        rows,
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    assert shapes == [(1, 2, 3), (2, 2, 3)]
