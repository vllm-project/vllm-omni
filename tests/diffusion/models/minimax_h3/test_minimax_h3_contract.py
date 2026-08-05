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


def test_fl2va_supports_first_last_and_explicit_frame_index_contracts():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_fl2va_keyframe_indices,
    )

    assert _resolve_fl2va_keyframe_indices({}, 1) == [0]
    assert _resolve_fl2va_keyframe_indices({}, 2) == [0, -1]
    assert _resolve_fl2va_keyframe_indices({"frame_index": -1}, 1) == [-1]
    assert _resolve_fl2va_keyframe_indices({"frame_indices": [0, -1]}, 2) == [0, -1]
    with pytest.raises(ValueError, match="frame_indices"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [0, 1]}, 2)


def test_minimax_h3_uses_the_official_output_canvas_policy():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_output_canvas,
    )

    assert _resolve_output_canvas(21 / 9, 768) == (672, 1536)
    assert _resolve_output_canvas(16 / 9, 768) == (768, 1344)
    assert _resolve_output_canvas(9 / 16, 768) == (1344, 768)
    with pytest.raises(ValueError, match="short_edge"):
        _resolve_output_canvas(16 / 9, 720)


def test_minimax_h3_accepts_sglang_auto_aspect_ratio_alias():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": 5.0, "aspect_ratio": "auto"},
    )
    height, width, *_ = pipeline._resolve_shape("ref2va", sampling, None)
    assert (height, width) == (768, 1344)


def test_minimax_h3_advertises_the_official_ref2va_image_limit():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    assert get_diffusion_model_metadata("MiniMaxH3Pipeline").max_multimodal_image_inputs == 9


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


@pytest.mark.parametrize(
    ("case", "extra", "image_count", "expected"),
    [
        ("F1", {"frame_index": -1}, 1, [-1]),
        ("F2", {"frame_indices": [0, -1]}, 2, [0, -1]),
    ],
)
def test_f1_f2_official_fl2va_keyframe_matrix(case, extra, image_count, expected):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_fl2va_keyframe_indices,
    )

    assert case in {"F1", "F2"}
    assert _resolve_fl2va_keyframe_indices(extra, image_count) == expected


@pytest.mark.parametrize(
    ("case", "counts"),
    [
        ("R1", (3, 0, 0)),
        ("R2", (1, 0, 0)),
        ("R3", (1, 1, 0)),
        ("R4", (0, 1, 1)),
        ("R5", (1, 1, 1)),
        ("R6", (1, 0, 2)),
    ],
)
def test_r1_r6_ref2va_reference_count_matrix(case, counts):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_ref2va_reference_counts,
    )

    assert case.startswith("R")
    _validate_ref2va_reference_counts(*counts)


def test_ref2va_reference_count_validation_preserves_client_error_metadata():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_ref2va_reference_counts,
    )
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match="at least one image or video"):
        _validate_ref2va_reference_counts(0, 0, 0)


@pytest.mark.parametrize(
    ("case", "start_time", "expected_duration"),
    [
        ("R7", None, 10.0),
        ("R8", 4.0, 6.0),
    ],
)
def test_r7_r8_ref2va_video_segment_matrix(monkeypatch, tmp_path, case, start_time, expected_duration):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    source = tmp_path / f"{case}.mp4"
    source.touch()
    metadata = {
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "frame_count": 240,
        "duration": 10.0,
        "format_names": ("mp4",),
        "video_codec": "h264",
        "audio_codecs": (),
        "file_size": 1024,
    }
    transcode_calls = []
    monkeypatch.setattr(reference_video_module, "_probe_video", lambda _path: metadata)
    monkeypatch.setattr(
        reference_video_module,
        "_transcode_reference_video",
        lambda source, **kwargs: transcode_calls.append((source, kwargs)) or "prepared.mp4",
    )

    prepared = reference_video_module.prepare_reference_videos(
        [str(source)],
        target_frame_count=209,
        workdir=str(tmp_path / "work"),
        start_time_seconds=start_time,
    )

    assert prepared[0]["duration_seconds"] == pytest.approx(expected_duration)
    assert prepared[0]["start_time_seconds"] == pytest.approx(start_time or 0.0)
    assert transcode_calls[0][1]["duration_seconds"] == pytest.approx(expected_duration)


def test_ref2va_two_video_recipe_tolerates_container_rounding(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.touch()
    second.touch()
    metadata = {
        str(first): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 241,
            "duration": 10.041667,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
        str(second): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 120,
            "duration": 4.966667,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
    }
    transcode_calls = []
    monkeypatch.setattr(reference_video_module, "_probe_video", lambda path: metadata[str(path)])
    monkeypatch.setattr(
        reference_video_module,
        "_transcode_reference_video",
        lambda source, **kwargs: transcode_calls.append((source, kwargs)) or f"{source}.prepared.mp4",
    )

    prepared = reference_video_module.prepare_reference_videos(
        [str(first), str(second)],
        target_frame_count=124,
        workdir=str(tmp_path / "work"),
    )

    assert [item["duration_seconds"] for item in prepared] == pytest.approx([10.041667, 4.958333])
    assert sum(item["duration_seconds"] for item in prepared) == pytest.approx(15.0)
    assert transcode_calls[1][1]["duration_seconds"] == pytest.approx(4.958333)


def test_ref2va_two_video_recipe_rejects_real_duration_overflow(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module
    from vllm_omni.errors import OmniClientError

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.touch()
    second.touch()
    metadata = {
        str(first): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 240,
            "duration": 10.0,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
        str(second): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 120,
            "duration": 5.02,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
    }
    monkeypatch.setattr(reference_video_module, "_probe_video", lambda path: metadata[str(path)])
    monkeypatch.setattr(reference_video_module, "_transcode_reference_video", lambda source, **kwargs: "prepared.mp4")

    with pytest.raises(OmniClientError, match="15 seconds"):
        reference_video_module.prepare_reference_videos(
            [str(first), str(second)],
            target_frame_count=124,
            workdir=str(tmp_path / "work"),
        )


@pytest.mark.parametrize("duration", [4.0, 15.0])
def test_g2_output_duration_accepts_official_boundaries(duration):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": duration},
    )

    height, width, *_ = pipeline._resolve_shape("ref2va", sampling, None)
    assert height > 0 and width > 0


@pytest.mark.parametrize("duration", [3.99, 15.01, float("nan"), "not-a-duration"])
def test_g2_output_duration_rejects_out_of_contract_values(duration):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": duration},
    )
    with pytest.raises(ValueError, match="duration"):
        pipeline._resolve_shape("ref2va", sampling, None)


def test_g1_fanout_uses_incrementing_output_seeds():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _minimax_h3_output_seeds,
        _resolve_minimax_h3_num_outputs,
    )

    assert _resolve_minimax_h3_num_outputs(3) == 3
    assert _minimax_h3_output_seeds(100, 3) == [100, 101, 102]
    with pytest.raises(ValueError, match="num_outputs_per_prompt"):
        _resolve_minimax_h3_num_outputs(11)


def test_g3_task_specific_aspect_ratio_policy():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_aspect_ratio,
    )

    image = Image.new("RGB", (1280, 720))
    assert _resolve_minimax_h3_aspect_ratio("fl2va", "9:16", image) == pytest.approx(16 / 9)
    assert _resolve_minimax_h3_aspect_ratio("ref2va", None, None) == pytest.approx(16 / 9)
    assert _resolve_minimax_h3_aspect_ratio("ref2va", "auto", None) == pytest.approx(16 / 9)
    with pytest.raises(ValueError, match="requires an explicit"):
        _resolve_minimax_h3_aspect_ratio("t2va", None, None)
    with pytest.raises(ValueError, match="one of"):
        _resolve_minimax_h3_aspect_ratio("t2va", "2:1", None)


def test_g3_t2va_shape_requires_a_named_ratio_and_fl2va_ignores_override():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    t2va_sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": 5.0},
    )
    with pytest.raises(ValueError, match="requires an explicit aspect_ratio"):
        pipeline._resolve_shape("t2va", t2va_sampling, None)

    fl2va_sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": 5.0, "aspect_ratio": "9:16"},
    )
    height, width, *_ = pipeline._resolve_shape("fl2va", fl2va_sampling, Image.new("RGB", (1280, 720)))
    assert (height, width) == (768, 1344)


def test_g4_reference_image_boundaries_and_aspect_ratio():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_reference_image,
    )

    _validate_reference_image(Image.new("RGB", (256, 256)))
    _validate_reference_image(Image.new("RGB", (5760, 2304)))
    with pytest.raises(ValueError, match="dimensions"):
        _validate_reference_image(Image.new("RGB", (255, 255)))
    with pytest.raises(ValueError, match="aspect ratio"):
        _validate_reference_image(Image.new("RGB", (256, 641)))


def test_g4_reference_image_file_format_and_size_contract(tmp_path):
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _load_image,
    )

    valid_path = tmp_path / "reference.png"
    Image.new("RGB", (256, 256)).save(valid_path)
    assert _load_image(valid_path).size == (256, 256)

    invalid_path = tmp_path / "reference.bmp"
    Image.new("RGB", (256, 256)).save(invalid_path)
    with pytest.raises(ValueError, match="must use"):
        _load_image(invalid_path)


def test_g4_standalone_audio_duration_and_total_duration_contract():
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        validate_reference_audio_waveforms,
    )

    sample_rate = 16000
    validate_reference_audio_waveforms(
        [
            (torch.zeros(1, 2 * sample_rate), sample_rate),
            (torch.zeros(1, 13 * sample_rate), sample_rate),
        ]
    )
    with pytest.raises(ValueError, match="duration"):
        validate_reference_audio_waveforms([(torch.zeros(1, int(1.9 * sample_rate)), sample_rate)])
    with pytest.raises(ValueError, match="at most 15 seconds in total"):
        validate_reference_audio_waveforms(
            [
                (torch.zeros(1, 8 * sample_rate), sample_rate),
                (torch.zeros(1, 8 * sample_rate), sample_rate),
            ]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 23.0, "FPS"),
        ("duration", 1.0, "duration"),
        ("format_names", ("avi",), "container"),
        ("video_codec", "vp9", "H.264"),
        ("audio_codecs", ("opus",), "AAC"),
        ("file_size", 50 * 1024 * 1024 + 1, "size"),
    ],
)
def test_g4_reference_video_metadata_validation(field, value, message, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        _validate_reference_video_metadata,
    )

    metadata = {
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "duration": 10.0,
        "format_names": ("mp4",),
        "video_codec": "h264",
        "audio_codecs": ("aac",),
        "file_size": 1024,
    }
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_reference_video_metadata(metadata, index=0, source=str(tmp_path / "reference.mp4"))
