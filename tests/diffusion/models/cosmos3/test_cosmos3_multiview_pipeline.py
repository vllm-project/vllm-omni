# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

MULTIVIEW_BUCKETS = [
    ("480", "1,1", 640, 640, (20, 20)),
    ("480", "4,3", 736, 544, (17, 23)),
    ("480", "3,4", 544, 736, (23, 17)),
    ("480", "16,9", 832, 480, (15, 26)),
    ("480", "9,16", 480, 832, (26, 15)),
    ("720", "1,1", 960, 960, (30, 30)),
    ("720", "4,3", 1104, 832, (26, 35)),
    ("720", "3,4", 832, 1104, (35, 26)),
    ("720", "16,9", 1280, 720, (23, 40)),
    ("720", "9,16", 720, 1280, (40, 23)),
]


def _views(cameras: tuple[str, ...], *, vision: bool = False) -> list[dict]:
    result = []
    for index, camera in enumerate(cameras):
        view = {"camera_key": camera, "control_path": f"control_{index}.mp4"}
        if vision:
            view["vision_path"] = f"vision_{index}.mp4"
        result.append(view)
    return result


def _deployment_config() -> dict:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import COSMOS3_MADS_CAMERAS

    return {
        "causal_training_strategy": "none",
        "attention_scope": "decomposed",
        "decomposed_temporal_window_seconds": None,
        "control_attends_sensor": True,
        "align_temporal_positions_across_views": True,
        "share_vision_temporal_positions": True,
        "backend": "triton",
        "cameras": list(COSMOS3_MADS_CAMERAS),
        "max_views": len(COSMOS3_MADS_CAMERAS),
    }


def _deployment_model_config(multiview: dict) -> dict:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    )

    return {
        "backbone_type": COSMOS3_MULTIVIEW_BACKBONE_TYPE,
        "multiview": multiview,
    }


def _versioned_deployment_config() -> dict:
    return {
        **_deployment_config(),
        "schema_version": 2,
        "separate_view_text_tokenization": True,
        "variable_view_count": True,
        "inference_defaults": {
            "resolution": "480",
            "fps": 30,
            "num_steps": 35,
            "guidance": 6,
            "shift": 10,
            "control_guidance": 1,
            "emphasize_control_in_prompt": True,
            "guidance_interval": None,
            "control_guidance_interval": None,
            "sigma_max": 80,
            "normalize_cfg": False,
            "negative_metadata_mode": "none",
        },
    }


@pytest.mark.parametrize("scale", [2.0, {"mode": "other"}])
@pytest.mark.parametrize("normalize", [False, True])
def test_multiview_cfg_delegates_non_transfer_scales(monkeypatch, scale, normalize):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    predictions = [torch.zeros(1, 2), torch.ones(1, 2)]
    expected = torch.full((1, 2), 5.0)

    def combine(self, actual_predictions, actual_scale, cfg_normalize=False):
        assert actual_predictions is predictions and actual_scale is scale
        assert cfg_normalize is normalize
        return expected

    monkeypatch.setattr(Cosmos3OmniDiffusersPipeline, "combine_multi_branch_cfg_noise", combine)
    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    assert pipeline.combine_multi_branch_cfg_noise(predictions, scale, normalize) is expected


def test_caption_lengths_follow_cfg_branch_identity_without_tensor_reductions():
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    texts = [torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[6, 7, 8, 9, 10]])]
    lengths = {texts[0].data_ptr(): (2, 3), texts[1].data_ptr(): (4, 1)}
    received = []

    def transformer(**kwargs):
        assert "_multiview_caption_lengths" not in kwargs
        received.append(kwargs["caption_lengths"])
        return kwargs["text_ids"]

    pipeline.transformer = transformer
    for text in (texts[0], texts[1], texts[0]):
        pipeline.predict_noise(text_ids=text, _multiview_caption_lengths=lengths)
    assert received == [(2, 3), (4, 1), (2, 3)]


@pytest.mark.parametrize("joint", [False, True])
def test_encoder_offload_declaration_only_includes_loaded_modules(monkeypatch, joint):
    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    config = _versioned_deployment_config()
    if joint:
        config["lidar"] = {}
    monkeypatch.setattr(module, "_validated_multiview_deployment_config", lambda _: config)
    monkeypatch.setattr(module, "validate_multiview_parallel_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.Cosmos3LidarEncoder, "from_pretrained", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(module.Cosmos3LidarDecoder, "from_pretrained", lambda *args: torch.nn.Identity())

    def initialize(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cuda")
        transformer = object.__new__(module.Cosmos3MultiviewVFMTransformer)
        torch.nn.Module.__init__(transformer)
        self.transformer = transformer

    monkeypatch.setattr(Cosmos3OmniDiffusersPipeline, "__init__", initialize)
    pipeline = module.Cosmos3MultiviewPipeline(
        od_config=SimpleNamespace(
            tf_model_config={}, parallel_config=None, enable_session_state_manager=False, model="unused"
        )
    )
    assert pipeline._encoder_modules == (["lidar_encoder"] if joint else [])
    assert pipeline._vae_modules == (["vae", "lidar_decoder"] if joint else ["vae"])
    assert all(isinstance(getattr(pipeline, name), torch.nn.Module) for name in pipeline._encoder_modules)
    assert module.Cosmos3MultiviewPipeline._encoder_modules == []


def test_multiview_padding_replicates_last_frame_and_rejects_empty_media() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _pad_multiview_view_video,
    )

    frames = torch.stack(
        [torch.full((3, 2, 3), value, dtype=torch.uint8) for value in (10, 20)],
        dim=1,
    )
    padded = _pad_multiview_view_video(frames, num_frames=5, height=2, width=3)
    assert padded.shape == (3, 5, 2, 3)
    assert padded.is_contiguous()
    assert padded[:, 0].unique().item() == 10
    assert padded[:, 1:].unique().tolist() == [20]

    # A longer clip is truncated, not padded.
    truncated = _pad_multiview_view_video(frames, num_frames=1, height=2, width=3)
    assert truncated.shape == (3, 1, 2, 3)
    assert truncated.unique().tolist() == [10]

    # Admission guarantees every camera has media, so an empty decode is a
    # failure rather than a silently gray camera.
    with pytest.raises(ValueError, match="zero frames"):
        _pad_multiview_view_video(frames[:, :0], num_frames=3, height=2, width=3)


def test_multiview_request_preserves_selected_camera_order_and_requires_hint_with_controls() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        Cosmos3MultiviewPipeline,
    )

    cameras = ("front", "left")
    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.multiview_cameras = cameras
    pipeline.multiview_config = {"schema_version": 2, "variable_view_count": True}
    sp = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": _views(cameras)}})
    multiview, parsed_views = pipeline._parse_multiview_request(sp)
    assert multiview["views"] == parsed_views

    reordered = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": _views(tuple(reversed(cameras)))}})
    _, reordered_views = pipeline._parse_multiview_request(reordered)
    assert [view["camera_key"] for view in reordered_views] == ["left", "front"]

    no_wsm = SimpleNamespace(extra_args={"multiview": {"views": _views(cameras)}})
    with pytest.raises(ValueError, match="exactly one"):
        pipeline._parse_multiview_request(no_wsm)


@pytest.mark.parametrize(
    "config", [{}, {"variable_view_count": True}, {"schema_version": 2, "variable_view_count": False}]
)
@pytest.mark.parametrize("selection", ["full", "subset", "reordered"])
def test_fixed_camera_requests_require_exported_order(config, selection):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.multiview_cameras = ("front", "rear")
    pipeline.multiview_config = config
    cameras = {"full": ("front", "rear"), "subset": ("front",), "reordered": ("rear", "front")}[selection]
    sp = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": _views(cameras)}})
    if selection == "full":
        _, views = pipeline._parse_multiview_request(sp)
        assert [view["camera_key"] for view in views] == list(cameras)
    else:
        with pytest.raises(ValueError, match="exported camera order"):
            pipeline._parse_multiview_request(sp)


def test_multiview_resolution_does_not_inherit_image_default() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _resolve_multiview_resolution,
    )

    sp = SimpleNamespace(resolution=640, extra_args={})
    assert _resolve_multiview_resolution(sp, {}) == "480"
    assert _resolve_multiview_resolution(sp, {"resolution": 480}) == "480"

    sp.extra_args["resolution"] = "480"
    assert _resolve_multiview_resolution(sp, {}) == "480"


@pytest.mark.parametrize("resolution", [480, "480", 720, "720"])
def test_multiview_resolution_accepts_both_buckets_and_preserves_precedence(resolution) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_resolution

    sp = SimpleNamespace(resolution=640, extra_args={"resolution": resolution})
    assert _resolve_multiview_resolution(sp, {}) == str(resolution)
    sp.extra_args["resolution"] = "unsupported"
    assert _resolve_multiview_resolution(sp, {"resolution": resolution}) == str(resolution)


@pytest.mark.parametrize("resolution", [256, 704, "1080", "720p", True])
def test_multiview_resolution_rejects_unsupported_buckets(resolution) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_resolution

    with pytest.raises(ValueError, match="supports resolutions '480' and '720'"):
        _resolve_multiview_resolution(SimpleNamespace(extra_args={}), {"resolution": resolution})


@pytest.mark.parametrize("resolution,width,height", [("480", 832, 480), ("720", 1280, 720)])
@pytest.mark.parametrize("dimension", ["width", "height"])
def test_multiview_rejects_mismatched_dimensions_before_loading_media(resolution, width, height, dimension) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.multiview_cameras = ("front", "rear")
    pipeline.vae_scale_factor_temporal = 4
    sp = SimpleNamespace(
        extra_args={
            "wsm": {},
            "aspect_ratio": "16:9",
            "multiview": {
                "views": _views(pipeline.multiview_cameras),
                "resolution": resolution,
            },
        },
        num_frames=29,
        width=width,
        height=height,
    )
    setattr(sp, dimension, getattr(sp, dimension) + 16)
    with pytest.raises(ValueError, match=f"resolution='{resolution}' requires {dimension}="):
        pipeline.forward(SimpleNamespace(prompts=["drive"], sampling_params=sp))


@pytest.mark.parametrize("resolution,aspect_ratio,width,height,patch_hw", MULTIVIEW_BUCKETS)
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("has_vision", [False, True])
def test_multiview_forward_propagates_resolution(
    monkeypatch, resolution, aspect_ratio, width, height, patch_hw, automatic, has_vision
) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.vae_scale_factor_spatial = 16
    pipeline.vae_scale_factor_temporal = 4
    pipeline.is_distilled_model = False
    pipeline.multiview_cameras = ("front", "rear")
    pipeline.multiview_align_temporal_positions_across_views = False
    pipeline.multiview_attention_scope = "decomposed"
    pipeline.multiview_decomposed_temporal_window_seconds = None
    pipeline.multiview_control_attends_sensor = False
    pipeline.multiview_backend = "triton"
    transformer = object.__new__(Cosmos3MultiviewVFMTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.latent_channel_size = 3
    transformer.latent_patch_size = 2
    pipeline.transformer = transformer
    pipeline.scheduler = SimpleNamespace(timesteps=torch.tensor([1]))
    monkeypatch.setattr(pipeline, "_set_timesteps", lambda *args, **kwargs: None)

    prepared = []

    def prepare(views, *, field, height, width, num_frames, keep_first, require_complete=False):
        prepared.append((field, height, width, num_frames, keep_first))
        # Broadcast camera markers to avoid allocating full-resolution input clips.
        markers = torch.tensor([0.25] * num_frames + [0.75] * num_frames)
        return markers.view(1, 1, -1, 1, 1).expand(1, 3, -1, height, width)

    monkeypatch.setattr(pipeline, "_prepare_camera_major_pixels", prepare)
    monkeypatch.setattr(pipeline, "_encode_video_tensor", lambda video: video[:, :, ::4, ::16, ::16].clone())

    def decode(latents):
        assert latents.shape[-2:] == (height // 16, width // 16)
        return latents[:, :, :1, :1, :1].expand(1, 3, 5, height, width)

    monkeypatch.setattr(pipeline, "_decode_latents", decode)
    prompts = []

    def tokenize(prompt, *args, **kwargs):
        prompts.append(prompt)
        return torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)

    monkeypatch.setattr(pipeline, "_tokenize_prompt", tokenize)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview.media_hw",
        lambda value: (height, width),
    )

    def diffuse(*, latents, shared_kwargs, velocity_mask, condition_latents, control_latents, **kwargs):
        from vllm_omni.diffusion.models.cosmos3.multiview_packing import pack_state, unpack_state

        shapes = shared_kwargs["packed_shapes"]
        latents = unpack_state(latents, shapes)[0]
        assert latents.shape == (1, 3, 4, height // 16, width // 16)
        assert velocity_mask.shape == (1, 1, 4, 1, 1)
        assert condition_latents.shape == latents.shape
        control = control_latents[0]
        assert control.shape == latents.shape
        assert shared_kwargs["video_shape"] == tuple(latents.shape[2:])
        layout = shared_kwargs["multiview_layout"]
        assert (layout.patch_height, layout.patch_width) == patch_hw
        assert layout.gen_tokens == 2 * 4 * patch_hw[0] * patch_hw[1]
        assert shared_kwargs["noisy_frame_mask"].flatten().tolist() == ([0, 1, 0, 1] if has_vision else [1, 1, 1, 1])
        if has_vision:
            torch.testing.assert_close(condition_latents[:, :, ::2], control[:, :, ::2])
        # Exercise the real patch padding/crop before the per-camera decoder.
        return pack_state(
            [transformer.unpatchify(transformer.patchify(control, *control.shape[2:]), *control.shape[2:])]
        )

    monkeypatch.setattr(pipeline, "diffuse_transfer", diffuse)
    sp = OmniDiffusionSamplingParams(
        num_frames=5,
        num_inference_steps=1,
        width=width,
        height=height,
        seed=42,
        extra_args={
            "wsm": {},
            "multiview": {
                "resolution": resolution,
                "aspect_ratio": "auto" if automatic else aspect_ratio,
                "views": _views(pipeline.multiview_cameras, vision=has_vision),
                "condition_video_as_image": True,
            },
        },
    )
    request_prompt = json.dumps({"caption": "drive", "aspect_ratio": "21:9"}) if automatic else "drive"
    result = pipeline.forward(SimpleNamespace(prompts=[request_prompt], sampling_params=sp))
    assert prepared == ([("vision", height, width, 5, True)] if has_vision else []) + [
        ("control", height, width, 5, False)
    ]
    assert len(prompts) == 2
    if automatic:
        metadata_prompt, _ = json.JSONDecoder().raw_decode(prompts[0])
        assert metadata_prompt["aspect_ratio"] == aspect_ratio
        assert metadata_prompt["resolution"] == {"H": height, "W": width}
    else:
        assert f"This video is of {height}x{width} resolution." in prompts[0]
    assert f"This video is of {height}x{width} resolution." in prompts[1]
    video = result.output["payload"]["video"]
    assert video.shape == (1, 3, 10, height, width)
    assert video[0, 0, :, -1, -1].tolist() == [0.25] * 5 + [0.75] * 5
    metadata = result.output["metadata"]["multiview"]
    assert metadata["cameras"] == ["front", "rear"]
    assert (metadata["resolution"], metadata["aspect_ratio"], metadata["width"], metadata["height"]) == (
        resolution,
        aspect_ratio,
        width,
        height,
    )


@pytest.mark.parametrize("resolution,aspect_ratio,width,height,patch_hw", MULTIVIEW_BUCKETS)
@pytest.mark.parametrize("kind", ["tensor", "array", "frames", "image_path", "video_path"])
def test_multiview_auto_uses_first_wsm_input(
    tmp_path, monkeypatch, resolution, aspect_ratio, width, height, patch_hw, kind
):
    import numpy as np
    from PIL import Image

    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_geometry

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if kind == "tensor":
        control = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(1)
    elif kind == "array":
        control = frame[None]
    elif kind == "frames":
        control = [Image.fromarray(frame)]
    elif kind == "image_path":
        control = tmp_path / "control.png"
        Image.fromarray(frame).save(control)
    else:
        import imageio.v3 as iio

        control = tmp_path / "control.mp4"

        def first_frame_only(path):
            assert path == control
            yield frame
            pytest.fail("Aspect ratio detection must not decode a second video frame")

        monkeypatch.setattr(iio, "imiter", first_frame_only)
    # Other cameras and even this camera's vision input cannot change selection.
    views = [
        {"camera_key": "front", "control": control, "vision_path": "unread-vision.mp4"},
        {"camera_key": "rear", "control_path": "unread-control.mp4"},
    ]
    sp = SimpleNamespace(extra_args={"resolution": resolution}, width=None, height=None)
    assert _resolve_multiview_geometry(sp, {}, views) == (resolution, aspect_ratio, width, height)


@pytest.mark.parametrize("source_hw", [(1080, 1920), (1920, 1080), (731, 991), (1000, 1000), (147, 170)])
@pytest.mark.parametrize("resolution", ["480", "720"])
def test_multiview_auto_matches_cosmos_bucket_selection(monkeypatch, source_hw, resolution):
    from vllm_omni.diffusion.models.cosmos3.action import find_closest_target_size
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_geometry

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview.media_hw",
        lambda _: source_hw,
    )
    sp = SimpleNamespace(extra_args={"resolution": resolution})
    result = _resolve_multiview_geometry(sp, {}, [{"camera_key": "front", "control": "input.mp4"}])
    assert result[2:] == find_closest_target_size(*source_hw, resolution)


def test_multiview_explicit_ratio_precedence_bypasses_detection(monkeypatch):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_geometry

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview.media_hw",
        lambda _: pytest.fail("Explicit aspect ratio must bypass input probing"),
    )
    sp = SimpleNamespace(extra_args={"aspect_ratio": "9:16"})
    assert _resolve_multiview_geometry(sp, {}, []) == ("480", "9,16", 480, 832)
    assert _resolve_multiview_geometry(sp, {"aspect_ratio": "32:18"}, []) == ("480", "16,9", 832, 480)


@pytest.mark.parametrize("probe_result", [None, (0, 640), (-1, 640), OSError("corrupt media")])
def test_multiview_auto_reports_unreadable_first_wsm(monkeypatch, probe_result):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_geometry

    def probe(_):
        if isinstance(probe_result, Exception):
            raise probe_result
        return probe_result

    monkeypatch.setattr("vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview.media_hw", probe)
    with pytest.raises(ValueError, match="camera 'front'.*'broken.mp4'"):
        _resolve_multiview_geometry(
            SimpleNamespace(extra_args={}),
            {},
            [{"camera_key": "front", "control_path": "broken.mp4"}],
        )


@pytest.mark.parametrize("dimension", ["width", "height"])
def test_multiview_auto_validates_dimension_constraints(monkeypatch, dimension):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _resolve_multiview_geometry

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview.media_hw",
        lambda _: (1080, 1080),
    )
    sp = SimpleNamespace(extra_args={}, **{dimension: 480})
    with pytest.raises(ValueError, match=f"requires {dimension}=640"):
        _resolve_multiview_geometry(sp, {}, [{"camera_key": "front", "control_path": "square.mp4"}])


def test_multiview_temporal_position_period_validates_actual_latent_geometry() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _resolve_temporal_position_period,
    )

    assert _resolve_temporal_position_period(6, 2, True) == 3
    assert _resolve_temporal_position_period(5, 2, False) is None
    with pytest.raises(ValueError, match="divisible by num_views"):
        _resolve_temporal_position_period(5, 2, True)


def test_multiview_skips_generic_dummy_warmup() -> None:
    from vllm_omni.diffusion.io_support import get_dummy_run_num_frames

    assert get_dummy_run_num_frames("Cosmos3MultiviewPipeline", supports_audio_input=False) == 0


def test_multiview_request_requires_all_or_none_per_view_media() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        Cosmos3MultiviewPipeline,
    )

    cameras = ("front", "left")
    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.multiview_cameras = cameras

    partial_vision = _views(cameras)
    partial_vision[0]["vision_path"] = "front.png"
    sp = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": partial_vision}})
    with pytest.raises(ValueError, match="complete RGB videos"):
        pipeline._parse_multiview_request(sp)

    mixed_control = _views(cameras)
    mixed_control[1]["control_path"] = "left.png"
    sp = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": mixed_control}})
    with pytest.raises(ValueError, match="all images or all videos"):
        pipeline._parse_multiview_request(sp)


def test_per_camera_vae_encode_and_decode_preserve_camera_major_order() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        Cosmos3MultiviewPipeline,
    )

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)

    def encode(video: torch.Tensor) -> torch.Tensor:
        # One latent frame per two source frames, retaining the camera marker.
        return video[:, :1, ::2]

    def decode(latents: torch.Tensor) -> torch.Tensor:
        return latents.repeat_interleave(2, dim=2)

    pipeline._encode_video_tensor = encode
    pipeline._decode_latents = decode
    pixels = torch.cat(
        [torch.full((1, 3, 5, 2, 2), value, dtype=torch.float32) for value in (1, 2)],
        dim=2,
    )
    latents = pipeline._encode_multiview_video(pixels, num_views=2, frames_per_view=5)
    assert latents.shape == (1, 1, 6, 2, 2)
    assert latents[:, :, :3].unique().tolist() == [1]
    assert latents[:, :, 3:].unique().tolist() == [2]

    decoded = pipeline._decode_multiview_latents(latents, num_views=2, latent_frames_per_view=3)
    assert decoded[:, :, :6].unique().tolist() == [1]
    assert decoded[:, :, 6:].unique().tolist() == [2]


def test_multiview_negative_prompt_is_caller_supplied() -> None:
    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module

    assert not hasattr(module.Cosmos3MultiviewPipeline, "_default_negative_prompt")
    assert not (Path(module.__file__).with_name("negative_prompt_multiview.json")).exists()


def test_multiview_negative_prompt_metadata_mode_defaults_to_same() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODE,
    )

    # The negative prompt carries the same duration/FPS and resolution
    # sentences as the positive one.
    assert COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODE == "same"


def test_multiview_frame_rate_is_request_driven_and_defaults_to_training_rate() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_DEFAULT_FPS,
        _resolve_multiview_frame_rate,
    )

    # 30 FPS is the training rate and only the fallback, not a pin.
    assert COSMOS3_MULTIVIEW_DEFAULT_FPS == 30.0
    assert _resolve_multiview_frame_rate(None) == 30.0
    assert _resolve_multiview_frame_rate(30) == 30.0
    assert _resolve_multiview_frame_rate(10) == 10.0
    assert _resolve_multiview_frame_rate(29.97) == pytest.approx(29.97)
    # Outside the recommended range is a warning, not an error.
    assert _resolve_multiview_frame_rate(60) == 60.0
    for bad in (0, -5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            _resolve_multiview_frame_rate(bad)
    with pytest.raises(TypeError):
        _resolve_multiview_frame_rate(True)
    with pytest.raises(TypeError):
        _resolve_multiview_frame_rate("fast")


def test_multiview_num_frames_round_up_to_vae_grid() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES,
        _resolve_multiview_num_frames,
    )

    assert COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES == 201
    assert _resolve_multiview_num_frames(None, 4) == 201
    # OmniDiffusionSamplingParams' legacy image default selects the variant default.
    assert _resolve_multiview_num_frames(1, 4) == 201
    assert _resolve_multiview_num_frames(93, 4) == 93
    # 200 is rounded up to 201 instead of being rejected.
    assert _resolve_multiview_num_frames(200, 4) == 201
    assert _resolve_multiview_num_frames(201, 4) == 201
    assert _resolve_multiview_num_frames("200", 4) == 201
    for bad in (0, -3):
        with pytest.raises(ValueError, match="greater than 1"):
            _resolve_multiview_num_frames(bad, 4)
    with pytest.raises(TypeError):
        _resolve_multiview_num_frames(True, 4)
    with pytest.raises(TypeError):
        _resolve_multiview_num_frames("many", 4)


def test_multiview_transformer_resolver() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import resolve_cosmos3_transformer_cls
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_BACKBONE_TYPE,
        Cosmos3MultiviewVFMTransformer,
    )

    assert (
        resolve_cosmos3_transformer_cls({"backbone_type": COSMOS3_MULTIVIEW_BACKBONE_TYPE})
        is Cosmos3MultiviewVFMTransformer
    )


def test_regular_cosmos3_import_keeps_multiview_flex_attention_lazy() -> None:
    code = """
import sys
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import resolve_cosmos3_transformer_cls

transformer_module = "vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview"
attention_module = "vllm_omni.diffusion.models.cosmos3.multiview_flex_attention"
assert transformer_module not in sys.modules
assert attention_module not in sys.modules
resolved = resolve_cosmos3_transformer_cls({"backbone_type": "cosmos3_multiview"})
assert resolved.__name__ == "Cosmos3MultiviewVFMTransformer"
assert transformer_module in sys.modules
assert attention_module in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_multiview_pipeline_registration_contract() -> None:
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_IR_OP_PRIORITY_FUNCS,
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _NO_CACHE_ACCELERATION,
    )

    assert _DIFFUSION_MODELS["Cosmos3MultiviewPipeline"] == (
        "cosmos3",
        "pipeline_cosmos3_multiview",
        "Cosmos3MultiviewPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3MultiviewPipeline"] == "get_cosmos3_post_process_func"
    assert _DIFFUSION_IR_OP_PRIORITY_FUNCS["Cosmos3MultiviewPipeline"] == "get_cosmos3_ir_op_priority_func"
    assert "Cosmos3MultiviewPipeline" in _NO_CACHE_ACCELERATION


def test_attention_backend_env_overrides_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend is an implementation choice, not model behavior, so it is
    overridable without editing the checkpoint -- and a bad name must fail at
    load time rather than on the first generated frame."""
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_BACKEND_ENV,
        Cosmos3MultiviewPipeline,
    )

    resolve = Cosmos3MultiviewPipeline._resolve_attention_backend

    monkeypatch.delenv(COSMOS3_MULTIVIEW_BACKEND_ENV, raising=False)
    assert resolve({"backend": "fa4"}) == "fa4"
    with pytest.raises(ValueError, match="requires field 'backend'"):
        resolve({})

    monkeypatch.setenv(COSMOS3_MULTIVIEW_BACKEND_ENV, "fa4")
    assert resolve({"backend": "triton"}) == "fa4"
    monkeypatch.setenv(COSMOS3_MULTIVIEW_BACKEND_ENV, "triton")
    assert resolve({"backend": "fa4"}) == "triton"

    # An unset-looking value must not shadow the checkpoint.
    monkeypatch.setenv(COSMOS3_MULTIVIEW_BACKEND_ENV, "")
    assert resolve({"backend": "fa4"}) == "fa4"

    monkeypatch.setenv(COSMOS3_MULTIVIEW_BACKEND_ENV, "fa5")
    with pytest.raises(ValueError, match=COSMOS3_MULTIVIEW_BACKEND_ENV):
        resolve({"backend": "triton"})

    monkeypatch.delenv(COSMOS3_MULTIVIEW_BACKEND_ENV, raising=False)
    with pytest.raises(ValueError, match="multiview.backend"):
        resolve({"backend": "tirton"})


def test_multiview_deployment_contract_accepts_current_generic_artifact() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    config = _deployment_config()
    assert _validated_multiview_deployment_config(_deployment_model_config(config)) == config


@pytest.mark.parametrize("backbone_type", [None, "cosmos3", "cosmos3_edge"])
def test_multiview_deployment_contract_rejects_wrong_backbone_before_multiview_fields(
    backbone_type: str | None,
) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    with pytest.raises(ValueError, match="backbone_type='cosmos3_multiview'"):
        _validated_multiview_deployment_config({"backbone_type": backbone_type})


@pytest.mark.parametrize("strategy", ["teacher_forcing", "teacher_forcing_dcm"])
def test_multiview_deployment_contract_rejects_teacher_forcing_before_generic_fields(strategy: str) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    with pytest.raises(ValueError, match="replay/cached-memory inference"):
        _validated_multiview_deployment_config(_deployment_model_config({"causal_training_strategy": strategy}))


@pytest.mark.parametrize(
    "missing_field",
    [
        "causal_training_strategy",
        "attention_scope",
        "decomposed_temporal_window_seconds",
        "control_attends_sensor",
        "align_temporal_positions_across_views",
        "share_vision_temporal_positions",
        "backend",
        "cameras",
        "max_views",
    ],
)
def test_multiview_deployment_contract_rejects_missing_fields(missing_field: str) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    config = _deployment_config()
    del config[missing_field]
    with pytest.raises(ValueError, match=missing_field):
        _validated_multiview_deployment_config(_deployment_model_config(config))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("causal_training_strategy", 0),
        ("attention_scope", "same_view_or_frame"),
        ("decomposed_temporal_window_seconds", True),
        ("control_attends_sensor", "true"),
        ("align_temporal_positions_across_views", 1),
        ("share_vision_temporal_positions", "true"),
        ("backend", 1),
        ("cameras", "camera_front_wide_120fov"),
        ("max_views", True),
    ],
)
def test_multiview_deployment_contract_rejects_malformed_fields(field: str, bad_value: object) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    config = _deployment_config()
    config[field] = bad_value
    with pytest.raises((TypeError, ValueError), match=field):
        _validated_multiview_deployment_config(_deployment_model_config(config))


@pytest.mark.parametrize("bad_value", [True, "0.5"])
def test_multiview_deployment_contract_rejects_temporal_window_type(bad_value: object) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    config = _deployment_config()
    config["decomposed_temporal_window_seconds"] = bad_value
    with pytest.raises(TypeError, match="decomposed_temporal_window_seconds"):
        _validated_multiview_deployment_config(_deployment_model_config(config))


@pytest.mark.parametrize("bad_value", [-0.1, float("inf"), float("nan")])
def test_multiview_deployment_contract_rejects_temporal_window_value(bad_value: float) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    config = _deployment_config()
    config["decomposed_temporal_window_seconds"] = bad_value
    with pytest.raises(ValueError, match="decomposed_temporal_window_seconds"):
        _validated_multiview_deployment_config(_deployment_model_config(config))


def test_versioned_deployment_defaults_are_required_and_legacy_remains_compatible():
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _validated_multiview_deployment_config

    config = _deployment_config()
    config.update(schema_version=2, separate_view_text_tokenization=True, variable_view_count=True)
    with pytest.raises(ValueError, match="inference_defaults"):
        _validated_multiview_deployment_config(_deployment_model_config(config))
    defaults = {
        "resolution": "480",
        "fps": 30,
        "num_steps": 35,
        "guidance": 6,
        "shift": 10,
        "control_guidance": 1,
        "emphasize_control_in_prompt": True,
        "guidance_interval": None,
        "control_guidance_interval": None,
        "sigma_max": 80,
        "normalize_cfg": False,
        "negative_metadata_mode": "none",
    }
    config["inference_defaults"] = defaults
    assert _validated_multiview_deployment_config(_deployment_model_config(config)) == config
    for missing in defaults:
        config["inference_defaults"] = {key: value for key, value in defaults.items() if key != missing}
        with pytest.raises(ValueError, match="inference_defaults"):
            _validated_multiview_deployment_config(_deployment_model_config(config))
    legacy = _deployment_config()
    assert _validated_multiview_deployment_config(_deployment_model_config(legacy)) == legacy


@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("selection", ["subset", "reordered"])
def test_custom_deployment_rigs_require_versioned_metadata(versioned, selection):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _validated_multiview_deployment_config

    config = _versioned_deployment_config() if versioned else _deployment_config()
    config["cameras"] = config["cameras"][:3] if selection == "subset" else config["cameras"][::-1]
    config["max_views"] = len(config["cameras"])
    if versioned:
        assert _validated_multiview_deployment_config(_deployment_model_config(config)) == config
    else:
        # A stray flag must not relax the unversioned artifact contract.
        config["variable_view_count"] = True
        with pytest.raises(ValueError, match="canonical MADS camera order"):
            _validated_multiview_deployment_config(_deployment_model_config(config))


@pytest.mark.parametrize("overrides", ["none", "extra", "nested"])
def test_forward_uses_deployment_resolution_fps_and_emphasis_defaults(monkeypatch, overrides):
    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module
    from vllm_omni.diffusion.models.cosmos3.multiview_prompts import control_emphasis
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = object.__new__(module.Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device, pipeline.dtype = torch.device("cpu"), torch.float32
    pipeline.vae_scale_factor_temporal, pipeline.vae_scale_factor_spatial = 4, 16
    pipeline.is_distilled_model = False
    pipeline.multiview_cameras = module.COSMOS3_MADS_CAMERAS
    pipeline.multiview_config = _versioned_deployment_config()
    pipeline.multiview_config["inference_defaults"].update(resolution="720", fps=10, emphasize_control_in_prompt=False)
    pipeline.multiview_align_temporal_positions_across_views = True
    pipeline.multiview_attention_scope = "decomposed"
    pipeline.multiview_decomposed_temporal_window_seconds = 0.4
    pipeline.multiview_control_attends_sensor = True
    pipeline.multiview_backend = "triton"
    pipeline.transformer = SimpleNamespace(
        latent_channel_size=1,
        _pad_to_patch_size=lambda h, w: ((h + 1) // 2, (w + 1) // 2, 0, 0),
    )
    pipeline._set_timesteps = lambda *args, **kwargs: None
    pipeline.scheduler = SimpleNamespace(timesteps=torch.tensor([1000.0]))
    extra = {
        "wsm": {},
        "multiview": {
            "aspect_ratio": "16,9",
            "views": [{"camera_key": module.COSMOS3_MADS_CAMERAS[0], "control_path": "map.mp4", "prompt": "Driving."}],
        },
    }
    if overrides != "none":
        extra.update(resolution="480", fps=24, emphasize_control_in_prompt=True)
    if overrides == "nested":
        extra["resolution"] = "720"
        extra["multiview"]["resolution"] = "480"
        extra["frame_rate"] = 12
    expected_resolution, width, height = ("720", 1280, 720) if overrides == "none" else ("480", 832, 480)
    expected_fps = {"none": 10, "extra": 24, "nested": 12}[overrides]
    prepared, prompts, diffusion_calls = [], [], []

    def prepare(views, **kwargs):
        prepared.append(kwargs)
        return torch.zeros(1, 3, 5, 1, 1)

    pipeline._prepare_camera_major_pixels = prepare
    pipeline._encode_video_tensor = lambda _: torch.zeros(1, 1, 2, height // 16, width // 16)
    pipeline._decode_latents = lambda _: torch.zeros(1, 3, 5, 2, 2)

    def tokenize(prompt, *args, **kwargs):
        prompts.append(prompt)
        return torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)

    pipeline._tokenize_prompt = tokenize

    def diffuse(**kwargs):
        diffusion_calls.append(kwargs)
        return kwargs["latents"]

    pipeline.diffuse_transfer = diffuse
    sp = OmniDiffusionSamplingParams(num_frames=5, num_inference_steps=1, seed=42, extra_args=extra)
    result = pipeline.forward(SimpleNamespace(prompts=["ignored"], sampling_params=sp))
    metadata = result.output["metadata"]["multiview"]
    assert (metadata["resolution"], metadata["width"], metadata["height"], metadata["fps"]) == (
        expected_resolution,
        width,
        height,
        expected_fps,
    )
    assert (prepared[0]["width"], prepared[0]["height"]) == (width, height)
    shared = diffusion_calls[0]["shared_kwargs"]
    assert shared["fps"] == expected_fps
    assert shared["multiview_layout"].seconds_per_frame == 4 / expected_fps
    assert f"{expected_fps} FPS" in prompts[0]
    assert f"{height}x{width}" in prompts[0]
    assert prompts[0].count(control_emphasis("wsm", joint=False)) == int(overrides != "none")
    assert prompts[1] == ""
