# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


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


def test_multiview_request_pins_camera_order_and_wsm() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        Cosmos3MultiviewPipeline,
    )

    cameras = ("front", "left")
    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.multiview_cameras = cameras
    sp = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": _views(cameras)}})
    multiview, parsed_views = pipeline._parse_multiview_request(sp)
    assert multiview["views"] == parsed_views

    reordered = SimpleNamespace(extra_args={"wsm": {}, "multiview": {"views": _views(tuple(reversed(cameras)))}})
    with pytest.raises(ValueError, match="camera order"):
        pipeline._parse_multiview_request(reordered)

    no_wsm = SimpleNamespace(extra_args={"multiview": {"views": _views(cameras)}})
    with pytest.raises(ValueError, match="exactly one"):
        pipeline._parse_multiview_request(no_wsm)


def test_multiview_resolution_does_not_inherit_image_default() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _resolve_multiview_resolution,
    )

    sp = SimpleNamespace(resolution=640, extra_args={})
    assert _resolve_multiview_resolution(sp, {}) == "480"
    assert _resolve_multiview_resolution(sp, {"resolution": 480}) == "480"

    sp.extra_args["resolution"] = "480"
    assert _resolve_multiview_resolution(sp, {}) == "480"


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
    with pytest.raises(ValueError, match="every camera or none"):
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
