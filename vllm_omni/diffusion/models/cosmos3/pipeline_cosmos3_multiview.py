# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 Multiview-AV pipeline.

Camera-only and joint V1.2 camera/LiDAR inference with independent sensor
geometries and per-camera captions. Only camera targets are decoded.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_extras.cosmos3 import (
    COSMOS3_MADS_CAMERAS,
    COSMOS3_TRANSFER_HINT_KEYS,
    normalize_multiview_aspect_ratio,
    validate_multiview_request,
)
from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames, required_lidar_sweeps

from .action import find_closest_target_size
from .lidar import Cosmos3LidarDecoder, Cosmos3LidarEncoder, validate_lidar_config
from .multiview_flex_attention import (
    DEFAULT_MAX_UND_TOKENS,
    MaskItem,
    MultiviewLayout,
    expand_multiview_condition_frame_indexes,
    validate_maskless_semantics,
    validate_multiview_backend,
)
from .multiview_packing import pack_state, unpack_state
from .multiview_parallel import validate_multiview_parallel_config
from .multiview_prompts import control_emphasis, format_camera_caption
from .pipeline_cosmos3 import (
    COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE,
    COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS,
    COSMOS3_TRANSFER_SYSTEM_PROMPT,
    COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT,
    Cosmos3OmniDiffusersPipeline,
    _ceil_video_num_frames,
    get_cosmos3_ir_op_priority_func,
    get_cosmos3_post_process_func,
)
from .transfer import (
    IMAGE_EXTENSIONS,
    as_bool,
    media_hw,
    media_to_uint8_cthw,
    uint8_cthw_to_normalized_5d,
)
from .transformer_cosmos3 import COSMOS3_MULTIVIEW_BACKBONE_TYPE, _tf_config_get
from .transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer
from .utils import VIDEO_RES_SIZE_INFO

logger = init_logger(__name__)

# Overrides transformer config multiview.backend, so the Triton and FA4 sparse
# attention paths can be compared without editing the checkpoint.
COSMOS3_MULTIVIEW_BACKEND_ENV = "VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND"

# Per-camera frame count when the request supplies none.
COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES = 201
# Frame rate when the request supplies none. The MADS WSM transfer recipes train
# on native 30 FPS clips, so the fps-modulated temporal mRoPE and the prompt
# metadata are on-distribution only at 30.
COSMOS3_MULTIVIEW_DEFAULT_FPS = 30.0
# Rates and frame counts outside these bounds are allowed with a warning.
COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE = (10.0, 30.0)
COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE = (24, 400)
# The negative prompt carries the same duration/FPS and resolution sentences as
# the positive prompt. Requests may override this through sampling params.
COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODE = "same"

# The tokenizer appends eos and vision_start after truncating. Derive the
# request ceiling from the sparse attention's single fixed UND capacity so the
# two cannot drift and accidentally trigger shape-specific recompilation.
COSMOS3_MULTIVIEW_PROMPT_FRAMING_TOKENS = 2
COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH = DEFAULT_MAX_UND_TOKENS - COSMOS3_MULTIVIEW_PROMPT_FRAMING_TOKENS
COSMOS3_MULTIVIEW_EMPHASIS = (
    "Follow the wsm control videos precisely for every camera view: shape, contour, position, and motion must "
    "align with the wsm signal at every frame."
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Cosmos3 multiview {name} must be an object, got {type(value).__name__}.")
    return value


def _media_kind(value: Any) -> str:
    if isinstance(value, str | Path):
        return "image" if Path(value).suffix.lower() in IMAGE_EXTENSIONS else "video"
    if isinstance(value, PIL.Image.Image):
        return "image"
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim == 5:
            return "image" if tensor.shape[2] == 1 else "video"
        if tensor.ndim == 4:
            temporal_dim = 1 if tensor.shape[0] in (3, 4) else 0
            return "image" if tensor.shape[temporal_dim] == 1 else "video"
        if tensor.ndim == 3:
            return "image"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "image" if len(value) == 1 else "video"
    raise TypeError(f"Unsupported Cosmos3 multiview media payload type: {type(value).__name__}.")


def _normalize_local_condition_indexes(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError("Cosmos3 multiview condition_frame_indexes_vision must be an int, list, or CSV string.")
    return sorted({int(index) for index in values})


def _pad_multiview_view_video(
    frames: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pad one camera to ``num_frames`` by replicating its last frame.

    The reference pads a view into a mid-gray canvas and then replicates the
    last decoded frame over the tail, so gray only ever survives for a view
    with no media at all.  Admission requires a control clip for every camera
    and vision for all cameras or none, so that case cannot reach here; an
    empty clip is a decode failure and is reported rather than silently
    generating a gray camera.
    """
    if frames.ndim != 4 or tuple(frames.shape[:1]) != (3,) or tuple(frames.shape[2:]) != (height, width):
        raise ValueError(
            "Cosmos3 multiview view frames must have shape [3, T, H, W] at the output resolution, "
            f"got {tuple(frames.shape)}."
        )
    fill = min(int(frames.shape[1]), num_frames)
    if fill <= 0:
        raise ValueError("Cosmos3 multiview view media decoded to zero frames.")
    video = frames[:, :fill]
    if fill < num_frames:
        video = torch.cat([video, video[:, -1:].expand(-1, num_frames - fill, -1, -1)], dim=1)
    return video.contiguous()


def _resolve_multiview_resolution(sp: Any, multiview: Mapping[str, Any], *, default: str = "480") -> str:
    """Resolve the variant-owned resolution without inheriting the image default."""
    resolution = multiview.get("resolution")
    if resolution is None:
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        resolution = extra.get("resolution")
    if resolution is None:
        resolution = default
    resolution = str(resolution)
    if resolution not in ("480", "720"):
        raise ValueError(f"Cosmos3 multiview supports resolutions '480' and '720', got {resolution!r}.")
    return resolution


def _resolve_multiview_geometry(
    sp: Any,
    multiview: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    *,
    default_resolution: str = "480",
) -> tuple[str, str, int, int]:
    """Select one bucket for every camera from an override or the first WSM."""
    resolution = _resolve_multiview_resolution(sp, multiview, default=default_resolution)
    extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
    requested_ratio = multiview.get("aspect_ratio")
    if requested_ratio is None:
        requested_ratio = extra.get("aspect_ratio")
    aspect_ratio = normalize_multiview_aspect_ratio(requested_ratio)
    sizes = VIDEO_RES_SIZE_INFO[resolution]
    if aspect_ratio == "auto":
        view = views[0]
        control = view.get("control_path", view.get("control"))
        if control is None:
            control = view.get("vision_path", view.get("vision"))
        if control is None:
            # Ordinary T2V has no source canvas; use the reference video bucket.
            return _resolve_multiview_geometry(
                sp, {**multiview, "aspect_ratio": "16,9"}, views, default_resolution=default_resolution
            )
        source = str(control) if isinstance(control, str | Path) else "decoded WSM input"
        try:
            source_hw = media_hw(control)
            if source_hw is None or min(source_hw) <= 0:
                raise ValueError("input has no readable positive spatial dimensions")
        except Exception as exc:
            raise ValueError(
                f"Cannot detect Cosmos3 multiview aspect ratio from camera {view['camera_key']!r} "
                f"WSM input {source!r}: {exc}"
            ) from exc
        width, height = find_closest_target_size(*source_hw, resolution)
        aspect_ratio = next(ratio for ratio, size in sizes.items() if size == (width, height))
    else:
        width, height = sizes[aspect_ratio]
    for key, expected in (("width", width), ("height", height)):
        requested = getattr(sp, key, None)
        if requested is not None and int(requested) != expected:
            raise ValueError(
                f"Cosmos3 multiview resolution={resolution!r} requires {key}={expected}, got {requested} "
                f"for aspect_ratio={aspect_ratio!r}."
            )
    return resolution, aspect_ratio, width, height


def _resolve_temporal_position_period(latent_frames: int, num_views: int, align_across_views: bool) -> int | None:
    if not align_across_views:
        return None
    if num_views <= 0 or latent_frames <= 0 or latent_frames % num_views:
        raise ValueError(
            "Aligning Cosmos3 multiview temporal positions requires positive latent frames divisible by num_views: "
            f"latent_frames={latent_frames}, num_views={num_views}."
        )
    return latent_frames // num_views


def _resolve_multiview_frame_rate(value: Any) -> float:
    """Resolve the request frame rate; ``None`` selects the default.

    Any finite positive rate is accepted: fps only feeds the fps-modulated
    temporal mRoPE, the prompt metadata, and the mask timestamps. Rates outside
    the recommended range are allowed with a warning.
    """
    if value is None:
        return COSMOS3_MULTIVIEW_DEFAULT_FPS
    if isinstance(value, bool):
        raise TypeError("Cosmos3 multiview FPS must be a number, not a boolean.")
    try:
        frame_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cosmos3 multiview FPS must be a number, got {value!r}.") from exc
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"Cosmos3 multiview FPS must be finite and positive, got {value!r}.")
    low, high = COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE
    if not low <= frame_rate <= high:
        logger.warning(
            "Cosmos3 multiview FPS %s is outside the recommended range [%s, %s]; the model was trained "
            "at %s FPS, so quality may be degraded.",
            frame_rate,
            low,
            high,
            COSMOS3_MULTIVIEW_DEFAULT_FPS,
        )
    return frame_rate


def _resolve_multiview_num_frames(value: Any, temporal_compression_factor: int) -> int:
    """Resolve the per-camera frame count, rounding up to the VAE grid.

    ``None`` and the ``OmniDiffusionSamplingParams`` legacy image default of one
    frame select the variant default. Other lengths are rounded up to the Wan
    VAE's ``4k+1`` grid instead of being rejected, so 200 becomes 201.
    """
    if value is None or (not isinstance(value, bool) and value == 1):
        return COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES
    if isinstance(value, bool):
        raise TypeError("Cosmos3 multiview num_frames must be an integer, not a boolean.")
    try:
        num_frames = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cosmos3 multiview num_frames must be an integer, got {value!r}.") from exc
    if num_frames <= 1:
        raise ValueError(f"Cosmos3 multiview num_frames must be greater than 1, got {value!r}.")
    rounded = _ceil_video_num_frames(num_frames, temporal_compression_factor)
    if rounded != num_frames:
        logger.info(
            "Rounded Cosmos3 multiview num_frames from %d to %d for temporal compression factor %d.",
            num_frames,
            rounded,
            temporal_compression_factor,
        )
    low, high = COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE
    if not low <= rounded <= high:
        logger.warning(
            "Cosmos3 multiview num_frames %d is outside the recommended range [%d, %d]; quality may be degraded.",
            rounded,
            low,
            high,
        )
    return rounded


def _required_deployment_field(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(f"Cosmos3 multiview transformer config requires field {name!r}.")
    return config[name]


def _validated_multiview_deployment_config(model_config: Any) -> dict[str, Any]:
    """Validate the flat exported contract before model initialization.

    The backbone is inspected before multiview-specific fields so selecting
    this pipeline for another Cosmos3 variant reports the actual mismatch.
    Within a multiview config, the training strategy is inspected first:
    teacher-forcing artifacts have a different replay/cached-memory runtime
    contract, so their generic fields must not obscure the targeted rejection.
    """
    backbone_type = _tf_config_get(model_config, "backbone_type", None)
    if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
        raise ValueError(
            "Cosmos3MultiviewPipeline requires transformer/config.json "
            f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
        )

    raw_config = _tf_config_get(model_config, "multiview", None)
    if raw_config is None:
        raise ValueError("Cosmos3 multiview transformer config must contain a 'multiview' object.")
    if hasattr(raw_config, "to_dict"):
        raw_config = raw_config.to_dict()
    config = _mapping(raw_config, "transformer config")

    strategy = _required_deployment_field(config, "causal_training_strategy")
    if not isinstance(strategy, str):
        raise TypeError("Cosmos3 multiview causal_training_strategy must be a string.")
    if strategy in {"teacher_forcing", "teacher_forcing_dcm"}:
        raise ValueError(
            f"Cosmos3 multiview {strategy} artifacts require replay/cached-memory inference, "
            "which vLLM-Omni does not support. Export a bidirectional causal_training_strategy='none' checkpoint."
        )
    if strategy != "none":
        raise ValueError(f"Cosmos3 multiview causal_training_strategy must be 'none' for vLLM-Omni, got {strategy!r}.")

    attention_scope = _required_deployment_field(config, "attention_scope")
    if not isinstance(attention_scope, str):
        raise TypeError("Cosmos3 multiview attention_scope must be a string.")
    if attention_scope not in {"all_views", "same_view", "decomposed"}:
        raise ValueError(
            "Cosmos3 multiview attention_scope must be one of ['all_views', 'decomposed', 'same_view']; "
            f"got {attention_scope!r}."
        )

    temporal_window = _required_deployment_field(config, "decomposed_temporal_window_seconds")
    if temporal_window is not None:
        if isinstance(temporal_window, bool) or not isinstance(temporal_window, int | float):
            raise TypeError("Cosmos3 multiview decomposed_temporal_window_seconds must be null or a number.")
        if not math.isfinite(temporal_window) or temporal_window < 0:
            raise ValueError("Cosmos3 multiview decomposed_temporal_window_seconds must be finite and non-negative.")
        temporal_window = float(temporal_window)

    for field_name in (
        "control_attends_sensor",
        "align_temporal_positions_across_views",
        "share_vision_temporal_positions",
    ):
        if not isinstance(_required_deployment_field(config, field_name), bool):
            raise TypeError(f"Cosmos3 multiview {field_name} must be boolean.")
    if not config["share_vision_temporal_positions"]:
        raise ValueError("Cosmos3 multiview requires share_vision_temporal_positions=true.")

    cameras = _required_deployment_field(config, "cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, str) and camera for camera in cameras)
    ):
        raise TypeError("Cosmos3 multiview cameras must be a non-empty list of strings.")
    if len(cameras) != len(set(cameras)):
        raise ValueError("Cosmos3 multiview cameras must be unique.")
    max_views = _required_deployment_field(config, "max_views")
    if isinstance(max_views, bool) or not isinstance(max_views, int):
        raise TypeError("Cosmos3 multiview max_views must be an integer.")
    if max_views != len(cameras):
        raise ValueError(
            "Cosmos3 multiview max_views must equal the exported camera list length: "
            f"max_views={max_views}, cameras={len(cameras)}."
        )

    version = config.get("schema_version")
    if version is not None and version != 2:
        raise ValueError(f"Unsupported Cosmos3 multiview schema_version={version}.")
    if version is None and tuple(cameras) != COSMOS3_MADS_CAMERAS:
        raise ValueError("Unversioned Cosmos3 multiview artifacts require the canonical MADS camera order.")
    if config.get("lidar") is not None:
        if version != 2:
            raise ValueError("Joint artifacts require versioned deployment metadata.")
        validate_lidar_config(dict(config["lidar"]))
    if version == 2:
        for field in ("per_view_captions", "variable_view_count"):
            if not isinstance(_required_deployment_field(config, field), bool):
                raise ValueError(f"Cosmos3 multiview {field} must be boolean.")
        defaults = _mapping(_required_deployment_field(config, "inference_defaults"), "inference_defaults")
        required_defaults = {
            "resolution",
            "fps",
            "num_steps",
            "guidance",
            "shift",
            "control_guidance",
            "emphasize_control_in_prompt",
            "guidance_interval",
            "control_guidance_interval",
            "sigma_max",
            "normalize_cfg",
            "negative_metadata_mode",
        }
        if missing := required_defaults - defaults.keys():
            raise ValueError(f"Incomplete inference_defaults metadata: {sorted(missing)}.")
        if defaults["resolution"] not in {"480", "720"}:
            raise ValueError("inference_defaults.resolution must be 480 or 720.")
        for name in ("fps", "num_steps", "guidance", "shift", "control_guidance", "sigma_max"):
            value = defaults[name]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"inference_defaults.{name} must be finite and non-negative.")
        if defaults["fps"] == 0 or defaults["num_steps"] < 1 or defaults["shift"] == 0:
            raise ValueError("inference_defaults requires positive FPS, step count and shift.")
    backend = _required_deployment_field(config, "backend")
    if not isinstance(backend, str):
        raise TypeError("Cosmos3 multiview backend must be a string.")
    validate_multiview_backend(backend)
    if backend == "maskless":
        if version != 2:
            raise ValueError("Maskless artifacts require multiview.schema_version=2; re-export the checkpoint.")
        validate_maskless_semantics(attention_scope, temporal_window, config["control_attends_sensor"])
    if not isinstance(config.get("lidar_attends_captions", True), bool):
        raise TypeError("Cosmos3 multiview lidar_attends_captions must be boolean.")
    return {
        **config,
        "decomposed_temporal_window_seconds": temporal_window,
    }


class Cosmos3MultiviewPipeline(Cosmos3OmniDiffusersPipeline):
    """Joint camera/LiDAR and camera-only generation with checkpoint-owned view layouts."""

    # The generic engine warmup has no per-camera WSM inputs and uses image
    # geometry that is invalid for this fixed-layout pipeline. Compile the
    # model on its first real request instead of weakening request validation.
    dummy_run_num_frames: ClassVar[int] = 0
    _encoder_modules: ClassVar[list[str]] = []

    def predict_noise(self, **kwargs):
        # Resolve branch metadata on the host before entering the transformer.
        lengths_by_text = kwargs.pop("_multiview_caption_lengths", None)
        if lengths_by_text is not None:
            kwargs["caption_lengths"] = lengths_by_text[kwargs["text_ids"].data_ptr()]
        return super().predict_noise(**kwargs)

    def combine_multi_branch_cfg_noise(self, predictions, true_cfg_scale, cfg_normalize=False):
        if not isinstance(true_cfg_scale, dict) or true_cfg_scale.get("mode") != "cosmos3_transfer":
            return super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize)
        combined = super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize=False)
        if cfg_normalize and true_cfg_scale.get("branch_mode") != "control_only":
            reference = predictions[0]
            if true_cfg_scale.get("branch_mode") == "control_and_text":
                reference = predictions[1] + true_cfg_scale["control_guidance"] * (predictions[0] - predictions[1])
            ratio = (reference.norm(dim=1, keepdim=True) / (combined.norm(dim=1, keepdim=True) + 1e-8)).clamp(0, 1)
            combined = combined * ratio
        return combined

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        multiview_config = _validated_multiview_deployment_config(od_config.tf_model_config)
        resolved_backend = self._resolve_attention_backend(multiview_config)
        validate_multiview_parallel_config(
            od_config.parallel_config,
            num_attention_heads=int(_tf_config_get(od_config.tf_model_config, "num_attention_heads", 32)),
            num_key_value_heads=int(_tf_config_get(od_config.tf_model_config, "num_key_value_heads", 8)),
            intermediate_size=int(_tf_config_get(od_config.tf_model_config, "intermediate_size", 12288)),
        )
        if od_config.enable_session_state_manager:
            raise ValueError("Cosmos3 multiview v1 does not support enable_session_state_manager.")
        super().__init__(od_config=od_config, prefix=prefix)
        if self.device.type != "cuda":
            raise ValueError("Cosmos3 multiview v1 requires CUDA for multiview attention.")
        if not isinstance(self.transformer, Cosmos3MultiviewVFMTransformer):
            raise ValueError(
                "Cosmos3MultiviewPipeline requires transformer/config.json backbone_type='cosmos3_multiview'."
            )

        self.multiview_config = multiview_config
        self._encoder_modules = []
        self._vae_modules = ["vae"]
        self.lidar_encoder = None
        self.lidar_decoder = None
        if multiview_config.get("lidar") is not None:
            self.lidar_encoder = Cosmos3LidarEncoder.from_pretrained(
                od_config.model, multiview_config["lidar"], self.device
            )
            self._encoder_modules = ["lidar_encoder"]
            self.lidar_decoder = Cosmos3LidarDecoder.from_pretrained(
                od_config.model, multiview_config["lidar"], self.device
            )
            self._vae_modules.append("lidar_decoder")
        self.multiview_cameras = tuple(multiview_config["cameras"])
        self.multiview_attention_scope = multiview_config["attention_scope"]
        self.multiview_decomposed_temporal_window_seconds = multiview_config["decomposed_temporal_window_seconds"]
        self.multiview_control_attends_sensor = multiview_config["control_attends_sensor"]
        self.multiview_align_temporal_positions_across_views = multiview_config["align_temporal_positions_across_views"]
        self.multiview_backend = resolved_backend

    @staticmethod
    def _resolve_attention_backend(multiview_config: Mapping[str, Any]) -> str:
        """Allow sparse kernel overrides, but never change checkpoint attention semantics.

        Triton and FA4 implement the same sparse predicate. Maskless intentionally
        counts overlapping branch keys twice and requires a matching checkpoint.
        """
        override = os.environ.get(COSMOS3_MULTIVIEW_BACKEND_ENV)
        backend = override if override else _required_deployment_field(multiview_config, "backend")
        if not isinstance(backend, str):
            raise TypeError("Cosmos3 multiview attention backend must be a string.")
        try:
            validate_multiview_backend(backend)
            source_backend = _required_deployment_field(multiview_config, "backend")
            if (backend == "maskless") != (source_backend == "maskless"):
                raise ValueError("Cannot override sparse attention with maskless or maskless with sparse attention.")
            return backend
        except ValueError as exc:
            source = (
                f"{COSMOS3_MULTIVIEW_BACKEND_ENV}={override!r}" if override else "transformer config multiview.backend"
            )
            raise ValueError(f"{exc} (from {source})") from exc

    def _parse_multiview_request(self, sp: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        config = getattr(self, "multiview_config", {})
        if extra.get("lidar") is not None and config.get("lidar") is None:
            raise ValueError("Joint camera+LiDAR requests require a complete joint checkpoint.")
        return validate_multiview_request(
            extra,
            self.multiview_cameras,
            media_kind=_media_kind,
            per_view_captions=config.get("per_view_captions", False),
            variable_view_count=config.get("schema_version") == 2 and config.get("variable_view_count") is True,
        )

    @staticmethod
    def _view_value(view: Mapping[str, Any], field: str) -> Any:
        return view.get(f"{field}_path", view.get(field))

    def _prepare_camera_major_pixels(
        self,
        views: Sequence[Mapping[str, Any]],
        *,
        field: str,
        height: int,
        width: int,
        num_frames: int,
        keep_first: bool,
        require_complete: bool = False,
    ) -> torch.Tensor:
        prepared = []
        for view in views:
            value = self._view_value(view, field)
            if value is None:
                if field == "vision":
                    prepared.append(torch.full((3, num_frames, height, width), 128, dtype=torch.uint8))
                    continue
                raise ValueError(f"Cosmos3 multiview camera {view['camera_key']!r} is missing {field} input.")
            frames = media_to_uint8_cthw(
                value,
                height=height,
                width=width,
                max_frames=1 if keep_first else num_frames,
            )
            if require_complete and frames.shape[1] < num_frames:
                raise ValueError(
                    f"Known camera {view['camera_key']!r} requires a complete RGB video of {num_frames} frames."
                )
            prepared.append(_pad_multiview_view_video(frames, num_frames=num_frames, height=height, width=width))
        camera_major = torch.cat(prepared, dim=1)
        return uint8_cthw_to_normalized_5d(camera_major, dtype=self.dtype)

    def _encode_multiview_video(
        self,
        camera_major_video: torch.Tensor,
        *,
        num_views: int,
        frames_per_view: int,
    ) -> torch.Tensor:
        expected_frames = num_views * frames_per_view
        if camera_major_video.ndim != 5 or camera_major_video.shape[2] != expected_frames:
            raise ValueError(
                "Cosmos3 multiview pixel video must be camera-major [1, 3, V*F, H, W]: "
                f"shape={tuple(camera_major_video.shape)}, V={num_views}, F={frames_per_view}."
            )
        per_view = [
            self._encode_video_tensor(camera_major_video[:, :, view * frames_per_view : (view + 1) * frames_per_view])
            for view in range(num_views)
        ]
        latent_frames = {int(latent.shape[2]) for latent in per_view}
        if len(latent_frames) != 1:
            raise ValueError(f"Cosmos3 multiview per-camera VAE encodes have unequal lengths: {latent_frames}.")
        return torch.cat(per_view, dim=2)

    def _decode_multiview_latents(
        self,
        camera_major_latents: torch.Tensor,
        *,
        num_views: int,
        latent_frames_per_view: int,
    ) -> torch.Tensor:
        if (
            num_views <= 0
            or latent_frames_per_view <= 0
            or camera_major_latents.ndim != 5
            or camera_major_latents.shape[2] != num_views * latent_frames_per_view
        ):
            raise ValueError(
                "Cosmos3 multiview latents must be camera-major before decode: "
                f"shape={tuple(camera_major_latents.shape)}, V={num_views}, F={latent_frames_per_view}."
            )
        decoded_output = None
        for view in range(num_views):
            decoded_view = self._decode_latents(
                camera_major_latents.narrow(2, view * latent_frames_per_view, latent_frames_per_view)
            )
            if decoded_view.ndim != 5 or decoded_view.shape[2] <= 0:
                raise ValueError("Cosmos3 multiview VAE decode must return non-empty [B, C, F, H, W] clips.")
            if decoded_output is None:
                view_shape = decoded_view.shape
                frames_per_view = view_shape[2]
                output_shape = (*view_shape[:2], num_views * frames_per_view, *view_shape[3:])
                decoded_output = torch.empty(output_shape, dtype=decoded_view.dtype, device="cpu")
            elif decoded_view.shape != view_shape or decoded_view.dtype != decoded_output.dtype:
                raise ValueError("Cosmos3 multiview VAE decoded camera clips must have matching shapes and dtypes.")
            # Copy each camera before decoding the next: retaining the GPU clips
            # and concatenating them would require two full multiview videos.
            decoded_output.narrow(2, view * frames_per_view, frames_per_view).copy_(decoded_view)
            del decoded_view

        assert decoded_output is not None
        return decoded_output

    def _mask_transfer_noise(
        self, noise: torch.Tensor, velocity_mask: torch.Tensor, shared_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        # Only camera frames are conditioned. Broadcast the compact temporal
        # mask on a view of the packed prediction; LiDAR needs no mask tensor.
        camera = unpack_state(noise, shared_kwargs["packed_shapes"])[0]
        camera.mul_(velocity_mask)
        return noise

    def _apply_transfer_condition(
        self,
        latents: torch.Tensor,
        velocity_mask: torch.Tensor,
        condition_latents: torch.Tensor,
        shared_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # UniPC and Euler return a new sample, separate from solver history.
        # Restore camera conditions there without repacking the LiDAR target.
        camera = unpack_state(latents, shared_kwargs["packed_shapes"])[0]
        camera.mul_(velocity_mask).add_((1.0 - velocity_mask) * condition_latents)
        return latents

    def _prepare_multiview_latents(
        self,
        *,
        target_pixels: torch.Tensor | None,
        condition_indexes: Sequence[int],
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
        generator: torch.Generator,
        injected_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_frames_per_view = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            self.transformer.latent_channel_size,
            num_views * latent_frames_per_view,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if injected_latents is None:
            noise = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:
            noise = injected_latents.to(device=self.device, dtype=self.dtype)
            if tuple(noise.shape) != shape:
                raise ValueError(
                    "Cosmos3 multiview injected latents have the wrong shape: "
                    f"expected={shape}, got={tuple(noise.shape)}."
                )
        condition_mask = torch.zeros(1, 1, shape[2], 1, 1, device=self.device, dtype=self.dtype)
        condition_latents = torch.zeros_like(noise)
        if condition_indexes:
            if target_pixels is None:
                raise ValueError("Cosmos3 multiview condition indexes require per-camera vision inputs.")
            encoded = self._encode_multiview_video(
                target_pixels,
                num_views=num_views,
                frames_per_view=num_frames,
            )
            if tuple(encoded.shape) != shape:
                raise ValueError(
                    f"Cosmos3 multiview target VAE latent shape mismatch: expected={shape}, got={tuple(encoded.shape)}."
                )
            for index in condition_indexes:
                condition_mask[:, :, index] = 1
                condition_latents[:, :, index : index + 1] = encoded[:, :, index : index + 1]
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        return latents, 1.0 - condition_mask, condition_latents

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("Cosmos3MultiviewPipeline supports exactly one prompt per request.")
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
            request_negative_prompt = None
        elif isinstance(prompt_data, Mapping):
            prompt = str(prompt_data.get("prompt", ""))
            request_negative_prompt = prompt_data.get("negative_prompt")
        else:
            raise TypeError(f"Unsupported Cosmos3 multiview prompt type: {type(prompt_data).__name__}.")

        sp = req.sampling_params
        multiview, views = self._parse_multiview_request(sp)
        num_views = len(views)
        deployment = getattr(self, "multiview_config", {})
        defaults = deployment.get("inference_defaults", {})
        lidar_request = self._get_sp_param(sp, "lidar", None)
        selected_hints = [key for key in COSMOS3_TRANSFER_HINT_KEYS if self._get_sp_param(sp, key, None) is not None]
        requested_num_frames = multiview.get("num_frames")
        if requested_num_frames is None:
            requested_num_frames = sp.num_frames
        num_frames = _resolve_multiview_num_frames(requested_num_frames, self.vae_scale_factor_temporal)
        resolution, aspect_ratio, width, height = _resolve_multiview_geometry(
            sp, multiview, views, default_resolution=defaults.get("resolution", "480")
        )
        frame_rate_value = self._get_sp_param(sp, "resolved_frame_rate", None)
        if frame_rate_value is None:
            frame_rate_value = self._get_sp_param(sp, "frame_rate", None)
        if frame_rate_value is None:
            frame_rate_value = self._get_sp_param(sp, "fps", None)
        if frame_rate_value is None:
            frame_rate_value = defaults.get("fps")
        frame_rate = _resolve_multiview_frame_rate(frame_rate_value)

        condition_video_as_image = as_bool(multiview.get("condition_video_as_image"), False)
        known_views = [index for index, view in enumerate(views) if self._view_value(view, "vision") is not None]
        has_vision = bool(known_views)
        completion = has_vision and len(known_views) < num_views
        vision_kind = _media_kind(self._view_value(views[known_views[0]], "vision")) if has_vision else None
        target_pixels = None
        if has_vision:
            target_pixels = self._prepare_camera_major_pixels(
                views,
                field="vision",
                height=height,
                width=width,
                num_frames=num_frames,
                keep_first=condition_video_as_image,
                require_complete=completion,
            )
        control_pixels = (
            self._prepare_camera_major_pixels(
                views,
                field="control",
                height=height,
                width=width,
                num_frames=num_frames,
                keep_first=False,
            )
            if selected_hints
            else None
        )

        latent_frames_per_view = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_t = num_views * latent_frames_per_view
        raw_indexes = multiview.get("condition_frame_indexes_vision")
        if raw_indexes is None:
            if not has_vision:
                local_indexes = []
            elif condition_video_as_image or vision_kind == "image":
                local_indexes = [0]
            else:
                local_indexes = [0, 1]
        else:
            local_indexes = _normalize_local_condition_indexes(raw_indexes)
        if completion:
            if raw_indexes is not None and local_indexes:
                raise ValueError("View completion conditions all frames of known views; omit condition frame indexes.")
            condition_indexes = [
                view * latent_frames_per_view + frame for view in known_views for frame in range(latent_frames_per_view)
            ]
        else:
            if any(index < 0 or index >= latent_frames_per_view for index in local_indexes):
                raise ValueError("Camera condition frame index is outside the generated latent clip.")
            condition_indexes = expand_multiview_condition_frame_indexes(local_indexes, num_views, latent_t)

        generator = sp.generator
        seed = self._resolve_seed(sp, generator)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        injected_latents = sp.latents if isinstance(sp.latents, torch.Tensor) else None
        latents, velocity_mask, condition_latents = self._prepare_multiview_latents(
            target_pixels=target_pixels,
            condition_indexes=condition_indexes,
            num_views=num_views,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
            injected_latents=injected_latents,
        )
        control_latents = (
            self._encode_multiview_video(
                control_pixels,
                num_views=num_views,
                frames_per_view=num_frames,
            )
            if control_pixels is not None
            else None
        )
        del target_pixels, control_pixels
        if control_latents is not None and control_latents.shape != latents.shape:
            raise ValueError(
                "Cosmos3 multiview WSM and target latent shapes must match: "
                f"control={tuple(control_latents.shape)}, target={tuple(latents.shape)}."
            )
        actual_latent_t = int(latents.shape[2])
        temporal_position_period = _resolve_temporal_position_period(
            actual_latent_t,
            num_views,
            self.multiview_align_temporal_positions_across_views,
        )

        max_sequence_length = int(
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH)
            or COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH
        )
        if max_sequence_length > COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Cosmos3 multiview max_sequence_length cannot exceed the variant ceiling the sparse "
                f"attention is sized for: requested={max_sequence_length}, "
                f"ceiling={COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH}."
            )
        patch_h, patch_w, _, _ = self.transformer._pad_to_patch_size(latents.shape[3], latents.shape[4])
        camera_shape = (actual_latent_t, patch_h, patch_w)
        camera_rate = self.vae_scale_factor_temporal / frame_rate
        items = (
            [MaskItem(camera_shape, num_views, is_control=True, seconds_per_frame=camera_rate)]
            if control_latents is not None
            else []
        )
        items.append(MaskItem(camera_shape, num_views, seconds_per_frame=camera_rate))
        targets = [latents]
        lidar_control_latents = None
        if lidar_request is not None:
            lidar_config = deployment["lidar"]
            sweeps = required_lidar_sweeps(num_frames, frame_rate, lidar_config["fps"])
            lidar_frames = load_lidar_frames(lidar_request["control_path"], num_sweeps=sweeps)
            lidar_control_latents = self.lidar_encoder(lidar_frames).to(device=self.device, dtype=self.dtype)
            del lidar_frames
            # Continue the request RNG after camera noise; reseeding would
            # reuse its initial stream and discard caller-supplied RNG state.
            lidar_noise = randn_tensor(
                lidar_control_latents.shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
            targets.append(lidar_noise)
            lt, lh, lw = lidar_noise.shape[2:]
            del lidar_noise
            lhp, lwp, _, _ = self.transformer._pad_to_patch_size(lh, lw)
            for is_control in (True, False):
                items.append(
                    MaskItem(
                        (lt, lhp, lwp),
                        1,
                        view_offset=num_views,
                        is_control=is_control,
                        is_lidar=True,
                        seconds_per_frame=lidar_config["temporal_compression_factor"] / lidar_config["fps"],
                    )
                )
        separate_captions = deployment.get("per_view_captions", False)
        layout = MultiviewLayout(
            attention_scope=self.multiview_attention_scope,  # type: ignore[arg-type]
            decomposed_temporal_window_seconds=self.multiview_decomposed_temporal_window_seconds,
            control_attends_sensor=self.multiview_control_attends_sensor,
            backend=self.multiview_backend,
            lidar_attends_captions=deployment.get("lidar_attends_captions", True),
            items=tuple(items),
            max_und_tokens=DEFAULT_MAX_UND_TOKENS * (num_views if separate_captions else 1),
        )

        # Same contract as the other Cosmos3 pipelines: no packaged default, an
        # unsupplied negative prompt is empty. Reference-parity runs must pass
        # the reference negative prompt explicitly (see the recipe); serializing
        # it with default json separators is the caller's job, exactly as it is
        # for Cosmos3-Nano and Cosmos3-Super.
        negative_prompt = request_negative_prompt
        if negative_prompt is None:
            negative_prompt = self._get_sp_param(sp, "negative_prompt", None)
        if negative_prompt is None:
            negative_prompt = ""
        negative_prompt = str(negative_prompt)
        emphasis = as_bool(
            self._get_sp_param(sp, "emphasize_control_in_prompt", defaults.get("emphasize_control_in_prompt", True)),
            True,
        )
        suffix = (
            control_emphasis(selected_hints[0], joint=lidar_request is not None)
            if selected_hints and emphasis
            else None
        )
        if deployment.get("schema_version") is None and selected_hints == ["wsm"] and emphasis:
            suffix = COSMOS3_MULTIVIEW_EMPHASIS
        captions = (
            [format_camera_caption(view["prompt"], view["camera_key"]) for view in views]
            if separate_captions
            else [prompt]
        )
        branches = []
        for caption in captions:
            branches.append(
                self._format_and_tokenize_prompts(
                    caption,
                    "" if separate_captions else negative_prompt,
                    num_frames,
                    frame_rate,
                    height,
                    width,
                    max_sequence_length,
                    sp,
                    use_system_prompt=True,
                    system_prompt=COSMOS3_TRANSFER_SYSTEM_PROMPT,
                    prompt_suffix=suffix,
                    use_duration_template=True,
                    use_resolution_template=True,
                    negative_metadata_mode="none"
                    if separate_captions
                    else str(
                        self._get_sp_param(
                            sp,
                            "negative_metadata_mode",
                            defaults.get("negative_metadata_mode", COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODE),
                        )
                    ),
                    aspect_ratio_override=aspect_ratio,
                )
            )

        # Positive integers identify separate text segments. Zero remains padding.
        # Each segment gets its own causal UND pass and shared position origin.
        def combine_branch(ids_index: int, mask_index: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
            ids, masks, lengths = [], [], []
            for index, branch in enumerate(branches, 1):
                real = branch[mask_index].bool()
                tokens = branch[ids_index][real].reshape(1, -1)
                ids.append(tokens)
                masks.append(torch.full_like(tokens, index))
                lengths.append(tokens.shape[1])
            return torch.cat(ids, dim=1), torch.cat(masks, dim=1), tuple(lengths)

        cond_ids, cond_mask, cond_lengths = combine_branch(0, 1)
        uncond_ids, uncond_mask, uncond_lengths = combine_branch(2, 3)

        guidance_scale = self._resolve_guidance_scale(sp, defaults.get("guidance", COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE))
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Cosmos3 multiview guidance must be finite and non-negative.")
        if deployment.get("schema_version") is None:
            guidance_scale = min(7.0, guidance_scale)
        num_inference_steps = int(
            sp.num_inference_steps or defaults.get("num_steps", COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS)
        )
        flow_shift = float(
            self._get_sp_param(sp, "flow_shift", defaults.get("shift", COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT))
        )
        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps
        self._set_flow_shift(flow_shift)
        self._set_timesteps(num_inference_steps, device=self.device, shift=flow_shift)

        video_shape = tuple(int(dim) for dim in latents.shape[2:])
        shared_kwargs = {
            "_multiview_caption_lengths": {
                cond_ids.data_ptr(): cond_lengths,
                uncond_ids.data_ptr(): uncond_lengths,
            },
            "video_shape": video_shape,
            "fps": frame_rate,
            "noisy_frame_mask": velocity_mask,
            "packed_shapes": tuple(tuple(tensor.shape[1:]) for tensor in targets),
            "lidar_control_latents": lidar_control_latents,
            "transfer_share_vision_temporal_positions": True,
            "temporal_position_period": temporal_position_period,
            "multiview_layout": layout,
        }
        # Transfer ownership of the initial state to the denoising loop. The
        # caller must not retain either source tensors or the packed sample.
        initial_state = [pack_state(targets)]
        del targets, latents
        packed = self.diffuse_transfer(
            latents=initial_state.pop(),
            timesteps=self.scheduler.timesteps,
            cond_ids=cond_ids,
            cond_mask=cond_mask,
            uncond_ids=uncond_ids,
            uncond_mask=uncond_mask,
            guidance_scale=guidance_scale,
            control_guidance=float(self._get_sp_param(sp, "control_guidance", defaults.get("control_guidance", 1.0))),
            control_guidance_interval=self._get_sp_param(
                sp, "control_guidance_interval", defaults.get("control_guidance_interval")
            ),
            guidance_interval=self._get_sp_param(sp, "guidance_interval", defaults.get("guidance_interval")),
            control_latents=[control_latents] if control_latents is not None else [],
            shared_kwargs=shared_kwargs,
            velocity_mask=velocity_mask,
            condition_latents=condition_latents,
            generator=generator,
            normalize_cfg=as_bool(self._get_sp_param(sp, "normalize_cfg", defaults.get("normalize_cfg", False)), False),
            open_guidance_interval=deployment.get("schema_version") == 2,
        )
        final_targets = unpack_state(packed, shared_kwargs["packed_shapes"])
        del condition_latents, control_latents, lidar_control_latents, shared_kwargs
        latents = final_targets[0]
        video = self._decode_multiview_latents(
            latents,
            num_views=num_views,
            latent_frames_per_view=latent_frames_per_view,
        ).clamp_(-1, 1)
        payload = {"video": video}
        lidar_metadata = {}
        if lidar_request is not None and lidar_request.get("return_output", False):
            lidar_output = self.lidar_decoder(final_targets[1])
            payload["lidar"] = lidar_output
            lidar_config = self.lidar_decoder.config
            lidar_metadata = {
                "lidar": {
                    "fps": lidar_config["fps"],
                    "num_frames": lidar_output.shape[2],
                    "shape": list(lidar_output.shape),
                    "dtype": "float32",
                    "channels": ["range", "intensity", "validity"],
                    "units": ["metres", "unit", "binary" if lidar_config["apply_validity_mask"] else "probability"],
                    "validity_threshold": lidar_config["range_projection"].get("validity_threshold", 0.5),
                    "apply_validity_mask": lidar_config["apply_validity_mask"],
                    "start_time_seconds": 0.0,
                    "range_projection": dict(lidar_config["range_projection"]),
                }
            }
        return DiffusionOutput(
            output={
                "payload": payload,
                "metadata": {
                    **lidar_metadata,
                    "multiview": {
                        "cameras": [view["camera_key"] for view in views],
                        "frames_per_view": num_frames,
                        "fps": frame_rate,
                        "resolution": resolution,
                        "aspect_ratio": aspect_ratio,
                        "width": width,
                        "height": height,
                    },
                },
            }
        )


__all__ = [
    "COSMOS3_MADS_CAMERAS",
    "Cosmos3MultiviewPipeline",
    "get_cosmos3_ir_op_priority_func",
    "get_cosmos3_post_process_func",
]
