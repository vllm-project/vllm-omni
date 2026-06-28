# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 transfer inference helpers.

The reference Cosmos Framework transfer path accepts one or more control hints
and packs them before the noisy target video.  This module keeps the vLLM Omni
pipeline-specific pieces small: request parsing, media normalization, and
optional on-the-fly edge/blur control generation.

Transfer is a video-output mode. It cannot be combined with T2I, sound
generation, or action generation.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

TRANSFER_HINT_KEYS: tuple[str, ...] = ("edge", "blur", "depth", "seg", "wsm")
TRANSFER_SAMPLE_DEFAULTS: dict[str, Any] = {
    "num_video_frames_per_chunk": 93,
    "num_conditional_frames": 1,
    "max_frames": 5000,
    "show_control_condition": False,
    "show_input": False,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
}
TRANSFER_DEFAULTS: dict[str, dict[str, Any]] = {
    "edge": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "blur": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "depth": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "seg": {"guidance_scale": 3.0, "control_guidance": 2.0, "flow_shift": 10.0},
    "wsm": {
        "guidance_scale": 1.0,
        "control_guidance": 3.0,
        "flow_shift": 10.0,
        "num_frames": 101,
        "fps": 10,
        "num_video_frames_per_chunk": 101,
    },
}
EDGE_PRESETS: dict[str, tuple[int, int]] = {
    "none": (20, 50),
    "very_low": (20, 50),
    "low": (50, 100),
    "medium": (100, 200),
    "high": (200, 300),
    "very_high": (300, 400),
}
BLUR_DOWNUP_PRESETS: dict[str, int] = {
    "none": 1,
    "very_low": 4,
    "low": 4,
    "medium": 10,
    "high": 16,
    "very_high": 16,
}
BLUR_PRE_BLUR_DOWNSCALE_PRESETS: dict[str, int] = {
    "none": 1,
    "very_low": 1,
    "low": 4,
    "medium": 2,
    "high": 1,
    "very_high": 4,
}
BILATERAL_REFERENCE_RESOLUTION = 720
BILATERAL_D = 30
BILATERAL_SIGMA_COLOR = 150
BILATERAL_SIGMA_SPACE = 100
BILATERAL_ITERATIONS = 1
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class Cosmos3TransferHint:
    key: str
    control_path: str | None = None
    control: Any | None = None
    preset_edge_threshold: str = "medium"
    preset_blur_strength: str = "medium"


@dataclass
class Cosmos3TransferConfig:
    hints: dict[str, Cosmos3TransferHint] = field(default_factory=dict)
    guidance_scale: float | None = None
    control_guidance: float = 1.0
    control_guidance_interval: tuple[float, float] | None = None
    flow_shift: float | None = None
    num_video_frames_per_chunk: int = 93
    num_conditional_frames: int = 1
    max_frames: int = 5000
    show_control_condition: bool = False
    show_input: bool = False
    num_first_chunk_conditional_frames: int = 0
    share_vision_temporal_positions: bool = True
    num_frames: int | None = None
    fps: float | None = None

    @property
    def ordered_hints(self) -> list[Cosmos3TransferHint]:
        return [self.hints[key] for key in TRANSFER_HINT_KEYS if key in self.hints]


def _extra_args(sp: Any) -> Mapping[str, Any]:
    extra = getattr(sp, "extra_args", None)
    return extra if isinstance(extra, Mapping) else {}


def _prompt_value(prompt_data: Any, key: str) -> Any:
    if not isinstance(prompt_data, Mapping):
        return None
    if prompt_data.get(key) is not None:
        return prompt_data[key]
    additional = prompt_data.get("additional_information")
    if isinstance(additional, Mapping):
        return additional.get(key)
    return None


def _param(extra: Mapping[str, Any], sp: Any, prompt_data: Any, key: str, default: Any = None) -> Any:
    if extra.get(key) is not None:
        return extra[key]
    value = getattr(sp, key, None)
    if value is not None:
        return value
    value = _prompt_value(prompt_data, key)
    return default if value is None else value


def _param_any(extra: Mapping[str, Any], sp: Any, prompt_data: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        value = _param(extra, sp, prompt_data, key, None)
        if value is not None:
            return value
    return default


def _is_user_field(extra: Mapping[str, Any], sp: Any, prompt_data: Any, *keys: str) -> bool:
    for key in keys:
        if key in extra and extra[key] is not None:
            return True
        if getattr(sp, key, None) is not None:
            return True
        if _prompt_value(prompt_data, key) is not None:
            return True
    return False


# fps, frame_rate, and resolved_frame_rate are aliases for the same request-level
# control. The serving layer leaves all of them ``None`` on the sampling params when
# the user did not provide fps, so a non-``None`` value here unambiguously means the
# user set it.
_FPS_KEYS: tuple[str, ...] = ("resolved_frame_rate", "frame_rate", "fps")


def _transfer_fps_param(extra: Mapping[str, Any], sp: Any, prompt_data: Any) -> Any:
    # extra_args is a model-specific override and wins over the generic sampling params.
    for key in _FPS_KEYS:
        if extra.get(key) is not None:
            return extra[key]
    return _param_any(extra, sp, prompt_data, _FPS_KEYS)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_interval(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Cosmos3 transfer control_guidance_interval must contain exactly two values.")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f"Cosmos3 transfer control_guidance_interval must be ordered as [lo, hi], got {(lo, hi)}.")
    return lo, hi


def has_transfer_hints(extra_args: Mapping[str, Any] | None) -> bool:
    if not isinstance(extra_args, Mapping):
        return False
    return any(extra_args.get(key) is not None for key in TRANSFER_HINT_KEYS)


def transfer_max_frames_from_extra_args(extra_args: Mapping[str, Any] | None) -> int:
    if not isinstance(extra_args, Mapping):
        return int(TRANSFER_SAMPLE_DEFAULTS["max_frames"])
    value = extra_args.get("max_frames", TRANSFER_SAMPLE_DEFAULTS["max_frames"])
    return max(1, int(value))


def resolve_transfer_config(sp: Any, prompt_data: Any = None) -> Cosmos3TransferConfig | None:
    extra = _extra_args(sp)
    hints: dict[str, Cosmos3TransferHint] = {}
    for key in TRANSFER_HINT_KEYS:
        raw = _param(extra, sp, prompt_data, key, None)
        if raw is None:
            continue
        if raw is True:
            raw = {}
        elif isinstance(raw, str | Path):
            raw = {"control_path": str(raw)}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Cosmos3 transfer hint '{key}' must be an object, path string, or true; got {type(raw)!r}."
            )
        hints[key] = Cosmos3TransferHint(
            key=key,
            control_path=str(raw["control_path"]) if raw.get("control_path") is not None else None,
            control=raw.get("control"),
            preset_edge_threshold=str(raw.get("preset_edge_threshold") or "medium").lower(),
            preset_blur_strength=str(raw.get("preset_blur_strength") or "medium").lower(),
        )

    if not hints:
        transfer_only = (
            "control_guidance",
            "control_guidance_interval",
            "num_video_frames_per_chunk",
            "num_conditional_frames",
            "max_frames",
            "show_control_condition",
            "show_input",
            "num_first_chunk_conditional_frames",
            "share_vision_temporal_positions",
        )
        if any(_is_user_field(extra, sp, prompt_data, key) for key in transfer_only):
            raise ValueError("Cosmos3 transfer options were provided, but no transfer hint was selected.")
        return None

    fps_value = _transfer_fps_param(extra, sp, prompt_data)
    config = Cosmos3TransferConfig(
        hints=hints,
        guidance_scale=(
            float(_param_any(extra, sp, prompt_data, ("guidance_scale", "guidance")))
            if _param_any(extra, sp, prompt_data, ("guidance_scale", "guidance")) is not None
            else None
        ),
        control_guidance=float(_param(extra, sp, prompt_data, "control_guidance", 1.0)),
        control_guidance_interval=_as_interval(_param(extra, sp, prompt_data, "control_guidance_interval", None)),
        flow_shift=(
            float(_param_any(extra, sp, prompt_data, ("flow_shift", "shift")))
            if _param_any(extra, sp, prompt_data, ("flow_shift", "shift")) is not None
            else None
        ),
        num_video_frames_per_chunk=int(
            _param(
                extra,
                sp,
                prompt_data,
                "num_video_frames_per_chunk",
                TRANSFER_SAMPLE_DEFAULTS["num_video_frames_per_chunk"],
            )
        ),
        num_conditional_frames=int(
            _param(extra, sp, prompt_data, "num_conditional_frames", TRANSFER_SAMPLE_DEFAULTS["num_conditional_frames"])
        ),
        max_frames=int(_param(extra, sp, prompt_data, "max_frames", TRANSFER_SAMPLE_DEFAULTS["max_frames"])),
        show_control_condition=_as_bool(
            _param(extra, sp, prompt_data, "show_control_condition", None),
            bool(TRANSFER_SAMPLE_DEFAULTS["show_control_condition"]),
        ),
        show_input=_as_bool(
            _param(extra, sp, prompt_data, "show_input", None),
            bool(TRANSFER_SAMPLE_DEFAULTS["show_input"]),
        ),
        num_first_chunk_conditional_frames=int(
            _param(
                extra,
                sp,
                prompt_data,
                "num_first_chunk_conditional_frames",
                TRANSFER_SAMPLE_DEFAULTS["num_first_chunk_conditional_frames"],
            )
        ),
        share_vision_temporal_positions=_as_bool(
            _param(extra, sp, prompt_data, "share_vision_temporal_positions", None),
            bool(TRANSFER_SAMPLE_DEFAULTS["share_vision_temporal_positions"]),
        ),
        num_frames=(
            int(_param(extra, sp, prompt_data, "num_frames"))
            if _param(extra, sp, prompt_data, "num_frames", None) is not None
            else None
        ),
        fps=(float(fps_value) if fps_value is not None else None),
    )

    if len(hints) == 1:
        hint_key = next(iter(hints))
        for field_name, default_value in TRANSFER_DEFAULTS[hint_key].items():
            if field_name == "guidance_scale":
                user_set = _is_user_field(extra, sp, prompt_data, "guidance_scale", "guidance")
            elif field_name == "flow_shift":
                user_set = _is_user_field(extra, sp, prompt_data, "flow_shift", "shift")
            elif field_name == "fps":
                user_set = _is_user_field(extra, sp, prompt_data, *_FPS_KEYS)
            else:
                user_set = _is_user_field(extra, sp, prompt_data, field_name)
            if not user_set:
                setattr(config, field_name, default_value)

    if config.num_video_frames_per_chunk <= 0:
        raise ValueError("Cosmos3 transfer num_video_frames_per_chunk must be positive.")
    if config.num_conditional_frames < 0:
        raise ValueError("Cosmos3 transfer num_conditional_frames must be non-negative.")
    if config.max_frames <= 0:
        raise ValueError("Cosmos3 transfer max_frames must be positive.")
    if config.num_first_chunk_conditional_frames < 0:
        raise ValueError("Cosmos3 transfer num_first_chunk_conditional_frames must be non-negative.")
    for hint in hints.values():
        if hint.key == "edge" and hint.preset_edge_threshold not in EDGE_PRESETS:
            raise ValueError(f"Unsupported Cosmos3 edge preset: {hint.preset_edge_threshold!r}.")
        if hint.key == "blur" and hint.preset_blur_strength not in BLUR_DOWNUP_PRESETS:
            raise ValueError(f"Unsupported Cosmos3 blur preset: {hint.preset_blur_strength!r}.")
    return config


def _import_cv2(hint_key: str):
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError(
            f"Cosmos3 transfer hint '{hint_key}' requires opencv-python for on-the-fly control generation. "
            "Install opencv-python in the environment, or provide a precomputed control_path for this hint."
        ) from exc


def pad_temporal_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Cosmos3 transfer frames must have shape [C, T, H, W], got {tuple(frames.shape)}.")
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError("Cosmos3 transfer target frame count must be positive.")
    if frames.shape[1] >= target_frames:
        return frames
    if frames.shape[1] == 0:
        raise ValueError("Cannot pad an empty Cosmos3 transfer frame tensor.")
    padded = frames
    while padded.shape[1] < target_frames:
        pad_len = min(padded.shape[1] - 1, target_frames - padded.shape[1])
        if pad_len <= 0:
            pad_frame = padded[:, -1:].repeat(1, target_frames - padded.shape[1], 1, 1)
            padded = torch.cat([padded, pad_frame], dim=1)
            break
        padded = torch.cat([padded, padded.flip(dims=[1])[:, :pad_len]], dim=1)
    return padded


def normalized_video_to_uint8_cthw(video: torch.Tensor) -> torch.Tensor:
    tensor = video.detach().cpu()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError("Cosmos3 transfer supports only batch size 1 video inputs.")
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(
            f"Cosmos3 transfer video must have shape [1, 3, T, H, W] or [3, T, H, W], got {tuple(video.shape)}."
        )
    if tensor.shape[0] != 3:
        if tensor.shape[-1] in (3, 4):
            tensor = tensor[..., :3].permute(3, 0, 1, 2)
        else:
            raise ValueError(f"Cosmos3 transfer video must have 3 RGB channels, got {tuple(video.shape)}.")
    if tensor.is_floating_point():
        if tensor.numel() and (tensor.min().item() < 0.0 or tensor.max().item() > 1.0):
            tensor = tensor.clamp(-1.0, 1.0).mul(0.5).add(0.5)
        tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    else:
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor.contiguous()


def uint8_cthw_to_normalized_5d(frames: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(f"Cosmos3 transfer frames must have shape [3, T, H, W], got {tuple(frames.shape)}.")
    return frames.to(dtype=dtype).div(127.5).sub(1.0).unsqueeze(0).contiguous()


def _pil_to_uint8_rgb(value: Any) -> np.ndarray:
    if isinstance(value, PIL.Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, str | Path):
        return np.asarray(PIL.Image.open(value).convert("RGB"), dtype=np.uint8)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
            tensor = tensor[:3].permute(1, 2, 0)
        if tensor.is_floating_point():
            if tensor.numel() and (tensor.min().item() < 0.0 or tensor.max().item() > 1.0):
                tensor = tensor.clamp(-1.0, 1.0).mul(0.5).add(0.5)
            tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        return tensor[..., :3].numpy().astype(np.uint8)
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            if array.size and (array.min() < 0.0 or array.max() > 1.0):
                array = np.clip(array, -1.0, 1.0) * 0.5 + 0.5
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return array[..., :3].astype(np.uint8)
    raise TypeError(f"Cosmos3 transfer expected an RGB frame, got {type(value)!r}.")


def media_hw(value: Any) -> tuple[int, int] | None:
    if isinstance(value, PIL.Image.Image):
        return value.height, value.width
    if isinstance(value, str | Path):
        media_path = Path(value)
        if media_path.suffix.lower() in IMAGE_EXTENSIONS:
            with PIL.Image.open(media_path) as image:
                return image.height, image.width
        try:
            import imageio.v3 as iio
        except ImportError:
            return None
        for frame in iio.imiter(media_path):
            array = np.asarray(frame)
            if array.ndim >= 2:
                return int(array.shape[0]), int(array.shape[1])
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4:
            if tensor.shape[0] in (3, 4) and tensor.shape[-1] not in (3, 4):
                return int(tensor.shape[-2]), int(tensor.shape[-1])
            if tensor.shape[-1] in (3, 4):
                return int(tensor.shape[1]), int(tensor.shape[2])
        if tensor.ndim == 3:
            if tensor.shape[0] in (3, 4) and tensor.shape[-1] not in (3, 4):
                return int(tensor.shape[-2]), int(tensor.shape[-1])
            if tensor.shape[-1] in (3, 4):
                return int(tensor.shape[0]), int(tensor.shape[1])
        return None
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4:
            if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                return int(array.shape[-2]), int(array.shape[-1])
            if array.shape[-1] in (3, 4):
                return int(array.shape[1]), int(array.shape[2])
        if array.ndim == 3:
            if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                return int(array.shape[-2]), int(array.shape[-1])
            if array.shape[-1] in (3, 4):
                return int(array.shape[0]), int(array.shape[1])
        return None
    if isinstance(value, list | tuple) and value:
        return media_hw(value[0])
    return None


def resize_center_crop_uint8_cthw(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(f"Cosmos3 transfer frames must have shape [3, T, H, W], got {tuple(frames.shape)}.")
    orig_h, orig_w = int(frames.shape[2]), int(frames.shape[3])
    scale = max(width / orig_w, height / orig_h)
    resize_h = int(np.ceil(scale * orig_h))
    resize_w = int(np.ceil(scale * orig_w))
    frames_tchw = frames.permute(1, 0, 2, 3).to(dtype=torch.float32)
    resized = F.interpolate(frames_tchw, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
    top = (resize_h - height) // 2
    left = (resize_w - width) // 2
    cropped = resized[:, :, top : top + height, left : left + width]
    return cropped.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def _path_media_to_uint8_cthw(path: str | Path, max_frames: int | None) -> torch.Tensor:
    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(f"Missing Cosmos3 transfer control_path: {media_path}")
    if media_path.suffix.lower() in IMAGE_EXTENSIONS:
        array = _pil_to_uint8_rgb(media_path)
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(1).contiguous()

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Cosmos3 transfer video control_path loading requires imageio. "
            "Install imageio[ffmpeg] or provide decoded control frames."
        ) from exc

    frames: list[torch.Tensor] = []
    limit = max_frames if max_frames is not None else None
    for frame in iio.imiter(media_path):
        frames.append(torch.from_numpy(_pil_to_uint8_rgb(frame)).permute(2, 0, 1))
        if limit is not None and len(frames) >= int(limit):
            break
    if not frames:
        raise ValueError(f"Cosmos3 transfer control_path produced no frames: {media_path}")
    return torch.stack(frames, dim=1).contiguous()


def media_to_uint8_cthw(value: Any, *, height: int, width: int, max_frames: int | None = None) -> torch.Tensor:
    if isinstance(value, str | Path):
        frames = _path_media_to_uint8_cthw(value, max_frames=max_frames)
    elif isinstance(value, torch.Tensor):
        frames = normalized_video_to_uint8_cthw(value)
    elif isinstance(value, np.ndarray):
        frames = normalized_video_to_uint8_cthw(torch.from_numpy(value))
    elif isinstance(value, list | tuple):
        if not value:
            raise ValueError("Cosmos3 transfer control frames cannot be empty.")
        selected = value[:max_frames] if max_frames is not None else value
        tensors = [torch.from_numpy(_pil_to_uint8_rgb(frame)).permute(2, 0, 1) for frame in selected]
        frames = torch.stack(tensors, dim=1).contiguous()
    else:
        raise TypeError(f"Unsupported Cosmos3 transfer control payload type: {type(value)!r}.")
    if max_frames is not None:
        frames = frames[:, : int(max_frames)]
    return resize_center_crop_uint8_cthw(frames, int(height), int(width))


def make_edge_control(frames: torch.Tensor, preset: str) -> torch.Tensor:
    cv2 = _import_cv2("edge")
    try:
        lower, upper = EDGE_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported Cosmos3 edge preset: {preset!r}.") from exc
    frames_np = frames.detach().cpu().numpy().astype(np.uint8)
    edge_maps = []
    for idx in range(frames_np.shape[1]):
        frame = np.ascontiguousarray(np.transpose(frames_np[:, idx], (1, 2, 0)))
        edge_maps.append(cv2.Canny(frame, lower, upper))
    edge = np.stack(edge_maps, axis=0)[None]
    return torch.from_numpy(edge).expand(3, -1, -1, -1).contiguous()


def _scale_for_bilateral_resolution(value: float, longest_side: int) -> float:
    if longest_side <= 0:
        return value
    return value * (longest_side / BILATERAL_REFERENCE_RESOLUTION)


def _scaled_bilateral_params(height: int, width: int) -> tuple[int, float, float]:
    longest_side = int(max(height, width))
    diameter = max(1, int(round(_scale_for_bilateral_resolution(float(BILATERAL_D), longest_side))))
    if diameter % 2 == 0:
        diameter += 1
    sigma_color = max(1.0, _scale_for_bilateral_resolution(float(BILATERAL_SIGMA_COLOR), longest_side))
    sigma_space = max(1.0, _scale_for_bilateral_resolution(float(BILATERAL_SIGMA_SPACE), longest_side))
    return diameter, sigma_color, sigma_space


def make_blur_control(frames: torch.Tensor, preset: str) -> torch.Tensor:
    cv2 = _import_cv2("blur")
    preset = preset.lower()
    if preset not in BLUR_DOWNUP_PRESETS:
        raise ValueError(f"Unsupported Cosmos3 blur preset: {preset!r}.")
    if preset == "none":
        return frames.clone()

    frames_np = frames.detach().cpu().numpy().astype(np.uint8)
    _, t, h, w = frames_np.shape
    pre_blur_factor = max(1, int(BLUR_PRE_BLUR_DOWNSCALE_PRESETS[preset]))
    downup_factor = max(1, int(BLUR_DOWNUP_PRESETS[preset]))
    result = np.empty_like(frames_np)
    for idx in range(t):
        frame = np.ascontiguousarray(np.transpose(frames_np[:, idx], (1, 2, 0)))
        if pre_blur_factor > 1:
            small_w = max(1, w // pre_blur_factor)
            small_h = max(1, h // pre_blur_factor)
            frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        diameter, sigma_color, sigma_space = _scaled_bilateral_params(frame.shape[0], frame.shape[1])
        for _ in range(BILATERAL_ITERATIONS):
            frame = cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)
        if pre_blur_factor > 1:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        if downup_factor > 1:
            small_w = max(1, w // downup_factor)
            small_h = max(1, h // downup_factor)
            frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        result[:, idx] = np.transpose(frame, (2, 0, 1))
    return torch.from_numpy(result).contiguous()


def load_or_compute_control_frames(
    hint: Cosmos3TransferHint,
    *,
    height: int,
    width: int,
    max_frames: int,
    input_frames: torch.Tensor | None,
) -> torch.Tensor:
    if hint.control is not None:
        return media_to_uint8_cthw(hint.control, height=height, width=width, max_frames=max_frames)
    if hint.control_path is not None:
        return media_to_uint8_cthw(hint.control_path, height=height, width=width, max_frames=max_frames)
    if hint.key == "edge":
        if input_frames is None:
            raise ValueError(
                "Cosmos3 transfer hint 'edge' requires either a video input for on-the-fly control generation "
                "or a precomputed control_path."
            )
        return make_edge_control(input_frames[:, :max_frames], hint.preset_edge_threshold)
    if hint.key == "blur":
        if input_frames is None:
            raise ValueError(
                "Cosmos3 transfer hint 'blur' requires either a video input for on-the-fly control generation "
                "or a precomputed control_path."
            )
        return make_blur_control(input_frames[:, :max_frames], hint.preset_blur_strength)
    raise FileNotFoundError(
        f"Cosmos3 transfer hint '{hint.key}' requires a precomputed control_path; "
        "on-the-fly generation is supported only for edge and blur."
    )
