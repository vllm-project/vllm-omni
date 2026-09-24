# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image
from vllm.logger import init_logger

logger = init_logger(__name__)

LINGBOT_RUNTIME_PROMPT_FIELDS = frozenset(
    {
        "duration",
        "seconds",
        "fps",
        "height",
        "width",
        "size",
        "num_frames",
        "resolution",
        "ratio",
    }
)

# Values are stored as (height, width), matching the official LingBot runner.
LINGBOT_RESOLUTION_PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "192p": {
        "1:1": (192, 192),
        "9:16": (192, 320),
        "16:9": (320, 192),
        "3:4": (192, 256),
        "4:3": (256, 192),
    },
    "480p": {
        "1:1": (480, 480),
        "9:16": (480, 832),
        "16:9": (832, 480),
        "3:4": (480, 640),
        "4:3": (640, 480),
    },
    "720p": {
        "1:1": (736, 736),
        "9:16": (736, 1280),
        "16:9": (1280, 736),
        "3:4": (736, 960),
        "4:3": (960, 736),
    },
    "1080p": {
        "1:1": (1088, 1088),
        "9:16": (1088, 1920),
        "16:9": (1920, 1088),
        "3:4": (1088, 1440),
        "4:3": (1440, 1088),
    },
    "2k": {
        "1:1": (1440, 1440),
        "9:16": (1440, 2560),
        "16:9": (2560, 1440),
        "3:4": (1440, 1920),
        "4:3": (1920, 1440),
    },
    "4k": {
        "1:1": (2176, 2176),
        "9:16": (2176, 3840),
        "16:9": (3840, 2176),
        "3:4": (2176, 2880),
        "4:3": (2880, 2176),
    },
}


class LingBotGenerationMode(str, Enum):
    T2I = "t2i"
    T2V = "t2v"
    TI2V = "ti2v"


@dataclass(frozen=True)
class LingBotRequestConfig:
    mode: LingBotGenerationMode
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float
    shift: float
    output_type: str
    input_image: Image.Image | None


def _json_caption(value: Any) -> str:
    try:
        caption = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LingBot structured prompt is not JSON serializable: {value!r}.") from exc
    if caption in {"", '""', "{}"}:
        raise ValueError("LingBot structured prompt must contain a non-empty caption.")
    return caption


def caption_from_lingbot_prompt(value: Any) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("LingBot `prompt` must not be empty.")
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"LingBot `prompt` must be a string or mapping, got {type(value)!r}.")

    if "caption" in value:
        caption = value["caption"]
        if isinstance(caption, str):
            if not caption:
                raise ValueError("LingBot `caption` must not be empty.")
            return caption
        return _json_caption(caption)

    caption_fields = {key: item for key, item in value.items() if key not in LINGBOT_RUNTIME_PROMPT_FIELDS}
    if not caption_fields:
        raise ValueError("LingBot structured prompt is empty after removing runtime fields.")
    return _json_caption(caption_fields)


def _prompt_envelope(prompt_obj: Any) -> Mapping[str, Any]:
    return prompt_obj if isinstance(prompt_obj, Mapping) else {}


def _prompt_value(prompt_obj: Any) -> Any:
    envelope = _prompt_envelope(prompt_obj)
    return envelope["prompt"] if "prompt" in envelope else prompt_obj


def _image_items(envelope: Mapping[str, Any]) -> list[Any]:
    multi_modal_data = envelope.get("multi_modal_data") or {}
    if not isinstance(multi_modal_data, Mapping):
        raise TypeError("LingBot `multi_modal_data` must be a mapping.")
    images = multi_modal_data.get("image")
    if images is None:
        return []
    return list(images) if isinstance(images, (list, tuple)) else [images]


def resolve_lingbot_mode(prompt_obj: Any) -> LingBotGenerationMode:
    envelope = _prompt_envelope(prompt_obj)
    images = _image_items(envelope)
    multi_modal_data = envelope.get("multi_modal_data") or {}
    unsupported = [name for name in ("video", "audio") if multi_modal_data.get(name) is not None]
    if unsupported:
        raise ValueError(f"LingBot does not support reference modalities {unsupported!r}; only one image is allowed.")

    modalities = envelope.get("modalities")
    if modalities is None:
        if images:
            raise ValueError("LingBot requests with an input image must explicitly set `modalities=['video']`.")
        logger.warning_once(
            "LingBot request omitted `modalities`; falling back to legacy text-to-video behavior. "
            "Set `modalities=['video']` explicitly."
        )
        return LingBotGenerationMode.T2V
    if not isinstance(modalities, (list, tuple)) or len(modalities) != 1:
        raise ValueError(f"LingBot `modalities` must contain exactly one output modality, got {modalities!r}.")

    modality = modalities[0]
    if modality == "image":
        if images:
            raise ValueError(f"LingBot text-to-image does not accept reference images, got {len(images)}.")
        return LingBotGenerationMode.T2I
    if modality == "video":
        if len(images) == 0:
            return LingBotGenerationMode.T2V
        if len(images) == 1:
            return LingBotGenerationMode.TI2V
        raise ValueError(f"LingBot text-image-to-video accepts exactly one reference image, got {len(images)}.")
    raise ValueError(f"LingBot output modality must be 'image' or 'video', got {modality!r}.")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"LingBot `{name}` must be a positive integer, got {value!r}.")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LingBot `{name}` must be a positive integer, got {value!r}.") from exc
    if not math.isfinite(numeric) or numeric != parsed or parsed <= 0:
        raise ValueError(f"LingBot `{name}` must be a positive integer, got {value!r}.")
    return parsed


def normalize_lingbot_num_frames(value: Any) -> int:
    """Round a video length up to LingBot's causal VAE ``4n+1`` grid."""
    num_frames = _positive_int(value, "num_frames")
    latent_frames = math.ceil((num_frames - 1) / 4)
    return latent_frames * 4 + 1


def resolve_lingbot_num_frames(duration: Any, fps: Any) -> int:
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LingBot `duration` must be a positive number, got {duration!r}.") from exc
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(f"LingBot `duration` must be a positive number, got {duration!r}.")
    fps_value = _positive_int(fps, "fps")
    raw_frames = int(duration_seconds * fps_value)
    if raw_frames < 1:
        raise ValueError(
            f"LingBot `duration` and `fps` must produce at least one frame, got duration={duration!r}, fps={fps!r}."
        )
    return normalize_lingbot_num_frames(raw_frames)


def _parse_size(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"LingBot `size` must use WIDTHxHEIGHT format, got {value!r}.")
        width, height = parts
        return _positive_int(height, "height"), _positive_int(width, "width")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
        return _positive_int(height, "height"), _positive_int(width, "width")
    raise ValueError(f"LingBot `size` must use WIDTHxHEIGHT format, got {value!r}.")


def resolve_lingbot_size(
    *,
    width: Any = None,
    height: Any = None,
    size: Any = None,
    resolution: Any = None,
    ratio: Any = None,
) -> tuple[int, int]:
    has_dimensions = width is not None or height is not None
    has_size = size is not None
    has_preset = resolution is not None or ratio is not None
    if sum((has_dimensions, has_size, has_preset)) > 1:
        raise ValueError("LingBot `size`, explicit `width`/`height`, and `resolution`/`ratio` are mutually exclusive.")

    if has_preset:
        if resolution is None or ratio is None:
            raise ValueError("LingBot `resolution` and `ratio` must be provided together.")
        resolution_value = str(resolution).lower()
        ratio_value = str(ratio)
        if resolution_value not in LINGBOT_RESOLUTION_PRESETS:
            raise ValueError(
                f"Unsupported LingBot `resolution` {resolution!r}; choices are {sorted(LINGBOT_RESOLUTION_PRESETS)!r}."
            )
        presets = LINGBOT_RESOLUTION_PRESETS[resolution_value]
        if ratio_value not in presets:
            raise ValueError(
                f"Unsupported LingBot `ratio` {ratio!r} for resolution {resolution_value!r}; "
                f"choices are {sorted(presets)!r}."
            )
        resolved_height, resolved_width = presets[ratio_value]
    elif has_size:
        resolved_height, resolved_width = _parse_size(size)
    else:
        if width is None or height is None:
            raise ValueError(
                f"LingBot `width` and `height` must be provided together, got width={width!r}, height={height!r}."
            )
        resolved_height = _positive_int(height, "height")
        resolved_width = _positive_int(width, "width")

    if resolved_height % 16 != 0 or resolved_width % 16 != 0:
        raise ValueError(
            "LingBot `height` and `width` must be multiples of 16, "
            f"got height={resolved_height}, width={resolved_width}."
        )
    return resolved_height, resolved_width


def _runtime_prompt_fields(prompt_obj: Any) -> Mapping[str, Any]:
    prompt_value = _prompt_value(prompt_obj)
    return prompt_value if isinstance(prompt_value, Mapping) else {}


def _pick(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    return value if value is not None else None


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def resolve_lingbot_output_dimensions(
    *,
    sampling_width: Any = None,
    sampling_height: Any = None,
    prompt_fields: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    default_width: int = 480,
    default_height: int = 480,
) -> tuple[int, int]:
    """Resolve the ``(width, height)`` that LingBot will use for generation."""
    prompt_fields = prompt_fields or {}
    extra_fields = extra_fields or {}
    # Online requests deliver `extra_params` through sampling.extra_args; the
    # offline path embeds the same fields in the prompt dict. Consult both,
    # extra_args first — the same precedence fps and num_frames use.
    requested_width = _first_not_none(_pick(extra_fields, "width"), _pick(prompt_fields, "width"))
    requested_height = _first_not_none(_pick(extra_fields, "height"), _pick(prompt_fields, "height"))
    size = _first_not_none(_pick(extra_fields, "size"), _pick(prompt_fields, "size"))
    resolution = _first_not_none(_pick(extra_fields, "resolution"), _pick(prompt_fields, "resolution"))
    ratio = _first_not_none(_pick(extra_fields, "ratio"), _pick(prompt_fields, "ratio"))
    if requested_width is not None or requested_height is not None:
        width, height = requested_width, requested_height
    elif size is not None or resolution is not None or ratio is not None:
        width = height = None
    else:
        width, height = sampling_width, sampling_height
    if width is None and height is None and size is None and resolution is None and ratio is None:
        width, height = default_width, default_height
    resolved_height, resolved_width = resolve_lingbot_size(
        width=width,
        height=height,
        size=size,
        resolution=resolution,
        ratio=ratio,
    )
    return resolved_width, resolved_height


def normalize_lingbot_request(
    request: Any,
    *,
    default_negative_prompt: str,
    default_image_negative_prompt: str,
    default_height: int = 480,
    default_width: int = 480,
    default_num_frames: int = 81,
    default_fps: int = 24,
    default_num_inference_steps: int = 40,
    default_guidance_scale: float = 6.0,
    default_shift: float = 3.0,
    default_output_type: str = "pt",
) -> LingBotRequestConfig:
    prompt_obj = request.prompt
    envelope = _prompt_envelope(prompt_obj)
    prompt_fields = _runtime_prompt_fields(prompt_obj)
    sampling = request.sampling_params
    extra_args = dict(sampling.extra_args or {})

    mode = resolve_lingbot_mode(prompt_obj)
    prompt = caption_from_lingbot_prompt(_prompt_value(prompt_obj))
    images = _image_items(envelope)
    input_image = images[0] if images else None
    if input_image is not None and not isinstance(input_image, Image.Image):
        raise TypeError(f"LingBot input image must be a PIL image after preprocessing, got {type(input_image)!r}.")

    prompt_negative = envelope.get("negative_prompt")
    if "negative_prompt" in extra_args:
        negative_prompt = extra_args["negative_prompt"]
    elif prompt_negative is not None:
        negative_prompt = prompt_negative
    else:
        negative_prompt = (
            default_image_negative_prompt if mode is LingBotGenerationMode.T2I else default_negative_prompt
        )
    if not isinstance(negative_prompt, str):
        raise TypeError(f"LingBot `negative_prompt` must be a string, got {type(negative_prompt)!r}.")

    fps_raw = _first_not_none(
        _pick(extra_args, "fps"),
        _pick(prompt_fields, "fps"),
        sampling.resolved_frame_rate,
        default_fps,
    )
    fps = _positive_int(fps_raw, "fps")

    extra_seconds = _pick(extra_args, "seconds")
    prompt_seconds = _pick(prompt_fields, "seconds")
    duration_value = _pick(extra_args, "duration")
    prompt_duration = _pick(prompt_fields, "duration")
    if any(value is not None for value in (extra_seconds, prompt_seconds)) and any(
        value is not None for value in (duration_value, prompt_duration)
    ):
        raise ValueError("LingBot `seconds` and `duration` cannot be provided together.")
    duration = next(
        (value for value in (extra_seconds, prompt_seconds, duration_value, prompt_duration) if value is not None),
        None,
    )

    model_num_frames = _first_not_none(
        _pick(extra_args, "num_frames"),
        _pick(prompt_fields, "num_frames"),
    )
    explicit_num_frames = model_num_frames
    if explicit_num_frames is None and sampling.num_frames not in (None, 1):
        # ``sampling.num_frames`` defaults to 1 (the image-model sentinel).
        # Treat any other value as an explicit request; ``1`` means "omitted".
        explicit_num_frames = sampling.num_frames

    if mode is LingBotGenerationMode.T2I:
        if duration is not None:
            raise ValueError(f"LingBot text-to-image does not accept `seconds` or `duration`, got {duration!r}.")
        if explicit_num_frames is not None and _positive_int(explicit_num_frames, "num_frames") != 1:
            raise ValueError(f"LingBot text-to-image requires `num_frames=1`, got {explicit_num_frames!r}.")
        num_frames = 1
    elif explicit_num_frames is not None:
        num_frames = normalize_lingbot_num_frames(explicit_num_frames)
    elif duration is not None:
        num_frames = resolve_lingbot_num_frames(duration, fps)
    else:
        num_frames = normalize_lingbot_num_frames(default_num_frames)

    width, height = resolve_lingbot_output_dimensions(
        sampling_width=sampling.width,
        sampling_height=sampling.height,
        prompt_fields=prompt_fields,
        extra_fields=extra_args,
        default_width=default_width,
        default_height=default_height,
    )
    num_inference_steps = sampling.num_inference_steps
    if num_inference_steps is None:
        num_inference_steps = _first_not_none(_pick(extra_args, "num_inference_steps"), default_num_inference_steps)
    num_inference_steps = _positive_int(num_inference_steps, "num_inference_steps")
    guidance_scale = (
        sampling.guidance_scale
        if sampling.guidance_scale_provided
        else _first_not_none(_pick(extra_args, "guidance_scale"), default_guidance_scale)
    )
    try:
        guidance_scale = float(guidance_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LingBot `guidance_scale` must be numeric, got {guidance_scale!r}.") from exc
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError(f"LingBot `guidance_scale` must be non-negative, got {guidance_scale!r}.")

    shift_raw = (
        extra_args["shift"]
        if _pick(extra_args, "shift") is not None
        else _first_not_none(_pick(extra_args, "flow_shift"), default_shift)
    )
    try:
        shift = float(shift_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LingBot `shift` must be numeric, got {shift_raw!r}.") from exc
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"LingBot `shift` must be positive, got {shift_raw!r}.")

    output_type = _first_not_none(
        sampling.output_type,
        _pick(extra_args, "output_type"),
        default_output_type,
    )
    if output_type not in {"pt", "np", "latent"}:
        raise ValueError(f"LingBot `output_type` must be 'pt', 'np', or 'latent', got {output_type!r}.")

    return LingBotRequestConfig(
        mode=mode,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        output_type=output_type,
        input_image=input_image,
    )
