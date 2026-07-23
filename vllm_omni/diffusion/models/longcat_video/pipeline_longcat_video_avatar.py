# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from PIL import Image
from torch import nn
from transformers import AutoFeatureExtractor, AutoTokenizer, UMT5EncoderModel, WhisperModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportImageInput
from vllm_omni.diffusion.models.longcat_video.longcat_video_avatar_transformer import (
    create_full_precision_avatar_dit,
    create_quantized_avatar_dit,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_DEFAULT_NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)
_DEFAULT_AT2V_RATIO = 480 / 832

# Whisper uses a 30-second context window: 30s * 16kHz raw samples,
# represented as 3000 mel frames after feature extraction.
_WHISPER_AUDIO_SAMPLE_RATE = 16000
_WHISPER_AUDIO_CHUNK_SAMPLES = 30 * _WHISPER_AUDIO_SAMPLE_RATE
_WHISPER_ENCODER_CHUNK_FRAMES = 3000

# Mirrors official LongCat-Video `longcat_video/utils/bukcet_config.py`
# for scale_factor_spatial in {16, 32}. Used for Avatar condition sizing
# and AT2V defaults without importing the official repository at runtime.
_LONGCAT_480P_BUCKETS = {
    "0.26": ([320, 1216], 1),
    "0.31": ([352, 1120], 1),
    "0.38": ([384, 1024], 1),
    "0.43": ([416, 960], 1),
    "0.52": ([448, 864], 1),
    "0.58": ([480, 832], 1),
    "0.67": ([512, 768], 1),
    "0.74": ([544, 736], 1),
    "0.86": ([576, 672], 1),
    "0.95": ([608, 640], 1),
    "1.05": ([640, 608], 1),
    "1.17": ([672, 576], 1),
    "1.29": ([704, 544], 1),
    "1.35": ([736, 544], 1),
    "1.50": ([768, 512], 1),
    "1.67": ([800, 480], 1),
    "1.73": ([832, 480], 1),
    "2.00": ([896, 448], 1),
    "2.31": ([960, 416], 1),
    "2.58": ([992, 384], 1),
    "2.75": ([1056, 384], 1),
    "3.09": ([1088, 352], 1),
    "3.70": ([1184, 320], 1),
    "3.80": ([1216, 320], 1),
    "3.90": ([1248, 320], 1),
    "4.00": ([1280, 320], 1),
}

_LONGCAT_720P_BUCKETS = {
    "0.25": ([480, 1920], 1),
    "0.29": ([512, 1792], 1),
    "0.32": ([544, 1696], 1),
    "0.36": ([576, 1600], 1),
    "0.40": ([608, 1504], 1),
    "0.49": ([672, 1376], 1),
    "0.54": ([704, 1312], 1),
    "0.59": ([736, 1248], 1),
    "0.69": ([800, 1152], 1),
    "0.74": ([832, 1120], 1),
    "0.82": ([864, 1056], 1),
    "0.88": ([896, 1024], 1),
    "0.94": ([928, 992], 1),
    "1.00": ([960, 960], 1),
    "1.07": ([992, 928], 1),
    "1.14": ([1024, 896], 1),
    "1.22": ([1056, 864], 1),
    "1.31": ([1088, 832], 1),
    "1.35": ([1120, 832], 1),
    "1.44": ([1152, 800], 1),
    "1.70": ([1248, 736], 1),
    "2.00": ([1344, 672], 1),
    "2.05": ([1376, 672], 1),
    "2.47": ([1504, 608], 1),
    "2.53": ([1536, 608], 1),
    "2.83": ([1632, 576], 1),
    "3.06": ([1664, 544], 1),
    "3.12": ([1696, 544], 1),
    "3.62": ([1856, 512], 1),
    "3.93": ([1888, 480], 1),
    "4.00": ([1920, 480], 1),
}


def _get_longcat_bucket_config(resolution: str) -> dict[str, tuple[list[int], int]]:
    if resolution == "480p":
        return _LONGCAT_480P_BUCKETS
    if resolution == "720p":
        return _LONGCAT_720P_BUCKETS
    raise ValueError(f"Unsupported LongCat-Video-Avatar resolution {resolution!r}. Expected '480p' or '720p'.")


def _default_at2v_shape(resolution: str) -> tuple[int, int]:
    bucket_config = _get_longcat_bucket_config(resolution)
    closest_bucket = sorted(bucket_config.keys(), key=lambda key: abs(float(key) - _DEFAULT_AT2V_RATIO))[0]
    target_h, target_w = bucket_config[closest_bucket][0]
    return target_h, target_w


def _avatar_model_allow_patterns(use_int8: bool) -> list[str]:
    weight_subfolder = "base_model_int8" if use_int8 else "base_model"
    return [
        "scheduler/*",
        f"{weight_subfolder}/*",
        "lora/dmd_lora.safetensors",
        "whisper-large-v3/config.json",
        "whisper-large-v3/generation_config.json",
        "whisper-large-v3/model.safetensors",
        "whisper-large-v3/preprocessor_config.json",
        "vocal_separator/*",
        "config.json",
        "model_index.json",
    ]


def _adjust_num_frames(num_frames: int, temporal_scale: int = 4) -> int:
    if num_frames % temporal_scale != 1:
        num_frames = num_frames // temporal_scale * temporal_scale + 1
    return max(num_frames, 1)


def _resolve_num_segments(
    value: Any,
    *,
    audio_duration: float,
    num_frames: int,
    num_cond_frames: int,
    save_fps: int,
) -> int:
    if value is None:
        return 1
    if isinstance(value, str) and value.lower() == "auto":
        first_segment_duration = num_frames / save_fps
        stride_duration = (num_frames - num_cond_frames) / save_fps
        if audio_duration <= first_segment_duration or stride_duration <= 0:
            return 1
        return 1 + math.ceil((audio_duration - first_segment_duration) / stride_duration)
    return max(1, int(value))


def _requested_num_segments(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, str) and value.lower() == "auto":
        return 0
    return max(1, int(value))


def _requested_audio_duration(
    *,
    auto_num_segments: bool,
    num_segments: int,
    num_frames: int,
    num_cond_frames: int,
    save_fps: int,
) -> float:
    if auto_num_segments:
        return 0.0
    return num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_asset_root_from_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve() if path.exists() else path
    parts = resolved.parts
    if "assets" not in parts:
        return resolved.parent
    asset_idx = parts.index("assets")
    if asset_idx == 0:
        return None
    return Path(*parts[:asset_idx])


def _resolve_path_from_roots(
    path: str | os.PathLike[str] | None,
    *,
    asset_roots: tuple[Path, ...],
) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return str(resolved)
    for asset_root in asset_roots:
        candidate = asset_root / resolved
        if candidate.exists():
            return str(candidate)
    return str(resolved)


def _speaker_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(name))
    return (int(match.group(1)) if match else 10_000, str(name))


def _normalize_audio_paths(
    audio_field: Any,
    *,
    asset_roots: tuple[Path, ...],
) -> dict[str, str]:
    if audio_field is None:
        return {}
    if isinstance(audio_field, dict):
        return {
            str(speaker): resolved
            for speaker, path in sorted(audio_field.items(), key=lambda item: _speaker_sort_key(str(item[0])))
            if (resolved := _resolve_path_from_roots(path, asset_roots=asset_roots)) is not None
        }
    if isinstance(audio_field, list | tuple):
        audio_paths: dict[str, str] = {}
        for idx, item in enumerate(audio_field, start=1):
            if isinstance(item, dict):
                audio_paths.update(_normalize_audio_paths(item, asset_roots=asset_roots))
                continue
            resolved = _resolve_path_from_roots(item, asset_roots=asset_roots)
            if resolved is not None:
                audio_paths[f"person{idx}"] = resolved
        return dict(sorted(audio_paths.items(), key=lambda item: _speaker_sort_key(item[0])))
    if isinstance(audio_field, str | os.PathLike):
        resolved = _resolve_path_from_roots(audio_field, asset_roots=asset_roots)
        return {"person1": resolved} if resolved is not None else {}
    raise TypeError(f"Unsupported LongCat-Video-Avatar audio input type {type(audio_field)}.")


def _prepare_multi_speaker_audio_arrays(
    speech_arrays: list[np.ndarray],
    *,
    audio_type: str,
    generate_duration: float,
    sample_rate: int,
) -> list[np.ndarray]:
    if not speech_arrays:
        raise ValueError("Multi-speaker Avatar requires at least one audio track.")

    arrays = [np.asarray(array, dtype=np.float32) for array in speech_arrays]
    min_samples = max(1, math.ceil(generate_duration * sample_rate))
    audio_type = audio_type.lower()

    if audio_type == "para":
        target_len = max(min_samples, *(len(array) for array in arrays))
        return [np.pad(array, (0, max(0, target_len - len(array)))) for array in arrays]

    if audio_type == "add":
        source_total = sum(len(array) for array in arrays)
        target_len = max(min_samples, source_total)
        outputs = []
        offset = 0
        for array in arrays:
            output = np.zeros(target_len, dtype=np.float32)
            output[offset : offset + len(array)] = array
            outputs.append(output)
            offset += len(array)
        return outputs

    raise NotImplementedError(f"Unsupported LongCat-Video-Avatar multi-speaker audio_type {audio_type!r}.")


def _bbox_to_mask(bbox: list[int] | tuple[int, int, int, int], height: int, width: int) -> torch.Tensor:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox [y_min, x_min, y_max, x_max], got {bbox!r}.")
    y_min, x_min, y_max, x_max = [int(value) for value in bbox]
    y_min = max(0, min(height, y_min))
    y_max = max(0, min(height, y_max))
    x_min = max(0, min(width, x_min))
    x_max = max(0, min(width, x_max))
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def _normalize_other_bboxes(raw_bboxes: Any) -> list[list[int]]:
    if raw_bboxes is None:
        return []
    if (
        isinstance(raw_bboxes, list | tuple)
        and raw_bboxes
        and all(isinstance(value, int | float) for value in raw_bboxes)
    ):
        if len(raw_bboxes) % 4 != 0:
            raise ValueError("bbox['others'] must contain a multiple of four values.")
        return [list(raw_bboxes[idx : idx + 4]) for idx in range(0, len(raw_bboxes), 4)]
    if isinstance(raw_bboxes, list | tuple):
        return [list(bbox) for bbox in raw_bboxes]
    raise TypeError("bbox['others'] must be a flat bbox list or a list of bboxes.")


def _build_multi_speaker_ref_target_masks(
    image: Image.Image,
    bbox: dict[str, Any] | None,
) -> tuple[torch.Tensor, bool]:
    src_width, src_height = image.size
    bbox = bbox or {}
    person1_bbox = bbox.get("person1")
    person2_bbox = bbox.get("person2")

    if person1_bbox is None and person2_bbox is None:
        face_scale = 0.1
        split_width = src_width // 2
        y_min = int(src_height * face_scale)
        y_max = int(src_height * (1 - face_scale))
        person1_bbox = [
            y_min,
            int(split_width * face_scale),
            y_max,
            int(split_width * (1 - face_scale)),
        ]
        person2_bbox = [
            y_min,
            int(split_width * face_scale + split_width),
            y_max,
            int(split_width * (1 - face_scale) + split_width),
        ]
    elif person1_bbox is None or person2_bbox is None:
        raise NotImplementedError("Multi-speaker Avatar requires both person1 and person2 bboxes when bbox is set.")

    human_mask1 = _bbox_to_mask(person1_bbox, src_height, src_width)
    human_mask2 = _bbox_to_mask(person2_bbox, src_height, src_width)
    background_mask = torch.where(human_mask1 + human_mask2 > 0, torch.zeros(()), torch.ones(()))
    total_masks = [human_mask1, human_mask2, background_mask]

    other_bboxes = _normalize_other_bboxes(bbox.get("others"))
    for other_bbox in other_bboxes:
        total_masks.append(_bbox_to_mask(other_bbox, src_height, src_width))

    return torch.stack(total_masks, dim=0), bool(other_bboxes)


def _ensure_local_dir(model: str | os.PathLike[str], allow_patterns: list[str] | None = None) -> Path:
    model_path = Path(model)
    if model_path.exists():
        return model_path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model), allow_patterns=allow_patterns))


def _ensure_avatar_model_index(model_dir: Path, use_int8: bool) -> None:
    model_index_path = model_dir / "model_index.json"
    model_index = {}
    if model_index_path.exists():
        with model_index_path.open("r", encoding="utf-8") as f:
            model_index = json.load(f)
    model_index.update(
        {
            "_class_name": "LongCatVideoAvatarPipeline",
            "_diffusers_version": "0.38.0",
            "model_name": "LongCat-Video-Avatar-1.5",
        }
    )
    model_index_path.write_text(json.dumps(model_index, indent=2, sort_keys=True), encoding="utf-8")

    source_subfolder = "base_model_int8" if use_int8 else "base_model"
    source_config = model_dir / source_subfolder / "config.json"
    target_config = model_dir / "transformer" / "config.json"
    if source_config.exists():
        target_config.parent.mkdir(parents=True, exist_ok=True)
        target_config.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")


def prepare_longcat_video_avatar_model_for_omni(model: str, use_int8: bool = True) -> str:
    model_dir = _ensure_local_dir(model, allow_patterns=_avatar_model_allow_patterns(use_int8))
    _ensure_avatar_model_index(model_dir, use_int8)
    return str(model_dir)


def _load_image(raw_image: Any) -> Image.Image:
    if isinstance(raw_image, list):
        if not raw_image:
            raise ValueError("LongCat-Video-Avatar received an empty image list.")
        raw_image = raw_image[0]
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")
    if isinstance(raw_image, np.ndarray):
        return Image.fromarray(raw_image).convert("RGB")
    if isinstance(raw_image, torch.Tensor):
        tensor = raw_image.detach().cpu()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, 0.0, 1.0) * 255).round().astype("uint8")
        return Image.fromarray(array).convert("RGB")
    if isinstance(raw_image, str | os.PathLike):
        return Image.open(raw_image).convert("RGB")
    raise TypeError(
        f"Unsupported LongCat-Video-Avatar image input type {type(raw_image)}. "
        "Pass a PIL image, numpy array, torch tensor, or file path."
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _video_tensor_to_pil_frames(video: Any) -> list[Image.Image]:
    if isinstance(video, torch.Tensor):
        array = video.detach().cpu().numpy()
    else:
        array = np.asarray(video)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[1] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.ndim != 4:
        raise ValueError(f"Expected video frames with shape [T,H,W,C], got {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255).round().astype("uint8")
    else:
        array = np.clip(array, 0, 255).astype("uint8")
    return [Image.fromarray(frame).convert("RGB") for frame in array]


def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def get_longcat_video_avatar_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(video: Any, sampling_params=None):
        fps = 25
        if sampling_params is not None:
            fps = int(
                sampling_params.extra_args.get("save_fps")
                or sampling_params.extra_args.get("fps")
                or sampling_params.fps
                or 25
            )
        return {
            "video": [_video_tensor_to_pil_frames(video)],
            "custom_output": {"fps": fps},
            "fps": fps,
        }

    return post_process_func


def get_longcat_video_avatar_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        prompt.setdefault("multi_modal_data", {})
        prompt.setdefault("additional_information", {})
        request.prompt = prompt
        return request

    return pre_process_func


class LongCatVideoAvatarPipeline(nn.Module, SupportImageInput, SupportAudioInput):
    """Native LongCat-Video-Avatar A2V/AI2V pipeline."""

    support_image_input: ClassVar[bool] = True
    support_audio_input: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 1

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        additional_config = getattr(od_config, "additional_config", {}) or {}
        self.model_type = str(additional_config.get("model_type") or "avatar-v1.5")
        if self.model_type != "avatar-v1.5":
            raise NotImplementedError("LongCat-Video-Avatar native MVP only supports avatar-v1.5.")
        self.use_distill = _as_bool(additional_config.get("use_distill"), True)
        self.use_int8 = _as_bool(additional_config.get("use_int8"), True)
        self.build_components_on_gpu = _as_bool(additional_config.get("build_components_on_gpu"), False)
        self.resolution = str(additional_config.get("resolution") or "480p")
        self.model_dir = _ensure_local_dir(
            od_config.model,
            allow_patterns=_avatar_model_allow_patterns(self.use_int8),
        )
        self.save_fps = 25
        self.audio_stride = 1
        self.default_num_frames = 93
        self.video_processor = VideoProcessor(vae_scale_factor=8)
        dit_subfolder = "base_model_int8" if self.use_int8 else "base_model"
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(self.model_dir),
                subfolder=dit_subfolder,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]

        self._load_components()

    def _base_model_dir(self) -> Path:
        configured = (getattr(self.od_config, "additional_config", {}) or {}).get("base_model_dir")
        if configured:
            return _ensure_local_dir(
                configured,
                allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*", "config.json", "model_index.json"],
            )
        sibling = self.model_dir.parent / "LongCat-Video"
        if sibling.exists():
            return sibling
        return _ensure_local_dir(
            "meituan-longcat/LongCat-Video",
            allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*", "config.json", "model_index.json"],
        )

    def _load_components(self) -> None:
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise ImportError(
                "LongCat-Video-Avatar requires audio-separator. "
                "Install the optional extra with `pip install 'vllm-omni[longcat-video-avatar]'`."
            ) from exc

        dtype = getattr(self.od_config, "dtype", torch.bfloat16)
        base_dir = self._base_model_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.model_dir), subfolder="scheduler")

        # vLLM initializes native pipelines under the target device context.
        # Build LongCat's large components on CPU first by default to avoid
        # transient full-precision CUDA allocations before int8 DiT replacement.
        build_context = nullcontext() if self.build_components_on_gpu else torch.device("cpu")
        enable_distill_lora = False
        with build_context:
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                str(base_dir), subfolder="text_encoder", torch_dtype=dtype
            )
            self.vae = DistributedAutoencoderKLWan.from_pretrained(str(base_dir), subfolder="vae", torch_dtype=dtype)

            if self.use_int8:
                self.transformer = create_quantized_avatar_dit(
                    str(self.model_dir),
                    subfolder="base_model_int8",
                    cp_split_hw=[1, 1],
                )
            else:
                self.transformer = create_full_precision_avatar_dit(
                    str(self.model_dir),
                    subfolder="base_model",
                    cp_split_hw=[1, 1],
                )
                self.transformer = self.transformer.to(dtype=dtype)

            if self.use_distill:
                lora_path = self.model_dir / "lora" / "dmd_lora.safetensors"
                if lora_path.exists():
                    self.transformer.load_lora(
                        str(lora_path),
                        "dmd",
                        multiplier=1.0,
                        lora_network_dim=128,
                        lora_network_alpha=64,
                    )
                    enable_distill_lora = True

            audio_model_path = self.model_dir / "whisper-large-v3"
            self.audio_encoder = WhisperModel.from_pretrained(str(audio_model_path)).eval()

        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.transformer = self.transformer.to(device=self.device)
        if enable_distill_lora:
            self.transformer.enable_loras(["dmd"])
        self.audio_encoder = self.audio_encoder.to(self.device)
        self.audio_encoder.requires_grad_(False)
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(str(audio_model_path))

        separator_path = self.model_dir / "vocal_separator" / "Kim_Vocal_2.onnx"
        self._audio_temp_dir = tempfile.TemporaryDirectory(prefix="longcat_avatar_audio_")
        self.audio_temp_dir = Path(self._audio_temp_dir.name)
        self.vocal_separator = Separator(
            output_dir=self.audio_temp_dir / "vocals",
            output_single_stem="vocals",
            model_file_dir=str(separator_path.parent),
        )
        self.vocal_separator.load_model(separator_path.name)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # .to() is overloaded (device / dtype / tensor); only treat args[0] as
        # a device when it actually is one, otherwise keep self.device.
        candidate = kwargs.get("device", args[0] if args else None)
        device = torch.device(candidate) if isinstance(candidate, str | torch.device | int) else self.device
        self.device = device
        for name in ("text_encoder", "vae", "transformer", "audio_encoder"):
            module = getattr(self, name, None)
            if module is not None:
                module.to(device)
        return self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load DiT weights using vLLM's diffusion weight loader."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip()

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)
        mask = mask.repeat_interleave(num_videos_per_prompt, dim=0)
        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.transformer.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        if not do_classifier_free_guidance:
            return prompt_embeds, prompt_attention_mask, None, None

        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @property
    def do_classifier_free_guidance(self):
        return self._text_guidance_scale > 1.0 or self._audio_guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return 1000

    @property
    def num_distill_sample_steps(self):
        return 8

    def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool = False) -> torch.Tensor:
        if use_distill:
            distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
            distill_indices = self.num_timesteps - distill_indices
            sigmas = torch.flip(torch.linspace(0, 1, self.num_timesteps), [0])
            sigmas = torch.flip(sigmas[distill_indices], [0]).float()
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        return sigmas.to(torch.float32)

    def _update_kv_cache_dict(self, kv_cache_dict):
        self.kv_cache_dict = kv_cache_dict

    def _cache_clean_latents(
        self,
        cond_latents: torch.Tensor,
        max_sequence_length: int,
        *,
        offload_kv_cache: bool,
        dtype: torch.dtype,
        audio_embs: torch.Tensor,
        num_cond_latents: int,
        num_ref_latents: int,
        ref_img_index: int,
    ) -> None:
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2], device=self.device, dtype=dtype)
        empty_embeds = torch.zeros(
            [cond_latents.shape[0], 1, max_sequence_length, self.text_encoder.config.d_model],
            device=self.device,
            dtype=dtype,
        )
        _, kv_cache_dict = self.transformer(
            hidden_states=cond_latents,
            timestep=timestep,
            encoder_hidden_states=empty_embeds,
            num_cond_latents=num_cond_latents,
            return_kv=True,
            skip_crs_attn=True,
            offload_kv_cache=offload_kv_cache,
            audio_embs=audio_embs,
            num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index,
        )
        self._update_kv_cache_dict(kv_cache_dict)

    def _get_kv_cache_dict(self):
        return getattr(self, "kv_cache_dict", {})

    def _clear_cache(self) -> None:
        self.kv_cache_dict = {}
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

    def normalize_latents(self, latents):
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) * latents_std

    def denormalize_latents(self, latents):
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return latents / latents_std + latents_mean

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        num_cond_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        need_encode: bool = True,
    ) -> torch.Tensor:
        if image is not None and video is not None:
            raise ValueError("Cannot provide both image and video conditioning.")
        if latents is None:
            num_latent_frames = (num_frames - 1) // self.vae.config.scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae.config.scale_factor_spatial,
                int(width) // self.vae.config.scale_factor_spatial,
            )
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        if image is not None or video is not None:
            condition_data = image if image is not None else video
            cond_latents = []
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae.config.scale_factor_temporal
            if need_encode:
                for idx in range(batch_size):
                    gen = generator[idx] if isinstance(generator, list) else generator
                    if image is not None:
                        encoded_input = condition_data[idx].unsqueeze(0).unsqueeze(2)
                    else:
                        encoded_input = condition_data[idx][:, -num_cond_frames:].unsqueeze(0)
                    latent = _retrieve_latents(self.vae.encode(encoded_input), gen, sample_mode="argmax")
                    cond_latents.append(latent)
                cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
                cond_latents = self.normalize_latents(cond_latents)
            else:
                cond_latents = condition_data[:, :, -num_cond_latents:].to(device=device, dtype=dtype)
            latents[:, :, :num_cond_latents] = cond_latents
        return latents

    def _resolve_request_inputs(self, req: OmniDiffusionRequest) -> dict[str, Any]:
        first_prompt = req.prompts[0] if req.prompts else {"prompt": ""}
        if isinstance(first_prompt, str):
            prompt_text = first_prompt
            mm_data: dict[str, Any] = {}
            info: dict[str, Any] = {}
            negative_prompt = None
        else:
            prompt_text = first_prompt.get("prompt") or ""
            mm_data = first_prompt.get("multi_modal_data") or {}
            info = first_prompt.get("additional_information") or {}
            negative_prompt = first_prompt.get("negative_prompt")

        extra = req.sampling_params.extra_args or {}
        input_json = extra.get("input_json") or info.get("input_json")
        input_asset_root = None
        if input_json:
            input_json_path = Path(input_json).expanduser()
            sample = _load_json(input_json_path)
            input_asset_root = _infer_asset_root_from_path(input_json_path)
        else:
            sample = {}
        asset_roots = (input_asset_root,) if input_asset_root is not None else ()
        prompt = prompt_text or sample.get("prompt") or ""
        negative_prompt = negative_prompt or extra.get("negative_prompt") or _DEFAULT_NEGATIVE_PROMPT

        audio_field = mm_data.get("audio")
        if audio_field is None:
            audio_field = extra.get("audio_path") or info.get("audio_path") or sample.get("cond_audio")
        audio_paths = _normalize_audio_paths(audio_field, asset_roots=asset_roots)
        if not audio_paths:
            raise ValueError("LongCat-Video-Avatar requires multi_modal_data['audio'] or extra_args['audio_path'].")
        audio_path = audio_paths[sorted(audio_paths, key=_speaker_sort_key)[0]]

        image_input = mm_data.get("image")
        image_path = extra.get("image_path") or info.get("image_path") or sample.get("cond_image")
        if image_input is None and image_path is not None:
            image_input = _resolve_path_from_roots(image_path, asset_roots=asset_roots)

        inferred_stage = "ai2v" if image_input is not None else "at2v"
        stage = str(extra.get("stage") or extra.get("stage_1") or info.get("stage") or inferred_stage).lower()
        resolution = str(extra.get("resolution") or info.get("resolution") or self.resolution)
        bbox = extra.get("bbox") or info.get("bbox") or sample.get("bbox") or {}
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "audio_path": audio_path,
            "audio_paths": audio_paths,
            "audio_type": str(extra.get("audio_type") or info.get("audio_type") or sample.get("audio_type") or "para"),
            "bbox": bbox,
            "image": image_input,
            "is_multi_speaker": len(audio_paths) > 1,
            "stage": stage,
            "resolution": resolution,
        }

    def _extract_vocal(self, audio_path: str) -> str:
        fd, path = tempfile.mkstemp(prefix="longcat_avatar_vocal_", suffix=".wav")
        os.close(fd)
        target_path = Path(path)
        outputs = self.vocal_separator.separate(audio_path)
        if not outputs:
            target_path.unlink(missing_ok=True)
            raise RuntimeError("Audio separation produced no vocal stem.")
        default_vocal_path = (self.audio_temp_dir / "vocals" / outputs[0]).resolve()
        shutil.move(str(default_vocal_path), str(target_path))
        return str(target_path)

    def _loudness_norm(self, audio_array, sr=16000, lufs=-23, threshold=100):
        try:
            import pyloudnorm as pyln
        except ImportError as exc:
            raise ImportError(
                "LongCat-Video-Avatar requires pyloudnorm. "
                "Install the optional extra with `pip install 'vllm-omni[longcat-video-avatar]'`."
            ) from exc

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > threshold:
            return audio_array
        return pyln.normalize.loudness(audio_array, loudness, lufs)

    @torch.no_grad()
    def _get_audio_embedding_whisper(self, speech_array, fps=25, sample_rate=16000):
        enc_fps = 50
        audio_duration = len(speech_array) / sample_rate
        video_length = int(audio_duration * fps)
        speech_array = self._loudness_norm(speech_array, sample_rate)

        mel_chunks = []
        for idx in range(0, len(speech_array), _WHISPER_AUDIO_CHUNK_SAMPLES):
            mel = self.audio_feature_extractor(
                speech_array[idx : idx + _WHISPER_AUDIO_CHUNK_SAMPLES],
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).input_features
            mel_chunks.append(mel)
        audio_features = torch.cat(mel_chunks, dim=-1).to(self.audio_encoder.dtype)

        enc_chunks = []
        for idx in range(0, audio_features.shape[-1], _WHISPER_ENCODER_CHUNK_FRAMES):
            chunk_hs = self.audio_encoder.encoder(
                audio_features[:, :, idx : idx + _WHISPER_ENCODER_CHUNK_FRAMES].to(self.device),
                output_hidden_states=True,
            ).hidden_states
            enc_chunks.append(torch.stack(chunk_hs, dim=2))
        audio_prompts = torch.cat(enc_chunks, dim=1)[:, : video_length * 2]

        def interpolate(features, input_fps, output_fps, output_len):
            features = features.transpose(1, 2)
            output_features = F.interpolate(features, size=output_len, align_corners=True, mode="linear")
            return output_features.transpose(1, 2)

        feat0 = interpolate(audio_prompts[:, :, 0:8].mean(dim=2), enc_fps, fps, video_length)
        feat1 = interpolate(audio_prompts[:, :, 8:16].mean(dim=2), enc_fps, fps, video_length)
        feat2 = interpolate(audio_prompts[:, :, 16:24].mean(dim=2), enc_fps, fps, video_length)
        feat3 = interpolate(audio_prompts[:, :, 24:32].mean(dim=2), enc_fps, fps, video_length)
        feat4 = interpolate(audio_prompts[:, :, 32], enc_fps, fps, video_length)
        return torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]

    def _audio_embedding_from_array(
        self,
        speech_array: np.ndarray,
        *,
        sample_rate: int,
        num_frames: int,
        save_fps: int,
    ) -> torch.Tensor:
        full_audio_emb = self._full_audio_embedding_from_array(
            speech_array,
            sample_rate=sample_rate,
            min_duration=num_frames / save_fps,
            save_fps=save_fps,
        )
        return self._slice_full_audio_embedding(full_audio_emb, audio_start_idx=0, num_frames=num_frames)

    def _full_audio_embedding_from_array(
        self,
        speech_array: np.ndarray,
        *,
        sample_rate: int,
        min_duration: float,
        save_fps: int,
    ) -> torch.Tensor:
        source_duration = len(speech_array) / sample_rate
        added_sample_nums = math.ceil((min_duration - source_duration) * sample_rate)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, np.zeros(added_sample_nums, dtype=np.float32))

        full_audio_emb = self._get_audio_embedding_whisper(
            speech_array,
            fps=save_fps * self.audio_stride,
            sample_rate=sample_rate,
        )
        if torch.isnan(full_audio_emb).any():
            raise ValueError("Audio embedding contains NaN values.")
        return full_audio_emb.to(self.device)

    def _slice_full_audio_embedding(
        self,
        full_audio_emb: torch.Tensor,
        *,
        audio_start_idx: int,
        num_frames: int,
    ) -> torch.Tensor:
        indices = torch.arange(5, device=full_audio_emb.device) - 2
        audio_end_idx = audio_start_idx + self.audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, self.audio_stride, device=full_audio_emb.device)
        center_indices = center_indices.unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        return full_audio_emb[center_indices][None, ...].to(self.device)

    def _audio_embedding(self, audio_path: str, num_frames: int, save_fps: int) -> torch.Tensor:
        full_audio_emb, _ = self._audio_full_embedding(
            audio_path,
            min_duration=num_frames / save_fps,
            save_fps=save_fps,
        )
        return self._slice_full_audio_embedding(full_audio_emb, audio_start_idx=0, num_frames=num_frames)

    def _audio_full_embedding(
        self, audio_path: str, *, min_duration: float, save_fps: int
    ) -> tuple[torch.Tensor, float]:
        import soundfile as sf

        temp_vocal_path = self._extract_vocal(audio_path)
        try:
            speech_array, sample_rate = self._load_audio(temp_vocal_path, target_sr=16000, soundfile_module=sf)
            effective_duration = max(len(speech_array) / sample_rate, min_duration)
            full_audio_emb = self._full_audio_embedding_from_array(
                speech_array,
                sample_rate=sample_rate,
                min_duration=effective_duration,
                save_fps=save_fps,
            )
            return full_audio_emb, effective_duration
        finally:
            Path(temp_vocal_path).unlink(missing_ok=True)

    def _multi_audio_embedding(
        self,
        audio_paths: dict[str, str],
        *,
        audio_type: str,
        num_frames: int,
        save_fps: int,
        include_background_silent_audio: bool,
    ) -> torch.Tensor:
        import soundfile as sf

        temp_vocal_paths: list[str] = []
        try:
            speech_arrays = []
            sample_rate = 16000
            for _, audio_path in sorted(audio_paths.items(), key=lambda item: _speaker_sort_key(item[0])):
                temp_vocal_path = self._extract_vocal(audio_path)
                temp_vocal_paths.append(temp_vocal_path)
                speech_array, sample_rate = self._load_audio(temp_vocal_path, target_sr=16000, soundfile_module=sf)
                speech_arrays.append(speech_array)

            aligned_arrays = _prepare_multi_speaker_audio_arrays(
                speech_arrays,
                audio_type=audio_type,
                generate_duration=num_frames / save_fps,
                sample_rate=sample_rate,
            )
            if include_background_silent_audio:
                aligned_arrays.append(np.zeros_like(aligned_arrays[0], dtype=np.float32))

            audio_embs = [
                self._audio_embedding_from_array(
                    speech_array,
                    sample_rate=sample_rate,
                    num_frames=num_frames,
                    save_fps=save_fps,
                )
                for speech_array in aligned_arrays
            ]
            return torch.cat(audio_embs, dim=0)
        finally:
            for temp_vocal_path in temp_vocal_paths:
                Path(temp_vocal_path).unlink(missing_ok=True)

    def _multi_audio_full_embeddings(
        self,
        audio_paths: dict[str, str],
        *,
        audio_type: str,
        min_duration: float,
        save_fps: int,
        include_background_silent_audio: bool,
    ) -> tuple[list[torch.Tensor], float]:
        import soundfile as sf

        temp_vocal_paths: list[str] = []
        try:
            speech_arrays = []
            sample_rate = 16000
            for _, audio_path in sorted(audio_paths.items(), key=lambda item: _speaker_sort_key(item[0])):
                temp_vocal_path = self._extract_vocal(audio_path)
                temp_vocal_paths.append(temp_vocal_path)
                speech_array, sample_rate = self._load_audio(temp_vocal_path, target_sr=16000, soundfile_module=sf)
                speech_arrays.append(speech_array)

            aligned_arrays = _prepare_multi_speaker_audio_arrays(
                speech_arrays,
                audio_type=audio_type,
                generate_duration=min_duration,
                sample_rate=sample_rate,
            )
            if include_background_silent_audio:
                aligned_arrays.append(np.zeros_like(aligned_arrays[0], dtype=np.float32))

            effective_duration = len(aligned_arrays[0]) / sample_rate
            full_embs = [
                self._full_audio_embedding_from_array(
                    speech_array,
                    sample_rate=sample_rate,
                    min_duration=effective_duration,
                    save_fps=save_fps,
                )
                for speech_array in aligned_arrays
            ]
            return full_embs, effective_duration
        finally:
            for temp_vocal_path in temp_vocal_paths:
                Path(temp_vocal_path).unlink(missing_ok=True)

    def _slice_multi_audio_embeddings(
        self,
        full_audio_embs: list[torch.Tensor],
        *,
        audio_start_idx: int,
        num_frames: int,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self._slice_full_audio_embedding(
                    full_audio_emb,
                    audio_start_idx=audio_start_idx,
                    num_frames=num_frames,
                )
                for full_audio_emb in full_audio_embs
            ],
            dim=0,
        )

    def _prepare_audio_embeddings(
        self,
        inputs: dict[str, Any],
        *,
        is_multi_speaker: bool,
        include_background_silent_audio: bool,
        needs_full_audio: bool,
        auto_num_segments: bool,
        num_segments_value: Any,
        num_segments: int,
        num_frames: int,
        num_cond_frames: int,
        save_fps: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, int]:
        if not needs_full_audio:
            if is_multi_speaker:
                audio_emb = self._multi_audio_embedding(
                    inputs["audio_paths"],
                    audio_type=inputs["audio_type"],
                    num_frames=num_frames,
                    save_fps=save_fps,
                    include_background_silent_audio=include_background_silent_audio,
                )
            else:
                audio_emb = self._audio_embedding(inputs["audio_path"], num_frames, save_fps)
            return audio_emb, None, num_segments

        requested_duration = _requested_audio_duration(
            auto_num_segments=auto_num_segments,
            num_segments=num_segments,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            save_fps=save_fps,
        )
        if is_multi_speaker:
            full_audio_embs, effective_audio_duration = self._multi_audio_full_embeddings(
                inputs["audio_paths"],
                audio_type=inputs["audio_type"],
                save_fps=save_fps,
                min_duration=requested_duration,
                include_background_silent_audio=include_background_silent_audio,
            )
        else:
            full_audio_emb, effective_audio_duration = self._audio_full_embedding(
                inputs["audio_path"],
                min_duration=requested_duration,
                save_fps=save_fps,
            )
            full_audio_embs = [full_audio_emb]

        num_segments = _resolve_num_segments(
            num_segments_value,
            audio_duration=effective_audio_duration,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            save_fps=save_fps,
        )
        audio_emb = self._slice_multi_audio_embeddings(
            full_audio_embs,
            audio_start_idx=0,
            num_frames=num_frames,
        )
        return audio_emb, full_audio_embs, num_segments

    @staticmethod
    def _load_audio(audio_path: str, target_sr: int, soundfile_module) -> tuple[np.ndarray, int]:
        speech_array, sr = soundfile_module.read(audio_path, dtype="float32", always_2d=True)
        speech = torch.from_numpy(speech_array.T).float()
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        if sr != target_sr:
            try:
                import torchaudio.functional as taF

                speech = taF.resample(speech, sr, target_sr)
            except ImportError:
                target_len = max(1, int(round(speech.shape[-1] * float(target_sr) / float(sr))))
                speech = F.interpolate(speech.unsqueeze(0), size=target_len, mode="linear", align_corners=False)[0]
            sr = target_sr
        return speech.squeeze(0).cpu().numpy(), sr

    def _condition_shape(self, image: Image.Image, resolution: str) -> tuple[int, int]:
        bucket_config = _get_longcat_bucket_config(resolution)
        ratio = image.height / image.width
        closest_bucket = sorted(bucket_config.keys(), key=lambda key: abs(float(key) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return target_h, target_w

    def _preprocess_image(self, image: Image.Image, height: int, width: int, resize_mode: str = "crop") -> torch.Tensor:
        try:
            return self.video_processor.preprocess(image, height=height, width=width, resize_mode=resize_mode)
        except TypeError:
            return self.video_processor.preprocess(image, height=height, width=width)

    def _preprocess_video_frames(
        self,
        video: list[Image.Image],
        height: int,
        width: int,
        resize_mode: str = "crop",
    ) -> torch.Tensor:
        if hasattr(self.video_processor, "preprocess_video"):
            try:
                return self.video_processor.preprocess_video(
                    video,
                    height=height,
                    width=width,
                    resize_mode=resize_mode,
                )
            except TypeError:
                return self.video_processor.preprocess_video(video, height=height, width=width)

        frame_tensors = [
            self._preprocess_image(frame.convert("RGB"), height, width, resize_mode=resize_mode) for frame in video
        ]
        return torch.stack([frame.squeeze(0) for frame in frame_tensors], dim=1).unsqueeze(0)

    def _resize_and_centercrop_tensor(
        self,
        tensor: torch.Tensor,
        height: int,
        width: int,
        resize_mode: str,
    ) -> torch.Tensor:
        if tensor.shape[-2:] == (height, width):
            return tensor

        tensor = tensor.float().unsqueeze(1)
        if resize_mode == "crop":
            src_height, src_width = tensor.shape[-2:]
            scale = max(height / src_height, width / src_width)
            resized_height = max(height, int(math.ceil(src_height * scale)))
            resized_width = max(width, int(math.ceil(src_width * scale)))
            tensor = F.interpolate(tensor, size=(resized_height, resized_width), mode="nearest")
            top = max(0, (resized_height - height) // 2)
            left = max(0, (resized_width - width) // 2)
            tensor = tensor[..., top : top + height, left : left + width]
        else:
            tensor = F.interpolate(tensor, size=(height, width), mode="nearest")
        return tensor.squeeze(1)

    def _decode_latents(self, latents: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return latents
        latents = self.denormalize_latents(latents.to(self.vae.dtype))
        output_video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(output_video, output_type=output_type)

    def _generate_at2v(
        self,
        prompt: str,
        negative_prompt: str,
        audio_emb: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
        text_guidance_scale: float,
        audio_guidance_scale: float,
        use_distill: bool,
        generator,
        latents: torch.Tensor | None,
        max_sequence_length: int,
        return_latents: bool = False,
    ):
        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        device = self.device
        dtype = self.transformer.dtype
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        audio_cond_embs = audio_emb
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_uncond_embs = torch.zeros_like(audio_cond_embs)
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)
        else:
            audio_uncond_embs = None

        sigmas = self.get_timesteps_sigmas(steps, use_distill=use_distill)
        self.scheduler.set_timesteps(steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        latents = self.prepare_latents(
            image=None,
            batch_size=1,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=0,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        with torch.no_grad():
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    audio_embs=audio_cond_embs,
                )
                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dtype)
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        audio_embs=audio_uncond_embs,
                    )
                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text)
                        + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
                    )
                latents = self.scheduler.step(-noise_pred, t, latents, return_dict=False)[0]
        output = self._decode_latents(latents, output_type="np")
        if return_latents:
            return output, latents.clone()
        return output

    def _generate_ai2v(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        audio_emb: torch.Tensor,
        resolution: str,
        num_frames: int,
        steps: int,
        text_guidance_scale: float,
        audio_guidance_scale: float,
        use_distill: bool,
        generator,
        latents: torch.Tensor | None,
        max_sequence_length: int,
        resize_mode: str,
        ref_target_masks: torch.Tensor | None = None,
    ):
        height, width = self._condition_shape(image, resolution)
        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        device = self.device
        dtype = self.transformer.dtype
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        audio_cond_embs = audio_emb
        audio_num = audio_cond_embs.shape[0]
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_uncond_embs = torch.zeros_like(audio_cond_embs)
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)
        else:
            audio_uncond_embs = None

        sigmas = self.get_timesteps_sigmas(steps, use_distill=use_distill)
        self.scheduler.set_timesteps(steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        image_tensor = self._preprocess_image(image, height, width, resize_mode=resize_mode).to(
            device=device, dtype=prompt_embeds.dtype
        )
        latents = self.prepare_latents(
            image=image_tensor,
            batch_size=1,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=1,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)
            ref_target_masks = ref_target_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :1] = 0
                if self.do_classifier_free_guidance and ref_target_masks is not None:
                    noise_pred_uncond_text = self.transformer(
                        hidden_states=latent_model_input[:1],
                        timestep=timestep[:1],
                        encoder_hidden_states=prompt_embeds[:1],
                        encoder_attention_mask=prompt_attention_mask[:1],
                        num_cond_latents=1,
                        audio_embs=audio_cond_embs[:audio_num],
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_cond = self.transformer(
                        hidden_states=latent_model_input[1:],
                        timestep=timestep[1:],
                        encoder_hidden_states=prompt_embeds[1:],
                        encoder_attention_mask=prompt_attention_mask[1:],
                        num_cond_latents=1,
                        audio_embs=audio_cond_embs[audio_num : 2 * audio_num],
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred = torch.cat([noise_pred_uncond_text, noise_pred_cond])
                else:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        num_cond_latents=1,
                        audio_embs=audio_cond_embs,
                        ref_target_masks=ref_target_masks,
                    )
                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep_uncond[:, :1] = 0
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=1,
                        audio_embs=audio_uncond_embs,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text)
                        + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
                    )
                latents[:, :, 1:] = self.scheduler.step(
                    -noise_pred[:, :, 1:],
                    t,
                    latents[:, :, 1:],
                    return_dict=False,
                )[0]
        return self._decode_latents(latents, output_type="np"), latents.clone(), height, width

    def _generate_avc(
        self,
        video: list[Image.Image],
        video_latent: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        audio_emb: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        num_cond_frames: int,
        steps: int,
        text_guidance_scale: float,
        audio_guidance_scale: float,
        use_distill: bool,
        generator,
        latents: torch.Tensor | None,
        max_sequence_length: int,
        resize_mode: str,
        ref_latent: torch.Tensor | None,
        ref_img_index: int,
        mask_frame_range: int,
        ref_target_masks: torch.Tensor | None = None,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = False,
    ):
        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        device = self.device
        dtype = self.transformer.dtype
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        audio_cond_embs = audio_emb
        audio_num = audio_cond_embs.shape[0]
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_uncond_embs = torch.zeros_like(audio_cond_embs)
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)
        else:
            audio_uncond_embs = None

        sigmas = self.get_timesteps_sigmas(steps, use_distill=use_distill)
        self.scheduler.set_timesteps(steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        video_tensor = self._preprocess_video_frames(video, height, width, resize_mode=resize_mode).to(
            device=device,
            dtype=prompt_embeds.dtype,
        )
        cond_videos = video_tensor[:, :, -num_cond_frames:]
        cond_videos_latents = _retrieve_latents(self.vae.encode(cond_videos), generator, sample_mode="argmax")
        cond_videos_latents = self.normalize_latents(cond_videos_latents)

        latents = self.prepare_latents(
            image=None,
            video=video_latent,
            batch_size=1,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
            need_encode=False,
        )
        num_cond_latents = 1 + (num_cond_frames - 1) // self.vae.config.scale_factor_temporal
        latents[:, :, :num_cond_latents] = cond_videos_latents

        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)
            ref_target_masks = ref_target_masks.to(device=device, dtype=torch.float32)

        num_ref_latents = 0
        if ref_latent is not None:
            num_cond_latents += 1
            num_ref_latents = 1
            latents = torch.cat([ref_latent.to(device=device, dtype=latents.dtype), latents], dim=2)

        if use_kv_cache:
            cond_latents = latents[:, :, :num_cond_latents]
            self._cache_clean_latents(
                cond_latents,
                max_sequence_length,
                offload_kv_cache=offload_kv_cache,
                dtype=dtype,
                audio_embs=audio_emb,
                num_cond_latents=num_cond_latents,
                num_ref_latents=num_ref_latents,
                ref_img_index=ref_img_index,
            )
            kv_cache_dict = self._get_kv_cache_dict()
            latents = latents[:, :, num_cond_latents:]
        else:
            cond_latents = None
            kv_cache_dict = {}

        with torch.no_grad():
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :num_cond_latents] = 0

                if self.do_classifier_free_guidance and ref_target_masks is not None:
                    noise_pred_uncond_text = self.transformer(
                        hidden_states=latent_model_input[:1],
                        timestep=timestep[:1],
                        encoder_hidden_states=prompt_embeds[:1],
                        encoder_attention_mask=prompt_attention_mask[:1],
                        num_cond_latents=num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_cond_embs[:audio_num],
                        num_ref_latents=num_ref_latents,
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_cond = self.transformer(
                        hidden_states=latent_model_input[1:],
                        timestep=timestep[1:],
                        encoder_hidden_states=prompt_embeds[1:],
                        encoder_attention_mask=prompt_attention_mask[1:],
                        num_cond_latents=num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_cond_embs[audio_num : 2 * audio_num],
                        num_ref_latents=num_ref_latents,
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred = torch.cat([noise_pred_uncond_text, noise_pred_cond])
                else:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        num_cond_latents=num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_cond_embs,
                        num_ref_latents=num_ref_latents,
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    if not use_kv_cache:
                        timestep_uncond[:, :num_cond_latents] = 0
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_uncond_embs,
                        num_ref_latents=num_ref_latents,
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text)
                        + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
                    )

                if use_kv_cache:
                    latents = self.scheduler.step(-noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(
                        -noise_pred[:, :, num_cond_latents:],
                        t,
                        latents[:, :, num_cond_latents:],
                        return_dict=False,
                    )[0]

        if use_kv_cache:
            latents = torch.cat([cond_latents, latents], dim=2)
            self._clear_cache()

        if ref_latent is not None:
            latents = latents[:, :, num_ref_latents:]
            num_cond_latents -= num_ref_latents
        latents[:, :, :num_cond_latents] = cond_videos_latents

        return self._decode_latents(latents, output_type="np"), latents.clone()

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if req.is_dummy_run():
            height = req.sampling_params.height or 512
            width = req.sampling_params.width or 512
            frame = torch.zeros((1, height, width, 3), dtype=torch.uint8)
            return DiffusionOutput(output=frame)

        started = time.perf_counter()
        inputs = self._resolve_request_inputs(req)
        extra = req.sampling_params.extra_args or {}
        stage = inputs["stage"]
        if stage not in {"at2v", "ai2v"}:
            raise ValueError(f"LongCat-Video-Avatar stage must be 'at2v' or 'ai2v', got {stage!r}.")
        resolution = inputs["resolution"]
        if resolution not in {"480p", "720p"}:
            raise ValueError(f"Unsupported LongCat-Video-Avatar resolution {resolution!r}.")

        num_frames = _adjust_num_frames(int(extra.get("num_frames") or req.sampling_params.num_frames or 93))
        save_fps = int(extra.get("save_fps") or req.sampling_params.fps or self.save_fps)
        num_cond_frames = int(extra.get("num_cond_frames") or 13)
        use_distill = _as_bool(extra.get("use_distill"), self.use_distill)
        steps = int(extra.get("num_inference_steps") or req.sampling_params.num_inference_steps or 50)
        text_guidance_scale = float(extra.get("text_guidance_scale") or req.sampling_params.guidance_scale or 4.0)
        audio_guidance_scale = float(extra.get("audio_guidance_scale") or req.sampling_params.guidance_scale_2 or 4.0)
        num_segments_value = extra.get("num_segments")
        resize_mode = str(extra.get("resize_mode") or "crop")
        use_kv_cache = _as_bool(extra.get("use_kv_cache"), True)
        offload_kv_cache = _as_bool(extra.get("offload_kv_cache"), False)
        ref_img_index = int(extra.get("ref_img_index") or 10)
        mask_frame_range = int(extra.get("mask_frame_range") or 3)
        if use_distill:
            steps = 8
            text_guidance_scale = 1.0
            audio_guidance_scale = 1.0

        generator = req.sampling_params.generator
        if generator is None:
            seed = req.sampling_params.seed if req.sampling_params.seed is not None else 42
            generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = req.sampling_params.latents
        max_sequence_length = req.sampling_params.max_sequence_length or int(extra.get("max_sequence_length") or 512)

        image = None
        ref_target_masks = None
        include_background_silent_audio = False
        explicit_num_segments = _requested_num_segments(num_segments_value)
        auto_num_segments = isinstance(num_segments_value, str) and num_segments_value.lower() == "auto"
        needs_full_audio = auto_num_segments or explicit_num_segments > 1
        num_segments = explicit_num_segments or 1
        if inputs["is_multi_speaker"]:
            if stage != "ai2v":
                raise NotImplementedError(
                    "LongCat-Video-Avatar multi-speaker generation requires AI2V "
                    "because speaker bounding boxes and masks are defined on a reference image."
                )
            image = _load_image(inputs["image"])
            ref_target_masks, include_background_silent_audio = _build_multi_speaker_ref_target_masks(
                image,
                inputs["bbox"],
            )

        audio_emb, full_audio_embs, num_segments = self._prepare_audio_embeddings(
            inputs,
            is_multi_speaker=inputs["is_multi_speaker"],
            include_background_silent_audio=include_background_silent_audio,
            needs_full_audio=needs_full_audio,
            auto_num_segments=auto_num_segments,
            num_segments_value=num_segments_value,
            num_segments=num_segments,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            save_fps=save_fps,
        )

        if num_segments > 1 and num_cond_frames >= num_frames:
            raise ValueError(
                "LongCat-Video-Avatar AVC continuation requires num_cond_frames to be smaller than num_frames."
            )

        latent = None
        if stage == "at2v":
            default_height, default_width = _default_at2v_shape(resolution)
            height = int(extra.get("height") or req.sampling_params.height or default_height)
            width = int(extra.get("width") or req.sampling_params.width or default_width)
            output = self._generate_at2v(
                prompt=inputs["prompt"],
                negative_prompt=inputs["negative_prompt"],
                audio_emb=audio_emb,
                height=height,
                width=width,
                num_frames=num_frames,
                steps=steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                use_distill=use_distill,
                generator=generator,
                latents=latents,
                max_sequence_length=max_sequence_length,
                return_latents=num_segments > 1,
            )
            if num_segments > 1:
                output, latent = output
        else:
            image = image or _load_image(inputs["image"])
            output, latent, height, width = self._generate_ai2v(
                image=image,
                prompt=inputs["prompt"],
                negative_prompt=inputs["negative_prompt"],
                audio_emb=audio_emb,
                resolution=resolution,
                num_frames=num_frames,
                steps=steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                use_distill=use_distill,
                generator=generator,
                latents=latents,
                max_sequence_length=max_sequence_length,
                resize_mode=resize_mode,
                ref_target_masks=ref_target_masks,
            )

        if num_segments > 1:
            if full_audio_embs is None or latent is None:
                raise RuntimeError("LongCat-Video-Avatar continuation requires full audio embeddings and latents.")
            segment_frames = output[0]
            current_video = [
                Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).round().astype("uint8")) for frame in segment_frames
            ]
            all_generated_frames = list(current_video)
            ref_latent = latent[:, :, :1].clone()
            audio_start_idx = 0
            for _segment_idx in range(1, num_segments):
                audio_start_idx += self.audio_stride * (num_frames - num_cond_frames)
                segment_audio_emb = self._slice_multi_audio_embeddings(
                    full_audio_embs,
                    audio_start_idx=audio_start_idx,
                    num_frames=num_frames,
                )
                segment_output, latent = self._generate_avc(
                    video=current_video,
                    video_latent=latent,
                    prompt=inputs["prompt"],
                    negative_prompt=inputs["negative_prompt"],
                    audio_emb=segment_audio_emb,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_cond_frames=num_cond_frames,
                    steps=steps,
                    text_guidance_scale=text_guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                    use_distill=use_distill,
                    generator=generator,
                    latents=None,
                    max_sequence_length=max_sequence_length,
                    resize_mode=resize_mode,
                    ref_latent=ref_latent,
                    ref_img_index=ref_img_index,
                    mask_frame_range=mask_frame_range,
                    ref_target_masks=ref_target_masks,
                    use_kv_cache=use_kv_cache,
                    offload_kv_cache=offload_kv_cache,
                )
                new_video = [
                    Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).round().astype("uint8"))
                    for frame in segment_output[0]
                ]
                all_generated_frames.extend(new_video[num_cond_frames:])
                current_video = new_video
            frames = np.asarray([np.asarray(frame) for frame in all_generated_frames], dtype=np.uint8)
        else:
            frames = (np.clip(output[0], 0.0, 1.0) * 255).round().astype("uint8")
        return DiffusionOutput(
            output=torch.from_numpy(frames),
            stage_durations={"avatar_generate_s": time.perf_counter() - started},
        )
