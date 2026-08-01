# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_NAVA_MODEL_TYPE = "NAVA"
DEFAULT_NAVA_MODEL_INDEX: dict[str, str] = {
    "_class_name": "NAVAPipeline",
    "nava_ckpt": "NAVA.safetensors",
    "fp8_ckpt": "NAVA_fp8.safetensors",
    "config": "configs/nava.yaml",
    "wan_dir": "Wan2.2-TI2V-5B",
    "audio_vae_dir": "params",
    "speaker_dir": "speaker",
}

NAVA_CONFIG_ALIAS_MAP = {
    "nava_ckpt": "ckpt_name",
    "fp8_ckpt": "fp8_ckpt_name",
    "config": "config_name",
    "num_inference_steps": "num_steps",
    "num_frames": "frames",
    "height": "log_height",
    "width": "log_width",
    "audio_vae_dir": "audio_vae_ckpt_dir",
    "wan_dir": "wan_dir",
    "speaker_dir": "speaker_dir",
}

_SPEECH_SPAN_RE = re.compile(r"<S>(.*?)<E>", re.DOTALL)
# Matches upstream NAVA defaults:
# https://github.com/ernie-research/NAVA/blob/main/nava_src/pipeline_nava.py#L397-L398
DEFAULT_NAVA_AUDIO_NEGATIVE_PROMPT = "机械音、闷糊、回音、失真、电流声、爆音、杂音"
DEFAULT_NAVA_VIDEO_NEGATIVE_PROMPT = (
    "画质模糊，多人同时说话，倒着走, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
    "画面，静止，整体发，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，"
    "音频带有机械音、闷糊、回音、失真、电流声、爆音、杂音"
)


@dataclass(frozen=True)
class NAVAConfig:
    model_type: str = DEFAULT_NAVA_MODEL_TYPE
    modality: str = "audio_video"
    use_bf16: bool = True
    audio_latent_ch: int = 128
    video_latent_ch: int = 48
    text_embed_dim: int = 4096
    speaker_embed_dim: int = 192
    latent_spatial_stride: int = 16
    latent_temporal_stride: int = 4
    log_height: int = 704
    log_width: int = 1280
    frames: int = 37
    fps: int = 24
    num_steps: int = 50
    audio_tokens_per_sec: float = 25.0
    audio_sample_rate: int = 16000
    video_guidance_scale: float = 3.0
    audio_guidance_scale: float = 2.0
    video_align_guidance_scale: float = 3.0
    audio_align_guidance_scale: float = 2.0
    timbre_align_guidance_scale: float = 3.0
    align_3d_cfg: bool = True
    timbre_cfg: bool = True
    negative_prompt: str = ""
    negative_prompt_mode: bool = True
    audio_negative_prompt: str = DEFAULT_NAVA_AUDIO_NEGATIVE_PROMPT
    video_negative_prompt: str = DEFAULT_NAVA_VIDEO_NEGATIVE_PROMPT
    ckpt_name: str = "NAVA.safetensors"
    fp8_ckpt_name: str = "NAVA_fp8.safetensors"
    config_name: str = "configs/nava.yaml"
    wan_dir: str = "Wan2.2-TI2V-5B"
    audio_vae_ckpt_dir: str = "params"
    speaker_dir: str = "speaker"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    hidden_size: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    num_heads: int = 24
    num_layers: int = 30
    num_double_layers: int = 10
    num_single_layers: int = 20
    text_len: int = 512

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> NAVAConfig:
        if not raw:
            return cls()
        data = dict(raw)
        explicit_keys = set(data.pop("_explicit_keys", ()))

        for old_key, new_key in NAVA_CONFIG_ALIAS_MAP.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        data_block = data.get("data")
        if isinstance(data_block, dict):
            data.setdefault("audio_tokens_per_sec", data_block.get("audio_tokens_per_sec", cls.audio_tokens_per_sec))
            if "video_fps" in data_block:
                data.setdefault("fps", data_block["video_fps"])

        model_block = data.get("model")
        if isinstance(model_block, dict):
            data.setdefault("audio_vae_ckpt_dir", model_block.get("audio_vae_ckpt_dir", cls.audio_vae_ckpt_dir))
            joint_config = model_block.get("joint_config_data")
            if isinstance(joint_config, dict):
                _merge_joint_config(data, joint_config, preserve_keys=explicit_keys)

        if isinstance(data.get("patch_size"), list):
            data["patch_size"] = tuple(data["patch_size"])
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid_keys})

    @property
    def target_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.use_bf16 else torch.float16

    def video_latent_hw(self, height: int | None = None, width: int | None = None) -> tuple[int, int]:
        resolved_height = int(height or self.log_height)
        resolved_width = int(width or self.log_width)
        if resolved_height % self.latent_spatial_stride or resolved_width % self.latent_spatial_stride:
            raise ValueError(
                "NAVA height and width must be divisible by "
                f"{self.latent_spatial_stride}: got height={resolved_height}, width={resolved_width}."
            )
        return resolved_height // self.latent_spatial_stride, resolved_width // self.latent_spatial_stride

    def normalize_output_frames(self, frames: int | None = None) -> int:
        resolved_frames = max(1, int(frames or self.frames))
        if resolved_frames % self.latent_temporal_stride != 1:
            resolved_frames = resolved_frames // self.latent_temporal_stride * self.latent_temporal_stride + 1
        return max(1, resolved_frames)

    def video_latent_frames(self, frames: int | None = None) -> int:
        resolved_frames = self.normalize_output_frames(frames)
        return (resolved_frames - 1) // self.latent_temporal_stride + 1

    def video_output_frames(self, latent_frames: int) -> int:
        return (max(1, int(latent_frames)) - 1) * self.latent_temporal_stride + 1

    def audio_latent_length(self, frames: int | None = None, fps: int | float | None = None) -> int:
        resolved_frames = max(1, int(frames or self.frames))
        resolved_fps = float(fps or self.fps)
        video_duration_s = ((resolved_frames - 1) * self.latent_temporal_stride + 1) / resolved_fps
        return max(1, math.ceil(video_duration_s * self.audio_tokens_per_sec))


@dataclass(frozen=True)
class NAVASpeakerCondition:
    wavs: list[Any]
    spans: list[str]


@dataclass(frozen=True)
class NAVARequestContext:
    prompt: str
    negative_prompt: str
    audio_negative_prompt: str
    video_negative_prompt: str
    image: Any | None
    speaker_condition: NAVASpeakerCondition | None
    height: int
    width: int
    frames: int
    fps: float
    seed: int
    num_steps: int
    video_guidance_scale: float
    audio_guidance_scale: float
    video_align_guidance_scale: float
    audio_align_guidance_scale: float
    timbre_align_guidance_scale: float
    align_3d_cfg: bool
    timbre_cfg: bool
    negative_prompt_mode: bool


def parse_speech_spans(prompt: str) -> list[str]:
    return [match.group(1) for match in _SPEECH_SPAN_RE.finditer(prompt or "")]


def inject_speaker_sentinel(prompt: str) -> str:
    return (prompt or "").replace("<S>", "<S><extra_id_2>")


def _merge_joint_config(
    data: dict[str, Any],
    joint_config: dict[str, Any],
    *,
    preserve_keys: set[str] | None = None,
) -> None:
    preserve_keys = preserve_keys or set()
    key_map = {
        "dim": "hidden_size",
        "vid_in_dim": "video_latent_ch",
        "audio_in_dim": "audio_latent_ch",
    }
    for key, value in joint_config.items():
        target = key_map.get(key, key)
        if target not in preserve_keys:
            data[target] = value
