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

logger = init_logger(__name__)

_DEFAULT_NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)
_DEFAULT_AT2V_RATIO = 480 / 832

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
        "whisper-large-v3/*",
        "vocal_separator/Kim_Vocal_2.onnx",
        "config.json",
        "model_index.json",
    ]


def _adjust_num_frames(num_frames: int, temporal_scale: int = 4) -> int:
    if num_frames % temporal_scale != 1:
        num_frames = num_frames // temporal_scale * temporal_scale + 1
    return max(num_frames, 1)


def _asset_root() -> Path | None:
    raw = os.environ.get("LONGCAT_VIDEO_ASSET_ROOT")
    if raw and Path(raw).exists():
        return Path(raw)
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path: str | os.PathLike[str] | None, *, asset_root: Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return str(resolved)
    if asset_root is not None:
        candidate = asset_root / resolved
        if candidate.exists():
            return str(candidate)
    return str(resolved)


def _ensure_local_dir(model: str | os.PathLike[str], allow_patterns: list[str] | None = None) -> Path:
    model_path = Path(model)
    if model_path.exists():
        return model_path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model), allow_patterns=allow_patterns))


def _load_image(raw_image: Any, *, asset_root: Path | None) -> Image.Image:
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
        image_path = _resolve_path(raw_image, asset_root=asset_root)
        if image_path is None:
            raise ValueError("LongCat-Video-Avatar image path resolved to None.")
        return Image.open(image_path).convert("RGB")
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
        for idx, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                request.prompts[idx] = {"prompt": prompt, "multi_modal_data": {}, "additional_information": {}}
                continue
            prompt.setdefault("multi_modal_data", {})
            prompt.setdefault("additional_information", {})
        return request

    return pre_process_func


class LongCatVideoAvatarPipeline(nn.Module, SupportImageInput, SupportAudioInput):
    """Native single-speaker LongCat-Video-Avatar A2V/AI2V pipeline."""

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
        self.resolution = str(additional_config.get("resolution") or "480p")
        self.model_dir = _ensure_local_dir(
            od_config.model,
            allow_patterns=_avatar_model_allow_patterns(self.use_int8),
        )
        self.asset_root = _asset_root()
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
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        self.vae = DistributedAutoencoderKLWan.from_pretrained(str(base_dir), subfolder="vae", torch_dtype=dtype).to(
            self.device
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.model_dir), subfolder="scheduler")

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
        self.transformer = self.transformer.to(device=self.device)
        if not self.use_int8:
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
                self.transformer.enable_loras(["dmd"])

        audio_model_path = self.model_dir / "whisper-large-v3"
        self.audio_encoder = WhisperModel.from_pretrained(str(audio_model_path)).eval().to(self.device)
        self.audio_encoder.requires_grad_(False)
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(str(audio_model_path))

        separator_path = self.model_dir / "vocal_separator" / "Kim_Vocal_2.onnx"
        self.audio_temp_dir = Path(tempfile.mkdtemp(prefix="longcat_avatar_audio_"))
        self.vocal_separator = Separator(
            output_dir=self.audio_temp_dir / "vocals",
            output_single_stem="vocals",
            model_file_dir=str(separator_path.parent),
        )
        self.vocal_separator.load_model(separator_path.name)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = torch.device(args[0]) if args else torch.device(kwargs.get("device", self.device))
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
    ) -> torch.Tensor:
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

        if image is not None:
            cond_latents = []
            for idx in range(batch_size):
                gen = generator[idx] if isinstance(generator, list) else generator
                encoded_input = image[idx].unsqueeze(0).unsqueeze(2)
                latent = _retrieve_latents(self.vae.encode(encoded_input), gen, sample_mode="argmax")
                cond_latents.append(latent)
            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
            cond_latents = self.normalize_latents(cond_latents)
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae.config.scale_factor_temporal
            latents[:, :, :num_cond_latents] = cond_latents
        return latents

    def _default_input_json(self) -> dict[str, Any]:
        if self.asset_root is None:
            return {}
        default_json = self.asset_root / "assets/avatar/single_example_1.json"
        return _load_json(default_json) if default_json.exists() else {}

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
        sample = (
            _load_json(Path(_resolve_path(input_json, asset_root=self.asset_root)))
            if input_json
            else self._default_input_json()
        )
        prompt = prompt_text or sample.get("prompt") or ""
        negative_prompt = negative_prompt or extra.get("negative_prompt") or _DEFAULT_NEGATIVE_PROMPT

        audio_field = mm_data.get("audio")
        if isinstance(audio_field, list):
            audio_field = audio_field[0] if audio_field else None
        audio_path = (
            audio_field
            if isinstance(audio_field, str | os.PathLike)
            else extra.get("audio_path") or info.get("audio_path") or (sample.get("cond_audio") or {}).get("person1")
        )
        audio_path = _resolve_path(audio_path, asset_root=self.asset_root)
        if not audio_path:
            raise ValueError("LongCat-Video-Avatar requires multi_modal_data['audio'] or extra_args['audio_path'].")

        image_input = mm_data.get("image")
        image_path = extra.get("image_path") or info.get("image_path") or sample.get("cond_image")
        if image_input is None and image_path is not None:
            image_input = _resolve_path(image_path, asset_root=self.asset_root)

        inferred_stage = "ai2v" if image_input is not None else "at2v"
        stage = str(extra.get("stage") or extra.get("stage_1") or info.get("stage") or inferred_stage).lower()
        resolution = str(extra.get("resolution") or info.get("resolution") or self.resolution)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "audio_path": audio_path,
            "image": image_input,
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
        mel_chunk = 750 * 640
        enc_chunk = 3000
        enc_fps = 50
        audio_duration = len(speech_array) / sample_rate
        video_length = int(audio_duration * fps)
        speech_array = self._loudness_norm(speech_array, sample_rate)

        mel_chunks = []
        for idx in range(0, len(speech_array), mel_chunk):
            mel = self.audio_feature_extractor(
                speech_array[idx : idx + mel_chunk],
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).input_features
            mel_chunks.append(mel)
        audio_features = torch.cat(mel_chunks, dim=-1).to(self.audio_encoder.dtype)

        enc_chunks = []
        for idx in range(0, audio_features.shape[-1], enc_chunk):
            chunk_hs = self.audio_encoder.encoder(
                audio_features[:, :, idx : idx + enc_chunk].to(self.device),
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

    def _audio_embedding(self, audio_path: str, num_frames: int, save_fps: int) -> torch.Tensor:
        import soundfile as sf

        temp_vocal_path = self._extract_vocal(audio_path)
        try:
            generate_duration = num_frames / save_fps
            speech_array, sr = self._load_audio(temp_vocal_path, target_sr=16000, soundfile_module=sf)
            source_duration = len(speech_array) / sr
            added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
            if added_sample_nums > 0:
                speech_array = np.append(speech_array, [0.0] * added_sample_nums)
            full_audio_emb = self._get_audio_embedding_whisper(
                speech_array,
                fps=save_fps * self.audio_stride,
                sample_rate=sr,
            )
            if torch.isnan(full_audio_emb).any():
                raise ValueError("Audio embedding contains NaN values.")

            indices = torch.arange(5, device=full_audio_emb.device) - 2
            audio_end_idx = self.audio_stride * num_frames
            center_indices = torch.arange(0, audio_end_idx, self.audio_stride, device=full_audio_emb.device)
            center_indices = center_indices.unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
            return full_audio_emb[center_indices][None, ...].to(self.device)
        finally:
            Path(temp_vocal_path).unlink(missing_ok=True)

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
        return self._decode_latents(latents, output_type="np")

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

        with torch.no_grad():
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :1] = 0
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=1,
                    audio_embs=audio_cond_embs,
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
        return self._decode_latents(latents, output_type="np"), height, width

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if req.is_dummy_run():
            height = req.sampling_params.height or 512
            width = req.sampling_params.width or 512
            frame = torch.zeros((1, height, width, 3), dtype=torch.uint8)
            return DiffusionOutput(output=frame, custom_output={"fps": self.save_fps})

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
        use_distill = _as_bool(extra.get("use_distill"), self.use_distill)
        steps = int(extra.get("num_inference_steps") or req.sampling_params.num_inference_steps or 50)
        text_guidance_scale = float(extra.get("text_guidance_scale") or req.sampling_params.guidance_scale or 4.0)
        audio_guidance_scale = float(extra.get("audio_guidance_scale") or req.sampling_params.guidance_scale_2 or 4.0)
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

        audio_emb = self._audio_embedding(inputs["audio_path"], num_frames, save_fps)
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
            )
        else:
            image = _load_image(inputs["image"], asset_root=self.asset_root)
            output, height, width = self._generate_ai2v(
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
                resize_mode=str(extra.get("resize_mode") or "crop"),
            )

        frames = output[0]
        frames = (np.clip(frames, 0.0, 1.0) * 255).round().astype("uint8")
        return DiffusionOutput(
            output=torch.from_numpy(frames),
            custom_output={
                "fps": save_fps,
                "audio_path": inputs["audio_path"],
                "stage": stage,
                "resolution": resolution,
                "height": height,
                "width": width,
            },
            stage_durations={"avatar_generate_s": time.perf_counter() - started},
        )
