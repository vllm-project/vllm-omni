# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import (
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.lingbot_video.image_condition import (
    LingBotImageCondition,
    apply_clean_prefix,
    prepare_ti2v_image_condition,
)
from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import LingBotVideoTransformer3DModel
from vllm_omni.diffusion.models.lingbot_video.request_utils import (
    LingBotGenerationMode,
    normalize_lingbot_request,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError

TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0
LOW_NOISE_TAIL_V1_DEFAULT_STEPS = 2

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", '
    '"pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", '
    '"underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], '
    '"artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", '
    '"cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", '
    '"signature", "logo", "subtitles", "pillarboxed", "side bars", '
    '"portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", '
    '"jittery", "motion blur", "temporal inconsistency", "warping", "morphing", '
    '"incoherent motion", "unnatural movement", "static object with sudden jump", '
    '"frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", '
    '"unrealistic texture", "deformed bottle", "liquid freezing improperly", '
    '"distorted reflections"]}}'
)
DEFAULT_NEGATIVE_PROMPT_IMAGE = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", '
    '"blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", '
    '"overexposed", "invisible subject", "subject hidden in darkness"], '
    '"artistic_style": ["painting", "illustration", "drawing", "cartoon", '
    '"3d render", "cgi", "sketch", "digital art"], "composition_and_content": '
    '["text", "watermark", "signature", "logo", "pillarboxed", "side bars", '
    '"portrait image in landscape frame"], "material_and_structure": '
    '["plastic-like glass", "unrealistic texture", "deformed bottle", '
    '"distorted reflections"]}}'
)


def _dtype_from_name(value: Any, default: torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported LingBot dtype: {value!r}.")


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _transformer_timestep(timestep: torch.Tensor, transformer_dtype: torch.dtype) -> torch.Tensor:
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    return (sigma * 1000.0).float()


def _transformer_autocast(device: torch.device, transformer_dtype: torch.dtype):
    if device.type != "cuda" or transformer_dtype not in {torch.bfloat16, torch.float16}:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=transformer_dtype)


def _group_global_rank(group: Any | None, group_rank: int) -> int:
    if group is None:
        return group_rank
    get_global_rank = getattr(dist, "get_global_rank", None)
    if get_global_rank is None:
        return group_rank
    return int(get_global_rank(group, group_rank))


def _validate_refiner_sigmas(sigmas: np.ndarray, t_thresh: float | None = None) -> np.ndarray:
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("refiner sigma schedule must be a non-empty 1D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("refiner sigma schedule contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"refiner sigma schedule values must be in [0, 1], got {arr.tolist()}")
    if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
        raise ValueError(f"refiner sigma schedule must be strictly descending, got {arr.tolist()}")
    if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
        raise ValueError(f"refiner sigma schedule must start at t_thresh={float(t_thresh)}, got {float(arr[0])}")
    return arr


def _compute_refiner_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float | None,
    tail_steps: int = 0,
) -> np.ndarray | None:
    if t_thresh is None:
        return None
    t_value = float(t_thresh)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps or 0)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
    shift_value = float(shift)
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = np.concatenate([[t_value], sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
        sigmas = np.concatenate([sigmas, extra])
    return _validate_refiner_sigmas(sigmas, t_value).astype(np.float32)


def _pad_prompt_embeds(
    embeds: torch.Tensor,
    mask: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeds.shape[0] != 1:
        raise ValueError(f"batched CFG helper expects batch=1 inputs, got {embeds.shape[0]}")
    if embeds.shape[1] > target_length:
        raise ValueError(f"cannot pad length {embeds.shape[1]} down to {target_length}")
    pad_len = target_length - embeds.shape[1]
    if pad_len == 0:
        return embeds, mask
    embed_pad = torch.zeros(embeds.shape[0], pad_len, embeds.shape[2], dtype=embeds.dtype, device=embeds.device)
    mask_pad = torch.zeros(mask.shape[0], pad_len, dtype=mask.dtype, device=mask.device)
    return torch.cat([embeds, embed_pad], dim=1), torch.cat([mask, mask_pad], dim=1)


def _batch_cfg_prompt_inputs(
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    null_cond_clone_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if null_cond_clone_zero:
        zero_negative = torch.zeros_like(prompt_embeds)
        return (
            torch.cat([prompt_embeds, zero_negative], dim=0),
            torch.cat([prompt_mask, prompt_mask.clone()], dim=0),
        )

    target_length = max(int(prompt_embeds.shape[1]), int(negative_embeds.shape[1]))
    prompt_padded, prompt_mask_padded = _pad_prompt_embeds(prompt_embeds, prompt_mask, target_length)
    negative_padded, negative_mask_padded = _pad_prompt_embeds(negative_embeds, negative_mask, target_length)
    return (
        torch.cat([prompt_padded, negative_padded], dim=0),
        torch.cat([prompt_mask_padded, negative_mask_padded], dim=0),
    )


def _load_lingbot_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        try:
            with Image.open(value) as image:
                return image.convert("RGB")
        except OSError as exc:
            raise OmniClientError(f"Unable to load LingBot input image from {value!r}: {exc}.") from exc
    raise OmniClientError(f"Unsupported LingBot image format {type(value)!r}; expected a PIL image or local path.")


def get_lingbot_video_pre_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            return request
        if not isinstance(prompt, Mapping):
            raise OmniClientError(f"LingBot prompt must be a string or mapping, got {type(prompt)!r}.")
        multi_modal_data = prompt.get("multi_modal_data")
        if multi_modal_data is None:
            return request
        if not isinstance(multi_modal_data, Mapping):
            raise OmniClientError("LingBot `multi_modal_data` must be a mapping.")
        raw_images = multi_modal_data.get("image")
        if raw_images is None:
            return request

        multiple = isinstance(raw_images, (list, tuple))
        image_values = list(raw_images) if multiple else [raw_images]
        images = [_load_lingbot_image(image) for image in image_values]
        normalized_prompt = dict(prompt)
        normalized_multi_modal_data = dict(multi_modal_data)
        normalized_multi_modal_data["image"] = images if multiple else images[0]
        normalized_prompt["multi_modal_data"] = normalized_multi_modal_data
        if request.is_dummy_run() and "modalities" not in normalized_prompt:
            normalized_prompt["modalities"] = ["video"]
        request.prompt = normalized_prompt
        return request

    return pre_process_func


def get_lingbot_video_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(frames: torch.Tensor | dict[str, torch.Tensor], sampling_params=None):
        output_key = None
        if isinstance(frames, dict):
            output_keys = [key for key in ("image", "video") if key in frames]
            if len(output_keys) != 1:
                raise ValueError(
                    f"LingBot output must contain exactly one of 'image' or 'video', got {sorted(frames)!r}."
                )
            output_key = output_keys[0]
            frames = frames[output_key]
        output_type = getattr(sampling_params, "output_type", None) or "pt"
        # The image serving path currently accepts PIL images or NumPy arrays,
        # while LingBot decodes T2I outputs to tensors. Keep that compatibility
        # conversion without discarding the model's image/video payload key.
        if (
            isinstance(frames, torch.Tensor)
            and output_type != "latent"
            and (output_type == "np" or output_key == "image")
        ):
            frames = frames.float().cpu().numpy()
        return {output_key: frames} if output_key is not None else frames

    return post_process_func


class LingBotVideoPipeline(
    nn.Module,
    SupportImageInput,
    ProgressBarMixin,
    SupportsComponentDiscovery,
):
    """Native vLLM-Omni entry for LingBot-Video checkpoints.

    The in-tree transformer supports both dense MLP blocks and routed MoE blocks.
    Fused expert kernels and the optional ``refiner/`` transformer are not loaded
    or executed. ``t_thresh`` only selects a low-noise sigma schedule for the
    primary transformer; it does not enable automatic refiner orchestration.
    """

    supports_step_execution: ClassVar[bool] = False
    max_outputs_per_prompt: ClassVar[int] = 1
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self.img_prompt_template = IMG_PROMPT_TEMPLATE
        self._crop_start: int | None = None

        model = od_config.model
        local_files_only = os.path.exists(model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = getattr(od_config, "model_config", None) or {}
        transformer_dtype = _dtype_from_name(model_config.get("transformer_dtype"), dtype)
        text_encoder_dtype = _dtype_from_name(model_config.get("text_encoder_dtype"), dtype)
        vae_dtype = _dtype_from_name(model_config.get("vae_dtype"), torch.float32)

        transformer_subfolder = str(model_config.get("transformer_subfolder", "transformer"))
        text_encoder_subfolder = str(model_config.get("text_encoder_subfolder", "text_encoder"))
        processor_subfolder = str(model_config.get("processor_subfolder", "processor"))
        vae_subfolder = str(model_config.get("vae_subfolder", "vae"))
        scheduler_subfolder = str(model_config.get("scheduler_subfolder", "scheduler"))

        self.transformer = LingBotVideoTransformer3DModel.from_pretrained(
            model,
            subfolder=transformer_subfolder,
            torch_dtype=transformer_dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        text_encoder_kwargs: dict[str, Any] = {
            "dtype": text_encoder_dtype,
            "local_files_only": local_files_only,
        }
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            subfolder=text_encoder_subfolder,
            **text_encoder_kwargs,
        ).to(self.device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model,
            subfolder=processor_subfolder,
            local_files_only=local_files_only,
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model,
            subfolder=vae_subfolder,
            torch_dtype=vae_dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.scheduler = FlowUniPCMultistepScheduler.from_pretrained(
            model,
            subfolder=scheduler_subfolder,
            local_files_only=local_files_only,
        )
        self.set_progress_bar_config(disable=bool(model_config.get("quiet_progress", True)))
        self.default_negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.default_image_negative_prompt = DEFAULT_NEGATIVE_PROMPT_IMAGE

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        super().to(*args, **kwargs)
        if device is not None:
            self.device = torch.device(device)
            self.transformer.to(device=self.device, non_blocking=non_blocking)
            self.text_encoder.to(device=self.device, non_blocking=non_blocking)
            self.vae.to(device=self.device, non_blocking=non_blocking)
        if dtype is not None:
            self.transformer.to(dtype=dtype, non_blocking=non_blocking)
        return self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"{self.__class__.__name__}.load_weights received {len(consumed)} weight tensors; "
                "LingBot components are loaded directly from subfolders during __init__."
            )
        return {name for name, _ in self.named_parameters()}

    @staticmethod
    def check_inputs(height: int, width: int, num_frames: int) -> None:
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be multiples of 16, got {height}x{width}.")

    @staticmethod
    def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
        return template.format(text)

    def _compute_crop_start(self) -> int:
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.prompt_template.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = self.processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _build_prompt_inputs(
        self,
        prompt: str | list[str],
        *,
        images: Any | None = None,
    ):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        visual_template = self.img_prompt_template if images is not None else ""
        texts = [self.apply_text_to_template(visual_template + text, self.prompt_template) for text in prompts]
        return self.processor(
            text=texts,
            images=images,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        *,
        images: Any | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device) if device is not None else self.device
        inputs = self._build_prompt_inputs(prompt, images=images).to(device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]
        return prompt_embeds, prompt_mask

    def _vision_patch_size(self) -> int:
        for obj in (
            getattr(getattr(self.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(self.processor, "image_processor", None), "config", None),
            getattr(self.processor, "image_processor", None),
        ):
            patch_size = getattr(obj, "patch_size", None)
            if patch_size is not None:
                return int(patch_size)
        return 16

    def prepare_ti2v_image_condition(
        self,
        image: Image.Image,
        *,
        height: int,
        width: int,
        generator: torch.Generator | None = None,
    ) -> LingBotImageCondition:
        return prepare_ti2v_image_condition(
            image,
            height=height,
            width=width,
            vae=self.vae,
            vision_patch_size=self._vision_patch_size(),
            device=self.device,
            generator=generator,
        )

    def prepare_latents(
        self,
        num_frames: int,
        height: int,
        width: int,
        generator: torch.Generator | None,
        latents: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (1, self.transformer.config.in_channels, latent_frames, latent_height, latent_width)
        if latents is None:
            return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        if tuple(latents.shape) != shape:
            raise ValueError(f"`latents` shape must be {shape}, got {tuple(latents.shape)}.")
        return latents.to(device=device, dtype=torch.float32)

    def _dit_latent_to_vae(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)
        mean = mean.view(1, -1, 1, 1, 1)
        std_inv = std_inv.view(1, -1, 1, 1, 1)
        return latents.float() / std_inv + mean

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_device = _module_device(self.vae)
        vae_dtype = _module_dtype(self.vae)
        vae_latents = self._dit_latent_to_vae(latents).to(device=vae_device, dtype=torch.float32)
        if vae_latents.ndim == 5:
            vae_latents = vae_latents.contiguous(memory_format=torch.channels_last_3d)
        autocast_dtype = (
            vae_dtype if vae_device.type == "cuda" and vae_dtype in {torch.bfloat16, torch.float16} else None
        )
        with torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None):
            decoded = self.vae.decode(vae_latents)
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = frames.float().clamp_(-1, 1)
        frames = (frames + 1.0) / 2.0
        frames = frames.permute(0, 2, 3, 4, 1).cpu()
        return frames[0]

    @torch.no_grad()
    def _generate(
        self,
        *,
        prompt: str,
        mode: LingBotGenerationMode,
        input_image: Image.Image | None = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 6.0,
        shift: float = 3.0,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        output_type: str = "pt",
        cfg_parallel_group: Any | None = None,
        batch_cfg: bool = False,
        null_cond_clone_zero: bool = False,
        t_thresh: float | None = None,
        refiner_sigma_tail_steps: int = LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
        offload_vae_during_denoise: bool = False,
        **extra_args,
    ) -> torch.Tensor:
        del extra_args
        self.check_inputs(height, width, num_frames)
        device = self.device
        do_cfg = guidance_scale > 1.0
        image_condition = None
        prompt_images = None
        # Public image/mode combinations are validated by resolve_lingbot_mode.
        # Keep these checks as defensive invariants for direct internal calls.
        if mode is LingBotGenerationMode.TI2V:
            if input_image is None:
                raise ValueError("LingBot TI2V generation requires one input image.")
            image_condition = self.prepare_ti2v_image_condition(
                input_image,
                height=height,
                width=width,
                generator=generator,
            )
            prompt_images = [image_condition.vlm_image]
        elif input_image is not None:
            raise ValueError(f"LingBot {mode.value} generation does not accept an input image.")
        effective_batch_cfg = bool(batch_cfg)
        cfg_parallel = cfg_parallel_group is not None
        cfg_parallel_rank = 0
        if cfg_parallel:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("`cfg_parallel_group` requires an initialized process group.")
            if effective_batch_cfg:
                raise ValueError("`cfg_parallel_group` and `batch_cfg` are mutually exclusive.")
            if not do_cfg:
                raise ValueError("CFG parallel requires `guidance_scale > 1.0`.")
            cfg_parallel_rank = dist.get_rank(cfg_parallel_group)
            cfg_parallel_world_size = dist.get_world_size(cfg_parallel_group)
            if cfg_parallel_world_size != 2:
                raise ValueError(f"CFG parallel currently requires exactly 2 ranks, got {cfg_parallel_world_size}.")

        if prompt_embeds is not None:
            if prompt_mask is None:
                raise ValueError("`prompt_mask` is required when `prompt_embeds` is provided.")
            prompt_embeds = prompt_embeds.to(device=device)
            prompt_mask = prompt_mask.to(device=device)
        if negative_prompt_embeds is not None:
            if negative_prompt_mask is None:
                raise ValueError("`negative_prompt_mask` is required when `negative_prompt_embeds` is provided.")
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)
            negative_prompt_mask = negative_prompt_mask.to(device=device)

        negative_embeds = None
        negative_mask = None
        if cfg_parallel and cfg_parallel_rank == 1:
            if negative_prompt_embeds is not None:
                negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
            else:
                negative_embeds, negative_mask = self.encode_prompt(
                    negative_prompt,
                    images=prompt_images,
                    device=device,
                )
            prompt_embeds = prompt_mask = None
        else:
            if prompt_embeds is None:
                prompt_embeds, prompt_mask = self.encode_prompt(
                    prompt,
                    images=prompt_images,
                    device=device,
                )
            if do_cfg and not cfg_parallel:
                if null_cond_clone_zero:
                    negative_embeds = torch.zeros_like(prompt_embeds)
                    negative_mask = prompt_mask.clone()
                elif negative_prompt_embeds is not None:
                    negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
                else:
                    negative_embeds, negative_mask = self.encode_prompt(
                        negative_prompt,
                        images=prompt_images,
                        device=device,
                    )

        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        if image_condition is not None:
            latents = apply_clean_prefix(latents, image_condition.clean_latent)
        sigmas = _compute_refiner_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=num_inference_steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=refiner_sigma_tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        else:
            self.scheduler.set_timesteps(int(sigmas.shape[0]), device=device, sigmas=sigmas, shift=1.0)

        transformer_dtype = _module_dtype(self.transformer)
        vae_restore_device: torch.device | None = None
        vae_offloaded = False
        if offload_vae_during_denoise and output_type != "latent":
            vae_device = _module_device(self.vae)
            if vae_device.type == "cuda":
                self.vae.to("cpu")
                torch.accelerator.empty_cache()
                vae_restore_device = vae_device
                vae_offloaded = True

        cfg_latent_src = _group_global_rank(cfg_parallel_group, 0)
        cfg_uncond_src = _group_global_rank(cfg_parallel_group, 1)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            if cfg_parallel:
                dist.broadcast(latents, src=cfg_latent_src, group=cfg_parallel_group)
            timestep_batch = _transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
            latent_model_input = latents
            if cfg_parallel:
                if cfg_parallel_rank == 0:
                    branch_embeds = prompt_embeds
                    branch_mask = prompt_mask
                else:
                    branch_embeds = negative_embeds
                    branch_mask = negative_mask
                if branch_embeds is None:
                    raise RuntimeError("CFG branch embeddings were not initialized.")
                with _transformer_autocast(device, transformer_dtype):
                    branch_noise_pred = self.transformer(
                        latent_model_input,
                        timestep_batch,
                        branch_embeds.to(transformer_dtype),
                        encoder_attention_mask=branch_mask,
                        return_dict=False,
                    )[0].float()
                if cfg_parallel_rank == 0:
                    noise_pred = branch_noise_pred
                    noise_pred_uncond = torch.empty_like(noise_pred)
                else:
                    noise_pred_uncond = branch_noise_pred
                dist.broadcast(noise_pred_uncond, src=cfg_uncond_src, group=cfg_parallel_group)
                if cfg_parallel_rank != 0:
                    continue
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            else:
                if prompt_embeds is None:
                    raise RuntimeError("Prompt embeddings were not initialized.")
                prompt_model_input = prompt_embeds.to(transformer_dtype)
                if do_cfg and effective_batch_cfg:
                    if negative_embeds is None or negative_mask is None:
                        raise RuntimeError("Negative embeddings were not initialized for CFG.")
                    cfg_embeds, cfg_mask = _batch_cfg_prompt_inputs(
                        prompt_model_input,
                        prompt_mask,
                        negative_embeds.to(transformer_dtype),
                        negative_mask,
                        null_cond_clone_zero=False,
                    )
                    cfg_latents = torch.cat([latent_model_input, latent_model_input], dim=0)
                    cfg_timesteps = torch.cat([timestep_batch, timestep_batch], dim=0)
                    with _transformer_autocast(device, transformer_dtype):
                        noise_batched = self.transformer(
                            cfg_latents,
                            cfg_timesteps,
                            cfg_embeds,
                            encoder_attention_mask=cfg_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred, noise_pred_uncond = noise_batched.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                else:
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred = self.transformer(
                            latent_model_input,
                            timestep_batch,
                            prompt_model_input,
                            encoder_attention_mask=prompt_mask,
                            return_dict=False,
                        )[0].float()

                if do_cfg and not effective_batch_cfg:
                    if negative_embeds is None or negative_mask is None:
                        raise RuntimeError("Negative embeddings were not initialized for CFG.")
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            timestep_batch,
                            negative_embeds.to(transformer_dtype),
                            encoder_attention_mask=negative_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
                generator=generator,
            )[0]
            if image_condition is not None:
                latents = apply_clean_prefix(latents, image_condition.clean_latent)

        if cfg_parallel:
            dist.barrier(group=cfg_parallel_group)
            if cfg_parallel_rank != 0:
                return latents if output_type == "latent" else []

        if output_type == "latent":
            return latents
        if output_type in {"pt", "np"}:
            if vae_offloaded and vae_restore_device is not None:
                self.vae.to(device=vae_restore_device)
                torch.accelerator.empty_cache()
            decoded = self._decode_latents(latents)
            if mode is LingBotGenerationMode.T2I:
                if decoded.shape[0] != 1:
                    raise RuntimeError(
                        f"LingBot T2I decode expected exactly one frame, got shape {tuple(decoded.shape)}."
                    )
                return decoded[0]
            return decoded
        raise ValueError(f"Unsupported output_type: {output_type}")

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError(f"LingBotVideoPipeline only supports one request per batch, got {req.num_reqs}.")
        request = req.requests[0]
        sampling = request.sampling_params
        extra_args = dict(sampling.extra_args or {})

        default_shift = getattr(self.od_config, "flow_shift", None) or 3.0
        default_output_type = getattr(self.od_config, "output_type", None) or "pt"
        if default_output_type not in {"pt", "np", "latent"}:
            default_output_type = "pt"
        try:
            if sampling.num_outputs_per_prompt != 1:
                raise ValueError(
                    f"LingBotVideoPipeline only supports one output per prompt, got {sampling.num_outputs_per_prompt}."
                )
            request_config = normalize_lingbot_request(
                request,
                default_negative_prompt=self.default_negative_prompt,
                default_image_negative_prompt=self.default_image_negative_prompt,
                default_shift=default_shift,
                default_output_type=default_output_type,
            )
        except (TypeError, ValueError) as exc:
            raise OmniClientError(str(exc)) from exc

        generator = sampling.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(sampling.seed))

        sampling.height = request_config.height
        sampling.width = request_config.width
        sampling.num_frames = request_config.num_frames
        sampling.fps = request_config.fps
        sampling.frame_rate = float(request_config.fps)
        sampling.num_inference_steps = request_config.num_inference_steps
        sampling.guidance_scale = request_config.guidance_scale
        sampling.output_type = request_config.output_type

        frames = self._generate(
            prompt=request_config.prompt,
            mode=request_config.mode,
            input_image=request_config.input_image,
            negative_prompt=request_config.negative_prompt,
            height=request_config.height,
            width=request_config.width,
            num_frames=request_config.num_frames,
            num_inference_steps=request_config.num_inference_steps,
            guidance_scale=request_config.guidance_scale,
            shift=request_config.shift,
            generator=generator,
            latents=sampling.latents,
            output_type=request_config.output_type,
            batch_cfg=bool(extra_args.get("batch_cfg", False)),
            null_cond_clone_zero=bool(extra_args.get("null_cond_clone_zero", False)),
            offload_vae_during_denoise=bool(extra_args.get("offload_vae_during_denoise", False)),
            t_thresh=extra_args.get("t_thresh"),
            refiner_sigma_tail_steps=int(
                extra_args.get(
                    "refiner_sigma_tail_steps",
                    LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
                )
            ),
        )
        output_key = "image" if request_config.mode is LingBotGenerationMode.T2I else "video"
        return DiffusionOutput(output={output_key: frames})
