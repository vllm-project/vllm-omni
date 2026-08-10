# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import (
    from_pretrained_with_prefetch,
    prefetch_subfolders,
)
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
from vllm_omni.diffusion.models.lingbot_video.refiner_utils import (
    LingBotRefinerConfig,
    LingBotRefinerInputs,
    align_refiner_first_frame,
    compute_refiner_frame_budget,
    compute_refiner_frame_indices,
    compute_refiner_sigmas,
    normalize_lingbot_refiner_config,
    prepare_refiner_latent,
    resize_refiner_video,
)
from vllm_omni.diffusion.models.lingbot_video.request_utils import (
    LingBotExecutionOptions,
    LingBotGenerationMode,
    LingBotRefinerOptions,
    normalize_lingbot_execution_options,
    normalize_lingbot_request,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)

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


def _restore_vae_device_after_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Restore the VAE device after generation, including exceptional exits."""

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        vae = getattr(self, "vae", None)
        if vae is None:
            return func(self, *args, **kwargs)

        original_device = _module_device(vae)
        active_error: BaseException | None = None
        try:
            return func(self, *args, **kwargs)
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            current_device = _module_device(vae)
            if current_device != original_device:
                try:
                    vae.to(device=original_device)
                    torch.accelerator.empty_cache()
                except Exception:
                    if active_error is None:
                        raise
                    logger.exception(
                        "Failed to restore LingBot VAE to %s while propagating %s",
                        original_device,
                        type(active_error).__name__,
                    )

    return wrapped


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


def _validate_low_noise_sigmas(sigmas: np.ndarray, threshold: float | None = None) -> np.ndarray:
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Base low-noise sigma schedule must be a non-empty 1D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Base low-noise sigma schedule contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"Base low-noise sigma schedule values must be in [0, 1], got {arr.tolist()}")
    if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
        raise ValueError(f"Base low-noise sigma schedule must be strictly descending, got {arr.tolist()}")
    if threshold is not None and abs(float(arr[0]) - float(threshold)) > 1e-6:
        raise ValueError(f"Base low-noise sigma schedule must start at {float(threshold)}, got {float(arr[0])}")
    return arr


def _compute_low_noise_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    threshold: float | None,
    tail_steps: int = 0,
) -> np.ndarray | None:
    if threshold is None:
        return None
    t_value = float(threshold)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"Base low-noise threshold must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps or 0)
    if tail < 0:
        raise ValueError(f"base_sigma_tail_steps must be >= 0, got {tail}")

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
    return _validate_low_noise_sigmas(sigmas, t_value).astype(np.float32)


@dataclass(frozen=True)
class LingBotStageCondition:
    prompt_embeds: torch.Tensor | None
    prompt_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_mask: torch.Tensor | None
    image_condition: LingBotImageCondition | None
    cfg_parallel_group: Any | None
    cfg_parallel_rank: int
    clean_prefix: torch.Tensor | None = None


@dataclass(frozen=True)
class LingBotStageSettings:
    num_inference_steps: int
    guidance_scale: float
    shift: float
    batch_cfg: bool
    base_low_noise_threshold: float | None
    base_sigma_tail_steps: int


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

    def post_process_func(frames: torch.Tensor | dict[str, Any], sampling_params=None):
        output_key = None
        envelope = False
        metadata: dict[str, Any] = {}
        payload: dict[str, Any] | None = None
        if isinstance(frames, dict) and isinstance(frames.get("payload"), Mapping):
            envelope = True
            payload = dict(frames["payload"])
            raw_metadata = frames.get("metadata")
            if isinstance(raw_metadata, Mapping):
                metadata = dict(raw_metadata)
            output_keys = [key for key in ("image", "video") if key in payload]
            if len(output_keys) != 1:
                raise ValueError(
                    f"LingBot output payload must contain exactly one of 'image' or 'video', got {sorted(payload)!r}."
                )
            output_key = output_keys[0]
            frames = payload[output_key]
        elif isinstance(frames, dict):
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
        if envelope:
            assert payload is not None and output_key is not None
            payload[output_key] = frames
            return {"payload": payload, "metadata": metadata}
        return {output_key: frames} if output_key is not None else frames

    return post_process_func


class LingBotVideoPipeline(
    nn.Module,
    SupportImageInput,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Native vLLM-Omni entry for LingBot-Video checkpoints.

    The Base transformer is always loaded through the native weight stream. The
    optional official Refiner is startup-configured as a second native DiT while
    sharing the text encoder, processor, and VAE with the Base stage.
    """

    supports_step_execution: ClassVar[bool] = False
    max_outputs_per_prompt: ClassVar[int] = 1
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = [
        "_prepare_base_condition",
        "diffuse",
        "_decode_latents_internal",
        "_prepare_refiner_inputs",
        "_prepare_refiner_condition",
        "_diffuse_refiner",
    ]

    @staticmethod
    def _validate_cache_dit_configuration(od_config: OmniDiffusionConfig) -> None:
        cache_backend = str(getattr(od_config, "cache_backend", "none") or "none").lower()
        if cache_backend != "cache_dit":
            return

        parallel_config = od_config.parallel_config
        unsupported = []
        checks = (
            (
                parallel_config.pipeline_parallel_size > 1,
                "pipeline parallelism",
            ),
            (
                parallel_config.tensor_parallel_size > 1,
                "tensor parallelism",
            ),
            (
                parallel_config.enable_expert_parallel,
                "expert parallelism",
            ),
            (
                (parallel_config.sequence_parallel_size or 1) > 1,
                "sequence parallelism",
            ),
            (parallel_config.cfg_parallel_size > 1, "CFG parallelism"),
            (
                parallel_config.vae_patch_parallel_size > 1,
                "VAE patch parallelism",
            ),
            (parallel_config.use_hsdp, "HSDP"),
            (
                od_config.enable_distributed_layerwise_offload,
                "distributed layerwise offload",
            ),
        )
        unsupported.extend(name for enabled, name in checks if enabled)
        if unsupported:
            raise ValueError(
                "LingBot Cache-DiT does not support the following combinations: "
                + ", ".join(unsupported)
                + ". Use an unsharded DiT stage; CPU offload and ordinary "
                "layerwise offload remain supported."
            )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self._validate_cache_dit_configuration(od_config)
        self.od_config = od_config
        self._cache_dit_stage_refreshers: dict[
            str,
            Callable[[Any, int, bool], None],
        ] = {}
        self._execution_device = get_local_device()
        self.device = self._execution_device
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self.img_prompt_template = IMG_PROMPT_TEMPLATE
        self._crop_start: int | None = None

        model = od_config.model
        revision = od_config.revision
        local_files_only = os.path.isdir(model)
        load_device = torch.get_default_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = getattr(od_config, "model_config", None) or {}
        self.refiner_config = normalize_lingbot_refiner_config(
            model_config,
            base_model=model,
            base_revision=revision,
        )
        self.refiner_transformer: LingBotVideoTransformer3DModel | None = None
        self.refiner_scheduler: FlowUniPCMultistepScheduler | None = None
        self._dit_modules = ["transformer"]
        transformer_dtype = _dtype_from_name(model_config.get("transformer_dtype"), dtype)
        text_encoder_dtype = _dtype_from_name(model_config.get("text_encoder_dtype"), dtype)
        vae_dtype = _dtype_from_name(model_config.get("vae_dtype"), torch.float32)

        transformer_subfolder = str(model_config.get("transformer_subfolder", "transformer"))
        text_encoder_subfolder = str(model_config.get("text_encoder_subfolder", "text_encoder"))
        processor_subfolder = str(model_config.get("processor_subfolder", "processor"))
        vae_subfolder = str(model_config.get("vae_subfolder", "vae"))
        scheduler_subfolder = str(model_config.get("scheduler_subfolder", "scheduler"))

        component_subfolders = [
            transformer_subfolder,
            text_encoder_subfolder,
            processor_subfolder,
            vae_subfolder,
            scheduler_subfolder,
        ]
        refiner_model = self.refiner_config.model_dir
        refiner_revision = self.refiner_config.revision
        refiner_subfolder = self.refiner_config.transformer_subfolder
        refiner_local_files_only = bool(refiner_model and os.path.isdir(refiner_model))
        refiner_shares_source = bool(
            self.refiner_config.enabled and refiner_model == model and refiner_revision == revision
        )
        if refiner_shares_source:
            component_subfolders.append(refiner_subfolder)
        prefetch_subfolders(
            model,
            local_files_only=local_files_only,
            subfolders=component_subfolders,
            revision=revision,
        )
        if self.refiner_config.enabled and not refiner_shares_source:
            if refiner_model is None:
                raise RuntimeError("LingBot Refiner is enabled without a resolved model root.")
            prefetch_subfolders(
                refiner_model,
                local_files_only=refiner_local_files_only,
                subfolders=[refiner_subfolder, scheduler_subfolder],
                revision=refiner_revision,
            )

        text_encoder_kwargs: dict[str, Any] = {
            "dtype": text_encoder_dtype,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3VLForConditionalGeneration.from_pretrained,
            model,
            subfolder=text_encoder_subfolder,
            prefetch_list=component_subfolders,
            **text_encoder_kwargs,
        ).to(load_device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model,
            subfolder=processor_subfolder,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLWan.from_pretrained,
            model,
            subfolder=vae_subfolder,
            prefetch_list=component_subfolders,
            torch_dtype=vae_dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(load_device)
        self.scheduler = FlowUniPCMultistepScheduler.from_pretrained(
            model,
            subfolder=scheduler_subfolder,
            local_files_only=local_files_only,
            revision=revision,
        )

        transformer_config = LingBotVideoTransformer3DModel.load_config(
            model,
            subfolder=transformer_subfolder,
            revision=revision,
            local_files_only=local_files_only,
        )
        transformer_kwargs = get_transformer_config_kwargs(
            TransformerConfig.from_dict(transformer_config),
            LingBotVideoTransformer3DModel,
        )
        self.transformer = LingBotVideoTransformer3DModel(**transformer_kwargs)
        self.transformer.to(dtype=transformer_dtype)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder=transformer_subfolder,
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        if self.refiner_config.enabled:
            if refiner_model is None:
                raise RuntimeError("LingBot Refiner is enabled without a resolved model root.")
            try:
                refiner_transformer_config = LingBotVideoTransformer3DModel.load_config(
                    refiner_model,
                    subfolder=refiner_subfolder,
                    revision=refiner_revision,
                    local_files_only=refiner_local_files_only,
                )
                refiner_transformer_kwargs = get_transformer_config_kwargs(
                    TransformerConfig.from_dict(refiner_transformer_config),
                    LingBotVideoTransformer3DModel,
                )
                self.refiner_transformer = LingBotVideoTransformer3DModel(**refiner_transformer_kwargs)
                self.refiner_transformer.to(dtype=transformer_dtype)
                self.refiner_scheduler = FlowUniPCMultistepScheduler.from_pretrained(
                    refiner_model,
                    subfolder=scheduler_subfolder,
                    local_files_only=refiner_local_files_only,
                    revision=refiner_revision,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize LingBot Refiner from "
                    f"model={refiner_model!r}, subfolder={refiner_subfolder!r}, "
                    f"revision={refiner_revision!r}: {exc}"
                ) from exc
            self._dit_modules = ["transformer", "refiner_transformer"]
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=refiner_model,
                    subfolder=refiner_subfolder,
                    revision=refiner_revision,
                    prefix="refiner_transformer.",
                    fall_back_to_pt=True,
                )
            )

        self.set_progress_bar_config(disable=bool(model_config.get("quiet_progress", True)))
        self.default_negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.default_image_negative_prompt = DEFAULT_NEGATIVE_PROMPT_IMAGE
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=list(self._PROFILER_TARGETS),
            enable_diffusion_pipeline_profiler=getattr(od_config, "enable_diffusion_pipeline_profiler", False),
        )

    def _refresh_cache_dit_stage(
        self,
        transformer_name: str,
        num_inference_steps: int,
    ) -> None:
        refresh = getattr(self, "_cache_dit_stage_refreshers", {}).get(transformer_name)
        if refresh is not None:
            refresh(self, int(num_inference_steps), False)

    def _validate_cache_dit_request(
        self,
        request_config: Any,
        execution_options: LingBotExecutionOptions,
        *,
        is_dummy_run: bool = False,
    ) -> None:
        if is_dummy_run or not getattr(self, "_cache_dit_stage_refreshers", {}):
            return

        unsupported = []
        if request_config.guidance_scale <= 1.0:
            unsupported.append("Base guidance_scale <= 1")
        if execution_options.batch_cfg:
            unsupported.append("Base batch_cfg=true")

        refiner_options = execution_options.refiner
        if refiner_options.run:
            if refiner_options.guidance_scale <= 1.0:
                unsupported.append("Refiner guidance_scale <= 1")
            if refiner_options.batch_cfg:
                unsupported.append("Refiner batch_cfg=true")

        if unsupported:
            raise ValueError(
                "LingBot Cache-DiT currently requires sequential two-pass CFG "
                "for every enabled denoising stage; unsupported request options: " + ", ".join(unsupported)
            )

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            move_kwargs: dict[str, Any] = {
                "device": device,
                "non_blocking": non_blocking,
            }
            if memory_format is not None:
                move_kwargs["memory_format"] = memory_format
            super().to(**move_kwargs)
            self._execution_device = torch.device(device)
            self.device = self._execution_device
        elif memory_format is not None:
            super().to(memory_format=memory_format)

        # Keep each component's configured dtype. In particular, the VAE stays
        # FP32 and the Transformer applies its own BF16-bulk/FP32-islands policy.
        if dtype is not None:
            self.transformer.to(dtype=dtype, non_blocking=non_blocking)
            if self.refiner_transformer is not None:
                self.refiner_transformer.to(dtype=dtype, non_blocking=non_blocking)
        return self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

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

    def _vae_latent_to_dit(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)
        mean = mean.view(1, -1, 1, 1, 1)
        std_inv = std_inv.view(1, -1, 1, 1, 1)
        return (latents.float() - mean) * std_inv

    @torch.no_grad()
    def _encode_video_latent(
        self,
        video: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"LingBot VAE encode expects [B,C,T,H,W], got {tuple(video.shape)}.")
        vae_device = _module_device(self.vae)
        normalized = (video.to(device=vae_device, dtype=torch.float32) - 0.5) / 0.5
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=vae_device.type == "cuda",
        ):
            encoded = self.vae.encode(normalized)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample(generator)
        else:
            latents = encoded[0] if isinstance(encoded, tuple) else encoded
        return self._vae_latent_to_dit(latents).to(latents)

    @torch.no_grad()
    def _decode_latents_internal(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode to the canonical ``[B, C, T, H, W]`` tensor on the VAE device."""

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
        return (frames + 1.0) / 2.0

    @staticmethod
    def _format_output(decoded: torch.Tensor, mode: LingBotGenerationMode) -> torch.Tensor:
        """Convert the canonical tensor to the existing public CPU layout."""

        if decoded.ndim != 5:
            raise RuntimeError(f"LingBot decode expected [B, C, T, H, W], got {tuple(decoded.shape)}.")
        frames = decoded.permute(0, 2, 3, 4, 1).cpu()
        if frames.shape[0] != 1:
            raise RuntimeError(f"LingBot only supports decode batch size 1, got {frames.shape[0]}.")
        if mode is LingBotGenerationMode.T2I:
            if frames.shape[1] != 1:
                raise RuntimeError(f"LingBot T2I decode expected exactly one frame, got shape {tuple(frames.shape)}.")
            return frames[0, 0]
        return frames[0]

    def _offload_vae_for_denoise(self, *, enabled: bool) -> torch.device | None:
        if not enabled:
            return None
        vae_device = _module_device(self.vae)
        if vae_device.type != "cuda":
            return None
        self.vae.to("cpu")
        torch.accelerator.empty_cache()
        return vae_device

    def _restore_vae_for_decode(self, restore_device: torch.device | None) -> None:
        if restore_device is None:
            return
        self.vae.to(device=restore_device)
        torch.accelerator.empty_cache()

    @torch.no_grad()
    def _prepare_base_condition(
        self,
        *,
        prompt: str,
        mode: LingBotGenerationMode,
        input_image: Image.Image | None,
        negative_prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        generator: torch.Generator | None,
        prompt_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_mask: torch.Tensor | None,
        cfg_parallel_group: Any | None,
        batch_cfg: bool,
        null_cond_clone_zero: bool,
    ) -> LingBotStageCondition:
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

        cfg_parallel = cfg_parallel_group is not None
        cfg_parallel_rank = 0
        if cfg_parallel:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("`cfg_parallel_group` requires an initialized process group.")
            if batch_cfg:
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

        return LingBotStageCondition(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_mask=negative_mask,
            image_condition=image_condition,
            cfg_parallel_group=cfg_parallel_group,
            cfg_parallel_rank=cfg_parallel_rank,
            clean_prefix=(image_condition.clean_latent if image_condition is not None else None),
        )

    @torch.no_grad()
    def diffuse(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        generator: torch.Generator | None,
        latents: torch.Tensor | None,
        condition: LingBotStageCondition,
        settings: LingBotStageSettings,
    ) -> torch.Tensor:
        device = self.device
        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        if condition.clean_prefix is not None:
            latents = apply_clean_prefix(latents, condition.clean_prefix)

        sigmas = _compute_low_noise_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=settings.num_inference_steps,
            shift=settings.shift,
            threshold=settings.base_low_noise_threshold,
            tail_steps=settings.base_sigma_tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(
                settings.num_inference_steps,
                device=device,
                shift=settings.shift,
            )
        else:
            self.scheduler.set_timesteps(
                int(sigmas.shape[0]),
                device=device,
                sigmas=sigmas,
                shift=1.0,
            )

        self._refresh_cache_dit_stage(
            "transformer",
            len(self.scheduler.timesteps),
        )

        return self._run_denoise_stage(
            transformer=self.transformer,
            scheduler=self.scheduler,
            generator=generator,
            latents=latents,
            condition=condition,
            settings=settings,
        )

    def _run_denoise_stage(
        self,
        *,
        transformer: nn.Module,
        scheduler: FlowUniPCMultistepScheduler,
        generator: torch.Generator | None,
        latents: torch.Tensor,
        condition: LingBotStageCondition,
        settings: LingBotStageSettings,
    ) -> torch.Tensor:
        device = self.device
        transformer_dtype = _module_dtype(transformer)
        cfg_parallel = condition.cfg_parallel_group is not None
        do_cfg = settings.guidance_scale > 1.0
        cfg_latent_src = _group_global_rank(condition.cfg_parallel_group, 0)
        cfg_uncond_src = _group_global_rank(condition.cfg_parallel_group, 1)

        for timestep in self.progress_bar(scheduler.timesteps):
            if cfg_parallel:
                dist.broadcast(latents, src=cfg_latent_src, group=condition.cfg_parallel_group)
            timestep_batch = _transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
            if cfg_parallel:
                if condition.cfg_parallel_rank == 0:
                    branch_embeds = condition.prompt_embeds
                    branch_mask = condition.prompt_mask
                else:
                    branch_embeds = condition.negative_prompt_embeds
                    branch_mask = condition.negative_prompt_mask
                if branch_embeds is None:
                    raise RuntimeError("CFG branch embeddings were not initialized.")
                with _transformer_autocast(device, transformer_dtype):
                    branch_noise_pred = transformer(
                        latents,
                        timestep_batch,
                        branch_embeds.to(transformer_dtype),
                        encoder_attention_mask=branch_mask,
                        return_dict=False,
                    )[0].float()
                if condition.cfg_parallel_rank == 0:
                    noise_pred = branch_noise_pred
                    noise_pred_uncond = torch.empty_like(noise_pred)
                else:
                    noise_pred_uncond = branch_noise_pred
                dist.broadcast(noise_pred_uncond, src=cfg_uncond_src, group=condition.cfg_parallel_group)
                if condition.cfg_parallel_rank != 0:
                    continue
                noise_pred = noise_pred_uncond + settings.guidance_scale * (noise_pred - noise_pred_uncond)
            else:
                if condition.prompt_embeds is None:
                    raise RuntimeError("Prompt embeddings were not initialized.")
                prompt_model_input = condition.prompt_embeds.to(transformer_dtype)
                if do_cfg and settings.batch_cfg:
                    if condition.negative_prompt_embeds is None or condition.negative_prompt_mask is None:
                        raise RuntimeError("Negative embeddings were not initialized for CFG.")
                    cfg_embeds, cfg_mask = _batch_cfg_prompt_inputs(
                        prompt_model_input,
                        condition.prompt_mask,
                        condition.negative_prompt_embeds.to(transformer_dtype),
                        condition.negative_prompt_mask,
                        null_cond_clone_zero=False,
                    )
                    cfg_latents = torch.cat([latents, latents], dim=0)
                    cfg_timesteps = torch.cat([timestep_batch, timestep_batch], dim=0)
                    with _transformer_autocast(device, transformer_dtype):
                        noise_batched = transformer(
                            cfg_latents,
                            cfg_timesteps,
                            cfg_embeds,
                            encoder_attention_mask=cfg_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred, noise_pred_uncond = noise_batched.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + settings.guidance_scale * (noise_pred - noise_pred_uncond)
                else:
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred = transformer(
                            latents,
                            timestep_batch,
                            prompt_model_input,
                            encoder_attention_mask=condition.prompt_mask,
                            return_dict=False,
                        )[0].float()

                if do_cfg and not settings.batch_cfg:
                    if condition.negative_prompt_embeds is None or condition.negative_prompt_mask is None:
                        raise RuntimeError("Negative embeddings were not initialized for CFG.")
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred_uncond = transformer(
                            latents,
                            timestep_batch,
                            condition.negative_prompt_embeds.to(transformer_dtype),
                            encoder_attention_mask=condition.negative_prompt_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred = noise_pred_uncond + settings.guidance_scale * (noise_pred - noise_pred_uncond)

            latents = scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
                generator=generator,
            )[0]
            if condition.clean_prefix is not None:
                latents = apply_clean_prefix(latents, condition.clean_prefix)

        if cfg_parallel:
            dist.broadcast(
                latents,
                src=cfg_latent_src,
                group=condition.cfg_parallel_group,
            )
            dist.barrier(group=condition.cfg_parallel_group)
        return latents

    @torch.no_grad()
    def _prepare_refiner_inputs(
        self,
        *,
        base_video: torch.Tensor,
        source_fps: float,
        source_height: int,
        source_width: int,
        input_image: Image.Image | None,
        generator: torch.Generator | None,
        options: LingBotRefinerOptions,
    ) -> LingBotRefinerInputs:
        sample_frames = compute_refiner_frame_budget(
            int(base_video.shape[2]),
            source_fps,
            sample_fps=options.sample_fps,
            vae_temporal_factor=self.vae_scale_factor_temporal,
            max_frames=options.max_video_frames,
        )
        indices = compute_refiner_frame_indices(
            int(base_video.shape[2]),
            sample_frames,
            device=base_video.device,
        )
        sampled = base_video.index_select(2, indices)
        resized = resize_refiner_video(
            sampled,
            height=options.height,
            width=options.width,
        )

        # Random-consumption parity with the official runner is intentional:
        # handoff encode -> optional TI2V clean-frame encode -> Refiner noise.
        x_up = self._encode_video_latent(resized, generator=generator)
        clean_prefix = None
        if input_image is not None:
            clean_frame = align_refiner_first_frame(
                input_image,
                target_height=options.height,
                target_width=options.width,
                source_height=source_height,
                source_width=source_width,
            )
            clean_x0 = self._encode_video_latent(clean_frame, generator=generator)
            clean_prefix = clean_x0[:, :, :1].contiguous()
            x_up = apply_clean_prefix(x_up, clean_prefix)

        noise = randn_tensor(
            x_up.shape,
            generator=generator,
            device=x_up.device,
            dtype=x_up.dtype,
        )
        initial_latents = prepare_refiner_latent(
            x_up,
            noise,
            options.t_thresh,
        ).to(device=self.device, dtype=torch.float32)
        if clean_prefix is not None:
            clean_prefix = clean_prefix.to(device=self.device, dtype=torch.float32)
            initial_latents = apply_clean_prefix(initial_latents, clean_prefix)
        return LingBotRefinerInputs(
            latents=initial_latents,
            clean_prefix=clean_prefix,
            num_frames=sample_frames,
            source_fps=float(source_fps),
            sample_fps=options.sample_fps,
        )

    @torch.no_grad()
    def _prepare_refiner_condition(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        mode: LingBotGenerationMode,
        base_condition: LingBotStageCondition,
        guidance_scale: float,
        cfg_parallel_group: Any | None,
        options: LingBotRefinerOptions,
        clean_prefix: torch.Tensor | None,
    ) -> LingBotStageCondition:
        device = self.device
        do_cfg = guidance_scale > 1.0
        cfg_parallel = cfg_parallel_group is not None
        cfg_parallel_rank = 0
        if cfg_parallel:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("`cfg_parallel_group` requires an initialized process group.")
            if options.batch_cfg:
                raise ValueError("`cfg_parallel_group` and `refiner_batch_cfg` are mutually exclusive.")
            if not do_cfg:
                raise ValueError("Refiner CFG parallel requires `refiner_guidance_scale > 1.0`.")
            cfg_parallel_rank = dist.get_rank(cfg_parallel_group)
            if dist.get_world_size(cfg_parallel_group) != 2:
                raise ValueError("Refiner CFG parallel currently requires exactly 2 ranks.")

        prompt_embeds = None
        prompt_mask = None
        negative_embeds = None
        negative_mask = None

        can_reuse_positive = (
            mode is LingBotGenerationMode.T2V
            and base_condition.prompt_embeds is not None
            and base_condition.prompt_mask is not None
        )
        if cfg_parallel and cfg_parallel_rank == 1:
            if options.null_cond_clone_zero:
                positive, positive_mask = self.encode_prompt(prompt, images=None, device=device)
                negative_embeds = torch.zeros_like(positive)
                negative_mask = positive_mask.clone()
            else:
                negative_embeds, negative_mask = self.encode_prompt(
                    negative_prompt,
                    images=None,
                    device=device,
                )
        else:
            if can_reuse_positive:
                prompt_embeds = base_condition.prompt_embeds
                prompt_mask = base_condition.prompt_mask
            else:
                prompt_embeds, prompt_mask = self.encode_prompt(
                    prompt,
                    images=None,
                    device=device,
                )
            if do_cfg and not cfg_parallel:
                if options.null_cond_clone_zero:
                    negative_embeds = torch.zeros_like(prompt_embeds)
                    negative_mask = prompt_mask.clone()
                else:
                    negative_embeds, negative_mask = self.encode_prompt(
                        negative_prompt,
                        images=None,
                        device=device,
                    )

        return LingBotStageCondition(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_mask=negative_mask,
            image_condition=None,
            cfg_parallel_group=cfg_parallel_group,
            cfg_parallel_rank=cfg_parallel_rank,
            clean_prefix=clean_prefix,
        )

    @torch.no_grad()
    def _diffuse_refiner(
        self,
        *,
        inputs: LingBotRefinerInputs,
        condition: LingBotStageCondition,
        generator: torch.Generator | None,
        options: LingBotRefinerOptions,
    ) -> torch.Tensor:
        if self.refiner_transformer is None or self.refiner_scheduler is None:
            raise RuntimeError("LingBot Refiner was requested but was not initialized.")

        sigmas = compute_refiner_sigmas(
            sigma_max=float(self.refiner_scheduler.sigma_max),
            sigma_min=float(self.refiner_scheduler.sigma_min),
            num_inference_steps=options.num_inference_steps,
            shift=options.shift,
            t_thresh=options.t_thresh,
            tail_steps=options.sigma_tail_steps,
        )
        self.refiner_scheduler.set_timesteps(
            int(sigmas.shape[0]),
            device=self.device,
            sigmas=sigmas,
            shift=1.0,
        )
        self._refresh_cache_dit_stage(
            "refiner_transformer",
            len(self.refiner_scheduler.timesteps),
        )
        settings = LingBotStageSettings(
            num_inference_steps=options.num_inference_steps,
            guidance_scale=options.guidance_scale,
            shift=options.shift,
            batch_cfg=options.batch_cfg,
            base_low_noise_threshold=options.t_thresh,
            base_sigma_tail_steps=options.sigma_tail_steps,
        )
        return self._run_denoise_stage(
            transformer=self.refiner_transformer,
            scheduler=self.refiner_scheduler,
            generator=generator,
            latents=inputs.latents,
            condition=condition,
            settings=settings,
        )

    @_restore_vae_device_after_call
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
        fps: float = 24.0,
        num_inference_steps: int = 40,
        guidance_scale: float = 6.0,
        shift: float = 3.0,
        generator: torch.Generator | None = None,
        refiner_generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        output_type: str = "pt",
        cfg_parallel_group: Any | None = None,
        execution_options: LingBotExecutionOptions | None = None,
    ) -> torch.Tensor:
        self.check_inputs(height, width, num_frames)
        if output_type not in {"pt", "np", "latent"}:
            raise ValueError(f"Unsupported output_type: {output_type}")
        options = execution_options or LingBotExecutionOptions()
        run_refiner = options.refiner.run
        if run_refiner and mode is LingBotGenerationMode.T2I:
            raise ValueError("LingBot Refiner is only supported for video modes.")
        if run_refiner and (self.refiner_transformer is None or self.refiner_scheduler is None):
            raise RuntimeError("LingBot Refiner was requested but is not initialized.")

        # Capture the initial seed before Base condition/latent generation consume
        # its generator. Refiner stochastic work must use an independent stream.
        if run_refiner and refiner_generator is None:
            refiner_seed = generator.initial_seed() if generator is not None else int(torch.seed())
            refiner_generator = torch.Generator(device=self.device).manual_seed(refiner_seed)

        settings = LingBotStageSettings(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            batch_cfg=options.batch_cfg,
            base_low_noise_threshold=options.base_low_noise_threshold,
            base_sigma_tail_steps=options.base_sigma_tail_steps,
        )
        condition = self._prepare_base_condition(
            prompt=prompt,
            mode=mode,
            input_image=input_image,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            cfg_parallel_group=cfg_parallel_group,
            batch_cfg=options.batch_cfg,
            null_cond_clone_zero=options.null_cond_clone_zero,
        )

        vae_restore_device = self._offload_vae_for_denoise(enabled=options.offload_vae_during_denoise)
        latents = self.diffuse(
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
            latents=latents,
            condition=condition,
            settings=settings,
        )

        if not run_refiner:
            if condition.cfg_parallel_group is not None and condition.cfg_parallel_rank != 0:
                return latents if output_type == "latent" else []
            if output_type == "latent":
                return latents
            self._restore_vae_for_decode(vae_restore_device)
            decoded = self._decode_latents_internal(latents)
            return self._format_output(decoded, mode)

        self._restore_vae_for_decode(vae_restore_device)
        base_video = self._decode_latents_internal(latents)
        refiner_inputs = self._prepare_refiner_inputs(
            base_video=base_video,
            source_fps=fps,
            source_height=height,
            source_width=width,
            input_image=input_image if mode is LingBotGenerationMode.TI2V else None,
            generator=refiner_generator,
            options=options.refiner,
        )
        refiner_vae_restore_device = self._offload_vae_for_denoise(
            enabled=self.refiner_config.offload_vae_during_denoise
        )
        refiner_condition = self._prepare_refiner_condition(
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            base_condition=condition,
            guidance_scale=options.refiner.guidance_scale,
            cfg_parallel_group=cfg_parallel_group,
            options=options.refiner,
            clean_prefix=refiner_inputs.clean_prefix,
        )
        latents = self._diffuse_refiner(
            inputs=refiner_inputs,
            condition=refiner_condition,
            generator=refiner_generator,
            options=options.refiner,
        )

        if refiner_condition.cfg_parallel_group is not None and refiner_condition.cfg_parallel_rank != 0:
            return latents if output_type == "latent" else []
        if output_type == "latent":
            return latents

        self._restore_vae_for_decode(refiner_vae_restore_device)
        decoded = self._decode_latents_internal(latents)
        return self._format_output(decoded, LingBotGenerationMode.T2V)

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
            execution_options = normalize_lingbot_execution_options(
                extra_args,
                default_base_sigma_tail_steps=LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
                refiner_config=getattr(
                    self,
                    "refiner_config",
                    LingBotRefinerConfig(),
                ),
                mode=request_config.mode,
            )
            self._validate_cache_dit_request(
                request_config,
                execution_options,
                is_dummy_run=request.is_dummy_run(),
            )
        except (TypeError, ValueError) as exc:
            raise OmniClientError(str(exc)) from exc

        generator = sampling.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        if generator is None:
            seed = int(sampling.seed) if sampling.seed is not None else int(torch.seed())
            generator = torch.Generator(device=self.device).manual_seed(seed)

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
            fps=float(request_config.fps),
            num_inference_steps=request_config.num_inference_steps,
            guidance_scale=request_config.guidance_scale,
            shift=request_config.shift,
            generator=generator,
            latents=sampling.latents,
            output_type=request_config.output_type,
            execution_options=execution_options,
        )
        output_key = "image" if request_config.mode is LingBotGenerationMode.T2I else "video"
        output: dict[str, Any] = {output_key: frames}
        if execution_options.refiner.run:
            sample_frames = compute_refiner_frame_budget(
                request_config.num_frames,
                float(request_config.fps),
                sample_fps=execution_options.refiner.sample_fps,
                vae_temporal_factor=self.vae_scale_factor_temporal,
                max_frames=execution_options.refiner.max_video_frames,
            )
            sampling.height = execution_options.refiner.height
            sampling.width = execution_options.refiner.width
            sampling.num_frames = sample_frames
            sampling.fps = execution_options.refiner.output_fps
            sampling.frame_rate = float(execution_options.refiner.output_fps)
            output = {
                "payload": {"video": frames},
                "metadata": {
                    "video": {
                        "fps": execution_options.refiner.output_fps,
                        "refined": True,
                        "source_fps": float(request_config.fps),
                        "sample_fps": execution_options.refiner.sample_fps,
                        "sample_frames": sample_frames,
                    }
                },
            }
        stage_durations = self.stage_durations if getattr(self, "enable_diffusion_pipeline_profiler", False) else {}
        return DiffusionOutput(
            output=output,
            stage_durations=stage_durations,
        )
