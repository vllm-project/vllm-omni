# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Hugging Face Diffusers commit
# 23ba73e1d2079c4b89959484ed0ca1c22e7ef998:
# src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
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
from vllm_omni.diffusion.models.joy_image.cfg_parallel import (
    JoyImageEditCFGParallelMixin,
)
from vllm_omni.diffusion.models.joy_image.joy_image_edit_transformer import (
    JoyImageEditTransformer3DModel,
)
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest


JOY_MAX_IMAGE_SEQ_LEN = 4096
# Reference: Hugging Face Diffusers commit 23ba73e1d2079c4b89959484ed0ca1c22e7ef998,
# src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py.
JOY_BUCKETS = [
    (512, 1792),
    (512, 1856),
    (512, 1920),
    (512, 1984),
    (512, 2048),
    (576, 1600),
    (576, 1664),
    (576, 1728),
    (576, 1792),
    (640, 1472),
    (640, 1536),
    (640, 1600),
    (704, 1344),
    (704, 1408),
    (704, 1472),
    (768, 1216),
    (768, 1280),
    (768, 1344),
    (832, 1152),
    (832, 1216),
    (896, 1088),
    (896, 1152),
    (960, 1024),
    (960, 1088),
    (1024, 960),
    (1024, 1024),
    (1088, 896),
    (1088, 960),
    (1152, 832),
    (1152, 896),
    (1216, 768),
    (1216, 832),
    (1280, 768),
    (1344, 704),
    (1344, 768),
    (1408, 704),
    (1472, 640),
    (1472, 704),
    (1536, 640),
    (1600, 576),
    (1600, 640),
    (1664, 576),
    (1728, 576),
    (1792, 512),
    (1792, 576),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
]
JOY_PROMPT_TEMPLATE_START_IDX = 34
# The upstream template is named "multiple_images", but this port currently
# inserts a single reference-image token and rejects multi-image prompts.
JOY_PROMPT_TEMPLATE = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "{}<|im_start|>assistant\n"
)
JOY_VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def _get_model_path(model_name: str) -> str:
    if os.path.exists(model_name):
        return model_name
    return download_weights_from_hf_specific(
        model_name,
        None,
        ["vae/config.json", "transformer/config.json"],
        require_all=True,
    )


def _get_transformer_config_kwargs_from_od_config(
    od_config: OmniDiffusionConfig,
) -> dict[str, Any]:
    tf_model_config = getattr(od_config, "tf_model_config", None)
    to_dict = getattr(tf_model_config, "to_dict", None)
    if not callable(to_dict):
        return {}
    return {key: value for key, value in to_dict().items() if not key.startswith("_")}


def _should_defer_component_device_placement(
    od_config: OmniDiffusionConfig,
) -> bool:
    return bool(
        getattr(od_config, "enable_cpu_offload", False) or getattr(od_config, "enable_layerwise_offload", False)
    )


def _raise_if_unsupported_hsdp(od_config: OmniDiffusionConfig) -> None:
    parallel_config = getattr(od_config, "parallel_config", None)
    if getattr(parallel_config, "use_hsdp", False):
        raise ValueError(
            "JoyImageEditPipeline does not support HSDP yet. "
            "Please disable `use_hsdp`, or add `_hsdp_shard_conditions` and HSDP parity tests."
        )


def _is_dummy_request(req: Any) -> bool:
    is_dummy_run = getattr(req, "is_dummy_run", None)
    if callable(is_dummy_run):
        return bool(is_dummy_run())
    return getattr(req, "request_id", None) == DUMMY_DIFFUSION_REQUEST_ID


def _format_qwen_multimodal_prompt(prompt: str) -> str:
    prompt = f"<image>\n{prompt}"
    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n"
    return prompt.replace("<image>\n", JOY_VISION_TOKEN)


def _find_best_bucket(height: int, width: int) -> tuple[int, int]:
    target_ratio = height / width
    return min(JOY_BUCKETS, key=lambda size: abs(size[0] / size[1] - target_ratio))


def _resize_center_crop(
    image: PIL.Image.Image,
    height: int,
    width: int,
) -> PIL.Image.Image:
    image = image.convert("RGB")
    source_width, source_height = image.size
    scale = max(height / source_height, width / source_width)
    resized_height = math.ceil(source_height * scale)
    resized_width = math.ceil(source_width * scale)
    image = image.resize((resized_width, resized_height), PIL.Image.Resampling.BILINEAR)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _pil_to_tensor(image: PIL.Image.Image, height: int, width: int) -> torch.Tensor:
    image = _resize_center_crop(image, height, width)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.unsqueeze(2).contiguous()


def _extract_single_image(
    prompt: dict[str, Any] | str,
) -> PIL.Image.Image | torch.Tensor | np.ndarray:
    multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else {}
    raw_image = multi_modal_data.get("image")
    if raw_image is None or (isinstance(raw_image, list) and len(raw_image) == 0):
        raise ValueError("Received no input image. JoyAI-Image-Edit requires exactly one input image.")
    if isinstance(raw_image, list):
        if len(raw_image) != 1:
            raise ValueError("Received multiple input images. JoyAI-Image-Edit v1 supports exactly one image.")
        raw_image = raw_image[0]
    if isinstance(raw_image, str):
        return PIL.Image.open(raw_image)
    return cast(PIL.Image.Image | torch.Tensor | np.ndarray, raw_image)


def _image_size(image: PIL.Image.Image | torch.Tensor | np.ndarray) -> tuple[int, int]:
    if isinstance(image, PIL.Image.Image):
        return image.size
    if isinstance(image, torch.Tensor):
        height, width = image.shape[-2:]
        return int(width), int(height)
    height, width = image.shape[:2]
    return int(width), int(height)


def _to_pil_image(
    image: PIL.Image.Image | torch.Tensor | np.ndarray,
) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.float().numpy()
    else:
        array = image
    if array.dtype != np.uint8:
        if array.min() < 0:
            array = (array + 1.0) / 2.0
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(array).convert("RGB")


def _cast_floating_model_inputs(model_inputs: Any, dtype: torch.dtype) -> Any:
    for key in ("pixel_values", "pixel_values_videos"):
        get_value = getattr(model_inputs, "get", None)
        tensor = get_value(key) if callable(get_value) else getattr(model_inputs, key, None)
        if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
            model_inputs[key] = tensor.to(dtype=dtype)
    return model_inputs


# Normalize the single reference image into a supported Joy bucket before
# forward(); default and explicitly requested sizes are both snapped to the
# nearest bucket, so the final output size can differ from user-provided
# height/width while keeping VAE/DiT latent grids within trained resolutions.
def get_joy_image_edit_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if len(request.prompts) != 1:
            raise ValueError("JoyAI-Image-Edit v1 supports exactly one prompt per request.")
        for prompt_index, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                prompt = {"prompt": prompt}
            prompt.setdefault("additional_information", {})
            image = _extract_single_image(prompt)
            original_width, original_height = _image_size(image)
            height_is_set = request.sampling_params.height is not None
            width_is_set = request.sampling_params.width is not None
            if height_is_set != width_is_set:
                raise ValueError("JoyAI-Image-Edit requires both `height` and `width`, or neither.")
            if not height_is_set and not width_is_set:
                height, width = _find_best_bucket(original_height, original_width)
            else:
                height, width = _find_best_bucket(
                    request.sampling_params.height,
                    request.sampling_params.width,
                )

            prompt_image = _resize_center_crop(_to_pil_image(image), height, width)
            image_tensor = _pil_to_tensor(prompt_image, height, width)
            prompt["additional_information"].update(
                {
                    "image_tensor": image_tensor,
                    "prompt_image": prompt_image,
                    "height": height,
                    "width": width,
                    "original_size": (original_width, original_height),
                    "resized_size": (width, height),
                }
            )
            request.sampling_params.height = height
            request.sampling_params.width = width
            request.prompts[prompt_index] = prompt
        return request

    return pre_process_func


def get_joy_image_edit_post_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[torch.Tensor], list[PIL.Image.Image]]:
    image_processor = VaeImageProcessor()

    def post_process_func(images: torch.Tensor) -> list[PIL.Image.Image]:
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func


def retrieve_latents(encoder_output: Any, generator: torch.Generator | None = None) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator=generator)
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided VAE encoder output.")


class JoyImageEditPipeline(
    nn.Module,
    SupportImageInput,
    JoyImageEditCFGParallelMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    _dit_modules = ["transformer"]
    # Joy's VAE is large enough that model-level CPU offload must treat it as
    # mutually exclusive with the DiT, instead of pinning it on GPU.
    _encoder_modules = ["text_encoder", "vae"]
    _vae_modules: list[str] = []
    _resident_modules: list[str] = []

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Joy has no `_hsdp_shard_conditions` yet, so fail before generic
        # HSDP setup reaches the transformer.
        _raise_if_unsupported_hsdp(od_config)
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        joy_subfolders = [
            "scheduler",
            "text_encoder",
            "vae",
            "tokenizer",
            "processor",
            "transformer",
        ]
        prefetch_subfolders(
            model,
            joy_subfolders,
            local_files_only=local_files_only,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.processor = from_pretrained_with_prefetch(
            Qwen3VLProcessor.from_pretrained,
            model,
            subfolder="processor",
            prefetch_list=joy_subfolders,
            local_files_only=local_files_only,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                model,
                subfolder="tokenizer",
                local_files_only=local_files_only,
            )
        defer_component_device_placement = _should_defer_component_device_placement(od_config)
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3VLForConditionalGeneration.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=joy_subfolders,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        if not defer_component_device_placement:
            self.text_encoder = self.text_encoder.to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=joy_subfolders,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        if not defer_component_device_placement:
            self.vae = self.vae.to(self.device)
        transformer_config_kwargs = _get_transformer_config_kwargs_from_od_config(od_config)
        if transformer_config_kwargs:
            self.transformer = JoyImageEditTransformer3DModel(
                od_config=od_config,
                **transformer_config_kwargs,
            )
        else:
            model_path = _get_model_path(model)
            transformer_config_path = Path(model_path) / "transformer" / "config.json"
            if not transformer_config_path.exists():
                raise FileNotFoundError(
                    "JoyAI-Image-Edit requires transformer/config.json to "
                    f"instantiate the DiT; could not find {transformer_config_path}."
                )
            self.transformer = JoyImageEditTransformer3DModel.from_config_file(
                transformer_config_path,
                od_config=od_config,
            )

        self.vae_scale_factor = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        self.latent_channels = int(getattr(self.vae.config, "z_dim", 16))
        self.tokenizer_max_length = JOY_MAX_IMAGE_SEQ_LEN
        self.prompt_template_encode = JOY_PROMPT_TEMPLATE
        self.prompt_template_encode_start_idx = JOY_PROMPT_TEMPLATE_START_IDX
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @staticmethod
    def resolve_effective_true_cfg_scale(
        req: OmniDiffusionRequest,
        default_true_cfg_scale: float = 4.0,
    ) -> float:
        sampling_params = req.sampling_params
        true_cfg_scale = sampling_params.true_cfg_scale
        guidance_provided = sampling_params.guidance_scale_provided
        guidance_scale = sampling_params.guidance_scale
        if true_cfg_scale is None:
            return float(guidance_scale if guidance_provided else default_true_cfg_scale)
        guidance_is_disabled_default = math.isclose(float(guidance_scale), 1.0)
        if (
            guidance_provided
            and not guidance_is_disabled_default
            and not math.isclose(float(guidance_scale), float(true_cfg_scale))
        ):
            raise ValueError(
                "JoyAI-Image-Edit treats `guidance_scale` as a Diffusers compatibility alias for "
                "`true_cfg_scale`. Provide only one value, or provide matching values."
            )
        return float(true_cfg_scale)

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    @staticmethod
    def _pad_prompt_embeds(
        split_hidden_states: list[torch.Tensor],
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than 0.")
        split_hidden_states = [item[-max_sequence_length:] for item in split_hidden_states]
        max_seq_len = max(item.shape[0] for item in split_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [item, item.new_zeros(max_seq_len - item.shape[0], item.shape[1])],
                    dim=0,
                )
                for item in split_hidden_states
            ],
            dim=0,
        )
        prompt_mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(item.shape[0], dtype=torch.long, device=item.device),
                        torch.zeros(
                            max_seq_len - item.shape[0],
                            dtype=torch.long,
                            device=item.device,
                        ),
                    ]
                )
                for item in split_hidden_states
            ],
            dim=0,
        )
        return prompt_embeds, prompt_mask

    def _get_last_layer_pre_norm_hidden(
        self,
        model_inputs: Any,
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

        handle = self.text_encoder.model.language_model.layers[-1].register_forward_hook(hook)
        try:
            input_items = getattr(model_inputs, "items", None)
            if callable(input_items):
                text_encoder_inputs = dict(input_items())
            else:
                text_encoder_inputs = dict(vars(model_inputs))
            text_encoder_inputs["output_hidden_states"] = False
            self.text_encoder(**text_encoder_inputs)
        finally:
            handle.remove()
        if "hidden_states" not in captured:
            raise RuntimeError("Failed to capture Qwen3VL last-layer pre-norm hidden states.")
        return captured["hidden_states"]

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image | list[PIL.Image.Image],
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = dtype or self.text_encoder.dtype
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        image_list = [image] if isinstance(image, PIL.Image.Image) else image
        texts = [_format_qwen_multimodal_prompt(item) for item in prompt_list]
        texts = [self.prompt_template_encode.format(item) for item in texts]
        model_inputs = self.processor(
            text=texts,
            images=image_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        model_inputs = _cast_floating_model_inputs(model_inputs, dtype)
        hidden_states = self._get_last_layer_pre_norm_hidden(model_inputs)
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [item[self.prompt_template_encode_start_idx :] for item in split_hidden_states]
        prompt_embeds, prompt_mask = self._pad_prompt_embeds(
            split_hidden_states,
            self.tokenizer_max_length,
        )
        return prompt_embeds.to(dtype=dtype, device=self.device), prompt_mask.to(device=self.device)

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image | list[PIL.Image.Image],
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt_list,
                image,
            )
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask must be provided with prompt_embeds.")
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds, prompt_embeds_mask

    def _latent_stats(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(
                device=device,
                dtype=dtype,
            )
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(
                device=device,
                dtype=dtype,
            )
        )
        return latents_mean, latents_std

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
    ):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[item_index : item_index + 1]))
                for item_index in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image))
        latents_mean, latents_std = self._latent_stats(image_latents.device, image_latents.dtype)
        return (image_latents - latents_mean) / latents_std

    def _prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_height = int(height) // self.vae_scale_factor
        latent_width = int(width) // self.vae_scale_factor
        expected_noise_shape = (
            batch_size,
            1,
            num_channels_latents,
            1,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} must match effective batch size {batch_size}.")

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image, generator)
        image_latents = image_latents.to(device=device, dtype=dtype)
        expected_image_tail = (
            num_channels_latents,
            1,
            latent_height,
            latent_width,
        )
        if image_latents.ndim != 5 or tuple(image_latents.shape[1:]) != expected_image_tail:
            raise ValueError(
                "Joy image latents must have shape "
                f"(B, {expected_image_tail[0]}, {expected_image_tail[1]}, "
                f"{expected_image_tail[2]}, {expected_image_tail[3]}), "
                f"got {tuple(image_latents.shape)}."
            )
        image_latents = image_latents.unsqueeze(1)
        if batch_size > image_latents.shape[0]:
            if batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate image batch size {image_latents.shape[0]} to effective batch {batch_size}."
                )
            repeats = batch_size // image_latents.shape[0]
            image_latents = image_latents.repeat_interleave(repeats, dim=0)
        elif batch_size < image_latents.shape[0]:
            raise ValueError(f"Image batch size {image_latents.shape[0]} exceeds effective batch size {batch_size}.")

        if latents is None:
            noise_latents = randn_tensor(
                expected_noise_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            noise_latents = latents.to(device=device, dtype=dtype)
            if noise_latents.ndim == 5:
                noise_latents = noise_latents.unsqueeze(1)
            if tuple(noise_latents.shape) != expected_noise_shape:
                raise ValueError(
                    f"Joy noise latents must have shape {expected_noise_shape}, got {tuple(noise_latents.shape)}."
                )
        return torch.cat([image_latents, noise_latents], dim=1), image_latents

    def _offload_transformer_if_deferred(self) -> None:
        od_config = getattr(self, "od_config", None)
        if od_config is None or not _should_defer_component_device_placement(od_config):
            return

        transformer = getattr(self, "transformer", None)
        if isinstance(transformer, nn.Module):
            pin_memory = getattr(od_config, "pin_cpu_memory", True)
            SequentialOffloadHook._move_params(
                transformer,
                torch.device("cpu"),
                non_blocking=pin_memory,
                pin_memory=pin_memory,
            )
            current_omni_platform.empty_cache()

    def _prepare_vae_for_decode(self) -> None:
        od_config = getattr(self, "od_config", None)
        if od_config is None or not _should_defer_component_device_placement(od_config):
            return

        self._offload_transformer_if_deferred()
        SequentialOffloadHook._move_params(self.vae, self.device, non_blocking=False)
        current_omni_platform.synchronize()

    def _vae_device(self, fallback: torch.device) -> torch.device:
        try:
            return next(self.vae.parameters()).device
        except (AttributeError, StopIteration):
            return fallback

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self._prepare_vae_for_decode()
        batch_size, num_items, channels, frames, height, width = latents.shape
        flat_latents = latents.reshape(batch_size * num_items, channels, frames, height, width)
        flat_latents = flat_latents.to(device=self._vae_device(flat_latents.device))
        latents_mean, latents_std = self._latent_stats(flat_latents.device, flat_latents.dtype)
        flat_latents = flat_latents * latents_std + latents_mean
        decoded = self.vae.decode(flat_latents, return_dict=False)[0]
        decoded = decoded.reshape(
            batch_size,
            num_items,
            frames,
            decoded.shape[1],
            decoded.shape[-2],
            decoded.shape[-1],
        )
        return decoded[:, -1, 0]

    @property
    def interrupt(self) -> bool:
        return getattr(self, "_interrupt", False)

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pil",
    ) -> DiffusionOutput:
        # Step 1: read the preprocessed single-image edit request.
        if len(req.prompts) != 1:
            raise ValueError("JoyImageEditPipeline supports exactly one prompt per request.")
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        additional_information = {} if isinstance(first_prompt, str) else first_prompt.get("additional_information", {})
        image = additional_information.get("image_tensor")
        prompt_image = additional_information.get("prompt_image")
        height = additional_information.get("height") or req.sampling_params.height
        width = additional_information.get("width") or req.sampling_params.width
        if image is None or prompt_image is None or height is None or width is None:
            raise ValueError("JoyImageEditPipeline requires preprocessed image information in the request.")

        num_inference_steps = req.sampling_params.num_inference_steps or 50
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = max(req.sampling_params.num_outputs_per_prompt, 1)
        true_cfg_scale = self.resolve_effective_true_cfg_scale(req)
        do_true_cfg = true_cfg_scale > 1.0
        self.check_cfg_parallel_validity(true_cfg_scale)
        negative_prompt = "" if isinstance(first_prompt, str) else first_prompt.get("negative_prompt") or ""

        self._current_timestep = None
        self._interrupt = False

        # Step 2: encode text plus the reference image through Qwen3-VL
        # (including its vision/ViT path) to produce DiT conditioning.
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            prompt_image,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                prompt_image,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        batch_size = prompt_embeds.shape[0]
        if req.sampling_params.seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        # Step 3: VAE-encode the reference image and create the target noise
        # latent; the latent stack is [reference image latents, target noise].
        latents, image_latents = self._prepare_latents(
            image=image,
            batch_size=batch_size,
            num_channels_latents=self.latent_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=generator,
            latents=latents,
        )
        # Step 4: build the scheduler timestep/sigma schedule.
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        # Step 5: run the DiT denoising loop. The reference image latents stay
        # fixed while the target noise latent is updated each scheduler step.
        latents = self.diffuse(
            latents=latents,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            timesteps=self.scheduler.timesteps,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=True,
        )
        if output_type == "latent" or _is_dummy_request(req):
            self._offload_transformer_if_deferred()
            return DiffusionOutput(output=latents[:, -1].detach().cpu())
        # Step 6: decode only the target latent back to image space.
        images = self._decode_latents(latents)
        return DiffusionOutput(output=images)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)
