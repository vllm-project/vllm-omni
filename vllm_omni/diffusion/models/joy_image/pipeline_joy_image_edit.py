# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Hugging Face Diffusers commit
# 23ba73e1d2079c4b89959484ed0ca1c22e7ef998:
# src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
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
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest


logger = init_logger(__name__)

JOY_DEFAULT_TARGET_AREA = 1024 * 1024
JOY_MAX_IMAGE_SEQ_LEN = 4096
JOY_PROMPT_TEMPLATE_START_IDX = 34
JOY_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are an assistant designed to generate high-quality images with the highest degree of "
    "image-text alignment based on textual prompts. <|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
)


def _get_model_path(model_name: str) -> str:
    if os.path.exists(model_name):
        return model_name
    return download_weights_from_hf_specific(
        model_name,
        None,
        ["vae/config.json", "transformer/config.json"],
        require_all=True,
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as config_file:
        return json.load(config_file)


def _get_transformer_config_kwargs_from_od_config(
    od_config: OmniDiffusionConfig,
) -> dict[str, Any]:
    tf_model_config = getattr(od_config, "tf_model_config", None)
    to_dict = getattr(tf_model_config, "to_dict", None)
    if not callable(to_dict):
        return {}
    return {key: value for key, value in to_dict().items() if not key.startswith("_")}


def _calculate_dimensions(
    target_area: int, ratio: float, alignment: int
) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = max(alignment, int(round(width / alignment) * alignment))
    height = max(alignment, int(round(height / alignment) * alignment))
    return width, height


def _fit_to_token_budget(height: int, width: int, alignment: int) -> tuple[int, int]:
    token_count = (height // alignment) * (width // alignment)
    if token_count <= JOY_MAX_IMAGE_SEQ_LEN:
        return height, width
    scale = math.sqrt(JOY_MAX_IMAGE_SEQ_LEN / token_count)
    height = max(alignment, int(math.floor((height * scale) / alignment) * alignment))
    width = max(alignment, int(math.floor((width * scale) / alignment) * alignment))
    while (height // alignment) * (width // alignment) > JOY_MAX_IMAGE_SEQ_LEN:
        if height >= width and height > alignment:
            height -= alignment
        elif width > alignment:
            width -= alignment
        else:
            break
    return height, width


def _normalize_explicit_size(
    height: int, width: int, alignment: int
) -> tuple[int, int]:
    height = max(alignment, int(round(height / alignment) * alignment))
    width = max(alignment, int(round(width / alignment) * alignment))
    token_count = (height // alignment) * (width // alignment)
    if token_count > JOY_MAX_IMAGE_SEQ_LEN:
        raise ValueError(
            "`height` and `width` exceed JoyAI-Image-Edit token budget: "
            f"({height} / {alignment}) * ({width} / {alignment}) = {token_count}, "
            f"max {JOY_MAX_IMAGE_SEQ_LEN}."
        )
    return height, width


def _pil_to_tensor(image: PIL.Image.Image, height: int, width: int) -> torch.Tensor:
    image = image.convert("RGB").resize((width, height), PIL.Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.unsqueeze(2).contiguous()


def _tensor_to_pil(images: torch.Tensor) -> list[PIL.Image.Image]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(
            f"Expected image tensor (B, C, H, W), got {tuple(images.shape)}."
        )
    images = images.detach().cpu().float().clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    images = (images * 255.0).round().to(torch.uint8)
    result = []
    for image in images:
        result.append(PIL.Image.fromarray(image.permute(1, 2, 0).numpy()))
    return result


def _extract_single_image(
    prompt: dict[str, Any] | str,
) -> PIL.Image.Image | torch.Tensor | np.ndarray:
    multi_modal_data = (
        prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else {}
    )
    raw_image = multi_modal_data.get("image")
    if raw_image is None or (isinstance(raw_image, list) and len(raw_image) == 0):
        raise ValueError(
            "Received no input image. JoyAI-Image-Edit requires exactly one input image."
        )
    if isinstance(raw_image, list):
        if len(raw_image) != 1:
            raise ValueError(
                "Received multiple input images. JoyAI-Image-Edit v1 supports exactly one image."
            )
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
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(
            np.uint8
        )
    return PIL.Image.fromarray(array).convert("RGB")


def get_joy_image_edit_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    model_path = _get_model_path(od_config.model)
    vae_config = _load_json(Path(model_path) / "vae" / "config.json")
    transformer_config = _load_json(Path(model_path) / "transformer" / "config.json")
    scale_factor_spatial = int(vae_config.get("scale_factor_spatial", 8))
    patch_size = transformer_config.get("patch_size", [1, 2, 2])
    alignment = scale_factor_spatial * int(patch_size[1])

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if len(request.prompts) != 1:
            raise ValueError(
                "JoyAI-Image-Edit v1 supports exactly one prompt per request."
            )
        for prompt_index, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                prompt = {"prompt": prompt}
            prompt.setdefault("additional_information", {})
            image = _extract_single_image(prompt)
            original_width, original_height = _image_size(image)
            height_is_set = request.sampling_params.height is not None
            width_is_set = request.sampling_params.width is not None
            if height_is_set != width_is_set:
                raise ValueError(
                    "JoyAI-Image-Edit requires both `height` and `width`, or neither."
                )
            if not height_is_set and not width_is_set:
                ratio = original_width / original_height
                width, height = _calculate_dimensions(
                    JOY_DEFAULT_TARGET_AREA, ratio, alignment
                )
                height, width = _fit_to_token_budget(height, width, alignment)
            else:
                height, width = _normalize_explicit_size(
                    request.sampling_params.height,
                    request.sampling_params.width,
                    alignment,
                )

            prompt_image = _to_pil_image(image).resize(
                (width, height), PIL.Image.Resampling.LANCZOS
            )
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
    def post_process_func(images: torch.Tensor) -> list[PIL.Image.Image]:
        return _tensor_to_pil(images)

    return post_process_func


def retrieve_latents(
    encoder_output: Any, generator: torch.Generator | None = None
) -> torch.Tensor:
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
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]
    _resident_modules: list[str] = []

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
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
        prefetch_subfolders(
            model,
            [
                "scheduler",
                "text_encoder",
                "vae",
                "tokenizer",
                "processor",
                "transformer",
            ],
            local_files_only=local_files_only,
        )
        from vllm_omni.diffusion.models.qwen3_vl.adapter_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model,
            subfolder="processor",
            local_files_only=local_files_only,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                model,
                subfolder="tokenizer",
                local_files_only=local_files_only,
            )
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        ).to(self.device)
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        ).to(self.device)
        transformer_config_kwargs = _get_transformer_config_kwargs_from_od_config(
            od_config
        )
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
        self.tokenizer_max_length = 1024
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
            return float(
                guidance_scale if guidance_provided else default_true_cfg_scale
            )
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
    def _extract_masked_hidden(
        hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> list[torch.Tensor]:
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
        split_hidden_states = [
            item[-max_sequence_length:] for item in split_hidden_states
        ]
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

        def hook(_module: nn.Module, args: tuple[Any, ...], _output: Any) -> None:
            captured["hidden_states"] = args[0]

        handle = self.text_encoder.model.language_model.layers[
            -1
        ].register_forward_hook(hook)
        try:
            self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=False,
            )
        finally:
            handle.remove()
        if "hidden_states" not in captured:
            raise RuntimeError(
                "Failed to capture Qwen3VL last-layer pre-norm hidden states."
            )
        return captured["hidden_states"]

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image | list[PIL.Image.Image],
        dtype: torch.dtype | None = None,
        prompt_name: str = "prompt",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = dtype or self.text_encoder.dtype
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        image_list = [image] if isinstance(image, PIL.Image.Image) else image
        texts = [self.prompt_template_encode.format(item) for item in prompt_list]
        model_inputs = self.processor(
            text=texts,
            images=image_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        hidden_states = self._get_last_layer_pre_norm_hidden(model_inputs)
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [
            item[self.prompt_template_encode_start_idx :]
            for item in split_hidden_states
        ]
        prompt_embeds, prompt_mask = self._pad_prompt_embeds(
            split_hidden_states,
            self.tokenizer_max_length,
        )
        return prompt_embeds.to(dtype=dtype, device=self.device), prompt_mask.to(
            device=self.device
        )

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image | list[PIL.Image.Image],
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        prompt_name: str = "prompt",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt_list,
                image,
                prompt_name=prompt_name,
            )
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask must be provided with prompt_embeds.")
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        return prompt_embeds, prompt_embeds_mask

    def _latent_stats(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                retrieve_latents(
                    self.vae.encode(image[item_index : item_index + 1]),
                    generator=generator[item_index],
                )
                for item_index in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator
            )
        latents_mean, latents_std = self._latent_stats(
            image_latents.device, image_latents.dtype
        )
        return (image_latents - latents_mean) / latents_std

    def prepare_latents(
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
            raise ValueError(
                f"Generator list length {len(generator)} must match effective batch size {batch_size}."
            )

        image = image.to(device=device, dtype=dtype)
        image_latents = (
            image
            if image.shape[1] == self.latent_channels
            else self._encode_vae_image(image, generator)
        )
        expected_image_tail = (
            num_channels_latents,
            1,
            latent_height,
            latent_width,
        )
        if (
            image_latents.ndim != 5
            or tuple(image_latents.shape[1:]) != expected_image_tail
        ):
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
            raise ValueError(
                f"Image batch size {image_latents.shape[0]} exceeds effective batch size {batch_size}."
            )

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
                    "Joy noise latents must have shape "
                    f"{expected_noise_shape}, got {tuple(noise_latents.shape)}."
                )
        return torch.cat([image_latents, noise_latents], dim=1), image_latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_items, channels, frames, height, width = latents.shape
        flat_latents = latents.reshape(
            batch_size * num_items, channels, frames, height, width
        )
        latents_mean, latents_std = self._latent_stats(
            flat_latents.device, flat_latents.dtype
        )
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
        if len(req.prompts) != 1:
            raise ValueError(
                "JoyImageEditPipeline supports exactly one prompt per request."
            )
        first_prompt = req.prompts[0]
        prompt = (
            first_prompt
            if isinstance(first_prompt, str)
            else (first_prompt.get("prompt") or "")
        )
        additional_information = (
            {}
            if isinstance(first_prompt, str)
            else first_prompt.get("additional_information", {})
        )
        image = additional_information.get("image_tensor")
        prompt_image = additional_information.get("prompt_image")
        height = additional_information.get("height") or req.sampling_params.height
        width = additional_information.get("width") or req.sampling_params.width
        if image is None or prompt_image is None or height is None or width is None:
            raise ValueError(
                "JoyImageEditPipeline requires preprocessed image information in the request."
            )

        num_inference_steps = req.sampling_params.num_inference_steps or 50
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = max(req.sampling_params.num_outputs_per_prompt, 1)
        true_cfg_scale = self.resolve_effective_true_cfg_scale(req)
        do_true_cfg = true_cfg_scale > 1.0
        self.check_cfg_parallel_validity(true_cfg_scale)
        negative_prompt = (
            None
            if isinstance(first_prompt, str)
            else first_prompt.get("negative_prompt")
        )
        if do_true_cfg and negative_prompt is None:
            negative_prompt = ""

        self._current_timestep = None
        self._interrupt = False

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            prompt_image,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_name="prompt",
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt or "",
                prompt_image,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_name="negative_prompt",
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        batch_size = prompt_embeds.shape[0]
        if req.sampling_params.seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(
                req.sampling_params.seed
            )

        latents, image_latents = self.prepare_latents(
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
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
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
            cfg_normalize=req.sampling_params.cfg_normalize,
        )
        if output_type == "latent":
            return DiffusionOutput(output=latents[:, -1].detach().cpu())
        images = self._decode_latents(latents)
        return DiffusionOutput(output=images)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)
