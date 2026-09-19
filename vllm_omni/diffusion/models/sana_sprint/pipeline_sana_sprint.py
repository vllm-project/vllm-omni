# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from Diffusers' Apache-2.0 licensed SanaSprintPipeline.

import os
from collections.abc import Callable, Iterable

import numpy as np
import torch
from diffusers import AutoencoderDC, SCMScheduler
from diffusers.image_processor import PixArtImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers import Gemma2Model, GemmaTokenizerFast
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .transformer_sana_sprint import SanaSprintTransformer2DModel

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

COMPLEX_HUMAN_INSTRUCTION = [
    (
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions "
        "suitable for image generation. Evaluate the level of detail in the user prompt:"
    ),
    (
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and"
        " spatial relationships to create vivid and concrete scenes."
    ),
    ("- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating."),
    ("Here are examples of how to transform or refine prompts:"),
    (
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round "
        "shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red "
        "flowers."
    ),
    (
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
        "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
        "passing by towering glass skyscrapers."
    ),
    (
        "Please generate only the enhanced description for the prompt below and avoid including any "
        "additional commentary or evaluations:"
    ),
    ("User Prompt: "),
]


def get_sana_sprint_post_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[torch.Tensor], list[Image.Image] | np.ndarray | torch.Tensor]:
    processor = PixArtImageProcessor(vae_scale_factor=32)

    def post_process(images: torch.Tensor) -> list[Image.Image] | np.ndarray | torch.Tensor:
        if od_config.output_type == "latent":
            return images
        return processor.postprocess(images.cpu(), output_type=od_config.output_type)

    return post_process


class SanaSprintPipeline(nn.Module, ProgressBarMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        parallel = od_config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.sequence_parallel_size != 1
            or parallel.cfg_parallel_size != 1
            or parallel.vae_patch_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.text_encoder_tp_size != 1
            or parallel.use_hsdp
        ):
            raise ValueError("Sana-Sprint currently supports single-device model execution only.")
        if od_config.quantization_config is not None or od_config.cache_backend != "none" or od_config.lora_path:
            raise ValueError("Sana-Sprint does not yet support quantization, diffusion caching, or LoRA.")
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model
        local = os.path.isdir(model)
        subfolders = ["scheduler", "tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(model, subfolders, local_files_only=local, revision=od_config.revision)
        self.scheduler = SCMScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local, revision=od_config.revision
        )
        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local, revision=od_config.revision
        )
        self.tokenizer.padding_side = "right"
        self.text_encoder = from_pretrained_with_prefetch(
            Gemma2Model.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local,
            revision=od_config.revision,
            torch_dtype=od_config.dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            AutoencoderDC.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local,
            revision=od_config.revision,
            torch_dtype=od_config.dtype,
        ).to(self.device)
        config = get_hf_file_to_dict("transformer/config.json", model, revision=od_config.revision)
        if config is None:
            raise ValueError("Sana-Sprint checkpoint is missing transformer/config.json.")
        self.transformer = SanaSprintTransformer2DModel(**{k: v for k, v in config.items() if not k.startswith("_")})
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                revision=od_config.revision,
                subfolder="transformer",
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["encode_prompt", "diffuse", "vae.decode"],
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def encode_prompt(
        self, prompts: list[str], num_images: int, max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instruction = "\n".join(COMPLEX_HUMAN_INSTRUCTION)
        max_length = len(self.tokenizer.encode(instruction)) + max_sequence_length - 2
        inputs = self.tokenizer(
            [instruction + prompt.lower().strip() for prompt in prompts],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)
        embeds = self.text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask)[0]
        indices = [0] + list(range(-max_sequence_length + 1, 0))
        return (
            embeds[:, indices].repeat_interleave(num_images, dim=0),
            inputs.attention_mask[:, indices].repeat_interleave(num_images, dim=0),
        )

    def diffuse(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        guidance: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        sigma_data = self.scheduler.config.sigma_data
        dtype = next(self.transformer.parameters()).dtype
        with self.progress_bar(total=len(self.scheduler.timesteps) - 1) as progress:
            for t in self.scheduler.timesteps[:-1]:
                timestep = t.expand(latents.shape[0])
                scm_t = timestep.sin() / (timestep.cos() + timestep.sin())
                a = scm_t[:, None, None, None]
                scale = (a.square() + (1 - a).square()).sqrt()
                model_input = latents / sigma_data * scale
                prediction = self.transformer(
                    model_input.to(dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=scm_t,
                )
                prediction = ((1 - 2 * a) * model_input + (1 - 2 * a + 2 * a.square()) * prediction) / scale
                latents, denoised = self.scheduler.step(
                    prediction.float() * sigma_data,
                    timestep,
                    latents,
                    generator=generator,
                    return_dict=False,
                )
                progress.update()
        return denoised / sigma_data

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        params = req.sampling_params
        prompts = []
        for prompt in req.prompts:
            if isinstance(prompt, dict):
                if prompt.get("negative_prompt") or prompt.get("multi_modal_data"):
                    raise ValueError("Sana-Sprint supports text-to-image without negative prompts or image inputs.")
                prompt = prompt.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("Sana-Sprint requires a text prompt.")
            prompts.append(prompt)
        height = params.height if params.height is not None else 1024
        width = params.width if params.width is not None else 1024
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("Sana-Sprint height and width must be positive multiples of 32.")
        steps = params.num_inference_steps if params.num_inference_steps is not None else 2
        if steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        sequence_length = params.max_sequence_length if params.max_sequence_length is not None else 300
        if sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2.")
        if params.lora_request is not None:
            raise ValueError("Sana-Sprint does not yet support LoRA.")
        count = params.num_outputs_per_prompt
        if count < 1:
            raise ValueError("num_outputs_per_prompt must be positive.")
        original_height, original_width = height, width
        binning = params.extra_args.get("use_resolution_binning", True)
        if binning:
            height, width = self.image_processor.classify_height_width_bin(height, width, ASPECT_RATIO_1024_BIN)
        prompt_embeds, mask = self.encode_prompt(prompts, count, sequence_length)
        self.scheduler.set_timesteps(
            steps,
            device=self.device,
            max_timesteps=1.57080,
            intermediate_timesteps=1.3 if steps == 2 else None,
        )
        shape = (
            len(prompts) * count,
            self.transformer.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        generator = params.generator
        if params.latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=torch.float32)
        else:
            if tuple(params.latents.shape) != shape:
                raise ValueError(f"Expected latents with shape {shape}, got {tuple(params.latents.shape)}.")
            latents = params.latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.scheduler.config.sigma_data
        guidance_scale = params.guidance_scale if params.guidance_scale_provided else 4.5
        guidance = torch.full((shape[0],), guidance_scale, device=self.device, dtype=prompt_embeds.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale
        latents = self.diffuse(latents, prompt_embeds, mask, guidance, generator)
        if self.od_config.output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
            if binning:
                image = self.image_processor.resize_and_crop_tensor(image, original_width, original_height)
        return DiffusionOutput(
            output=image, stage_durations=self.stage_durations if self.enable_diffusion_pipeline_profiler else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
