# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import PIL
import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torch import nn
from transformers import (
    AutoTokenizer,
    Qwen2VLProcessor,
)
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import (
    LongCatImageTransformer2DModel,
)

logger = init_logger(__name__)


class LongcatImageEditPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
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

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_processor = Qwen2VLProcessor.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )
        self.transformer = LongCatImageTransformer2DModel(od_config=od_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        self.prompt_template_encode_prefix = (
            "<|im_start|>system\n"
            "As an image captioning expert, generate a descriptive text prompt based on an image content,"
            " suitable for input to a text-to-image model.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"

        self.default_sample_size = 128
        self.tokenizer_max_length = 512

    def _encode_prompt(self, prompt, image):
        pass

    def encode_prompt(
        self,
        prompt: list[str] = None,
        image: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
    ):
        pass

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        prompt_embeds_length,
        device,
        generator,
        latents=None,
    ):
        pass

    def check_inputs(
        self, prompt, height, width, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        pass

    def forward(
        self,
        image: PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ):
        pass
