# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
    HunyuanVideo15Transformer3DModel,
)
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    extract_glyph_texts,
    format_text_input,
    get_hunyuan_video_15_post_process_func,
    retrieve_latents,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def get_omniweaving_post_process_func(od_config: OmniDiffusionConfig):
    return get_hunyuan_video_15_post_process_func(od_config)


def get_omniweaving_pre_process_func(od_config: OmniDiffusionConfig):
    max_area = 480 * 832
    divisor = 16

    def pre_process_func(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if not req.prompts:
            return req
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            return req

        multi_modal_data = prompt_data.get("multi_modal_data", {})
        raw_image = multi_modal_data.get("image", None)
        if raw_image is None:
            return req
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            return req

        height, width = req.sampling_params.height, req.sampling_params.width
        if height is None or width is None:
            w, h = image.size
            aspect = w / h
            target_h = int((max_area / aspect) ** 0.5)
            target_w = int(target_h * aspect)
            req.sampling_params.height = height or ((target_h + divisor - 1) // divisor * divisor)
            req.sampling_params.width = width or ((target_w + divisor - 1) // divisor * divisor)
        return req

    return pre_process_func


class OmniWeavingPipeline(
    nn.Module,
    CFGParallelMixin,
    SupportImageInput,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    support_image_input = True
    color_format = "RGB"

    # Default external sub-model paths
    DEFAULT_QWEN_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    DEFAULT_SIGLIP_PATH = "google/siglip-so400m-patch14-384"
    DEFAULT_T2V_PATH = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

    @staticmethod
    def _resolve_external_model_path(
        od_config: OmniDiffusionConfig,
        arg_name: str,
        default_path: str,
    ) -> str:
        custom_args = od_config.custom_pipeline_args or {}
        if isinstance(custom_args, dict):
            value = custom_args.get(arg_name)
            if isinstance(value, str) and value:
                return value
        return default_path

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)
        self.qwen_path = self._resolve_external_model_path(od_config, "qwen_path", self.DEFAULT_QWEN_PATH)
        self.siglip_path = self._resolve_external_model_path(od_config, "siglip_path", self.DEFAULT_SIGLIP_PATH)
        self.t2v_path = self._resolve_external_model_path(od_config, "t2v_path", self.DEFAULT_T2V_PATH)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.qwen_path)
        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.qwen_path, torch_dtype=dtype).to(
            self.device
        )

        ckpt_dir = os.path.join(model, "text_encoder")
        safetensors_files = glob.glob(os.path.join(ckpt_dir, "**", "*.safetensors"), recursive=True)
        if safetensors_files:
            from safetensors.torch import load_file

            for f in safetensors_files:
                te_weights = load_file(f)
                if "__metadata__" in te_weights:
                    del te_weights["__metadata__"]
                self.mllm.load_state_dict(te_weights, strict=False)

        try:
            self.tokenizer_2 = AutoTokenizer.from_pretrained(
                model, subfolder="tokenizer_2", local_files_only=local_files_only
            )
            t5_config = AutoConfig.from_pretrained(model, subfolder="text_encoder_2", local_files_only=local_files_only)
            self.text_encoder_2 = T5EncoderModel(t5_config, prefix="text_encoder_2").to(dtype=dtype, device=self.device)
            self.has_t5_2 = True
        except Exception:
            self.tokenizer_2 = None
            self.text_encoder_2 = None
            self.has_t5_2 = False
            self.t5_2_d_model = 1472

        try:
            self.image_encoder = SiglipVisionModel.from_pretrained(
                model,
                subfolder="image_encoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(self.device)
            self.feature_extractor = SiglipImageProcessor.from_pretrained(
                model,
                subfolder="feature_extractor",
                local_files_only=local_files_only,
            )
        except Exception:
            self.image_encoder = SiglipVisionModel.from_pretrained(self.siglip_path, torch_dtype=dtype).to(self.device)
            self.feature_extractor = SiglipImageProcessor.from_pretrained(self.siglip_path)

        self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            self.t2v_path, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.t2v_path, subfolder="scheduler")
        if od_config.flow_shift is not None:
            self.scheduler._shift = od_config.flow_shift

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, HunyuanVideo15Transformer3DModel)
        self.transformer = HunyuanVideo15Transformer3DModel(od_config=od_config, **transformer_kwargs)
        self.use_meanflow = getattr(od_config.tf_model_config, "use_meanflow", False)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder_2",
                revision=None,
                prefix="text_encoder_2.",
                fall_back_to_pt=True,
            ),
        ]

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 16
        )
        self.num_channels_latents = self.vae.config.latent_channels if hasattr(self.vae, "config") else 32

        self.system_message = (
            "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video."
        )

        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def _get_mllm_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        num_hidden_layers_to_skip: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_formatted = format_text_input(prompt, self.system_message)
        text_inputs = self.tokenizer.apply_chat_template(
            prompt_formatted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)
        with torch.no_grad():
            outputs = self.mllm.model(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
            )
        prompt_embeds = outputs.hidden_states[-(num_hidden_layers_to_skip + 1)]
        crop_start = self.prompt_template_encode_start_idx
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]
        return (
            torch.clamp(prompt_embeds, min=-65504.0, max=65504.0).to(dtype=dtype),
            prompt_attention_mask,
        )

    def _get_t5_2_prompt_embeds(
        self, prompt: list[str], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds_list, prompt_embeds_mask_list = [], []
        for p in prompt:
            glyph_text = extract_glyph_texts(p)
            if glyph_text is None or not self.has_t5_2:
                glyph_text_embeds = torch.zeros(
                    (1, self.tokenizer_2_max_length, self.t5_2_d_model),
                    device=device,
                    dtype=dtype,
                )
                glyph_text_embeds_mask = torch.ones((1, self.tokenizer_2_max_length), device=device, dtype=torch.int64)
            else:
                txt_tokens = self.tokenizer_2(
                    glyph_text,
                    padding="max_length",
                    max_length=self.tokenizer_2_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)
                glyph_text_embeds = self.text_encoder_2(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0].to(device=device, dtype=dtype)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)
            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)
        return torch.cat(prompt_embeds_list, dim=0), torch.cat(prompt_embeds_mask_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image | None,
        device: torch.device,
        dtype: torch.dtype,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = False,
    ) -> tuple:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt, device, dtype)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_t5_2_prompt_embeds(prompt, device, dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype)

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        negative_prompt_embeds_2 = None
        negative_prompt_embeds_mask_2 = None

        if do_classifier_free_guidance:
            neg_p = (
                [negative_prompt]
                if isinstance(negative_prompt, str)
                else (negative_prompt if negative_prompt else [""])
            )
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
            ) = self._get_mllm_prompt_embeds(neg_p, device, dtype)
            (
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self._get_t5_2_prompt_embeds(neg_p, device, dtype)
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(dtype=dtype)
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(dtype=dtype)
        return (
            prompt_embeds,
            prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
        )

    def _get_image_embeds(self, image: PIL.Image.Image, device: torch.device) -> torch.Tensor:
        image_encoder_dtype = next(self.image_encoder.parameters()).dtype
        pixel_values = self.feature_extractor(images=image, do_resize=True, return_tensors="pt").pixel_values.to(
            device=device, dtype=image_encoder_dtype
        )
        return torch.clamp(
            self.image_encoder(pixel_values).last_hidden_state,
            min=-65504.0,
            max=65504.0,
        )

    def _get_image_latents(self, image: PIL.Image.Image, height: int, width: int, device: torch.device) -> torch.Tensor:
        image_tensor = (
            VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            .preprocess(image, height=height, width=width)
            .unsqueeze(2)
            .to(device=device, dtype=self.vae.dtype)
        )
        return retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax") * self.vae.config.scaling_factor

    def prepare_cond_latents_and_mask(
        self,
        latents: torch.Tensor,
        image: PIL.Image.Image,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, frames, lat_height, lat_width = latents.shape
        latent_condition = self._get_image_latents(image, height, width, device).repeat(batch, 1, frames, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        latent_mask = torch.zeros(batch, 1, frames, lat_height, lat_width, dtype=dtype, device=device)
        latent_mask[:, :, 0, :, :] = 1.0
        return latent_condition.to(device=device, dtype=dtype), latent_mask

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (
            batch_size,
            self.num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 121,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if not req.prompts:
            raise ValueError("Prompt is required.")
        prompt_data = req.prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else prompt_data.get("prompt")
        negative_prompt = None if isinstance(prompt_data, str) else prompt_data.get("negative_prompt")
        multi_modal_data = prompt_data.get("multi_modal_data", {}) if not isinstance(prompt_data, str) else {}

        raw_image = multi_modal_data.get("image", None)
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        image = None
        if raw_image is not None:
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image.convert("RGB")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames_val = req.sampling_params.num_frames or num_frames
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale_provided else guidance_scale
        )
        do_cfg = guidance_scale > 1.0

        device = self.device
        dtype = self.transformer.transformer_blocks[0].norm1.linear.weight.dtype
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        enc_tuple = self.encode_prompt(prompt, image, device, dtype, negative_prompt, do_cfg)

        latents = self.prepare_latents(
            enc_tuple[0].shape[0],
            height,
            width,
            num_frames_val,
            dtype,
            device,
            generator,
            req.sampling_params.latents,
        )

        if image is not None:
            image_embeds = self._get_image_embeds(image, device).to(dtype=dtype)
            cond_latents, mask = self.prepare_cond_latents_and_mask(latents, image, height, width, dtype, device)
        else:
            batch, channels, frames, lat_height, lat_width = latents.shape
            cond_latents = torch.zeros_like(latents)
            mask = torch.zeros(batch, 1, frames, lat_height, lat_width, dtype=dtype, device=device)
            image_embeds = torch.zeros((batch, 729, 1152), dtype=dtype, device=device)

        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, num_steps + 1)[:-1], device=device)
        timesteps = self.scheduler.timesteps

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents, cond_latents, mask], dim=1)
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                timestep_r = torch.tensor([0.0], device=device) if i == len(timesteps) - 1 else timesteps[i + 1]
                timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype) if self.use_meanflow else None

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "timestep_r": timestep_r,
                    "encoder_hidden_states": enc_tuple[0],
                    "encoder_attention_mask": enc_tuple[1],
                    "encoder_hidden_states_2": enc_tuple[2],
                    "encoder_attention_mask_2": enc_tuple[3],
                    "image_embeds": image_embeds,
                    "return_dict": False,
                }
                negative_kwargs = (
                    {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "timestep_r": timestep_r,
                        "encoder_hidden_states": enc_tuple[4],
                        "encoder_attention_mask": enc_tuple[5],
                        "encoder_hidden_states_2": enc_tuple[6],
                        "encoder_attention_mask_2": enc_tuple[7],
                        "image_embeds": image_embeds,
                        "return_dict": False,
                    }
                    if do_cfg and enc_tuple[4] is not None
                    else None
                )

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=bool(negative_kwargs),
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=req.sampling_params.cfg_normalize,
                )
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg=bool(negative_kwargs))
                pbar.update()

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        if output_type == "latent":
            return DiffusionOutput(output=latents)
        return DiffusionOutput(
            output=self.vae.decode(
                latents.to(self.vae.dtype) / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
        )

    # NATIVE ARCHITECTURE ROUTER: Tencent Original -> HF Diffusers
    def _translate_tencent_to_hf(self, raw_tf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped_tf = {}

        def map_k(src: str, dst: str):
            if src in raw_tf:
                mapped_tf[dst] = raw_tf[src]

        map_k("img_in.proj.weight", "x_embedder.proj.weight")
        map_k("img_in.proj.bias", "x_embedder.proj.bias")
        map_k("final_layer.linear.weight", "proj_out.weight")
        map_k("final_layer.linear.bias", "proj_out.bias")
        map_k("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight")
        map_k("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias")
        map_k("txt_in.input_embedder.weight", "context_embedder.proj_in.weight")
        map_k("txt_in.input_embedder.bias", "context_embedder.proj_in.bias")

        # Time Text Embedder
        map_k(
            "txt_in.c_embedder.linear_1.weight",
            "context_embedder.time_text_embed.text_embedder.linear_1.weight",
        )
        map_k(
            "txt_in.c_embedder.linear_1.bias",
            "context_embedder.time_text_embed.text_embedder.linear_1.bias",
        )
        map_k(
            "txt_in.c_embedder.linear_2.weight",
            "context_embedder.time_text_embed.text_embedder.linear_2.weight",
        )
        map_k(
            "txt_in.c_embedder.linear_2.bias",
            "context_embedder.time_text_embed.text_embedder.linear_2.bias",
        )
        map_k(
            "txt_in.t_embedder.mlp.0.weight",
            "context_embedder.time_text_embed.timestep_embedder.linear_1.weight",
        )
        map_k(
            "txt_in.t_embedder.mlp.0.bias",
            "context_embedder.time_text_embed.timestep_embedder.linear_1.bias",
        )
        map_k(
            "txt_in.t_embedder.mlp.2.weight",
            "context_embedder.time_text_embed.timestep_embedder.linear_2.weight",
        )
        map_k(
            "txt_in.t_embedder.mlp.2.bias",
            "context_embedder.time_text_embed.timestep_embedder.linear_2.bias",
        )

        # Time Embedder
        map_k("time_in.mlp.0.weight", "time_embed.timestep_embedder.linear_1.weight")
        map_k("time_in.mlp.0.bias", "time_embed.timestep_embedder.linear_1.bias")
        map_k("time_in.mlp.2.weight", "time_embed.timestep_embedder.linear_2.weight")
        map_k("time_in.mlp.2.bias", "time_embed.timestep_embedder.linear_2.bias")
        map_k("cond_type_embedding.weight", "cond_type_embed.weight")

        # Vision Embedder
        vision_proj_map = {
            "vision_in.proj.0.weight": "image_embedder.norm_in.weight",
            "vision_in.proj.0.bias": "image_embedder.norm_in.bias",
            "vision_in.proj.1.weight": "image_embedder.linear_1.weight",
            "vision_in.proj.1.bias": "image_embedder.linear_1.bias",
            "vision_in.proj.3.weight": "image_embedder.linear_2.weight",
            "vision_in.proj.3.bias": "image_embedder.linear_2.bias",
            "vision_in.proj.4.weight": "image_embedder.norm_out.weight",
            "vision_in.proj.4.bias": "image_embedder.norm_out.bias",
        }
        for s_k, d_k in vision_proj_map.items():
            map_k(s_k, d_k)

        # T5-2 Projections
        t5_2_prefix = "by" + "t5_in"
        if f"{t5_2_prefix}.fc1.weight" in raw_tf:
            w1 = raw_tf[f"{t5_2_prefix}.fc1.weight"]
            mapped_tf["context_embedder_2.linear_1.weight"] = w1

            map_k(f"{t5_2_prefix}.fc1.bias", "context_embedder_2.linear_1.bias")
            map_k(f"{t5_2_prefix}.fc2.weight", "context_embedder_2.linear_2.weight")
            map_k(f"{t5_2_prefix}.fc2.bias", "context_embedder_2.linear_2.bias")
            map_k(f"{t5_2_prefix}.fc3.weight", "context_embedder_2.linear_3.weight")
            map_k(f"{t5_2_prefix}.fc3.bias", "context_embedder_2.linear_3.bias")
            map_k(f"{t5_2_prefix}.layernorm.weight", "context_embedder_2.norm.weight")
            map_k(f"{t5_2_prefix}.layernorm.bias", "context_embedder_2.norm.bias")

        # Refiner Blocks
        for layer in range(2):
            p = f"txt_in.individual_token_refiner.blocks.{layer}"
            if f"{p}.self_attn_qkv.weight" in raw_tf:
                q_w, k_w, v_w = raw_tf[f"{p}.self_attn_qkv.weight"].chunk(3, dim=0)
                q_b, k_b, v_b = raw_tf[f"{p}.self_attn_qkv.bias"].chunk(3, dim=0)
                pref = f"context_embedder.token_refiner.refiner_blocks.{layer}"

                mapped_tf[f"{pref}.attn.to_q.weight"] = q_w
                mapped_tf[f"{pref}.attn.to_k.weight"] = k_w
                mapped_tf[f"{pref}.attn.to_v.weight"] = v_w
                mapped_tf[f"{pref}.attn.to_q.bias"] = q_b
                mapped_tf[f"{pref}.attn.to_k.bias"] = k_b
                mapped_tf[f"{pref}.attn.to_v.bias"] = v_b

                map_k(f"{p}.self_attn_proj.weight", f"{pref}.attn.to_out.0.weight")
                map_k(f"{p}.self_attn_proj.bias", f"{pref}.attn.to_out.0.bias")
                map_k(f"{p}.norm1.weight", f"{pref}.norm1.weight")
                map_k(f"{p}.norm1.bias", f"{pref}.norm1.bias")
                map_k(f"{p}.norm2.weight", f"{pref}.norm2.weight")
                map_k(f"{p}.norm2.bias", f"{pref}.norm2.bias")
                map_k(f"{p}.mlp.fc1.weight", f"{pref}.ff.net.0.proj.weight")
                map_k(f"{p}.mlp.fc1.bias", f"{pref}.ff.net.0.proj.bias")
                map_k(f"{p}.mlp.fc2.weight", f"{pref}.ff.net.2.weight")
                map_k(f"{p}.mlp.fc2.bias", f"{pref}.ff.net.2.bias")
                map_k(f"{p}.adaLN_modulation.1.weight", f"{pref}.norm_out.linear.weight")
                map_k(f"{p}.adaLN_modulation.1.bias", f"{pref}.norm_out.linear.bias")

        # Double Blocks
        double_block_keys = set(k.split(".")[1] for k in raw_tf.keys() if k.startswith("double_blocks."))
        for i_str in double_block_keys:
            p = f"double_blocks.{i_str}"
            t_p = f"transformer_blocks.{i_str}"

            map_k(f"{p}.img_attn_q.weight", f"{t_p}.attn.to_q.weight")
            map_k(f"{p}.img_attn_k.weight", f"{t_p}.attn.to_k.weight")
            map_k(f"{p}.img_attn_v.weight", f"{t_p}.attn.to_v.weight")
            map_k(f"{p}.img_attn_q.bias", f"{t_p}.attn.to_q.bias")
            map_k(f"{p}.img_attn_k.bias", f"{t_p}.attn.to_k.bias")
            map_k(f"{p}.img_attn_v.bias", f"{t_p}.attn.to_v.bias")

            map_k(f"{p}.txt_attn_q.weight", f"{t_p}.attn.add_q_proj.weight")
            map_k(f"{p}.txt_attn_k.weight", f"{t_p}.attn.add_k_proj.weight")
            map_k(f"{p}.txt_attn_v.weight", f"{t_p}.attn.add_v_proj.weight")
            map_k(f"{p}.txt_attn_q.bias", f"{t_p}.attn.add_q_proj.bias")
            map_k(f"{p}.txt_attn_k.bias", f"{t_p}.attn.add_k_proj.bias")
            map_k(f"{p}.txt_attn_v.bias", f"{t_p}.attn.add_v_proj.bias")

            map_k(f"{p}.img_attn_q_norm.weight", f"{t_p}.attn.norm_q.weight")
            map_k(f"{p}.img_attn_k_norm.weight", f"{t_p}.attn.norm_k.weight")
            map_k(f"{p}.txt_attn_q_norm.weight", f"{t_p}.attn.norm_added_q.weight")
            map_k(f"{p}.txt_attn_k_norm.weight", f"{t_p}.attn.norm_added_k.weight")

            map_k(f"{p}.img_attn_proj.weight", f"{t_p}.attn.to_out.0.weight")
            map_k(f"{p}.img_attn_proj.bias", f"{t_p}.attn.to_out.0.bias")
            map_k(f"{p}.txt_attn_proj.weight", f"{t_p}.attn.to_add_out.weight")
            map_k(f"{p}.txt_attn_proj.bias", f"{t_p}.attn.to_add_out.bias")

            map_k(f"{p}.img_mlp.fc1.weight", f"{t_p}.ff.net.0.proj.weight")
            map_k(f"{p}.img_mlp.fc1.bias", f"{t_p}.ff.net.0.proj.bias")
            map_k(f"{p}.img_mlp.fc2.weight", f"{t_p}.ff.net.2.weight")
            map_k(f"{p}.img_mlp.fc2.bias", f"{t_p}.ff.net.2.bias")

            map_k(f"{p}.txt_mlp.fc1.weight", f"{t_p}.ff_context.net.0.proj.weight")
            map_k(f"{p}.txt_mlp.fc1.bias", f"{t_p}.ff_context.net.0.proj.bias")
            map_k(f"{p}.txt_mlp.fc2.weight", f"{t_p}.ff_context.net.2.weight")
            map_k(f"{p}.txt_mlp.fc2.bias", f"{t_p}.ff_context.net.2.bias")

            map_k(f"{p}.img_mod.linear.weight", f"{t_p}.norm1.linear.weight")
            map_k(f"{p}.img_mod.linear.bias", f"{t_p}.norm1.linear.bias")
            map_k(f"{p}.txt_mod.linear.weight", f"{t_p}.norm1_context.linear.weight")
            map_k(f"{p}.txt_mod.linear.bias", f"{t_p}.norm1_context.linear.bias")

        # Single Blocks
        single_block_keys = set(k.split(".")[1] for k in raw_tf.keys() if k.startswith("single_blocks."))
        for i_str in single_block_keys:
            p = f"single_blocks.{i_str}"
            t_p = f"single_transformer_blocks.{i_str}"

            map_k(f"{p}.attn_q.weight", f"{t_p}.attn.to_q.weight")
            map_k(f"{p}.attn_k.weight", f"{t_p}.attn.to_k.weight")
            map_k(f"{p}.attn_v.weight", f"{t_p}.attn.to_v.weight")
            map_k(f"{p}.attn_q.bias", f"{t_p}.attn.to_q.bias")
            map_k(f"{p}.attn_k.bias", f"{t_p}.attn.to_k.bias")
            map_k(f"{p}.attn_v.bias", f"{t_p}.attn.to_v.bias")

            map_k(f"{p}.attn_proj.weight", f"{t_p}.attn.to_out.0.weight")
            map_k(f"{p}.attn_proj.bias", f"{t_p}.attn.to_out.0.bias")
            map_k(f"{p}.mlp.fc1.weight", f"{t_p}.proj_mlp.weight")
            map_k(f"{p}.mlp.fc1.bias", f"{t_p}.proj_mlp.bias")
            map_k(f"{p}.mlp.fc2.weight", f"{t_p}.proj_out.weight")
            map_k(f"{p}.mlp.fc2.bias", f"{t_p}.proj_out.bias")
            map_k(f"{p}.mod.linear.weight", f"{t_p}.norm.linear.weight")
            map_k(f"{p}.mod.linear.bias", f"{t_p}.norm.linear.bias")

        return mapped_tf

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raw_tf, raw_t5, other_weights = {}, {}, []
        for k, v in weights:
            if k.startswith("transformer."):
                raw_tf[k[len("transformer.") :]] = v
            elif k.startswith("text_encoder_2."):
                raw_t5[k[len("text_encoder_2.") :]] = v
            else:
                other_weights.append((k, v))

        if any(k.startswith("double_blocks.") for k in raw_tf.keys()):
            logger.info("Detecting Tencent original weight format. Routing through native Auto-Translator...")
            tf_weights_to_load = self._translate_tencent_to_hf(raw_tf).items()
        else:
            tf_weights_to_load = raw_tf.items()

        loaded_keys = set()

        # Load Main Backbone
        if hasattr(self.transformer, "load_weights"):
            tf_loaded = self.transformer.load_weights(tf_weights_to_load)
            loaded_keys.update([f"transformer.{k}" for k in tf_loaded])
        else:
            loader = AutoWeightsLoader(self.transformer)
            tf_loaded = loader.load_weights(tf_weights_to_load)
            loaded_keys.update([f"transformer.{k}" for k in tf_loaded])

        # Load T5-2
        if self.has_t5_2 and raw_t5:
            t5_items = raw_t5.items()
            if hasattr(self.text_encoder_2, "load_weights"):
                t5_loaded = self.text_encoder_2.load_weights(t5_items)
                loaded_keys.update([f"text_encoder_2.{k}" for k in t5_loaded])
            else:
                loader = AutoWeightsLoader(self.text_encoder_2)
                t5_loaded = loader.load_weights(t5_items)
                loaded_keys.update([f"text_encoder_2.{k}" for k in t5_loaded])

        # Load Everything Else
        if other_weights:
            loader = AutoWeightsLoader(self)
            loaded_keys.update(loader.load_weights(other_weights))

        for name, p in self.transformer.named_parameters():
            if f"transformer.{name}" not in loaded_keys and (torch.isnan(p).any() or p.std().item() == 0.0):
                logger.warning(
                    f"Parameter transformer.{name} was not loaded from the checkpoint. "
                    "This might cause incorrect outputs."
                )

        return loaded_keys
