# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
AMD Micro-World Image-to-World (I2W) pipeline.

Generates action-controlled video from an input image and keyboard/mouse actions.
Based on the Wan2.1-I2V-14B architecture with AdaLN-style action injection.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import retrieve_latents
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

from .micro_world_transformer import MicroWorldAdaLNTransformer

logger = logging.getLogger(__name__)


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/config.json")
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> MicroWorldAdaLNTransformer:
    """Create MicroWorldAdaLNTransformer from a HF config dict."""
    kwargs: dict[str, Any] = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_heads" in config:
        kwargs["num_attention_heads"] = config["num_heads"]
        if "dim" in config:
            kwargs["attention_head_dim"] = config["dim"] // config["num_heads"]
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    # in_dim (36 for I2V) overrides in_channels (16) so patch_embed loads correctly.
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "in_dim" in config:
        kwargs["in_channels"] = config["in_dim"]
    if "out_dim" in config:
        kwargs["out_channels"] = config["out_dim"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
    # AMD I2V config omits these; default to Wan2.1 I2V values.
    if config.get("model_type") == "i2v":
        kwargs.setdefault("image_dim", 1280)
        kwargs.setdefault("added_kv_proj_dim", config.get("dim", 5120))

    # Micro-World specific action params
    action_kwargs: dict[str, Any] = {}
    if "keyboard_dim" in config:
        action_kwargs["keyboard_dim"] = config["keyboard_dim"]
    if "mouse_dim" in config:
        action_kwargs["mouse_dim"] = config["mouse_dim"]
    if "action_dim" in config:
        action_kwargs["action_dim"] = config["action_dim"]

    return MicroWorldAdaLNTransformer(**action_kwargs, **kwargs)


def get_micro_world_i2w_post_process_func(od_config: OmniDiffusionConfig):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_micro_world_i2w_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process: load and resize input image, validate actions."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}

        if raw_image is None:
            raise ValueError(
                "No image provided. This model requires an image for I2W generation. "
                'Set "multi_modal_data": {"image": <path_or_PIL_Image>}'
            )
        if not isinstance(raw_image, (str, PIL.Image.Image)):
            raise TypeError(f"Unsupported image format {raw_image.__class__}.")

        image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

        # Calculate dimensions based on aspect ratio if not provided
        if request.sampling_params.height is None or request.sampling_params.width is None:
            max_area = 480 * 832
            aspect_ratio = image.height / image.width
            mod_value = 16
            h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            if request.sampling_params.height is None:
                request.sampling_params.height = h
            if request.sampling_params.width is None:
                request.sampling_params.width = w

        image = image.resize(
            (request.sampling_params.width, request.sampling_params.height),
            PIL.Image.Resampling.LANCZOS,
        )
        prompt["multi_modal_data"]["image"] = image

        prompt["additional_information"]["preprocessed_image"] = video_processor.preprocess(
            image, height=request.sampling_params.height, width=request.sampling_params.width
        )
        request.prompt = prompt
        return request

    return pre_process_func


def _resize_mask(mask: torch.Tensor, latent: torch.Tensor, process_first_frame_only: bool = True) -> torch.Tensor:
    """Resize a video mask to match latent dimensions."""
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame = F.interpolate(mask[:, :, 0:1, :, :], size=target_size, mode="trilinear", align_corners=False)

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining = F.interpolate(mask[:, :, 1:, :, :], size=target_size, mode="trilinear", align_corners=False)
            return torch.cat([first_frame, remaining], dim=2)
        return first_frame
    else:
        return F.interpolate(mask, size=list(latent_size[2:]), mode="trilinear", align_corners=False)


class MicroWorldI2WPipeline(
    nn.Module, SupportImageInput, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    """AMD Micro-World Image-to-World pipeline.

    Generates action-controlled video from an input image with keyboard/mouse inputs.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Weight sources
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        # CLIP image encoder
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model, subfolder="image_processor", local_files_only=local_files_only
        )
        self.image_encoder = CLIPVisionModel.from_pretrained(
            model, subfolder="image_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        # VAE
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        # Transformer
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)
        self.transformer_config = self.transformer.config
        self._text_len = int(transformer_config.get("text_len", 512))

        # MICRO_WORLD_SKIP_LORA=1 skips the LoRA merge for baseline debugging.
        self._lora_path: str | None = None
        lora_path = os.path.join(model, "lora_diffusion_pytorch_model.safetensors") if local_files_only else None
        if lora_path and os.path.exists(lora_path):
            if os.environ.get("MICRO_WORLD_SKIP_LORA") == "1":
                logger.info(f"Skipping LoRA weights at {lora_path} (MICRO_WORLD_SKIP_LORA=1)")
            else:
                logger.info(f"Found LoRA weights at {lora_path}")
                self._lora_path = lora_path
        self._lora_weight: float = 1.0

        # Scheduler
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

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
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        if self._lora_path is not None:
            self._merge_lora(self._lora_path, self._lora_weight)
        return loaded

    def _merge_lora(self, lora_path: str, multiplier: float = 1.0) -> None:
        import re
        from collections import defaultdict

        from safetensors.torch import load_file

        state_dict = load_file(lora_path)
        layers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in state_dict.items():
            if ".lora_up." in key or ".lora_down." in key:
                layer, sub = key.split(".", 1)
                layers[layer][sub] = value
            elif key.endswith(".alpha"):
                layer = key[: -len(".alpha")]
                layers[layer]["alpha"] = value

        _TOP_LEVEL_REMAP = [
            ("head_head", "proj_out"),
            ("time_embedding_0", "condition_embedder.time_embedder.linear_1"),
            ("time_embedding_2", "condition_embedder.time_embedder.linear_2"),
            ("time_projection_1", "condition_embedder.time_proj"),
            ("text_embedding_0", "condition_embedder.text_embedder.linear_1"),
            ("text_embedding_2", "condition_embedder.text_embedder.linear_2"),
            ("img_emb_proj_1", "condition_embedder.image_embedder.ff.net.0.proj"),
            ("img_emb_proj_3", "condition_embedder.image_embedder.ff.net.2"),
        ]
        _BLOCK_REST_MAP = {
            "self_attn_q": "attn1.to_q",
            "self_attn_k": "attn1.to_k",
            "self_attn_v": "attn1.to_v",
            "self_attn_o": "attn1.to_out",
            "cross_attn_q": "attn2.to_q",
            "cross_attn_k": "attn2.to_k",
            "cross_attn_v": "attn2.to_v",
            "cross_attn_o": "attn2.to_out",
            "cross_attn_k_img": "attn2.add_k_proj",
            "cross_attn_v_img": "attn2.add_v_proj",
            "ffn_0": "ffn.net_0.proj",
            "ffn_2": "ffn.net_2",
        }

        def _remap(lora_key: str) -> str | None:
            if not lora_key.startswith("lora_unet__"):
                return None
            name = lora_key[len("lora_unet__") :]
            for old, new in _TOP_LEVEL_REMAP:
                if name == old:
                    return new
            m = re.match(r"^(blocks|action_blocks)_(\d+)_(.+)$", name)
            if not m:
                return None
            prefix, idx, rest = m.group(1), m.group(2), m.group(3)
            sub = _BLOCK_REST_MAP.get(rest)
            if sub is None:
                return None
            return f"{prefix}.{idx}.{sub}"

        # Self-attention q/k/v are fused into attn1.to_qkv.
        from collections import defaultdict as _dd

        fused_qkv: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, float]]] = _dd(dict)
        _qkv_re = re.compile(r"^lora_unet__(blocks|action_blocks)_(\d+)_self_attn_(q|k|v)$")

        named_modules = dict(self.transformer.named_modules())
        merged = 0
        skipped = 0
        for lora_key, elems in layers.items():
            up = elems.get("lora_up.weight")
            down = elems.get("lora_down.weight")
            if up is None or down is None:
                skipped += 1
                continue
            alpha_t = elems.get("alpha")
            alpha = float(alpha_t.item()) / up.shape[1] if alpha_t is not None else 1.0

            qkv_match = _qkv_re.match(lora_key)
            if qkv_match:
                prefix, idx, shard_id = qkv_match.group(1), qkv_match.group(2), qkv_match.group(3)
                fused_target = f"{prefix}.{idx}.attn1.to_qkv"
                fused_qkv[fused_target][shard_id] = (up, down, alpha)
                continue

            target = _remap(lora_key)
            if target is None or target not in named_modules:
                skipped += 1
                continue
            module = named_modules[target]
            if not hasattr(module, "weight") or module.weight is None:
                skipped += 1
                continue
            with torch.no_grad():
                w = module.weight
                up_t = up.to(device=w.device, dtype=w.dtype)
                down_t = down.to(device=w.device, dtype=w.dtype)
                delta = torch.mm(up_t, down_t)
                w.data.add_(multiplier * alpha * delta)
            merged += 1

        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        for fused_target, shards in fused_qkv.items():
            module = named_modules.get(fused_target)
            if module is None or not hasattr(module, "weight"):
                skipped += len(shards)
                continue
            qkv_param = module.weight
            head_size = getattr(module, "head_size", None)
            n_heads_q = getattr(module, "total_num_heads", None)
            n_heads_kv = getattr(module, "total_num_kv_heads", n_heads_q)
            if head_size is None or n_heads_q is None:
                skipped += len(shards)
                continue
            q_size = n_heads_q * head_size
            kv_size = n_heads_kv * head_size
            offsets = {
                "q": (0, q_size),
                "k": (q_size, q_size + kv_size),
                "v": (q_size + kv_size, q_size + 2 * kv_size),
            }
            weight_loader = getattr(qkv_param, "weight_loader", default_weight_loader)
            with torch.no_grad():
                for shard_id, (up, down, alpha) in shards.items():
                    start, end = offsets[shard_id]
                    up_t = up.to(device=qkv_param.device, dtype=qkv_param.dtype)
                    down_t = down.to(device=qkv_param.device, dtype=qkv_param.dtype)
                    delta = multiplier * alpha * torch.mm(up_t, down_t)
                    existing = qkv_param.data[start:end].clone()
                    weight_loader(qkv_param, existing + delta, shard_id)
                    merged += 1

        logger.info("LoRA merge: %d layers merged, %d skipped (of %d)", merged, skipped, len(layers))

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompts using UMT5 text encoder."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        # Zero T5 output at PAD positions to match reference; otherwise
        # cross-attention over padded K/V collapses the output to flat color.
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_mask = neg_inputs.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(neg_input_ids, attention_mask=neg_mask)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds * neg_mask.unsqueeze(-1).to(negative_prompt_embeds.dtype)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image: PIL.Image.Image, device: torch.device) -> torch.Tensor:
        # Reference uses torch bicubic interpolate, not HF CLIPImageProcessor's
        # PIL bicubic; the delta compounds across 32 ViT layers.
        import torchvision.transforms.functional as TF

        x = TF.to_tensor(image).sub_(0.5).div_(0.5).unsqueeze(0)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = x.mul_(0.5).add_(0.5)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        pixel_values = ((x - mean) / std).to(device=device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Prepare noise latents for diffusion."""
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _parse_actions(
        self,
        extra_args: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
        num_action_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract action tensors from extra_args.

        If actions are missing and ``num_action_frames`` is given, returns
        zero-filled defaults (used for warmup / dummy run).
        """
        mouse_actions = extra_args.get("mouse_actions") if extra_args else None
        keyboard_actions = extra_args.get("keyboard_actions") if extra_args else None

        if mouse_actions is None or keyboard_actions is None:
            if num_action_frames is not None:
                mouse_actions = torch.zeros((num_action_frames, 2), dtype=dtype, device=device)
                keyboard_actions = torch.zeros((num_action_frames, 7), dtype=dtype, device=device)
            else:
                raise ValueError("Both 'mouse_actions' and 'keyboard_actions' must be provided in extra_params.")

        if not isinstance(mouse_actions, torch.Tensor):
            mouse_actions = torch.tensor(mouse_actions, dtype=dtype, device=device)
        else:
            mouse_actions = mouse_actions.to(dtype=dtype, device=device)

        if not isinstance(keyboard_actions, torch.Tensor):
            keyboard_actions = torch.tensor(keyboard_actions, dtype=dtype, device=device)
        else:
            keyboard_actions = keyboard_actions.to(dtype=dtype, device=device)

        if mouse_actions.ndim == 2:
            mouse_actions = mouse_actions.unsqueeze(0)
        if keyboard_actions.ndim == 2:
            keyboard_actions = keyboard_actions.unsqueeze(0)

        return mouse_actions, keyboard_actions

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        mouse_actions: torch.Tensor | None = None,
        keyboard_actions: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through transformer to predict noise."""
        if current_model is None:
            current_model = self.transformer
        return current_model(
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            **kwargs,
        )[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        image: PIL.Image.Image | torch.Tensor | None = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        frame_num: int = 49,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Extract prompt
        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required.")

        # Extract image
        if image is None:
            multi_modal_data = (
                req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
            )
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if raw_image is None:
                raise ValueError("Image is required for I2W generation.")
            if isinstance(raw_image, list):
                raw_image = raw_image[0]
            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image).convert("RGB")
            else:
                image = cast(PIL.Image.Image | torch.Tensor, raw_image)

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames or frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        # Dimension alignment
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Parse actions
        extra_args = req.sampling_params.extra_args if hasattr(req.sampling_params, "extra_args") else None
        mouse_actions, keyboard_actions = self._parse_actions(extra_args, device, dtype, num_action_frames=num_frames)

        # Encode prompt
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                device=device,
                dtype=dtype,
            )

        # encode_image does its own 224x224 interpolate; pass raw image.
        clip_fea = self.encode_image(image, device=device)

        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        image_tensor = video_processor.preprocess(image, height=height, width=width)

        num_channels_latents = self.transformer_config.out_channels
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # Build I2V conditioning y = cat([mask(4ch), masked_video(16ch)]) → 20ch,
        # later concatenated with 16ch noise to form the 36ch patch_embed input.
        bs = prompt_embeds.shape[0]
        init_video = image_tensor.unsqueeze(2).to(device=device, dtype=torch.float32)  # (1, 3, 1, H, W)
        if init_video.shape[2] < num_frames:
            pad_frames = torch.zeros(
                init_video.shape[0],
                init_video.shape[1],
                num_frames - init_video.shape[2],
                height,
                width,
                dtype=init_video.dtype,
                device=device,
            )
            masked_video = torch.cat([init_video, pad_frames], dim=2)
        else:
            masked_video = init_video[:, :, :num_frames]
        masked_video = masked_video.to(self.vae.dtype)

        masked_video_latents = retrieve_latents(self.vae.encode(masked_video), sample_mode="argmax")
        masked_video_latents = masked_video_latents.repeat(bs, 1, 1, 1, 1).to(torch.float32)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(masked_video_latents.device, masked_video_latents.dtype)
        )
        latents_std_inv = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            masked_video_latents.device, masked_video_latents.dtype
        )
        masked_video_latents = (masked_video_latents - latents_mean) * latents_std_inv

        # 4ch latent mask: replicate frame-0 mask 4× temporally (VAE temporal
        # compression), then fold every 4 frames into the channel dim.
        mask_video = torch.ones(1, 1, num_frames, height, width, dtype=torch.float32, device=device)
        mask_video[:, :, 0] = 0
        first_frame_chunk = mask_video[:, :, 0:1].repeat(1, 1, 4, 1, 1)
        rest_chunk = mask_video[:, :, 1:]
        mask_video = torch.cat([first_frame_chunk, rest_chunk], dim=2)
        t_lat_full = mask_video.shape[2] // 4
        mask_video = mask_video.view(1, t_lat_full, 4, height, width).transpose(1, 2)
        mask_latents = _resize_mask(1.0 - mask_video, masked_video_latents, process_first_frame_only=True).to(
            device, dtype=torch.float32
        )
        mask_latents = mask_latents.repeat(bs, 1, 1, 1, 1)

        y_cond = torch.cat([mask_latents, masked_video_latents], dim=1)

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        # Denoising loop
        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t

                latent_model_input = torch.cat([latents, y_cond], dim=1).to(dtype)
                timestep = t.expand(latents.shape[0])

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": clip_fea,
                    "return_dict": False,
                    "current_model": self.transformer,
                    "mouse_actions": mouse_actions,
                    "keyboard_actions": keyboard_actions,
                }
                if do_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_hidden_states_image": clip_fea,
                        "return_dict": False,
                        "current_model": self.transformer,
                        "mouse_actions": mouse_actions,
                        "keyboard_actions": keyboard_actions,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_cfg)
                pbar.update()

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # Decode
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )
