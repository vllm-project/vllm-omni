# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NextStep-1.1 (https://huggingface.co/stepfun-ai/NextStep-1.1)

import os
import re
from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.nextstep_1_1.modeling_flux_vae import AutoencoderKL
from vllm_omni.diffusion.models.nextstep_1_1.modeling_nextstep import (
    NextStepConfig,
    NextStepModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = init_logger(__name__)


def layer_norm(input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(input, normalized_shape, None, None, eps)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_pil(image: torch.Tensor, mode: str = "11") -> Image.Image:
    """Convert tensor to PIL Image."""
    if mode == "11":
        # Assuming image is in [-1, 1] range
        image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def hw2str(h: int, w: int) -> str:
    """Convert height and width to string format."""
    return f"{h}*{w}"


DEFAULT_IMAGE_AREA_TOKEN = "<|image_area|>"


def get_nextstep11_post_process_func(od_config: OmniDiffusionConfig):
    """Return post-processing function for NextStep-1.1 pipeline outputs."""
    vae_scale_factor = 8  # Default for NextStep VAE

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class NextStep11Pipeline(nn.Module):
    """
    NextStep-1.1 Pipeline for text-to-image generation.

    This pipeline implements the autoregressive flow-based image generation
    model from StepFun. It uses an LLM backbone with a flow matching head
    to generate images autoregressively.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self._execution_device = get_local_device()

        model_path = od_config.model
        local_files_only = os.path.exists(model_path)

        if not local_files_only:
            model_path = download_weights_from_hf_specific(model_path, None, ["*"])

        # Load tokenizer (still uses trust_remote_code for tokenizer only)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            model_max_length=512,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )
        self.tokenizer.add_eos_token = False

        # Load model from local TP-aware code (weights loaded later via load_weights)
        config = NextStepConfig.from_json(os.path.join(model_path, "config.json"))
        self.model = NextStepModel(config)
        self.model.eval()

        # Load config
        self.config = self.model.config

        # Load VAE
        vae_path = getattr(self.config, "vae_name_or_path", None)
        if vae_path is None:
            vae_path = os.path.join(model_path, "vae")
        elif not os.path.isabs(vae_path):
            # Resolve relative vae_name_or_path (e.g. "vae/") against model dir
            vae_path = os.path.join(model_path, vae_path)

        if os.path.exists(vae_path):
            self.vae = AutoencoderKL.from_pretrained(vae_path)
        else:
            # Try loading from model directory
            vae_checkpoint = os.path.join(model_path, "vae", "checkpoint.pt")
            vae_config = os.path.join(model_path, "vae", "config.json")
            if os.path.exists(vae_checkpoint) and os.path.exists(vae_config):
                self.vae = AutoencoderKL.from_pretrained(os.path.join(model_path, "vae"))
            else:
                raise ValueError(f"Could not find VAE at {vae_path}")

        self.vae.eval()

        # Calculate down factor
        vae_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_patch_size = getattr(self.config, "latent_patch_size", 2)
        self.down_factor = vae_factor * latent_patch_size

        # Get VAE parameters
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)

        # Get special token IDs from config
        self.boi = getattr(self.config, "boi", None)
        self.eoi = getattr(self.config, "eoi", None)
        self.image_placeholder_id = getattr(self.config, "image_placeholder_id", None)

        # Image processing
        self.pil2tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.__device = self._execution_device
        self.__dtype = od_config.dtype

        # Weight sources: model weights from safetensors, prefixed with "model."
        # so AutoWeightsLoader dispatches to self.model.load_weights()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="model.",
                fall_back_to_pt=True,
                allow_patterns_overrides=["model-*.safetensors", "model.safetensors"],
            )
        ]

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    def to(self, device=None, dtype=None):
        if device is not None:
            self.__device = device
        if dtype is not None:
            self.__dtype = dtype
        self.model.to(self.__device, dtype=self.__dtype)
        self.vae.to(self.__device, dtype=self.__dtype)
        return self

    def _image_str(self, hw: tuple[int, int] = (256, 256)):
        """Generate image token string for given height/width."""
        latent_hw = (hw[0] // self.down_factor, hw[1] // self.down_factor)
        image_ids = [self.boi] + [self.image_placeholder_id] * (latent_hw[0] * latent_hw[1]) + [self.eoi]
        image_str = DEFAULT_IMAGE_AREA_TOKEN + hw2str(*latent_hw) + self.tokenizer.decode(image_ids)
        return image_str

    def _check_input(
        self, captions: str | list[str], images: Image.Image | list[Image.Image] | None
    ) -> tuple[list[str], list[Image.Image] | None]:
        """Validate and process input captions and images."""
        if not isinstance(captions, list):
            captions = [captions]

        if images is not None:
            if not isinstance(images, list):
                images = [images]

            # Validate image count matches <image> tokens in captions
            image_token_count = 0
            for caption in captions:
                num_image_token = len(re.findall(r"<image>", caption))
                if num_image_token != 1:
                    raise ValueError(
                        f"Caption must contain exactly one <image> token. "
                        f"Found {num_image_token} in: {caption[:100]}..."
                    )
                image_token_count += num_image_token
            if image_token_count != len(images):
                raise ValueError(
                    f"Number of images ({len(images)}) does not match number of image tokens ({image_token_count})."
                )

            hws = [(image.size[1], image.size[0]) for image in images]

            # Replace <image> tokens with corresponding image_str
            processed_captions = []
            image_idx = 0
            for caption in captions:
                processed_caption = caption
                num_image_tokens = processed_caption.count("<image>")

                for _ in range(num_image_tokens):
                    processed_caption = processed_caption.replace("<image>", self._image_str(hws[image_idx]), 1)
                    image_idx += 1

                processed_captions.append(processed_caption)

            captions = processed_captions
        return captions, images

    @staticmethod
    def _resolve_cfg_layout(cfg: float, cfg_img: float, has_image_conditions: bool) -> tuple[int, float]:
        """Resolve the active CFG branch layout for the current request."""
        use_text_cfg = cfg > 1.0
        # Image CFG branch is only meaningful when the request has an image condition.
        use_img_cfg = use_text_cfg and has_image_conditions and cfg_img != 1.0
        cfg_mult = 1 + int(use_text_cfg) + int(use_img_cfg)
        effective_cfg_img = cfg_img if use_img_cfg else 1.0
        return cfg_mult, effective_cfg_img

    def _build_captions(
        self,
        captions: str | list[str],
        images: list[Image.Image] | None = None,
        num_images_per_caption: int = 1,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
    ) -> tuple[list[str], list[Image.Image] | None, int, float]:
        """Build captions with CFG support."""
        if not isinstance(captions, list):
            captions = [captions]
        captions = [caption for caption in captions for _ in range(num_images_per_caption)]
        if images is not None:
            images = [image for image in images for _ in range(num_images_per_caption)]

        # Add positive prompt
        if positive_prompt is not None and positive_prompt != "":
            captions = [f"{caption} {positive_prompt}" for caption in captions]

        cfg_mult, effective_cfg_img = self._resolve_cfg_layout(cfg, cfg_img, images is not None)

        # Add negative prompt for CFG
        if negative_prompt is None:
            negative_prompt = ""
        num_samples = len(captions)
        if cfg_mult == 3:
            w, h = images[0].size
            captions = captions + [self._image_str((h, w)) + negative_prompt] * num_samples
            images = images + images
            captions = captions + [negative_prompt] * num_samples
        elif cfg_mult == 2:
            captions = captions + [negative_prompt] * num_samples

        return captions, images, cfg_mult, effective_cfg_img

    def _add_prefix_ids(self, hw: tuple[int, int], input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Add prefix IDs for image generation."""
        prefix_str = DEFAULT_IMAGE_AREA_TOKEN + hw2str(hw[0] // self.down_factor, hw[1] // self.down_factor)
        prefix_output = self.tokenizer(prefix_str, truncation=False, add_special_tokens=True, return_tensors="pt")
        prefix_input_ids = prefix_output.input_ids.to(input_ids.device, dtype=input_ids.dtype)
        prefix_attention_mask = prefix_output.attention_mask.to(attention_mask.device, dtype=attention_mask.dtype)

        # Remove bos token
        if self.tokenizer.bos_token is not None:
            prefix_input_ids = prefix_input_ids[:, 1:]
            prefix_attention_mask = prefix_attention_mask[:, 1:]

        # Add boi token
        prefix_input_ids = torch.cat(
            [
                prefix_input_ids,
                prefix_input_ids.new_tensor([self.boi]).unsqueeze(0),
            ],
            dim=1,
        )
        prefix_attention_mask = torch.cat(
            [
                prefix_attention_mask,
                prefix_attention_mask.new_ones((prefix_attention_mask.shape[0], 1)),
            ],
            dim=1,
        )

        bsz = input_ids.shape[0]
        input_ids = torch.cat([input_ids, prefix_input_ids.expand(bsz, -1)], dim=1)
        attention_mask = torch.cat([attention_mask, prefix_attention_mask.expand(bsz, -1)], dim=1)

        return input_ids, attention_mask

    @torch.no_grad()
    def decoding(
        self,
        c: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
        max_new_len: int,
        num_images_per_caption: int,
        use_norm: bool = False,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        cfg_mult: int = 1,
        cfg_schedule: Literal["linear", "constant"] = "constant",
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        progress: bool = True,
        hw: tuple[int, int] = (256, 256),
    ):
        """Autoregressive image token decoding with optional CFG-Parallel."""
        if cfg_mult <= 0:
            raise ValueError(f"Invalid cfg_mult={cfg_mult}; expected a positive value.")

        full_bsz = c.shape[0]
        if full_bsz % cfg_mult != 0:
            raise ValueError(
                f"Invalid CFG layout: batch size {full_bsz} is not divisible by active CFG multiplier {cfg_mult}."
            )

        # CFG-Parallel: each rank handles one portion of the CFG batch
        cfg_world_size = get_classifier_free_guidance_world_size()
        cfg_parallel = cfg_world_size > 1 and cfg_mult > 1
        if cfg_parallel and cfg_world_size != cfg_mult:
            logger.warning(
                "CFG parallel world size (%d) does not match the number of active "
                "CFG branches (%d); falling back to non-parallel CFG.",
                cfg_world_size,
                cfg_mult,
            )
            cfg_parallel = False
        if cfg_parallel:
            cfg_rank = get_classifier_free_guidance_rank()
            cfg_group = get_cfg_group()
            # Split batch: rank 0 gets positive, rank 1 gets first uncond, etc.
            batch_per_rank = full_bsz // cfg_mult
            start = cfg_rank * batch_per_rank
            end = start + batch_per_rank
            c = c[start:end]
            attention_mask = attention_mask[start:end]
            # Slice the StaticCache to keep only this rank's portion.
            # We rebuild a new StaticCache for this rank's batch size and copy
            # the relevant slices from the pre-filled cache.
            if isinstance(past_key_values, StaticCache):
                old_cache = past_key_values
                new_cache = StaticCache(
                    config=self.config,
                    max_cache_len=old_cache.get_max_cache_shape(),
                )
                for layer_idx, layer in enumerate(old_cache.layers):
                    if not layer.is_initialized:
                        continue
                    # Force-initialize the new layer with sliced tensors
                    new_layer = new_cache.layers[layer_idx]
                    sliced_keys = layer.keys[start:end]
                    new_layer.lazy_initialization(sliced_keys)
                    new_layer.keys.copy_(sliced_keys)
                    new_layer.values.copy_(layer.values[start:end])
                    # Preserve cumulative_length for sliding window layers
                    # (StaticSlidingWindowLayer tracks seq position via this
                    # counter, not by counting non-zero entries)
                    if hasattr(layer, "cumulative_length"):
                        new_layer.cumulative_length = layer.cumulative_length
                past_key_values = new_cache
        else:
            cfg_rank = 0
            cfg_group = None

        token_dim = self.config.latent_channels * self.config.latent_patch_size**2
        indices = list(range(max_new_len))
        indices = tqdm(indices, desc="Generating") if progress else indices

        tokens = None
        for step in indices:
            # CFG schedule
            if cfg_schedule == "linear":
                tokens_len = 0 if tokens is None else tokens.shape[1]
                cfg_iter = 1 + (cfg - 1) * (max_new_len - tokens_len) / max_new_len
                cfg_img_iter = 1 + (cfg_img - 1) * (max_new_len - tokens_len) / max_new_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
                cfg_img_iter = cfg_img
            else:
                raise NotImplementedError(f"Unknown cfg_schedule: {cfg_schedule}")

            if cfg_parallel:
                # Each rank projects its own portion
                c_proj = self.model.image_out_projector(c)
                # Gather projected context from all CFG ranks
                c_gathered = cfg_group.all_gather(c_proj, separate_tensors=True)
                c_full = torch.cat(c_gathered, dim=0)  # (full_bsz, 1, hidden)

                # Rank 0 runs the FM head sampling (it needs full CFG batch)
                if cfg_rank == 0:
                    token_sampled = self.model.image_head.sample(
                        c=c_full.squeeze(1),
                        cfg=cfg_iter,
                        cfg_img=cfg_img_iter,
                        cfg_mult=cfg_mult,
                        timesteps_shift=timesteps_shift,
                        num_sampling_steps=num_sampling_steps,
                        noise_repeat=num_images_per_caption,
                    )
                else:
                    token_sampled = torch.empty(
                        batch_per_rank,
                        token_dim,
                        device=c.device,
                        dtype=c.dtype,
                    )
                # Broadcast sampled token from rank 0 to all ranks
                token_sampled = token_sampled.contiguous()
                cfg_group.broadcast(token_sampled, src=0)
            else:
                c_proj = self.model.image_out_projector(c)
                token_sampled = self.model.image_head.sample(
                    c=c_proj.squeeze(1),
                    cfg=cfg_iter,
                    cfg_img=cfg_img_iter,
                    cfg_mult=cfg_mult,
                    timesteps_shift=timesteps_shift,
                    num_sampling_steps=num_sampling_steps,
                    noise_repeat=num_images_per_caption,
                )

            if use_norm:
                token_sampled = layer_norm(token_sampled, normalized_shape=token_sampled.size()[1:])

            if tokens is not None:
                tokens = torch.cat([tokens, token_sampled.unsqueeze(1)], dim=1)
            else:
                tokens = token_sampled.unsqueeze(1)

            # Prepare input embeds for next LLM step
            cur_inputs_embeds = self.model.image_in_projector(tokens[:, -1:])
            if not cfg_parallel and cfg_mult > 1:
                # Non-parallel: duplicate embeds for all active CFG branches.
                cur_inputs_embeds = torch.cat([cur_inputs_embeds] * cfg_mult, dim=0)
            # In CFG-parallel mode, each rank already has its portion — no duplication needed

            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            outputs = self.model.forward_model(
                inputs_embeds=cur_inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            c = outputs.last_hidden_state[:, -1:]
            if getattr(self.config, "use_gen_pos_embed", False):
                c = c + self.model.gen_pos_embed_with_ar(hw[0], hw[1])[:, step + 1 : step + 2, :]

        return tokens

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """
        Generate images from text prompts.

        Args:
            req: OmniDiffusionRequest containing generation parameters
            prompt: Text prompt(s) for generation
            height: Output image height
            width: Output image width
            num_inference_steps: Number of sampling steps (default 28 for NextStep-1.1)
            guidance_scale: CFG scale
            negative_prompt: Negative prompt for CFG
            num_images_per_prompt: Number of images per prompt
            generator: Random generator for reproducibility
            seed: Random seed

        Returns:
            DiffusionOutput containing generated images
        """
        # Extract parameters from request
        # req.prompts is a list of str or dict; req.sampling_params holds all generation params
        first_prompt = req.prompts[0] if req.prompts else None
        if first_prompt is not None:
            if isinstance(first_prompt, str):
                prompt = first_prompt
            else:
                prompt = first_prompt.get("prompt") or prompt
                negative_prompt = first_prompt.get("negative_prompt", negative_prompt)

        height = req.sampling_params.height or height or 512
        width = req.sampling_params.width or width or 512
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )
        seed = req.sampling_params.seed if req.sampling_params.seed is not None else seed

        # NextStep-specific parameters from request extra
        cfg_img = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else req.sampling_params.extra_args.get("cfg_img", 1.0)
        )
        timesteps_shift = req.sampling_params.extra_args.get("timesteps_shift", 1.0)
        use_norm = req.sampling_params.extra_args.get("use_norm", False)
        cfg_schedule = req.sampling_params.extra_args.get("cfg_schedule", "constant")
        positive_prompt = req.sampling_params.extra_args.get("positive_prompt", None)

        # Set seed for reproducibility (use generator if provided, else fall back to seed)
        if generator is None and seed is not None:
            set_seed(seed)
        elif generator is not None:
            torch.manual_seed(generator.initial_seed())

        # Prepare hw tuple
        hw = (height, width)

        # Check and process inputs (no image inputs for t2i)
        captions, images = self._check_input(prompt, None)

        # Build captions with CFG
        captions, images, cfg_mult, effective_cfg_img = self._build_captions(
            captions,
            images,
            num_images_per_prompt,
            positive_prompt,
            negative_prompt,
            guidance_scale,
            cfg_img,
        )

        # No input images for text-to-image
        latents = None

        # Add BOS token to captions before tokenizing
        captions = [
            self.tokenizer.bos_token + caption if self.tokenizer.bos_token is not None else caption
            for caption in captions
        ]

        # Tokenize captions and add prefix ids
        output = self.tokenizer(
            captions,
            padding="longest",
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = output.input_ids.to(self.device)
        attention_mask = output.attention_mask.to(self.device)
        input_ids, attention_mask = self._add_prefix_ids(hw, input_ids, attention_mask)

        # LLM prefill
        max_new_len = (hw[0] // self.down_factor) * (hw[1] // self.down_factor)
        max_cache_len = input_ids.shape[1] + max_new_len
        past_key_values = StaticCache(
            config=self.config,
            max_cache_len=max_cache_len,
        )
        inputs_embeds = self.model.prepare_inputs_embeds(input_ids, latents)
        outputs = self.model.forward_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        c = outputs.last_hidden_state[:, -1:]
        if getattr(self.config, "use_gen_pos_embed", False):
            c = c + self.model.gen_pos_embed_with_ar(height, width)[:, 0:1, :]

        # Decoding
        tokens = self.decoding(
            c=c,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_len=max_new_len,
            num_images_per_caption=num_images_per_prompt,
            use_norm=use_norm,
            cfg=guidance_scale,
            cfg_img=effective_cfg_img,
            cfg_mult=cfg_mult,
            cfg_schedule=cfg_schedule,
            timesteps_shift=timesteps_shift,
            num_sampling_steps=num_inference_steps,
            progress=True,
            hw=hw,
        )

        # Unpatchify
        latents = self.model.unpatchify(tokens)
        latents = (latents / self.scaling_factor) + self.shift_factor

        # Decode latents
        sampled_images = self.vae.decode(latents.to(self.vae.dtype)).sample
        sampled_images = sampled_images.detach().cpu().to(torch.float32)

        return DiffusionOutput(output=sampled_images)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
