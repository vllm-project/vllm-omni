# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/OpenMOSS/MOVA
"""
MOVA Pipeline for synchronized video-audio generation.

Orchestrates the dual-tower (video + audio) diffusion process with
cross-modal bridge conditioning, paired timestep scheduling, and
boundary-ratio expert switching.
"""

import html
import json
import os
import re
from glob import glob

import torch
import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .dac_vae import DAC
from .mova_audio_transformer import MovaAudioTransformer
from .mova_bridge import MovaBridge
from .mova_video_transformer import MovaVideoTransformer, sinusoidal_embedding_1d
from .scheduling_mova import FlowMatchPairScheduler

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Component loading helpers
# ---------------------------------------------------------------------------


def _load_config_from_subfolder(model_path: str, subfolder: str) -> dict:
    """Load config.json from a model subfolder."""
    config_path = os.path.join(model_path, subfolder, "config.json")
    if not os.path.exists(config_path):
        # Try scheduler_config.json for scheduler subfolder
        config_path = os.path.join(model_path, subfolder, "scheduler_config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    # Remove diffusers metadata keys
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _load_weights_from_subfolder(model_path: str, subfolder: str) -> dict[str, torch.Tensor]:
    """Load safetensors weights from a model subfolder."""
    subfolder_path = os.path.join(model_path, subfolder)
    safetensor_files = sorted(glob(os.path.join(subfolder_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {subfolder_path}")
    state_dict: dict[str, torch.Tensor] = {}
    for sf in safetensor_files:
        state_dict.update(load_safetensors(sf))
    return state_dict


def _load_module_from_subfolder(
    cls: type,
    model_path: str,
    subfolder: str,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Instantiate an nn.Module from config.json + safetensors in a subfolder."""
    cfg = _load_config_from_subfolder(model_path, subfolder)
    model = cls(**cfg)
    state_dict = _load_weights_from_subfolder(model_path, subfolder)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Missing keys when loading {subfolder}: {result.missing_keys[:10]}. Model checkpoint may be incompatible."
        )
    if result.unexpected_keys:
        logger.warning("Unexpected keys in %s (ignored): %s", subfolder, result.unexpected_keys[:10])
    model = model.to(dtype=dtype, device=device)
    return model


def _load_scheduler_from_subfolder(model_path: str, subfolder: str) -> FlowMatchPairScheduler:
    """Load FlowMatchPairScheduler from config in a subfolder."""
    cfg = _load_config_from_subfolder(model_path, subfolder)
    return FlowMatchPairScheduler(**cfg)


# ---------------------------------------------------------------------------
# Text cleaning utilities (from upstream)
# ---------------------------------------------------------------------------


def _basic_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return text.strip()


def _prompt_clean(text: str) -> str:
    text = _basic_clean(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _crop_and_resize_image(image: Image.Image, height: int, width: int) -> Image.Image:
    """Match upstream MOVA's PIL crop-and-resize preprocessing."""
    image_width, image_height = image.size
    target_ratio = width / height
    image_ratio = image_width / image_height

    if image_ratio > target_ratio:
        cropped_width = int(image_height * target_ratio)
        left = (image_width - cropped_width) // 2
        image = image.crop((left, 0, left + cropped_width, image_height))
    else:
        cropped_height = int(image_width / target_ratio)
        top = (image_height - cropped_height) // 2
        image = image.crop((0, top, image_width, top + cropped_height))

    return image.resize((width, height))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MovaPipeline(nn.Module, CFGParallelMixin, SupportImageInput, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    """
    MOVA pipeline for synchronized video-audio generation.

    Drives the dual-tower denoising loop with cross-modal bridge conditioning.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()

        model_path = od_config.model
        device = get_local_device()
        dtype = od_config.dtype
        self.parallel_config = od_config.parallel_config

        if getattr(self.parallel_config, "sequence_parallel_size", 1) > 1:
            raise NotImplementedError(
                "MOVA does not yet support sequence/context parallel execution in vllm-omni. "
                "Use single-GPU direct inference for parity checks; upstream SGLang workflow references "
                "that rely on context parallelism are not directly comparable."
            )

        # Load text encoder + tokenizer (diffusers/transformers native)
        logger.info("Loading text encoder from %s", model_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # Load video VAE (diffusers native)
        logger.info("Loading video VAE from %s", model_path)
        self.video_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="video_vae", torch_dtype=dtype).to(
            device
        )
        self.video_vae.eval()
        for p in self.video_vae.parameters():
            p.requires_grad_(False)

        self.vae_scale_factor_spatial = self.video_vae.config.scale_factor_spatial
        self.vae_scale_factor_temporal = self.video_vae.config.scale_factor_temporal
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Load audio VAE (DAC - custom nn.Module, manual loading)
        logger.info("Loading audio VAE from %s", model_path)
        self.audio_vae = _load_module_from_subfolder(DAC, model_path, "audio_vae", dtype, device)
        self.audio_vae.eval()
        for p in self.audio_vae.parameters():
            p.requires_grad_(False)

        self.audio_vae_scale_factor = int(self.audio_vae.hop_length)
        self.audio_sample_rate = self.audio_vae.sample_rate

        # Load scheduler (custom, manual loading)
        logger.info("Loading scheduler from %s", model_path)
        self.scheduler = _load_scheduler_from_subfolder(model_path, "scheduler")

        # Load components - device placement is managed by vllm-omni's offloader
        # framework (via enable_cpu_offload / enable_layerwise_offload).
        # Pipeline should NOT manually .to(device) — the diffusers_loader
        # controls load_device based on offload config.
        load_device = torch.device("cpu") if od_config.enable_cpu_offload else device

        # Load video DiT (high-noise expert)
        logger.info("Loading video_dit from %s", model_path)
        self.video_dit = _load_module_from_subfolder(MovaVideoTransformer, model_path, "video_dit", dtype, load_device)
        self.video_dit.eval()

        # Load video DiT 2 (low-noise expert) — always CPU initially
        # (swapped to GPU at boundary_ratio during inference)
        logger.info("Loading video_dit_2 from %s (to CPU, swap at boundary)", model_path)
        self.video_dit_2 = _load_module_from_subfolder(
            MovaVideoTransformer, model_path, "video_dit_2", dtype, torch.device("cpu")
        )
        self.video_dit_2.eval()

        # Load audio DiT
        logger.info("Loading audio_dit from %s", model_path)
        self.audio_dit = _load_module_from_subfolder(MovaAudioTransformer, model_path, "audio_dit", dtype, load_device)
        self.audio_dit.eval()

        # Load bridge
        logger.info("Loading dual_tower_bridge from %s", model_path)
        self.dual_tower_bridge = _load_module_from_subfolder(
            MovaBridge, model_path, "dual_tower_bridge", dtype, load_device
        )
        self.dual_tower_bridge.eval()

        # Config
        custom_args = od_config.custom_pipeline_args or {}
        self.boundary_ratio = custom_args.get("boundary_ratio", 0.9)
        self._use_cpu_offload = od_config.enable_cpu_offload

    def load_weights(self, weights):
        # Weights are loaded in __init__ via _load_module_from_subfolder.
        # This no-op satisfies the framework's load_weights call.
        pass

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def _retrieve_latents(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if hasattr(encoder_output, "latent_dist"):
            return encoder_output.latent_dist.mode()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        raise AttributeError("Could not access latents of provided encoder_output")

    def _normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        inv_std = (
            1.0 / torch.tensor(self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype)
        ).view(1, self.video_vae.config.z_dim, 1, 1, 1)
        return (latents - mean) * inv_std

    def _denormalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        std = torch.tensor(self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        return latents * std + mean

    def _prepare_video_latents(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video noise latents + first-frame conditioning."""
        num_channels = self.video_vae.config.z_dim
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial

        latents = torch.randn(1, num_channels, num_latent_frames, latent_h, latent_w, device=device, dtype=dtype)

        # First-frame conditioning
        image = image.unsqueeze(2)  # [1, C, 1, H, W]
        video_condition = torch.cat([image, image.new_zeros(1, image.shape[1], num_frames - 1, height, width)], dim=2)
        video_condition = video_condition.to(device=device, dtype=self.video_vae.dtype)
        latent_condition = self._retrieve_latents(self.video_vae.encode(video_condition))
        latent_condition = latent_condition.to(dtype)
        latent_condition = self._normalize_video_latents(latent_condition)

        # Build mask
        mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=device)
        mask[:, :, 1:] = 0
        first_frame_mask = mask[:, :, 0:1].repeat(1, 1, self.vae_scale_factor_temporal, 1, 1)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        mask = mask.view(1, -1, self.vae_scale_factor_temporal, latent_h, latent_w).transpose(1, 2)

        condition = torch.cat([mask, latent_condition], dim=1)
        return latents, condition

    def _prepare_audio_latents(
        self,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare audio noise latents."""
        latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
        return torch.randn(1, self.audio_vae.latent_dim, latent_t, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int = 512,
    ) -> torch.Tensor:
        prompt = _prompt_clean(prompt)
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        seq_len = mask.gt(0).sum(dim=1).long()

        embeds = self.text_encoder(input_ids, mask).last_hidden_state
        embeds = embeds.to(dtype=dtype, device=device)
        # Trim to actual length, then pad back to max_seq_len
        embeds = torch.stack(
            [torch.cat([e[:s], e.new_zeros(max_seq_len - s, e.size(1))]) for e, s in zip(embeds, seq_len)]
        )
        return embeds

    # ------------------------------------------------------------------
    # Single denoising step
    # ------------------------------------------------------------------

    def _inference_single_step(
        self,
        visual_dit: MovaVideoTransformer,
        visual_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        video_fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one denoising step through both towers + bridge."""
        # Time embeddings
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = visual_dit.time_embedding(sinusoidal_embedding_1d(visual_dit.freq_dim, timestep))
            visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))
            audio_t = self.audio_dit.time_embedding(sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep))
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))

        model_dtype = next(visual_dit.parameters()).dtype
        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)

        # Text embeddings
        visual_context = visual_dit.text_embedding(context)
        audio_context = self.audio_dit.text_embedding(context)

        # Patchify
        visual_x = visual_latents.to(model_dtype)
        audio_x = audio_latents.to(model_dtype)

        visual_x, grid_size = visual_dit.patchify(visual_x)
        t, h, w = grid_size
        audio_x, audio_grid = self.audio_dit.patchify(audio_x)
        f = audio_grid[0]

        # Assemble 3D visual RoPE: self.freqs is tuple of 3 tensors
        f_freqs, h_freqs, w_freqs = tuple(freq.to(visual_x.device) for freq in visual_dit.freqs)
        vf = torch.cat(
            [
                f_freqs[:t].view(t, 1, 1, -1).expand(t, h, w, -1),
                h_freqs[:h].view(1, h, 1, -1).expand(t, h, w, -1),
                w_freqs[:w].view(1, 1, w, -1).expand(t, h, w, -1),
            ],
            dim=-1,
        ).reshape(t * h * w, 1, -1)

        # Assemble 1D audio RoPE: self.freqs is tuple of 3 tensors
        af_parts = tuple(freq.to(audio_x.device) for freq in self.audio_dit.freqs)
        af = torch.cat(
            [af_parts[0][:f], af_parts[1][:f], af_parts[2][:f]],
            dim=-1,
        ).reshape(f, 1, -1)

        # Dual-tower forward with bridge
        visual_x, audio_x = self._forward_dual_tower_dit(
            visual_dit=visual_dit,
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context,
            audio_context=audio_context,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            visual_freqs=vf,
            audio_freqs=af,
            grid_size=(t, h, w),
            video_fps=video_fps,
        )

        # Head + unpatchify
        visual_output = visual_dit.head(visual_x, visual_t)
        visual_output = visual_dit.unpatchify(visual_output, (t, h, w))

        audio_output = self.audio_dit.head(audio_x, audio_t)
        audio_output = self.audio_dit.unpatchify(audio_output, (f,))

        return visual_output, audio_output

    # ------------------------------------------------------------------
    # Dual-tower forward (block loop with bridge)
    # ------------------------------------------------------------------

    def _forward_dual_tower_dit(
        self,
        visual_dit: MovaVideoTransformer,
        visual_x: torch.Tensor,
        audio_x: torch.Tensor,
        visual_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_t_mod: torch.Tensor,
        audio_t_mod: torch.Tensor,
        visual_freqs: torch.Tensor,
        audio_freqs: torch.Tensor,
        grid_size: tuple[int, int, int],
        video_fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run paired DiT blocks with bridge conditioning interleaved."""
        min_layers = min(len(visual_dit.blocks), len(self.audio_dit.blocks))
        visual_layers = len(visual_dit.blocks)

        # Cross-modal RoPE for bridge
        if self.dual_tower_bridge.apply_cross_rope:
            visual_rope_cos_sin, audio_rope_cos_sin = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=audio_x.shape[1],
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        # Paired block loop
        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # Bridge conditioning (at designated layers)
            if self.dual_tower_bridge.should_interact(layer_idx, "a2v"):
                visual_x, audio_x = self.dual_tower_bridge(
                    layer_idx,
                    visual_x,
                    audio_x,
                    x_freqs=visual_rope_cos_sin,
                    y_freqs=audio_rope_cos_sin,
                    video_grid_size=grid_size,
                )

            visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)
            audio_x = audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)

        # Remaining visual-only blocks
        for layer_idx in range(min_layers, visual_layers):
            visual_x = visual_dit.blocks[layer_idx](visual_x, visual_context, visual_t_mod, visual_freqs)

        return visual_x, audio_x

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Main inference entry point."""
        # Extract parameters from request
        r_prompts = req.prompts[0]
        if isinstance(r_prompts, str):
            prompt = r_prompts
            negative_prompt = ""
        else:
            prompt = r_prompts.get("prompt", "")
            negative_prompt = r_prompts.get("negative_prompt", "")

        multi_modal_data = r_prompts.get("multi_modal_data", {}) if not isinstance(r_prompts, str) else {}
        raw_image = multi_modal_data.get("image", None)
        if raw_image is None:
            # Only allow empty image for engine warmup (guidance_scale=0 is the warmup signal)
            if req.sampling_params.guidance_scale == 0.0:
                logger.debug("Warmup run (no image, guidance_scale=0), returning empty output")
                return DiffusionOutput(output=None)
            raise ValueError("MOVA I2VA mode requires an image input in multi_modal_data['image'].")
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        height = req.sampling_params.height or 352
        width = req.sampling_params.width or 640
        num_frames = req.sampling_params.num_frames or 193
        num_inference_steps = req.sampling_params.num_inference_steps or 50
        cfg_scale = req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else 5.0
        seed = req.sampling_params.seed if req.sampling_params.seed is not None else 42

        extra = req.sampling_params.extra_args or {}
        video_fps = extra.get("video_fps", 24.0)
        visual_shift = extra.get("visual_shift", 5.0)
        audio_shift = extra.get("audio_shift", 5.0)
        boundary_ratio = extra.get("boundary_ratio", self.boundary_ratio)

        if isinstance(raw_image, Image.Image):
            raw_image = _crop_and_resize_image(raw_image.convert("RGB"), height=height, width=width)

        device = get_local_device()
        use_offload = getattr(self, "_use_cpu_offload", False)

        # Restore device state for multi-request safety:
        # ensure encoding modules are on GPU at the start of each request.
        if use_offload:
            self.text_encoder = self.text_encoder.to(device)
            self.video_vae = self.video_vae.to(device)
            self.audio_vae = self.audio_vae.to(device)

        model_dtype = next(self.video_dit.parameters()).dtype

        # Check inputs
        target_div = self.vae_scale_factor_spatial * 2
        if height % target_div != 0 or width % target_div != 0:
            raise ValueError(f"height and width must be divisible by {target_div}, got {height} and {width}.")

        audio_num_samples = int(self.audio_sample_rate * num_frames / video_fps)

        # Set up scheduler with paired timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if hasattr(self.scheduler, "set_pair_postprocess_by_name"):
            self.scheduler.set_pair_postprocess_by_name(
                "dual_sigma_shift",
                visual_shift=visual_shift,
                audio_shift=audio_shift,
            )
        paired_timesteps = self.scheduler.get_pairs()

        # Prepare image
        image = self.video_processor.preprocess(raw_image, height=height, width=width).to(device, dtype=torch.float32)

        # Prepare latents
        torch.manual_seed(seed)
        latents, condition = self._prepare_video_latents(image, height, width, num_frames, device, torch.float32)
        audio_latents = self._prepare_audio_latents(audio_num_samples, device, torch.float32)

        # Encode prompts
        prompt_embeds = self._encode_prompt(prompt, device, model_dtype)
        negative_prompt_embeds = self._encode_prompt(negative_prompt, device, model_dtype)

        # Component-wise offload: free encoding modules, load DiT components.
        if use_offload:
            logger.info("CPU offload: moving text_encoder + VAEs to CPU, DiT components to GPU")
            self.text_encoder = self.text_encoder.cpu()
            self.video_vae = self.video_vae.cpu()
            self.audio_vae = self.audio_vae.cpu()
            torch.cuda.empty_cache()
            self.audio_dit = self.audio_dit.to(device)
            self.dual_tower_bridge = self.dual_tower_bridge.to(device)
            self.video_dit = self.video_dit.to(device)

        # Denoising loop
        cur_visual_dit = self.video_dit
        total_steps = paired_timesteps.shape[0]
        switched = False
        boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps

        for idx_step in tqdm(range(total_steps), desc="MOVA denoising"):
            timestep, audio_timestep = paired_timesteps[idx_step]

            # Switch to low-noise expert at boundary
            if not switched and timestep.item() < boundary_timestep:
                logger.info("Boundary (step %d/%d): switching to video_dit_2", idx_step, total_steps)
                if use_offload:
                    self.video_dit = self.video_dit.cpu()
                    torch.cuda.empty_cache()
                # Ensure video_dit_2 is on the right device (it starts on CPU)
                self.video_dit_2 = self.video_dit_2.to(device)
                cur_visual_dit = self.video_dit_2
                switched = True

            latent_model_input = torch.cat([latents, condition], dim=1)
            timestep_t = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_timestep_t = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)

            # Positive prediction
            noise_pred_pos = self._inference_single_step(
                visual_dit=cur_visual_dit,
                visual_latents=latent_model_input,
                audio_latents=audio_latents,
                context=prompt_embeds,
                timestep=timestep_t,
                audio_timestep=audio_timestep_t,
                video_fps=video_fps,
            )

            if cfg_scale == 1.0:
                visual_noise_pred = noise_pred_pos[0].float()
                audio_noise_pred = noise_pred_pos[1].float()
            else:
                # Negative prediction for CFG
                noise_pred_neg = self._inference_single_step(
                    visual_dit=cur_visual_dit,
                    visual_latents=latent_model_input,
                    audio_latents=audio_latents,
                    context=negative_prompt_embeds,
                    timestep=timestep_t,
                    audio_timestep=audio_timestep_t,
                    video_fps=video_fps,
                )
                v_pos, a_pos = noise_pred_pos[0].float(), noise_pred_pos[1].float()
                v_neg, a_neg = noise_pred_neg[0].float(), noise_pred_neg[1].float()
                visual_noise_pred = v_neg + cfg_scale * (v_pos - v_neg)
                audio_noise_pred = a_neg + cfg_scale * (a_pos - a_neg)

            # Scheduler step
            next_t = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
            next_a = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
            latents = self.scheduler.step_from_to(visual_noise_pred, timestep_t, next_t, latents)
            audio_latents = self.scheduler.step_from_to(audio_noise_pred, audio_timestep_t, next_a, audio_latents)

        # Swap DiT off GPU and reload VAEs for decoding (only with offload)
        if use_offload:
            if cur_visual_dit is self.video_dit_2:
                self.video_dit_2 = self.video_dit_2.cpu()
            else:
                self.video_dit = self.video_dit.cpu()
            self.audio_dit = self.audio_dit.cpu()
            self.dual_tower_bridge = self.dual_tower_bridge.cpu()
            torch.cuda.empty_cache()

            logger.info("CPU offload: moving VAEs back to GPU for decoding")
            self.video_vae = self.video_vae.to(device)

        # Decode video
        video_latents = self._denormalize_video_latents(latents)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = self.video_vae.decode(video_latents).sample
        video = self.video_processor.postprocess_video(video, output_type="pil")

        if use_offload:
            self.video_vae = self.video_vae.cpu()
            torch.cuda.empty_cache()
            self.audio_vae = self.audio_vae.to(device)

        # Decode audio
        with torch.autocast("cuda", dtype=torch.float32):
            audio = self.audio_vae.decode(audio_latents)

        return DiffusionOutput(output=(video, audio))


# ---------------------------------------------------------------------------
# Post-process function (required by registry)
# ---------------------------------------------------------------------------


def get_mova_post_process_func(od_config: OmniDiffusionConfig):
    """Return post-process function for MOVA outputs."""

    def post_process_func(output, output_type: str = "np"):
        # output is already (video_pil, audio_tensor) from pipeline forward
        # No additional post-processing needed
        return output

    return post_process_func
