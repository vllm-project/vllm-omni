# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Marey video generation pipeline for vllm_omni.

Orchestrates:
    - Text encoding (UL2 + CLIP + ByT5)
    - Rectified flow DDPM diffusion sampling
    - VAE decoding
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterable

import torch
import yaml
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.marey_serving_runtime import (
    MareyVAEInitializationError,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.marey.marey_transformer import MareyTransformer
from vllm_omni.diffusion.models.marey.vae_loader import load_vae
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


DEFAULT_NEGATIVE_PROMPT = (
    "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, "
    "vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, "
    "VR, transition, flare, saturation, distorted, warped, wide angle, contrast, "
    "saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, "
    "cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, "
    "horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, "
    "fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, "
    "boring, static"
)

# Checkpoint key remapping (training → inference naming)
_CHECKPOINT_KEY_REMAP = [
    ("y_embedder.vector_embedding_0", "y_embedder.vector_embedding"),
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_yaml_config(model_path: str) -> dict:
    """Load config.yaml from model directory."""
    config_path = os.path.join(model_path, "config.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(od_config: OmniDiffusionConfig) -> dict:
    """Load config.yaml from od_config.model, then apply od_config.model_config overrides.

    Relative paths in the ``vae.cp_path`` field are resolved against the model
    directory (where config.yaml lives) so a checkpoint with
    ``cp_path: vae.ckpt`` works regardless of the process CWD or the mount
    point (``/model``, ``/app/hf_checkpoints/…``, etc.).
    """
    config = _load_yaml_config(od_config.model)
    if od_config.model_config:
        _deep_update(config, od_config.model_config)

    vae_cfg = config.get("vae")
    if isinstance(vae_cfg, dict):
        cp_path = vae_cfg.get("cp_path", "")
        if cp_path and not os.path.isabs(cp_path):
            vae_cfg["cp_path"] = os.path.join(od_config.model, cp_path)

    return config


def _idx_to_latent_offset(
    idx: torch.Tensor,
    temporal_downsample_factor: int,
    frame_chunk_len: int | None,
    latent_time_offset: float,
    encode_as_images: bool,
    clip_as_keyframes: bool,
) -> torch.Tensor:
    """Map pixel-frame indices to fractional latent-time keyframe positions."""

    num_blocks = (idx // frame_chunk_len) + 1
    padding = (temporal_downsample_factor - frame_chunk_len % temporal_downsample_factor) % temporal_downsample_factor
    idx = idx + num_blocks * padding
    offset = idx / temporal_downsample_factor
    if encode_as_images:
        offset += latent_time_offset
    elif clip_as_keyframes:
        offset = offset[::temporal_downsample_factor]
    else:
        raise ValueError("Trying to encode a video, but clip_as_keyframes not enabled.")
    return offset


# ---------------------------------------------------------------------------
# Transformer construction
# ---------------------------------------------------------------------------


def _create_transformer_from_config(
    model_cfg: dict,
    te_cfg: dict,
    in_channels: int,
    caption_channels: list[int],
    vector_cond_channels: int | None,
) -> MareyTransformer:
    """Build a MareyTransformer from merged config sections."""
    depth = model_cfg.get("depth", 42)
    num_heads = model_cfg.get("num_heads", 40)
    head_dim = model_cfg.get("head_dim", 128)
    hidden_size = model_cfg.get("hidden_size", num_heads * head_dim)
    patch_size = tuple(model_cfg.get("patch_size", [1, 2, 2]))
    depth_single_blocks = model_cfg.get("depth_single_blocks", 28)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
    rope_dim = model_cfg.get("rope_dim", -1)
    rope_channels_ratio = model_cfg.get("rope_channels_ratio", 0.5)
    qk_norm = model_cfg.get("qk_norm", True)
    add_pos_embed_at_every_block = model_cfg.get("add_pos_embed_at_every_block", True)
    learned_pe = model_cfg.get("learned_pe", True)

    ul2_max_length = te_cfg.get("ul2_max_length", 300)
    byt5_max_length = te_cfg.get("byt5_max_length", 0)
    if byt5_max_length > 0:
        model_max_length = [ul2_max_length, byt5_max_length]
    else:
        model_max_length = ul2_max_length

    out_channels = model_cfg.get("out_channels", 2 * in_channels)
    extra_features_config = model_cfg.get("extra_features_embedders", None)
    camera_dim = model_cfg.get("camera_dim") if model_cfg.get("sequence_camera_condition", False) else None

    return MareyTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_size=patch_size,
        caption_channels=caption_channels,
        model_max_length=model_max_length,
        vector_cond_channels=vector_cond_channels,
        qk_norm=qk_norm,
        rope_channels_ratio=rope_channels_ratio,
        rope_dim=rope_dim,
        add_pos_embed_at_every_block=add_pos_embed_at_every_block,
        class_dropout_prob=0.0,
        learned_pe=learned_pe,
        extra_features_config=extra_features_config,
        camera_dim=camera_dim,
    )


# ---------------------------------------------------------------------------
# Extra features
# ---------------------------------------------------------------------------


def _build_extra_features(
    model_cfg: dict,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build extra feature tensors from config eval values for inference."""
    extra_cfg = model_cfg.get("extra_features_embedders", {})
    features: dict[str, torch.Tensor] = {}
    for feat, params in extra_cfg.items():
        if feat == "fps":
            continue
        ftype = params.get("type", "")
        if ftype == "SizeEmbedder":
            if feat == "ar":
                val = float(height) / float(width)
            else:
                val = 1.0
            features[feat] = torch.tensor([val], device=device, dtype=dtype)
        elif ftype in ("LabelEmbedder", "OrderedEmbedder"):
            eval_val = params.get("eval_value", -1)
            features[feat] = torch.tensor([eval_val], device=device, dtype=torch.long)
    return features


# ---------------------------------------------------------------------------
# Timestep / guidance schedule
# ---------------------------------------------------------------------------


def _create_flow_timesteps(
    num_steps: int,
    shift: float | None = None,
    num_train_timesteps: int = 1000,
    tmin: float = 0.001,
    tmax: float = 1.0,
    teacher_steps: int = 100,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Create rectified-flow timesteps for distilled inference.

    Matches the reference RFLOW.draw_time + RFlowScheduler.timestep_shift
    computation order exactly (float64 intermediate via .tolist()).
    """
    nt = num_train_timesteps
    sigmas_f64 = torch.linspace(tmax, tmin, teacher_steps).tolist()
    timesteps_all = [torch.tensor([t * nt], device=device) for t in sigmas_f64]

    if shift is not None and shift > 0:
        for i, t in enumerate(timesteps_all):
            t_norm = t / nt
            timesteps_all[i] = (shift * t_norm / (1.0 + (shift - 1.0) * t_norm)) * nt

    stride = max(1, teacher_steps // num_steps)
    return [timesteps_all[i * stride] for i in range(num_steps)]


def _build_guidance_schedule(
    num_steps: int,
    guidance_scale: float,
    warmup_steps: int = 4,
    cooldown_steps: int = 18,
    guidance_every_n_steps: int = 2,
) -> list[float]:
    """Build an oscillating guidance schedule matching the reference RFLOW scheduler.

    During warmup: CFG is always active.
    During middle: CFG oscillates (active every ``guidance_every_n_steps``).
    During cooldown: CFG is off (scale=1.0).
    """
    cooldown_start = num_steps - cooldown_steps

    schedule = []
    for i in range(num_steps):
        if i < warmup_steps:
            schedule.append(guidance_scale)
        elif i >= cooldown_start:
            schedule.append(1.0)
        else:
            middle_idx = i - warmup_steps
            if middle_idx > 0 and (middle_idx - 1) % guidance_every_n_steps == 0:
                schedule.append(guidance_scale)
            else:
                schedule.append(1.0)
    return schedule


# ---------------------------------------------------------------------------
# Post / pre process functions
# ---------------------------------------------------------------------------


def get_marey_post_process_func(od_config: OmniDiffusionConfig):
    """Return a function that converts decoded latents to output frames."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_marey_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process conditioning images for Marey I2V.

    Mirrors mv `marey_inference.py:_frame_conditions` (lines 230-297): for each
    conditioning image, letterbox-resize via ``ImageOps.fit`` to the target
    resolution, ``to_tensor``, then ``Normalize(mean=[0.5], std=[0.5])`` to land
    in ``[-1, 1]``. Stack into ``(1, C, T_keyframes, H, W)``. Store along with
    the parallel list of frame indices in ``additional_information`` for the
    pipeline's forward to pick up.
    """
    import numpy as np
    import PIL.Image
    from PIL import ImageOps
    from torchvision import transforms

    def _infer_target_size(image: PIL.Image.Image) -> tuple[int, int]:
        max_area = 720 * 1280
        aspect_ratio = image.height / image.width
        mod_value = 16
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return int(width), int(height)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                from vllm_omni.inputs.data import OmniTextPrompt

                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            multi_modal_data = prompt.get("multi_modal_data", {})
            raw_images = multi_modal_data.get("images")
            raw_frame_indices = multi_modal_data.get("frame_indices")

            # Back-compat: fall back to a single-image request via ``image`` key.
            if not raw_images:
                raw_image = multi_modal_data.get("image")
                if raw_image is None:
                    request.prompts[i] = prompt
                    continue
                raw_images = [raw_image]
                if raw_frame_indices is None:
                    raw_frame_indices = [0]

            if raw_frame_indices is None or len(raw_frame_indices) != len(raw_images):
                raise ValueError(
                    f"Marey I2V preprocessing: expected frame_indices with the same length as images "
                    f"(got {len(raw_frame_indices) if raw_frame_indices is not None else 'None'} indices "
                    f"for {len(raw_images)} images)."
                )

            # Load and pick a target (W, H) from the first image if not provided.
            pil_images = [
                PIL.Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB") for img in raw_images
            ]
            if request.sampling_params.height is None or request.sampling_params.width is None:
                w, h = _infer_target_size(pil_images[0])
                if request.sampling_params.height is None:
                    request.sampling_params.height = h
                if request.sampling_params.width is None:
                    request.sampling_params.width = w
            target_w = int(request.sampling_params.width)
            target_h = int(request.sampling_params.height)

            # Letterbox-fit to match mv (marey_inference.py:284-286).
            fitted = [ImageOps.fit(img, size=(target_w, target_h)) for img in pil_images]
            tensors = [transforms.functional.to_tensor(img) for img in fitted]
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tensors = [normalize(t) for t in tensors]
            cond_images = torch.stack(tensors, dim=1).unsqueeze(0)  # (1, C, T_keyframes, H, W) in [-1, 1]

            # Reject dense keyframe groups (consecutive runs of 4 starting at idx%4==0)
            # which would need the deferred split_vae_blocks_and_frames VAE-block path
            # (mv marey_inference.py:63-95). Phase 2 only implements the individual-frame path.
            sorted_idx = sorted(int(x) for x in raw_frame_indices)
            for k in range(len(sorted_idx) - 3):
                run = sorted_idx[k : k + 4]
                if run[0] % 4 == 0 and run == list(range(run[0], run[0] + 4)):
                    raise NotImplementedError(
                        f"Dense keyframe group {run} requires the VAE-block encoding path, "
                        f"which is not implemented yet. Use sparse keyframes (e.g. every 4+ frames)."
                    )

            prompt["multi_modal_data"]["images"] = fitted
            prompt["multi_modal_data"]["image"] = fitted[0]  # back-compat alias
            prompt["multi_modal_data"]["frame_indices"] = [int(x) for x in raw_frame_indices]
            prompt["additional_information"]["cond_images"] = cond_images
            prompt["additional_information"]["cond_frame_indices"] = [int(x) for x in raw_frame_indices]
            logger.info(
                "Marey I2V preprocess: cond_images.shape=%s cond_frame_indices=%s",
                tuple(cond_images.shape),
                prompt["additional_information"]["cond_frame_indices"],
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _extract_quotes(text: str) -> str:
    """Extract text between quotes for ByT5 encoding.
    Returns the full prompt if no quotes are found."""
    matches = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', text)
    return " ".join(matches) if matches else text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MareyPipeline(nn.Module, ProgressBarMixin):
    """Marey MMDiT video generation pipeline.

    All configuration is read from config.yaml in the model directory
    (od_config.model), with od_config.model_config providing overrides.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        # Load merged config
        self.config = _load_config(od_config)
        te_cfg = self.config.get("text_encoder", {})
        vae_cfg = self.config.get("vae", {})
        model_cfg = self.config.get("model", {})
        sched_cfg = self.config.get("scheduler", {})

        # Weight sources for DiffusersPipelineLoader
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # -- Text encoders ----------------------------------------------------
        ul2_name = te_cfg.get("ul2_pretrained", "google/ul2")
        self.ul2_max_length = te_cfg.get("ul2_max_length", 300)

        self.ul2_tokenizer = AutoTokenizer.from_pretrained(ul2_name)
        self.ul2_model = T5EncoderModel.from_pretrained(ul2_name, torch_dtype=dtype, device_map="cpu").eval()

        # Second text encoder: MetaCLIP-sequence or CLIP-vector + ByT5
        metaclip_name = te_cfg.get("metaclip_pretrained")
        if metaclip_name:
            # ul2-metaclip: MetaCLIP provides sequence features (last_hidden_state)
            self._metaclip_as_seq = True
            self.clip_max_length = te_cfg.get("metaclip_max_length", 77)
            self.byt5_max_length = te_cfg.get("byt5_max_length", self.clip_max_length)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(metaclip_name)
            self.clip_model = CLIPTextModel.from_pretrained(metaclip_name, torch_dtype=dtype, device_map="cpu").eval()
            # Alias MetaCLIP into the byt5 slot for reuse in encode_prompt
            self.byt5_tokenizer = self.clip_tokenizer
            self.byt5_model = self.clip_model
            caption_channels = [self.ul2_model.config.d_model, self.clip_model.config.hidden_size]
            vector_cond_channels = None
        else:
            # ul2-clip-quote (30B): CLIP pooled vector + ByT5 sequence
            self._metaclip_as_seq = False
            clip_name = te_cfg.get("clip_pretrained", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
            byt5_name = te_cfg.get("byt5_pretrained", "google/byt5-large")
            self.clip_max_length = te_cfg.get("clip_max_length", 77)
            self.byt5_max_length = te_cfg.get("byt5_max_length", 70)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)
            self.clip_model = CLIPTextModel.from_pretrained(clip_name, torch_dtype=dtype, device_map="cpu").eval()
            self.byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_name)
            self.byt5_model = T5EncoderModel.from_pretrained(byt5_name, torch_dtype=dtype, device_map="cpu").eval()
            caption_channels = [self.ul2_model.config.d_model, self.byt5_model.config.d_model]
            vector_cond_channels = self.clip_model.config.hidden_size

        try:
            self.vae = load_vae(vae_cfg, "cpu", dtype)
        except Exception as e:
            raise MareyVAEInitializationError(f"Marey VAE initialization failed. {str(e)}")

        self.vae_downsample_factors = tuple(self.vae.model.get_downsample_factors(0))
        self.vae_scale_factor_temporal = self.vae_downsample_factors[0]
        self.vae_scale_factor_spatial = self.vae_downsample_factors[1]

        # -- Transformer ------------------------------------------------------
        in_channels = self.vae.latent_dim
        self.transformer = _create_transformer_from_config(
            model_cfg,
            te_cfg,
            in_channels,
            caption_channels,
            vector_cond_channels,
        )

        # -- Scheduler config -------------------------------------------------
        self.num_train_timesteps = 1000
        self.sched_tmin = sched_cfg.get("tmin", 0.001)
        self.sched_tmax = sched_cfg.get("tmax", 1.0)
        self.sched_teacher_steps = sched_cfg.get("num_sampling_steps", 100)
        # flow_shift comes from the CLI (--flow-shift). `0.0` means "skip the
        # timestep transform" (see `shift=... if > 0 else None` below); pass
        # `3.0` for 30B (distilled, transform on) and `0.0` for 7B
        self.flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0

        # -- Guidance defaults ------------------------------------------------
        self.skip_uncond = True  # distilled models skip uncond when scale=1.0
        self.default_clip_value = 10.0
        self.default_warmup_steps = 4
        self.default_cooldown_steps = 18
        self.default_guidance_every_n_steps = 2

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    # -- Properties -----------------------------------------------------------

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

    # -- Text encoding -------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        quote_override: str | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Encode text using UL2 (sequence), CLIP (vector), and ByT5 (quotes).

        Returns (seq_cond, seq_cond_masks, vector_cond).
        """
        # UL2
        ul2_inputs = self.ul2_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.ul2_max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ul2_mask = ul2_inputs["attention_mask"].to(device, torch.bool)
        with torch.no_grad():
            ul2_output = self.ul2_model(
                input_ids=ul2_inputs.input_ids.to(device),
                attention_mask=ul2_inputs.attention_mask.to(device),
            )
            ul2_seq = ul2_output.last_hidden_state.to(dtype)

        # CLIP
        clip_inputs = self.clip_tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            clip_output = self.clip_model(input_ids=clip_inputs["input_ids"].to(device))
            vector_cond = clip_output.pooler_output.to(dtype)

        # ByT5 / MetaCLIP-sequence
        if self._metaclip_as_seq:
            seq2_text = prompt
        else:
            seq2_text = quote_override if quote_override is not None else _extract_quotes(prompt)
        byt5_inputs = self.byt5_tokenizer(
            seq2_text,
            padding="max_length",
            max_length=self.byt5_max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        byt5_mask = byt5_inputs["attention_mask"].to(device, torch.bool)
        with torch.no_grad():
            byt5_output = self.byt5_model(
                input_ids=byt5_inputs.input_ids.to(device),
                attention_mask=byt5_inputs.attention_mask.to(device),
            )
            byt5_seq = byt5_output.last_hidden_state.to(dtype)

        seq_cond = [ul2_seq, byt5_seq]
        seq_cond_masks = [ul2_mask, byt5_mask]
        return seq_cond, seq_cond_masks, vector_cond

    # -- Latent preparation --------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        time_pad = (
            0
            if (num_frames % self.vae_scale_factor_temporal == 0)
            else (self.vae_scale_factor_temporal - num_frames % self.vae_scale_factor_temporal)
        )
        num_latent_frames = (num_frames + time_pad) // self.vae_scale_factor_temporal
        latent_h = math.ceil(height / self.vae_scale_factor_spatial)
        latent_w = math.ceil(width / self.vae_scale_factor_spatial)
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_h, latent_w)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # -- Conditioning-frame encoding (I2V) -----------------------------------

    def _encode_cond_frames(
        self,
        cond_images: torch.Tensor,
        cond_frame_indices: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode I2V conditioning images → ``cond_frames`` latent + ``cond_offsets``.

        Uses VAE on each keyframe independently. Only the
        individual-frame path is implemented; dense VAE-block keyframe groups
        are rejected earlier in the preprocess hook.

        Args:
            cond_images: ``(1, C=3, T_keyframes, H, W)`` in ``[-1, 1]``.
            cond_frame_indices: List of target frame indices in the output video.

        Returns:
            ``(cond_frames, cond_offsets)`` — ``cond_frames`` shaped
            ``(1, C_vae, T_keyframes, H_lat, W_lat)`` in transformer ``dtype``;
            ``cond_offsets`` a 1-D int64 tensor of length ``T_keyframes``.
        """
        from einops import rearrange

        model_cfg = self.config.get("model", {})
        latent_time_offset = model_cfg.get("latent_time_offset", -0.75)

        # (1, C, T_kf, H, W) → (T_kf, C, 1, H, W): VAE treats each keyframe independently.
        x = cond_images.to(device=device, dtype=dtype)
        T_kf = x.shape[2]
        x = rearrange(x, "B C (Tl L) H W -> (B Tl) C L H W", L=1)

        with torch.no_grad():
            enc = self.vae.encode(x)  # (T_kf, C_vae, 1, H_lat, W_lat)
        enc = enc.squeeze(2)
        cond_frames = rearrange(enc, "(B T) C H W -> B C T H W", T=T_kf).contiguous().to(dtype=dtype)

        indices_tensor = torch.tensor(cond_frame_indices, device=device, dtype=torch.int64)
        cond_offsets = _idx_to_latent_offset(
            indices_tensor,
            self.vae_scale_factor_temporal,
            self.vae.cfg.frame_chunk_len,
            latent_time_offset,
            encode_as_images=True,
            clip_as_keyframes=True,
        )
        return cond_frames, cond_offsets

    # -- Forward (DDPM flow-matching loop) -----------------------------------

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # -- Extract parameters from request ----------------------------------
        if len(req.prompts) != 1:
            raise ValueError("This model only supports a single prompt per request.")

        raw_prompt = req.prompts[0]
        prompt = raw_prompt if isinstance(raw_prompt, str) else raw_prompt.get("prompt")
        negative_prompt = None if isinstance(raw_prompt, str) else raw_prompt.get("negative_prompt")
        if prompt is None:
            raise ValueError("Prompt is required.")

        sp = req.sampling_params
        height = sp.height or 720
        width = sp.width or 1280
        num_frames = sp.num_frames if sp.num_frames else 33
        num_steps = sp.num_inference_steps or 100
        guidance_scale = sp.guidance_scale if sp.guidance_scale_provided else 7.5
        self._guidance_scale = guidance_scale

        device = self.device
        dtype = self.transformer.dtype

        generator = sp.generator
        if generator is None and sp.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sp.seed)
        elif generator is None:
            generator = torch.Generator(device=device).manual_seed(0)  # default seed

        # -- Text encoding (offload transformer, load encoders) ---------------
        self.transformer.to("cpu")
        torch.cuda.empty_cache()
        self.ul2_model.to(device)
        self.clip_model.to(device)
        self.byt5_model.to(device)

        # Reference passes quote_override="" (ByT5 encodes empty string)
        prompt_embeds, prompt_masks, vector_cond = self.encode_prompt(
            prompt,
            device,
            dtype,
            quote_override="",
        )

        use_cfg = guidance_scale > 1.0
        negative_prompt_embeds = None
        negative_prompt_masks = None
        negative_vector_cond = None
        if use_cfg:
            neg_text = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
            negative_prompt_embeds, negative_prompt_masks, negative_vector_cond = self.encode_prompt(
                neg_text,
                device,
                dtype,
                quote_override="",
            )

        # Offload encoders, reload transformer
        self.ul2_model.to("cpu")
        self.clip_model.to("cpu")
        self.byt5_model.to("cpu")
        torch.cuda.empty_cache()

        # -- I2V: encode conditioning frames (VAE only, transformer still offloaded) --
        cond_frames: torch.Tensor | None = None
        cond_offsets: torch.Tensor | None = None
        additional_information = raw_prompt.get("additional_information", {}) if not isinstance(raw_prompt, str) else {}
        cond_images = additional_information.get("cond_images")
        cond_frame_indices = additional_information.get("cond_frame_indices")
        if cond_images is not None and cond_frame_indices:
            self.vae.to(device)
            cond_frames, cond_offsets = self._encode_cond_frames(
                cond_images,
                cond_frame_indices,
                device=device,
                dtype=dtype,
            )
            self.vae.to("cpu")
            torch.cuda.empty_cache()
            logger.info(
                "Marey I2V forward: cond_frames.shape=%s cond_offsets=%s",
                tuple(cond_frames.shape),
                cond_offsets.tolist(),
            )

        self.transformer.to(device)

        # -- Timesteps --------------------------------------------------------
        timesteps = _create_flow_timesteps(
            num_steps=num_steps,
            shift=self.flow_shift if self.flow_shift > 0 else None,
            num_train_timesteps=self.num_train_timesteps,
            tmin=self.sched_tmin,
            tmax=self.sched_tmax,
            teacher_steps=self.sched_teacher_steps,
            device=device,
        )
        self._num_timesteps = len(timesteps)

        # -- Guidance schedule ------------------------------------------------
        guidance_schedule: list[float] | None = None
        if use_cfg:
            guidance_schedule = _build_guidance_schedule(
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                warmup_steps=self.default_warmup_steps,
                cooldown_steps=self.default_cooldown_steps,
                guidance_every_n_steps=self.default_guidance_every_n_steps,
            )

        clip_value = self.default_clip_value

        # -- Prepare latents --------------------------------------------------
        num_channels_latents = self.transformer.in_channels

        # print(f'sp.latents: {sp.latents}')

        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=sp.latents,
        )

        # -- Extra features ---------------------------------------------------
        model_cfg = self.config.get("model", {})
        extra_features = _build_extra_features(model_cfg, height, width, device, dtype)

        # Quality guidance: cond gets quality=0, uncond gets quality=9
        uncond_extra_features = None
        # if use_cfg:
        #     uncond_extra_features = dict(extra_features)
        #     for qkey in ("dover_technical", "aesthetics_score_total"):
        #         if qkey in extra_features:
        #             uncond_extra_features[qkey] = torch.tensor([9], device=device, dtype=torch.long)
        #             extra_features[qkey] = torch.tensor([0], device=device, dtype=torch.long)

        _uncond_ef = uncond_extra_features if uncond_extra_features is not None else extra_features

        # -- Conditioning tensors ---------------------------------------------
        height_t = torch.tensor([height], device=device, dtype=dtype)
        width_t = torch.tensor([width], device=device, dtype=dtype)
        fps_value = sp.fps if sp.fps else 24
        fps_t = torch.tensor([float(fps_value)], device=device, dtype=dtype)

        has_neg = negative_prompt_embeds is not None
        in_channels = self.transformer.in_channels

        def _model_forward(z_in, t_in, text_emb, vec_cond, text_mask=None, ef=None):
            raw = self.transformer(
                hidden_states=z_in.to(dtype),
                timestep=t_in,
                encoder_hidden_states=text_emb,
                encoder_hidden_states_mask=text_mask,
                vector_cond=vec_cond,
                height=height_t,
                width=width_t,
                fps=fps_t,
                extra_features=ef if ef is not None else extra_features,
                cond_frames=cond_frames,
                cond_offsets=cond_offsets,
                return_dict=False,
            )[0]
            if raw.shape[1] != in_channels:
                raw = raw[:, :in_channels]
            return raw

        # -- DDPM flow-matching denoising loop --------------------------------
        z = latents
        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                t_input = t.expand(1)

                # Per-step guidance scale
                gs_i = guidance_schedule[i] if guidance_schedule is not None else guidance_scale
                use_uncond = has_neg and ((not self.skip_uncond) or (gs_i > 1.0))

                pred_cond = _model_forward(
                    z,
                    t_input,
                    prompt_embeds,
                    vector_cond,
                    prompt_masks,
                    ef=extra_features,
                )

                if use_uncond:
                    pred_uncond = _model_forward(
                        z,
                        t_input,
                        negative_prompt_embeds,
                        negative_vector_cond,
                        negative_prompt_masks,
                        ef=_uncond_ef,
                    )
                    v_pred = pred_uncond + gs_i * (pred_cond - pred_uncond)
                else:
                    v_pred = pred_cond

                # DDPM flow-matching step
                sigma_t = (t / self.num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x0 = z - sigma_t * v_pred
                if clip_value is not None and clip_value > 0:
                    x0 = torch.clamp(x0, -clip_value, clip_value)

                if i < len(timesteps) - 1:
                    sigma_s = (
                        (timesteps[i + 1] / self.num_train_timesteps)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    alpha_t = 1.0 - sigma_t
                    alpha_s = 1.0 - sigma_s
                    alpha_ts = alpha_t / alpha_s
                    alpha_ts_sq = alpha_ts**2
                    sigma_s_div_t_sq = (sigma_s / sigma_t) ** 2
                    sigma_ts_div_t_sq = 1.0 - alpha_ts_sq * sigma_s_div_t_sq

                    mean = alpha_ts * sigma_s_div_t_sq * z + alpha_s * sigma_ts_div_t_sq * x0
                    variance = sigma_ts_div_t_sq * sigma_s**2

                    noise = torch.randn_like(z, generator=generator)
                    z = mean + torch.sqrt(variance) * noise
                else:
                    z = x0

                pbar.update()

        self._current_timestep = None

        # -- VAE decode -------------------------------------------------------
        self.transformer.to("cpu")
        torch.cuda.empty_cache()
        output_type = sp.output_type or self.od_config.output_type or "np"
        if output_type == "latent":
            output = z
        else:
            self.vae.to(device)
            vae_t_ds = self.vae_downsample_factors[0]
            num_latent_t = z.shape[2]
            num_pixel_frames = num_latent_t * vae_t_ds
            chunk = self.vae.cfg.frame_chunk_len
            if num_latent_t <= 1:
                num_pixel_frames = 1
            elif chunk is not None and num_pixel_frames % chunk != 0:
                num_pixel_frames = (num_pixel_frames // chunk) * chunk

            with torch.no_grad():
                output = self.vae.decode(
                    z.to(dtype),
                    num_frames=num_pixel_frames,
                    spatial_size=(height, width),
                )
            if isinstance(output, tuple):
                output = output[0]
            self.vae.to("cpu")
        self.transformer.to(device)
        torch.cuda.empty_cache()

        return DiffusionOutput(output=output)

    # -- Weight loading ------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Ignore the provided weights iterator (dummy safetensors) and load
        # from the actual checkpoint, matching the offline inference loader.
        import glob

        import safetensors.torch

        model_dir = self.od_config.model
        patterns = [
            os.path.join(model_dir, "epoch0-*", "ema_inference_ckpt.safetensors"),
            os.path.join(model_dir, "**", "ema_inference_ckpt.safetensors"),
        ]
        ckpt_path = None
        for pat in patterns:
            matches = sorted(glob.glob(pat, recursive=True))
            if matches:
                ckpt_path = matches[0]
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"Cannot find ema_inference_ckpt.safetensors under {model_dir}")

        logger.info("Loading transformer weights from %s", ckpt_path)
        state_dict = safetensors.torch.load_file(ckpt_path)

        def _remap_items(sd: dict[str, torch.Tensor]):
            for name, tensor in sd.items():
                r = name
                for old, new in _CHECKPOINT_KEY_REMAP:
                    if old in r:
                        r = r.replace(old, new)
                        break
                yield r, tensor

        self.transformer.load_weights(_remap_items(state_dict))

        del state_dict
        torch.cuda.empty_cache()

        return {name for name, _ in self.named_parameters()}
