# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Helios (https://github.com/BestWishYsh/Helios)

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.diffusion.models.helios.scheduling_helios import HeliosScheduler
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

logger = logging.getLogger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def optimized_scale(positive_flat, negative_flat):
    positive_flat = positive_flat.float()
    negative_flat = negative_flat.float()
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm
    return st_star


def load_json_config(model_path: str, subfolder: str, filename: str, local_files_only: bool = True) -> dict:
    """Load a JSON config file from a local path or HuggingFace Hub repo."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, filename)
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/{filename}",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    return load_json_config(model_path, subfolder, "config.json", local_files_only)


def create_transformer_from_config(
    config: dict, quant_config: QuantizationConfig | None = None
) -> HeliosTransformer3DModel:
    kwargs = {}

    key_map = [
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "qk_norm",
        "eps",
        "added_kv_proj_dim",
        "rope_dim",
        "rope_theta",
        "guidance_cross_attn",
        "zero_history_timestep",
        "has_multi_term_memory_patch",
        "is_amplify_history",
        "history_scale_mode",
    ]
    for key in key_map:
        if key in config:
            val = config[key]
            if key in ("patch_size", "rope_dim") and isinstance(val, list):
                val = tuple(val)
            kwargs[key] = val

    return HeliosTransformer3DModel(quant_config=quant_config, **kwargs)


def get_helios_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_helios_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return request

    return pre_process_func


class HeliosPipeline(
    nn.Module, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery
):
    """Helios text-to-video / image-to-video / video-to-video pipeline for vllm-omni.

    Supports T2V, I2V (with image input), and V2V (with video input).
    Implements chunked video generation with multi-term memory history context.
    """

    supports_step_execution: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

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

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        # Helios checkpoints store embed_tokens under ``shared.weight`` only,
        # but the published config sets ``tie_word_embeddings=False``.  When
        # transformers sees ``tie=False`` it creates a separate
        # ``encoder.embed_tokens.weight`` that is never loaded from the
        # checkpoint, leaving it as all-zeros.  This silently destroys prompt
        # encoding and produces grey/meaningless video output.  Force tying so
        # that ``embed_tokens`` shares ``shared.weight`` as intended.
        text_enc_cfg = AutoConfig.from_pretrained(model, subfolder="text_encoder", local_files_only=local_files_only)
        text_enc_cfg.tie_word_embeddings = True
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", config=text_enc_cfg, torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(
            transformer_config, quant_config=od_config.quantization_config
        )

        # Read scheduler config to determine scheduler type
        sched_cfg = load_json_config(model, "scheduler", "scheduler_config.json", local_files_only)
        scheduler_kwargs = {}
        passthrough_keys = [
            "num_train_timesteps",
            "shift",
            "stages",
            "stage_range",
            "gamma",
            "thresholding",
            "prediction_type",
            "solver_order",
            "predict_x0",
            "solver_type",
            "lower_order_final",
            "disable_corrector",
            "use_flow_sigmas",
            "scheduler_type",
            "use_dynamic_shifting",
            "time_shift_type",
        ]
        for key in passthrough_keys:
            if key in sched_cfg:
                scheduler_kwargs[key] = sched_cfg[key]

        self.scheduler = HeliosScheduler(**scheduler_kwargs)

        self.is_distilled = scheduler_kwargs.get("scheduler_type") == "dmd"

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

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

    def _stage1_sigmas(self, num_steps: int) -> np.ndarray:
        # DMD drops the last timestep in set_timesteps(). Only compensate for
        # the one-step dummy-run edge case; otherwise preserve the historical
        # stage-1 schedule used by request-level forward().
        sigma_count = num_steps + 2 if self.is_distilled and num_steps == 1 else num_steps + 1
        return np.linspace(0.999, 0.0, sigma_count)[:-1]

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Initialize Helios request state for chunk-wise step execution."""
        del kwargs
        # Wrap the single request in a DiffusionRequestBatch so the batch
        # compatibility properties (`prompts`, etc.) used below are available;
        # OmniDiffusionRequest itself only exposes a singular `prompt`.
        req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt=state.prompt,
                    sampling_params=state.sampling,
                    request_id=state.request_id,
                )
            ]
        )
        extra = getattr(state.sampling, "extra_args", {}) or {}

        history_sizes = sorted(extra.get("history_sizes", [16, 2, 1]), reverse=True)
        num_latent_frames_per_chunk = int(extra.get("num_latent_frames_per_chunk", 9))
        keep_first_frame = bool(extra.get("keep_first_frame", True))
        frame_num = int(extra.get("frame_num", 132))
        height = (int(state.sampling.height or extra.get("height", 384)) // 16) * 16
        width = (int(state.sampling.width or extra.get("width", 640)) // 16) * 16
        num_frames = max(int(state.sampling.num_frames or frame_num), 1)
        num_steps = int(state.sampling.num_inference_steps or extra.get("num_inference_steps", 50))
        output_type = extra.get("output_type", "np")

        image = extra.get("image")
        video = extra.get("video")
        if image is not None and video is not None:
            raise ValueError("image and video cannot be provided simultaneously")
        if len(req.prompts) > 1:
            raise ValueError("Helios step execution supports a single prompt, not a batched request.")

        prompt = None
        negative_prompt = None
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None:
            raise ValueError("Prompt is required for Helios generation.")

        guidance_scale = float(extra.get("guidance_scale", 5.0))
        if state.sampling.guidance_scale_provided:
            guidance_scale = state.sampling.guidance_scale
        self._guidance_scale = guidance_scale

        device = self.device
        dtype = self.transformer.dtype
        generator = state.sampling.generator
        if generator is None and state.sampling.seed is not None:
            generator = torch.Generator(device=device).manual_seed(state.sampling.seed)
            state.sampling.generator = generator

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=state.sampling.num_outputs_per_prompt or 1,
            max_sequence_length=state.sampling.max_sequence_length or 226,
            device=device,
            dtype=dtype,
        )
        batch_size = prompt_embeds.shape[0]

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )

        add_noise_to_image_latents = bool(extra.get("add_noise_to_image_latents", True))
        image_noise_sigma_min = float(extra.get("image_noise_sigma_min", 0.111))
        image_noise_sigma_max = float(extra.get("image_noise_sigma_max", 0.135))
        add_noise_to_video_latents = bool(extra.get("add_noise_to_video_latents", True))
        video_noise_sigma_min = float(extra.get("video_noise_sigma_min", 0.111))
        video_noise_sigma_max = float(extra.get("video_noise_sigma_max", 0.135))

        fake_image_latents = None
        image_latents = None
        if image is not None:
            image_latents, fake_image_latents = self.prepare_image_latents(
                image,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

        if image_latents is not None and add_noise_to_image_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            fake_image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min)
                + video_noise_sigma_min
            )
            fake_image_latents = (
                fake_image_noise_sigma * randn_tensor(fake_image_latents.shape, generator=generator, device=device)
                + (1 - fake_image_noise_sigma) * fake_image_latents
            )

        video_latents = None
        if video is not None:
            image_latents, video_latents = self.prepare_video_latents(
                video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

        if video_latents is not None and add_noise_to_video_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            noisy_latents_chunks = []
            num_latent_chunks = video_latents.shape[2] // num_latent_frames_per_chunk
            for i in range(num_latent_chunks):
                chunk_start = i * num_latent_frames_per_chunk
                chunk_end = chunk_start + num_latent_frames_per_chunk
                latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]
                chunk_frames = latent_chunk.shape[2]
                frame_sigmas = (
                    torch.rand(chunk_frames, device=device, generator=generator)
                    * (video_noise_sigma_max - video_noise_sigma_min)
                    + video_noise_sigma_min
                )
                frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
                noisy_chunk = (
                    frame_sigmas * randn_tensor(latent_chunk.shape, generator=generator, device=device)
                    + (1 - frame_sigmas) * latent_chunk
                )
                noisy_latents_chunks.append(noisy_chunk)
            video_latents = torch.cat(noisy_latents_chunks, dim=2)

        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        num_latent_chunk = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        num_history_latent_frames = sum(history_sizes)
        total_generated_latent_frames = 0
        if not keep_first_frame:
            history_sizes[-1] = history_sizes[-1] + 1

        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents[:, :, :-1, :, :], fake_image_latents], dim=2)
            total_generated_latent_frames += 1
        if video_latents is not None:
            history_frames = history_latents.shape[2]
            video_frames = video_latents.shape[2]
            if video_frames < history_frames:
                keep_frames = history_frames - video_frames
                history_latents = torch.cat([history_latents[:, :, :keep_frames, :, :], video_latents], dim=2)
            else:
                history_latents = video_latents
            total_generated_latent_frames += video_latents.shape[2]

        if keep_first_frame:
            indices = torch.arange(0, sum([1, *history_sizes, num_latent_frames_per_chunk]))
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        else:
            indices = torch.arange(0, sum([*history_sizes, num_latent_frames_per_chunk]))
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)

        state.prompt_embeds = prompt_embeds
        state.negative_prompt_embeds = negative_prompt_embeds
        state.do_true_cfg = self.do_classifier_free_guidance
        state.chunk_index = 0
        state.step_in_chunk = 0
        state.total_chunks = num_latent_chunk
        state.extra.update(
            {
                "attention_kwargs": extra.get("attention_kwargs", {}) or {},
                "batch_size": batch_size,
                "dtype": dtype,
                "generator": generator,
                "guidance_scale": guidance_scale,
                "height": height,
                "history_latents": history_latents,
                "history_sizes": history_sizes,
                "history_video": None,
                "image_latents": image_latents,
                "indices_hidden_states": indices_hidden_states.unsqueeze(0),
                "indices_latents_history_long": indices_latents_history_long.unsqueeze(0),
                "indices_latents_history_mid": indices_latents_history_mid.unsqueeze(0),
                "indices_latents_history_short": indices_latents_history_short.unsqueeze(0),
                "is_amplify_first_chunk": bool(extra.get("is_amplify_first_chunk", False)),
                "is_enable_stage2": bool(extra.get("is_enable_stage2", False)),
                "is_skip_first_chunk": bool(extra.get("is_skip_first_chunk", False)),
                "keep_first_frame": keep_first_frame,
                "latents_mean": latents_mean,
                "latents_std": latents_std,
                "num_channels_latents": num_channels_latents,
                "num_history_latent_frames": num_history_latent_frames,
                "num_latent_frames_per_chunk": num_latent_frames_per_chunk,
                "num_steps": num_steps,
                "output_type": output_type,
                "pyramid_num_inference_steps_list": extra.get("pyramid_num_inference_steps_list", [10, 10, 10]),
                "pyramid_num_stages": int(extra.get("pyramid_num_stages", 3)),
                "total_generated_latent_frames": total_generated_latent_frames,
                "use_cfg_zero_star": bool(extra.get("use_cfg_zero_star", False)),
                "use_zero_init": bool(extra.get("use_zero_init", True)),
                "vae_dtype": self.vae.dtype,
                "width": width,
                "window_num_frames": window_num_frames,
                "zero_steps": int(extra.get("zero_steps", 1)),
            }
        )
        self._prepare_next_chunk(state)
        return state

    def _prepare_next_chunk(self, state: DiffusionRequestState) -> None:
        extra = state.extra
        k = state.chunk_index
        is_first_chunk = k == 0
        is_second_chunk = k == 1
        history_sizes = extra["history_sizes"]
        history_latents = extra["history_latents"]
        keep_first_frame = extra["keep_first_frame"]
        image_latents = extra["image_latents"]
        num_history_latent_frames = extra["num_history_latent_frames"]

        if keep_first_frame:
            latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                :, :, -num_history_latent_frames:
            ].split(history_sizes, dim=2)
            if image_latents is None and is_first_chunk:
                latents_prefix = torch.zeros(
                    (
                        extra["batch_size"],
                        extra["num_channels_latents"],
                        1,
                        latents_history_1x.shape[-2],
                        latents_history_1x.shape[-1],
                    ),
                    device=latents_history_1x.device,
                    dtype=latents_history_1x.dtype,
                )
            else:
                latents_prefix = image_latents
            latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
        else:
            latents_history_long, latents_history_mid, latents_history_short = history_latents[
                :, :, -num_history_latent_frames:
            ].split(history_sizes, dim=2)

        state.latents = self.prepare_latents(
            extra["batch_size"],
            extra["num_channels_latents"],
            extra["height"],
            extra["width"],
            extra["window_num_frames"],
            dtype=torch.float32,
            device=self.device,
            generator=extra["generator"],
            latents=None,
        )
        extra["stage1_start_latents"] = state.latents
        extra.update(
            {
                "is_first_chunk": is_first_chunk,
                "is_second_chunk": is_second_chunk,
                "latents_history_long": latents_history_long,
                "latents_history_mid": latents_history_mid,
                "latents_history_short": latents_history_short,
            }
        )

        if not extra["is_enable_stage2"]:
            patch_size = self.transformer.config.patch_size
            image_seq_len = (state.latents.shape[-1] * state.latents.shape[-2] * state.latents.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            sigmas = self._stage1_sigmas(extra["num_steps"])
            mu = calculate_shift(image_seq_len)
            self.scheduler.set_timesteps(extra["num_steps"], device=self.device, sigmas=sigmas, mu=mu)
            state.timesteps = self.scheduler.timesteps
            state.chunk_num_steps = len(state.timesteps)
        else:
            self._prepare_stage2_chunk(state)

        state.step_in_chunk = 0
        state.step_index = 0
        self._num_timesteps = state.chunk_num_steps

    def _prepare_stage2_chunk(self, state: DiffusionRequestState) -> None:
        extra = state.extra
        batch_size, num_channel, num_frames_lat, height, width = state.latents.shape
        latents_flat = state.latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames_lat, num_channel, height, width
        )
        for _ in range(extra["pyramid_num_stages"] - 1):
            height //= 2
            width //= 2
            latents_flat = F.interpolate(latents_flat, size=(height, width), mode="bilinear") * 2
        state.latents = latents_flat.reshape(batch_size, num_frames_lat, num_channel, height, width).permute(
            0, 2, 1, 3, 4
        )
        extra["stage2_height"] = height
        extra["stage2_width"] = width
        extra["stage2_start_point_list"] = [state.latents] if self.is_distilled else None
        extra["stage_index"] = 0
        extra["stage_step_index"] = 0
        amplify_first_chunk = extra["is_amplify_first_chunk"] and extra["is_first_chunk"]
        state.chunk_num_steps = sum(
            self._stage2_effective_num_steps(int(num_steps), amplify_first_chunk)
            for num_steps in extra["pyramid_num_inference_steps_list"]
        )
        self._set_stage2_timesteps(state)

    def _stage2_effective_num_steps(self, num_steps: int, is_amplify_first_chunk: bool) -> int:
        if self.is_distilled and is_amplify_first_chunk:
            return num_steps * 2
        return num_steps

    def _set_stage2_timesteps(self, state: DiffusionRequestState) -> None:
        extra = state.extra
        patch_size = self.transformer.config.patch_size
        image_seq_len = (state.latents.shape[-1] * state.latents.shape[-2] * state.latents.shape[-3]) // (
            patch_size[0] * patch_size[1] * patch_size[2]
        )
        mu = calculate_shift(image_seq_len)
        stage_index = extra["stage_index"]
        self.scheduler.set_timesteps(
            extra["pyramid_num_inference_steps_list"][stage_index],
            stage_index=stage_index,
            device=self.device,
            mu=mu,
            is_amplify_first_chunk=extra["is_amplify_first_chunk"] and extra["is_first_chunk"],
        )
        state.timesteps = self.scheduler.timesteps
        state.step_index = extra["stage_step_index"]

    def denoise_step(
        self,
        input_batch: InputBatch,
        states: Sequence[DiffusionRequestState],
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del kwargs
        if len(states) != 1:
            raise ValueError("Helios step execution supports a single request, not a batched request.")
        state = states[0]
        if state.extra["is_enable_stage2"]:
            return self._denoise_stage2_step(state)
        return self._denoise_stage1_step(state, input_batch.latents, input_batch.timesteps)

    def _denoise_stage1_step(
        self,
        state: DiffusionRequestState,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        extra = state.extra
        batch_size = latents.shape[0]
        t = timesteps
        self._current_timestep = t
        timestep = t.expand(batch_size)
        transformer_kwargs = {
            "hidden_states": latents.to(extra["dtype"]),
            "timestep": timestep,
            "indices_hidden_states": extra["indices_hidden_states"],
            "indices_latents_history_short": extra["indices_latents_history_short"],
            "indices_latents_history_mid": extra["indices_latents_history_mid"],
            "indices_latents_history_long": extra["indices_latents_history_long"],
            "latents_history_short": extra["latents_history_short"].to(extra["dtype"]),
            "latents_history_mid": extra["latents_history_mid"].to(extra["dtype"]),
            "latents_history_long": extra["latents_history_long"].to(extra["dtype"]),
            "attention_kwargs": extra["attention_kwargs"],
            "return_dict": False,
        }
        if extra["use_cfg_zero_star"] and state.do_true_cfg:
            noise_pred = self.transformer(
                encoder_hidden_states=state.prompt_embeds,
                **transformer_kwargs,
            )[0]
            noise_uncond = self.transformer(
                encoder_hidden_states=state.negative_prompt_embeds,
                **transformer_kwargs,
            )[0]
            positive_flat = noise_pred.view(batch_size, -1)
            negative_flat = noise_uncond.view(batch_size, -1)
            alpha_cfg = optimized_scale(positive_flat, negative_flat)
            alpha_cfg = alpha_cfg.view(batch_size, *([1] * (len(noise_pred.shape) - 1))).to(noise_pred.dtype)
            if (state.step_in_chunk <= extra["zero_steps"]) and extra["use_zero_init"]:
                return noise_pred * 0.0
            return noise_uncond * alpha_cfg + extra["guidance_scale"] * (noise_pred - noise_uncond * alpha_cfg)

        positive_kwargs = {
            "encoder_hidden_states": state.prompt_embeds,
            **transformer_kwargs,
        }
        negative_kwargs = (
            {
                "encoder_hidden_states": state.negative_prompt_embeds,
                **transformer_kwargs,
            }
            if state.do_true_cfg
            else None
        )
        return self.predict_noise_maybe_with_cfg(
            do_true_cfg=state.do_true_cfg,
            true_cfg_scale=extra["guidance_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )

    def _denoise_stage2_step(self, state: DiffusionRequestState) -> torch.Tensor:
        extra = state.extra
        latents = state.latents
        assert latents is not None
        batch_size = latents.shape[0]
        t = state.current_timestep
        assert t is not None
        self._current_timestep = t
        timestep = t.expand(batch_size).to(torch.int64)
        transformer_kwargs = {
            "hidden_states": latents.to(extra["dtype"]),
            "timestep": timestep,
            "indices_hidden_states": extra["indices_hidden_states"],
            "indices_latents_history_short": extra["indices_latents_history_short"],
            "indices_latents_history_mid": extra["indices_latents_history_mid"],
            "indices_latents_history_long": extra["indices_latents_history_long"],
            "latents_history_short": extra["latents_history_short"].to(extra["dtype"]),
            "latents_history_mid": extra["latents_history_mid"].to(extra["dtype"]),
            "latents_history_long": extra["latents_history_long"].to(extra["dtype"]),
            "attention_kwargs": extra["attention_kwargs"],
            "return_dict": False,
        }
        noise_pred = self.transformer(encoder_hidden_states=state.prompt_embeds, **transformer_kwargs)[0]
        if state.do_true_cfg:
            noise_uncond = self.transformer(encoder_hidden_states=state.negative_prompt_embeds, **transformer_kwargs)[0]
            if extra["use_cfg_zero_star"]:
                positive_flat = noise_pred.view(batch_size, -1)
                negative_flat = noise_uncond.view(batch_size, -1)
                alpha_cfg = optimized_scale(positive_flat, negative_flat)
                alpha_cfg = alpha_cfg.view(batch_size, *([1] * (len(noise_pred.shape) - 1))).to(noise_pred.dtype)
                if (
                    extra["stage_index"] == 0
                    and extra["stage_step_index"] <= extra["zero_steps"]
                    and extra["use_zero_init"]
                ):
                    noise_pred = noise_pred * 0.0
                else:
                    noise_pred = noise_uncond * alpha_cfg + extra["guidance_scale"] * (
                        noise_pred - noise_uncond * alpha_cfg
                    )
            else:
                noise_pred = noise_uncond + extra["guidance_scale"] * (noise_pred - noise_uncond)
        return noise_pred

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if state.extra["is_enable_stage2"]:
            self._step_scheduler_stage2(state, noise_pred)
        else:
            t = state.current_timestep
            assert t is not None
            if self.is_distilled:
                state.latents = self.scheduler.step(
                    noise_pred,
                    t,
                    state.latents,
                    return_dict=False,
                    cur_sampling_step=state.step_in_chunk,
                    dmd_noisy_tensor=state.extra["stage1_start_latents"],
                    dmd_sigmas=self.scheduler.sigmas,
                    dmd_timesteps=self.scheduler.timesteps,
                    all_timesteps=state.timesteps,
                )[0]
            else:
                state.latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, state.latents, state.do_true_cfg)
            state.step_in_chunk += 1
            state.step_index = state.step_in_chunk

    def _step_scheduler_stage2(self, state: DiffusionRequestState, noise_pred: torch.Tensor) -> None:
        extra = state.extra
        t = state.current_timestep
        assert t is not None and state.latents is not None
        start_points = extra["stage2_start_point_list"]
        state.latents = self.scheduler.step(
            noise_pred,
            t,
            state.latents,
            return_dict=False,
            cur_sampling_step=extra["stage_step_index"],
            dmd_noisy_tensor=start_points[extra["stage_index"]] if start_points is not None else None,
            dmd_sigmas=self.scheduler.sigmas,
            dmd_timesteps=self.scheduler.timesteps,
            all_timesteps=state.timesteps,
        )[0]
        extra["stage_step_index"] += 1
        state.step_in_chunk += 1
        state.step_index = extra["stage_step_index"]

        if extra["stage_step_index"] < len(state.timesteps):
            return
        if extra["stage_index"] >= extra["pyramid_num_stages"] - 1:
            return

        extra["stage_index"] += 1
        extra["stage_step_index"] = 0
        extra["stage2_height"] *= 2
        extra["stage2_width"] *= 2
        batch_size, num_channel, num_frames_cur = state.latents.shape[:3]
        latents_flat = state.latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames_cur,
            num_channel,
            extra["stage2_height"] // 2,
            extra["stage2_width"] // 2,
        )
        latents_flat = F.interpolate(latents_flat, size=(extra["stage2_height"], extra["stage2_width"]), mode="nearest")
        state.latents = latents_flat.reshape(
            batch_size,
            num_frames_cur,
            num_channel,
            extra["stage2_height"],
            extra["stage2_width"],
        ).permute(0, 2, 1, 3, 4)

        ori_sigma = 1 - self.scheduler.ori_start_sigmas[extra["stage_index"]]
        gamma = self.scheduler.config.gamma
        alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
        beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
        patch_size = self.transformer.config.patch_size
        noise = self.sample_block_noise(
            batch_size,
            num_channel,
            state.latents.shape[2],
            extra["stage2_height"],
            extra["stage2_width"],
            patch_size,
            generator=extra["generator"],
        ).to(device=self.device, dtype=extra["dtype"])
        state.latents = alpha * state.latents + beta * noise
        if self.is_distilled and extra["stage2_start_point_list"] is not None:
            extra["stage2_start_point_list"].append(state.latents)
        self._set_stage2_timesteps(state)

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        del kwargs
        extra = state.extra
        is_first_chunk = extra["is_first_chunk"]
        is_second_chunk = extra["is_second_chunk"]
        if extra["keep_first_frame"] and (
            (is_first_chunk and extra["image_latents"] is None) or (extra["is_skip_first_chunk"] and is_second_chunk)
        ):
            extra["image_latents"] = state.latents[:, :, 0:1, :, :]

        extra["total_generated_latent_frames"] += state.latents.shape[2]
        extra["history_latents"] = torch.cat([extra["history_latents"], state.latents], dim=2)
        real_history_latents = extra["history_latents"][:, :, -extra["total_generated_latent_frames"] :]
        current_latents = (
            real_history_latents[:, :, -extra["num_latent_frames_per_chunk"] :].to(extra["vae_dtype"])
            / extra["latents_std"]
            + extra["latents_mean"]
        )
        current_video = self.vae.decode(current_latents, return_dict=False)[0]
        if extra["history_video"] is None:
            extra["history_video"] = current_video
        else:
            extra["history_video"] = torch.cat([extra["history_video"], current_video], dim=2)

        output = current_latents if extra["output_type"] == "latent" else current_video
        completed_chunk_index = state.chunk_index
        state.chunk_index += 1
        finished = state.request_denoise_completed
        if not finished:
            self._prepare_next_chunk(state)
        else:
            self._current_timestep = None
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else {},
            chunk_index=completed_chunk_index,
            total_chunks=state.total_chunks,
            finished=finished,
        )

    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 384,
        width: int = 640,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        frame_num: int = 132,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        # Helios-specific
        history_sizes: list | None = None,
        num_latent_frames_per_chunk: int = 9,
        keep_first_frame: bool = True,
        # I2V
        image: torch.Tensor | None = None,
        add_noise_to_image_latents: bool = True,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        # V2V
        video: torch.Tensor | None = None,
        add_noise_to_video_latents: bool = True,
        video_noise_sigma_min: float = 0.111,
        video_noise_sigma_max: float = 0.135,
        # Stage 2 (pyramid multi-stage denoising)
        is_enable_stage2: bool = False,
        pyramid_num_stages: int = 3,
        pyramid_num_inference_steps_list: list | None = None,
        is_skip_first_chunk: bool = False,
        # DMD
        is_amplify_first_chunk: bool = False,
        # CFG Zero Star
        use_cfg_zero_star: bool = False,
        use_zero_init: bool = True,
        zero_steps: int = 1,
        **kwargs,
    ) -> DiffusionOutput:
        if pyramid_num_inference_steps_list is None:
            pyramid_num_inference_steps_list = [10, 10, 10]
        if history_sizes is None:
            history_sizes = [16, 2, 1]

        # Read Helios-specific params from extra_args
        extra = getattr(req.sampling_params, "extra_args", {}) or {}
        is_enable_stage2 = extra.get("is_enable_stage2", is_enable_stage2)
        pyramid_num_stages = extra.get("pyramid_num_stages", pyramid_num_stages)
        pyramid_num_inference_steps_list = extra.get(
            "pyramid_num_inference_steps_list", pyramid_num_inference_steps_list
        )
        is_amplify_first_chunk = extra.get("is_amplify_first_chunk", is_amplify_first_chunk)
        use_cfg_zero_star = extra.get("use_cfg_zero_star", use_cfg_zero_star)
        use_zero_init = extra.get("use_zero_init", use_zero_init)
        zero_steps = extra.get("zero_steps", zero_steps)
        is_skip_first_chunk = extra.get("is_skip_first_chunk", is_skip_first_chunk)

        image = extra.get("image", image)
        video = extra.get("video", video)
        add_noise_to_image_latents = extra.get("add_noise_to_image_latents", add_noise_to_image_latents)
        image_noise_sigma_min = extra.get("image_noise_sigma_min", image_noise_sigma_min)
        image_noise_sigma_max = extra.get("image_noise_sigma_max", image_noise_sigma_max)
        add_noise_to_video_latents = extra.get("add_noise_to_video_latents", add_noise_to_video_latents)
        video_noise_sigma_min = extra.get("video_noise_sigma_min", video_noise_sigma_min)
        video_noise_sigma_max = extra.get("video_noise_sigma_max", video_noise_sigma_max)

        if image is not None and video is not None:
            raise ValueError("image and video cannot be provided simultaneously")

        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt, not a batched request.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")

        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Helios generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        height = (height // 16) * 16
        width = (width // 16) * 16
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Encode prompts
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 226,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        batch_size = prompt_embeds.shape[0]

        history_sizes = sorted(history_sizes, reverse=True)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )

        # Prepare I2V image latents
        fake_image_latents = None
        image_latents = None
        if image is not None:
            image_latents, fake_image_latents = self.prepare_image_latents(
                image,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

        if image_latents is not None and add_noise_to_image_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            fake_image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min)
                + video_noise_sigma_min
            )
            fake_image_latents = (
                fake_image_noise_sigma * randn_tensor(fake_image_latents.shape, generator=generator, device=device)
                + (1 - fake_image_noise_sigma) * fake_image_latents
            )

        # Prepare V2V video latents
        video_latents = None
        if video is not None:
            image_latents, video_latents = self.prepare_video_latents(
                video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

        if video_latents is not None and add_noise_to_video_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )

            noisy_latents_chunks = []
            num_latent_chunks = video_latents.shape[2] // num_latent_frames_per_chunk
            for i in range(num_latent_chunks):
                chunk_start = i * num_latent_frames_per_chunk
                chunk_end = chunk_start + num_latent_frames_per_chunk
                latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]
                chunk_frames = latent_chunk.shape[2]
                frame_sigmas = (
                    torch.rand(chunk_frames, device=device, generator=generator)
                    * (video_noise_sigma_max - video_noise_sigma_min)
                    + video_noise_sigma_min
                )
                frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
                noisy_chunk = (
                    frame_sigmas * randn_tensor(latent_chunk.shape, generator=generator, device=device)
                    + (1 - frame_sigmas) * latent_chunk
                )
                noisy_latents_chunks.append(noisy_chunk)
            video_latents = torch.cat(noisy_latents_chunks, dim=2)

        # Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        num_latent_chunk = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        num_history_latent_frames = sum(history_sizes)
        history_video = None
        total_generated_latent_frames = 0

        if not keep_first_frame:
            history_sizes[-1] = history_sizes[-1] + 1

        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )

        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents[:, :, :-1, :, :], fake_image_latents], dim=2)
            total_generated_latent_frames += 1

        if video_latents is not None:
            history_frames = history_latents.shape[2]
            video_frames = video_latents.shape[2]
            if video_frames < history_frames:
                keep_frames = history_frames - video_frames
                history_latents = torch.cat([history_latents[:, :, :keep_frames, :, :], video_latents], dim=2)
            else:
                history_latents = video_latents
            total_generated_latent_frames += video_latents.shape[2]

        # Prepare frame indices
        if keep_first_frame:
            indices = torch.arange(0, sum([1, *history_sizes, num_latent_frames_per_chunk]))
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        else:
            indices = torch.arange(0, sum([*history_sizes, num_latent_frames_per_chunk]))
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)

        indices_hidden_states = indices_hidden_states.unsqueeze(0)
        indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        if attention_kwargs is None:
            attention_kwargs = {}

        vae_dtype = self.vae.dtype

        # Chunked denoising loop
        for k in range(num_latent_chunk):
            is_first_chunk = k == 0
            is_second_chunk = k == 1

            if keep_first_frame:
                latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                    :, :, -num_history_latent_frames:
                ].split(history_sizes, dim=2)
                if image_latents is None and is_first_chunk:
                    latents_prefix = torch.zeros(
                        (
                            batch_size,
                            num_channels_latents,
                            1,
                            latents_history_1x.shape[-2],
                            latents_history_1x.shape[-1],
                        ),
                        device=latents_history_1x.device,
                        dtype=latents_history_1x.dtype,
                    )
                else:
                    latents_prefix = image_latents
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                latents_history_long, latents_history_mid, latents_history_short = history_latents[
                    :, :, -num_history_latent_frames:
                ].split(history_sizes, dim=2)

            # Prepare noise for this chunk
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            if not is_enable_stage2:
                # Stage 1 only: single-stage denoising
                patch_size = self.transformer.config.patch_size
                image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                    patch_size[0] * patch_size[1] * patch_size[2]
                )
                sigmas = np.linspace(0.999, 0.0, num_steps + 1)[:-1]
                mu = calculate_shift(image_seq_len)
                self.scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas, mu=mu)
                timesteps = self.scheduler.timesteps
                self._num_timesteps = len(timesteps)

                latents = self._stage1_sample(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    attention_kwargs=attention_kwargs,
                    transformer_dtype=dtype,
                    use_cfg_zero_star=use_cfg_zero_star,
                    use_zero_init=use_zero_init,
                    zero_steps=zero_steps,
                )
            else:
                # Stage 2: pyramid multi-stage denoising
                latents = self._stage2_sample(
                    latents=latents,
                    pyramid_num_stages=pyramid_num_stages,
                    pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    attention_kwargs=attention_kwargs,
                    transformer_dtype=dtype,
                    is_amplify_first_chunk=is_amplify_first_chunk and is_first_chunk,
                    use_cfg_zero_star=use_cfg_zero_star,
                    use_zero_init=use_zero_init,
                    zero_steps=zero_steps,
                    device=device,
                    generator=generator,
                )

            if keep_first_frame and (
                (is_first_chunk and image_latents is None) or (is_skip_first_chunk and is_second_chunk)
            ):
                image_latents = latents[:, :, 0:1, :, :]

            total_generated_latent_frames += latents.shape[2]
            history_latents = torch.cat([history_latents, latents], dim=2)
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
            index_slice = (
                slice(None),
                slice(None),
                slice(-num_latent_frames_per_chunk, None),
            )

            current_latents = real_history_latents[index_slice].to(vae_dtype) / latents_std + latents_mean
            current_video = self.vae.decode(current_latents, return_dict=False)[0]

            if history_video is None:
                history_video = current_video
            else:
                history_video = torch.cat([history_video, current_video], dim=2)

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        if output_type == "latent":
            output = real_history_latents
        else:
            output = history_video

        return DiffusionOutput(
            output=output, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def _stage1_sample(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        timesteps: torch.Tensor,
        guidance_scale: float,
        indices_hidden_states: torch.Tensor,
        indices_latents_history_short: torch.Tensor,
        indices_latents_history_mid: torch.Tensor,
        indices_latents_history_long: torch.Tensor,
        latents_history_short: torch.Tensor,
        latents_history_mid: torch.Tensor,
        latents_history_long: torch.Tensor,
        attention_kwargs: dict,
        transformer_dtype: torch.dtype,
        use_cfg_zero_star: bool = False,
        use_zero_init: bool = True,
        zero_steps: int = 1,
    ) -> torch.Tensor:
        """Single-stage denoising loop for one chunk."""
        batch_size = latents.shape[0]
        do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                timestep = t.expand(batch_size)

                transformer_kwargs = {
                    "hidden_states": latents.to(transformer_dtype),
                    "timestep": timestep,
                    "indices_hidden_states": indices_hidden_states,
                    "indices_latents_history_short": indices_latents_history_short,
                    "indices_latents_history_mid": indices_latents_history_mid,
                    "indices_latents_history_long": indices_latents_history_long,
                    "latents_history_short": latents_history_short.to(transformer_dtype),
                    "latents_history_mid": latents_history_mid.to(transformer_dtype),
                    "latents_history_long": latents_history_long.to(transformer_dtype),
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }

                if use_cfg_zero_star and do_true_cfg:
                    noise_pred = self.transformer(
                        encoder_hidden_states=prompt_embeds,
                        **transformer_kwargs,
                    )[0]

                    noise_uncond = self.transformer(
                        encoder_hidden_states=negative_prompt_embeds,
                        **transformer_kwargs,
                    )[0]

                    positive_flat = noise_pred.view(batch_size, -1)
                    negative_flat = noise_uncond.view(batch_size, -1)
                    alpha_cfg = optimized_scale(positive_flat, negative_flat)
                    alpha_cfg = alpha_cfg.view(batch_size, *([1] * (len(noise_pred.shape) - 1)))
                    alpha_cfg = alpha_cfg.to(noise_pred.dtype)

                    if (i <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred * 0.0
                    else:
                        noise_pred = noise_uncond * alpha_cfg + guidance_scale * (noise_pred - noise_uncond * alpha_cfg)
                else:
                    positive_kwargs = {
                        "encoder_hidden_states": prompt_embeds,
                        **transformer_kwargs,
                    }
                    if do_true_cfg:
                        negative_kwargs = {
                            "encoder_hidden_states": negative_prompt_embeds,
                            **transformer_kwargs,
                        }
                    else:
                        negative_kwargs = None

                    noise_pred = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_true_cfg,
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                    )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

                pbar.update()

        return latents

    def _stage2_sample(
        self,
        latents: torch.Tensor,
        pyramid_num_stages: int,
        pyramid_num_inference_steps_list: list[int],
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        indices_hidden_states: torch.Tensor,
        indices_latents_history_short: torch.Tensor,
        indices_latents_history_mid: torch.Tensor,
        indices_latents_history_long: torch.Tensor,
        latents_history_short: torch.Tensor,
        latents_history_mid: torch.Tensor,
        latents_history_long: torch.Tensor,
        attention_kwargs: dict,
        transformer_dtype: torch.dtype,
        is_amplify_first_chunk: bool = False,
        use_cfg_zero_star: bool = False,
        use_zero_init: bool = True,
        zero_steps: int = 1,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        """Pyramid multi-stage denoising for one chunk."""
        batch_size, num_channel, num_frames_lat, height, width = latents.shape

        # Downsample latents to the smallest pyramid level
        latents_flat = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames_lat, num_channel, height, width)
        for _ in range(pyramid_num_stages - 1):
            height //= 2
            width //= 2
            latents_flat = F.interpolate(latents_flat, size=(height, width), mode="bilinear") * 2
        latents = latents_flat.reshape(batch_size, num_frames_lat, num_channel, height, width).permute(0, 2, 1, 3, 4)

        start_point_list = None
        if self.is_distilled:
            start_point_list = [latents]

        do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        for i_s in range(pyramid_num_stages):
            patch_size = self.transformer.config.patch_size
            image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            mu = calculate_shift(image_seq_len)
            self.scheduler.set_timesteps(
                pyramid_num_inference_steps_list[i_s],
                stage_index=i_s,
                device=device,
                mu=mu,
                is_amplify_first_chunk=is_amplify_first_chunk,
            )
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                # Upsample latents to next pyramid level
                height *= 2
                width *= 2
                num_frames_cur = latents.shape[2]
                latents_flat = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames_cur, num_channel, height // 2, width // 2
                )
                latents_flat = F.interpolate(latents_flat, size=(height, width), mode="nearest")
                latents = latents_flat.reshape(batch_size, num_frames_cur, num_channel, height, width).permute(
                    0, 2, 1, 3, 4
                )

                # Add block noise to fix artifacts between stages
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                noise = self.sample_block_noise(
                    batch_size, num_channel, latents.shape[2], height, width, patch_size, generator=generator
                )
                noise = noise.to(device=device, dtype=transformer_dtype)
                latents = alpha * latents + beta * noise

                if self.is_distilled and start_point_list is not None:
                    start_point_list.append(latents)

            with self.progress_bar(total=len(timesteps)) as pbar:
                for idx, t in enumerate(timesteps):
                    self._current_timestep = t
                    timestep = t.expand(latents.shape[0]).to(torch.int64)

                    transformer_kwargs = {
                        "hidden_states": latents.to(transformer_dtype),
                        "timestep": timestep,
                        "indices_hidden_states": indices_hidden_states,
                        "indices_latents_history_short": indices_latents_history_short,
                        "indices_latents_history_mid": indices_latents_history_mid,
                        "indices_latents_history_long": indices_latents_history_long,
                        "latents_history_short": latents_history_short.to(transformer_dtype),
                        "latents_history_mid": latents_history_mid.to(transformer_dtype),
                        "latents_history_long": latents_history_long.to(transformer_dtype),
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                    }

                    noise_pred = self.transformer(
                        encoder_hidden_states=prompt_embeds,
                        **transformer_kwargs,
                    )[0]

                    if do_true_cfg:
                        noise_uncond = self.transformer(
                            encoder_hidden_states=negative_prompt_embeds,
                            **transformer_kwargs,
                        )[0]

                        if use_cfg_zero_star:
                            positive_flat = noise_pred.view(batch_size, -1)
                            negative_flat = noise_uncond.view(batch_size, -1)
                            alpha_cfg = optimized_scale(positive_flat, negative_flat)
                            alpha_cfg = alpha_cfg.view(batch_size, *([1] * (len(noise_pred.shape) - 1)))
                            alpha_cfg = alpha_cfg.to(noise_pred.dtype)

                            if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                                noise_pred = noise_pred * 0.0
                            else:
                                noise_pred = noise_uncond * alpha_cfg + guidance_scale * (
                                    noise_pred - noise_uncond * alpha_cfg
                                )
                        else:
                            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        return_dict=False,
                        cur_sampling_step=idx,
                        dmd_noisy_tensor=start_point_list[i_s] if start_point_list is not None else None,
                        dmd_sigmas=self.scheduler.sigmas,
                        dmd_timesteps=self.scheduler.timesteps,
                        all_timesteps=timesteps,
                    )[0]

                    pbar.update()

        return latents

    def sample_block_noise(self, batch_size, channel, num_frames, height, width, patch_size=(1, 2, 2), generator=None):
        gamma = self.scheduler.config.gamma
        _, ph, pw = patch_size
        block_size = ph * pw

        device = generator.device if generator is not None else self.device

        cov = torch.eye(block_size) * (1 + gamma) - torch.ones(block_size, block_size) * gamma
        cov += torch.eye(block_size) * 1e-8
        cov = cov.float()  # Upcast to fp32 for numerical stability — cholesky is unreliable in fp16/bf16.

        L = torch.linalg.cholesky(cov).to(device)
        block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)
        z = torch.randn(block_number, block_size, generator=generator, device=device)
        noise = z @ L.T

        noise = noise.view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
        noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)

        return noise

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_image_latents(
        self,
        image: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        num_latent_frames_per_chunk: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single image into VAE latent space for I2V generation.

        Returns (image_latents, fake_image_latents) where fake_image_latents
        is the last-frame latent of a repeated-frame video, used to seed the
        history buffer for the first denoising chunk.
        """
        device = device or self.device
        image = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
        latents = self.vae.encode(image).latent_dist.sample(generator=generator)
        latents = (latents - latents_mean) * latents_std

        min_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        fake_video = image.repeat(1, 1, min_frames, 1, 1)
        fake_latents_full = self.vae.encode(fake_video).latent_dist.sample(generator=generator)
        fake_latents_full = (fake_latents_full - latents_mean) * latents_std
        fake_latents = fake_latents_full[:, :, -1:, :, :]

        return latents.to(device=device, dtype=dtype), fake_latents.to(device=device, dtype=dtype)

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        num_latent_frames_per_chunk: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a video into VAE latent space for V2V generation.

        Returns (first_frame_latent, video_latents) where first_frame_latent
        is used as the image prefix, and video_latents fills the history buffer.
        """
        device = device or self.device
        video = video.to(device=device, dtype=self.vae.dtype)

        num_frames = video.shape[2]
        min_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        num_chunks = num_frames // min_frames
        if num_chunks == 0:
            raise ValueError(
                f"Video must have at least {min_frames} frames (got {num_frames}). "
                f"Required: (num_latent_frames_per_chunk - 1) * {self.vae_scale_factor_temporal} + 1"
            )
        total_valid_frames = num_chunks * min_frames
        start_frame = num_frames - total_valid_frames

        first_frame = video[:, :, 0:1, :, :]
        first_frame_latent = self.vae.encode(first_frame).latent_dist.sample(generator=generator)
        first_frame_latent = (first_frame_latent - latents_mean) * latents_std

        latents_chunks = []
        for i in range(num_chunks):
            chunk_start = start_frame + i * min_frames
            chunk_end = chunk_start + min_frames
            video_chunk = video[:, :, chunk_start:chunk_end, :, :]
            chunk_latents = self.vae.encode(video_chunk).latent_dist.sample(generator=generator)
            chunk_latents = (chunk_latents - latents_mean) * latents_std
            latents_chunks.append(chunk_latents)
        video_latents = torch.cat(latents_chunks, dim=2)

        return first_frame_latent.to(device=device, dtype=dtype), video_latents.to(device=device, dtype=dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
