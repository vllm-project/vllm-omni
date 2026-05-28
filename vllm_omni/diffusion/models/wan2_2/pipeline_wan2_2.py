# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

logger = logging.getLogger(__name__)
DEBUG_PERF = False
WAN_SAMPLE_SOLVER_CHOICES = {"unipc", "euler"}


def build_wan_scheduler(sample_solver: str, flow_shift: float) -> Any:
    if sample_solver == "unipc":
        return FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )
    if sample_solver == "euler":
        return WanEulerScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )

    raise ValueError(
        f"Unsupported Wan sample_solver: {sample_solver}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}"
    )


def resolve_wan_sample_solver(req: OmniDiffusionRequest, default: str = "unipc") -> str:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw = extra_args.get("sample_solver", default)
    sample_solver = str(raw).strip().lower()
    if sample_solver not in WAN_SAMPLE_SOLVER_CHOICES:
        raise ValueError(f"Invalid sample_solver={raw!r}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}")
    return sample_solver


def resolve_wan_flow_shift(req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> float:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw_flow_shift = extra_args.get("flow_shift")
    if raw_flow_shift is None:
        raw_flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0

    try:
        return float(raw_flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flow_shift={raw_flow_shift!r}. flow_shift must be a float.") from exc


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict, quant_config: QuantizationConfig | None = None, prefix: str = ""
) -> WanTransformer3DModel:
    """Create WanTransformer3DModel from config dict."""
    kwargs: dict = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
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
    if "pos_embed_seq_len" in config:
        kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])

    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    if prefix:
        kwargs["prefix"] = prefix

    return WanTransformer3DModel(**kwargs)


def get_wan22_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return video
        custom_output = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            custom_output["video_fps_multiplier"] = multiplier
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "custom_output": custom_output,
        }

    return post_process_func


def get_wan22_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for Wan2.2: optionally load and resize input image for I2V mode."""
    import numpy as np

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                continue

            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(
                    f"""Unsupported image format {raw_image.__class__}.""",
                    """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
                )
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

            # Calculate dimensions based on aspect ratio if not provided
            if request.sampling_params.height is None or request.sampling_params.width is None:
                # Default max area for 720P
                max_area = 720 * 1280
                aspect_ratio = image.height / image.width

                # Calculate dimensions maintaining aspect ratio
                mod_value = 16  # Must be divisible by 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width

            # Resize image to target dimensions
            image = image.resize(
                (request.sampling_params.width, request.sampling_params.height),  # type: ignore # Above has ensured that width & height are not None
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence already checked above

            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan22Pipeline(
    nn.Module, PipelineParallelMixin, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    supports_step_execution: ClassVar[bool] = True
    supports_micro_step_execution: ClassVar[bool] = True

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

        # Read model_index.json to detect expand_timesteps mode (for TI2V-5B)
        self.expand_timesteps = False
        self.has_transformer_2 = False
        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
            # Check if this is a two-stage model (MoE with transformer_2)
            transformer_2_path = os.path.join(model, "transformer_2")
            self.has_transformer_2 = os.path.exists(transformer_2_path)
        else:
            # For remote models, download and read model_index.json
            try:
                from huggingface_hub import hf_hub_download

                model_index_path = hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    # Check transformer_2 from model_index
                    transformer_2_info = model_index.get("transformer_2", [None, None])
                    self.has_transformer_2 = transformer_2_info[0] is not None
            except Exception:
                pass

        self.boundary_ratio = od_config.boundary_ratio

        # Determine which transformers to load based on boundary_ratio
        # boundary_ratio=1.0: only load transformer_2 (low-noise stage only)
        # boundary_ratio=0.0: only load transformer (high-noise stage only)
        # otherwise: load both transformers
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = self.has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        # Set up weights sources for transformer(s)
        self.weights_sources = []
        if load_transformer:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            )
        if load_transformer_2:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        prefetch_subfolders(
            model,
            ["tokenizer", "text_encoder", "vae"],
            local_files_only=local_files_only,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        z_dim = self.vae.config.z_dim
        self._latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        self._latents_inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1)

        # Initialize transformers with correct config (weights loaded via load_weights)
        if load_transformer:
            transformer_config = load_transformer_config(model, "transformer", local_files_only)
            self.transformer = self._create_transformer(transformer_config)
        else:
            self.transformer = None

        if load_transformer_2:
            transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
            self.transformer_2 = self._create_transformer(transformer_2_config)
        else:
            self.transformer_2 = None

        # Store the active transformer config
        if load_transformer:
            self.transformer_config = self.transformer.config
        elif load_transformer_2:
            self.transformer_config = self.transformer_2.config
        else:
            raise RuntimeError("No transformer loaded")

        self._sample_solver = "unipc"
        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        """Create a transformer from a config dict. Respects od_config.quantization_config."""
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(config, quant_config=quant_config)

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

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any],
        latent_condition: torch.Tensor | None = None,
        first_frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                set_forward_context_denoise_step_idx(step_idx)

                # Select model based on timestep and boundary_ratio
                # High noise stage (t >= boundary_timestep): use transformer
                # Low noise stage (t < boundary_timestep): use transformer_2
                if boundary_timestep is not None and t < boundary_timestep:
                    # Low noise stage - always use guidance_high for this stage
                    current_guidance_scale = guidance_high
                    if self.transformer_2 is not None:
                        current_model = self.transformer_2
                    elif self.transformer is not None:
                        # Fallback to transformer if transformer_2 not loaded
                        current_model = self.transformer
                    else:
                        raise RuntimeError("No transformer available for low-noise stage")
                else:
                    # High noise stage - always use guidance_low for this stage
                    current_guidance_scale = guidance_low
                    if self.transformer is not None:
                        current_model = self.transformer
                    elif self.transformer_2 is not None:
                        # Fallback to transformer_2 if transformer not loaded
                        current_model = self.transformer_2
                    else:
                        raise RuntimeError("No transformer available for high-noise stage")

                if self.expand_timesteps and latent_condition is not None:
                    # I2V mode: blend condition with latents using mask
                    latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(dtype)

                    # Expand timesteps per patch - use floor division to match patch embedding
                    patch_size = self.transformer_config.patch_size
                    patch_height = latents.shape[3] // patch_size[1]
                    patch_width = latents.shape[4] // patch_size[2]

                    # Create mask at patch resolution (same as hidden states sequence length)
                    patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                    patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]  # Ensure correct dimensions
                    temp_ts = (patch_mask[0][0] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # T2V mode: standard forward
                    latent_model_input = latents.to(dtype)
                    timestep = t.expand(latents.shape[0])

                do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": current_model,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float | tuple[float, float] = 4.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Get parameters from request or arguments
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )
        if len(req.prompts) == 1:  # If req.prompt is empty, default to prompt & neg_prompt in param list
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Wan2.2 generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num

        # Ensure dimensions are compatible with VAE and patch size
        # For expand_timesteps mode, we need latent dims to be even (divisible by patch_size)
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        # Respect per-request guidance_scale when explicitly provided.
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        guidance_high = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1
                else guidance_low
            )
        )

        # record guidance for properties
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Prefer engine-configured boundary_ratio, but allow per-request fallback.
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else req.sampling_params.boundary_ratio

        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for T2V generation. using default value 0.875")

        # validate shapes
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high if boundary_ratio is not None else None,
            boundary_ratio=boundary_ratio,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        # Get dtype from whichever transformer is loaded
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            # Fallback to text_encoder dtype if no transformer loaded
            dtype = self.text_encoder.dtype

        # Seed / generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        if DEBUG_PERF:
            # Sync GPU before timing to ensure accurate measurements
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_low > 1.0 or guidance_high > 1.0,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            elif guidance_low > 1.0 or guidance_high > 1.0:
                raise ValueError(
                    "negative_prompt_embeds must be provided when prompt_embeds are given and guidance > 1."
                )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        sample_solver = resolve_wan_sample_solver(req, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(req, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        if DEBUG_PERF:
            _t_latent_prep_start = time.perf_counter()
        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                logger.warning(
                    """Received a list of image. Only a single image is supported by this model."""
                    """Taking only the first image for now."""
                )
            raw_image = raw_image[0]
        if raw_image is None:
            image = None
        elif isinstance(raw_image, str):
            image = PIL.Image.open(raw_image)
        else:
            image = cast(PIL.Image.Image | torch.Tensor, raw_image)

        latent_condition = None
        first_frame_mask = None

        if self.expand_timesteps and image is not None:
            # I2V mode: encode image and prepare condition
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            # Preprocess image
            if isinstance(image, PIL.Image.Image):
                image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                image_tensor = video_processor.preprocess(image, height=height, width=width)
            else:
                image_tensor = image

            # Use out_channels for noise latents (not in_channels which includes condition)
            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]

            # Prepare noise latents
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
            )

            # Encode image condition
            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]

            image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            # Normalize condition latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std
            latent_condition = latent_condition.to(torch.float32)

            # Create mask: 0 for first frame (condition), 1 for rest (to denoise)
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode: standard latent preparation
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
            )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if attention_kwargs is None:
            attention_kwargs = {}

        if DEBUG_PERF:
            _t_denoise_start = time.perf_counter()
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=boundary_timestep,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            latent_condition=latent_condition,
            first_frame_mask=first_frame_mask,
        )

        # Wan2.2 is prone to out of memory errors when predicting large videos
        # so we empty the cache here to avoid OOM before vae decoding.
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        # For I2V mode: blend final latents with condition
        if self.expand_timesteps and latent_condition is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        if DEBUG_PERF:
            _t_decode_start = time.perf_counter()
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

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms

            if _is_rank_zero():
                logger.info(
                    "Pipeline stage timing summary: "
                    "TextEncoding=%.2f ms, LatentPreparation=%.2f ms, "
                    "Denoising=%.2f ms (%d steps), Decoding=%.2f ms, "
                    "StagesSum=%.2f ms, PipelineWall=%.2f ms, Unaccounted=%.2f ms",
                    _t_text_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                    _t_pipeline_wall_ms - _t_stages_sum,
                )

        return DiffusionOutput(
            output=output, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        if current_model is None:
            current_model = self.transformer
        result = current_model(**kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
        boundary_ratio=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")

    # ── Micro-step execution (SupportsMicroStepExecution) Wan2.1 model (single-stage) ──

    def _extract_prompts(
        self,
        state: DiffusionRequestState,
    ) -> tuple[str | None, str | None]:
        """Extract prompt and negative prompt from *state*."""
        prompt: str | None = None
        negative_prompt: str | None = None
        if state.prompts:
            p = state.prompts[0]
            if isinstance(p, str):
                prompt = p
            else:
                prompt = p.get("prompt")
                negative_prompt = p.get("negative_prompt")
        return prompt, negative_prompt

    def _resolve_generation_params(
        self,
        state: DiffusionRequestState,
    ) -> dict[str, Any]:
        """Extract and validate generation parameters from *state*.

        Returns a dict with resolved height, width, num_frames, guidance_low,
        guidance_high, boundary_ratio, boundary_timestep, device, dtype, and
        generator.
        """
        sampling = state.sampling

        height = sampling.height or 480
        width = sampling.width or 832
        num_frames = sampling.num_frames or 81
        chunk_frames = sampling.chunk_frames or 8
        num_steps = sampling.num_inference_steps or 40

        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        guidance_high = (
            sampling.guidance_scale_2
            if sampling.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1
                else guidance_low
            )
        )

        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else sampling.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = 0.875

        device = self.device
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            dtype = self.text_encoder.dtype

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sampling.seed)

        return {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "chunk_frames": chunk_frames,
            "num_steps": num_steps,
            "guidance_low": guidance_low,
            "guidance_high": guidance_high,
            "boundary_ratio": boundary_ratio,
            "boundary_timestep": boundary_ratio * 1000,  # num_train_timesteps
            "device": device,
            "dtype": dtype,
            "generator": generator,
        }

    def _prepare_latent_input(
        self,
        state: DiffusionRequestState,
        t: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare latent_model_input and timestep tensor for one denoise step.

        Handles both T2V (passthrough) and I2V (condition blending + per-patch
        timestep expansion) modes.

        Returns:
            (latent_model_input, timestep_tensor)
        """
        latent_condition = state.extra.get("latent_condition")
        first_frame_mask = state.extra.get("first_frame_mask")
        expand_timesteps = state.extra.get("expand_timesteps", False)

        if expand_timesteps and latent_condition is not None:
            # I2V mode: blend condition with latents, expand timesteps per patch
            latent_model_input = ((1 - first_frame_mask) * latent_condition + first_frame_mask * state.latents).to(
                dtype
            )
            patch_size = self.transformer_config.patch_size
            patch_height = state.latents.shape[3] // patch_size[1]
            patch_width = state.latents.shape[4] // patch_size[2]
            patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
            patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]
            temp_ts = (patch_mask[0][0] * t).flatten()
            timestep_tensor = temp_ts.unsqueeze(0).expand(state.latents.shape[0], -1)
        else:
            # T2V mode
            latent_model_input = state.latents.to(dtype)
            timestep_tensor = t.expand(state.latents.shape[0]) if t.ndim == 0 else t

        return latent_model_input, timestep_tensor

    def _select_model_for_timestep(
        self,
        t: torch.Tensor,
        boundary_timestep: float | None,
    ) -> tuple[nn.Module, float]:
        """Return (transformer, guidance_scale) for a given timestep."""
        guidance_low = self._guidance_scale or 4.0
        guidance_high = self._guidance_scale_2 or guidance_low
        if boundary_timestep is not None and t < boundary_timestep:
            model = self.transformer_2 if self.transformer_2 is not None else self.transformer
            return model, guidance_high
        model = self.transformer if self.transformer is not None else self.transformer_2
        return model, guidance_low

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """One-time request setup: encode prompt, prepare latents, init scheduler."""
        # Extract prompts
        prompt, negative_prompt = self._extract_prompts(state)

        params = self._resolve_generation_params(state)
        height = params["height"]
        width = params["width"]
        num_frames = params["num_frames"]
        chunk_frames = params["chunk_frames"]
        device = params["device"]
        dtype = params["dtype"]
        generator = params["generator"]
        guidance_low = params["guidance_low"]
        guidance_high = params["guidance_high"]

        # Store guidance for properties and for denoise_step model selection
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Encode prompt
        do_cfg = guidance_low > 1.0 or guidance_high > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=state.sampling.num_outputs_per_prompt or 1,
            max_sequence_length=state.sampling.max_sequence_length or 512,
            device=device,
            dtype=dtype,
        )

        # Multi-modal inputs (I2V image, V2V video)
        multi_modal_data = (
            state.prompts[0].get("multi_modal_data", {})
            if state.prompts and not isinstance(state.prompts[0], str)
            else None
        )
        raw_image = multi_modal_data.get("image", None) if multi_modal_data else None
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        v2v_video = multi_modal_data.get("video", None) if multi_modal_data else None
        noise_scale = float((state.sampling.extra_args or {}).get("noise_scale", 0.8))
        sigma_start = noise_scale if (v2v_video is not None and 0.0 < noise_scale < 1.0) else 1.0

        # Scheduler
        self.scheduler.set_timesteps(params["num_steps"], device=device, sigma_start=sigma_start)
        req_scheduler = copy.deepcopy(self.scheduler)

        latent_condition = None
        first_frame_mask = None

        if v2v_video is not None:
            # V2V mode
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=chunk_frames,
                dtype=self.vae.dtype,
                device="meta",  # NOTE: stream mode; latents are prepared per-chunk in `encode_chunk_inputs`
                generator=None,
                latents=state.sampling.latents,
            )
        elif self.expand_timesteps and raw_image is not None:
            # I2V mode
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image)
            else:
                image = cast(PIL.Image.Image | torch.Tensor, raw_image)

            if isinstance(image, PIL.Image.Image):
                image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                image_tensor = video_processor.preprocess(image, height=height, width=width)
            else:
                image_tensor = image

            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]

            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=state.sampling.latents,
            )

            image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = ((latent_condition - latents_mean) * latents_std).to(torch.float32)

            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=state.sampling.latents,
            )

        # Populate state
        state.prompt_embeds = prompt_embeds
        state.negative_prompt_embeds = negative_prompt_embeds
        state.latents = latents
        state.timesteps = req_scheduler.timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = do_cfg and negative_prompt_embeds is not None
        state.guidance = torch.tensor([guidance_low], device=device)

        state.extra["guidance_low"] = guidance_low
        state.extra["guidance_high"] = guidance_high
        state.extra["boundary_timestep"] = params["boundary_timestep"]
        state.extra["expand_timesteps"] = self.expand_timesteps
        state.extra["latent_condition"] = latent_condition
        state.extra["first_frame_mask"] = first_frame_mask
        state.extra["height"] = height
        state.extra["width"] = width

        return state

    def set_pp_recv_dict_buffers(self, state: DiffusionRequestState) -> None:
        from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

        pp_group = get_pp_group()
        if pp_group.world_size == 1:
            return

        slo_fps = getattr(state.sampling, "slo_fps", None)
        slo_max_batch = getattr(state.sampling, "slo_max_batch", 1)
        slo_max_batch = max(1, slo_max_batch if slo_fps else 1)
        num_inference_steps = state.sampling.num_inference_steps or 4

        _, channels, latent_chunk_frames, height, width = state.latents.shape
        p_t, p_h, p_w = self.transformer_config.patch_size
        inner_dim = self.transformer_config.num_attention_heads * self.transformer_config.attention_head_dim
        it_dtype = (self.transformer or self.transformer_2).dtype
        latents_dtype = state.latents.dtype
        
        seq_len = (latent_chunk_frames // p_t) * (height // p_h) * (width // p_w)

        cfg_branches = 2 if state.do_true_cfg else 1

        for batch_size in range(1, slo_max_batch * num_inference_steps + 1):
            latents_template = {
                "latents": torch.empty(batch_size, channels, latent_chunk_frames, height, width, dtype=latents_dtype, device="meta")
            }
            it_template = {
                "hidden_states": torch.empty(batch_size, seq_len, inner_dim, dtype=it_dtype, device="meta")
            }
            pp_group.set_recv_dict_buffer("latents", -1, latents_template, batch_size=batch_size)
            for seg in range(cfg_branches):
                pp_group.set_recv_dict_buffer("intermediate", seg, it_template, batch_size=batch_size)

    def denoise_step(
        self,
        state: DiffusionRequestState,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Run one denoising iteration.

        When ``state.batched_timesteps`` is set (stream-batch path with
        multiple chunks at different ``step_index`` fused along dim 0),
        it overrides ``current_timestep`` and ``_prepare_latent_input``
        forwards the per-row timesteps directly to the transformer.
        Model selection uses the per-row max so a batch straddling the
        high/low noise boundary picks the high-noise transformer (NOTE correct
        for single-stage Wan2.1).
        """
        t = state.batched_timesteps if state.batched_timesteps is not None else state.current_timestep
        self._current_timestep = t

        boundary_timestep = state.extra.get("boundary_timestep")
        t_select = t.max() if t.ndim > 0 else t
        current_model, current_guidance_scale = self._select_model_for_timestep(t_select, boundary_timestep)

        latent_model_input, timestep = self._prepare_latent_input(state, t, current_model.dtype)

        do_true_cfg = current_guidance_scale > 1.0 and state.negative_prompt_embeds is not None

        # Stream-batch path fuses B chunks along dim 0
        B = latent_model_input.shape[0]
        encoder_pos = state.prompt_embeds
        if encoder_pos is not None and encoder_pos.shape[0] == 1 and B > 1:
            encoder_pos = encoder_pos.expand(B, *encoder_pos.shape[1:]).contiguous()
        encoder_neg = state.negative_prompt_embeds
        if encoder_neg is not None and encoder_neg.shape[0] == 1 and B > 1:
            encoder_neg = encoder_neg.expand(B, *encoder_neg.shape[1:]).contiguous()

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_pos,
            "attention_kwargs": {},
            "return_dict": False,
            "current_model": current_model,
        }
        negative_kwargs = (
            {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_neg,
                "attention_kwargs": {},
                "return_dict": False,
                "current_model": current_model,
            }
            if do_true_cfg
            else None
        )
        preposted_its = state.extra.pop("preposted_its", None)

        return self.predict_noise_maybe_with_cfg(
            do_true_cfg=do_true_cfg,
            true_cfg_scale=current_guidance_scale,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            buf_idx=state.step_index % 2,
            batch_size=batch_size,
            preposted_its=preposted_its,
        )

    def prefetch_tensors(self, state: DiffusionRequestState, batch_size: int = 1) -> None:
        """Prefetch next-step tensors: latents on first rank, ITs on others."""
        do_true_cfg = state.do_true_cfg
        buf_idx = state.step_index % 2

        preposted = self.prefetch_tensors_maybe_with_cfg(
            do_true_cfg=do_true_cfg,
            buf_idx=buf_idx,
            batch_size=batch_size,
        )

        if isinstance(preposted, AsyncLatents):
            state.latents = preposted
        elif preposted is not None:
            state.extra["preposted_its"] = preposted

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        *,
        per_request_scheduler: Any | list[Any] | None = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        """Run one scheduler step: update latents and advance step_index.

        ``per_request_scheduler`` may be a single scheduler (B=1) or a list of
        per-chunk schedulers (B>1, last rank loops one ``step()`` per row).
        """
        t = state.batched_timesteps if state.batched_timesteps is not None else state.current_timestep
        do_true_cfg = state.do_true_cfg

        if per_request_scheduler is None:
            per_request_scheduler = state.scheduler

        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            do_true_cfg,
            per_request_scheduler=per_request_scheduler,
            batch_size=batch_size,
        )

        state.step_index += 1

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents after denoising completes."""
        self._sync_pp_send()
        self._current_timestep = None

        # I2V: blend final latents with condition
        latent_condition = state.extra.get("latent_condition")
        first_frame_mask = state.extra.get("first_frame_mask")
        if state.extra.get("expand_timesteps") and latent_condition is not None:
            state.latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * state.latents

        latents = state.latents.to(self.vae.dtype)
        
        latents_mean = self._latents_mean.to(latents.device, latents.dtype)
        latents_inv_std = self._latents_inv_std.to(
            latents.device, latents.dtype
        )

        latents = latents / latents_inv_std + latents_mean
        output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def encode_chunk_inputs(
        self,
        state: DiffusionRequestState,
        new_idxs: list[int],
    ) -> torch.Tensor:
        """Streaming V2V initial latents (StreamDiffusionV2-style).

        For each newly admitted chunk: VAE-encode the source frames at
        ``chunk_idx * chunk_frames : (idx+1) * chunk_frames``, normalize, and
        linearly blend with Gaussian noise via ``extra_args["noise_scale"]``
        (default 0.8). The transformer then runs the full schedule starting at
        ``step_index=0`` on this partially-noised latent.
        """
        batch_size = len(new_idxs)
        noise_scale = float((state.sampling.extra_args or {}).get("noise_scale", 0.8))
        chunk_frames = state.sampling.chunk_frames
        prompt = state.prompts[0] if state.prompts else None
        video = None
        if prompt is not None and not isinstance(prompt, str):
            video = (prompt.get("multi_modal_data") or {}).get("video")
        if video is None:
            raise ValueError(
                "encode_chunk_inputs requires V2V source frames in prompts[0]['multi_modal_data']['video']"
            )

        latents_mean = self._latents_mean.to(self.device)
        latents_inv_std = self._latents_inv_std.to(self.device)

        controls: list[torch.Tensor] = []
        for idx in new_idxs:
            start = idx * chunk_frames
            end = start + chunk_frames
            if isinstance(video, list):
                frames = torch.stack(list(video[start:end]), dim=0)
            else:
                frames = video[start:end]

            if frames.shape[0] < chunk_frames:
                pad = torch.zeros(
                    (chunk_frames - frames.shape[0], *frames.shape[1:]),
                    dtype=frames.dtype, device=frames.device,
                )
                frames = torch.cat([frames, pad], dim=0)

            # [n, C, H, W] -> [1, C, n, H, W]
            controls.append(
                frames.permute(1, 0, 2, 3)
                .unsqueeze(0)
                .to(device=self.device, dtype=self.vae.dtype)
            )
        control = torch.cat(controls, dim=0)

        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.transformer_config.in_channels,
            height=state.extra["height"],
            width=state.extra["width"],
            num_frames=chunk_frames,
            dtype=self.vae.dtype,
            device=self.device,
            generator=state.sampling.generator,
            latents=None,
        )

        latent_condition = retrieve_latents(self.vae.encode(control), sample_mode="argmax")
        latent_condition = (
            (latent_condition.float() - latents_mean.to(latent_condition.dtype))
            * latents_inv_std.to(latent_condition.dtype)
        ).to(self.vae.dtype)

        return latents * noise_scale + latent_condition * (1.0 - noise_scale)
