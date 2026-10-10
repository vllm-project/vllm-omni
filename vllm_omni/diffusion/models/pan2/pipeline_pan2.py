# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN2 video generation pipeline: text-to-video, and image-to-video from a first frame."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, TypeAlias, cast

import numpy as np
import PIL.Image
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Qwen3_5TextModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend, RequestScopedCacheDiTRuntime
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.pan2.pan2_transformer import PAN2Transformer3DModel
from vllm_omni.diffusion.models.pan2.quality_policy import PAN2_GENERIC_CACHE_KEY, PAN2QualityPolicy
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest, resolve_video_num_frames
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

PAN2_SYSTEM_MESSAGE = "You are a helpful assistant. Generate the video with the following instructions."
# PAN2 conditions on `hidden_states[-3]` of the text encoder, i.e. it skips its last two layers.
PAN2_TEXT_ENCODER_LAYERS_TO_SKIP = 2
PAN2_MAX_SEQUENCE_LENGTH = 1536
PAN2_DEFAULT_HEIGHT = 704
PAN2_DEFAULT_WIDTH = 1248
PAN2_DEFAULT_NUM_FRAMES = 313
PAN2_MAX_NUM_OUTPUTS = 10

# Latents, the frames of one video (PIL images), or the frames as arrays.
PAN2PostProcessOutput: TypeAlias = torch.Tensor | list[PIL.Image.Image] | np.ndarray | list[np.ndarray]


def get_pan2_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    from vllm_omni.diffusion.models.pan2.guardrails import check_text_safety, ensure_initialized, is_guardrails_enabled

    # Eager-load the guardrail models at pipeline build time when the server-level gate is on. Per-request overrides
    # only decide whether the loaded models are invoked.
    if is_guardrails_enabled(od_config):
        ensure_initialized(od_config)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        # Reject unsupported requests here, in the engine process: an OmniClientError raised in `forward` on the
        # workers reaches the client as a 500 under the multi-process executor, which drops its 4xx status.
        _check_pan2_request(request.sampling_params, request.prompt)
        if is_guardrails_enabled(od_config, request.sampling_params):
            prompt = request.prompt
            check_text_safety(prompt if isinstance(prompt, str) else prompt.get("prompt") or "")
        return request

    return pre_process_func


def get_pan2_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., PAN2PostProcessOutput]:
    from vllm_omni.diffusion.models.pan2.guardrails import check_video_safety, is_guardrails_enabled

    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "pil",
        sampling_params: OmniDiffusionSamplingParams | None = None,
    ) -> PAN2PostProcessOutput:
        if sampling_params is not None and sampling_params.output_type is not None:
            output_type = sampling_params.output_type
        if output_type == "latent":
            return video
        checked_frames = check_video_safety(video) if is_guardrails_enabled(od_config, sampling_params) else None
        if checked_frames is not None and output_type == "pil":
            # Reuse the uint8 frames the guardrail checked; they are the pixels `postprocess_video` would deliver.
            result = [[PIL.Image.fromarray(frame) for frame in frames] for frames in checked_frames]
        else:
            result = video_processor.postprocess_video(video, output_type=output_type)
        # postprocess_video returns a batch of frame lists; a single video is returned as its frames.
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return result

    return post_process_func


def _resolve_pan2_num_outputs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OmniClientError(f"PAN2 num_outputs_per_prompt must be an integer in [1, {PAN2_MAX_NUM_OUTPUTS}]")
    if not 1 <= value <= PAN2_MAX_NUM_OUTPUTS:
        raise OmniClientError(f"PAN2 num_outputs_per_prompt must be in [1, {PAN2_MAX_NUM_OUTPUTS}], got {value}")
    return value


def _check_pan2_request(sampling_params: OmniDiffusionSamplingParams, prompt: OmniPromptType) -> None:
    """Reject the inputs PAN2 does not support."""
    if sampling_params.sigmas is not None or sampling_params.timesteps is not None:
        raise OmniClientError(
            "PAN2 uses its own flow-matching schedule; custom `sigmas`/`timesteps` are not supported."
        )
    multi_modal_data = (prompt.get("multi_modal_data") or {}) if isinstance(prompt, dict) else {}
    if multi_modal_data.get("video") is not None:
        raise OmniClientError("PAN2 does not accept video input; pass a single first-frame image for image-to-video.")


class PAN2Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportsComponentDiscovery,
):
    """PAN2 video generator. A request with an image in ``multi_modal_data`` runs image-to-video from that frame."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    # Distributed layerwise offload: the text refiner streams with the main blocks, and the Qwen3.5 text encoder
    # streams its layers.
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        block_attrs={"transformer": ("context_refiner", "transformer_blocks")},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("layers",)},
        encoder_dlo_weight_replication=frozenset({"text_encoder"}),
    )
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        parallel_config = od_config.parallel_config
        if parallel_config.ring_degree > 1 or parallel_config.allgather_degree > 1:
            raise NotImplementedError(
                "PAN2 does not support ring or all-gather sequence parallel: its padded joint video/text sequence "
                "needs an attention mask they do not carry. Use Ulysses (--usp) and CFG parallelism instead."
            )
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        revision = od_config.revision
        local_files_only = os.path.exists(model)
        subfolders = ["tokenizer", "text_encoder", "vae", "scheduler"]
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only, revision=revision)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only, revision=revision
        )
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3_5TextModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            revision=revision,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            revision=revision,
            torch_dtype=dtype,
        ).to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only, revision=revision
        )
        if od_config.flow_shift is not None:
            # The scheduler exposes `shift` as a read-only property.
            self.scheduler._shift = od_config.flow_shift

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, PAN2Transformer3DModel)
        self.transformer = PAN2Transformer3DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )
        # Cache-DiT tells the CFG branches apart by counting transformer calls in pairs, which only holds when the
        # branches run as two calls on the same rank. CFG parallelism makes one call per step on each rank.
        self.sequential_cfg_branches = od_config.parallel_config.cfg_parallel_size == 1
        self.transformer._cache_dit_adapter_config = dataclasses.replace(
            self.transformer._cache_dit_adapter_config, has_separate_cfg=self.sequential_cfg_branches
        )
        self._quality_policy = PAN2QualityPolicy(od_config)
        self._cache_dit_runtime = RequestScopedCacheDiTRuntime(self)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.num_channels_latents = self.vae.config.z_dim
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self._guidance_scale: float | None = None
        self._num_timesteps: int | None = None
        self._current_timestep: torch.Tensor | None = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self) -> float | None:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int | None:
        return self._num_timesteps

    @property
    def current_timestep(self) -> torch.Tensor | None:
        return self._current_timestep

    def encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        # An empty prompt is encoded as a single space, as in training.
        messages = [{"role": "system", "content": PAN2_SYSTEM_MESSAGE}, {"role": "user", "content": prompt or " "}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True
        ).input_ids

        # PAN2 conditions on the tokens after the `<|im_start|>user\n` header: the prompt and the assistant header.
        header_ids = self.tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        crop_start = next(
            (i + len(header_ids) for i in range(len(input_ids)) if input_ids[i : i + len(header_ids)] == header_ids),
            None,
        )
        if crop_start is None:
            raise ValueError("The PAN2 tokenizer's chat template has no `<|im_start|>user\\n` header.")
        input_ids = torch.tensor([input_ids[: crop_start + PAN2_MAX_SEQUENCE_LENGTH]], device=device)

        prompt_embeds = self.text_encoder(input_ids=input_ids, output_hidden_states=True).hidden_states[
            -(PAN2_TEXT_ENCODER_LAYERS_TO_SKIP + 1)
        ]
        return prompt_embeds[:, crop_start:]

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        # The latents stay in float32 across the denoising loop and are cast to the transformer dtype per step.
        if latents is not None:
            if latents.shape[0] != batch_size:
                raise ValueError(f"`latents` must have batch size {batch_size}, got {latents.shape[0]}.")
            return latents.to(device=device, dtype=torch.float32)
        shape = (
            batch_size,
            self.num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

    def prepare_condition_latents(
        self, latents: torch.Tensor, image: PIL.Image.Image | None, height: int, width: int
    ) -> torch.Tensor:
        """Conditioning latents and mask stacked along channels; image-to-video fills the first latent frame.

        The image is encoded once and shared by every video of the batch.
        """
        batch_size, num_channels, num_frames, latent_height, latent_width = latents.shape
        condition_latents = latents.new_zeros(batch_size, num_channels + 1, num_frames, latent_height, latent_width)
        if image is None:
            return condition_latents

        # Scale the image to cover the canvas, rounding its size, and center crop it, as PAN2 was trained.
        scale = max(width / image.width, height / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), resample=PIL.Image.LANCZOS)
        left, top = round((image.width - width) / 2), round((image.height - height) / 2)
        image = image.crop((left, top, left + width, top + height))
        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.unsqueeze(2).to(device=latents.device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(image_latents)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(image_latents)
        image_latents = (image_latents - latents_mean) / latents_std

        condition_latents[:, :num_channels, :1] = image_latents
        condition_latents[:, num_channels:, :1] = 1.0
        return condition_latents

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        """Take over the Cache-DiT backend the runner enabled at startup, so each request can switch it."""
        self._cache_dit_runtime.adopt(backend, installation_key=PAN2_GENERIC_CACHE_KEY)

    def is_cache_dit_enabled(self) -> bool:
        return self._cache_dit_runtime.is_enabled

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)

    def forward(
        self,
        req: DiffusionRequestBatch,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        height: int = PAN2_DEFAULT_HEIGHT,
        width: int = PAN2_DEFAULT_WIDTH,
        num_frames: int = PAN2_DEFAULT_NUM_FRAMES,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("PAN2 takes a single prompt per request.")
        sampling_params = req.sampling_params
        num_outputs = _resolve_pan2_num_outputs(sampling_params.num_outputs_per_prompt or 1)
        prompt_data = req.prompts[0]
        _check_pan2_request(sampling_params, prompt_data)
        if isinstance(prompt_data, str):
            prompt, negative_prompt, multi_modal_data = prompt_data, None, {}
        else:
            prompt = prompt_data.get("prompt")
            negative_prompt = prompt_data.get("negative_prompt")
            multi_modal_data = prompt_data.get("multi_modal_data") or {}

        image = multi_modal_data.get("image")
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(f"PAN2 conditions on a single first frame, got {len(image)} images.")
            image = image[0]
        if isinstance(image, str):
            image = PIL.Image.open(image)
        if image is not None:
            image = cast(PIL.Image.Image, image).convert("RGB")

        output_type = sampling_params.output_type or output_type
        height = sampling_params.height or height
        width = sampling_params.width or width
        num_frames = resolve_video_num_frames(
            sampling_params.num_frames, default_num_frames=num_frames, is_dummy_run=req.is_dummy_run()
        )
        num_steps = sampling_params.num_inference_steps or num_inference_steps
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale
        self._guidance_scale = guidance_scale
        do_cfg = guidance_scale > 1.0

        cache_dit_spec = self._quality_policy.resolve(quality=sampling_params.quality, num_inference_steps=num_steps)
        if cache_dit_spec is not None and self.sequential_cfg_branches and not do_cfg:
            logger.warning_once(
                "PAN2 runs without Cache-DiT for requests without CFG when CFG branches run sequentially."
            )
            cache_dit_spec = None
        self._cache_dit_runtime.prepare(cache_dit_spec)

        multiple = self.vae_scale_factor_spatial * self.transformer.patch_size[1]
        if height % multiple != 0 or width % multiple != 0:
            raise ValueError(f"`height` and `width` must be divisible by {multiple}, got {height} and {width}.")
        if (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            raise ValueError(
                f"`num_frames - 1` must be divisible by {self.vae_scale_factor_temporal}, got {num_frames}."
            )

        device = self.device
        dtype = self.transformer.x_embedder.weight.dtype

        if generator is None:
            # One generator per video: a single request generator draws the videos' noise in turn.
            generator = req.collate_request_generators(num_outputs, None)
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=sampling_params.generator_device or device).manual_seed(
                sampling_params.seed
            )

        # The videos share one prompt, so their text embeddings are repeated without padding.
        prompt_embeds = self.encode_prompt(prompt, device).to(dtype).repeat_interleave(num_outputs, dim=0)
        negative_prompt_embeds = None
        if do_cfg:
            negative_prompt_embeds = (
                self.encode_prompt(negative_prompt or "", device).to(dtype).repeat_interleave(num_outputs, dim=0)
            )

        latents = self.prepare_latents(
            num_outputs, height, width, num_frames, device, generator, sampling_params.latents
        )
        condition_latents = self.prepare_condition_latents(latents, image, height, width).to(dtype)

        # PAN2 samples on uniformly spaced sigmas from 1 down to (excluding) 0; the scheduler applies the shift.
        sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t

                latent_model_input = torch.cat([latents.to(dtype), condition_latents], dim=1)
                timestep = t.expand(latent_model_input.shape[0])

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                }
                negative_kwargs = None
                if do_cfg:
                    negative_kwargs = {**positive_kwargs, "encoder_hidden_states": negative_prompt_embeds}

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=sampling_params.cfg_normalize,
                )
                latents = self.scheduler_step_maybe_with_cfg(noise_pred.float(), t, latents, do_true_cfg=do_cfg)
                pbar.update()

        self._current_timestep = None
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        if output_type == "latent":
            output = latents
        else:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
            latents = (latents * latents_std + latents_mean).to(self.vae.dtype)
            # Decode one video at a time, so the VAE peak memory does not grow with the number of videos.
            output = torch.cat(
                [self.vae.decode(video_latents, return_dict=False)[0] for video_latents in latents.split(1)]
            )

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
