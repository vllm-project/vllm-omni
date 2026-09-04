# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2025 SANA-Video Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501

import inspect
import os
import warnings
from collections.abc import Iterable

import numpy as np
import PIL.Image
import torch
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoModel, Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
    DistributedAutoencoderKLLTX2Video,
)
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin, _wrap
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import resolve_video_num_frames
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .pipeline_output import SanaVideoPipelineOutput
from .transformer_sana_video import SanaVideoTransformer3DModel, validate_sana_video_parallel_config


def _resolve_vae_class_and_dtype(
    vae_class_name: str,
    pipeline_dtype: torch.dtype,
) -> tuple[type[DistributedAutoencoderKLLTX2Video | DistributedAutoencoderKLWan], torch.dtype]:
    vae_classes = {
        "AutoencoderKLLTX2Video": DistributedAutoencoderKLLTX2Video,
        "AutoencoderKLWan": DistributedAutoencoderKLWan,
    }
    if vae_class_name not in vae_classes:
        raise ValueError(f"Unsupported SANA-Video VAE class: {vae_class_name!r}")
    vae_dtype = torch.float32 if vae_class_name == "AutoencoderKLWan" else pipeline_dtype
    return vae_classes[vae_class_name], vae_dtype


def _load_sana_tokenizer(
    model: str,
    component_subfolders: list[str],
    local_files_only: bool,
) -> GemmaTokenizer:
    # Transformers v5 AutoTokenizer first tries to resolve an AutoConfig from
    # the requested subfolder. Released SANA-Video repositories correctly
    # contain tokenizer metadata under tokenizer/, but their model config lives
    # under text_encoder/, so that extra tokenizer/config.json lookup fails for
    # Hub model IDs. Both released checkpoints use GemmaTokenizer; loading the
    # concrete class also matches Diffusers and avoids the unrelated config
    # lookup.
    return from_pretrained_with_prefetch(
        GemmaTokenizer.from_pretrained,
        model,
        subfolder="tokenizer",
        prefetch_list=component_subfolders,
        local_files_only=local_files_only,
    )


ASPECT_RATIO_480_BIN = {
    "0.5": [448.0, 896.0],
    "0.57": [480.0, 832.0],
    "0.68": [528.0, 768.0],
    "0.78": [560.0, 720.0],
    "1.0": [624.0, 624.0],
    "1.13": [672.0, 592.0],
    "1.29": [720.0, 560.0],
    "1.46": [768.0, 528.0],
    "1.67": [816.0, 496.0],
    "1.75": [832.0, 480.0],
    "2.0": [896.0, 448.0],
}


ASPECT_RATIO_720_BIN = {
    "0.5": [672.0, 1344.0],
    "0.57": [704.0, 1280.0],
    "0.68": [800.0, 1152.0],
    "0.78": [832.0, 1088.0],
    "1.0": [960.0, 960.0],
    "1.13": [1024.0, 896.0],
    "1.29": [1088.0, 832.0],
    "1.46": [1152.0, 800.0],
    "1.67": [1248.0, 736.0],
    "1.75": [1280.0, 704.0],
    "2.0": [1344.0, 672.0],
}


SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION = (
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
)


def get_sana_video_default_resolution(sample_size: int) -> tuple[int, int]:
    if sample_size == 30:
        return 480, 832
    if sample_size == 22:
        return 704, 1280
    raise ValueError(f"Unsupported SANA-Video sample size: {sample_size}")


def resolve_sana_video_sample_size(od_config: OmniDiffusionConfig) -> int:
    tf_model_config = getattr(od_config, "tf_model_config", None)
    sample_size = tf_model_config.get("sample_size", None) if tf_model_config is not None else None
    model = getattr(od_config, "model", None)
    if sample_size is None and model:
        try:
            transformer_config = get_hf_file_to_dict("transformer/config.json", model) or {}
            sample_size = transformer_config.get("sample_size")
        except (OSError, ValueError):
            pass
    if sample_size is None:
        sample_size = 22 if model and "720p" in str(model).lower() else 30
    return int(sample_size)


def get_sana_video_post_process_func(od_config: OmniDiffusionConfig):
    video_processor = VideoProcessor()
    is_diffusers_adapter = od_config.diffusion_load_format == "diffusers"

    def post_process_func(video: torch.Tensor, output_type: str = "np", sampling_params=None):
        if output_type == "latent" or (
            sampling_params is not None and getattr(sampling_params, "output_type", None) == "latent"
        ):
            return video

        first_frame = video
        while isinstance(first_frame, (list, tuple)) and first_frame:
            first_frame = first_frame[0]
        is_postprocessed_frame_sequence = isinstance(video, np.ndarray) or isinstance(
            first_frame,
            (PIL.Image.Image, np.ndarray),
        )
        if is_diffusers_adapter and is_postprocessed_frame_sequence:
            if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
                raise ValueError(
                    "Frame interpolation is not supported for SANA-Video with the Diffusers adapter because "
                    "Diffusers returns already postprocessed frames."
                )
            return {"payload": {"video": video}, "metadata": {}}

        video_metadata = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            video_metadata["video_fps_multiplier"] = multiplier
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {"video": video_metadata} if video_metadata else {},
        }

    return post_process_func


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    scheduler_device = torch.device("cpu")
    # vLLM initializes models under a CUDA default-device context. Diffusers'
    # schedulers are not nn.Modules, so their tensors can be created on CUDA and
    # are not moved by pipeline.to(). DPM solvers also perform NumPy operations
    # while constructing their schedule. Move scheduler state and temporarily
    # override the global default device so every intermediate remains on CPU.
    for name, value in vars(scheduler).items():
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            setattr(scheduler, name, value.cpu())
    previous_default_device = torch.get_default_device()
    try:
        torch.set_default_device(scheduler_device)
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=scheduler_device, **kwargs)
            num_inference_steps = len(scheduler.timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=scheduler_device, **kwargs)
            num_inference_steps = len(scheduler.timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=scheduler_device, **kwargs)
    finally:
        torch.set_default_device(previous_default_device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _validate_cache_offload_parallelism(od_config: OmniDiffusionConfig) -> None:
    cache_backend = od_config.cache_backend
    if cache_backend not in ("none", "cache_dit"):
        raise NotImplementedError(
            f"Cache backend {cache_backend!r} is not supported by the native SANA-Video pipeline; "
            "use 'cache_dit' or 'none'."
        )
    if cache_backend == "cache_dit" and od_config.enable_distributed_layerwise_offload:
        raise NotImplementedError(
            "SANA-Video does not support Cache-DiT with distributed layerwise offload; "
            "per-rank cache skips desynchronize the weight AllGather."
        )

    offload_enabled = (
        od_config.enable_cpu_offload
        or od_config.enable_layerwise_offload
        or od_config.enable_distributed_layerwise_offload
    )
    if cache_backend == "none" and not offload_enabled:
        return

    parallel_config = od_config.parallel_config
    for name, size in (
        ("tensor_parallel_size", parallel_config.tensor_parallel_size),
        ("cfg_parallel_size", parallel_config.cfg_parallel_size),
        ("sequence_parallel_size", parallel_config.sequence_parallel_size),
    ):
        if size > 1:
            raise NotImplementedError(
                f"SANA-Video cache/offload is currently supported only with TP1, CFG1, and SP1; got {name}={size}."
            )


class SanaVideoPipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    r"""
    vLLM-Omni pipeline for text-to-video generation using
    [Sana](https://huggingface.co/papers/2509.24695).

    Args:
        tokenizer ([`GemmaTokenizer`] or [`GemmaTokenizerFast`]):
            The tokenizer used to tokenize the prompt.
        text_encoder ([`Gemma2PreTrainedModel`]):
            Text encoder model to encode the input prompts.
        vae ([`DistributedAutoencoderKLWan`] or [`DistributedAutoencoderKLLTX2Video`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer ([`SanaVideoTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`DPMSolverMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]
    supports_step_execution = False
    default_num_inference_steps = 50

    def __init__(
        self,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast | None = None,
        text_encoder: Gemma2PreTrainedModel | None = None,
        vae: DistributedAutoencoderKLLTX2Video | DistributedAutoencoderKLWan | None = None,
        transformer: SanaVideoTransformer3DModel | None = None,
        scheduler: DPMSolverMultistepScheduler | None = None,
        *,
        od_config: OmniDiffusionConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        if od_config is not None:
            validate_sana_video_parallel_config(od_config.parallel_config)

        self.od_config = od_config
        self.device = get_local_device()
        self.weights_sources: list[DiffusersPipelineLoader.ComponentSource] = []
        if od_config is not None:
            _validate_cache_offload_parallelism(od_config)
            tokenizer, text_encoder, vae, transformer, scheduler = self._load_components(od_config, prefix)

        if tokenizer is None or text_encoder is None or vae is None or transformer is None or scheduler is None:
            raise ValueError("SanaVideoPipeline requires tokenizer, text_encoder, vae, transformer, and scheduler.")

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        if od_config is None:
            first_parameter = next(self.transformer.parameters(), None)
            if first_parameter is not None:
                self.device = first_parameter.device

        if isinstance(self.vae, DistributedAutoencoderKLLTX2Video):
            self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio
            self.vae_scale_factor_spatial = self.vae.config.spatial_compression_ratio
        elif isinstance(self.vae, DistributedAutoencoderKLWan):
            self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
            self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        else:
            raise TypeError(f"Unsupported SANA-Video VAE type: {type(self.vae).__name__}")

        self.vae_scale_factor = self.vae_scale_factor_spatial

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        if od_config is not None:
            self.setup_diffusion_pipeline_profiler(
                enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
                profiler_targets=["forward"],
            )

    def _load_components(
        self,
        od_config: OmniDiffusionConfig,
        prefix: str,
    ) -> tuple[
        GemmaTokenizer | GemmaTokenizerFast,
        Gemma2PreTrainedModel,
        DistributedAutoencoderKLLTX2Video | DistributedAutoencoderKLWan,
        SanaVideoTransformer3DModel,
        DPMSolverMultistepScheduler,
    ]:
        model = od_config.model
        local_files_only = os.path.exists(model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        # The loader loads components under a default-device context that is CPU
        # when offload is enabled; runtime placement stays with the offloader.
        component_load_device = torch.get_default_device()
        # Transformer weights are streamed by DiffusersPipelineLoader below.
        # Prefetch only components loaded through from_pretrained here.
        component_subfolders = ["tokenizer", "text_encoder", "vae", "scheduler"]
        prefetch_subfolders(model, component_subfolders, local_files_only=local_files_only)

        tokenizer = _load_sana_tokenizer(
            model,
            component_subfolders,
            local_files_only=local_files_only,
        )
        text_encoder = from_pretrained_with_prefetch(
            AutoModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(component_load_device)

        model_index = _load_json(model, "model_index.json", local_files_only)
        vae_class_name = model_index.get("vae", [None, "AutoencoderKLWan"])[1]
        # The released 480p checkpoint uses Wan VAE weights that are decoded
        # in FP32 in the upstream reference recipe. The 720p LTX2 VAE is
        # validated in BF16.
        vae_class, vae_dtype = _resolve_vae_class_and_dtype(vae_class_name, dtype)
        vae = from_pretrained_with_prefetch(
            vae_class.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=vae_dtype,
        ).to(component_load_device)

        transformer_config = _load_json(model, "transformer/config.json", local_files_only)
        transformer = SanaVideoTransformer3DModel.from_config(transformer_config).to(
            dtype=dtype,
            device=component_load_device,
        )
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.weights_sources[:] = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix=f"{prefix}transformer." if prefix else "transformer.",
                fall_back_to_pt=True,
            )
        ]
        return tokenizer, text_encoder, vae, transformer, scheduler

    # Copied from diffusers.pipelines.sana.pipeline_sana.SanaPipeline._get_gemma_prompt_embeds
    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: list[str] | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)

        # prepare complex human instruction
        if not complex_human_instruction:
            max_length_all = max_sequence_length
        else:
            chi_prompt = "\n".join(complex_human_instruction)
            prompt = [chi_prompt + p for p in prompt]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + max_sequence_length - 2

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0].to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_videos_per_prompt: int = 1,
        device: torch.device | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: list[str] | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                number of videos that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """

        if device is None:
            device = self.device

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        # See Section 3.1. of the paper.
        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=complex_human_instruction,
            )

            prompt_embeds = prompt_embeds[:, select_index]
            prompt_attention_mask = prompt_attention_mask[:, select_index]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=False,
            )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_videos_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        patch_size = getattr(self.transformer.config, "patch_size", (1, 1, 1))
        spatial_alignment = self.vae_scale_factor_spatial * max(patch_size[-2:])
        if height % spatial_alignment != 0 or width % spatial_alignment != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {spatial_alignment} but are {height} and {width}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption:
            raise ValueError(
                "SANA-Video native pipeline does not support `clean_caption=True`; "
                "pre-clean the prompt or use the Diffusers adapter."
            )
        if not isinstance(text, (tuple, list)):
            text = [text]
        return [str(item).lower().strip() for item in text]

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
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
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def predict_noise(self, *args, **kwargs) -> torch.Tensor:
        # return_dict=False: the dataclass output is not subscriptable.
        return self.transformer(*args, **kwargs, return_dict=False)[0]

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale,
        cfg_normalize=False,
        kwargs=None,
    ):
        # fp32 combine so the bf16 all-gather path matches CFG1 bit-for-bit.
        return super().combine_cfg_noise(
            tuple(p.float() for p in _wrap(positive_noise_pred)),
            tuple(n.float() for n in _wrap(negative_noise_pred)),
            true_cfg_scale,
            cfg_normalize,
            kwargs,
        )

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        guidance_scale: float,
        extra_step_kwargs: dict,
        dtype: torch.dtype,
        output_slice: int | None,
    ) -> torch.Tensor:
        do_true_cfg = self.do_classifier_free_guidance
        # Single-process runs never initialize the parallel groups.
        cfg_parallel = model_parallel_is_initialized() and get_classifier_free_guidance_world_size() > 1
        if cfg_parallel:
            self.check_cfg_parallel_validity(guidance_scale, negative_prompt_embeds is not None)

        if do_true_cfg and not cfg_parallel:
            # Concatenate neg/pos for a single batch-2 forward.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for t in timesteps:
                if self.interrupt:
                    continue

                if cfg_parallel:
                    timestep = t.expand(latents.shape[0])
                    latent_model_input = latents.to(dtype)
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "encoder_hidden_states": prompt_embeds.to(dtype),
                        "encoder_attention_mask": prompt_attention_mask,
                        "timestep": timestep,
                    }
                    negative_kwargs = (
                        {
                            "hidden_states": latent_model_input,
                            "encoder_hidden_states": negative_prompt_embeds.to(dtype),
                            "encoder_attention_mask": negative_prompt_attention_mask,
                            "timestep": timestep,
                        }
                        if do_true_cfg
                        else None
                    )
                    noise_pred = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_true_cfg,
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                        output_slice=output_slice,
                    ).float()
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_true_cfg else latents
                    timestep = t.expand(latent_model_input.shape[0])
                    noise_pred = self.transformer(
                        latent_model_input.to(dtype),
                        encoder_hidden_states=prompt_embeds.to(dtype),
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=timestep,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float()
                    if do_true_cfg:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if output_slice is not None:
                        noise_pred = noise_pred[:, :output_slice]

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                progress_bar.update()

        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        transformer_weights = (
            (name.removeprefix("transformer."), tensor) for name, tensor in weights if name.startswith("transformer.")
        )
        return {f"transformer.{name}" for name in self.transformer.load_weights(transformer_weights)}

    def __call__(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        return self.forward(req)

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("SANA-Video currently supports exactly one prompt per request.")
        prompt_obj = req.prompts[0]
        prompt = prompt_obj if isinstance(prompt_obj, str) else (prompt_obj.get("prompt") or "")
        negative_prompt = "" if isinstance(prompt_obj, str) else (prompt_obj.get("negative_prompt") or "")
        if not prompt:
            raise ValueError("Prompt is required for SANA-Video generation.")

        sampling = req.sampling_params
        default_height, default_width = get_sana_video_default_resolution(self.transformer.config.sample_size)
        height = sampling.height or default_height
        width = sampling.width or default_width
        num_frames = resolve_video_num_frames(
            sampling.num_frames,
            default_num_frames=81,
            is_dummy_run=req.is_dummy_run(),
        )
        num_steps = (
            sampling.num_inference_steps
            if sampling.num_inference_steps is not None
            else self.default_num_inference_steps
        )
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 6.0
        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=get_local_device()).manual_seed(sampling.seed)

        extra_args = sampling.extra_args or {}
        motion_score = extra_args.get("motion_score")
        if motion_score is not None:
            prompt = f"{prompt} motion score: {int(motion_score)}."

        result = self._generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            timesteps=sampling.timesteps,
            sigmas=sampling.sigmas,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=sampling.num_outputs_per_prompt or 1,
            height=height,
            width=width,
            frames=num_frames,
            eta=sampling.eta,
            generator=generator,
            latents=sampling.latents,
            output_type="latent" if sampling.output_type == "latent" else "raw",
            clean_caption=bool(extra_args.get("clean_caption", False)),
            use_resolution_binning=bool(extra_args.get("use_resolution_binning", True)),
            max_sequence_length=sampling.max_sequence_length or 300,
        )
        return DiffusionOutput(output=result.frames)

    @torch.no_grad()
    def _generate(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        timesteps: list[int] = None,
        sigmas: list[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int | None = 1,
        height: int = 480,
        width: int = 832,
        frames: int = 81,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        complex_human_instruction: list[str] | None = None,
    ) -> SanaVideoPipelineOutput | tuple:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            timesteps (`list[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`list[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to
                the text `prompt`, usually at the expense of lower video quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            height (`int`, *optional*, defaults to 480):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 832):
                The width in pixels of the generated video.
            frames (`int`, *optional*, defaults to 81):
                The number of frames in the generated video.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video. Choose between mp4 or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SanaVideoPipelineOutput`] instead of a plain tuple.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Advanced caption cleaning is not supported by the native pipeline. Passing `True` raises an error;
                pre-clean the prompt or use the Diffusers adapter.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_480_BIN` or `ASPECT_RATIO_720_BIN`. After the produced latents are decoded into videos,
                they are resized back to the requested resolution. Useful for generating non-square videos.
            max_sequence_length (`int` defaults to `300`):
                Maximum sequence length to use with the `prompt`.
            complex_human_instruction (`list[str]`, *optional*):
                Instructions for complex human attention:
                https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

        Examples:

        Returns:
            [`~pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated videos
        """
        if complex_human_instruction is None:
            complex_human_instruction = list(SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION)

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 30:
                aspect_ratio_bin = ASPECT_RATIO_480_BIN
            elif self.transformer.config.sample_size == 22:
                aspect_ratio_bin = ASPECT_RATIO_720_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.video_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
        )
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            height,
            width,
            frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        # learned sigma: transformer predicts 2x latent channels; keep the first half.
        output_slice = latent_channels if self.transformer.config.out_channels // 2 == latent_channels else None
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            dtype=self.transformer.dtype,
            output_slice=output_slice,
        )

        if output_type == "latent":
            video = latents
        else:
            latents = latents.to(self.vae.dtype)
            if isinstance(self.vae, DistributedAutoencoderKLLTX2Video):
                latents_mean = self.vae.latents_mean
                latents_std = self.vae.latents_std
                z_dim = self.vae.config.latent_channels
            elif isinstance(self.vae, DistributedAutoencoderKLWan):
                latents_mean = torch.tensor(self.vae.config.latents_mean)
                latents_std = torch.tensor(self.vae.config.latents_std)
                z_dim = self.vae.config.z_dim
            else:
                latents_mean = torch.zeros(latents.shape[1], device=latents.device, dtype=latents.dtype)
                latents_std = torch.ones(latents.shape[1], device=latents.device, dtype=latents.dtype)
                z_dim = latents.shape[1]

            latents_mean = latents_mean.view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / latents_std.view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            try:
                video = self.vae.decode(latents, return_dict=False)[0]
            except torch.OutOfMemoryError as e:
                warnings.warn(
                    f"{e}. \n"
                    f"Try to use VAE tiling for large images. For example: \n"
                    f"pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)"
                )
                raise

            if use_resolution_binning:
                video = self.video_processor.resize_and_crop_tensor(video, orig_width, orig_height)

            if output_type != "raw":
                video = self.video_processor.postprocess_video(video, output_type=output_type)

        if not return_dict:
            return (video,)

        return SanaVideoPipelineOutput(frames=video)
