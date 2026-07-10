# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base implementation for the independent LTX-2.3 pipelines in vLLM-Omni.

This implementation does NOT inherit from LTX2Pipeline because:
- LTX-2.3 connectors run per_token_rms_norm + per-modality video/audio
  projection internally (per_modality_projections=True),
  versus LTX-2's per_layer_masked_mean_norm + shared projection path
- LTX-2.3 uses a BWE vocoder outputting 48kHz audio (not 16kHz)
- LTX-2.3 transformer requires the sigma parameter for prompt modulation
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from diffusers import AutoencoderKLLTX2Audio, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import DistributedAutoencoderKLLTX2Video
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

from . import ltx2_3_guidance as guidance_ops
from . import ltx2_3_latent_preparation as latent_prep
from .ltx2_3_connector import install_ltx23_official_connector_processors
from .ltx2_3_denoise_scheduling import LTX23SchedulerMixin
from .ltx2_3_guidance_parallel import LTX23GuidanceParallelMixin
from .ltx2_3_misc import (
    LTX23RequestMixin,
    _LTX23RequestInputs,
    create_transformer_from_config,
    load_ltx23_weights,
    load_transformer_config,
)
from .ltx2_3_misc import (
    detect_vocoder_output_sample_rate as _detect_vocoder_output_sample_rate,
)
from .ltx2_3_recipes import LTX23PipelineRecipe

logger = init_logger(__name__)


def _is_output_rank() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _vae_decode_needs_all_ranks(vae: Any) -> bool:
    if not torch.distributed.is_initialized():
        return False
    is_distributed_enabled = getattr(vae, "is_distributed_enabled", None)
    if not callable(is_distributed_enabled):
        return False
    try:
        return bool(is_distributed_enabled())
    except Exception:
        return False


def _should_decode_video_on_rank(vae: Any) -> bool:
    return _is_output_rank() or _vae_decode_needs_all_ranks(vae)


# Try to import LTX2VocoderWithBWE (diffusers >= 0.38.0)
try:
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
except ImportError:
    LTX2VocoderWithBWE = None


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    """Factory for the LTX-2.3 post-process function.

    Detects the vocoder output sample rate at factory time and captures it
    in the closure so that the audio_sample_rate flows through
    DiffusionEngine -> OmniRequestOutput -> serving_video.
    """
    output_sr = _detect_vocoder_output_sample_rate(od_config.model)

    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            result: dict[str, Any] = {"video": video, "audio": audio}
            if output_sr is not None:
                result["audio_sample_rate"] = output_sr
            return result
        return output

    return post_process_func


@dataclass
class _LTX23PromptContext:
    batch_size: int
    connector_prompt_embeds: torch.Tensor
    connector_audio_prompt_embeds: torch.Tensor
    connector_attention_mask: torch.Tensor
    positive_connector_prompt_embeds: torch.Tensor
    positive_connector_audio_prompt_embeds: torch.Tensor
    positive_connector_attention_mask: torch.Tensor
    negative_connector_prompt_embeds: torch.Tensor | None
    negative_connector_audio_prompt_embeds: torch.Tensor | None
    negative_connector_attention_mask: torch.Tensor | None


@dataclass
class _LTX23ForwardContext:
    req: DiffusionRequestBatch
    request_inputs: _LTX23RequestInputs
    prompt_context: _LTX23PromptContext
    device: torch.device
    cfg_parallel_ready: bool
    attention_kwargs: dict[str, Any] | None
    latent_num_frames: int
    latent_height: int
    latent_width: int
    latent_mel_bins: int
    original_audio_num_frames: int
    padded_audio_num_frames: int
    timesteps: torch.Tensor
    audio_scheduler: Any
    video_audio_scheduler: Any

    @property
    def batch_size(self) -> int:
        return self.prompt_context.batch_size

    @property
    def num_videos_per_prompt(self) -> int:
        return self.request_inputs.num_videos_per_prompt


@dataclass
class _LTX23DenoiseContext:
    latents: torch.Tensor
    audio_latents: torch.Tensor
    video_coords: torch.Tensor
    audio_coords: torch.Tensor
    conditioning_mask: torch.Tensor | None = None
    conditioning_mask_for_model: torch.Tensor | None = None


class LTX23PipelineBase(
    nn.Module,
    LTX23RequestMixin,
    LTX23SchedulerMixin,
    LTX23GuidanceParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Base class for independent LTX-2.3 pipelines.

    Key differences from LTX2Pipeline:
    - Text encoding: uses ALL 49 hidden states from Gemma-3-12B, flattened
    - Connectors: uses padding_side API (not additive_mask)
    - Vocoder: uses LTX2VocoderWithBWE (48kHz output)
    - Transformer: passes sigma for prompt_adaln
    """

    supports_request_batch = True
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "video_cfg_scale",
            "audio_cfg_scale",
            "video_cfg_guidance_scale",
            "audio_cfg_guidance_scale",
            "video_stg_scale",
            "audio_stg_scale",
            "video_stg_guidance_scale",
            "audio_stg_guidance_scale",
            "video_modality_scale",
            "audio_modality_scale",
            "a2v_guidance_scale",
            "v2a_guidance_scale",
            "video_rescale_scale",
            "audio_rescale_scale",
            "video_stg_blocks",
            "audio_stg_blocks",
        }
    )
    # Audio is diffused jointly with video; warmup must size audio tokens.
    dummy_run_num_frames = 2
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "connectors"]
    _vae_modules: ClassVar[list[str]] = ["vae", "audio_vae"]
    _resident_modules: ClassVar[list[str]] = ["vocoder"]
    _ltx23_recipe: ClassVar[LTX23PipelineRecipe]

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

        # Weight sources for transformer (loaded via AutoWeightsLoader)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # See ``hub_prefetch.py`` for the transformers v5 multi-worker subfolder
        # race; prefetch the whole component set before any from_pretrained.
        ltx2_subfolders = [
            "tokenizer",
            "text_encoder",
            "connectors",
            "vae",
            "audio_vae",
            "vocoder",
            "scheduler",
        ]
        prefetch_subfolders(model, ltx2_subfolders, local_files_only=local_files_only)

        # --- Tokenizer (lightweight, stays wherever) ---
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        # --- Text encoder ---
        with torch.device("cpu"):
            self.text_encoder = from_pretrained_with_prefetch(
                Gemma3ForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=ltx2_subfolders,
                local_files_only=local_files_only,
                torch_dtype=dtype,
            )

        # --- Connectors (LTX-2.3 connectors include caption projection) ---
        self.connectors = from_pretrained_with_prefetch(
            LTX2TextConnectors.from_pretrained,
            model,
            subfolder="connectors",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        install_ltx23_official_connector_processors(self.connectors)

        # --- VAE, Audio VAE ---
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLLTX2Video.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        self.audio_vae = from_pretrained_with_prefetch(
            AutoencoderKLLTX2Audio.from_pretrained,
            model,
            subfolder="audio_vae",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )

        # --- Vocoder: prefer BWE vocoder (48kHz) for LTX-2.3 ---
        vocoder_cls = LTX2VocoderWithBWE or LTX2Vocoder
        try:
            self.vocoder = vocoder_cls.from_pretrained(
                model, subfolder="vocoder", torch_dtype=dtype, local_files_only=local_files_only
            )
        except (TypeError, OSError, ValueError):
            self.vocoder = LTX2Vocoder.from_pretrained(
                model, subfolder="vocoder", torch_dtype=dtype, local_files_only=local_files_only
            )

        # --- Transformer: created empty, weights loaded via AutoWeightsLoader ---
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        quant_config = getattr(self.od_config, "quantization_config", None)
        self.transformer = create_transformer_from_config(transformer_config, quant_config=quant_config)
        self._place_aux_components()

        # --- Scheduler ---
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # --- Derived compression ratios ---
        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio if self.vae is not None else 32
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio if self.vae is not None else 8
        self.audio_vae_mel_compression_ratio = self.audio_vae.mel_compression_ratio if self.audio_vae is not None else 4
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if self.audio_vae is not None else 4
        )
        self.transformer_spatial_patch_size = self.transformer.config.patch_size if self.transformer is not None else 1
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if self.transformer is not None else 1
        )
        self.audio_sampling_rate = self.audio_vae.config.sample_rate if self.audio_vae is not None else 16000
        self.audio_hop_length = self.audio_vae.config.mel_hop_length if self.audio_vae is not None else 160

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

        # Tokenizer max length
        tokenizer_max_length = 1024
        if self.tokenizer is not None:
            tokenizer_max_length = self.tokenizer.model_max_length
            if tokenizer_max_length is None or tokenizer_max_length > 100000:
                encoder_config = getattr(self.text_encoder, "config", None)
                config_max_len = getattr(encoder_config, "max_position_embeddings", None)
                if config_max_len is None:
                    config_max_len = getattr(encoder_config, "max_seq_len", None)
                tokenizer_max_length = config_max_len or 1024
        self.tokenizer_max_length = int(tokenizer_max_length)

        # Pipeline state
        self._guidance_scale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _place_aux_components(self) -> None:
        parallel_config = getattr(self.od_config, "parallel_config", None)
        use_managed_placement = bool(
            getattr(self.od_config, "enable_cpu_offload", False)
            or getattr(self.od_config, "enable_layerwise_offload", False)
            or getattr(parallel_config, "use_hsdp", False)
        )
        if use_managed_placement:
            return

        modules = ModuleDiscovery.discover(self)
        for module in (*modules.encoders, *modules.vaes, *modules.resident_modules):
            module.to(self.device)

    # ------------------------------------------------------------------
    # Text Encoding (LTX-2.3 specific)
    # ------------------------------------------------------------------

    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode prompts using Gemma-3-12B, returning ALL 49 hidden states flattened.

        Stacks all 49 hidden states and flattens to [B, seq, hidden * 49]. The
        connectors unflatten, apply per_token_rms_norm, and project internally
        (same shape contract as LTX-2 since the `diffusers==0.38` connector
        migration; the two differ only in the connector's internal norm path).
        """
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )

        hidden_states = text_encoder_outputs.hidden_states

        # LTX-2.3: Stack ALL 49 hidden states and flatten
        # [49 x (B, seq, 3840)] -> [B, seq, 3840, 49] -> [B, seq, 188160]
        prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=dtype)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_embeds = latent_prep.repeat_prompt_tensor_for_outputs(prompt_embeds, num_videos_per_prompt)
        prompt_attention_mask = latent_prep.repeat_prompt_tensor_for_outputs(
            prompt_attention_mask, num_videos_per_prompt
        )

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        negative_prompt_embeds_provided = negative_prompt_embeds is not None

        if do_classifier_free_guidance and prompt_embeds is None and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if isinstance(negative_prompt, list) and batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            combined_prompt_embeds, combined_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt + negative_prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            prompt_embeds = combined_prompt_embeds[:batch_size]
            negative_prompt_embeds = combined_prompt_embeds[batch_size:]
            prompt_attention_mask = combined_prompt_attention_mask[:batch_size]
            negative_prompt_attention_mask = combined_prompt_attention_mask[batch_size:]
            if num_videos_per_prompt > 1:
                prompt_embeds = latent_prep.repeat_prompt_tensor_for_outputs(
                    prompt_embeds,
                    num_videos_per_prompt,
                )
                prompt_attention_mask = latent_prep.repeat_prompt_tensor_for_outputs(
                    prompt_attention_mask, num_videos_per_prompt
                )
                negative_prompt_embeds = latent_prep.repeat_prompt_tensor_for_outputs(
                    negative_prompt_embeds,
                    num_videos_per_prompt,
                )
                negative_prompt_attention_mask = latent_prep.repeat_prompt_tensor_for_outputs(
                    negative_prompt_attention_mask,
                    num_videos_per_prompt,
                )
        else:
            if prompt_embeds is None:
                prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
            elif num_videos_per_prompt > 1:
                prompt_embeds = latent_prep.repeat_prompt_tensor_for_outputs(
                    prompt_embeds,
                    num_videos_per_prompt,
                )
                prompt_attention_mask = latent_prep.repeat_prompt_tensor_for_outputs(
                    prompt_attention_mask, num_videos_per_prompt
                )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if isinstance(negative_prompt, list) and batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        elif do_classifier_free_guidance and negative_prompt_embeds_provided and num_videos_per_prompt > 1:
            negative_prompt_embeds = latent_prep.repeat_prompt_tensor_for_outputs(
                negative_prompt_embeds, num_videos_per_prompt
            )
            negative_prompt_attention_mask = latent_prep.repeat_prompt_tensor_for_outputs(
                negative_prompt_attention_mask,
                num_videos_per_prompt,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # ------------------------------------------------------------------
    # Latent utilities (shared with LTX2Pipeline)
    # ------------------------------------------------------------------

    _pack_latents = staticmethod(latent_prep.pack_latents)
    _unpack_latents = staticmethod(latent_prep.unpack_latents)
    _normalize_latents = staticmethod(latent_prep.normalize_latents)
    _normalize_audio_latents = staticmethod(latent_prep.normalize_audio_latents)
    _denormalize_latents = staticmethod(latent_prep.denormalize_latents)
    _denormalize_audio_latents = staticmethod(latent_prep.denormalize_audio_latents)
    _pack_audio_latents = staticmethod(latent_prep.pack_audio_latents)
    _unpack_audio_latents = staticmethod(latent_prep.unpack_audio_latents)
    _unpad_audio_latents = staticmethod(latent_prep.unpad_audio_latents)
    _get_sp_padded_audio_latent_length = staticmethod(latent_prep.get_sp_padded_audio_latent_length)

    def _resolve_audio_latent_length(self, audio_latent_length: int, audio_latents: torch.Tensor | None) -> int:
        if audio_latents is None or audio_latents.ndim != 4:
            return audio_latent_length

        provided_latent_length = audio_latents.shape[2]
        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_latent_length = self._get_sp_padded_audio_latent_length(audio_latent_length, int(sp_size))

        # Keep requested duration semantics when callers pass 4D latents that
        # are already padded for SP; other 4D lengths retain shape inference.
        if provided_latent_length in {audio_latent_length, padded_latent_length}:
            return audio_latent_length
        return provided_latent_length

    def _decode_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        output_type: str,
        connector_prompt_embeds: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        prompt_batch_size: int,
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(
                output=(latents, audio_latents),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep_decode, decode_noise_scale_t = latent_prep.prepare_decode_timestep_conditioning(
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                prompt_batch_size=prompt_batch_size,
                effective_batch_size=latents.shape[0],
                device=device,
                dtype=latents.dtype,
            )
            latents = (1 - decode_noise_scale_t) * latents + decode_noise_scale_t * noise

        if _should_decode_video_on_rank(self.vae):
            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep_decode, return_dict=False)[0]
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if not _is_output_rank():
            return DiffusionOutput(
                output=(
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                ),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        if video.numel() > 0:
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        audio_latents = audio_latents.to(self.audio_vae.dtype)
        generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = self.vocoder(generated_mel_spectrograms)

        return DiffusionOutput(
            output=(video, audio),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 5:
                latents = self._normalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
            if latents.ndim != 3:
                raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = noise_scale * noise + (1 - noise_scale) * latents
            return latents.to(device=device, dtype=dtype)

        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        shape = (
            batch_size,
            (num_frames // self.transformer_temporal_patch_size)
            * (height // self.transformer_spatial_patch_size)
            * (width // self.transformer_spatial_patch_size),
            num_channels_latents
            * self.transformer_temporal_patch_size
            * self.transformer_spatial_patch_size
            * self.transformer_spatial_patch_size,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        original_latent_length = audio_latent_length
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_latent_length = self._get_sp_padded_audio_latent_length(original_latent_length, int(sp_size))

        if latents is not None:
            if latents.ndim == 4:
                latents = self._pack_audio_latents(latents)
            if latents.ndim != 3:
                raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")
            latents = self._normalize_audio_latents(latents, self.audio_vae.latents_mean, self.audio_vae.latents_std)
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = noise_scale * noise + (1 - noise_scale) * latents

            if latents.shape[1] not in {original_latent_length, padded_latent_length}:
                raise ValueError(
                    "Provided `audio_latents` has incompatible audio frame count "
                    f"{latents.shape[1]}; expected {original_latent_length} or {padded_latent_length}."
                )

            if latents.shape[1] == original_latent_length and padded_latent_length > original_latent_length:
                padding = torch.zeros(
                    latents.shape[0],
                    padded_latent_length - original_latent_length,
                    latents.shape[2],
                    dtype=latents.dtype,
                    device=latents.device,
                )
                latents = torch.cat([latents, padding], dim=1)

            return latents.to(device=device, dtype=dtype), original_latent_length, padded_latent_length

        shape = (batch_size, padded_latent_length, num_channels_latents * latent_mel_bins)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents, original_latent_length, padded_latent_length

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale != 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 32 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

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
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when "
                    "passed directly, but got: `prompt_attention_mask` "
                    f"{prompt_attention_mask.shape} != `negative_prompt_attention_mask` "
                    f"{negative_prompt_attention_mask.shape}."
                )

    # ------------------------------------------------------------------
    # Cache context
    # ------------------------------------------------------------------

    def _transformer_cache_context(self, context_name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        if callable(cache_context):
            return cache_context(context_name)
        return nullcontext()

    def _prepare_prompt_context(
        self,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> _LTX23PromptContext:
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
        )

        negative_connector_prompt_embeds = None
        negative_connector_audio_prompt_embeds = None
        negative_connector_attention_mask = None
        if self.do_classifier_free_guidance:
            # Official LTX-2.3 processes positive and negative prompt
            # connectors independently. Batched connector attention changes
            # numerics enough to diverge from the reference trajectory.
            (
                positive_connector_prompt_embeds,
                positive_connector_audio_prompt_embeds,
                positive_connector_attention_mask,
            ) = self.connectors(
                prompt_embeds,
                prompt_attention_mask,
                padding_side=getattr(self.tokenizer, "padding_side", "left"),
            )
            (
                negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask,
            ) = self.connectors(
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                padding_side=getattr(self.tokenizer, "padding_side", "left"),
            )
            connector_prompt_embeds = torch.cat(
                [negative_connector_prompt_embeds, positive_connector_prompt_embeds], dim=0
            )
            connector_audio_prompt_embeds = torch.cat(
                [negative_connector_audio_prompt_embeds, positive_connector_audio_prompt_embeds], dim=0
            )
            connector_attention_mask = torch.cat(
                [negative_connector_attention_mask, positive_connector_attention_mask], dim=0
            )
        else:
            (
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                connector_attention_mask,
            ) = self.connectors(
                prompt_embeds,
                prompt_attention_mask,
                padding_side=getattr(self.tokenizer, "padding_side", "left"),
            )
            positive_connector_prompt_embeds = connector_prompt_embeds
            positive_connector_audio_prompt_embeds = connector_audio_prompt_embeds
            positive_connector_attention_mask = connector_attention_mask

        return _LTX23PromptContext(
            batch_size=batch_size,
            connector_prompt_embeds=connector_prompt_embeds,
            connector_audio_prompt_embeds=connector_audio_prompt_embeds,
            connector_attention_mask=connector_attention_mask,
            positive_connector_prompt_embeds=positive_connector_prompt_embeds,
            positive_connector_audio_prompt_embeds=positive_connector_audio_prompt_embeds,
            positive_connector_attention_mask=positive_connector_attention_mask,
            negative_connector_prompt_embeds=negative_connector_prompt_embeds,
            negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds,
            negative_connector_attention_mask=negative_connector_attention_mask,
        )

    def _setup_forward_runtime(
        self,
        request_inputs: _LTX23RequestInputs,
        attention_kwargs: dict[str, Any] | None,
    ) -> bool:
        self._guidance_scale = 2.0 if request_inputs.guidance_params.do_cfg else 1.0
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None
        return self._setup_cfg_parallel_runtime(request_inputs.guidance_params)

    def _check_forward_inputs(
        self,
        request_inputs: _LTX23RequestInputs,
        image: Any | None = None,
    ) -> None:
        self.check_inputs(
            prompt=request_inputs.prompt,
            height=request_inputs.height,
            width=request_inputs.width,
            prompt_embeds=request_inputs.prompt_embeds,
            negative_prompt_embeds=request_inputs.negative_prompt_embeds,
            prompt_attention_mask=request_inputs.prompt_attention_mask,
            negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
        )

    def _resolve_video_latent_dimensions(
        self,
        request_inputs: _LTX23RequestInputs,
    ) -> tuple[int, int, int]:
        latent_num_frames = (request_inputs.num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = request_inputs.height // self.vae_spatial_compression_ratio
        latent_width = request_inputs.width // self.vae_spatial_compression_ratio
        if request_inputs.latents is not None and request_inputs.latents.ndim == 5:
            _, _, latent_num_frames, latent_height, latent_width = request_inputs.latents.shape
        return latent_num_frames, latent_height, latent_width

    def _prepare_video_latents_stage(
        self,
        request_inputs: _LTX23RequestInputs,
        prompt_context: _LTX23PromptContext,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            num_channels_latents,
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            noise_scale,
            prompt_context.connector_prompt_embeds.dtype,
            device,
            request_inputs.generator,
            request_inputs.latents,
        )
        return latents, None

    def _prepare_audio_latents_stage(
        self,
        request_inputs: _LTX23RequestInputs,
        prompt_context: _LTX23PromptContext,
        *,
        device: torch.device,
        noise_scale: float,
    ) -> tuple[torch.Tensor, int, int, int]:
        duration_s = request_inputs.num_frames / request_inputs.frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        audio_num_frames = self._resolve_audio_latent_length(audio_num_frames, request_inputs.audio_latents)

        num_mel_bins = self.audio_vae.config.mel_bins if self.audio_vae is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = self.audio_vae.config.latent_channels if self.audio_vae is not None else 8
        audio_latents, original_audio_num_frames, padded_audio_num_frames = self.prepare_audio_latents(
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=prompt_context.connector_audio_prompt_embeds.dtype,
            device=device,
            generator=request_inputs.generator,
            latents=request_inputs.audio_latents,
        )
        return audio_latents, original_audio_num_frames, padded_audio_num_frames, latent_mel_bins

    def _prepare_rope_coords_stage(
        self,
        forward_ctx: _LTX23ForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_inputs = forward_ctx.request_inputs
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0],
            forward_ctx.latent_num_frames,
            forward_ctx.latent_height,
            forward_ctx.latent_width,
            latents.device,
            fps=request_inputs.frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            forward_ctx.padded_audio_num_frames,
            audio_latents.device,
        )
        return video_coords, audio_coords

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
    ) -> _LTX23DenoiseContext:
        return denoise_ctx

    @staticmethod
    def _expand_step_to_tokens(ts: torch.Tensor, token_count: int) -> torch.Tensor:
        return ts.reshape(-1, 1).expand(-1, token_count)

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "timestep": self._expand_step_to_tokens(ts, video_token_count),
            "audio_timestep": self._expand_step_to_tokens(ts, audio_token_count),
            "sigma": ts,
            "audio_sigma": ts,
        }

    @staticmethod
    def _denoise_pass_count(guidance_params: guidance_ops._LTX23GuidanceParams) -> int:
        return guidance_ops.denoise_pass_count(guidance_params)

    def _build_transformer_kwargs(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        ts: torch.Tensor,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_inputs = forward_ctx.request_inputs
        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            **self._denoise_timestep_kwargs(
                ts,
                forward_ctx,
                denoise_ctx,
                video_token_count=hidden_states.shape[1],
                audio_token_count=audio_hidden_states.shape[1],
            ),
            "num_frames": forward_ctx.latent_num_frames,
            "height": forward_ctx.latent_height,
            "width": forward_ctx.latent_width,
            "fps": request_inputs.frame_rate,
            "audio_num_frames": forward_ctx.padded_audio_num_frames,
            "video_coords": denoise_ctx.video_coords if video_coords is None else video_coords,
            "audio_coords": denoise_ctx.audio_coords if audio_coords is None else audio_coords,
            "attention_kwargs": forward_ctx.attention_kwargs if attention_kwargs is None else attention_kwargs,
            "return_dict": False,
        }

    _x0_from_noise = staticmethod(guidance_ops.x0_from_noise)
    _noise_from_x0 = staticmethod(guidance_ops.noise_from_x0)
    _euler_step_from_velocity = staticmethod(guidance_ops.euler_step_from_velocity)
    _combine_guided_x0 = staticmethod(guidance_ops.combine_guided_x0)
    _repeat_batch_tensor = staticmethod(guidance_ops.repeat_batch_tensor)
    _build_ltx23_perturbation_kwargs = staticmethod(guidance_ops.build_ltx23_perturbation_kwargs)

    def _predict_guided_noise(
        self,
        *,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
        t: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_inputs = forward_ctx.request_inputs
        prompt_context = forward_ctx.prompt_context
        guidance_params = request_inputs.guidance_params

        pass_names = ["cond"]
        video_contexts = [prompt_context.positive_connector_prompt_embeds]
        audio_contexts = [prompt_context.positive_connector_audio_prompt_embeds]

        if guidance_params.do_cfg:
            if (
                prompt_context.negative_connector_prompt_embeds is None
                or prompt_context.negative_connector_audio_prompt_embeds is None
                or prompt_context.negative_connector_attention_mask is None
            ):
                raise ValueError("Negative prompt context is required when LTX2.3 CFG guidance is enabled.")
            pass_names.append("uncond")
            video_contexts.append(prompt_context.negative_connector_prompt_embeds)
            audio_contexts.append(prompt_context.negative_connector_audio_prompt_embeds)

        if guidance_params.do_stg:
            pass_names.append("ptb")
            video_contexts.append(prompt_context.positive_connector_prompt_embeds)
            audio_contexts.append(prompt_context.positive_connector_audio_prompt_embeds)

        if guidance_params.do_modality_guidance:
            pass_names.append("mod")
            video_contexts.append(prompt_context.positive_connector_prompt_embeds)
            audio_contexts.append(prompt_context.positive_connector_audio_prompt_embeds)

        pass_count = len(pass_names)
        latent_model_input = self._repeat_batch_tensor(denoise_ctx.latents, pass_count).to(
            prompt_context.connector_prompt_embeds.dtype
        )
        audio_latent_model_input = self._repeat_batch_tensor(denoise_ctx.audio_latents, pass_count).to(
            prompt_context.connector_prompt_embeds.dtype
        )
        video_coords = self._repeat_batch_tensor(denoise_ctx.video_coords, pass_count)
        audio_coords = self._repeat_batch_tensor(denoise_ctx.audio_coords, pass_count)
        encoder_hidden_states = torch.cat(video_contexts, dim=0)
        audio_encoder_hidden_states = torch.cat(audio_contexts, dim=0)
        ts = t.expand(latent_model_input.shape[0])

        attention_kwargs = dict(forward_ctx.attention_kwargs or {})
        perturbation_kwargs = self._build_ltx23_perturbation_kwargs(
            pass_names=pass_names,
            batch_size=denoise_ctx.latents.shape[0],
            guidance_params=guidance_params,
            device=latent_model_input.device,
            dtype=latent_model_input.dtype,
        )
        if perturbation_kwargs:
            attention_kwargs["ltx23_perturbation_kwargs"] = perturbation_kwargs

        transformer_kwargs = self._build_transformer_kwargs(
            forward_ctx,
            denoise_ctx,
            hidden_states=latent_model_input,
            audio_hidden_states=audio_latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            ts=ts,
            video_coords=video_coords,
            audio_coords=audio_coords,
            attention_kwargs=attention_kwargs,
        )
        with self._transformer_cache_context("guided"):
            noise_pred_video, noise_pred_audio = self.transformer(**transformer_kwargs)

        video_splits = dict(zip(pass_names, noise_pred_video.float().chunk(pass_count), strict=True))
        audio_splits = dict(zip(pass_names, noise_pred_audio.float().chunk(pass_count), strict=True))
        video_sigma = self.scheduler.sigmas[step_index]
        audio_sigma = forward_ctx.audio_scheduler.sigmas[step_index]

        cond_video_x0 = self._x0_from_noise(denoise_ctx.latents, video_splits["cond"], video_sigma)
        cond_audio_x0 = self._x0_from_noise(denoise_ctx.audio_latents, audio_splits["cond"], audio_sigma)
        uncond_video_x0 = (
            self._x0_from_noise(denoise_ctx.latents, video_splits["uncond"], video_sigma)
            if "uncond" in video_splits
            else 0.0
        )
        uncond_audio_x0 = (
            self._x0_from_noise(denoise_ctx.audio_latents, audio_splits["uncond"], audio_sigma)
            if "uncond" in audio_splits
            else 0.0
        )
        ptb_video_x0 = (
            self._x0_from_noise(denoise_ctx.latents, video_splits["ptb"], video_sigma) if "ptb" in video_splits else 0.0
        )
        ptb_audio_x0 = (
            self._x0_from_noise(denoise_ctx.audio_latents, audio_splits["ptb"], audio_sigma)
            if "ptb" in audio_splits
            else 0.0
        )
        mod_video_x0 = (
            self._x0_from_noise(denoise_ctx.latents, video_splits["mod"], video_sigma) if "mod" in video_splits else 0.0
        )
        mod_audio_x0 = (
            self._x0_from_noise(denoise_ctx.audio_latents, audio_splits["mod"], audio_sigma)
            if "mod" in audio_splits
            else 0.0
        )

        guided_video_x0 = self._combine_guided_x0(
            cond=cond_video_x0,
            uncond_text=uncond_video_x0,
            uncond_perturbed=ptb_video_x0,
            uncond_modality=mod_video_x0,
            cfg_scale=guidance_params.video_cfg_scale,
            stg_scale=guidance_params.video_stg_scale,
            modality_scale=guidance_params.video_modality_scale,
            rescale_scale=guidance_params.video_rescale_scale,
        )
        guided_audio_x0 = self._combine_guided_x0(
            cond=cond_audio_x0,
            uncond_text=uncond_audio_x0,
            uncond_perturbed=ptb_audio_x0,
            uncond_modality=mod_audio_x0,
            cfg_scale=guidance_params.audio_cfg_scale,
            stg_scale=guidance_params.audio_stg_scale,
            modality_scale=guidance_params.audio_modality_scale,
            rescale_scale=guidance_params.audio_rescale_scale,
        )

        return (
            self._noise_from_x0(denoise_ctx.latents, guided_video_x0, video_sigma),
            self._noise_from_x0(denoise_ctx.audio_latents, guided_audio_x0, audio_sigma),
        )

    def _step_denoised_latents(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
        noise_pred_video: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        t: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forward_ctx.cfg_parallel_ready:
            latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                (noise_pred_video, noise_pred_audio),
                (t, t),
                (denoise_ctx.latents, denoise_ctx.audio_latents),
                do_true_cfg=self.do_classifier_free_guidance,
                per_request_scheduler=forward_ctx.video_audio_scheduler,
            )
            return self._synchronize_cfg_parallel_step_output(
                (latents, audio_latents),
                do_true_cfg=self.do_classifier_free_guidance,
            )

        latents = self._euler_step_from_velocity(
            denoise_ctx.latents,
            noise_pred_video,
            self.scheduler.sigmas,
            step_index,
        )
        audio_latents = self._euler_step_from_velocity(
            denoise_ctx.audio_latents,
            noise_pred_audio,
            forward_ctx.audio_scheduler.sigmas,
            step_index,
        )
        return latents, audio_latents

    def _denoise_loop(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_inputs = forward_ctx.request_inputs
        prompt_context = forward_ctx.prompt_context
        guidance_params = request_inputs.guidance_params
        audio_scheduler = forward_ctx.audio_scheduler

        with self.progress_bar(total=len(forward_ctx.timesteps)) as pbar:
            for i, t in enumerate(forward_ctx.timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if forward_ctx.cfg_parallel_ready:
                    latent_model_input = denoise_ctx.latents.to(prompt_context.positive_connector_prompt_embeds.dtype)
                    audio_latent_model_input = denoise_ctx.audio_latents.to(
                        prompt_context.positive_connector_prompt_embeds.dtype
                    )
                    ts = t.expand(latent_model_input.shape[0])
                    positive_kwargs = self._build_transformer_kwargs(
                        forward_ctx,
                        denoise_ctx,
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latent_model_input,
                        encoder_hidden_states=prompt_context.positive_connector_prompt_embeds,
                        audio_encoder_hidden_states=prompt_context.positive_connector_audio_prompt_embeds,
                        ts=ts,
                    )
                    negative_kwargs = {
                        **positive_kwargs,
                        "encoder_hidden_states": prompt_context.negative_connector_prompt_embeds,
                        "audio_encoder_hidden_states": prompt_context.negative_connector_audio_prompt_embeds,
                    }
                    noise_pred_video, noise_pred_audio = self.predict_noise_with_parallel_cfg(
                        true_cfg_scale=guidance_params.video_cfg_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                        video_latents=denoise_ctx.latents,
                        audio_latents=denoise_ctx.audio_latents,
                        video_sigma=self.scheduler.sigmas[i],
                        audio_sigma=audio_scheduler.sigmas[i],
                    )
                else:
                    noise_pred_video, noise_pred_audio = self._predict_guided_noise(
                        forward_ctx=forward_ctx,
                        denoise_ctx=denoise_ctx,
                        t=t,
                        step_index=i,
                    )

                denoise_ctx.latents, denoise_ctx.audio_latents = self._step_denoised_latents(
                    forward_ctx,
                    denoise_ctx,
                    noise_pred_video,
                    noise_pred_audio,
                    t,
                    i,
                )
                pbar.update()

        return denoise_ctx.latents, denoise_ctx.audio_latents

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: _LTX23ForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self._unpack_latents(
            latents,
            forward_ctx.latent_num_frames,
            forward_ctx.latent_height,
            forward_ctx.latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        audio_latents = self._unpad_audio_latents(audio_latents, forward_ctx.original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents,
            forward_ctx.original_audio_num_frames,
            num_mel_bins=forward_ctx.latent_mel_bins,
        )
        return latents, audio_latents

    def _decode_and_split(
        self,
        forward_ctx: _LTX23ForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> list[DiffusionOutput]:
        request_inputs = forward_ctx.request_inputs
        return split_diffusion_output_by_request(
            self._decode_output(
                latents=latents,
                audio_latents=audio_latents,
                output_type=request_inputs.output_type,
                connector_prompt_embeds=forward_ctx.prompt_context.connector_prompt_embeds,
                generator=request_inputs.generator,
                device=forward_ctx.device,
                decode_timestep=request_inputs.decode_timestep,
                decode_noise_scale=request_inputs.decode_noise_scale,
                prompt_batch_size=forward_ctx.batch_size,
            ),
            forward_ctx.req,
            num_outputs_per_prompt=forward_ctx.num_videos_per_prompt,
        )

    def _forward_impl(
        self,
        req: DiffusionRequestBatch,
        request_inputs: _LTX23RequestInputs,
        *,
        noise_scale: float,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        image: Any | None = None,
    ) -> list[DiffusionOutput]:
        self._check_forward_inputs(request_inputs, image=image)
        cfg_parallel_ready = self._setup_forward_runtime(request_inputs, attention_kwargs)
        device = self.device
        prompt_context = self._prepare_prompt_context(
            prompt=request_inputs.prompt,
            negative_prompt=request_inputs.negative_prompt,
            prompt_embeds=request_inputs.prompt_embeds,
            negative_prompt_embeds=request_inputs.negative_prompt_embeds,
            prompt_attention_mask=request_inputs.prompt_attention_mask,
            negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
            num_videos_per_prompt=request_inputs.num_videos_per_prompt,
            max_sequence_length=request_inputs.max_sequence_length,
        )

        latent_num_frames, latent_height, latent_width = self._resolve_video_latent_dimensions(request_inputs)
        latents, conditioning_mask = self._prepare_video_latents_stage(
            request_inputs,
            prompt_context,
            device=device,
            noise_scale=noise_scale,
            image=image,
        )
        audio_latents, original_audio_num_frames, padded_audio_num_frames, latent_mel_bins = (
            self._prepare_audio_latents_stage(
                request_inputs,
                prompt_context,
                device=device,
                noise_scale=noise_scale,
            )
        )
        audio_scheduler, video_audio_scheduler, timesteps_tensor = self._prepare_scheduler_stage(
            request_inputs,
            device=device,
            sigmas=sigmas,
            timesteps=timesteps,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        forward_ctx = _LTX23ForwardContext(
            req=req,
            request_inputs=request_inputs,
            prompt_context=prompt_context,
            device=device,
            cfg_parallel_ready=cfg_parallel_ready,
            attention_kwargs=attention_kwargs,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_mel_bins=latent_mel_bins,
            original_audio_num_frames=original_audio_num_frames,
            padded_audio_num_frames=padded_audio_num_frames,
            timesteps=timesteps_tensor,
            audio_scheduler=audio_scheduler,
            video_audio_scheduler=video_audio_scheduler,
        )
        video_coords, audio_coords = self._prepare_rope_coords_stage(forward_ctx, latents, audio_latents)
        denoise_ctx = _LTX23DenoiseContext(
            latents=latents,
            audio_latents=audio_latents,
            video_coords=video_coords,
            audio_coords=audio_coords,
            conditioning_mask=conditioning_mask,
        )
        denoise_ctx = self._prepare_denoise_context_for_cfg(forward_ctx, denoise_ctx)
        latents, audio_latents = self._denoise_loop(forward_ctx, denoise_ctx)
        latents, audio_latents = self._unpack_and_denormalize_stage(forward_ctx, latents, audio_latents)
        return self._decode_and_split(forward_ctx, latents, audio_latents)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] | None = None,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> list[DiffusionOutput]:
        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )
        return self._forward_impl(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_ltx23_weights(self, weights)
