# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Fully independent LTX-2.3 pipeline for vLLM-Omni.

This pipeline does NOT inherit from LTX2Pipeline because:
- LTX-2.3 connectors run per_token_rms_norm + per-modality video/audio
  projection internally (per_modality_projections=True),
  versus LTX-2's per_layer_masked_mean_norm + shared projection path
- LTX-2.3 uses a BWE vocoder outputting 48kHz audio (not 16kHz)
- LTX-2.3 transformer requires the sigma parameter for prompt modulation
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import torch
from diffusers import AutoencoderKLLTX2Audio, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import DistributedAutoencoderKLLTX2Video
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_components import (
    LTX23_COMPONENT_PROFILE,
    create_transformer_from_config,
    load_transformer_config,
)
from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
from .ltx2_latents import LTXAVState
from .ltx2_pipeline_base import (
    LTXPipelineBase,
    LTXPromptContext,
    LTXRequestInputs,
)
from .ltx2_recipes import LTX23_ONE_STAGE_RECIPE

logger = init_logger(__name__)


def _get_audio_latents_from_sampling(sampling: Any) -> torch.Tensor | None:
    if sampling.audio_latents is not None:
        return sampling.audio_latents
    return sampling.extra_args.get("audio_latents")


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


def _repeat_prompt_tensor_for_outputs(tensor: torch.Tensor, num_videos_per_prompt: int) -> torch.Tensor:
    if num_videos_per_prompt == 1:
        return tensor
    return tensor.repeat_interleave(num_videos_per_prompt, dim=0)


def _detect_vocoder_output_sample_rate(model: str) -> int | None:
    """Detect the vocoder output sample rate from vocoder/config.json.

    This runs at factory time (engine process) so the rate is captured in
    the post-process closure and doesn't need cross-process communication.

    Returns:
        Output sample rate (e.g. 48000 for LTX-2.3 BWE vocoder) or None.
    """
    vocoder_config_path = os.path.join(model, "vocoder", "config.json")
    if not os.path.exists(vocoder_config_path):
        try:
            vocoder_config_path = hf_hub_download(model, "vocoder/config.json")
        except Exception:
            return None
    try:
        with open(vocoder_config_path) as f:
            cfg = json.load(f)
        return cfg.get("output_sampling_rate")
    except Exception:
        return None


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


def _expand_per_prompt_decode_value(
    value: float | list[float],
    *,
    prompt_batch_size: int,
    effective_batch_size: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list):
        return [value] * effective_batch_size
    if len(value) == 1:
        return value * effective_batch_size
    if len(value) == effective_batch_size:
        return value
    if prompt_batch_size > 0 and len(value) == prompt_batch_size and effective_batch_size % prompt_batch_size == 0:
        repeats = effective_batch_size // prompt_batch_size
        return [item for item in value for _ in range(repeats)]
    raise ValueError(
        f"`{field_name}` must have length 1, prompt batch size ({prompt_batch_size}), or effective batch size"
        f" ({effective_batch_size}); got {len(value)}."
    )


def _prepare_decode_timestep_conditioning(
    *,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    prompt_batch_size: int,
    effective_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_timestep_values = _expand_per_prompt_decode_value(
        decode_timestep,
        prompt_batch_size=prompt_batch_size,
        effective_batch_size=effective_batch_size,
        field_name="decode_timestep",
    )
    if decode_noise_scale is None:
        decode_noise_scale_values = decode_timestep_values
    else:
        decode_noise_scale_values = _expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )


class LTX23Pipeline(LTXPipelineBase, DiffusionPipelineProfilerMixin):
    """Fully independent LTX-2.3 pipeline.

    Key differences from LTX2Pipeline:
    - Text encoding: uses ALL 49 hidden states from Gemma-3-12B, flattened
    - Connectors: uses padding_side API (not additive_mask)
    - Vocoder: uses LTX2VocoderWithBWE (48kHz output)
    - Transformer: passes sigma for prompt_adaln
    """

    supports_request_batch = True
    # Audio is diffused jointly with video; warmup must size audio tokens.
    dummy_run_num_frames = 2
    component_profile = LTX23_COMPONENT_PROFILE
    one_stage_recipe = LTX23_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)

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
        prompt_embeds = _repeat_prompt_tensor_for_outputs(prompt_embeds, num_videos_per_prompt)
        prompt_attention_mask = _repeat_prompt_tensor_for_outputs(prompt_attention_mask, num_videos_per_prompt)

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

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        elif num_videos_per_prompt > 1:
            prompt_embeds = _repeat_prompt_tensor_for_outputs(prompt_embeds, num_videos_per_prompt)
            prompt_attention_mask = _repeat_prompt_tensor_for_outputs(prompt_attention_mask, num_videos_per_prompt)

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
            negative_prompt_embeds = _repeat_prompt_tensor_for_outputs(negative_prompt_embeds, num_videos_per_prompt)
            negative_prompt_attention_mask = _repeat_prompt_tensor_for_outputs(
                negative_prompt_attention_mask,
                num_videos_per_prompt,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

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
            timestep_decode, decode_noise_scale_t = _prepare_decode_timestep_conditioning(
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

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

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
    # CFG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_x0_space_cfg(
        sample: torch.Tensor,
        positive_noise_pred: torch.Tensor,
        negative_noise_pred: torch.Tensor,
        sigma: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        x0_cond = sample - positive_noise_pred * sigma
        x0_uncond = sample - negative_noise_pred * sigma
        x0_guided = x0_cond + (guidance_scale - 1) * (x0_cond - x0_uncond)
        return (sample - x0_guided) / sigma

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale,
        cfg_normalize=False,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        video_sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ):
        if video_latents is None or audio_latents is None or video_sigma is None or audio_sigma is None:
            raise ValueError("LTX23Pipeline applies CFG in x0-space and requires video/audio latents and sigmas.")

        video_pos, audio_pos = positive_noise_pred
        video_neg, audio_neg = negative_noise_pred
        video_combined = self._combine_x0_space_cfg(
            video_latents,
            video_pos,
            video_neg,
            video_sigma,
            true_cfg_scale,
        )
        audio_combined = self._combine_x0_space_cfg(
            audio_latents,
            audio_pos,
            audio_neg,
            audio_sigma,
            true_cfg_scale,
        )
        if cfg_normalize:
            video_combined = self.cfg_normalize_function(video_pos, video_combined)
            audio_combined = self.cfg_normalize_function(audio_pos, audio_combined)
        return video_combined, audio_combined

    def predict_noise_with_parallel_cfg(
        self,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any],
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        video_sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def maybe_slice(pred: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if output_slice is None:
                return pred
            return pred[0][:, :output_slice], pred[1][:, :output_slice]

        cfg_world_size = get_classifier_free_guidance_world_size()
        if cfg_world_size != 2:
            raise ValueError(f"LTX23Pipeline parallel CFG requires cfg_parallel_size 2, but got {cfg_world_size}.")

        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        branch_kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
        local_video_pred, local_audio_pred = maybe_slice(self.predict_noise(**branch_kwargs))

        gathered_video = cfg_group.all_gather(local_video_pred, separate_tensors=True)
        gathered_audio = cfg_group.all_gather(local_audio_pred, separate_tensors=True)
        positive_noise_pred = (gathered_video[0], gathered_audio[0])
        negative_noise_pred = (gathered_video[1], gathered_audio[1])

        return self.combine_cfg_noise(
            positive_noise_pred,
            negative_noise_pred,
            true_cfg_scale,
            cfg_normalize,
            video_latents=video_latents,
            audio_latents=audio_latents,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

    def _resolve_request_inputs(
        self,
        req: DiffusionRequestBatch,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        frame_rate: float | None,
        num_inference_steps: int | None,
        timesteps: list[int] | None,
        guidance_scale: float,
        num_videos_per_prompt: int | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        audio_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        output_type: str,
        max_sequence_length: int | None,
    ) -> LTXRequestInputs:
        sampling_params_list = req.sampling_params_list
        common_sampling_params = sampling_params_list[0]
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = common_sampling_params.height or height or self.one_stage_recipe.height
        width = common_sampling_params.width or width or self.one_stage_recipe.width
        num_frames = common_sampling_params.num_frames or num_frames or self.one_stage_recipe.num_frames
        frame_rate = common_sampling_params.resolved_frame_rate or frame_rate or self.one_stage_recipe.frame_rate
        num_inference_steps = (
            common_sampling_params.num_inference_steps
            or num_inference_steps
            or self.one_stage_recipe.num_inference_steps
        )
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchEulerDiscreteScheduler.")

        num_videos_per_prompt = (
            common_sampling_params.num_outputs_per_prompt
            if common_sampling_params.num_outputs_per_prompt > 0
            else num_videos_per_prompt or 1
        )
        max_sequence_length = (
            common_sampling_params.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        )

        if common_sampling_params.guidance_scale_provided:
            guidance_scale = common_sampling_params.guidance_scale

        if generator is None:
            generator = req.collate_request_generators(num_videos_per_prompt, generator)

        latents = req.collate_request_tensors("latents", latents)
        audio_latents = DiffusionRequestBatch.collate_tensors(
            [_get_audio_latents_from_sampling(sampling) for sampling in sampling_params_list],
            "audio_latents",
            audio_latents,
        )

        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            req.prompts,
            {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_attention_mask": negative_prompt_attention_mask,
            },
            field_aliases={
                "prompt_attention_mask": ("prompt_attention_mask", "attention_mask"),
                "negative_prompt_attention_mask": (
                    "negative_prompt_attention_mask",
                    "negative_attention_mask",
                ),
            },
        )
        prompt_embeds = prompt_fields["prompt_embeds"]
        negative_prompt_embeds = prompt_fields["negative_prompt_embeds"]
        prompt_attention_mask = prompt_fields["prompt_attention_mask"]
        negative_prompt_attention_mask = prompt_fields["negative_prompt_attention_mask"]
        if prompt_embeds is not None:
            prompt = None
        if negative_prompt_embeds is not None:
            negative_prompt = None

        if common_sampling_params.decode_timestep is not None:
            decode_timestep = common_sampling_params.decode_timestep
        if common_sampling_params.decode_noise_scale is not None:
            decode_noise_scale = common_sampling_params.decode_noise_scale
        if common_sampling_params.output_type is not None:
            output_type = common_sampling_params.output_type

        return LTXRequestInputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            guidance_rescale=0.0,
            num_videos_per_prompt=int(num_videos_per_prompt),
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
            max_sequence_length=int(max_sequence_length),
        )

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
    ) -> LTXPromptContext:
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

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds,
            prompt_attention_mask,
            padding_side=getattr(self.tokenizer, "padding_side", "left"),
        )

        positive_connector_prompt_embeds = connector_prompt_embeds
        positive_connector_audio_prompt_embeds = connector_audio_prompt_embeds
        positive_connector_attention_mask = connector_attention_mask
        negative_connector_prompt_embeds = None
        negative_connector_audio_prompt_embeds = None
        negative_connector_attention_mask = None
        if self.do_classifier_free_guidance:
            split_batch = batch_size * num_videos_per_prompt
            negative_connector_prompt_embeds = connector_prompt_embeds[:split_batch]
            positive_connector_prompt_embeds = connector_prompt_embeds[split_batch:]
            negative_connector_audio_prompt_embeds = connector_audio_prompt_embeds[:split_batch]
            positive_connector_audio_prompt_embeds = connector_audio_prompt_embeds[split_batch:]
            negative_connector_attention_mask = connector_attention_mask[:split_batch]
            positive_connector_attention_mask = connector_attention_mask[split_batch:]

        return LTXPromptContext(
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
        request_inputs: LTXRequestInputs,
        attention_kwargs: dict[str, Any] | None,
    ) -> bool:
        self._guidance_scale = request_inputs.guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None
        cfg_world_size = get_classifier_free_guidance_world_size()
        if self.do_classifier_free_guidance and cfg_world_size not in (1, 2):
            raise ValueError(
                f"LTX23Pipeline supports CFG parallelism with cfg_parallel_size 1 or 2, but got {cfg_world_size}."
            )
        return self.do_classifier_free_guidance and cfg_world_size > 1

    def _check_forward_inputs(
        self,
        request_inputs: LTXRequestInputs,
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

    def _scheduler_shift_sequence_length(
        self,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> int:
        # Diffusers LTX2Pipeline hardcodes max_image_seq_len for this path.
        return self.scheduler.config.get("max_image_seq_len", 4096)

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        if self.do_classifier_free_guidance and not forward_ctx.cfg_parallel_ready:
            denoise_ctx.video_coords = denoise_ctx.video_coords.repeat(
                (2,) + (1,) * (denoise_ctx.video_coords.ndim - 1)
            )
            denoise_ctx.audio_coords = denoise_ctx.audio_coords.repeat(
                (2,) + (1,) * (denoise_ctx.audio_coords.ndim - 1)
            )
        return denoise_ctx

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        return {"timestep": ts, "sigma": ts}

    def _step_denoised_latents(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
        noise_pred_video: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        t: torch.Tensor,
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

        latents = self.scheduler.step(noise_pred_video, t, denoise_ctx.latents, return_dict=False)[0]
        audio_latents = forward_ctx.audio_scheduler.step(
            noise_pred_audio,
            t,
            denoise_ctx.audio_latents,
            return_dict=False,
        )[0]
        return latents, audio_latents

    def _denoise_step(
        self,
        i: int,
        t: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXAVState:
        request_inputs = forward_ctx.request_inputs
        prompt_context = forward_ctx.prompt_context
        guidance_scale = request_inputs.guidance_scale
        audio_scheduler = forward_ctx.audio_scheduler
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        if forward_ctx.cfg_parallel_ready:
            latent_model_input = state.video.to(prompt_context.positive_connector_prompt_embeds.dtype)
            audio_latent_model_input = state.audio.to(prompt_context.positive_connector_prompt_embeds.dtype)
            ts = t.expand(latent_model_input.shape[0])
            positive_kwargs = self._build_transformer_kwargs(
                forward_ctx,
                denoise_ctx,
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=prompt_context.positive_connector_prompt_embeds,
                audio_encoder_hidden_states=prompt_context.positive_connector_audio_prompt_embeds,
                encoder_attention_mask=prompt_context.positive_connector_attention_mask,
                audio_encoder_attention_mask=prompt_context.positive_connector_attention_mask,
                ts=ts,
            )
            negative_kwargs = {
                **positive_kwargs,
                "encoder_hidden_states": prompt_context.negative_connector_prompt_embeds,
                "audio_encoder_hidden_states": prompt_context.negative_connector_audio_prompt_embeds,
                "encoder_attention_mask": prompt_context.negative_connector_attention_mask,
                "audio_encoder_attention_mask": prompt_context.negative_connector_attention_mask,
            }
            noise_pred_video, noise_pred_audio = self.predict_noise_with_parallel_cfg(
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
                video_latents=state.video,
                audio_latents=state.audio,
                video_sigma=self.scheduler.sigmas[i],
                audio_sigma=audio_scheduler.sigmas[i],
            )
        else:
            latent_model_input = torch.cat([state.video] * 2) if self.do_classifier_free_guidance else state.video
            latent_model_input = latent_model_input.to(prompt_context.connector_prompt_embeds.dtype)
            audio_latent_model_input = torch.cat([state.audio] * 2) if self.do_classifier_free_guidance else state.audio
            audio_latent_model_input = audio_latent_model_input.to(prompt_context.connector_prompt_embeds.dtype)
            ts = t.expand(latent_model_input.shape[0])

            transformer_kwargs = self._build_transformer_kwargs(
                forward_ctx,
                denoise_ctx,
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=prompt_context.connector_prompt_embeds,
                audio_encoder_hidden_states=prompt_context.connector_audio_prompt_embeds,
                encoder_attention_mask=prompt_context.connector_attention_mask,
                audio_encoder_attention_mask=prompt_context.connector_attention_mask,
                ts=ts,
            )
            with self._transformer_cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = self.transformer(**transformer_kwargs)

            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            if self.do_classifier_free_guidance:
                noise_pred_video_uncond, noise_pred_video_cond = noise_pred_video.chunk(2)
                noise_pred_video = self._combine_x0_space_cfg(
                    state.video,
                    noise_pred_video_cond,
                    noise_pred_video_uncond,
                    self.scheduler.sigmas[i],
                    guidance_scale,
                )

                noise_pred_audio_uncond, noise_pred_audio_cond = noise_pred_audio.chunk(2)
                noise_pred_audio = self._combine_x0_space_cfg(
                    state.audio,
                    noise_pred_audio_cond,
                    noise_pred_audio_uncond,
                    audio_scheduler.sigmas[i],
                    guidance_scale,
                )

        video, audio = self._step_denoised_latents(
            forward_ctx,
            denoise_ctx,
            noise_pred_video,
            noise_pred_audio,
            t,
        )
        return LTXAVState(video=video, audio=audio)

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
        guidance_scale: float = LTX23_ONE_STAGE_RECIPE.guidance_scale,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
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
