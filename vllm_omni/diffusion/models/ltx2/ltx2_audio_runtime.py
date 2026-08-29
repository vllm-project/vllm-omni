# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audio-only runtime for full LTX family checkpoints."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from dataclasses import replace
from typing import ClassVar

import torch
from diffusers import AutoencoderKLLTX2Audio, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from torch import nn
from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group as get_guidance_parallel_group,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank as get_guidance_parallel_rank,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size as get_guidance_parallel_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.interface import (
    SupportAudioOutput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from . import ltx2_latents as latent_ops
from .ltx2_audio_cuda_graph import LTX2AudioCUDAGraphConfig, LTX2AudioCUDAGraphRunner
from .ltx2_components import (
    LTXComponentProfile,
    _install_connector_attention,
    _load_component,
    _place_aux_components,
    create_audio_transformer_from_config,
    detect_ltx_model_version,
    load_transformer_config,
    resolve_ltx_checkpoint_kind,
    resolve_ltx_component_profile,
)
from .ltx2_conditioning import LTXTextConditioningMixin
from .ltx2_denoise import _official_ltx_sigmas
from .ltx2_guidance import (
    LTX_GUIDANCE_EXECUTOR,
    LTXGuidancePlan,
    _repeat_batch,
    build_perturbation_kwargs,
    euler_step_from_velocity,
)
from .ltx2_recipes import LTXPipelineRecipe, resolve_ltx_pipeline_recipe
from .ltx2_request import (
    LTXRequestMixin,
    resolve_ltx_audio_num_frames,
    validate_ltx_checkpoint,
    validate_pipeline_request,
)
from .ltx2_runtime import _run_ltx_vocoder

logger = init_logger(__name__)

_LTX_AUDIO_COMPONENT_SUBFOLDERS = (
    "tokenizer",
    "text_encoder",
    "connectors",
    "audio_vae",
    "vocoder",
    "scheduler",
)


def initialize_audio_pipeline_components(pipeline, od_config) -> None:
    """Build an LTX graph containing no video Transformer or video VAE."""
    profile = pipeline.component_profile
    pipeline.od_config = od_config
    pipeline.device = get_local_device()
    dtype = getattr(od_config, "dtype", torch.bfloat16)
    model = od_config.model
    revision = getattr(od_config, "revision", None)
    local_files_only = os.path.exists(model)

    pipeline.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(
            model_or_path=model,
            subfolder=profile.transformer_subfolder,
            revision=revision,
            prefix="transformer.",
            fall_back_to_pt=True,
            weight_name_patterns=(
                "audio_*",
                "transformer_blocks.*.audio_*",
            ),
        )
    ]
    prefetch_subfolders(
        model,
        _LTX_AUDIO_COMPONENT_SUBFOLDERS,
        local_files_only=local_files_only,
        revision=revision,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model,
        subfolder="scheduler",
        local_files_only=local_files_only,
        revision=revision,
    )

    # LTX-2.5 publishes the distilled scheduler as the repository default,
    # while the Full/SFT transformer (used by the regular one-stage and T2A
    # profiles) requires dynamic shifting.  Apply the profile override before
    # validating the checkpoint kind; otherwise a valid LTX-2.5 Full/SFT
    # checkpoint is rejected based on the deliberately shared distilled
    # scheduler config.
    if profile.scheduler_use_dynamic_shifting:
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            use_dynamic_shifting=True,
            shift_terminal=profile.scheduler_shift_terminal,
        )

    # Reject incompatible checkpoints before allocating the text encoder,
    # connectors, VAE, vocoder, or Transformer.
    validate_ltx_checkpoint(
        pipeline.scheduler.config,
        expected_kind=resolve_ltx_checkpoint_kind(pipeline.pipeline_kind),
        pipeline_name=type(pipeline).__name__,
    )
    pipeline.tokenizer = AutoTokenizer.from_pretrained(
        model,
        subfolder="tokenizer",
        local_files_only=local_files_only,
        revision=revision,
    )
    if profile.text_encoder_cls is None:
        raise ImportError("LTX-2.5 requires Gemma4UnifiedForConditionalGeneration; install transformers>=5.10.1,<5.15.")
    with torch.device("cpu"):
        pipeline.text_encoder = _load_component(
            profile.text_encoder_cls,
            model,
            "text_encoder",
            local_files_only=local_files_only,
            dtype=dtype,
            revision=revision,
        )
    pipeline.connectors = _load_component(
        LTX2TextConnectors,
        model,
        "connectors",
        local_files_only=local_files_only,
        dtype=dtype,
        revision=revision,
    )
    _install_connector_attention(
        pipeline.connectors,
        preserve_learned_register_mask=profile.preserve_connector_attention_mask,
    )
    pipeline.audio_vae = _load_component(
        AutoencoderKLLTX2Audio,
        model,
        "audio_vae",
        local_files_only=local_files_only,
        dtype=dtype,
        revision=revision,
    )
    try:
        pipeline.vocoder = _load_component(
            profile.vocoder_cls,
            model,
            "vocoder",
            local_files_only=local_files_only,
            dtype=dtype,
            revision=revision,
        )
    except (TypeError, OSError, ValueError):
        if profile.vocoder_fallback_cls is None or profile.vocoder_fallback_cls is profile.vocoder_cls:
            raise
        pipeline.vocoder = _load_component(
            profile.vocoder_fallback_cls,
            model,
            "vocoder",
            local_files_only=local_files_only,
            dtype=dtype,
            revision=revision,
        )

    transformer_config = load_transformer_config(
        model,
        profile.transformer_subfolder,
        local_files_only,
        revision=revision,
    )
    pipeline.transformer = create_audio_transformer_from_config(
        transformer_config,
        quant_config=getattr(od_config, "quantization_config", None),
    )
    pipeline.audio_vae_mel_compression_ratio = pipeline.audio_vae.mel_compression_ratio
    pipeline.audio_vae_temporal_compression_ratio = pipeline.audio_vae.temporal_compression_ratio
    pipeline.audio_sampling_rate = pipeline.audio_vae.config.sample_rate
    pipeline.audio_hop_length = pipeline.audio_vae.config.mel_hop_length
    tokenizer_max_length = pipeline.tokenizer.model_max_length
    if tokenizer_max_length is None or tokenizer_max_length > 100000:
        encoder_config = getattr(pipeline.text_encoder, "config", None)
        tokenizer_max_length = getattr(encoder_config, "max_position_embeddings", None)
        if tokenizer_max_length is None:
            tokenizer_max_length = getattr(encoder_config, "max_seq_len", None)
    pipeline.tokenizer_max_length = int(tokenizer_max_length or 1024)
    pipeline._interrupt = False
    _place_aux_components(pipeline)


class LTXAudioRuntime(
    LTXRequestMixin,
    LTXTextConditioningMixin,
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    SupportAudioOutput,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Shared one-stage runtime for audio-only LTX generation."""

    pipeline_kind: ClassVar[str] = "text_to_audio"
    component_profile: ClassVar[LTXComponentProfile]
    pipeline_recipe: ClassVar[LTXPipelineRecipe]
    support_audio_output = True
    support_image_input = False
    supports_request_batch = False
    # The generic diffusion warmup uses this class-level value when the
    # pipeline is registered by name.  LTX's causal audio clock requires
    # ``8 * k + 1`` frames; 9 is the smallest valid warmup shape.
    dummy_run_num_frames: ClassVar[int] = 9
    connector_batches_cfg = False
    preserve_sp_padded_audio_duration = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        parallel_config = getattr(od_config, "parallel_config", None)
        if getattr(parallel_config, "ulysses_mode", "strict") == "advanced_uaa":
            raise ValueError(f"{type(self).__name__} does not support ulysses_mode='advanced_uaa'; use 'strict'.")

        self._audio_cuda_graph_config = LTX2AudioCUDAGraphConfig.from_additional_config(
            getattr(od_config, "additional_config", None)
        )
        if self._audio_cuda_graph_config.enabled:
            self._validate_audio_cuda_graph_support(od_config, get_local_device())

        self.model_version = detect_ltx_model_version(
            od_config.model,
            revision=getattr(od_config, "revision", None),
        )

        self.component_profile = resolve_ltx_component_profile(self.pipeline_kind, self.model_version)
        self.pipeline_recipe = resolve_ltx_pipeline_recipe(self.pipeline_kind, self.model_version)
        if getattr(od_config, "cache_backend", "none") == "cache_dit":
            raise ValueError(f"{type(self).__name__} does not support cache_backend='cache_dit'.")
        self._dit_modules = list(self.component_profile.dit_modules)
        self._encoder_modules = list(self.component_profile.encoder_modules)
        self._vae_modules = list(self.component_profile.vae_modules)
        self._resident_modules = list(self.component_profile.resident_modules)

        super().__init__()
        initialize_audio_pipeline_components(self, od_config)
        self.audio_graph_runner = (
            LTX2AudioCUDAGraphRunner(
                self.transformer,
                max_graphs=self._audio_cuda_graph_config.max_entries,
                device=self.device,
            )
            if self._audio_cuda_graph_config.enabled
            else None
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(od_config, "enable_diffusion_pipeline_profiler", False)
        )

    @staticmethod
    def _validate_audio_cuda_graph_support(od_config: OmniDiffusionConfig, device: torch.device) -> None:
        parallel = od_config.parallel_config
        errors: list[str] = []
        if device.type != "cuda":
            errors.append("device must be CUDA")
        if od_config.dtype != torch.bfloat16:
            errors.append("dtype must be bfloat16")
        if parallel.tensor_parallel_size != 1:
            errors.append("tensor_parallel_size must be 1")
        if parallel.sequence_parallel_size != 1:
            errors.append("sequence_parallel_size must be 1")
        if od_config.enable_cpu_offload:
            errors.append("CPU offload is unsupported")
        if od_config.enable_layerwise_offload:
            errors.append("layerwise offload is unsupported")
        if getattr(od_config, "enable_distributed_layerwise_offload", False):
            errors.append("distributed layerwise offload is unsupported")
        if od_config.quantization_config is not None:
            errors.append("quantization is unsupported")
        if od_config.lora_path is not None:
            errors.append("LoRA is unsupported")
        if errors:
            raise ValueError("LTX2 audio CUDA Graph configuration is unsupported: " + "; ".join(errors))

    def setup_audio_cuda_graph_compile(self) -> None:
        """Compile the audio Transformer without compiler-managed graphing."""
        runner = self.audio_graph_runner
        if runner is None:
            return

        try:
            self.transformer = regionally_compile(
                self.transformer,
                dynamic=self.od_config.diffusion_compile_dynamic,
                options={
                    "triton.cudagraphs": False,
                    "triton.cudagraph_trees": False,
                },
            )
            runner.transformer = self.transformer
        except Exception as exc:
            logger.warning(
                "LTX2 audio regional compilation for manual CUDA Graph failed (%s); "
                "continuing with the eager Transformer inside the manual graph.",
                exc,
            )

        logger.info(
            "LTX2 audio Runtime-owned CUDA Graph replay enabled (max_entries=%d); "
            "compiler-managed Transformer graphing is bypassed.",
            self._audio_cuda_graph_config.max_entries,
        )

    def release_captured_graphs(self) -> None:
        """Synchronously release captured audio graphs while retaining the runner."""
        if self.audio_graph_runner is not None:
            self.audio_graph_runner.clear()

    @property
    def guidance_scale(self):
        return self._guidance_plan.spec.audio.cfg_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_plan.spec.audio.do_cfg

    @property
    def do_guidance(self):
        return len(self._guidance_plan.passes) > 1

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def _reject_video_options(sampling) -> None:
        extra = sampling.extra_args or {}
        unsupported = (
            "video_cfg_scale",
            "video_cfg_guidance_scale",
            "video_stg_scale",
            "video_stg_guidance_scale",
            "video_modality_scale",
            "a2v_guidance_scale",
            "video_rescale_scale",
            "video_stg_blocks",
            "image_crf",
            "latents",
        )
        supplied = [name for name in unsupported if extra.get(name) is not None]
        if sampling.latents is not None:
            supplied.append("latents")
        if supplied:
            names = ", ".join(sorted(set(supplied)))
            raise ValueError(f"LTX text-to-audio does not accept video-only option(s): {names}.")

    def _resolve_audio_request_inputs(self, req: DiffusionRequestBatch):
        if len(req.sampling_params_list) != 1:
            raise ValueError("LTX text-to-audio currently accepts one request at a time.")
        sampling = req.sampling_params_list[0]
        self._reject_video_options(sampling)
        extra = sampling.extra_args or {}
        audio_length = extra.get("audio_length")
        if audio_length is None and extra.get("audio_end_in_s") is not None:
            audio_length = float(extra["audio_end_in_s"]) - float(extra.get("audio_start_in_s", 0.0))
        exact_num_frames = extra.get("num_frames")
        if exact_num_frames is None and sampling.num_frames not in (None, 1):
            exact_num_frames = sampling.num_frames
        frame_rate = float(sampling.resolved_frame_rate or self.pipeline_recipe.frame_rate)
        resolved_num_frames = resolve_ltx_audio_num_frames(
            audio_length=None if audio_length is None else float(audio_length),
            num_frames=exact_num_frames,
            frame_rate=frame_rate,
            default_num_frames=self.pipeline_recipe.num_frames,
        )

        inputs = self._resolve_request_inputs(
            req,
            prompt=None,
            negative_prompt=None,
            height=self.pipeline_recipe.height,
            width=self.pipeline_recipe.width,
            num_frames=resolved_num_frames,
            frame_rate=frame_rate,
            num_inference_steps=None,
            guidance_scale=None,
            num_videos_per_prompt=1,
            generator=None,
            latents=None,
            audio_latents=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            decode_timestep=0.0,
            decode_noise_scale=None,
            output_type="np",
            max_sequence_length=None,
        )
        inputs = replace(
            inputs,
            height=self.pipeline_recipe.height,
            width=self.pipeline_recipe.width,
            num_frames=resolved_num_frames,
            frame_rate=frame_rate,
            latents=None,
        )
        validate_pipeline_request(
            inputs,
            pipeline_recipe=self.pipeline_recipe,
            vae_spatial_compression_ratio=32,
            vae_temporal_compression_ratio=8,
            pipeline_name=type(self).__name__,
            request_sigmas=self._resolve_request_sigmas(req, None),
        )
        if inputs.guidance.audio.do_modality_guidance:
            raise ValueError("LTX text-to-audio requires `audio_modality_scale=1.0` because no video branch exists.")
        return inputs

    def prepare_audio_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        audio_latent_length: int,
        num_mel_bins: int,
        *,
        noise_scale: float,
        dtype: torch.dtype,
        device: torch.device,
        generator,
        latents,
    ):
        return latent_ops.prepare_audio_latents(
            self,
            batch_size,
            num_channels_latents,
            audio_latent_length,
            num_mel_bins,
            noise_scale,
            dtype,
            device,
            generator,
            latents,
        )

    def _prepare_audio_state(self, inputs, prompt_context):
        duration_s = inputs.num_frames / inputs.frame_rate
        latent_rate = self.audio_sampling_rate / self.audio_hop_length / self.audio_vae_temporal_compression_ratio
        requested_frames = round(duration_s * latent_rate)
        num_mel_bins = self.audio_vae.config.mel_bins
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        audio_latents, original_frames, padded_frames = self.prepare_audio_latents(
            prompt_context.batch_size * inputs.num_videos_per_prompt,
            self.audio_vae.config.latent_channels,
            requested_frames,
            num_mel_bins,
            noise_scale=self.pipeline_recipe.phases[0].noise_scale,
            dtype=prompt_context.positive_connector_audio_prompt_embeds.dtype,
            device=self.device,
            generator=inputs.generator,
            latents=inputs.audio_latents,
        )
        return audio_latents, original_frames, padded_frames, latent_mel_bins

    @staticmethod
    def _set_audio_sigmas(scheduler, sigmas: torch.Tensor) -> torch.Tensor:
        scheduler.sigmas = sigmas.to(torch.float32)
        scheduler.timesteps = scheduler.sigmas[:-1] * scheduler.config.get("num_train_timesteps", 1000)
        scheduler.num_inference_steps = len(scheduler.timesteps)
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler.timesteps

    def _run_audio_denoise(
        self,
        audio_latents: torch.Tensor,
        prompt_context,
        inputs,
        *,
        original_num_frames: int,
        padded_num_frames: int,
        request_sigmas: list[float] | None,
    ) -> torch.Tensor:
        audio_scheduler = copy.deepcopy(self.scheduler)
        if request_sigmas is None:
            sigmas = _official_ltx_sigmas(audio_scheduler, inputs.num_inference_steps, self.device)
        else:
            sigmas = torch.as_tensor(request_sigmas, dtype=torch.float32, device=self.device)
            if sigmas.ndim != 1 or sigmas.numel() < 2:
                raise ValueError("An LTX custom sigma schedule must contain at least two boundary values.")
            if sigmas[-1] != 0:
                sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        timesteps = self._set_audio_sigmas(audio_scheduler, sigmas)
        plan = self._guidance_plan
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            padded_num_frames,
            audio_latents.device,
        )
        ring_degree = getattr(self.od_config.parallel_config, "ring_degree", 1) or 1
        if padded_num_frames > original_num_frames and ring_degree > 1:
            raise ValueError(
                "LTX audio padding requires an attention mask, which Ring sequence parallelism does not support. "
                "Use Ulysses-only SP or choose a duration whose audio latent length is divisible by the SP size."
            )
        audio_attention_mask = (
            torch.arange(padded_num_frames, device=audio_latents.device)
            .lt(original_num_frames)
            .unsqueeze(0)
            .expand(audio_latents.shape[0], -1)
            if padded_num_frames > original_num_frames
            else None
        )
        guidance_world_size = get_guidance_parallel_world_size()
        guidance_parallel_ready = self.do_guidance and guidance_world_size > 1
        LTX_GUIDANCE_EXECUTOR.validate_guidance_world_size(plan, guidance_world_size)
        LTX_GUIDANCE_EXECUTOR.warn_if_imbalanced(plan, guidance_world_size, "generate_audio")

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for index, timestep in enumerate(timesteps):
                if self.interrupt:
                    continue
                if guidance_parallel_ready:
                    assignments = LTX_GUIDANCE_EXECUTOR._parallel_assignments(len(plan.passes), guidance_world_size)
                    local_indices = assignments[get_guidance_parallel_rank()]
                    model_pass_count = max(len(indices) for indices in assignments)
                    padded_indices: list[int | None] = local_indices + [None] * (model_pass_count - len(local_indices))
                    local_passes = tuple(
                        plan.passes[0] if pass_index is None else plan.passes[pass_index]
                        for pass_index in padded_indices
                    )
                    local_plan = LTXGuidancePlan(spec=plan.spec, passes=local_passes)
                else:
                    local_passes = plan.passes
                    local_plan = plan
                    model_pass_count = len(plan.passes)

                contexts = []
                for denoise_pass in local_passes:
                    context = (
                        prompt_context.negative_connector_audio_prompt_embeds
                        if denoise_pass.negative_audio_context
                        else prompt_context.positive_connector_audio_prompt_embeds
                    )
                    if context is None:
                        raise ValueError("Negative audio prompt context is required when LTX T2A CFG is enabled.")
                    contexts.append(context)
                model_input = _repeat_batch(audio_latents, model_pass_count).to(
                    prompt_context.positive_connector_audio_prompt_embeds.dtype
                )
                perturbations = build_perturbation_kwargs(local_plan, audio_latents.shape[0], model_input)
                expanded_timestep = timestep.expand(model_input.shape[0])
                encoder_hidden_states = torch.cat(contexts)
                model_audio_attention_mask = (
                    None if audio_attention_mask is None else _repeat_batch(audio_attention_mask, model_pass_count)
                )
                model_audio_coords = _repeat_batch(audio_coords, model_pass_count)
                model_timestep = expanded_timestep[:, None].expand(-1, model_input.shape[1])
                model_sigma = audio_scheduler.sigmas[index].expand(model_input.shape[0])
                velocity = self._run_audio_transformer(
                    audio_hidden_states=model_input,
                    audio_encoder_hidden_states=encoder_hidden_states,
                    audio_timestep=model_timestep,
                    audio_sigma=model_sigma,
                    audio_coords=model_audio_coords,
                    audio_attention_mask=model_audio_attention_mask,
                    perturbation_mask=perturbations.get("audio_self_attention_mask"),
                    stg_blocks=perturbations.get("audio_self_attention_blocks"),
                )
                if guidance_parallel_ready:
                    local_slots = velocity.chunk(model_pass_count)
                    group = get_guidance_parallel_group()
                    gathered_slots = [group.all_gather(value, separate_tensors=True) for value in local_slots]
                    splits = {
                        denoise_pass.name: gathered_slots[pass_index // guidance_world_size][
                            pass_index % guidance_world_size
                        ]
                        for pass_index, denoise_pass in enumerate(plan.passes)
                    }
                else:
                    splits = dict(zip(plan.names, velocity.chunk(model_pass_count), strict=True))
                guided_velocity = LTX_GUIDANCE_EXECUTOR._guide_modality(
                    audio_latents,
                    splits,
                    audio_scheduler.sigmas[index],
                    plan.spec.audio,
                    rescale_token_count=original_num_frames,
                )
                audio_latents = euler_step_from_velocity(
                    audio_latents,
                    guided_velocity,
                    audio_scheduler.sigmas,
                    index,
                )
                audio_latents = latent_ops.clear_audio_padding(audio_latents, original_num_frames)
                progress_bar.update()
        return audio_latents

    def _run_audio_transformer(
        self,
        *,
        audio_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        audio_timestep: torch.Tensor,
        audio_sigma: torch.Tensor,
        audio_coords: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
        stg_blocks,
    ) -> torch.Tensor:
        """Run one normalized audio Transformer invocation eagerly or by graph."""
        audio_graph_runner = getattr(self, "audio_graph_runner", None)
        if audio_graph_runner is not None:
            return audio_graph_runner(
                audio_hidden_states=audio_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                audio_timestep=audio_timestep,
                audio_sigma=audio_sigma,
                audio_coords=audio_coords,
                audio_attention_mask=audio_attention_mask,
                perturbation_mask=perturbation_mask,
                stg_blocks=stg_blocks,
            )

        attention_kwargs = {}
        if perturbation_mask is not None:
            attention_kwargs["ltx_perturbation_kwargs"] = {
                "audio_self_attention_mask": perturbation_mask,
                "audio_self_attention_blocks": stg_blocks,
            }
        return self.transformer(
            audio_hidden_states=audio_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            audio_timestep=audio_timestep,
            audio_sigma=audio_sigma,
            audio_coords=audio_coords,
            audio_attention_mask=audio_attention_mask,
            attention_kwargs=attention_kwargs,
        )

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        inputs = self._resolve_audio_request_inputs(req)
        self._interrupt = False
        self._guidance_plan = LTXGuidancePlan.build(inputs.guidance)
        prompt_context = self._prepare_prompt_context(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            prompt_embeds=inputs.prompt_embeds,
            negative_prompt_embeds=inputs.negative_prompt_embeds,
            prompt_attention_mask=inputs.prompt_attention_mask,
            negative_prompt_attention_mask=inputs.negative_prompt_attention_mask,
            num_videos_per_prompt=inputs.num_videos_per_prompt,
            max_sequence_length=inputs.max_sequence_length,
        )
        audio_latents, original_frames, padded_frames, latent_mel_bins = self._prepare_audio_state(
            inputs,
            prompt_context,
        )
        request_sigmas = self._resolve_request_sigmas(req, None)
        audio_latents = self._run_audio_denoise(
            audio_latents,
            prompt_context,
            inputs,
            original_num_frames=original_frames,
            padded_num_frames=padded_frames,
            request_sigmas=request_sigmas,
        )
        if inputs.output_type == "latent":
            return DiffusionOutput(output=audio_latents)
        waveform = self._decode_audio_latents(
            audio_latents,
            original_num_frames=original_frames,
            latent_mel_bins=latent_mel_bins,
        )
        return DiffusionOutput(output=waveform)

    def _decode_audio_latents(
        self,
        audio_latents,
        *,
        original_num_frames: int,
        latent_mel_bins: int,
    ):
        """Undo audio packing/normalization and synthesize the waveform."""
        audio_latents = latent_ops.unpad_audio_latents(audio_latents, original_num_frames)
        audio_latents = latent_ops.denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = latent_ops.unpack_audio_latents(audio_latents, num_mel_bins=latent_mel_bins)
        generated_mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        return _run_ltx_vocoder(self.vocoder, generated_mel)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
