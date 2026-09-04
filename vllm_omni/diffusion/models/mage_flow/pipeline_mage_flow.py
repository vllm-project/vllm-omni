# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Request-batched native Mage-Flow text-to-image and image-edit pipeline."""

import copy
import os
import posixpath
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_exists

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.model_loader.hub_prefetch import (
    from_pretrained_with_prefetch,
    prefetch_subfolders,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .autoencoder_mage import MageVAE
from .mage_flow_transformer import MageFlowTransformer2DModel
from .prompt_utils import (
    MageFlowVariantDefaults,
    encode_mage_flow_prompt,
    get_mage_flow_variant_defaults,
    normalize_mage_flow_reference_images,
    parse_mage_flow_extra_args,
    preprocess_mage_flow_reference_for_vae,
    validate_mage_flow_size,
)


@dataclass(frozen=True)
class _MageFlowCheckpoint:
    """Component layout and variant defaults resolved from ``model_index.json``."""

    text_encoder_subfolder: str
    vae_subfolder: str
    vae_filename: str
    vae_config: dict[str, Any]
    transformer_config: dict[str, Any]
    variant_defaults: MageFlowVariantDefaults


@dataclass
class _MageFlowBatchItem:
    sampling: Any
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    output_type: str
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    target_latents: torch.Tensor
    reference_latents: torch.Tensor
    image_grids: list[tuple[int, int]]


def make_mage_flow_request_scheduler(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    num_inference_steps: int,
    device: torch.device,
) -> FlowMatchEulerDiscreteScheduler:
    """Create an isolated static-shift schedule for one request."""
    request_scheduler = copy.deepcopy(scheduler)
    base_sigmas = torch.linspace(
        1.0,
        1.0 / num_inference_steps,
        num_inference_steps,
    ).tolist()
    request_scheduler.set_timesteps(sigmas=base_sigmas, device=device)
    return request_scheduler


MAGE_FLOW_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        # ``y_embedder``'s encoder and bottleneck belong to the training
        # pipeline alone and have no inference counterpart. Order matters:
        # ``orig_to_new_prefix`` is applied in insertion order, so the dropped
        # prefixes must precede the broader ``vae.pipeline.`` rewrite that
        # would otherwise claim them first.
        "vae.pipeline.y_embedder.encoder.": None,
        "vae.pipeline.y_embedder.bottleneck.": None,
        "vae.student.dconv_encoder.": "vae.dconv_encoder.",
        "vae.pipeline.": "vae.decoder_model.",
    }
)


def get_mage_flow_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Validate the shared T2I/Edit prompt contract before batching."""
    try:
        model_index = get_hf_file_to_dict(
            "model_index.json",
            od_config.model,
            od_config.revision,
        )
    except Exception:
        model_index = None
    variant_defaults = get_mage_flow_variant_defaults(od_config.model, model_index=model_index)

    def pre_process_func(request: OmniDiffusionRequest):
        prompt = request.prompt
        if isinstance(prompt, str):
            return request
        multi_modal_data = prompt.get("multi_modal_data", {})
        images = multi_modal_data.get("image")
        if images is not None:
            if variant_defaults.supports_reference_images:
                normalize_mage_flow_reference_images(images)
            elif request.is_dummy_run():
                # The engine's warmup request attaches a dummy image to every
                # multimodal pipeline. A T2I checkpoint cannot consume one, so
                # drop it and let warmup exercise the T2I path; that also keeps
                # the Qwen3-VL vision tower out of T2I startup entirely.
                multi_modal_data.pop("image")
            else:
                raise ValueError(
                    f"Mage-Flow checkpoint {od_config.model} is text-to-image and does not accept "
                    "reference images; serve an Edit checkpoint (e.g. microsoft/Mage-Flow-Edit) "
                    "for image-editing requests"
                )
        if not isinstance(prompt.get("prompt"), str):
            raise ValueError("Mage-Flow prompt must contain a string 'prompt'")
        return request

    return pre_process_func


def get_mage_flow_post_process_func(
    od_config: OmniDiffusionConfig,
):
    image_processor = VaeImageProcessor(
        vae_scale_factor=MageVAE.downsample_factor,
        do_convert_rgb=True,
    )

    def post_process_func(
        images: torch.Tensor,
        *,
        sampling_params: Any,
    ):
        output_type = sampling_params.output_type or od_config.output_type or "pil"
        if output_type == "latent":
            return images
        if output_type == "tensor":
            return image_processor.postprocess(images, output_type="pt")
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func


class MageFlowPipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Native request-batched T2I and multi-reference Edit pipeline."""

    supports_request_batch = True
    supports_step_execution = False
    support_image_input = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self._validate_runtime_config()

        model = od_config.model
        local_files_only = os.path.isdir(model)
        checkpoint = self._load_checkpoint_metadata()
        text_encoder_subfolder = checkpoint.text_encoder_subfolder
        vae_subfolder = checkpoint.vae_subfolder
        # Resolved once from the checkpoint's own identity, so no request has to
        # re-derive it and no request can steer it.
        self.variant_defaults = checkpoint.variant_defaults
        component_subfolders = [
            "scheduler",
            text_encoder_subfolder,
            "transformer",
            vae_subfolder,
        ]
        prefetch_subfolders(
            model,
            component_subfolders,
            local_files_only=local_files_only,
        )
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder=vae_subfolder,
                revision=od_config.revision,
                prefix="vae.",
                fall_back_to_pt=False,
                allow_patterns_overrides=[checkpoint.vae_filename],
            ),
        ]

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )
        if self.scheduler.config.get("use_dynamic_shifting", False):
            raise ValueError("Mage-Flow requires a static scheduler shift")
        if float(self.scheduler.config.get("shift", 0.0)) != 6.0:
            raise ValueError(f"Mage-Flow requires scheduler shift=6.0, got {self.scheduler.config.get('shift')!r}")
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3VLForConditionalGeneration.from_pretrained,
            model,
            subfolder=text_encoder_subfolder,
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
            revision=od_config.revision,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model,
            subfolder=text_encoder_subfolder,
            local_files_only=local_files_only,
            revision=od_config.revision,
        )
        self.transformer = MageFlowTransformer2DModel(
            od_config=od_config,
            **checkpoint.transformer_config,
        )
        # ``sample_posterior`` is not forwarded: _load_checkpoint_metadata
        # already rejects any value but False, so the VAE only ever needs the
        # posterior mode.
        self.vae = MageVAE(
            od_config=od_config,
            latent_channels=int(checkpoint.vae_config["latent_channels"]),
            downsample_factor=int(checkpoint.vae_config["downsample_factor"]),
        )
        if self.transformer.in_channels != self.vae.latent_channels:
            raise ValueError(
                "Mage-Flow transformer/VAE channel mismatch: "
                f"{self.transformer.in_channels} != {self.vae.latent_channels}"
            )

        self.default_sample_size = 1024
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=[
                "text_encoder.forward",
                "vae.encode",
                "vae.decode",
                "diffuse_batch",
            ],
            enable_diffusion_pipeline_profiler=(od_config.enable_diffusion_pipeline_profiler),
        )

    def _validate_runtime_config(self) -> None:
        if self.device.type != "cuda":
            raise ValueError(f"Mage-Flow currently supports only CUDA execution, got {self.device.type}")
        if self.od_config.dtype != torch.bfloat16:
            raise ValueError(f"Mage-Flow currently supports only BF16, got {self.od_config.dtype}")
        if self.od_config.quantization_config is not None:
            raise ValueError("Mage-Flow does not yet support quantized loading")
        if self.od_config.lora_path is not None:
            raise ValueError("Mage-Flow does not yet support LoRA")
        if self.od_config.enable_prompt_embed_cache:
            raise ValueError("Mage-Flow does not yet support the shared prompt embedding cache")
        if self.od_config.max_num_seqs < 1:
            raise ValueError("Mage-Flow max_num_seqs must be at least 1")
        if self.od_config.vae_use_slicing or self.od_config.vae_use_tiling:
            raise ValueError("Mage-Flow's native VAE does not yet support slicing or tiling")
        parallel = self.od_config.parallel_config
        # Tensor, CFG and sequence parallelism are supported. TP shards attention
        # heads and FFN (validated against num_heads at build time); CFG parallel
        # splits the positive/negative branches across exactly two ranks; SP
        # shards the image token stream via the transformer's _sp_plan.
        unsupported = {
            "pipeline parallel": parallel.pipeline_parallel_size,
            "VAE patch parallel": parallel.vae_patch_parallel_size,
        }
        enabled = [f"{name}={size}" for name, size in unsupported.items() if size not in (None, 1)]
        if enabled:
            raise ValueError(
                "Mage-Flow currently supports only tensor, CFG and sequence "
                "parallelism for multi-GPU execution; unsupported settings: " + ", ".join(enabled)
            )
        cfg_parallel_size = parallel.cfg_parallel_size
        if cfg_parallel_size not in (None, 1, 2):
            raise ValueError(
                "Mage-Flow CFG parallelism splits one positive and one negative "
                f"branch, so cfg_parallel_size must be 1 or 2, got {cfg_parallel_size}"
            )
        if parallel.sequence_parallel_size > 1 and self.od_config.max_num_seqs > 1:
            # Sharding splits the padded sequence blindly, so the transformer
            # rejects a batch whose requests differ in token count. Prompt
            # length is not part of the request-batch key, so any two concurrent
            # requests can trip that. Refusing here turns a mid-generation
            # failure into a startup one.
            raise ValueError(
                f"Mage-Flow sequence parallelism requires max_num_seqs=1, got "
                f"{self.od_config.max_num_seqs}. Requests of differing token counts "
                f"share a batch and cannot be sharded; run with --max-num-seqs 1, or "
                f"drop --ulysses-degree/--ring-degree to keep request batching."
            )
        if parallel.use_hsdp:
            raise ValueError("Mage-Flow does not yet support HSDP")

    def _load_checkpoint_metadata(self) -> _MageFlowCheckpoint:
        model_index = get_hf_file_to_dict(
            "model_index.json",
            self.od_config.model,
            self.od_config.revision,
        )
        if model_index is None:
            raise ValueError(f"Mage-Flow model_index.json not found for {self.od_config.model}")
        if model_index.get("_class_name") != "MageFlowPipeline":
            raise ValueError(
                f"Expected _class_name='MageFlowPipeline' in model_index.json, got {model_index.get('_class_name')!r}"
            )

        text_encoder_subfolder = self._validate_component_path(
            "_text_encoder_path",
            model_index.get("_text_encoder_path", "text_encoder"),
        )
        vae_source = self._validate_component_path(
            "_vae_source",
            model_index.get(
                "_vae_source",
                "vae/diffusion_pytorch_model.safetensors",
            ),
        )
        vae_subfolder, vae_filename = posixpath.split(vae_source)
        if not vae_subfolder or not vae_filename:
            raise ValueError(f"Mage-Flow _vae_source must include a subfolder and filename, got {vae_source!r}")
        if not os.path.isdir(self.od_config.model) and not file_exists(
            self.od_config.model,
            vae_source,
            revision=self.od_config.revision,
        ):
            raise ValueError(f"Mage-Flow _vae_source resolves to missing Hub path {self.od_config.model}/{vae_source}")

        vae_config = get_hf_file_to_dict(
            f"{vae_subfolder}/config.json",
            self.od_config.model,
            self.od_config.revision,
        )
        if vae_config is None:
            raise ValueError(f"Mage-Flow VAE config not found at {vae_subfolder}/config.json")
        expected_vae_config = {
            "_class_name": "MageVAE",
            "latent_channels": 128,
            "downsample_factor": 16,
            "sample_posterior": False,
        }
        mismatches = {
            key: (vae_config.get(key), expected)
            for key, expected in expected_vae_config.items()
            if vae_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Unsupported Mage-Flow VAE config values: {mismatches}")
        transformer_config = get_hf_file_to_dict(
            "transformer/config.json",
            self.od_config.model,
            self.od_config.revision,
        )
        if transformer_config is None:
            raise ValueError("Mage-Flow Transformer config not found at transformer/config.json")
        return _MageFlowCheckpoint(
            text_encoder_subfolder=text_encoder_subfolder,
            vae_subfolder=vae_subfolder,
            vae_filename=vae_filename,
            vae_config=vae_config,
            transformer_config=transformer_config,
            variant_defaults=get_mage_flow_variant_defaults(
                self.od_config.model,
                model_index=model_index,
            ),
        )

    def _validate_component_path(self, key: str, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Mage-Flow {key} must be a non-empty relative path, got {value!r}")
        normalized = posixpath.normpath(value)
        if posixpath.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
            raise ValueError(f"Mage-Flow {key} must stay inside the checkpoint, got {value!r}")
        if os.path.isdir(self.od_config.model):
            resolved_path = os.path.join(
                self.od_config.model,
                *normalized.split("/"),
            )
            if not os.path.exists(resolved_path):
                raise ValueError(f"Mage-Flow {key} resolves to missing path {resolved_path}")
        return normalized

    def _request_generator(
        self,
        generator: torch.Generator | list[torch.Generator] | None,
        seed: int | None,
    ) -> torch.Generator:
        if isinstance(generator, list):
            raise ValueError("Mage-Flow currently supports exactly one request generator")
        if generator is not None:
            if generator.device != self.device:
                local_generator = torch.Generator(device=self.device)
                local_generator.manual_seed(generator.initial_seed())
                return local_generator
            return generator
        local_generator = torch.Generator(device=self.device)
        # OmniDiffusionRequest assigns a random seed when the caller supplies
        # neither a seed nor a generator, so a real request always arrives with
        # one. Draw the same way it does for the paths that do not go through
        # it (the engine's warmup request), rather than pinning a constant that
        # would make every such run produce the identical image.
        local_generator.manual_seed(random.randint(0, 2**31 - 1) if seed is None else seed)
        return local_generator

    def _prepare_latents(
        self,
        *,
        height: int,
        width: int,
        generator: torch.Generator,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = (
            1,
            self.vae.latent_channels,
            height // self.vae.downsample_factor,
            width // self.vae.downsample_factor,
        )
        if latents is None:
            # Draw from the request's own generator rather than the global RNG,
            # so a seeded request reproduces regardless of what else ran before.
            latent_image = torch.randn(
                shape,
                generator=generator,
                device=self.device,
                dtype=self.od_config.dtype,
            )
        else:
            if latents.shape == (
                1,
                shape[2] * shape[3],
                shape[1],
            ):
                latent_image = latents.transpose(1, 2).reshape(shape)
            elif tuple(latents.shape) == shape:
                latent_image = latents
            else:
                raise ValueError(
                    f"Mage-Flow latents must have shape {shape} or "
                    f"(1, {shape[2] * shape[3]}, {shape[1]}), got "
                    f"{tuple(latents.shape)}"
                )
            if latent_image.dtype != self.od_config.dtype:
                raise ValueError(
                    f"Mage-Flow supplied latents must use {self.od_config.dtype}, got {latent_image.dtype}"
                )
            if latent_image.device != self.device:
                raise ValueError(f"Mage-Flow supplied latents must be on {self.device}, got {latent_image.device}")
        return latent_image.flatten(2).transpose(1, 2).contiguous()

    def _prepare_reference_latents(
        self,
        reference_images,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        pixel_values = torch.stack(
            [
                preprocess_mage_flow_reference_for_vae(
                    image,
                    height=height,
                    width=width,
                )
                for image in reference_images
            ]
        ).to(device=self.device, dtype=self.vae.dtype)
        reference_latents = self.vae.encode(pixel_values)
        return (
            reference_latents.flatten(2)
            .transpose(1, 2)
            .reshape(1, -1, self.vae.latent_channels)
            .to(dtype=self.od_config.dtype)
            .contiguous()
        )

    @staticmethod
    def _pad_token_sequences(
        sequences: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not sequences:
            raise ValueError("Mage-Flow cannot pad an empty token sequence list")
        normalized = []
        lengths = []
        feature_dim = sequences[0].shape[-1]
        for sequence in sequences:
            if sequence.ndim != 3 or sequence.shape[0] != 1:
                raise ValueError(f"Mage-Flow token sequences must have shape [1, S, D], got {tuple(sequence.shape)}")
            if sequence.shape[-1] != feature_dim:
                raise ValueError("Mage-Flow padded token sequences must share the feature dimension")
            normalized.append(sequence[0])
            lengths.append(sequence.shape[1])
        max_length = max(lengths)
        padded = sequences[0].new_zeros(
            len(sequences),
            max_length,
            feature_dim,
        )
        mask = torch.zeros(
            len(sequences),
            max_length,
            dtype=torch.bool,
            device=sequences[0].device,
        )
        for index, (sequence, length) in enumerate(zip(normalized, lengths)):
            padded[index, :length] = sequence
            mask[index, :length] = True
        return padded, mask, lengths

    def predict_noise(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_grid_hw: list[tuple[int, int]] | list[list[tuple[int, int]]],
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_grid_hw=image_grid_hw,
            image_attention_mask=image_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )[0]

    def _predict_noise_packed_cfg(
        self,
        *,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        guidance_scale: float,
        cfg_normalize: bool,
    ) -> torch.Tensor:
        if negative_kwargs is None:
            return self.predict_noise(**positive_kwargs)

        batch_size = positive_kwargs["hidden_states"].shape[0]
        packed_kwargs: dict[str, Any] = {}
        for key, positive_value in positive_kwargs.items():
            negative_value = negative_kwargs[key]
            if key == "image_grid_hw":
                packed_kwargs[key] = list(positive_value) + list(negative_value)
            elif isinstance(positive_value, torch.Tensor):
                if (
                    key
                    in {
                        "encoder_hidden_states",
                        "encoder_attention_mask",
                    }
                    and positive_value.shape[1] != negative_value.shape[1]
                ):
                    max_length = max(
                        positive_value.shape[1],
                        negative_value.shape[1],
                    )
                    positive_padding = max_length - positive_value.shape[1]
                    negative_padding = max_length - negative_value.shape[1]
                    if positive_padding:
                        positive_value = torch.nn.functional.pad(
                            positive_value,
                            (0, positive_padding) if positive_value.ndim == 2 else (0, 0, 0, positive_padding),
                        )
                    if negative_padding:
                        negative_value = torch.nn.functional.pad(
                            negative_value,
                            (0, negative_padding) if negative_value.ndim == 2 else (0, 0, 0, negative_padding),
                        )
                packed_kwargs[key] = torch.cat(
                    [positive_value, negative_value],
                    dim=0,
                )
            else:
                raise TypeError(f"Unsupported Mage-Flow packed CFG field {key!r}")

        packed_prediction = self.predict_noise(**packed_kwargs)
        positive_prediction = packed_prediction[:batch_size]
        negative_prediction = packed_prediction[batch_size:]
        combined = self.combine_cfg_noise(
            positive_prediction,
            negative_prediction,
            guidance_scale,
            cfg_normalize,
        )
        if not isinstance(combined, torch.Tensor):
            raise TypeError("Mage-Flow packed CFG expects a tensor prediction")
        return combined

    def diffuse_batch(
        self,
        *,
        target_latents: torch.Tensor,
        target_attention_mask: torch.Tensor,
        target_lengths: list[int],
        reference_latents: list[torch.Tensor],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        image_grid_hw: list[list[tuple[int, int]]],
        num_inference_steps: int,
        guidance_scale: float,
        cfg_normalize: bool,
    ) -> torch.Tensor:
        """Denoise a padded request batch with one packed CFG forward per step."""
        batch_size = target_latents.shape[0]
        if len(target_lengths) != batch_size or len(reference_latents) != batch_size:
            raise ValueError("Mage-Flow batch metadata must match the latent batch size")
        scheduler = make_mage_flow_request_scheduler(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=self.device,
        )
        do_cfg = guidance_scale > 1.0
        cfg_parallel = get_classifier_free_guidance_world_size() > 1
        # Packed CFG pads the shorter guidance branch, and sequence parallelism
        # cannot shard a padded sequence -- the transformer refuses one. Run the
        # branches as two separate unpadded forwards instead, which shard
        # cleanly. Pairing SP with --cfg-parallel-size 2 is faster because it
        # splits the branches across ranks, but SP alone has to stay correct
        # rather than failing once the first guided request arrives.
        sequence_parallel = self.od_config.parallel_config.sequence_parallel_size > 1

        with self.progress_bar(total=len(scheduler.timesteps)) as progress:
            for step_index, timestep in enumerate(scheduler.timesteps):
                combined_sequences = [
                    torch.cat(
                        [
                            target_latents[index : index + 1, :target_length],
                            reference_latents[index],
                        ],
                        dim=1,
                    )
                    for index, target_length in enumerate(target_lengths)
                ]
                combined_latents, image_attention_mask, _ = self._pad_token_sequences(combined_sequences)
                sigma = scheduler.sigmas[step_index].to(
                    device=self.device,
                    dtype=target_latents.dtype,
                )
                positive_kwargs = {
                    "hidden_states": combined_latents,
                    "encoder_hidden_states": prompt_embeds,
                    "timestep": sigma.expand(batch_size),
                    "image_grid_hw": image_grid_hw,
                    "image_attention_mask": image_attention_mask,
                    "encoder_attention_mask": prompt_attention_mask,
                }
                negative_kwargs = (
                    {
                        "hidden_states": combined_latents,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "timestep": sigma.expand(batch_size),
                        "image_grid_hw": image_grid_hw,
                        "image_attention_mask": image_attention_mask,
                        "encoder_attention_mask": negative_prompt_attention_mask,
                    }
                    if do_cfg
                    else None
                )
                if cfg_parallel or (do_cfg and sequence_parallel):
                    # One branch per rank under CFG parallelism; two sequential
                    # unpadded forwards under sequence parallelism. Packing both
                    # branches into the batch dimension would defeat the split
                    # in the first case and pad the shard in the second, so the
                    # mixin's own path replaces it here.
                    full_noise_prediction = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_cfg,
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=cfg_normalize,
                    )
                else:
                    full_noise_prediction = self._predict_noise_packed_cfg(
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        guidance_scale=guidance_scale,
                        cfg_normalize=cfg_normalize,
                    )
                target_noise_prediction = torch.zeros_like(target_latents)
                for index, target_length in enumerate(target_lengths):
                    target_noise_prediction[index, :target_length] = full_noise_prediction[index, :target_length]
                target_latents = self.scheduler_step(
                    target_noise_prediction,
                    timestep,
                    target_latents,
                    per_request_scheduler=scheduler,
                )
                target_latents = target_latents * target_attention_mask[..., None]
                progress.update()
        return target_latents

    def _prepare_batch_item(
        self,
        prompt_data: Any,
        sampling: Any,
        *,
        is_dummy_warmup: bool = False,
    ) -> _MageFlowBatchItem:
        defaults = self.variant_defaults
        if sampling.num_outputs_per_prompt != 1:
            raise ValueError("Mage-Flow currently supports num_outputs_per_prompt=1")
        reference_images = []
        if isinstance(prompt_data, str):
            prompt = prompt_data
            negative_prompt = " "
        else:
            prompt = prompt_data.get("prompt") or ""
            negative_prompt = prompt_data.get("negative_prompt") or " "
            images = prompt_data.get("multi_modal_data", {}).get("image")
            reference_images = normalize_mage_flow_reference_images(images)
        if not prompt:
            raise ValueError("Mage-Flow prompt must not be empty")
        if reference_images and not defaults.supports_reference_images:
            # pre_process_func rejects these requests upstream; this guards the
            # paths that reach the pipeline directly. A T2I transformer's
            # reference tokens were never trained, so running Edit on one would
            # silently produce unconditioned output.
            raise ValueError(
                "Mage-Flow reference images require an Edit checkpoint; "
                f"{self.od_config.model} is a text-to-image variant"
            )

        height = sampling.height if sampling.height is not None else self.default_sample_size
        width = sampling.width if sampling.width is not None else self.default_sample_size
        validate_mage_flow_size(height, width)
        num_inference_steps = (
            sampling.num_inference_steps if sampling.num_inference_steps is not None else defaults.num_inference_steps
        )
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else defaults.guidance_scale
        if guidance_scale < 1.0:
            if not is_dummy_warmup:
                raise ValueError("Mage-Flow guidance_scale must be at least 1.0")
            # The engine's warmup request pins guidance_scale=0.0 to keep CFG
            # off. Mage-Flow spells that same intent as 1.0, so accept it here
            # rather than failing startup, while real requests stay validated.
            guidance_scale = 1.0
        output_type = sampling.output_type or "pil"
        if output_type not in {"pil", "tensor", "latent"}:
            raise ValueError("Mage-Flow output_type must be 'pil', 'tensor', or 'latent'")
        vision_long_edge = parse_mage_flow_extra_args(sampling.extra_args)

        is_edit = bool(reference_images)
        encode_kwargs = (
            {
                "reference_images": reference_images,
                "vision_long_edge": vision_long_edge,
            }
            if is_edit
            else {}
        )
        prompt_embeds = encode_mage_flow_prompt(
            self.text_encoder,
            self.processor,
            prompt,
            device=self.device,
            **encode_kwargs,
        )
        do_cfg = guidance_scale > 1.0
        negative_prompt_embeds = (
            encode_mage_flow_prompt(
                self.text_encoder,
                self.processor,
                negative_prompt,
                device=self.device,
                **encode_kwargs,
            )
            if do_cfg
            else None
        )
        generator = self._request_generator(
            sampling.generator,
            sampling.seed,
        )
        latents = self._prepare_latents(
            height=height,
            width=width,
            generator=generator,
            latents=sampling.latents,
        )
        if is_edit:
            reference_latents = self._prepare_reference_latents(
                reference_images,
                height=height,
                width=width,
            )
        else:
            reference_latents = latents.new_empty(
                1,
                0,
                latents.shape[-1],
            )

        latent_grid = (
            height // self.vae.downsample_factor,
            width // self.vae.downsample_factor,
        )
        return _MageFlowBatchItem(
            sampling=sampling,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            target_latents=latents,
            reference_latents=reference_latents,
            image_grids=[latent_grid] * (1 + len(reference_images)),
        )

    @staticmethod
    def _validate_batch_compatibility(
        items: list[_MageFlowBatchItem],
    ) -> None:
        first = items[0]
        for item in items[1:]:
            incompatible = []
            for name in (
                "num_inference_steps",
                "guidance_scale",
                "output_type",
            ):
                if getattr(item, name) != getattr(first, name):
                    incompatible.append(name)
            if item.sampling.cfg_normalize != first.sampling.cfg_normalize:
                incompatible.append("cfg_normalize")
            if incompatible:
                raise ValueError("Mage-Flow request batch has incompatible sampling fields: " + ", ".join(incompatible))

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        if not req.num_reqs:
            raise ValueError("Mage-Flow request batch must not be empty")

        is_dummy_warmup = req.is_dummy_run()
        items = [
            self._prepare_batch_item(prompt_data, sampling, is_dummy_warmup=is_dummy_warmup)
            for prompt_data, sampling in zip(
                req.prompts,
                req.sampling_params_list,
            )
        ]
        self._validate_batch_compatibility(items)

        target_latents, target_attention_mask, target_lengths = self._pad_token_sequences(
            [item.target_latents for item in items]
        )
        prompt_embeds, prompt_attention_mask, _ = self._pad_token_sequences([item.prompt_embeds for item in items])
        do_cfg = items[0].guidance_scale > 1.0
        if do_cfg:
            negative_prompt_embeds, negative_prompt_attention_mask, _ = self._pad_token_sequences(
                [item.negative_prompt_embeds for item in items if item.negative_prompt_embeds is not None]
            )
            if negative_prompt_embeds.shape[0] != len(items):
                raise ValueError("Mage-Flow CFG request batch is missing negative prompt embeddings")
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        target_latents = self.diffuse_batch(
            target_latents=target_latents,
            target_attention_mask=target_attention_mask,
            target_lengths=target_lengths,
            reference_latents=[item.reference_latents for item in items],
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            image_grid_hw=[item.image_grids for item in items],
            num_inference_steps=items[0].num_inference_steps,
            guidance_scale=items[0].guidance_scale,
            cfg_normalize=items[0].sampling.cfg_normalize,
        )

        stage_durations = self.stage_durations if hasattr(self, "_stage_durations") else None
        outputs = []
        for index, (item, target_length) in enumerate(zip(items, target_lengths)):
            latent_height = item.height // self.vae.downsample_factor
            latent_width = item.width // self.vae.downsample_factor
            latent_image = (
                target_latents[index : index + 1, :target_length]
                .transpose(1, 2)
                .reshape(
                    1,
                    self.vae.latent_channels,
                    latent_height,
                    latent_width,
                )
            )
            output = (
                latent_image
                if item.output_type == "latent"
                else self.vae.decode(latent_image.to(dtype=self.vae.dtype)).clamp(-1, 1)
            )
            outputs.append(
                DiffusionOutput(
                    output=output,
                    stage_durations=stage_durations,
                )
            )
        return outputs

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Rewrite the checkpoint's MageVAE prefixes and load each component.

        ``AutoWeightsLoader`` walks to the owning submodule and hands off to its
        own ``load_weights``, which is what keeps Q/K/V fusion confined to the
        transformer: the VAE's ``.q``/``.k``/``.v`` convolutions live under a
        module that knows nothing about the fused projections and so can never
        be rewritten by them.
        """
        return AutoWeightsLoader(self).load_weights(weights, mapper=MAGE_FLOW_WEIGHTS_MAPPER)
