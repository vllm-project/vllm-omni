# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion pipeline for Ming-Image Design and Design-Layer checkpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import (
    set_forward_context_direct_condition,
    set_forward_context_ref_latent,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch
from vllm_omni.diffusion.models.ming_image.condition import MingImageConditioning
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)

_DESIGN_PIPELINE = "MingImageDiffusionPipeline"
_LAYERED_PIPELINE = "MingImageLayeredDiffusionPipeline"
_VENDOR_TRANSFORMER_CLASS = "DiffusionTransformer"
_DIFFUSION_REQUIRED_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "transformer/**",
    "vae/**",
    "mlp/**",
    "connector/**",
]


def _validate_variant_config(
    model_index: dict[str, Any] | None,
    transformer_config: Any,
) -> bool:
    """Validate checkpoint-owned variant metadata and return layered mode."""
    declared_transformer = getattr(transformer_config, "_class_name", None)
    if declared_transformer != _VENDOR_TRANSFORMER_CLASS:
        raise ValueError(
            "Ming-Image transformer/config.json must preserve the vendor "
            f"_class_name={_VENDOR_TRANSFORMER_CLASS!r}, got {declared_transformer!r}. "
            "vLLM-Omni loads those weights through MingImageTransformer2DModel."
        )

    alignment_padding_mode = getattr(transformer_config, "alignment_padding_mode", None)
    multi_frame_output = getattr(transformer_config, "multi_frame_output", None)
    variant = (alignment_padding_mode, multi_frame_output)
    if variant == ("zero_masked", False):
        is_layer_decomposition = False
    elif variant == ("learned", True):
        is_layer_decomposition = True
    else:
        raise ValueError(
            "Ming-Image transformer/config.json must use either "
            "alignment_padding_mode='zero_masked' with multi_frame_output=False "
            "or alignment_padding_mode='learned' with multi_frame_output=True; "
            f"got alignment_padding_mode={alignment_padding_mode!r}, "
            f"multi_frame_output={multi_frame_output!r}."
        )

    declared_pipeline = (model_index or {}).get("_class_name")
    if declared_pipeline is not None:
        if declared_pipeline not in {_DESIGN_PIPELINE, _LAYERED_PIPELINE}:
            raise ValueError(
                "Ming-Image model_index.json must declare "
                f"{_DESIGN_PIPELINE!r} or {_LAYERED_PIPELINE!r}, got {declared_pipeline!r}."
            )
        expected_pipeline = _LAYERED_PIPELINE if is_layer_decomposition else _DESIGN_PIPELINE
        if declared_pipeline != expected_pipeline:
            raise ValueError(
                f"Ming-Image model_index.json declares {declared_pipeline!r}, but "
                f"transformer/config.json describes {expected_pipeline!r}."
            )
    return is_layer_decomposition


class MingImageDiffusionPipeline(ZImagePipeline):
    """Ming-Image component adapter around the canonical Z-Image loop."""

    supports_request_batch = False

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["conditioning"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        nn.Module.__init__(self)

        model_path = od_config.model
        if not os.path.exists(model_path):
            model_path = download_weights_from_hf_specific(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=_DIFFUSION_REQUIRED_PATTERNS,
                revision=od_config.revision,
                require_all=True,
            )
        local_files_only = os.path.isdir(model_path)
        dtype = od_config.dtype

        self.od_config = od_config
        self._execution_device = get_local_device()
        self.device = self._execution_device
        model_index = get_diffusion_model_index(model_path, revision=od_config.revision) or {}
        transformer_config = od_config.tf_model_config
        self.is_layer_decomposition = _validate_variant_config(
            model_index,
            transformer_config,
        )

        self.default_num_inference_steps = 12
        self.default_guidance_scale = 2.0 if self.is_layer_decomposition else 1.0
        self._num_frames_per_prompt = 1
        self._pending_prompt_embeds: list[torch.Tensor] | None = None
        self._pending_negative_prompt_embeds: list[torch.Tensor] | None = None
        self._uses_cudagraph_trees = False

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        subfolders = ["scheduler", "transformer", "vae"]

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLQwenImage.from_pretrained,
            model_path,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae.eval()
        vae_channels = int(getattr(self.vae.config, "input_channels", 3))
        if vae_channels != 4:
            raise ValueError(f"Ming-Image requires an RGBA VAE, got input_channels={vae_channels}.")

        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config,
            MingImageTransformer2DModel,
        )
        self.transformer = MingImageTransformer2DModel(
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )
        self.conditioning = MingImageConditioning(
            model_path,
            device=self.device,
            dtype=dtype,
        )
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** len(self.vae.config.temperal_downsample)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            do_convert_rgb=False,
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def setup_compile(self) -> None:
        # Keep request preparation, scheduling, and VAE work eager,
        # while capturing repeated DiT blocks for CUDAGraph Trees replay.
        if self.od_config.diffusion_compile_granularity != "regional":
            logger.warning(
                "Ming-Image CUDA Graph uses regional DiT compilation; diffusion_compile_granularity=%r is ignored.",
                self.od_config.diffusion_compile_granularity,
            )
        self.transformer = regionally_compile(
            self.transformer,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=self.od_config.diffusion_compile_dynamic,
        )
        self._uses_cudagraph_trees = True

    def encode_prompt(self, *args, **kwargs):
        del args, kwargs
        if self._pending_prompt_embeds is None or self._pending_negative_prompt_embeds is None:
            raise RuntimeError("Ming-Image conditioning is only available during forward.")
        return self._pending_prompt_embeds, self._pending_negative_prompt_embeds

    def prepare_latents(self, batch_size, *args, **kwargs):
        frames = self._num_frames_per_prompt
        flat = super().prepare_latents(batch_size * frames, *args, **kwargs)
        return torch.stack(flat.chunk(frames, dim=0), dim=2)

    def _encode_reference(
        self,
        reference: Any | None,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        if reference is None:
            return None
        if isinstance(reference, list):
            if len(reference) != 1:
                raise ValueError("Ming-Image currently accepts exactly one reference image.")
            reference = reference[0]
        if isinstance(reference, Image.Image):
            reference = reference.convert("RGBA")
        if not isinstance(reference, torch.Tensor):
            reference = self.image_processor.preprocess(reference, height=height, width=width)
        if reference.ndim != 4 or reference.shape[1] != 4:
            raise ValueError("Ming-Image reference must be a batched RGBA tensor or an RGBA-compatible image.")
        reference = reference.to(device=self.device, dtype=self.vae.dtype).unsqueeze(2)
        latent = self.vae.encode(reference).latent_dist.mode()
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @staticmethod
    def _get_prompt_extra(req: DiffusionRequestBatch) -> dict[str, Any]:
        if not req.prompts:
            return {}
        prompt = req.prompts[0]
        if isinstance(prompt, dict):
            return dict(prompt.get("extra") or {})
        if hasattr(prompt, "_asdict"):
            return dict(prompt._asdict().get("extra") or {})
        return {}

    def _configure_output_frames(
        self,
        *,
        reference: Any | None,
        num_layers: int,
        is_dummy_run: bool,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.is_layer_decomposition:
            if reference is None and not is_dummy_run:
                raise ValueError("Ming-Image Design-Layer requires a reference image.")
            self._num_frames_per_prompt = num_layers + 1
            return
        if num_layers != 1:
            raise ValueError("Ming-Image Design supports exactly one output frame.")
        self._num_frames_per_prompt = 1

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        sampling = req.sampling_params
        if sampling.num_outputs_per_prompt != 1:
            # TODO(yuanheng-zhao): enable after supporting batching
            raise ValueError(
                f"Ming-Image currently supports num_outputs_per_prompt=1 only, got {sampling.num_outputs_per_prompt}."
            )

        extra = self._get_prompt_extra(req)
        extra_args = sampling.extra_args or {}

        is_dummy_run = req.is_dummy_run()
        query_hidden = extra.get("query_hidden_states")
        direct_hidden = extra.get("direct_hidden_states")
        if query_hidden is None or direct_hidden is None:
            if not is_dummy_run:
                raise ValueError("Ming-Image requests require query and direct conditions.")
            logger.warning("Ming-Image conditions are absent during warmup; using zero tensors.")
            query_hidden = torch.zeros((256, 2048), device=self.device, dtype=self.od_config.dtype)
            direct_hidden = torch.zeros((1, 6144), device=self.device, dtype=self.od_config.dtype)
        if not isinstance(query_hidden, torch.Tensor) or not isinstance(direct_hidden, torch.Tensor):
            raise TypeError("Ming-Image query and direct conditions must be tensors.")
        if query_hidden.ndim == 2:
            query_hidden = query_hidden.unsqueeze(0)
        if direct_hidden.ndim == 2:
            direct_hidden = direct_hidden.unsqueeze(0)
        query_hidden = query_hidden.to(device=self.device, dtype=self.od_config.dtype)
        direct_hidden = direct_hidden.to(device=self.device, dtype=self.od_config.dtype)
        cap_feats, direct_condition = self.conditioning(query_hidden, direct_hidden)

        reference = extra.get("reference_image")
        num_layers = int(extra_args.get("num_layers", extra.get("num_layers", 1)))
        self._configure_output_frames(
            reference=reference,
            num_layers=num_layers,
            is_dummy_run=is_dummy_run,
        )

        height = int(extra_args.get("height") or sampling.height or 1024)
        width = int(extra_args.get("width") or sampling.width or 1024)
        steps = int(sampling.num_inference_steps or self.default_num_inference_steps)
        cfg = float(sampling.guidance_scale if sampling.guidance_scale is not None else self.default_guidance_scale)
        seed = extra_args.get("seed", sampling.seed)
        generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else sampling.generator

        ref_latent = self._encode_reference(reference, height, width)
        positive = [item for item in cap_feats]
        negative = [torch.zeros_like(item) for item in positive]
        self._pending_prompt_embeds = positive
        self._pending_negative_prompt_embeds = negative

        apply_cfg = cfg > 1
        context_direct = (
            torch.cat([direct_condition, torch.zeros_like(direct_condition)], dim=0) if apply_cfg else direct_condition
        )
        context_ref = ref_latent
        if apply_cfg and context_ref is not None:
            context_ref = context_ref.repeat(2, 1, 1, 1, 1)

        inner_sampling = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            output_type="latent",
        )
        inner_req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt={"prompt": ""},
                    sampling_params=inner_sampling,
                    request_id=req.request_id or "ming-image",
                )
            ]
        )

        set_forward_context_ref_latent(context_ref)
        set_forward_context_direct_condition(context_direct)
        try:
            latent_output = super().forward(inner_req)
            if not isinstance(latent_output.output, torch.Tensor):
                raise TypeError("Ming-Image denoising must return latent tensors.")
            image = self._decode_latent_frames(latent_output.output)
            return DiffusionOutput(
                output=image,
                stage_durations=latent_output.stage_durations,
            )
        finally:
            set_forward_context_ref_latent(None)
            set_forward_context_direct_condition(None)
            self._pending_prompt_embeds = None
            self._pending_negative_prompt_embeds = None
            self._num_frames_per_prompt = 1

    @staticmethod
    def _flatten_latent_frames(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            return latents
        if latents.ndim != 5:
            raise ValueError(f"Expected 4D or 5D Ming-Image latents, got {tuple(latents.shape)}")
        batch, channels, frames, height, width = latents.shape
        return latents.permute(2, 0, 1, 3, 4).reshape(
            frames * batch,
            channels,
            height,
            width,
        )

    def _decode_latent_frames(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self._flatten_latent_frames(latents).to(self.vae.dtype).unsqueeze(2)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        if image.ndim == 5:
            if image.shape[2] != 1:
                raise ValueError(
                    f"Ming-Image VAE returned multiple decoded frames for a single latent frame: {tuple(image.shape)}"
                )
            image = image.squeeze(2)
        return image

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)
        loaded |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded |= {f"conditioning.{name}" for name, _ in self.conditioning.named_parameters()}
        return loaded


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=False)

    def post_process(images: torch.Tensor):
        if images.ndim == 5:
            images = images.permute(0, 2, 1, 3, 4).flatten(0, 1)
        return image_processor.postprocess(images.float())

    return post_process


__all__ = ["MingImageDiffusionPipeline", "get_ming_image_post_process_func"]
