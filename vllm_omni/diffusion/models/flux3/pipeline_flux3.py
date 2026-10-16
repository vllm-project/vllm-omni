# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any, ClassVar

import PIL.Image
import PIL.ImageOps
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoConfig, AutoProcessor
from vllm.config.vllm import get_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.cache.cachedit.config import CacheDiTAdapterConfig
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.flux3.flux3_conditioning import (
    pack_i2i_image_condition,
    sample_image_latents,
)
from vllm_omni.diffusion.models.flux3.flux3_image_vae import Flux3ImageVAE
from vllm_omni.diffusion.models.flux3.flux3_positional import (
    batched_prc_txt,
    scatter_ids,
)
from vllm_omni.diffusion.models.flux3.flux3_prims_ts_config import (
    flux3_prims_ts_autotune_context,
    prims_ts_autotune_enabled,
)
from vllm_omni.diffusion.models.flux3.flux3_text_encoder import Flux3TextEncoder
from vllm_omni.diffusion.models.flux3.flux3_transformer import (
    Flux3Transformer2DModel,
)
from vllm_omni.diffusion.models.flux3.lora import load_flux3_raw_lora
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.utils.flux3_package import (
    _resolve_flux3_component_path,
    _resolve_flux3_package_selection,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)

_TextBranch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_TextConditioning = tuple[_TextBranch, _TextBranch]

_F3_PACKAGE_WEIGHTS_NAME = "model.safetensors"
_F3_PACKAGE_VAE_NAME = "ae.safetensors"
_F3_DEFAULT_TEXT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
# Precomputed entries with a text-sequence axis, zero-padded when the CFG branches'
# text windows differ; the rest are concatenated.
_TEXT_SEQ_KEYS = frozenset({"ctx_emb", "vec_ctx", "pe_ctx", "mod_ctx"})
_F3_DEFAULT_TEXT_MAX_TOKENS = 8192
_F3_DEFAULT_TEXT_PAD_MULTIPLE = 80
_F3_DEFAULT_GUIDANCE_SCALE = 3.5
_F3_DEFAULT_FLOW_SHIFT = 6.93
_F3_DEFAULT_STEPS = 50
_F3_DEFAULT_SIZE = (1024, 1024)  # width, height
_F3_DEFAULT_IMAGE_ROPE_SCALE = 0.5
_F3_DEFAULT_VAE_SCALE_FACTOR = 8
_F3_MAX_REFERENCE_AREA_SIDE = 4096


def _flux3_transformer_config_kwargs(package_config: dict[str, Any] | None) -> dict[str, Any]:
    if not package_config:
        return {}
    dit = package_config.get("dit") or {}
    in_channels = dit.get("in_channels") or {}
    kwargs: dict[str, Any] = {}
    mapping = {
        "hidden": "hidden_size",
        "heads": "num_heads",
        "depth": "depth",
        "depth_single": "depth_single_blocks",
        "mlp_ratio": "mlp_ratio",
        "theta": "theta",
        "vec_in_dim": "vec_in_dim",
        "out_modes": "out_modes",
        "flow_map": "flow_map",
    }
    for src, dst in mapping.items():
        if src in dit:
            kwargs[dst] = dit[src]
    if "axes_dim" in dit:
        kwargs["axes_dim"] = tuple(dit["axes_dim"])
    if "text" in in_channels:
        kwargs["context_in_dim"] = in_channels["text"]
    if "image" in in_channels:
        kwargs["in_channels"] = in_channels["image"]
    if "out_modes" in kwargs:
        kwargs["out_modes"] = tuple(kwargs["out_modes"])
    return kwargs


def _flux3_text_config(package_config: dict[str, Any] | None) -> dict[str, Any]:
    text_cfg = (package_config or {}).get("text_encoder") or {}
    return {
        "path": text_cfg.get("path", "text_encoder"),
        "tokenizer": text_cfg.get("tokenizer", "tokenizer"),
        "layers": list(text_cfg.get("layers", _F3_DEFAULT_TEXT_LAYERS)),
        "pad_multiple": int(text_cfg.get("pad_multiple", _F3_DEFAULT_TEXT_PAD_MULTIPLE)),
        "max_tokens": int(text_cfg.get("max_tokens", _F3_DEFAULT_TEXT_MAX_TOKENS)),
    }


def _flux3_sampler_config(package_config: dict[str, Any] | None) -> dict[str, Any]:
    sampler_cfg = (package_config or {}).get("sampler") or {}
    size = tuple(sampler_cfg.get("size", _F3_DEFAULT_SIZE))
    if len(size) != 2:
        size = _F3_DEFAULT_SIZE
    guidance = sampler_cfg.get("guidance", _F3_DEFAULT_GUIDANCE_SCALE)
    return {
        "steps": int(sampler_cfg.get("steps", _F3_DEFAULT_STEPS)),
        "guidance": None if guidance is None else float(guidance),
        "alpha": float(sampler_cfg.get("alpha", _F3_DEFAULT_FLOW_SHIFT)),
        "size": (int(size[0]), int(size[1])),
    }


def _should_autotune_flux3_prims_ts(transformer: Any) -> bool:
    adapter = getattr(transformer, "kernel_adapter", None)
    if getattr(adapter, "backend", None) != "prims_ts" or getattr(adapter, "_configured_blocks", 0) <= 0:
        return False
    return prims_ts_autotune_enabled()


def _flux3_image_rope_scale(package_config: dict[str, Any] | None) -> float:
    dit = (package_config or {}).get("dit") or {}
    return float(dit.get("image_rope_scale", _F3_DEFAULT_IMAGE_ROPE_SCALE))


def _flux3_image_vae_file(package_config: dict[str, Any] | None) -> str:
    vae_cfg = (package_config or {}).get("vae") or {}
    return vae_cfg.get("path") or _F3_PACKAGE_VAE_NAME


@contextmanager
def _enable_native_fp8_quant_for_flux3():
    """Select vLLM's native FP8 quantizer while building Flux3."""
    custom_ops = get_current_vllm_config().compilation_config.custom_ops
    marker = "+quant_fp8"
    added = marker not in custom_ops
    if added:
        custom_ops.append(marker)
    try:
        yield
    finally:
        if added:
            custom_ops.remove(marker)


def get_flux3_pre_process_func(od_config: OmniDiffusionConfig):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniTextPrompt

    selection = _resolve_flux3_package_selection(od_config.model, od_config.additional_config)
    package_config = selection.package_config
    sampler_config = _flux3_sampler_config(package_config)
    default_width, default_height = sampler_config["size"]
    align = _F3_DEFAULT_VAE_SCALE_FACTOR * 2
    image_processor = Flux3ImageProcessor(vae_scale_factor=align)

    def _prepared_pil_to_tensor(pil: PIL.Image.Image) -> torch.Tensor:
        """Normalise a prepared reference image without resizing to the output size."""
        return image_processor.preprocess(pil, height=pil.height, width=pil.width, resize_mode="crop")

    def _as_pil(image: PIL.Image.Image | str) -> PIL.Image.Image:
        return PIL.Image.open(image) if isinstance(image, str) else image

    def _reference_output_size(images: list[PIL.Image.Image]) -> tuple[int, int]:
        if not images:
            return default_height, default_width
        return Flux3ImageProcessor.default_size(
            images[0],
            default_width=default_width,
            default_height=default_height,
            align=align,
        )

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else {}
        if not multi_modal_data:
            return request
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)

        h: int | None = request.sampling_params.height
        w: int | None = request.sampling_params.width

        raw_image = multi_modal_data.get("image")
        if raw_image is not None and not isinstance(raw_image, torch.Tensor):
            if isinstance(raw_image, (str, PIL.Image.Image)):
                pil = _as_pil(raw_image)
                prepared = Flux3ImageProcessor.prepare_reference(pil, align=align)
                if h is None or w is None:
                    default_h, default_w = _reference_output_size([prepared])
                    h = h or default_h
                    w = w or default_w
                    request.sampling_params.height = h
                    request.sampling_params.width = w
                raw_image = [_prepared_pil_to_tensor(prepared)]
            elif isinstance(raw_image, (list, tuple)):
                prepared_or_tensor: list[PIL.Image.Image | torch.Tensor] = []
                prepared_refs: list[PIL.Image.Image] = []
                for img in raw_image:
                    if isinstance(img, torch.Tensor):
                        prepared_or_tensor.append(img)
                        continue
                    prepared = Flux3ImageProcessor.prepare_reference(_as_pil(img), align=align)
                    prepared_or_tensor.append(prepared)
                    prepared_refs.append(prepared)
                if prepared_refs and (h is None or w is None):
                    default_h, default_w = _reference_output_size(prepared_refs)
                    h = h or default_h
                    w = w or default_w
                    request.sampling_params.height = h
                    request.sampling_params.width = w
                raw_image = [
                    (img if isinstance(img, torch.Tensor) else _prepared_pil_to_tensor(img))
                    for img in prepared_or_tensor
                ]
            prompt["multi_modal_data"]["image"] = raw_image

        unsupported_modalities = sorted(set(multi_modal_data) - {"image"})
        if unsupported_modalities:
            raise ValueError(
                "Flux3Pipeline supports only image multi_modal_data; got: " + ", ".join(unsupported_modalities)
            )

        request.prompt = prompt
        return request

    return pre_process_func


def get_flux3_post_process_func(od_config: OmniDiffusionConfig):
    image_processor = Flux3ImageProcessor(vae_scale_factor=_F3_DEFAULT_VAE_SCALE_FACTOR * 2)

    def post_process_func(decoded: dict):
        if "image" not in decoded:
            raise ValueError("Flux3Pipeline expected an image tensor")
        return {"image": image_processor.postprocess(decoded["image"])}

    return post_process_func


def _pad_seq(t: torch.Tensor, length: int) -> torch.Tensor:
    """Zero-pad dim 1 (the text sequence) to ``length``; broadcast rows stay as they are."""
    if t.shape[1] in (1, length):
        return t
    pad = t.new_zeros((t.shape[0], length - t.shape[1], *t.shape[2:]))
    return torch.cat([t, pad], dim=1)


def _merge_padded_cfg(neg: dict, pos: dict, text_len: int) -> dict:
    """Concatenate per-branch precomputed conditioning into one B=2 batch."""

    def merge(key: str, a: Any, b: Any) -> Any:
        if isinstance(a, dict):
            return {k: merge(k, a[k], b[k]) for k in a}
        if isinstance(a, tuple):
            return tuple(merge(key, x, y) for x, y in zip(a, b, strict=True))
        if key in _TEXT_SEQ_KEYS:
            a, b = _pad_seq(a, text_len), _pad_seq(b, text_len)
        return torch.cat([a, b], dim=0)

    # Tuple keys are the transformer's per-request step caches, rebuilt for the batch.
    return {k: merge(k, neg[k], pos[k]) for k in pos if not isinstance(k, tuple)}


class Flux3ImageProcessor(VaeImageProcessor):
    """Image processor to preprocess the conditioning image for Flux3."""

    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_convert_rgb=do_convert_rgb,
        )

    @staticmethod
    def prepare_reference(image: PIL.Image.Image, *, align: int = 16) -> PIL.Image.Image:
        """Orient, convert to RGB, limit area, and center-crop to the VAE grid."""
        Flux3ImageProcessor.check_image_input(image, min_side_length=align, max_area=_F3_MAX_REFERENCE_AREA_SIDE**2)
        image = PIL.ImageOps.exif_transpose(image).convert("RGB")
        scale = min(1.0, _F3_MAX_REFERENCE_AREA_SIDE / math.sqrt(image.width * image.height))
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                PIL.Image.Resampling.LANCZOS,
            )
        width = (image.width // align) * align
        height = (image.height // align) * align
        if width <= 0 or height <= 0:
            raise ValueError(f"Image too small after {align}px alignment: {image.width}x{image.height}")
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        return image.crop((left, top, left + width, top + height))

    @staticmethod
    def default_size(
        reference: PIL.Image.Image,
        *,
        default_width: int,
        default_height: int,
        align: int = 16,
    ) -> tuple[int, int]:
        """Return ``(height, width)`` preserving the first reference aspect at default area."""
        scale = math.sqrt((default_width * default_height) / (reference.width * reference.height))

        def grid(value: int) -> int:
            return max(align, round(value * scale / align) * align)

        return grid(reference.height), grid(reference.width)

    @staticmethod
    def check_image_input(
        image: PIL.Image.Image,
        max_aspect_ratio: int = 8,
        min_side_length: int = 64,
        max_area: int = 1024 * 1024,
    ) -> PIL.Image.Image:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")

        width, height = image.size
        if width < min_side_length or height < min_side_length:
            raise ValueError(f"Image too small: {width}x{height}. Both dimensions must be at least {min_side_length}px")

        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            raise ValueError(
                f"Aspect ratio too extreme: {width}x{height} (ratio: {aspect_ratio:.1f}:1). "
                f"Maximum allowed ratio is {max_aspect_ratio}:1"
            )

        if width * height > max_area:
            logger.warning("Image area exceeds recommended maximum; resizing will be applied.")

        return image


class Flux3Pipeline(
    nn.Module,
    CFGParallelMixin,
    SupportImageInput,
    DiffusionPipelineProfilerMixin,
    ProgressBarMixin,
    SupportsComponentDiscovery,
):
    """Flux3 pipeline for text-to-image and image-to-image generation."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        batch_cfg = od_config.additional_config.get("batch_cfg")
        if batch_cfg is not None and not isinstance(batch_cfg, bool):
            raise TypeError(f"additional_config['batch_cfg'] must be a bool or None, got {type(batch_cfg).__name__}")
        cfg_parallel_size = int(getattr(od_config.parallel_config, "cfg_parallel_size", 1) or 1)
        # Fuse negative and positive conditioning by default when both branches
        # execute on the same rank. CFG parallelism requires one branch per rank.
        self.batch_cfg = cfg_parallel_size == 1 if batch_cfg is None else batch_cfg
        if cfg_parallel_size > 1 and self.batch_cfg:
            logger.warning("Disabling Flux3 batch CFG because cfg_parallel_size=%d", cfg_parallel_size)
            self.batch_cfg = False

        self._execution_device = get_local_device()
        model = od_config.model
        if not model:
            raise ValueError("Flux3 requires a model path")
        selection = _resolve_flux3_package_selection(
            model,
            od_config.additional_config,
        )
        package_config = selection.package_config
        text_config = _flux3_text_config(package_config)
        sampler_config = _flux3_sampler_config(package_config)
        shared_root = selection.shared_root
        transformer_root = selection.transformer_root
        text_encoder_root, text_encoder_subfolder = _resolve_flux3_component_path(shared_root, text_config["path"])
        tokenizer_root, tokenizer_subfolder = _resolve_flux3_component_path(shared_root, text_config["tokenizer"])
        self._default_num_inference_steps = sampler_config["steps"]
        self._default_guidance_scale = sampler_config["guidance"]
        self._flow_map = bool((package_config.get("dit") or {}).get("flow_map", False))
        if self._flow_map and self._default_guidance_scale is not None:
            raise ValueError("A flow-map Flux3 variant must set sampler.guidance to null")
        self._default_flow_shift = sampler_config["alpha"]
        self._default_size = sampler_config["size"]
        self._image_rope_scale = _flux3_image_rope_scale(package_config)
        vae_root, image_vae_file = _resolve_flux3_component_path(shared_root, _flux3_image_vae_file(package_config))

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=transformer_root,
                subfolder=None,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
                allow_patterns_overrides=[_F3_PACKAGE_WEIGHTS_NAME],
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=vae_root,
                subfolder=None,
                revision=od_config.revision,
                prefix="vae._model.",
                fall_back_to_pt=True,
                allow_patterns_overrides=[image_vae_file],
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=text_encoder_root,
                subfolder=text_encoder_subfolder,
                revision=od_config.revision,
                prefix="text_encoder.text_encoder.",
                fall_back_to_pt=True,
            ),
        ]

        # Construct every tensor-bearing component without loading its checkpoint.
        # Non-tensor assets remain component-specific: Transformers builds the
        # processor/config metadata.
        self.vae = Flux3ImageVAE(device=self._execution_device)

        transformer_kwargs = _flux3_transformer_config_kwargs(package_config)
        transformer_kwargs.update(get_transformer_config_kwargs(od_config.tf_model_config, Flux3Transformer2DModel))
        with _enable_native_fp8_quant_for_flux3():
            self.transformer = Flux3Transformer2DModel(
                quant_config=od_config.quantization_config,
                od_config=od_config,
                **transformer_kwargs,
            )
        cache_dit_adapter = getattr(self.transformer, "_cache_dit_adapter_config", None)
        if isinstance(cache_dit_adapter, CacheDiTAdapterConfig):
            # Cache-DiT counts one forward per step for fused B=2 CFG and two
            # forwards per step for split CFG.
            self.transformer._cache_dit_adapter_config = replace(
                cache_dit_adapter,
                has_separate_cfg=not self.batch_cfg,
            )

        self.processor = AutoProcessor.from_pretrained(
            tokenizer_root,
            subfolder=tokenizer_subfolder,
            local_files_only=os.path.isdir(tokenizer_root),
        )
        text_encoder_config = AutoConfig.from_pretrained(
            text_encoder_root,
            subfolder=text_encoder_subfolder,
            local_files_only=os.path.isdir(text_encoder_root),
        )
        self.text_encoder = Flux3TextEncoder.from_config(
            text_encoder_config,
            processor=self.processor,
            context_in_dim=self.transformer.config.context_in_dim,
            device=self._execution_device,
            vec_in_dim=self.transformer.config.vec_in_dim,
            output_layer=text_config["layers"],
            max_sequence_length=text_config["max_tokens"],
            pad_multiple=text_config["pad_multiple"],
            torch_dtype=od_config.dtype,
        )

        self.image_vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_channels = self.vae.config.latent_channels if hasattr(self.vae, "config") else 16
        self.image_processor = Flux3ImageProcessor(vae_scale_factor=self.image_vae_scale_factor * 2)
        self.tokenizer_max_length = text_config["max_tokens"]
        self.default_sample_size = 128

        # CFG scale, set per-request in forward(); read by CFGParallelMixin.
        self._guidance_scale = None
        self._raw_lora_adapter_ids: set[int] = set()
        # Scheduler is rebuilt only when (num_inference_steps, flow_shift) changes.
        self._scheduler_key: tuple[int, float] | None = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _flux3_lora_weights_mapper(self) -> Any | None:
        """Return the first available Flux3 transformer LoRA key mapper."""
        for name in ("transformer", "transformer_2", "dit"):
            component = getattr(self, name, None)
            mapper = getattr(component, "lora_weights_mapper", None)
            if mapper is not None:
                return mapper
        return None

    def _load_diffusion_lora_adapter(
        self,
        *,
        lora_request: LoRARequest,
        lora_path: str,
        dtype: torch.dtype,
    ) -> tuple[LoRAModel, PEFTHelper] | None:
        """Load Flux3 raw adapters while leaving PEFT adapters to the shared manager."""
        self._raw_lora_adapter_ids.discard(lora_request.lora_int_id)
        loaded = load_flux3_raw_lora(
            lora_request=lora_request,
            lora_path=lora_path,
            dtype=dtype,
            weights_mapper=self._flux3_lora_weights_mapper(),
        )
        if loaded is not None:
            self._raw_lora_adapter_ids.add(lora_request.lora_int_id)
        return loaded

    def _validate_diffusion_lora_binding(
        self,
        *,
        lora_model: LoRAModel,
        bound_lora_names: frozenset[str],
    ) -> None:
        """Require every raw Flux3 LoRA tensor to bind to a pipeline layer."""
        if lora_model.id not in self._raw_lora_adapter_ids:
            return
        missing = sorted(set(lora_model.loras) - bound_lora_names)
        if missing:
            raise ValueError(
                "Flux3 raw LoRA binding is incomplete: "
                f"bound={len(bound_lora_names)}/{len(lora_model.loras)}, missing={missing[:5]}"
            )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # ── Generation ──

    def _text_conditioning(
        self,
        prompt: str | list[str] | None,
        device: torch.device,
        negative_prompt: str = "",
    ) -> _TextConditioning:
        """Encode negative/positive text separately.

        Each CFG branch is padded to its own token window before entering the
        DiT. Separate encoding preserves those lengths when the prompts round
        to different token counts.
        """
        pos = prompt if isinstance(prompt, str) else (prompt[0] if prompt else "")
        return self._encode_text_branch(negative_prompt, device), self._encode_text_branch(pos, device)

    def _encode_text_branch(self, prompt: str, device: torch.device) -> _TextBranch:
        ctx, vector = self.text_encoder.encode(prompt)
        _, ctx_ids = batched_prc_txt(ctx)
        timesteps_ctx = torch.zeros(ctx.shape[:2], device=device, dtype=ctx.dtype)
        return ctx.to(device), ctx_ids.to(device), vector.to(device), timesteps_ctx

    def _decode_image(self, grid: torch.Tensor) -> torch.Tensor:
        """Decode a packed image latent grid ``(1, 128, h, w)`` to pixels in [-1, 1]."""
        return self.vae.decode(grid)  # (1, 3, 16h, 16w) in [-1, 1]

    def _encode_cond_images(
        self, image: torch.Tensor | list[torch.Tensor] | None, device: torch.device
    ) -> list[torch.Tensor]:
        """VAE-encode one or more conditioning images to packed latents ``(1, 128, h, w)``."""
        if image is None:
            raise ValueError("image-to-image generation requires a conditioning image")
        refs = list(image) if isinstance(image, (list, tuple)) else [image]
        vae_dtype = next(self.vae._model.parameters()).dtype
        return [self.vae.encode(r.to(device=device, dtype=vae_dtype)) for r in refs]

    # ── CFGParallelMixin interface ──

    def predict_noise(self, **kwargs) -> torch.Tensor:
        """Forward the transformer and extract the relevant output stream(s)."""
        result = self.transformer(**kwargs)
        return result["x_image"]

    def predict_noise_maybe_with_cfg(
        self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs=None, **kwargs
    ):
        """CFG forward pass — fuses negative and positive inputs into a single B=2 forward pass.

        Note: the fused B=2 path and two separate B=1 passes produce different
        bfloat16 matmul accumulation, so this choice affects numerical output.

        Falls through to the mixin's two-pass path when CFG-parallel is active,
        since each rank must own exactly one branch.
        """
        batch_cfg = kwargs.pop("batch_cfg", self.batch_cfg)
        if do_true_cfg and batch_cfg and negative_kwargs is not None:
            try:
                from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size

                cfg_parallel = get_classifier_free_guidance_world_size() > 1
            except (AssertionError, ImportError):
                cfg_parallel = False
            if not cfg_parallel:
                fused = {}
                for k, v in positive_kwargs.items():
                    neg_v = negative_kwargs.get(k)
                    if isinstance(v, torch.Tensor) and isinstance(neg_v, torch.Tensor):
                        fused[k] = torch.cat([neg_v, v], dim=0)
                    else:
                        fused[k] = v
                pred = self.predict_noise(**fused)
                neg, pos = pred.chunk(2)
                return self.combine_cfg_noise(pos, neg, true_cfg_scale)
        return super().predict_noise_maybe_with_cfg(
            do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize=False, **kwargs
        )

    def diffuse(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        text_conditioning: _TextConditioning,
        guidance_scale: float,
        cond_kwargs: dict | None = None,
        pag_single_layers: list[int] | None = None,
        pag_strength: float = 1.0,
    ) -> torch.Tensor:
        """Rectified-flow Euler loop using the CFGParallelMixin interface.

        ``text_conditioning`` contains separate [uncond, cond] branches, fused into
        a B=2 forward unless ``batch_cfg`` is off. Branches of different text length
        keep their prompt-local padding: the shorter one is zero-padded and attends
        only to its own tokens.
        """
        compute_dtype = self.od_config.dtype
        latent_key = "x_image"
        ids_key = "x_image_ids"
        ts_key = "x_image_timesteps"

        negative_text, positive_text = text_conditioning
        ctx_neg, ctx_ids_neg, vector_neg, timesteps_ctx_neg = negative_text
        ctx_pos, ctx_ids_pos, vector_pos, timesteps_ctx_pos = positive_text
        text_len_neg, text_len_pos = ctx_neg.shape[1], ctx_pos.shape[1]
        # Branches whose text windows differ (e.g. a long prompt and an empty negative)
        # still share one B=2 forward: the shorter text is zero-padded and each sample
        # attends only to its own tokens, matching two B=1 forwards.
        # Cache-DiT's two-stream wrapper has no per-sample attention lengths, so it keeps
        # two B=1 forwards for unequal lengths.
        cache_dit = getattr(self.transformer, "_is_cached", False)
        pad_cfg_text = self.batch_cfg and text_len_neg != text_len_pos and not cache_dit
        batch_cfg = self.batch_cfg and (text_len_neg == text_len_pos or pad_cfg_text)

        # Precompute text and positional encodings that are constant across all denoising steps.
        # ids_by_mod collects all latent positional ID tensors (fixed geometry, never change).
        ids_by_mod: dict[str, torch.Tensor] = {latent_key: latent_ids}
        if cond_kwargs:
            ids_by_mod.update({k[:-4]: v for k, v in cond_kwargs.items() if k.endswith("_ids")})

        # Conditioning streams are fixed for the full denoising loop. Cache
        # their embedding and Phase 1 output alongside the text conditioning;
        # generated image streams remain step-dependent.
        static_content: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if cond_kwargs:
            for key, value in cond_kwargs.items():
                if key.endswith("_cond"):
                    static_content[key] = (value, cond_kwargs[f"{key}_timesteps"])

        if pad_cfg_text:
            # Fused B=2 forward over text padded to the longer branch. Static text
            # streams are computed per branch at their own length, then padded.
            text_len = max(text_len_neg, text_len_pos)
            precomputed_pos = _merge_padded_cfg(
                *(
                    self.transformer.precompute_static_conditioning(
                        c, i, v, t, ids_by_mod=ids_by_mod, static_content=static_content
                    )
                    for c, i, v, t in (negative_text, positive_text)
                ),
                text_len,
            )
            precomputed_pos["text_lens"] = (text_len_neg, text_len_pos)
            precomputed_neg = None
            ctx_neg, ctx_ids_neg, timesteps_ctx_neg, ctx_pos, ctx_ids_pos, timesteps_ctx_pos = (
                _pad_seq(t, text_len)
                for t in (ctx_neg, ctx_ids_neg, timesteps_ctx_neg, ctx_pos, ctx_ids_pos, timesteps_ctx_pos)
            )
        elif batch_cfg:
            # Single fused B=2 forward — precompute once for the full (neg+pos) batch.
            # Latent IDs are fixed geometry: repeat to B=2 so pe_by_mod matches the doubled latents.
            ctx = torch.cat([ctx_neg, ctx_pos], dim=0)
            ctx_ids = torch.cat([ctx_ids_neg, ctx_ids_pos], dim=0)
            vector = torch.cat([vector_neg, vector_pos], dim=0)
            timesteps_ctx = torch.cat([timesteps_ctx_neg, timesteps_ctx_pos], dim=0)
            ids_by_mod_b2 = {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in ids_by_mod.items()}
            static_content_b2 = {
                k: tuple(v.repeat(2, *([1] * (v.ndim - 1))) for v in values) for k, values in static_content.items()
            }
            precomputed_full = self.transformer.precompute_static_conditioning(
                ctx,
                ctx_ids,
                vector,
                timesteps_ctx,
                ids_by_mod=ids_by_mod_b2,
                static_content=static_content_b2,
            )
            precomputed_pos = precomputed_full
            precomputed_neg = None  # not used: batch_cfg merges both into positive_kwargs
        else:
            # Two separate B=1 forwards — precompute independently for neg and pos.
            precomputed_neg = self.transformer.precompute_static_conditioning(
                ctx_neg,
                ctx_ids_neg,
                vector_neg,
                timesteps_ctx_neg,
                ids_by_mod=ids_by_mod,
                static_content=static_content,
            )
            precomputed_pos = self.transformer.precompute_static_conditioning(
                ctx_pos,
                ctx_ids_pos,
                vector_pos,
                timesteps_ctx_pos,
                ids_by_mod=ids_by_mod,
                static_content=static_content,
            )

        stable_pos = dict(
            ctx=ctx_pos,
            ctx_ids=ctx_ids_pos,
            vector=vector_pos,
            timesteps_ctx=timesteps_ctx_pos,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
            precomputed=precomputed_pos,
            **(cond_kwargs or {}),
        )
        stable_neg = dict(
            ctx=ctx_neg,
            ctx_ids=ctx_ids_neg,
            vector=vector_neg,
            timesteps_ctx=timesteps_ctx_neg,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
            precomputed=precomputed_neg,
            **(cond_kwargs or {}),
        )
        with self.progress_bar(total=len(timesteps) - 1) as pbar:
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=True):
                x_in = latents.to(compute_dtype)
                ts = t_curr * torch.ones(x_in.shape[:2], device=x_in.device, dtype=x_in.dtype)
                positive_kwargs: dict[str, Any] = {
                    latent_key: x_in,
                    ids_key: latent_ids,
                    ts_key: ts,
                    **stable_pos,
                }
                negative_kwargs: dict[str, Any] = {
                    latent_key: x_in,
                    ids_key: latent_ids,
                    ts_key: ts,
                    **stable_neg,
                }
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=True,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    batch_cfg=batch_cfg,
                )
                latents = latents + (t_prev - t_curr) * noise_pred
                pbar.update()

        return latents

    def _diffuse_flow_map(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        *,
        text: _TextBranch,
        timesteps: torch.Tensor,
        cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run single-branch Euler sampling with the target time embedded by the DiT."""
        ctx, ctx_ids, vector, timesteps_ctx = text
        compute_dtype = self.od_config.dtype
        with self.progress_bar(total=len(timesteps) - 1) as pbar:
            for t_curr, t_next in zip(timesteps[:-1], timesteps[1:], strict=True):
                x_in = latents.to(compute_dtype)
                ts = t_curr.to(device=x_in.device, dtype=x_in.dtype).expand(x_in.shape[:2])
                ts_ref = t_next.to(device=x_in.device, dtype=x_in.dtype).expand(x_in.shape[0])
                prediction = self.predict_noise(
                    x_image=x_in,
                    x_image_ids=latent_ids,
                    x_image_timesteps=ts,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    vector=vector,
                    timesteps_ctx=timesteps_ctx,
                    timesteps_ref=ts_ref,
                    **(cond_kwargs or {}),
                )
                latents = latents + (t_next - t_curr) * prediction
                pbar.update()
        return latents

    def generate(
        self,
        *,
        prompt: str | list[str] | None,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float | None,
        seed: int,
        negative_prompt: str,
        flow_shift: float,
        image: torch.Tensor | list[torch.Tensor] | None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run generation with optional image conditioning; returns raw image tensors.

        Runs text encoding, scheduling, seeded latent sampling, denoising,
        spatial scattering, and VAE decoding.
        Pixel-space postprocessing (PIL conversion) is handled by the registered
        post_process_func.
        """
        device = device or self._execution_device
        if self._flow_map:
            if guidance_scale is not None:
                raise ValueError("The sd variant does not support guidance_scale")
            if negative_prompt:
                raise ValueError("The sd variant does not support negative_prompt")
            prompt_text = prompt if isinstance(prompt, str) else (prompt or [""])[0]
            text_conditioning = self._encode_text_branch(prompt_text, device)
        else:
            if guidance_scale is None:
                raise ValueError("Flux3 requires a guidance scale for this variant")
            text_conditioning = self._text_conditioning(prompt, device, negative_prompt=negative_prompt)
        sched_key = (num_inference_steps, flow_shift)
        if self._scheduler_key != sched_key:
            self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_inference_steps, shift=flow_shift)
            self._scheduler_key = sched_key
        timesteps = torch.cat([self.scheduler.sigmas, self.scheduler.sigmas.new_zeros(1)]).to(device)

        decoded: dict[str, torch.Tensor] = {}
        img_cond_kwargs: dict[str, torch.Tensor] = {}
        if image is not None:
            cond_latents = self._encode_cond_images(image, device)
            xc, xc_ids, xc_ts = pack_i2i_image_condition(
                [lat[0] for lat in cond_latents],
                device=device,
                image_rope_scale=self._image_rope_scale,
            )
            img_cond_kwargs = dict(x_image_cond=xc, x_image_cond_ids=xc_ids, x_image_cond_timesteps=xc_ts)
        x, x_ids, orig_ids = sample_image_latents(
            height=height,
            width=width,
            seed=seed,
            device=device,
            image_rope_scale=self._image_rope_scale,
        )
        if self._flow_map:
            final = self._diffuse_flow_map(
                x,
                x_ids,
                text=text_conditioning,
                timesteps=timesteps,
                cond_kwargs=img_cond_kwargs,
            )
        else:
            final = self.diffuse(
                x,
                x_ids,
                timesteps=timesteps,
                text_conditioning=text_conditioning,
                guidance_scale=guidance_scale,
                cond_kwargs=img_cond_kwargs,
            )
        grid = scatter_ids(final.to(torch.bfloat16), orig_ids)[0]
        decoded["image"] = self._decode_image(grid)
        return decoded

    def forward(
        self,
        req: DiffusionRequestBatch,
        *,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        height: int | None = None,
        width: int | None = None,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Parse the request and generate with optional image conditioning."""
        if len(req.prompts) > 1:
            logger.warning(
                "This model only supports a single prompt, not a batched request. Taking only the first prompt for now."
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        # Guard against a double-nested prompt dict (e.g. OmniTextPrompt wrapping a
        # {"prompt": str, "negative_prompt": str} dict passed as the "prompt" value).
        if isinstance(prompt, dict):
            negative_prompt = prompt.get("negative_prompt") or ""
            prompt = prompt.get("prompt") or ""
        else:
            negative_prompt = "" if isinstance(first_prompt, str) else (first_prompt.get("negative_prompt") or "")
        multi_modal_data = {} if isinstance(first_prompt, str) else (first_prompt.get("multi_modal_data") or {})

        sp = req.sampling_params

        # Conditioning tensors are produced by get_flux3_pre_process_func before forward() runs.
        image = multi_modal_data.get("image")  # None | list[Tensor] | Tensor
        unsupported_modalities = sorted(set(multi_modal_data) - {"image"})
        if unsupported_modalities:
            raise ValueError(
                "Flux3Pipeline supports only image multi_modal_data; got: " + ", ".join(unsupported_modalities)
            )
        default_width, default_height = self._default_size
        flow_shift = float(sp.extra_args.get("flow_shift", self.od_config.flow_shift or self._default_flow_shift))
        height = sp.height or height or default_height
        width = sp.width or width or default_width
        num_inference_steps = (
            sp.num_inference_steps
            if sp.num_inference_steps is not None
            else (num_inference_steps if num_inference_steps is not None else self._default_num_inference_steps)
        )
        # The generic engine warmup injects guidance_scale=0.0 even for models
        # that do not use CFG. Flow-map sampling has no guidance parameter, so
        # ignore that framework-only value for the dummy request.
        is_dummy_flow_map_warmup = self._flow_map and req.is_dummy_run()
        if sp.guidance_scale_provided and not is_dummy_flow_map_warmup:
            guidance_scale = sp.guidance_scale
            if self._flow_map:
                raise ValueError("The sd variant does not support guidance_scale")
        if guidance_scale is None:
            guidance_scale = self._default_guidance_scale
        if self._flow_map and negative_prompt:
            raise ValueError("The sd variant does not support negative_prompt")

        seed = sp.seed
        if seed is None:
            gen = sp.generator or generator
            if isinstance(gen, (list, tuple)):
                gen = gen[0] if gen else None
            seed = gen.initial_seed() if isinstance(gen, torch.Generator) else 0

        self._guidance_scale = guidance_scale

        tune_context = (
            flux3_prims_ts_autotune_context(
                self.od_config.model,
                tune_mode=req.is_dummy_run(),
            )
            if _should_autotune_flux3_prims_ts(getattr(self, "transformer", None))
            else nullcontext()
        )
        with tune_context:
            decoded = self.generate(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=int(seed),
                image=image,
                negative_prompt=negative_prompt,
                flow_shift=flow_shift,
            )
        return DiffusionOutput(
            output=decoded, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        self.text_encoder.finalize_after_loading()
        return loaded_weights

    def remap_checkpoint_key(self, name: str) -> str | None:
        """Let checkpoint adapters remap pipeline-prefixed transformer weight names."""
        transformer_mapper = getattr(self.transformer, "remap_checkpoint_key", None)
        if callable(transformer_mapper) and name.startswith("transformer."):
            return transformer_mapper(name)
        return None

    def setup_compile(self) -> None:
        """Compile the transformer and the complete tensor-only text forward."""
        dynamic = self.od_config.diffusion_compile_dynamic
        if self.od_config.diffusion_compile_granularity == "full":
            self.transformer.compile(dynamic=dynamic)
        else:
            self.transformer = regionally_compile(self.transformer, dynamic=dynamic)
        self.text_encoder.setup_compile(dynamic=dynamic)
