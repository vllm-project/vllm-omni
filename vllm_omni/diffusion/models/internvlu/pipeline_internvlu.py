# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Stage-1 pipeline for InternVL-U image generation and editing."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.stage_input_processors.internvlu import (
    CFG_ROLES,
    CONDITIONAL_ROLE,
    PARTIAL_ROLE,
    UNCONDITIONAL_ROLE,
)

from .internvlu_transformer import InternVLUTransformer2DModel

logger = init_logger(__name__)

_EXPECTED_DECODER_CONFIG = {
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 20,
    "attention_head_dim": 128,
    "num_attention_heads": 12,
    "axes_dims_rope": [16, 56, 56],
    "video_position_scale_factor": 1.0,
    "vlm_cond_position_scale_factor": 0.25,
}


def get_internvlu_post_process_func(od_config: OmniDiffusionConfig):
    """InternVL-U returns the final PIL image directly."""
    del od_config

    def post_process_func(output):
        return output

    return post_process_func


def _component_config_path(
    model: str,
    revision: str | None,
    component: str,
) -> str:
    filename = f"{component}/config.json"
    if os.path.isdir(model):
        path = os.path.join(model, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing InternVL-U component config: {path}")
        return path
    return hf_hub_download(model, filename=filename, revision=revision)


def _load_generation_config(model: str, revision: str | None) -> dict[str, Any]:
    with open(
        _component_config_path(model, revision, "generation_decoder"),
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)
    _validate_generation_config(config)
    return config


def _validate_generation_config(config: dict[str, Any]) -> None:
    expected_wrapper = {
        "decoder_projector_type": "identity",
        "padding_encoder_hidden_states": True,
        "pad_to_fix_length": False,
        "max_sequence_length": 768,
        "vae_downsample_factor": 8,
    }
    for key, expected in expected_wrapper.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(
                f"unsupported InternVL-U generation config {key}={actual!r}; checkpoint contract requires {expected!r}"
            )
    decoder_config = config.get("decoder_config")
    if not isinstance(decoder_config, dict):
        raise ValueError("generation_decoder config is missing decoder_config")
    for key, expected in _EXPECTED_DECODER_CONFIG.items():
        actual = decoder_config.get(key)
        if actual != expected:
            raise ValueError(
                f"unsupported InternVL-U decoder config {key}={actual!r}; checkpoint contract requires {expected!r}"
            )

    # The conditioning width is checkpoint configuration, not a pipeline
    # constant: Stage 1 sizes every conditioning shape check and dummy
    # tensor from input_hidden_size, so the three declarations of the width
    # must agree.
    hidden_size = config.get("input_hidden_size")
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
        raise ValueError(f"generation_decoder input_hidden_size must be a positive integer, got {hidden_size!r}")
    output_hidden_size = config.get("output_hidden_size")
    if output_hidden_size != hidden_size:
        raise ValueError(
            f"generation_decoder output_hidden_size={output_hidden_size!r} must match input_hidden_size={hidden_size}"
        )
    joint_attention_dim = decoder_config.get("joint_attention_dim")
    if joint_attention_dim != hidden_size:
        raise ValueError(
            f"generation_decoder joint_attention_dim={joint_attention_dim!r} must match input_hidden_size={hidden_size}"
        )


def _retrieve_mode(encoder_output: Any) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Qwen-Image VAE encoder output has no latent distribution")


def _coerce_positive_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _as_pil_image(image: Any) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        return PIL.Image.open(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().float().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(
            f"reference images must be PIL images, paths, numpy arrays, or tensors; got {type(image).__name__}"
        )
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unsupported reference image shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("floating reference images must contain values in [0, 1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return PIL.Image.fromarray(array).convert("RGB")


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 16,
    min_pixels: int = 4096 * 8 * 8,
    max_pixels: int = 16384 * 8 * 8,
    max_length: int = 1024,
) -> tuple[int, int]:
    """Released dynamic generation-image resize heuristic."""
    if height < factor or width < factor:
        raise ValueError(f"reference image dimensions must both be at least {factor}, got {(height, width)}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("reference image aspect ratio must be at most 200")
    if max(height, width) > max_length:
        scale = max_length / max(height, width)
        height = int(height * scale)
        width = int(width * scale)

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    if max(resized_height, resized_width) > max_length:
        scale = max_length / max(resized_height, resized_width)
        resized_height = math.floor(resized_height * scale / factor) * factor
        resized_width = math.floor(resized_width * scale / factor) * factor
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError("reference image resize produced an empty dimension")
    return resized_height, resized_width


def _preprocess_reference_images(
    images: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize references independently, then pad them for batched VAE input.

    The returned grids describe each image's valid VAE-latent extent. Padding
    is deliberately absent from those grids so it cannot become conditioning.
    """
    pil_images = [_as_pil_image(image) for image in images]
    if not pil_images:
        raise ValueError("at least one reference image is required")

    tensors: list[torch.Tensor] = []
    generation_grids: list[tuple[int, int, int]] = []
    for image in pil_images:
        image_width, image_height = image.size
        height, width = _smart_resize(image_height, image_width)
        resized = image.resize((width, height), resample=PIL.Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0)
        generation_grids.append((1, height // 8, width // 8))

    max_height = max(tensor.shape[-2] for tensor in tensors)
    max_width = max(tensor.shape[-1] for tensor in tensors)
    padded = tensors[0].new_zeros(len(tensors), 3, max_height, max_width)
    for index, tensor in enumerate(tensors):
        height, width = tensor.shape[-2:]
        padded[index, :, :height, :width] = tensor
    return padded, torch.tensor(generation_grids, dtype=torch.long)


def process_diff_norm(norm: torch.Tensor, k: float = 0.4) -> torch.Tensor:
    """Reference high-timestep CFG delta normalization denominator."""
    powered = torch.pow(norm, k)
    return torch.where(
        norm > 1.0,
        powered,
        torch.where(norm < 1.0, torch.ones_like(norm), norm),
    )


def _move_conditioning_to_device(
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_image_mask: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place Stage-0 conditioning on the device used by the diffusion model."""
    return (
        prompt_embeds.to(device=device, dtype=dtype),
        prompt_attention_mask.to(device=device),
        prompt_image_mask.to(device=device),
    )


class InternVLUPipeline(
    nn.Module,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
    ProgressBarMixin,
):
    """Two-stage InternVL-U diffusion/VAE pipeline."""

    # Requests execute serially in forward(); real static batching is future work.
    supports_request_batch = False
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        tp_size = int(od_config.parallel_config.tensor_parallel_size)
        if tp_size != 1:
            raise ValueError(f"InternVL-U currently supports tensor_parallel_size=1 only, got {tp_size}")

        model = od_config.model
        local_files_only = os.path.isdir(model)
        self.generation_config = _load_generation_config(model, od_config.revision)
        # Conditioning width comes from the checkpoint's generation config
        # (validated for input/output/joint_attention_dim consistency above).
        self.vlm_cond_hidden_size = int(self.generation_config["input_hidden_size"])
        decoder_config = self.generation_config["decoder_config"]
        self.transformer = InternVLUTransformer2DModel(
            od_config=od_config,
            patch_size=int(decoder_config["patch_size"]),
            in_channels=int(decoder_config["in_channels"]),
            out_channels=int(decoder_config["out_channels"]),
            num_layers=int(decoder_config["num_layers"]),
            attention_head_dim=int(decoder_config["attention_head_dim"]),
            num_attention_heads=int(decoder_config["num_attention_heads"]),
            joint_attention_dim=int(decoder_config["joint_attention_dim"]),
            axes_dims_rope=decoder_config["axes_dims_rope"],
            video_position_scale_factor=float(decoder_config["video_position_scale_factor"]),
            vlm_cond_position_scale_factor=float(decoder_config["vlm_cond_position_scale_factor"]),
            max_sequence_length=int(self.generation_config["max_sequence_length"]),
            quant_config=od_config.quantization_config,
            prefix="transformer",
        )
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="generation_decoder",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            revision=od_config.revision,
            local_files_only=local_files_only,
        )
        self.scheduler.config.flow_shift = float(self.generation_config.get("flow_shift", 3.0))
        self.vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model,
            subfolder="vae",
            revision=od_config.revision,
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
        ).to(self.device)
        self.vae.eval()
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        if self.vae_scale_factor != 8:
            raise ValueError(f"InternVL-U requires Qwen-Image VAE scale factor 8, got {self.vae_scale_factor}")
        self.latent_channels = int(self.vae.config.z_dim)
        if self.latent_channels != 16:
            raise ValueError(f"InternVL-U requires 16-channel VAE latents, got {self.latent_channels}")
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def _parse_branch_payload(
        self,
        payload: dict[str, Any],
        *,
        role: str,
        raw_image_count: int,
    ) -> dict[str, Any]:
        hidden_states = payload.get("encoder_hidden_states")
        image_mask = payload.get("encoder_image_token_mask")
        image_fhw = payload.get("image_fhw_cond")
        reference_indices = payload.get("reference_image_indices")
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError(f"{role} branch is missing encoder_hidden_states")
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.vlm_cond_hidden_size:
            raise ValueError(
                f"{role} conditioning must be [T,{self.vlm_cond_hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if not isinstance(image_mask, torch.Tensor):
            raise ValueError(f"{role} branch is missing encoder_image_token_mask")
        if image_mask.ndim != 1 or image_mask.shape[0] != hidden_states.shape[0]:
            raise ValueError(f"{role} image mask is not aligned with conditioning")
        if not isinstance(image_fhw, torch.Tensor):
            image_fhw = torch.as_tensor(image_fhw, dtype=torch.long)
        if image_fhw.ndim != 2 or image_fhw.shape[-1] != 3:
            raise ValueError(f"{role} image_fhw_cond must be [N,3]")
        expected_image_tokens = int(image_fhw.prod(dim=-1).sum().item())
        if int(image_mask.bool().sum().item()) != expected_image_tokens:
            raise ValueError(f"{role} VLM image-token count does not match image_fhw_cond")
        if not isinstance(reference_indices, torch.Tensor):
            reference_indices = torch.as_tensor(reference_indices, dtype=torch.long)
        reference_indices = reference_indices.long().flatten()
        if reference_indices.numel() != image_fhw.shape[0]:
            raise ValueError(f"{role} reference_image_indices must align with image_fhw_cond")
        if reference_indices.numel() and (
            int(reference_indices.min()) < 0 or int(reference_indices.max()) >= raw_image_count
        ):
            raise ValueError(f"{role} reference image index is out of range")
        return {
            "encoder_hidden_states": hidden_states,
            "encoder_image_token_mask": image_mask.bool(),
            "image_fhw_cond": image_fhw.long(),
            "reference_image_indices": reference_indices,
        }

    def _parse_prompt(
        self,
        prompt: Any,
        sampling: Any,
        *,
        is_dummy: bool,
    ) -> dict[str, Any]:
        if is_dummy:
            default_height = int(sampling.height or 1024)
            default_width = int(sampling.width or 1024)
            height = max(16, default_height // 16 * 16)
            width = max(16, default_width // 16 * 16)
            branches = {
                role: {
                    "encoder_hidden_states": torch.zeros(1, self.vlm_cond_hidden_size),
                    "encoder_image_token_mask": torch.zeros(1, dtype=torch.bool),
                    "image_fhw_cond": torch.empty(0, 3, dtype=torch.long),
                    "reference_image_indices": torch.empty(0, dtype=torch.long),
                }
                for role in CFG_ROLES
            }
            return {
                "branches": branches,
                "raw_images": [],
                "height": height,
                "width": width,
                "requested_height": default_height,
                "requested_width": default_width,
                "image_grid_thw_gen": torch.tensor([[1, height // 8, width // 8]]),
                "is_cot": False,
            }

        if not isinstance(prompt, dict):
            raise ValueError("InternVL-U Stage 1 requires the ar2diffusion prompt dictionary")
        extra = prompt.get("extra")
        private = extra.get("internvlu") if isinstance(extra, dict) else None
        if not isinstance(private, dict):
            raise ValueError("InternVL-U Stage 1 prompt is missing extra['internvlu']")
        branches_payload = private.get("branches")
        if not isinstance(branches_payload, dict):
            raise ValueError("InternVL-U bridge payload is missing branches")
        if set(branches_payload) != set(CFG_ROLES):
            raise ValueError("InternVL-U bridge must provide exactly conditional, partial, and unconditional branches")

        multi_modal_data = prompt.get("multi_modal_data")
        raw_images = multi_modal_data.get("image") if isinstance(multi_modal_data, dict) else None
        if raw_images is None:
            raw_images = []
        elif not isinstance(raw_images, (list, tuple)):
            raw_images = [raw_images]
        else:
            raw_images = list(raw_images)

        branches = {
            role: self._parse_branch_payload(
                branches_payload[role],
                role=role,
                raw_image_count=len(raw_images),
            )
            for role in CFG_ROLES
        }
        expected_indices = list(range(len(raw_images)))
        conditional_indices = branches[CONDITIONAL_ROLE]["reference_image_indices"].tolist()
        partial_indices = branches[PARTIAL_ROLE]["reference_image_indices"].tolist()
        unconditional_indices = branches[UNCONDITIONAL_ROLE]["reference_image_indices"].tolist()
        if conditional_indices != expected_indices or partial_indices != expected_indices:
            raise ValueError("editing conditional and partial branches must retain every reference image in order")
        if unconditional_indices:
            raise ValueError("unconditional branch must not retain reference images")

        requested_height = _coerce_positive_int(
            private.get("requested_height", sampling.height or prompt.get("height", 1024)),
            name="requested_height",
        )
        requested_width = _coerce_positive_int(
            private.get("requested_width", sampling.width or prompt.get("width", 1024)),
            name="requested_width",
        )
        height = _coerce_positive_int(prompt.get("height"), name="height")
        width = _coerce_positive_int(prompt.get("width"), name="width")
        if height % 16 or width % 16:
            raise ValueError(f"InternVL-U execution size must be stride-16 aligned, got {(height, width)}")
        expected_grid = torch.tensor([[1, height // 8, width // 8]], dtype=torch.long)
        image_grid = private.get("image_grid_thw_gen")
        if image_grid is not None:
            image_grid = torch.as_tensor(image_grid, dtype=torch.long).cpu()
            if image_grid.shape != expected_grid.shape or not torch.equal(image_grid, expected_grid):
                raise ValueError(
                    f"generation grid {image_grid.tolist()} does not match execution size {(height, width)}"
                )
        return {
            "branches": branches,
            "raw_images": raw_images,
            "height": height,
            "width": width,
            "requested_height": requested_height,
            "requested_width": requested_width,
            "image_grid_thw_gen": expected_grid,
            "is_cot": bool(private.get("is_cot", False)),
        }

    def _latent_stats(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel VAE latent mean/std, shaped for [B,C,F,H,W] tensors."""
        means = torch.tensor(
            self.vae.config.latents_mean,
            device=device,
            dtype=dtype,
        ).view(1, self.latent_channels, 1, 1, 1)
        stds = torch.tensor(
            self.vae.config.latents_std,
            device=device,
            dtype=dtype,
        ).view(1, self.latent_channels, 1, 1, 1)
        return means, stds

    @torch.inference_mode()
    def _encode_reference_images(
        self,
        raw_images: Sequence[Any],
    ) -> list[torch.Tensor]:
        if not raw_images:
            return []
        pixels, generation_grids = _preprocess_reference_images(raw_images)
        vae_dtype = self.vae.dtype
        pixels = pixels.to(device=self.device, dtype=vae_dtype).unsqueeze(2)
        encoded = _retrieve_mode(self.vae.encode(pixels))
        means, stds = self._latent_stats(device=encoded.device, dtype=encoded.dtype)
        encoded = ((encoded - means) / stds).squeeze(2)
        expected_padded_shape = (
            pixels.shape[-2] // self.vae_scale_factor,
            pixels.shape[-1] // self.vae_scale_factor,
        )
        if encoded.ndim != 4 or encoded.shape[-2:] != expected_padded_shape:
            raise ValueError("Qwen-Image VAE reference latent shape does not match its scale factor")

        reference_latents: list[torch.Tensor] = []
        for index, grid in enumerate(generation_grids.tolist()):
            frame, valid_height, valid_width = (int(value) for value in grid)
            if frame != 1 or valid_height > encoded.shape[-2] or valid_width > encoded.shape[-1]:
                raise ValueError(f"invalid InternVL-U reference generation grid {grid}")
            reference_latents.append(encoded[index, :, :valid_height, :valid_width])
        return reference_latents

    @torch.inference_mode()
    def _decode_latents(
        self,
        latents: torch.Tensor,
        *,
        requested_height: int,
        requested_width: int,
    ) -> PIL.Image.Image:
        vae_dtype = self.vae.dtype
        latents = latents.to(dtype=vae_dtype).unsqueeze(2)
        means, stds = self._latent_stats(device=latents.device, dtype=latents.dtype)
        decoded = self.vae.decode(latents * stds + means, return_dict=False)[0]
        decoded = decoded[:, :, 0]
        decoded = ((127.5 * decoded + 128.0) / 255.0).clamp(0.0, 1.0)
        array = (decoded[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
        image = PIL.Image.fromarray(array)
        if image.size != (requested_width, requested_height):
            image = image.resize(
                (requested_width, requested_height),
                resample=PIL.Image.Resampling.LANCZOS,
            )
        return image

    @torch.inference_mode()
    def diffuse(
        self,
        latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_image_mask: torch.Tensor,
        branch_reference_latents: list[list[torch.Tensor]],
        branch_image_fhw: list[torch.Tensor],
        all_cfg_scale: float,
        part_cfg_scale: float,
        timestep_trunc: float,
    ) -> torch.Tensor:
        """Run the dual-CFG denoising loop over the prepared scheduler steps."""
        transformer_dtype = self.transformer.dtype
        # prompt_embeds rows follow CFG_ROLES order; the unbind below relies on it.
        num_branches = len(CFG_ROLES)
        with self.progress_bar(total=len(self.scheduler.timesteps)) as pbar:
            for timestep in self.scheduler.timesteps:
                model_input = latents.repeat(num_branches, 1, 1, 1).to(transformer_dtype)
                timestep_batch = timestep.expand(num_branches)
                noise = self.transformer(
                    model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep_batch,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_image_token_mask=prompt_image_mask,
                    reference_latents=branch_reference_latents,
                    image_fhw_cond=branch_image_fhw,
                    return_dict=False,
                )[0].float()
                conditional, partial, unconditional = noise.unbind(dim=0)
                partial_delta = partial - unconditional
                conditional_delta = conditional - partial
                if timestep_trunc > 0 and float(timestep.item()) > timestep_trunc:
                    partial_norm = torch.linalg.vector_norm(
                        partial_delta,
                        dim=0,
                        keepdim=True,
                    )
                    conditional_norm = torch.linalg.vector_norm(
                        conditional_delta,
                        dim=0,
                        keepdim=True,
                    )
                    partial_delta = partial_delta / process_diff_norm(partial_norm, 0.4)
                    conditional_delta = conditional_delta / process_diff_norm(conditional_norm, 0.4)
                velocity = (
                    unconditional + part_cfg_scale * partial_delta + all_cfg_scale * conditional_delta
                ).unsqueeze(0)
                latents = self.scheduler.step(
                    velocity.float(),
                    timestep,
                    latents.float(),
                    return_dict=False,
                )[0].float()
                pbar.update()
        return latents

    @torch.inference_mode()
    def _forward_one(self, request: Any) -> DiffusionOutput:
        sampling = request.sampling_params
        if int(sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("InternVL-U currently supports one output per prompt")
        parsed = self._parse_prompt(
            request.prompt,
            sampling,
            is_dummy=request.is_dummy_run(),
        )
        branches = parsed["branches"]
        reference_latents = self._encode_reference_images(parsed["raw_images"])

        # CFG_ROLES order is the batch layout diffuse() unpacks positionally.
        hidden_states = [branches[role]["encoder_hidden_states"] for role in CFG_ROLES]
        image_masks = [branches[role]["encoder_image_token_mask"] for role in CFG_ROLES]
        prompt_embeds, prompt_attention_mask, prompt_image_mask = self.transformer.prepare_forward_input(
            hidden_states,
            image_masks,
        )
        transformer_dtype = self.transformer.dtype
        prompt_embeds, prompt_attention_mask, prompt_image_mask = _move_conditioning_to_device(
            prompt_embeds,
            prompt_attention_mask,
            prompt_image_mask,
            device=self.device,
            dtype=transformer_dtype,
        )
        branch_reference_latents: list[list[torch.Tensor]] = []
        branch_image_fhw: list[torch.Tensor] = []
        for role in CFG_ROLES:
            indices = branches[role]["reference_image_indices"].tolist()
            branch_reference_latents.append([reference_latents[index] for index in indices])
            branch_image_fhw.append(
                branches[role]["image_fhw_cond"].to(
                    device=self.device,
                    dtype=torch.long,
                )
            )

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(sampling.seed))
        elif isinstance(generator, list):
            if len(generator) != 1:
                raise ValueError("InternVL-U currently supports one output per prompt")
            generator = generator[0]

        height = parsed["height"]
        width = parsed["width"]
        latents = randn_tensor(
            (1, self.latent_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        extra_args = sampling.extra_args or {}
        num_steps = int(sampling.num_inference_steps or 20)
        if num_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        # The reference inference implementation defaults to weaker guidance
        # (3.5/1.5) when the conditioning includes generated CoT text.
        if parsed["is_cot"]:
            default_all_cfg, default_part_cfg = 3.5, 1.5
        else:
            default_all_cfg = self.generation_config.get("all_cfg_scale", 4.5)
            default_part_cfg = self.generation_config.get("part_cfg_scale", 2.0)
        all_cfg_scale = float(extra_args.get("all_cfg_scale", default_all_cfg))
        part_cfg_scale = float(extra_args.get("part_cfg_scale", default_part_cfg))
        timestep_trunc = float(extra_args.get("timestep_trunc", 930.0))
        flow_shift = float(
            extra_args.get(
                "flow_shift",
                extra_args.get(
                    "timestep_shift",
                    self.generation_config.get("flow_shift", 3.0),
                ),
            )
        )
        self.scheduler.config.flow_shift = flow_shift
        # The framework moves pipeline tensor attributes to CUDA, but this
        # scheduler constructs its schedule through NumPy. Restore its static
        # FP32 lookup tables to their intended host device before construction;
        # set_timesteps still places the emitted timesteps on the model device.
        for name in ("alphas_cumprod", "alpha_t", "sigma_t", "lambda_t"):
            value = getattr(self.scheduler, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self.scheduler, name, value.float().cpu())
        self.scheduler.set_timesteps(num_steps, device=self.device)

        latents = self.diffuse(
            latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            prompt_image_mask=prompt_image_mask,
            branch_reference_latents=branch_reference_latents,
            branch_image_fhw=branch_image_fhw,
            all_cfg_scale=all_cfg_scale,
            part_cfg_scale=part_cfg_scale,
            timestep_trunc=timestep_trunc,
        )

        image = self._decode_latents(
            latents,
            requested_height=parsed["requested_height"],
            requested_width=parsed["requested_width"],
        )
        return DiffusionOutput(
            output=image,
            stage_durations=(self.stage_durations if hasattr(self, "stage_durations") else None),
        )

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        return [self._forward_one(request) for request in req.requests]

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def normalize():
            for name, weight in weights:
                # The checkpoint nests the MMDiT under ``decoder.``; the native
                # module is the flat ``transformer`` (its sibling
                # ``encoder_padding_token`` already lands there via the prefix).
                if name.startswith("transformer.decoder."):
                    name = "transformer." + name[len("transformer.decoder.") :]
                # The native module stores the row-parallel projection directly;
                # Diffusers wrapped the same projection in a one-element list.
                name = name.replace(".attn.to_out.0.", ".attn.to_out.")
                yield name, weight

        return AutoWeightsLoader(self).load_weights(normalize())


__all__ = [
    "InternVLUPipeline",
    "get_internvlu_post_process_func",
    "process_diff_norm",
]
