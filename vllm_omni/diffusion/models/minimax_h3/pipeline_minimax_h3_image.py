# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3-Image: text-to-image and image editing on the H3 DiT.

The image release reuses the H3 DiT (``audio_enabled: false``) and Qwen3-VL
text encoder (loaded from the base MiniMax-H3 repository named in
``model_index.json``) with a native 2D-decoder VAE (``image_vae.py``). It is a
PDD checkpoint: ``pdd_fused_heads.safetensors`` holds one fused output head per
denoise step and ``pdd_config.json`` lists the sigma grid they were distilled
on, so every request runs exactly that many single-branch (CFG-distilled)
Euler steps.

Per request:

1. A request smaller than ``min_generation_area`` pixels (2048x2048 by
   default) is generated on a larger 32px-aligned canvas with the same aspect
   ratio and resized back to the requested size with LANCZOS after decoding.
2. One ``torch.Generator`` seeded with the request seed draws, in order, the
   target noise, each reference's anchor noise, and each reference's VAE
   posterior sample.
3. Qwen3-VL encodes ``Picture i: <image>`` per edit reference, then the prompt.
4. The packed ref2va layout carries text, the reference rows (pinned at the
   conditioning timestep) and the target rows; each step forwards the positive
   branch once and applies ``x += (sigma_k - sigma_{k+1}) * v`` to the target.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import groupby
from math import sqrt
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_torch_threads_for_runtime

from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader import OffloadPlan
from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    IMAGE_PAD,
    VISION_END,
    VISION_START,
)
from vllm_omni.platforms import current_omni_platform

from .denoise_loop import (
    MINIMAX_H3_AUDIO_ROW_WIDTH,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
)
from .encoder import MiniMaxH3Qwen3VLEncoder
from .image_resize import resize_lanczos_uint8
from .image_vae import MiniMaxH3ImageVAE
from .minimax_h3_transformer import MiniMaxH3DiTModel
from .packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from .packed_tokens import minimax_h3_patchify_video_latent, minimax_h3_unpatchify_video_tokens
from .pipeline_minimax_h3 import MiniMaxH3Pipeline, _broadcast_tensor, _dit_rank_world

logger = init_logger(__name__)

MINIMAX_H3_IMAGE_PIXEL_GRID = 32
MINIMAX_H3_IMAGE_DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
# 32x32 never enlarges a request; 4096x4096 is the top tier.
MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE = (32 * 32, 4096 * 4096)
MINIMAX_H3_IMAGE_DEFAULT_SIZE = 1024
# A reference is resized to a 1024px short edge, so its aspect ratio bounds the
# pixels the Qwen preprocessing and the VAE encoder receive (a 32x16384 upload
# would otherwise become a 1024x524288 input). The output size is bounded by the
# server's --max-generated-image-size.
MINIMAX_H3_IMAGE_MAX_REFERENCE_ASPECT = 8
# Edit accepts up to 16 ordered references ("Picture 1" .. "Picture 16"); keep
# model_metadata's max_multimodal_image_inputs in sync.
MINIMAX_H3_IMAGE_MAX_REFERENCES = 16
# Edit references are resized per consumer so that their short edge lies in
# [min, max] (a short edge already inside keeps the original pixels).
MINIMAX_H3_IMAGE_QWEN_SHORT_EDGE = (1024, 1024)
MINIMAX_H3_IMAGE_VAE_SHORT_EDGE = (1024, 2048)
_TEXT_SEGMENT_MAX_TOKENS = 4096
_TEXT_TAG = 1
_VISUAL_TAG = 0
_LATENT_CHANNELS = 24
_PATCH = (1, 2, 2)


# ---------------------------------------------------------------------------
# Request geometry
# ---------------------------------------------------------------------------


def _nearest_grid_multiple(value: float, grid: int = MINIMAX_H3_IMAGE_PIXEL_GRID) -> int:
    # Python round: ties to even.
    return max(grid, int(round(value / grid)) * grid)


def minimax_h3_validate_min_generation_area(value: object) -> int:
    low, high = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise OmniClientError(
            f"min_generation_area must be an integer pixel area within [{low}, {high}], got {value!r}"
        )
    return value


def minimax_h3_image_generation_canvas(width: int, height: int, min_generation_area: int | None) -> tuple[int, int]:
    """The 32px-aligned ``(width, height)`` the DiT generates for a request."""
    if min_generation_area is None:
        return width, height
    scale = max(1.0, sqrt(min_generation_area / (width * height)))
    return _nearest_grid_multiple(width * scale), _nearest_grid_multiple(height * scale)


def minimax_h3_image_reference_size(
    width: int, height: int, *, min_short_edge: int, max_short_edge: int
) -> tuple[int, int] | None:
    """Size of an edit reference for one consumer; None keeps the original pixels."""
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return None
    bound = min_short_edge if short_edge < min_short_edge else max_short_edge
    scale = bound / short_edge
    return _nearest_grid_multiple(width * scale), _nearest_grid_multiple(height * scale)


def _bound_reference(image: Image.Image, bounds: tuple[int, int]) -> Image.Image:
    size = minimax_h3_image_reference_size(*image.size, min_short_edge=bounds[0], max_short_edge=bounds[1])
    if size is None or size == image.size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def minimax_h3_image_reference_grid(width: int, height: int) -> tuple[int, int]:
    """``(latent_h, latent_w)`` of an edit reference: 16px cells, even count."""
    return height // 16 // 2 * 2, width // 16 // 2 * 2


def minimax_h3_quantize_image(pixels: torch.Tensor) -> torch.Tensor:
    """Decoded ``[B, 3, H, W]`` pixels in [0, 1] to uint8, as the released writer
    quantizes them (truncating ``(x * 255).clamp(0, 255)``)."""
    return (pixels * 255).clamp(0, 255).to(torch.uint8)


def minimax_h3_resize_image_output(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize uint8 ``[B, 3, H, W]`` images to ``width x height`` with PIL LANCZOS.

    The reference resizes the writer-quantized 8-bit image and hands back
    ``uint8 / 255``; that value quantizes back to the same byte, so the uint8
    result is returned directly.
    """
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected uint8 [B, 3, H, W] images, got {images.dtype} {list(images.shape)}")
    if images.is_cuda:
        # Same fixed-point arithmetic as Pillow, on the device (image_resize.py).
        return resize_lanczos_uint8(images, width, height)
    resized = []
    for image in images.permute(0, 2, 3, 1).contiguous().cpu().numpy():
        output = Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)
        resized.append(torch.from_numpy(np.array(output)).permute(2, 0, 1))
    return torch.stack(resized)


def minimax_h3_image_pdd_grid(config: dict[str, Any]) -> tuple[float, ...]:
    """The validated ``sigmas`` of an image release's pdd_config.json."""
    sigmas, steps = config.get("sigmas"), config.get("num_inference_steps")
    if not isinstance(sigmas, list) or any(isinstance(s, bool) or not isinstance(s, (int, float)) for s in sigmas):
        raise ValueError(f"image pdd_config.json sigmas must be a list of numbers, got {sigmas!r}")
    sigmas = tuple(float(s) for s in sigmas)
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or len(sigmas) != steps + 1
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(a <= b for a, b in zip(sigmas, sigmas[1:]))
    ):
        raise ValueError(
            "image pdd_config.json sigmas must decrease strictly from 1.0 to 0.0 in "
            f"num_inference_steps + 1 points; got num_inference_steps={steps!r}, sigmas={list(sigmas)!r}"
        )
    return sigmas


def minimax_h3_image_run_sigmas(pdd_sigmas: Sequence[float]) -> list[float]:
    """The PDD grid as the loop runs it: each sigma rounded through its FP32 timestep."""
    timesteps = torch.tensor([1.0 - sigma for sigma in pdd_sigmas[:-1]], dtype=torch.float32)
    return torch.cat((1.0 - timesteps, torch.zeros(1, dtype=torch.float32))).tolist()


def minimax_h3_image_presentation(
    tokenizer: Any, *, prompt: str, image_token_counts: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Qwen input: per reference ``Picture i: `` and its image block, then the prompt.

    Each text segment is right-truncated to 4096 tokens and the ids are
    right-padded with 0 to a multiple of 8. Returns ``(padded_ids, tags,
    text_len)``; Qwen is causal, so the padding never changes the first
    ``text_len`` hidden states.
    """

    def text(value: str) -> list[int]:
        return list(tokenizer(value, add_special_tokens=False)["input_ids"])[:_TEXT_SEGMENT_MAX_TOKENS]

    ids: list[int] = []
    tags: list[int] = []
    start, end, pad = (tokenizer.convert_tokens_to_ids(token) for token in (VISION_START, VISION_END, IMAGE_PAD))
    for index, count in enumerate(image_token_counts, start=1):
        label = text(f"Picture {index}: ")
        ids += label
        tags += [_TEXT_TAG] * len(label)
        block = [start] + [pad] * int(count) + [end]
        ids += block
        tags += [_VISUAL_TAG] * len(block)
    body = text(prompt)
    ids += body
    tags += [_TEXT_TAG] * len(body)
    text_len = len(ids)
    padded = ids + [0] * (-text_len % 8)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(tags, dtype=torch.long), text_len


def _require_reference_prefix(update_mask: torch.Tensor, n_cond: int) -> None:
    """The step below slices ``rows[n_cond:]``; that equals the packed
    ``update_mask`` selection only while the reference rows are a prefix."""
    mask = update_mask.view(-1).to(torch.bool).cpu()
    if int(mask.numel()) <= n_cond or mask[:n_cond].any() or not mask[n_cond:].all():
        raise RuntimeError(
            f"MiniMax-H3-Image expects {n_cond} reference rows before the target rows, "
            f"got update_mask with {int((~mask).sum())} condition rows out of {mask.numel()}"
        )


def _euler_target_step(rows: torch.Tensor, velocity: torch.Tensor, n_cond: int, delta: torch.Tensor) -> None:
    """In-place eta-0 Euler update of the target rows: ``x + (sigma_k - sigma_{k+1}) * v``.

    Algebraically equal to ``minimax_h3_euler_eta0_step`` on ``x0 = x + sigma_k * v``;
    kept in this form because it is the released sampler's arithmetic.
    """
    target = rows[n_cond:]
    target.copy_(target + delta * velocity[n_cond:].float())


def _image_packed_seq_len(used_rows: int) -> int:
    """Used rows (text + target + references) rounded up to 16, plus 16 rows."""
    return ((used_rows + 15) // 16) * 16 + 16


def _upright_rgb(image: Image.Image) -> Image.Image:
    if getattr(image, "n_frames", 1) != 1:
        raise OmniClientError("image reference must not be animated")
    return ImageOps.exif_transpose(image).convert("RGB")


def _load_reference(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        image = _upright_rgb(value)
    elif isinstance(value, (str, os.PathLike)):
        with Image.open(value) as source:
            image = _upright_rgb(source)
    else:
        raise OmniClientError(f"unsupported image reference type {type(value).__name__}")
    if min(image.size) < 32:
        raise OmniClientError("image reference must retain at least 32 pixels on each axis")
    if max(image.size) > MINIMAX_H3_IMAGE_MAX_REFERENCE_ASPECT * min(image.size):
        raise OmniClientError(
            f"image reference aspect ratio must be at most {MINIMAX_H3_IMAGE_MAX_REFERENCE_ASPECT}:1, "
            f"got {image.size[0]}x{image.size[1]}"
        )
    return image


def get_minimax_h3_image_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(images: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return images
        if images.dtype == torch.uint8 and images.ndim == 4 and images.shape[-1] == 3:
            # The pipeline's uint8 [B, H, W, 3] output.
            arrays = images.cpu().contiguous().numpy()
        else:
            if images.ndim == 3:
                images = images[None]
            if images.dtype != torch.uint8:
                # Truncating quantization, as the released writer.
                images = minimax_h3_quantize_image(images.float())
            arrays = images.cpu().permute(0, 2, 3, 1).contiguous().numpy()
        if output_type == "np":
            return arrays
        return [Image.fromarray(array) for array in arrays]

    return post_process_func


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MiniMaxH3ImagePipeline(
    nn.Module, SupportImageInput, SupportsComponentDiscovery, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    """MiniMax-H3-Image t2i / edit (one image per request)."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    # Layerwise offload streams the DiT blocks and the Qwen3-VL encoder (the
    # same encoder class as MiniMax-H3). The image VAE stays resident: its
    # decode tiles replay CUDA graphs that capture the weight addresses.
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        offload_submodules={"token_refiner": "blocks"},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_component_types={"text_encoder": TEXT_ENCODER_COMPONENT},
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder"}),
    )
    _PROFILER_TARGETS: ClassVar[list[str]] = ["_encode_text", "_encode_references", "_denoise", "vae.decode_latent"]

    # The text-encoder TP group helpers are shared with the video pipeline.
    _build_text_encoder_group = MiniMaxH3Pipeline._build_text_encoder_group
    _resident_dit_layers_on_device = MiniMaxH3Pipeline._resident_dit_layers_on_device
    # An offloaded text encoder is staged for the encode and released after it.
    _uses_manual_component_offload = MiniMaxH3Pipeline._uses_manual_component_offload
    _component_on_device = MiniMaxH3Pipeline._component_on_device

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        parallel = od_config.parallel_config
        if int(parallel.cfg_parallel_size) != 1:
            raise ValueError("MiniMax-H3-Image is CFG-distilled (one positive branch); cfg_parallel_size must be 1")
        if getattr(parallel, "use_hsdp", False):
            raise ValueError("MiniMax-H3-Image does not support HSDP")

        root = self._resolve_model_root(str(od_config.model), od_config.revision)
        self.model_root = root
        model_index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
        if model_index.get("_class_name") != "MiniMaxH3ImagePipeline":
            raise ValueError(f"{root} is not a MiniMaxH3ImagePipeline release")
        release = model_index.get("_minimax_h3") or {}
        if str(release.get("partition", "")).lower() != "image":
            raise ValueError(f"{root} is not a MiniMax H3 image release partition")
        self.supported_tasks = frozenset(str(task).lower() for task in release.get("tasks", ()))
        if not self.supported_tasks or not self.supported_tasks <= {"t2i", "edit"}:
            raise ValueError(f"MiniMax H3 image tasks must be a subset of t2i/edit, got {sorted(self.supported_tasks)}")
        heads = release.get("pdd_fused_heads")
        if not isinstance(heads, str) or not heads:
            raise ValueError("MiniMax-H3-Image serves PDD releases only: _minimax_h3.pdd_fused_heads is required")
        self.pdd_heads_path = root / heads
        self.pdd_sigmas = minimax_h3_image_pdd_grid(
            json.loads((self.pdd_heads_path.parent / "pdd_config.json").read_text(encoding="utf-8"))
        )
        self.num_steps = len(self.pdd_sigmas) - 1
        additional = od_config.additional_config or {}
        self.min_generation_area: int | None = additional.get(
            "min_generation_area", MINIMAX_H3_IMAGE_DEFAULT_MIN_GENERATION_AREA
        )
        if self.min_generation_area is not None:
            minimax_h3_validate_min_generation_area(self.min_generation_area)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(od_config.model),
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        # Honor --quantization (e.g. online fp8) for the DiT linears, as the video
        # pipeline does; the fused PDD heads stay FP32.
        from vllm_omni.quantization import resolve_component_quant_config

        transformer_quant_config = resolve_component_quant_config(od_config.quantization_config, "transformer")
        self.transformer = MiniMaxH3DiTModel(od_config, quant_config=transformer_quant_config)
        if self.transformer.arch.audio_enabled:
            raise ValueError("MiniMax-H3-Image requires an audio-free DiT (audio_enabled: false)")

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(str(root), subfolder="tokenizer", local_files_only=True)
        self.processor = Qwen3VLProcessor.from_pretrained(str(root), subfolder="processor", local_files_only=True)

        dit_group, rank, dit_world = _dit_rank_world()
        self._dit_rank = rank
        # VAE tiles are spread over every DiT rank; each tile keeps its
        # single-rank kernels, so outputs do not depend on the rank count.
        tile_parallel = (od_config.additional_config or {}).get("vae_tile_parallel", True)
        if type(tile_parallel) is not bool:
            raise ValueError(f"vae_tile_parallel must be a bool, got {tile_parallel!r}")
        self._vae_group = dit_group if dit_world > 1 and tile_parallel else None
        # decode_latent only captures graphs on a cuda-like platform (CUDA,
        # ROCm and MUSA); elsewhere (e.g. XPU) the tiles decode eagerly.
        self._vae_cuda_graph = (
            current_omni_platform.is_cuda_alike()
            and bool(getattr(od_config, "enable_cuda_graph_decode", True))
            and not bool(getattr(od_config, "enforce_eager", False))
        )
        tp_size = int(getattr(parallel, "text_encoder_tp_size", 1))
        if not 1 <= tp_size <= dit_world or 64 % tp_size or 8 % tp_size:
            raise ValueError(
                "text_encoder_tp_size must divide Qwen3-VL's 64 query / 8 KV heads and not exceed "
                f"the DiT group size ({dit_world}), got {tp_size}"
            )
        self.text_encoder_tp_size = tp_size
        self.text_encoder_group = self._build_text_encoder_group(tp_size)
        encoder_root = self._resolve_text_encoder_root(model_index, root)
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            os.path.join(encoder_root, "text_encoder"),
            device=self.device,
            load_model=rank < tp_size,
            encoder_group=self.text_encoder_group,
        )
        if rank >= tp_size:
            # Ranks outside the text-encoder TP group hold no encoder weights.
            self._encoder_modules = []
            self._offload_plan = replace(
                self._offload_plan,
                encoder_component_types={},
                encoder_block_attrs={},
                on_demand_component_paths=frozenset(),
            )
        if rank < tp_size:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=str(encoder_root),
                    subfolder="text_encoder",
                    revision=None,
                    prefix="text_encoder.",
                    fall_back_to_pt=False,
                )
            )
        cudnn_sdp = additional.get("text_encoder_cudnn_sdp", True)
        if type(cudnn_sdp) is not bool:
            raise ValueError(f"text_encoder_cudnn_sdp must be a bool, got {cudnn_sdp!r}")
        # False: same seed, same image across restarts, at a small cost in
        # agreement with the reference (MiniMaxH3Qwen3VLEncoder.cudnn_sdp).
        self.text_encoder.cudnn_sdp = cudnn_sdp
        self.vae = MiniMaxH3ImageVAE.from_pretrained(root / "vae").to(self.device)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def setup_compile(self) -> None:
        """Keep the DiT eager; CUDA graphs still replay the VAE tile decode.

        Each request packs its own text/reference/target row counts, so
        regional compilation of the DiT blocks recompiles past dynamo's limit
        and falls back to eager without a measured speedup, while changing
        the DiT numerics. The VAE graphs (``enable_cuda_graph_decode``) keep
        the eager kernels and are therefore output-preserving.
        """
        logger.info("MiniMax-H3-Image: DiT stays eager; VAE tile decode uses CUDA graphs: %s", self._vae_cuda_graph)

    # -- checkpoint resolution ----------------------------------------------

    @staticmethod
    def _resolve_model_root(model: str, revision: str | None) -> Path:
        if os.path.isdir(model):
            return Path(model)
        return Path(
            download_weights_from_hf_specific(
                model,
                None,
                ["model_index.json", "pdd_*", "tokenizer/*", "processor/*", "vae/*"],
                revision=revision,
            )
        )

    @staticmethod
    def _resolve_text_encoder_root(model_index: dict[str, Any], root: Path) -> str:
        """Root holding ``text_encoder/``: this release, or the base repository it names."""
        spec = model_index.get("text_encoder")
        source = spec[2] if isinstance(spec, list) and len(spec) == 3 and isinstance(spec[2], dict) else None
        if source is None:
            return str(root)
        repo, subfolder = source.get("pretrained_model_name_or_path"), source.get("subfolder")
        if subfolder != "text_encoder" or not isinstance(repo, str) or not repo:
            raise ValueError(f"unsupported MiniMax H3 image text_encoder source {source!r}")
        if (root / "text_encoder").is_dir():
            # The release bundles a copy of the base repository's text encoder
            # (same shards) while model_index.json still names the base repo.
            # Prefer the local copy: it needs no second download.
            logger.info("MiniMax-H3-Image: using bundled text_encoder/ instead of %s", repo)
            return str(root)
        if os.path.isdir(repo):
            return repo
        return download_weights_from_hf_specific(
            repo, None, ["text_encoder/*"], revision=source.get("revision"), require_all=True
        )

    # -- weights --------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def component_of(item: tuple[str, torch.Tensor]) -> str:
            component = item[0].partition(".")[0]
            if component not in ("transformer", "text_encoder"):
                raise ValueError(f"unexpected MiniMax-H3-Image weight {item[0]!r}")
            return component

        loaded: set[str] = set()
        seen: set[str] = set()
        for component, group in groupby(weights, key=component_of):
            if component in seen:
                raise ValueError(f"MiniMax-H3-Image weight source {component!r} is not contiguous")
            seen.add(component)
            names = getattr(self, component).load_weights(
                (name[len(component) + 1 :], tensor) for name, tensor in group
            )
            loaded.update(f"{component}.{name}" for name in names)
        self.transformer.post_load_weights()
        steps = self.transformer.final_layer.load_fused_pdd_heads(str(self.pdd_heads_path))
        if steps != self.num_steps:
            raise ValueError(f"{self.pdd_heads_path} holds {steps} heads, but pdd_config.json lists {self.num_steps}")
        logger.info("MiniMax-H3-Image: %d fused PDD output heads loaded from %s", steps, self.pdd_heads_path)
        loaded.update(f"vae.{name}" for name, _ in self.vae.named_parameters())
        # Startup is done: serve with vLLM's runtime CPU-thread policy. Each
        # rank otherwise keeps one OpenMP thread per core, and the per-request
        # host work (noise, Qwen image preprocessing, VAE input) spends tens of
        # milliseconds in spin-wait contention across the ranks.
        set_torch_threads_for_runtime()
        return loaded

    # -- request parsing -------------------------------------------------------

    def _parse_request(self, request: DiffusionRequestBatch) -> dict[str, Any]:
        if len(request.prompts) != 1:
            raise OmniClientError("MiniMax-H3-Image generates one image per request")
        raw = request.prompts[0]
        sampling = request.sampling_params
        if isinstance(raw, str):
            prompt, media, negative = raw, {}, None
        else:
            prompt = raw.get("prompt")
            media = raw.get("multi_modal_data") or {}
            negative = raw.get("negative_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise OmniClientError("prompt must be a non-empty string")
        if negative:
            raise OmniClientError("negative_prompt is not supported: MiniMax-H3-Image runs one positive branch")
        if int(getattr(sampling, "num_outputs_per_prompt", 1) or 1) != 1:
            raise OmniClientError("MiniMax-H3-Image produces exactly one image per request")
        steps = sampling.num_inference_steps
        if request.is_dummy_run():
            # Startup warmup runs a prefix of the PDD grid and carries generic
            # CFG settings that a real request may not set.
            run_steps = min(max(1, int(steps or 1)), self.num_steps)
        else:
            if getattr(sampling, "guidance_scale_provided", False) and sampling.guidance_scale not in (None, 1, 1.0):
                raise OmniClientError("MiniMax-H3-Image is CFG-distilled: guidance_scale must be 1 or omitted")
            if steps is not None and steps != self.num_steps:
                raise OmniClientError(
                    f"this MiniMax-H3-Image PDD release has {self.num_steps} fused output heads, "
                    f"so num_inference_steps must be {self.num_steps} or omitted, got {steps!r}"
                )
            run_steps = self.num_steps
        images = media.get("image")
        images = [] if images is None else list(images) if isinstance(images, (list, tuple)) else [images]
        task = "edit" if images else "t2i"
        if task not in self.supported_tasks:
            raise OmniClientError(
                f"task {task!r} is not served by this release; declared {sorted(self.supported_tasks)}"
            )
        if len(images) > MINIMAX_H3_IMAGE_MAX_REFERENCES:
            raise OmniClientError(
                f"MiniMax-H3-Image accepts at most {MINIMAX_H3_IMAGE_MAX_REFERENCES} reference images, "
                f"got {len(images)}"
            )
        references = [_load_reference(image) for image in images]
        width = sampling.width if not getattr(sampling, "width_not_provided", False) else None
        height = sampling.height if not getattr(sampling, "height_not_provided", False) else None
        width = int(width or MINIMAX_H3_IMAGE_DEFAULT_SIZE)
        height = int(height or MINIMAX_H3_IMAGE_DEFAULT_SIZE)
        extra = getattr(sampling, "extra_args", None) or {}
        min_area = self.min_generation_area
        if "min_generation_area" in extra:
            if min_area is None:
                raise OmniClientError("min_generation_area is disabled on this server")
            min_area = minimax_h3_validate_min_generation_area(extra["min_generation_area"])
        if width < 1 or height < 1:
            raise OmniClientError("width and height must be positive")
        if min_area is None and (width % 32 or height % 32 or width < 32 or height < 32):
            raise OmniClientError("width and height must be multiples of 32 when min_generation_area is disabled")
        canvas_w, canvas_h = minimax_h3_image_generation_canvas(width, height, min_area)
        device = sampling.generator_device or current_omni_platform.device_type
        return {
            "prompt": prompt,
            "references": references,
            "width": width,
            "height": height,
            "canvas": (canvas_w, canvas_h),
            "seed": int(sampling.seed if sampling.seed is not None else 0),
            "steps": run_steps,
            "generator_device": "cpu" if device == "cpu" else current_omni_platform.device_type,
            "output_type": sampling.output_type or "pil",
        }

    # -- stages ----------------------------------------------------------------

    @staticmethod
    def _noise_rows(generator: torch.Generator, latent_h: int, latent_w: int) -> torch.Tensor:
        noise = torch.randn(
            1,
            _LATENT_CHANNELS,
            1,
            latent_h,
            latent_w,
            generator=generator,
            device=generator.device,
            dtype=torch.float32,
        )
        return minimax_h3_patchify_video_latent(noise.cpu(), patch_size=list(_PATCH))

    def _encode_text(self, prompt: str, references: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer-50 Qwen states of the positive presentation, on every DiT rank."""
        vision: dict[str, torch.Tensor] = {}
        counts: list[int] = []
        if references:
            images = [_bound_reference(image, MINIMAX_H3_IMAGE_QWEN_SHORT_EDGE) for image in references]
            processed = self.processor.image_processor(images=images, return_tensors="pt")
            vision = {key: processed[key] for key in ("pixel_values", "image_grid_thw")}
            merge = int(self.processor.image_processor.merge_size) ** 2
            counts = [int(row.prod()) // merge for row in vision["image_grid_thw"]]
        ids, tags, text_len = minimax_h3_image_presentation(self.tokenizer, prompt=prompt, image_token_counts=counts)
        hidden = None
        if self._dit_rank < self.text_encoder_tp_size:
            if getattr(self, "_model_cpu_offload_modules", None):
                hidden = self.text_encoder(ids, **vision)
            elif self._uses_manual_component_offload(self.text_encoder):
                with self._component_on_device(self.text_encoder):
                    hidden = self.text_encoder.encode_ids(ids, **vision)
            else:
                self.text_encoder.load_to_device()
                hidden = self.text_encoder.encode_ids(ids, **vision)
            hidden = hidden[:text_len]
        hidden = _broadcast_tensor(hidden, dtype=torch.bfloat16, device=self.device)
        return hidden.cpu(), tags

    def _encode_references(
        self, references: list[Image.Image], grids: list[tuple[int, int]], generator: torch.Generator
    ) -> list[torch.Tensor]:
        """Clean normalized rows of VAE-bounded edit references on their latent grids (draw 3)."""
        rows = []
        for image, (latent_h, latent_w) in zip(references, grids):
            latent = self.vae.encode_image_latent(image, generator=generator, group=self._vae_group).cpu()
            latent = self.vae.normalize_latent(latent[..., :latent_h, :latent_w])
            rows.append(minimax_h3_patchify_video_latent(latent, patch_size=list(_PATCH)).to(torch.float32))
        return rows

    def _denoise(
        self,
        *,
        hidden: torch.Tensor,
        text_tags: torch.Tensor,
        grids: list[tuple[int, int]],
        anchors: torch.Tensor,
        target_noise: torch.Tensor,
        latent_h: int,
        latent_w: int,
        num_steps: int,
    ) -> torch.Tensor:
        text_len = int(hidden.shape[0])
        blocks = [{"kind": "image", "latent_h": h, "latent_w": w} for h, w in grids]
        used = text_len + latent_h * latent_w // 4 + sum(h * w // 4 for h, w in grids)
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=text_len,
            latent_t=1,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=0,
            ref_blocks=blocks,
            seq_len=_image_packed_seq_len(used),
        )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"].view(-1)] = text_tags
        branch = MiniMaxH3DenoiseBranch(packed=packed, text_embeddings=hidden, token_tags=tags, device=self.device)
        branch.prepare_rope_table(self.transformer)
        n_cond = int(anchors.shape[0])
        _require_reference_prefix(packed["update_mask"], n_cond)
        anchors = anchors.to(self.device)
        video_rows = torch.cat((anchors, target_noise.to(self.device)), dim=0)
        audio_rows = torch.empty((0, MINIMAX_H3_AUDIO_ROW_WIDTH), device=self.device, dtype=torch.float32)
        sigmas = minimax_h3_image_run_sigmas(self.pdd_sigmas)
        sigma_values = torch.tensor(sigmas, dtype=torch.float32, device=self.device)
        deltas = sigma_values[:-1] - sigma_values[1:]
        with self.progress_bar(total=num_steps) as progress_bar:
            for step in range(num_steps):
                check_request_cancellation()
                t = 1.0 - sigmas[step]
                kwargs = branch.forward_kwargs(
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    t_video=t,
                    t_audio=t,
                    imgvid_cond_timestep=MINIMAX_H3_IMGVID_COND_TIMESTEP,
                    audio_ref_cond_timestep=1.0,
                )
                # Only the target rows are read back below, and the update
                # writes only them, so the anchor rows need neither re-copying
                # nor a zeroed velocity.
                kwargs["skip_mask_out_condition"] = True
                velocity, _ = self.transformer(pdd_step=step, **kwargs)
                _euler_target_step(video_rows, velocity, n_cond, deltas[step])
                progress_bar.update()
        return video_rows[n_cond:]

    # no_grad, not inference_mode: the runner selects no_grad for layerwise
    # offload (DLO), whose hooks read tensor version counters.
    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        check_request_cancellation()
        parsed = self._parse_request(request)
        canvas_w, canvas_h = parsed["canvas"]
        latent_h, latent_w = canvas_h // 16, canvas_w // 16
        references = parsed["references"]

        generator = torch.Generator(device=parsed["generator_device"]).manual_seed(parsed["seed"])
        # Draw 1: target noise; draw 2: per-reference anchor noise; draw 3 (in
        # the VAE encode): per-reference posterior samples.
        target_noise = self._noise_rows(generator, latent_h, latent_w)
        vae_refs = [_bound_reference(image, MINIMAX_H3_IMAGE_VAE_SHORT_EDGE) for image in references]
        grids = [minimax_h3_image_reference_grid(*image.size) for image in vae_refs]
        anchor_noise = [self._noise_rows(generator, h, w) for h, w in grids]
        clean_rows = self._encode_references(vae_refs, grids, generator)
        cond_t = torch.tensor(MINIMAX_H3_IMGVID_COND_TIMESTEP, dtype=torch.float32)
        anchors = (
            torch.cat([cond_t * clean + (1.0 - cond_t) * noise for clean, noise in zip(clean_rows, anchor_noise)])
            if clean_rows
            else torch.empty((0, _LATENT_CHANNELS * 4), dtype=torch.float32)
        )

        check_request_cancellation()
        hidden, text_tags = self._encode_text(parsed["prompt"], references)
        # DLO keeps resident DiT layers on the host outside the denoise loop.
        with self._resident_dit_layers_on_device():
            target = self._denoise(
                hidden=hidden,
                text_tags=text_tags,
                grids=grids,
                anchors=anchors,
                target_noise=target_noise,
                latent_h=latent_h,
                latent_w=latent_w,
                num_steps=parsed["steps"],
            )
        latent = minimax_h3_unpatchify_video_tokens(
            target.float(), latent_shape=(1, latent_h // 2, latent_w // 2, _LATENT_CHANNELS), patch_size=_PATCH
        )
        if parsed["output_type"] == "latent":
            return DiffusionOutput(output=latent)
        check_request_cancellation()
        pixels = self.vae.decode_latent(latent, group=self._vae_group, use_cuda_graph=self._vae_cuda_graph)
        # Quantize on the device (the same FP32 arithmetic as on the host) so
        # only uint8 bytes cross to the host.
        pixels = minimax_h3_quantize_image(pixels)
        if (canvas_w, canvas_h) != (parsed["width"], parsed["height"]):
            pixels = minimax_h3_resize_image_output(pixels, parsed["width"], parsed["height"])
        # Hand the host uint8 HWC images: the writer needs no layout copy.
        pixels = pixels.permute(0, 2, 3, 1).contiguous().cpu()
        return DiffusionOutput(output=pixels)


__all__ = [
    "MiniMaxH3ImagePipeline",
    "get_minimax_h3_image_post_process_func",
    "minimax_h3_image_generation_canvas",
    "minimax_h3_image_pdd_grid",
    "minimax_h3_image_presentation",
]
