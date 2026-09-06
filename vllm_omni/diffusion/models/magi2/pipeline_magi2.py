# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native vLLM-Omni pipeline for ``sand-ai/MAGI-2-preview``.

The model, packing, scheduler, checkpoint mapping, and decoder integration are
owned by vLLM-Omni.  SandAI's Apache-2.0 implementation was used as the
accuracy reference, but is neither imported nor required at runtime.

This first native integration intentionally supports the released Preview
stage (272p and 540p).  The separately checkpointed 1080p refiner is not
silently routed through an external implementation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader import OffloadPlan, PinnedModuleStager
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.platforms import current_omni_platform

from .audio_decoder import Magi2AudioDecoder
from .conditioning import Magi2Qwen35TextEncoder
from .configuration_magi2 import (
    MAGI2_GENERATION_CONFIG,
    MAGI2_PREVIEW_CONFIG,
)
from .parallel import get_magi2_replica_group
from .preview_data_proxy import Magi2DataProxy
from .sampler_magi2 import (
    CFGConfig,
    Magi2PreviewSampler,
    SamplerInput,
    build_magi2_preview_schedulers,
)
from .turbo_vae import Magi2TurboVAEDecoder

logger = logging.getLogger(__name__)

MAGI2_MODEL_ID = "sand-ai/MAGI-2-preview"
MAGI2_MODEL_REVISION = "2dea51b64db47ee5b4402d36fd90829a0c58913b"
MAGI2_AUDIO_SAMPLE_RATE = MAGI2_GENERATION_CONFIG.audio_sample_rate

_RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "272p": (448, 256),
    "540p": (896, 512),
}
_PRESET_BY_SIZE = {size: name for name, size in _RESOLUTION_PRESETS.items()}

_REQUIRED_CHECKPOINT_PATHS = (
    "preview/model.safetensors.index.json",
    "text_encoder/model.safetensors.index.json",
    "vae/Wan2.2_VAE.pth",
    "turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json",
    "turbo_vae/checkpoint.ckpt",
    "stable-audio-open-1.0/model_config.json",
    "stable-audio-open-1.0/model.safetensors",
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)
DEFAULT_NEGATIVE_PROMPT += (
    ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, "
    "compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, "
    "room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, "
    "thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, "
    "looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, "
    "disorganized, messy, cheap sound"
)
DEFAULT_NEGATIVE_PROMPT += (
    ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, "
    "undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, "
    "exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, "
    "stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky "
    "voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, "
    "coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural "
    "pauses, awkward silence, choppy, disconnected, multiple speakers, two voices, background talking, out of tune, "
    "off-key, autotune artifacts"
)


class _PeakReservedMonitor:
    def __init__(self, device: int) -> None:
        self.device = device
        self.peak_bytes = torch.accelerator.memory_reserved(device)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(
                self.peak_bytes,
                torch.accelerator.memory_reserved(self.device),
            )
            self._stop.wait(0.05)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(
            self.peak_bytes,
            torch.accelerator.memory_reserved(self.device),
        )


def _config_value(
    od_config: OmniDiffusionConfig,
    key: str,
    env_key: str,
    default: Any = None,
) -> Any:
    custom = od_config.custom_pipeline_args or {}
    additional = od_config.additional_config or {}
    if key in custom:
        return custom[key]
    if key in additional:
        return additional[key]
    return os.environ.get(env_key, default)


def _env_flag(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_native_resolution(sampling: object, extra: Mapping[str, object]) -> tuple[str, int, int]:
    """Resolve MAGI-2 generation geometry from the shared sampling fields."""
    resolution_value = extra.get("resolution")
    resolution = None if resolution_value is None else str(resolution_value).lower()
    if resolution is not None and resolution not in _RESOLUTION_PRESETS:
        raise OmniClientError(
            f"Unsupported native MAGI-2 resolution {resolution!r}; choose "
            f"from {sorted(_RESOLUTION_PRESETS)}. The 1080p refiner is not "
            "part of the Preview-only native PR."
        )

    requested_width = getattr(sampling, "width", None)
    requested_height = getattr(sampling, "height", None)
    if (requested_width is None) != (requested_height is None):
        raise OmniClientError("MAGI-2 native width and height must be supplied together")

    requested_size: tuple[int, int] | None = None
    if requested_width is not None:
        try:
            requested_size = (int(requested_width), int(requested_height))
        except (TypeError, ValueError) as exc:
            raise OmniClientError("MAGI-2 native dimensions must be positive integers") from exc
        if requested_size[0] <= 0 or requested_size[1] <= 0:
            raise OmniClientError("MAGI-2 native dimensions must be positive integers")

    if resolution is None:
        if requested_size is None:
            resolution = "540p"
        else:
            resolution = _PRESET_BY_SIZE.get(requested_size)
            if resolution is None:
                raise OmniClientError(
                    f"Unsupported native MAGI-2 size {requested_size[0]}x{requested_size[1]}; "
                    f"choose from {sorted(_PRESET_BY_SIZE)} or use output_width/output_height for final resizing."
                )
    elif requested_size is not None and requested_size != _RESOLUTION_PRESETS[resolution]:
        expected_width, expected_height = _RESOLUTION_PRESETS[resolution]
        raise OmniClientError(
            f"MAGI-2 resolution {resolution!r} requires {expected_width}x{expected_height}; "
            f"got {requested_size[0]}x{requested_size[1]}."
        )

    width, height = _RESOLUTION_PRESETS[resolution]
    return resolution, width, height


def _resolve_checkpoint_root(model: str, revision: str | None) -> str:
    path = Path(model).expanduser()
    if path.is_dir():
        root = path.resolve()
    else:
        normalized = model.strip().rstrip("/")
        prefix = "https://huggingface.co/"
        if normalized.lower().startswith(prefix):
            normalized = normalized[len(prefix) :].split("/tree/", 1)[0]
        if normalized.lower() != MAGI2_MODEL_ID.lower():
            raise ValueError(
                "MAGI-2 expects a local checkpoint directory or the official "
                f"model ID {MAGI2_MODEL_ID!r}; got {model!r}."
            )
        from vllm.transformers_utils.repo_utils import hf_api

        pinned_revision = revision or MAGI2_MODEL_REVISION
        logger.warning(
            "Resolving the pinned MAGI-2 snapshot from Hugging Face. Pre-download it for predictable startup."
        )
        root = Path(hf_api().snapshot_download(repo_id=normalized, revision=pinned_revision)).resolve()

    missing = [relative for relative in _REQUIRED_CHECKPOINT_PATHS if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Incomplete MAGI-2 checkpoint at {root}: missing " + ", ".join(missing))
    return str(root)


def _validate_native_topology(od_config: OmniDiffusionConfig) -> None:
    parallel = od_config.parallel_config
    unsupported: list[str] = []
    for name in (
        "pipeline_parallel_size",
        "ring_degree",
        "allgather_degree",
        "text_encoder_tp_size",
    ):
        value = getattr(parallel, name, 1)
        if value not in (None, 1):
            unsupported.append(f"{name}={value}")
    if getattr(parallel, "enable_expert_parallel", False):
        unsupported.append("enable_expert_parallel")
    if od_config.quantization_config is not None:
        unsupported.append("quantization")
    if unsupported:
        raise ValueError(
            "MAGI-2 Preview uses Ulysses sequence parallelism and MoE-head "
            "parallelism. Unsupported options: " + ", ".join(unsupported)
        )

    dp_size = int(getattr(parallel, "data_parallel_size", 1) or 1)
    tp_size = int(getattr(parallel, "tensor_parallel_size", 1) or 1)
    sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
    cfg_size = int(getattr(parallel, "cfg_parallel_size", 1) or 1)
    vae_pp_size = int(getattr(parallel, "vae_patch_parallel_size", 1) or 1)
    ulysses = int(getattr(parallel, "ulysses_degree", 1) or 1)
    if cfg_size not in (1, 2):
        raise ValueError(f"MAGI-2 has exactly two CFG branches; cfg_parallel_size must be 1 or 2, got {cfg_size}.")
    if sp_size != ulysses:
        raise ValueError(f"MAGI-2 requires sequence_parallel_size == ulysses_degree; got {sp_size} and {ulysses}.")

    attention_shards = tp_size * sp_size
    if MAGI2_PREVIEW_CONFIG.num_heads_q % attention_shards:
        raise ValueError(
            "MAGI-2 query heads must divide TP x SP; got "
            f"{MAGI2_PREVIEW_CONFIG.num_heads_q} heads, TP={tp_size}, SP={sp_size}."
        )
    if MAGI2_PREVIEW_CONFIG.num_heads_kv % attention_shards:
        raise ValueError(
            "MAGI-2 KV heads must divide TP x SP; got "
            f"{MAGI2_PREVIEW_CONFIG.num_heads_kv} heads, TP={tp_size}, SP={sp_size}."
        )
    dense_intermediate = (
        int(MAGI2_PREVIEW_CONFIG.hidden_size * MAGI2_PREVIEW_CONFIG.intermediate_factor * 2 / 3) // 128 * 128
    )
    tp_dimensions = {
        "hidden_size": MAGI2_PREVIEW_CONFIG.hidden_size,
        "dense_intermediate_size": dense_intermediate,
        "MoE heads": MAGI2_PREVIEW_CONFIG.moe.num_heads,
        "shared_expert_intermediate_size": MAGI2_PREVIEW_CONFIG.moe.shared_expert_intermediate_size,
        "modality_shared_expert_intermediate_size": (MAGI2_PREVIEW_CONFIG.moe.modality_shared_expert_intermediate_size),
    }
    invalid_tp_dimensions = [name for name, size in tp_dimensions.items() if size % tp_size]
    if invalid_tp_dimensions:
        raise ValueError(f"MAGI-2 tensor_parallel_size={tp_size} does not divide: " + ", ".join(invalid_tp_dimensions))

    configured_world_size = dp_size * cfg_size * tp_size * sp_size
    cpu_offload = bool(od_config.enable_cpu_offload)
    layerwise_offload = bool(od_config.enable_layerwise_offload)
    distributed_offload = bool(od_config.enable_distributed_layerwise_offload)
    if cpu_offload and not layerwise_offload:
        raise ValueError(
            "MAGI-2 already stages its auxiliary components from CPU, while "
            "the complete Preview transformer cannot fit on one qualified GPU. "
            "Combine --enable-cpu-offload with --enable-layerwise-offload, or "
            "use --enable-layerwise-offload alone."
        )
    if layerwise_offload and distributed_offload:
        raise ValueError("MAGI-2 ordinary and distributed layerwise offload are mutually exclusive")
    if layerwise_offload and configured_world_size != 1:
        raise ValueError(
            "MAGI-2 ordinary layerwise offload is a single-worker path; "
            f"got world_size={configured_world_size}. Use distributed layerwise "
            "offload for multi-worker layouts."
        )
    if layerwise_offload and getattr(parallel, "use_hsdp", False):
        raise ValueError("MAGI-2 ordinary layerwise offload cannot be combined with HSDP")
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != configured_world_size:
        raise ValueError(
            "MAGI-2 parallel dimensions do not cover the worker world; got "
            f"world_size={dist.get_world_size()}, DP={dp_size}, CFG={cfg_size}, TP={tp_size}, SP={sp_size}."
        )

    if vae_pp_size not in (1, configured_world_size):
        raise ValueError(
            "MAGI-2 VAE patch parallelism currently uses the complete DiT group; "
            f"expected vae_patch_parallel_size=1 or {configured_world_size}, got {vae_pp_size}."
        )

    dlo_allgather = bool(getattr(od_config, "dlo_use_allgather", True))
    if cfg_size > 1 and dp_size > 1:
        raise ValueError("MAGI-2 CFG parallelism is not yet combined with DLO data parallelism")
    if distributed_offload and getattr(parallel, "use_hsdp", False):
        raise ValueError("MAGI-2 HSDP and distributed layerwise offload are alternative transformer memory modes")
    if dp_size > 1 and not distributed_offload:
        raise ValueError("MAGI-2 data parallelism currently requires distributed layerwise offload")
    if dp_size > 1 and tp_size > 1:
        raise ValueError("MAGI-2 DLO data-parallel replicas currently require tensor_parallel_size=1")
    if distributed_offload and dlo_allgather:
        if dp_size <= 1:
            raise ValueError(
                "MAGI-2 DLO AllGather requires data_parallel_size > 1. SP ranks "
                "own different MoE-head shards; use --dlo-no-use-allgather for SP-only DLO."
            )
        if tp_size > 1:
            raise ValueError("MAGI-2 DLO AllGather does not support tensor_parallel_size > 1")

    allow_unsupported = _env_flag(
        _config_value(
            od_config,
            "magi2_allow_unsupported_topology",
            "MAGI2_ALLOW_UNSUPPORTED_TOPOLOGY",
        )
    )
    supported_single_worker = layerwise_offload and configured_world_size == 1
    if configured_world_size not in {4, 8} and not supported_single_worker and not allow_unsupported:
        raise ValueError(
            "MAGI-2 Preview is qualified for one worker with ordinary layerwise "
            "offload, or four/eight workers; got "
            f"DP={dp_size}, CFG={cfg_size}, TP={tp_size}, SP={sp_size} "
            f"(world_size={configured_world_size}). Set "
            "MAGI2_ALLOW_UNSUPPORTED_TOPOLOGY=1 only for controlled bring-up."
        )


def _seed_request(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _resolve_request_seed(sampling: object) -> int:
    seed = getattr(sampling, "seed", None)
    if seed is not None:
        return int(seed)
    generator = getattr(sampling, "generator", None)
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is not None and hasattr(generator, "initial_seed"):
        return int(generator.initial_seed())
    return 42


def _single_image(value: object) -> str | Image.Image | None:
    if value is None:
        return None
    if isinstance(value, list | tuple):
        if len(value) != 1:
            raise OmniClientError(f"MAGI-2 accepts at most one input image, got {len(value)}")
        value = value[0]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, str | Image.Image):
        return value
    raise OmniClientError(
        f"MAGI-2 image input must be a PIL image, path, or one-element image list; got {type(value).__name__}."
    )


def _resizepad(image: Image.Image, height: int, width: int) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"Invalid input image size {image.size}")
    scale = min(width / image.width, height / image.height)
    target_width = max(1, int(round(image.width * scale)))
    target_height = max(1, int(round(image.height * scale)))
    resized = image.convert("RGB").resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(
        resized,
        ((width - target_width) // 2, (height - target_height) // 2),
    )
    return canvas


def _resize_video(video: np.ndarray, width: int, height: int) -> np.ndarray:
    frames = torch.from_numpy(video)
    if frames.ndim != 4:
        raise ValueError(f"Expected video shape (T,H,W,C), got {tuple(frames.shape)}")
    was_uint8 = frames.dtype == torch.uint8
    frames = frames.float().div_(255.0) if was_uint8 else frames.float()
    frames = torch.nn.functional.interpolate(
        frames.permute(3, 0, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(1, 2, 3, 0)
    if was_uint8:
        return frames.clamp_(0, 1).mul_(255).round_().byte().numpy()
    return frames.numpy()


def _magi2_post_process(output: object) -> object:
    return dict(output) if isinstance(output, Mapping) else output


def get_magi2_post_process_func(_od_config: OmniDiffusionConfig):
    return _magi2_post_process


class _Magi2StagedComponent(nn.Module):
    """Keep one immutable CPU copy and materialize a component per stage.

    MAGI-2's text encoder and three codec components are mutually exclusive at
    runtime.  Giving each one an explicit lifecycle lets both layerwise
    offload backends leave them on the host at startup instead of eagerly
    co-residing roughly 60 GiB of auxiliary weights on the output rank.
    """

    def __init__(
        self,
        module: nn.Module,
        device: str | torch.device,
        *,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.module = module
        self._stager = PinnedModuleStager(
            module,
            torch.device(device),
            pin_memory=pin_memory,
        )

    def load_to_device(self) -> None:
        self._stager.load()

    def offload_to_cpu(self) -> None:
        self._stager.offload()

    def to(self, *_args: object, **_kwargs: object) -> _Magi2StagedComponent:
        """Keep lifecycle-managed auxiliaries on the host between stages.

        The generic HSDP loader places discovered non-DiT components on the
        execution device after sharding. MAGI-2 owns those components through
        ``PinnedModuleStager`` instead, so a direct placement request must not
        make Qwen and all three codecs resident together.
        """

        return self

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)


class Magi2Pipeline(
    nn.Module,
    ProgressBarMixin,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportAudioOutput,
):
    """Native MAGI-2 Preview text/image-to-video-and-audio pipeline.

    One pipeline instance is supported per worker process. Initialization sets
    process-wide deterministic state, so tests or deployments that need more
    than one pipeline must isolate them in separate worker processes.
    """

    support_image_input: ClassVar[bool] = True
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = MAGI2_AUDIO_SAMPLE_RATE
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = [
        "image_vae",
        "video_decoder",
        "audio_decoder",
    ]
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        on_demand_component_paths=frozenset(
            {
                "text_encoder",
                "image_vae",
                "video_decoder",
                "audio_decoder",
            }
        )
    )

    @staticmethod
    def _remap_ckpt_key(checkpoint_key: str) -> str | None:
        """Map released Preview keys to the native pipeline namespace.

        The DLO host-weight planner looks this hook up by name when it builds
        a direct mmap plan; the ordinary loader uses ``load_weights`` below.
        """

        checkpoint_key = checkpoint_key.removeprefix("transformer.")
        if checkpoint_key.startswith(("block.", "pre_adapter.", "post_adapter.")):
            return f"transformer.{checkpoint_key}"
        return None

    def __init__(self, od_config: OmniDiffusionConfig, **kwargs: object) -> None:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MAGI-2 pipeline initialization argument(s): {names}")
        super().__init__()
        if not od_config.model:
            raise ValueError("MAGI-2 requires od_config.model")
        _validate_native_topology(od_config)
        if not current_omni_platform.is_cuda() or not current_omni_platform.is_available():
            raise RuntimeError("MAGI-2 Preview requires CUDA GPUs")

        self.od_config = od_config
        self.dtype = od_config.dtype or torch.bfloat16
        self.device_str = f"cuda:{torch.accelerator.current_device_index()}"
        self.checkpoint_root = _resolve_checkpoint_root(
            str(od_config.model),
            od_config.revision,
        )
        self.deterministic = _env_flag(
            _config_value(
                od_config,
                "magi2_deterministic",
                "MAGI2_DETERMINISTIC",
            ),
            default=bool(getattr(od_config, "fa_deterministic", False)),
        )
        os.environ["MAGI2_DETERMINISTIC"] = str(int(self.deterministic))
        # Each diffusion worker owns one startup-fixed pipeline. Configure the
        # process-wide PyTorch mode once instead of mutating it per request;
        # warn-only avoids rejecting otherwise valid third-party CUDA kernels.
        torch.use_deterministic_algorithms(self.deterministic, warn_only=True)
        self._parallel_group = get_magi2_replica_group(
            int(getattr(od_config.parallel_config, "data_parallel_size", 1) or 1)
        )
        self._is_output_rank = self._parallel_group.rank == 0
        self._offload_aux_after_use = True
        self._transformer_is_layerwise_offloaded = bool(
            od_config.enable_layerwise_offload or od_config.enable_distributed_layerwise_offload
        )
        self._transformer_is_hsdp = bool(getattr(od_config.parallel_config, "use_hsdp", False))
        self._distributed_video_decode = int(od_config.parallel_config.vae_patch_parallel_size) > 1

        # Importing here keeps config-only model detection light.  The class is
        # an in-tree implementation; there is no dynamic remote-code import.
        from .modeling_magi2 import Magi2PreviewTransformer

        MAGI2_PREVIEW_CONFIG.validate()
        mmap_dlo = bool(
            od_config.enable_distributed_layerwise_offload and getattr(od_config, "dlo_use_allgather", True)
        )
        if mmap_dlo:
            # AllGather DLO binds checkpoint tensors as mmap views and copies
            # only this rank's orthogonal DP shard.  Constructing the 212-GiB
            # Preview transformer on CPU first would defeat that memory model.
            with torch.device("meta"):
                self.transformer = Magi2PreviewTransformer(MAGI2_PREVIEW_CONFIG)
            # The mmap path is inference-only. Removing autograd ownership here
            # lets the shared DLO backend shard views without a MAGI-specific
            # detach branch.
            self.transformer.requires_grad_(False)
        else:
            self.transformer = Magi2PreviewTransformer(MAGI2_PREVIEW_CONFIG)
        self.data_proxy = Magi2DataProxy()
        self.sampler = Magi2PreviewSampler(
            self.transformer,
            self.data_proxy,
            device=self.device_str,
            dtype=self.dtype,
        )

        root = Path(self.checkpoint_root)
        # Auxiliaries start on CPU and are staged one at a time. Distributed
        # TurboVAE decode is the only case that materializes a codec on every
        # rank; Qwen, image VAE, and Oobleck remain output-rank-only.
        with torch.device("cpu"):
            if self._is_output_rank:
                text_encoder: nn.Module = Magi2Qwen35TextEncoder(
                    str(root / "text_encoder"),
                    dtype=self.dtype,
                )
                from vllm_omni.diffusion.models.lance.wan_vae import LanceWanVAE

                image_vae: nn.Module = LanceWanVAE(
                    str(root / "vae" / "Wan2.2_VAE.pth"),
                    dtype=torch.float32,
                    device="cpu",
                )
                audio_decoder: nn.Module = Magi2AudioDecoder(
                    root / "stable-audio-open-1.0",
                    device="cpu",
                    dtype=torch.float32,
                )
            else:
                text_encoder = nn.Identity()
                image_vae = nn.Identity()
                audio_decoder = nn.Identity()

            if self._is_output_rank or self._distributed_video_decode:
                video_decoder: nn.Module = Magi2TurboVAEDecoder(
                    root / "turbo_vae" / "TurboV3-Wan22-TinyShallow_7_7.json",
                    root / "turbo_vae" / "checkpoint.ckpt",
                    device="cpu",
                    dtype=self.dtype,
                )
            else:
                video_decoder = nn.Identity()

        pin_memory = bool(getattr(od_config, "pin_cpu_memory", True))
        self.text_encoder = _Magi2StagedComponent(
            text_encoder,
            self.device_str,
            pin_memory=pin_memory,
        )
        self.image_vae = _Magi2StagedComponent(
            image_vae,
            self.device_str,
            pin_memory=pin_memory,
        )
        self.video_decoder = _Magi2StagedComponent(
            video_decoder,
            self.device_str,
            pin_memory=pin_memory,
        )
        self.audio_decoder = _Magi2StagedComponent(
            audio_decoder,
            self.device_str,
            pin_memory=pin_memory,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=2 * MAGI2_GENERATION_CONFIG.video_vae_stride[1])
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.checkpoint_root,
                subfolder="preview",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=[
                "_encode_prompts",
                "_encode_reference_image",
                "_pool_figure_token",
                "sampler.sample",
                "_decode_video",
                "_decode_audio",
            ],
            enable_diffusion_pipeline_profiler=bool(getattr(od_config, "enable_diffusion_pipeline_profiler", False)),
        )

    @property
    def vae(self) -> nn.Module:
        """Expose TurboVAE through the shared distributed-VAE contract."""

        return self.video_decoder.module

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        prefix = "transformer."

        def stripped_weights():
            for name, tensor in weights:
                if not name.startswith(prefix):
                    raise ValueError(f"Unexpected MAGI-2 checkpoint key {name!r}")
                yield name[len(prefix) :], tensor

        if not hasattr(self.transformer, "load_weights"):
            raise RuntimeError("Native MAGI-2 transformer has no strict weight loader")
        loaded = self.transformer.load_weights(stripped_weights())
        return {prefix + name for name in loaded}

    @contextmanager
    def _component_on_device(self, component: _Magi2StagedComponent):
        component.load_to_device()
        try:
            yield component.module
        finally:
            if self._offload_aux_after_use:
                component.offload_to_cpu()

    @contextmanager
    def _conditioning_memory_window(self):
        """Make room for the 27B text encoder on the output rank.

        The text encoder and a resident Preview shard may not fit together on
        the qualified devices.  Layerwise offloading already keeps the DiT on
        the host; otherwise stage the output rank's shard to CPU for the two
        prompt encodes and restore it before any model collective begins.
        """

        stage_transformer = (
            self._is_output_rank
            and not self._transformer_is_layerwise_offloaded
            and not getattr(self, "_transformer_is_hsdp", False)
        )
        if stage_transformer:
            self.transformer.to("cpu")
            torch.accelerator.empty_cache()
        try:
            yield
        finally:
            if stage_transformer:
                self.transformer.to(self.device_str)

    def _broadcast_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor:
        group = self._parallel_group
        if group.world_size == 1:
            if tensor is None:
                raise RuntimeError("source rank did not provide a tensor")
            return tensor.to(self.device_str)

        source_global_rank = dist.get_global_rank(group.group, 0)
        metadata: list[tuple[tuple[int, ...], torch.dtype] | None] = [
            (tuple(tensor.shape), tensor.dtype) if group.rank == 0 else None
        ]
        dist.broadcast_object_list(
            metadata,
            src=source_global_rank,
            group=group.group,
            device=torch.device(self.device_str),
        )
        resolved_metadata = metadata[0]
        if resolved_metadata is None:
            raise RuntimeError("source rank did not broadcast tensor metadata")
        shape, dtype = resolved_metadata
        if group.rank == 0:
            assert tensor is not None
            output = tensor.to(device=self.device_str, dtype=dtype).contiguous()
        else:
            output = torch.empty(shape, device=self.device_str, dtype=dtype)
        dist.broadcast(output, src=source_global_rank, group=group.group)
        return output

    def _encode_prompts(self, prompts: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
        encoded: list[torch.Tensor | None] = [None] * len(prompts)
        if self._is_output_rank:
            with self._component_on_device(self.text_encoder) as text_encoder:
                # Call ``encode`` directly so framework on-demand hooks do not
                # evict the encoder between the positive and negative prompt.
                encoded = [text_encoder.encode(prompt).to(self.dtype) for prompt in prompts]
        return tuple(self._broadcast_tensor(value) for value in encoded)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self._encode_prompts((prompt,))[0]

    @staticmethod
    def _ensure_figure_token(prompt: str) -> str:
        try:
            value = json.loads(prompt)
            if not isinstance(value, dict):
                raise ValueError
            value["reference_layer"] = ["The first frame refers to <Figure 1>"]
            return json.dumps(value, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            return prompt + "\nreference_layer:The first frame refers to <Figure 1>"

    def _encode_reference_image(
        self,
        image: str | Image.Image,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent: torch.Tensor | None = None
        if self._is_output_rank:
            pil_image = load_image(image)
            max_length = max(height, width)
            if pil_image.width > pil_image.height:
                target_width = max_length
                target_height = int(pil_image.height * max_length / pil_image.width)
            else:
                target_height = max_length
                target_width = int(pil_image.width * max_length / pil_image.height)
            padded = _resizepad(pil_image, target_height, target_width)
            pixels = self.video_processor.preprocess(
                padded,
                height=target_height,
                width=target_width,
            )
            pixels = pixels[:, :3].to(torch.float32).unsqueeze(2)
            with self._component_on_device(self.image_vae) as image_vae:
                latent = image_vae.encode_video(
                    pixels.to(self.device_str),
                    use_sample=False,
                )
        latent = self._broadcast_tensor(latent)
        feature = latent.unsqueeze(1)
        lengths = torch.tensor(
            [[[latent.shape[-2], latent.shape[-1]]]],
            dtype=torch.long,
            device=latent.device,
        )
        return feature, lengths

    def _pool_figure_token(
        self,
        prompt: str,
        context: torch.Tensor,
    ) -> torch.Tensor:
        special: torch.Tensor | None = None
        if self._is_output_rank:
            # This helper only tokenizes the prompt and pools rows that are
            # already present in ``context``; it does not read encoder weights.
            # Keep it outside the staged encoder window so CPU-only pooling
            # cannot trigger an unnecessary device transfer.
            special = self.text_encoder.module.pool_figure_tokens(
                prompt,
                ["<Figure 1>"],
                context,
            ).unsqueeze(0)
        return self._broadcast_tensor(special)

    def _decode_video(self, latent: torch.Tensor) -> np.ndarray | None:
        distributed_video_decode = getattr(self, "_distributed_video_decode", False)
        if not self._is_output_rank and not distributed_video_decode:
            return None
        with self._component_on_device(self.video_decoder) as video_decoder:
            decoded = video_decoder.decode(
                latent.to(self.device_str, dtype=self.dtype),
                output_offload=True,
            )
        if not self._is_output_rank:
            return None
        if decoded.ndim != 5:
            raise RuntimeError(f"TurboVAE returned unexpected shape {tuple(decoded.shape)}")
        video = decoded[0].float().mul_(0.5).add_(0.5).clamp_(0, 1)
        return video.permute(1, 2, 3, 0).mul_(255).byte().cpu().numpy()

    def _decode_audio(self, latent: torch.Tensor) -> np.ndarray | None:
        if not self._is_output_rank:
            return None
        with self._component_on_device(self.audio_decoder) as audio_decoder:
            waveform = audio_decoder.decode(latent.transpose(1, 2).to(self.device_str))
        audio = waveform.squeeze(0).transpose(0, 1).float().cpu().numpy()
        # MAGI emits 25 latent frames/s while Stable Audio's native latent
        # rate is 44,100/2,048. Preserve the reference Fourier resampling.
        from scipy.signal import resample

        target_length = int(audio.shape[0] * 441 / 512)
        return resample(audio, target_length)

    @torch.inference_mode()
    def _evaluate_preview(
        self,
        *,
        prompt: str,
        image: str | Image.Image | None,
        width: int,
        height: int,
        num_inference_steps: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        ref_image_feat = None
        ref_image_feat_len = None
        if image is not None:
            ref_image_feat, ref_image_feat_len = self._encode_reference_image(
                image,
                height,
                width,
            )
            prompt = self._ensure_figure_token(prompt)

        negative_prompt = os.environ.get(
            "MAGI2_NEGATIVE_PROMPT",
            DEFAULT_NEGATIVE_PROMPT,
        )
        with self._conditioning_memory_window():
            context, context_null = self._encode_prompts((prompt, negative_prompt))
            special = self._pool_figure_token(prompt, context) if ref_image_feat is not None else None

        latent_height = height // MAGI2_GENERATION_CONFIG.video_vae_stride[1]
        latent_width = width // MAGI2_GENERATION_CONFIG.video_vae_stride[2]
        public_frame_length = int(round(MAGI2_GENERATION_CONFIG.duration_seconds * MAGI2_GENERATION_CONFIG.fps))
        if public_frame_length != MAGI2_GENERATION_CONFIG.output_frames:
            raise RuntimeError(
                "MAGI-2 generation config is inconsistent: duration_seconds * fps must equal output_frames"
            )
        # The Preview DiT/decoder operates on the internal 25-FPS timeline,
        # while the public output contract is 12.5 FPS. The internal timeline
        # is therefore twice as long; the decoded result is muxed at 12.5 FPS.
        video_frame_length = public_frame_length * 2
        latent_length = (video_frame_length - 1) // MAGI2_GENERATION_CONFIG.video_vae_stride[0] + 1
        audio_length = int(round(MAGI2_GENERATION_CONFIG.duration_seconds * MAGI2_GENERATION_CONFIG.audio_latent_fps))
        # Draw order is parity-critical: video noise precedes audio noise.
        latent = torch.randn(
            1,
            MAGI2_GENERATION_CONFIG.video_latent_channels,
            latent_length,
            latent_height,
            latent_width,
            device=self.device_str,
            dtype=torch.float32,
        )
        audio_latent = torch.randn(
            1,
            audio_length,
            MAGI2_GENERATION_CONFIG.audio_latent_channels,
            device=self.device_str,
            dtype=torch.float32,
        )
        ref_audio = torch.zeros(
            1,
            0,
            MAGI2_GENERATION_CONFIG.audio_latent_channels,
            device=self.device_str,
            dtype=torch.float32,
        )
        video_scheduler, audio_scheduler = build_magi2_preview_schedulers(
            num_inference_steps,
            device=self.device_str,
            shift=MAGI2_GENERATION_CONFIG.shift,
        )
        sampler_input = SamplerInput(
            video_t_list=video_scheduler.timesteps,
            audio_t_list=audio_scheduler.timesteps,
            latent=latent,
            audio_latent=audio_latent,
            txt_feat=context,
            null_txt_feat=context_null,
            ref_audio_feat=ref_audio,
            ref_video_feat=None,
            video_scheduler=video_scheduler,
            audio_scheduler=audio_scheduler,
            cfg_config=CFGConfig(
                video_txt_guidance_scale=(MAGI2_GENERATION_CONFIG.video_guidance_scale),
                audio_txt_guidance_scale=(MAGI2_GENERATION_CONFIG.audio_guidance_scale),
            ),
            ref_image_feat=ref_image_feat,
            ref_image_feat_len=ref_image_feat_len,
            ref_image_special_token_embedding=special,
        )
        latent, audio_latent = self.sampler.sample(sampler_input)
        video = self._decode_video(latent)
        audio = self._decode_audio(audio_latent)
        return video, audio

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise OmniClientError("MAGI-2 currently supports one request at a time")

        raw_prompt = req.prompts[0]
        image_value: object = None
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
            multimodal: Mapping[str, object] = {}
        elif isinstance(raw_prompt, Mapping):
            prompt = str(raw_prompt.get("prompt") or "")
            multimodal = raw_prompt.get("multi_modal_data") or {}
            image_value = raw_prompt.get("image_path")
        else:
            raise OmniClientError(f"Unsupported MAGI-2 prompt type: {type(raw_prompt).__name__}")
        if not prompt:
            raise OmniClientError("MAGI-2 requires a non-empty prompt")
        if not isinstance(multimodal, Mapping):
            raise OmniClientError("MAGI-2 multi_modal_data must be a mapping")
        unexpected = set(multimodal) - {"image"}
        if unexpected:
            raise OmniClientError(
                "MAGI-2 accepts text and at most one still image; unsupported "
                "modalities: " + ", ".join(sorted(unexpected))
            )
        if "image" in multimodal:
            image_value = multimodal["image"]

        sampling = req.sampling_params
        num_outputs = int(getattr(sampling, "num_outputs_per_prompt", 1) or 1)
        if num_outputs != 1:
            raise OmniClientError(f"MAGI-2 supports exactly one output per prompt; got {num_outputs}")
        extra = sampling.extra_args or {}
        if image_value is None:
            image_value = extra.get("image_path")
        image = _single_image(image_value)

        _, width, height = _resolve_native_resolution(sampling, extra)
        if _env_flag(extra.get("use_refiner")):
            raise OmniClientError(
                "The MAGI-2 1080p refiner is a separate model and is not yet enabled by the native Preview pipeline."
            )
        seconds = float(extra.get("seconds", extra.get("duration", 10.0)))
        if not math.isfinite(seconds) or seconds != 10.0:
            raise OmniClientError("MAGI-2 Preview supports 10-second clips only")

        requested_fps = getattr(sampling, "resolved_frame_rate", None)
        if requested_fps is None:
            requested_fps = getattr(sampling, "frame_rate", None)
        if requested_fps is None:
            requested_fps = getattr(sampling, "fps", None)
        if requested_fps is not None and float(requested_fps) != 12.5:
            raise OmniClientError(f"MAGI-2 Preview output is fixed at 12.5 fps; got {requested_fps}.")
        requested_frames = getattr(sampling, "num_frames", None)
        # ``1`` is the shared image-model engine default and therefore means
        # unset here. Serving validates an explicitly supplied frame count
        # before applying MAGI-2's model-owned default.
        if requested_frames not in {None, 1, 125}:
            raise OmniClientError(f"MAGI-2 Preview output is fixed at 125 frames; got {requested_frames}.")
        requested_steps = sampling.num_inference_steps
        steps = MAGI2_GENERATION_CONFIG.preview_steps if requested_steps is None else int(requested_steps)
        if steps <= 0:
            raise OmniClientError("MAGI-2 inference steps must be positive")

        output_width = extra.get("output_width")
        output_height = extra.get("output_height")
        if (output_width is None) != (output_height is None):
            raise OmniClientError("MAGI-2 output resize requires output_width and output_height")
        if output_width is not None:
            try:
                output_width = int(output_width)
                output_height = int(output_height)
            except (TypeError, ValueError) as exc:
                raise OmniClientError("MAGI-2 output dimensions must be positive integers") from exc
            if output_width <= 0 or output_height <= 0:
                raise OmniClientError("MAGI-2 output dimensions must be positive integers")

        requested_deterministic = extra.get("deterministic")
        if requested_deterministic is not None and _env_flag(requested_deterministic) != self.deterministic:
            raise OmniClientError(
                "MAGI-2 deterministic mode is fixed at worker startup; restart "
                "with MAGI2_DETERMINISTIC set to the requested value."
            )
        seed = _resolve_request_seed(sampling)
        _seed_request(seed)

        has_cuda = current_omni_platform.is_cuda() and current_omni_platform.is_available()
        device_index = torch.accelerator.current_device_index() if has_cuda else None
        # Sampling reserved memory is qualification instrumentation, not part
        # of ordinary serving. It starts only when the pipeline profiler is
        # explicitly enabled, so every CUDA request avoids a 20 Hz thread.
        monitor = (
            _PeakReservedMonitor(device_index)
            if device_index is not None and getattr(self, "enable_diffusion_pipeline_profiler", False)
            else None
        )
        monitor_started = False
        started = time.perf_counter()
        try:
            if monitor is not None:
                monitor.start()
                monitor_started = True
                torch.accelerator.synchronize()
            started = time.perf_counter()
            video, audio = self._evaluate_preview(
                prompt=prompt,
                image=image,
                width=width,
                height=height,
                num_inference_steps=steps,
            )
            if has_cuda:
                torch.accelerator.synchronize()
        finally:
            if monitor_started:
                monitor.stop()
        elapsed = time.perf_counter() - started

        if video is not None and output_width is not None:
            video = _resize_video(video, output_width, output_height)

        peak_memory_mb = monitor.peak_bytes / 1024**2 if monitor is not None else 0.0
        if has_cuda and dist.is_available() and dist.is_initialized() and self._parallel_group.world_size > 1:
            peak = torch.tensor(peak_memory_mb, device=self.device_str)
            dist.all_reduce(
                peak,
                op=dist.ReduceOp.MAX,
                group=self._parallel_group.group,
            )
            peak_memory_mb = float(peak.item())

        stage_durations = {"magi2_preview_e2e": elapsed}
        if getattr(self, "enable_diffusion_pipeline_profiler", False):
            stage_durations.update(self.stage_durations)

        return DiffusionOutput(
            output={
                "payload": {"video": video, "audio": audio},
                "metadata": {
                    "video": {"fps": 12.5},
                    "audio": {"sample_rate": MAGI2_AUDIO_SAMPLE_RATE},
                },
            },
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
        )
