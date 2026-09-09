# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sana-WM pipeline integration.

This module wires the registry-visible surface, release-layout validation, and
the Stage-1 sampling loop. The DiT, Gated-DeltaNet and camera stack are
vLLM-Omni-native layers (``sana_wm_transformer.py`` / ``ucpe.py``); nothing here
calls into NVlabs code at runtime.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.ltx2.ltx2_latents import (
    denormalize_latents,
    normalize_latents,
    pack_latents,
    resolve_video_latent_shape,
    unpack_latents,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.sana_wm.camera_control import (
    SanaWmCameraCondition,
    build_plucker_condition,
)
from vllm_omni.diffusion.models.sana_wm.config import (
    SANA_WM_VAE_SPATIAL_COMPRESSION,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
    SanaWmConfig,
)
from vllm_omni.diffusion.models.sana_wm.request import normalize_sana_wm_payload
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SANA_WM_STAGE1_PROMPT_CHANNELS,
    SanaWmTransformer3DModel,
)
from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)

# The loader expects the standard diffusers layout (transformer/config.json +
# transformer/diffusion_pytorch_model.safetensors + model_index.json), so this
# repo is an offline conversion of the NVlabs release into that layout. It holds
# Stage-1 only — transformer/ + vae/, with model_index.json naming
# SanaWmPipeline — because startup prefetches the whole model repo
# (allow_patterns=["*"]); a repo that also carried the refiner would pull tens of
# GB this path never loads.
SANA_WM_MODEL_ID = "BBBBruce/SANA-WM_bidirectional-stage1-diffusers"

SANA_WM_STAGE1_DIT_FILE = "transformer/diffusion_pytorch_model.safetensors"
SANA_WM_CONFIG_FILE = "transformer/config.json"
SANA_WM_VAE_CONFIG_FILE = "vae/config.json"
SANA_WM_VAE_WEIGHT_FILE = "vae/diffusion_pytorch_model.safetensors"
SANA_WM_STAGE1_TEXT_ENCODER_ID = "google/gemma-2-2b-it"
SANA_WM_STAGE1_TEXT_ENCODER_FALLBACK_ID = "Efficient-Large-Model/gemma-2-2b-it"
SANA_WM_STAGE1_TEXT_ENCODER_ENV = "VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER"
SANA_WM_OUTPUT_HEIGHT = 704
SANA_WM_OUTPUT_WIDTH = 1280
SANA_WM_NUM_TRAIN_TIMESTEPS = 1000
# Maximum supported clip length (20s @16fps) — the native envelope, not a
# request default. Request-side defaults live in
# ``models/sana_wm/request.py``.
SANA_WM_NATIVE_NUM_FRAMES = 321
# Fail-fast latent-token cap for the native Stage-1 path, sized to the native
# 704x1280 output at the maximum clip length so it only rejects genuinely
# oversized requests. Overridable per-request via ``sana_wm_native_max_tokens``.
SANA_WM_NATIVE_MAX_TOKENS = (
    ((SANA_WM_NATIVE_NUM_FRAMES - 1) // SANA_WM_VAE_TEMPORAL_COMPRESSION + 1)
    * (SANA_WM_OUTPUT_HEIGHT // SANA_WM_VAE_SPATIAL_COMPRESSION)
    * (SANA_WM_OUTPUT_WIDTH // SANA_WM_VAE_SPATIAL_COMPRESSION)
)
# The NVlabs production inference settings for this release. They belong in the
# model rather than in a deploy YAML: single-stage diffusion models resolve
# through ``create_default_diffusion``, which never reads a YAML's
# ``default_sampling_params``, so a caller that omits either field lands here.
SANA_WM_DEFAULT_NUM_INFERENCE_STEPS = 60
SANA_WM_DEFAULT_GUIDANCE_SCALE = 5.0

SANA_WM_STAGE1_PATTERNS = (
    SANA_WM_CONFIG_FILE,
    SANA_WM_STAGE1_DIT_FILE,
    SANA_WM_VAE_CONFIG_FILE,
    SANA_WM_VAE_WEIGHT_FILE,
)
SANA_WM_STAGE1_DIT_SUBFOLDER = "transformer"
SANA_WM_STAGE1_DIT_BASENAME = Path(SANA_WM_STAGE1_DIT_FILE).name


@dataclass(frozen=True)
class SanaWmLocalPaths:
    """Resolved local file paths for a SANA-WM snapshot."""

    root: Path
    config: Path
    stage1_dit: Path
    vae_config: Path
    vae_weights: Path


@dataclass(frozen=True)
class SanaWmNativeParams:
    """Resolved generation settings for one Stage-1 request.

    Built by ``_native_params`` from the request payload and sampling params,
    and consumed by ``_run_native_backend`` — this is the production path.
    """

    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    seed: int
    # Defaults to the NVlabs production scale, so a caller that never mentions
    # guidance still gets the two-branch CFG forward the model was tuned for.
    # An explicit ``guidance_scale <= 1.0`` still selects the single-branch path.
    cfg_scale: float = SANA_WM_DEFAULT_GUIDANCE_SCALE


def build_sana_wm_download_patterns() -> tuple[str, ...]:
    """Return the minimal HF allow-patterns needed for SANA-WM."""

    return tuple(SANA_WM_STAGE1_PATTERNS)


def resolve_sana_wm_local_paths(snapshot_dir: str | Path) -> SanaWmLocalPaths:
    root = Path(snapshot_dir)
    return SanaWmLocalPaths(
        root=root,
        config=root / SANA_WM_CONFIG_FILE,
        stage1_dit=root / SANA_WM_STAGE1_DIT_FILE,
        vae_config=root / SANA_WM_VAE_CONFIG_FILE,
        vae_weights=root / SANA_WM_VAE_WEIGHT_FILE,
    )


def validate_sana_wm_local_paths(paths: SanaWmLocalPaths) -> None:
    required = [
        paths.config,
        paths.stage1_dit,
        paths.vae_config,
        paths.vae_weights,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing SANA-WM checkpoint files: {joined}")


def resolve_or_download_sana_wm_checkpoint(
    model: str = SANA_WM_MODEL_ID,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> SanaWmLocalPaths:
    """Resolve a local SANA-WM tree or download the required HF files."""

    root = Path(model)
    if root.is_dir():
        snapshot_dir = root
    else:
        snapshot_dir = Path(
            download_weights_from_hf_specific(
                model,
                cache_dir,
                list(build_sana_wm_download_patterns()),
                revision=revision,
                require_all=True,
            )
        )
    paths = resolve_sana_wm_local_paths(snapshot_dir)
    validate_sana_wm_local_paths(paths)
    return paths


def build_sana_wm_output_envelope(
    *,
    output: Any,
    output_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Wrap a Stage-1 result in the canonical output envelope.

    ``normalize_diffusion_postprocess_output`` splits ``{"payload": ...,
    "metadata": ...}`` into the API-facing payload and the metadata groups, so
    the model-specific diagnostics ride along under a ``sana_wm`` group instead
    of the removed ``DiffusionOutput.custom_output`` field.
    """
    payload_key = "latents" if output_type == "latent" else "video"
    return {"payload": {payload_key: output}, "metadata": {"sana_wm": metadata}}


def get_sana_wm_pre_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def pre_process_func(request: Any) -> Any:
        from vllm_omni.inputs.data import OmniTextPrompt

        sampling_params = getattr(request, "sampling_params", None)
        prompt = request.prompt
        prompt_mapping = OmniTextPrompt(prompt=prompt) if isinstance(prompt, str) else prompt
        prompt_mapping = dict(prompt_mapping)

        # The camera block travels as extra_params["sana_wm"], which serving
        # merges into extra_args. Offline callers may also put it straight on
        # the prompt.
        if sampling_params is not None:
            extra_args = getattr(sampling_params, "extra_args", None)
            if isinstance(extra_args, Mapping):
                block = extra_args.get("sana_wm")
                if isinstance(block, Mapping):
                    merged = dict(block)
                    merged.update(prompt_mapping.get("sana_wm") or {})
                    prompt_mapping["sana_wm"] = merged
            # Geometry always comes from the standard sampling fields.
            for field in ("num_frames", "height", "width"):
                value = getattr(sampling_params, field, None)
                if value is not None and not (field == "num_frames" and value == 1):
                    prompt_mapping[field] = value

        request.prompt = normalize_sana_wm_payload(prompt_mapping)

        # Preprocess is the only place the request is normalised, so publish the
        # canonical geometry back onto sampling_params for the rest of the stack.
        if sampling_params is not None:
            payload = request.prompt["additional_information"]["sana_wm"]
            sampling_params.num_frames = payload["num_frames"]
            sampling_params.height = payload["height"]
            sampling_params.width = payload["width"]
        return request

    return pre_process_func


class SanaWmPipeline(
    nn.Module,
    CFGParallelMixin,
    SupportImageInput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """Stage-1 SANA-WM image-to-video pipeline."""

    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"
    # Skip the startup warmup: the generic dummy request carries a blank image
    # with no camera poses and no action string, which SANA-WM's request
    # validation always rejects. The first real request does the warming.
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig | None = None, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix

        self.sana_wm_config = SanaWmConfig()
        self.quant_config = getattr(od_config, "quantization_config", None) if od_config is not None else None

        self.tokenizer: Any | None = None
        self.text_encoder: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.release_paths: SanaWmLocalPaths | None = None
        self.weights_sources = []
        if od_config is not None and od_config.model is not None:
            # Resolve the checkpoint once at construction (the standard
            # startup-time flow, like every other diffusion pipeline) so the
            # transformer below is built with the transformer/config.json shape
            # rather than defaults.
            self.resolve_checkpoint()
            self.weights_sources = [
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder=SANA_WM_STAGE1_DIT_SUBFOLDER,
                    revision=od_config.revision,
                    # The subfolder file carries module-local keys (standard
                    # diffusers layout); prefix them into the pipeline
                    # namespace for the loader's strict-coverage check.
                    prefix="transformer.",
                    fall_back_to_pt=False,
                    allow_patterns_overrides=[SANA_WM_STAGE1_DIT_BASENAME],
                )
            ]
        # Built eagerly: the loader constructs the pipeline under the target
        # device / default-dtype context, and the weight loader then streams
        # checkpoint tensors directly into these modules.
        self.transformer = SanaWmTransformer3DModel(
            config=self.sana_wm_config,
            quant_config=self.quant_config,
            prefix=f"{prefix}.transformer" if prefix else "transformer",
        )
        self.device = get_local_device()
        if od_config is not None and od_config.model is not None:
            self._build_aux_components()

    def resolve_checkpoint(self) -> SanaWmLocalPaths:
        if self.release_paths is not None:
            return self.release_paths
        if self.od_config is None or self.od_config.model is None:
            raise ValueError("Sana-WM checkpoint resolution requires od_config.model.")
        self.release_paths = resolve_or_download_sana_wm_checkpoint(
            self.od_config.model,
            revision=self.od_config.revision,
        )
        self.sana_wm_config = SanaWmConfig.from_json(self.release_paths.config)
        return self.release_paths

    @staticmethod
    def _extra_args(sampling_params: Any | None) -> dict[str, Any]:
        extra_args = getattr(sampling_params, "extra_args", None) if sampling_params is not None else None
        return dict(extra_args or {})

    def _native_params(self, payload: dict[str, Any], sampling_params: Any | None) -> SanaWmNativeParams:
        extra_args = self._extra_args(sampling_params)
        height = int(getattr(sampling_params, "height", None) or payload["height"])
        width = int(getattr(sampling_params, "width", None) or payload["width"])
        num_frames = int(getattr(sampling_params, "num_frames", None) or payload["num_frames"])
        steps = int(
            getattr(sampling_params, "num_inference_steps", None)
            or extra_args.get("num_inference_steps")
            or SANA_WM_DEFAULT_NUM_INFERENCE_STEPS
        )
        seed = int(getattr(sampling_params, "seed", None) or extra_args.get("seed", 0))
        if min(height, width, num_frames, steps) <= 0:
            raise ValueError("Sana-WM native height, width, num_frames, and steps must be positive.")
        cfg_scale = SANA_WM_DEFAULT_GUIDANCE_SCALE
        if sampling_params is not None and getattr(sampling_params, "guidance_scale_provided", False):
            cfg_scale = float(getattr(sampling_params, "guidance_scale", 1.0) or 1.0)
        return SanaWmNativeParams(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            seed=seed,
            cfg_scale=cfg_scale,
        )

    def _native_dtype(self, device: torch.device) -> torch.dtype:
        dtype = getattr(self.od_config, "dtype", None) if self.od_config is not None else None
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.float32 if device.type == "cpu" else torch.bfloat16

    def _runtime_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        return self.device, self._native_dtype(self.device)

    def _build_aux_components(self) -> None:
        """Construct the text encoder and VAE at startup, on CPU.

        ``ModuleDiscovery`` skips attributes that are ``None``, so components
        created lazily on the first request are invisible to sequential
        offload, HSDP auxiliary placement, and the shared device/memory
        lifecycle. Building them here and deferring placement to the common
        logic mirrors LTX-2's ``initialize_pipeline_components``.
        """
        dtype = self._native_dtype(self.device)
        self._load_stage1_text_encoder(dtype=dtype)
        self._load_vae(dtype=dtype)
        self._place_aux_components()

    def _place_aux_components(self) -> None:
        """Move the discovered auxiliary components onto the local device.

        Skipped when offload or HSDP owns placement — those paths expect the
        components to still be on CPU when they take over.
        """
        parallel_config = getattr(self.od_config, "parallel_config", None)
        if (
            getattr(self.od_config, "enable_cpu_offload", False)
            or getattr(self.od_config, "enable_layerwise_offload", False)
            or getattr(parallel_config, "use_hsdp", False)
        ):
            return
        modules = ModuleDiscovery.discover(self)
        for module in (*modules.encoders, *modules.vaes, *modules.resident_modules):
            module.to(self.device)

    @staticmethod
    def _component_local_files_only(model_id: Any) -> bool:
        """Decide download policy per component, from its own model id.

        The Stage-1 release directory is local once ``resolve_checkpoint`` has
        run, but the Gemma text encoder lives in a separate Hub repo. Deriving
        locality from the release paths forced ``local_files_only=True`` onto
        Gemma as well, which forbids the download on a box that has not cached
        it yet.
        """
        return Path(str(model_id)).expanduser().exists()

    def _load_stage1_text_encoder(self, *, dtype: torch.dtype) -> None:
        from transformers import AutoModel, AutoTokenizer

        model_ids = [os.environ.get(SANA_WM_STAGE1_TEXT_ENCODER_ENV, "").strip()]
        model_ids.extend([SANA_WM_STAGE1_TEXT_ENCODER_ID, SANA_WM_STAGE1_TEXT_ENCODER_FALLBACK_ID])
        errors: list[str] = []
        for model_id in dict.fromkeys(model_id for model_id in model_ids if model_id):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=self._component_local_files_only(model_id),
                )
                text_encoder_model_id = model_id
                break
            except OSError as exc:
                errors.append(f"{model_id}: {exc}")
        else:
            raise OSError("Could not load SANA-WM Stage-1 text encoder. Tried: " + "; ".join(errors))
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        with torch.device("cpu"):
            # AutoModel, not AutoModelForCausalLM: only the final hidden state is
            # used, so the LM head would compute a [seq, 256k] logits tensor
            # that is thrown away on every prompt encode.
            self.text_encoder = AutoModel.from_pretrained(
                text_encoder_model_id,
                torch_dtype=dtype,
                local_files_only=self._component_local_files_only(text_encoder_model_id),
            )

    def _ensure_stage1_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.text_encoder is None:
            self._load_stage1_text_encoder(dtype=dtype)
        self.text_encoder.to(device=device, dtype=dtype)

    def _load_vae(self, *, dtype: torch.dtype) -> None:
        if self.release_paths is None and self.od_config is not None and self.od_config.model is not None:
            self.resolve_checkpoint()
        if self.release_paths is None:
            raise ValueError("Sana-WM VAE loading requires a resolved checkpoint.")

        from diffusers import AutoencoderKLLTX2Video

        with torch.device("cpu"):
            self.vae = AutoencoderKLLTX2Video.from_pretrained(
                str(self.release_paths.root),
                subfolder="vae",
                torch_dtype=dtype,
                local_files_only=self._component_local_files_only(self.release_paths.root),
            )
        self.vae.use_framewise_encoding = True
        self.vae.use_framewise_decoding = True
        # NVlabs reads these from its YAML VAE config with 64/96 defaults; the
        # diffusers VAE config.json only ships scaling_factor, so the defaults
        # are load-bearing (the keys are genuinely absent here).
        self.vae.tile_sample_stride_num_frames = int(getattr(self.vae.config, "tile_sample_stride_num_frames", 64))
        self.vae.tile_sample_min_num_frames = int(getattr(self.vae.config, "tile_sample_min_num_frames", 96))

    def _ensure_vae(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.vae is None:
            self._load_vae(dtype=dtype)
        # Tiling is re-asserted here, not in _load_vae: registry.py assigns
        # ``model.vae.use_tiling = od_config.vae_use_tiling`` after the pipeline
        # is constructed, and that flag defaults to False. Matching NVlabs'
        # inference_video_scripts/inference_sana_wm.py, the SANA VAE has to
        # tile — a 161-frame decode costs about 9 GiB more without it, and a
        # 321-frame request (41 latent frames) decodes as one 3D volume and
        # OOMs. Overriding is deliberate, but say so rather than dropping the
        # caller's setting silently.
        if not self.vae.use_tiling:
            logger.warning("Sana-WM forces VAE tiling on; ignoring vae_use_tiling=False (the VAE OOMs without it).")
        self.vae.use_tiling = True
        self.vae.to(device=device, dtype=dtype)

    @staticmethod
    def _preprocess_first_frame(
        image: Any,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the first-frame image as ``[1, 3, 1, H, W]`` in ``[-1, 1]``.

        ``VideoProcessor.preprocess`` covers the PIL / ndarray / tensor cases.
        ``resample="bicubic"`` keeps the filter PIL's ``Image.resize`` defaulted
        to; the processor's own default is lanczos, which would move the
        conditioning latent.
        """
        import numpy as np
        from diffusers.video_processor import VideoProcessor

        # The processor normalises to [-1, 1] assuming float inputs in [0, 1].
        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        processor = VideoProcessor(vae_scale_factor=SANA_WM_VAE_SPATIAL_COMPRESSION, resample="bicubic")
        frame = processor.preprocess(image, height=height, width=width)
        return frame.unsqueeze(2).to(device=device, dtype=dtype)

    def _vae_normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """LTX-2 VAE per-channel latent normalisation matching NVlabs.

        ``z_norm = (z_raw - latents_mean) * scaling_factor / latents_std``

        The LTX-2 VAE (``AutoencoderKLLTX2Video`` in diffusers) ships with
        per-channel ``latents_mean`` and ``latents_std`` tensors that must be
        applied alongside ``scaling_factor`` for the round-trip to be
        identity. Applying `* scaling_factor` alone gives decoded videos a
        systematic G/B colour shift (pred RGB means
        ~(110, 220, 208) vs ref ~(94, 114, 138)).

        Returns the normalised latent in the same dtype as the input.
        """
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        return normalize_latents(
            latent,
            self.vae.latents_mean,
            self.vae.latents_std,
            float(self.vae.config.scaling_factor),
        )

    def _vae_denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_vae_normalize_latent` — match NVlabs decode."""
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        return denormalize_latents(
            latent,
            self.vae.latents_mean,
            self.vae.latents_std,
            float(self.vae.config.scaling_factor),
        )

    def _vae_encode_first_frame(
        self,
        image: Any,
        *,
        height: int,
        width: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode the first-frame image to latent space via the LTX-2 VAE.

        Returns shape ``[1, 128, 1, latent_height, latent_width]``.
        """
        self._ensure_vae(device=device, dtype=dtype)
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        frame = self._preprocess_first_frame(image, height=height, width=width, device=device, dtype=dtype)
        with torch.inference_mode():
            encoded = self.vae.encode(frame)
            first_latent = encoded.latent_dist.mean
            first_latent = self._vae_normalize_latent(first_latent.to(dtype=dtype))
        # The encoder's own rounding can land a pixel off the latent grid the
        # noise tensor was built on, and the two are concatenated below.
        if first_latent.shape[-2:] != (latent_height, latent_width):
            first_latent = F.interpolate(
                first_latent.squeeze(2),
                size=(latent_height, latent_width),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(2)
        return first_latent  # [1, 128, 1, lh, lw]

    def _stage1_prompt_text(self, prompt: dict[str, Any]) -> str:
        user_prompt = str(prompt.get("prompt") or "")
        chi_prompt = "\n".join(part for part in self.sana_wm_config.chi_prompt if part)
        if chi_prompt:
            return chi_prompt + user_prompt
        return user_prompt

    def _native_prompt_embeds(
        self,
        prompt: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prompt_text = self._stage1_prompt_text(prompt)
        self._ensure_stage1_text_encoder(device=device, dtype=dtype)
        tokenizer = self.tokenizer
        if tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Sana-WM Stage-1 text encoder did not initialize.")
        chi_prompt = "\n".join(part for part in self.sana_wm_config.chi_prompt if part)
        model_max_length = int(self.sana_wm_config.model_max_length)
        max_length_all = model_max_length
        if chi_prompt:
            # NVlabs encodes the chi-prompt-prefixed text with extra room for
            # the chi tokens, then keeps BOS plus the final model window.
            max_length_all = len(tokenizer.encode(chi_prompt)) + model_max_length - 2
        encoded = tokenizer(
            [prompt_text],
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        # last_hidden_state is the same tensor the CausalLM wrapper exposed as
        # hidden_states[-1] (both are post-final-norm), without retaining every
        # intermediate layer.
        hidden_states = self.text_encoder(**encoded).last_hidden_state
        attention_mask = encoded.attention_mask
        if chi_prompt:
            select_index = [0] + list(range(-model_max_length + 1, 0))
            hidden_states = hidden_states[:, select_index]
            attention_mask = attention_mask[:, select_index]
        if hidden_states.shape[-1] != SANA_WM_STAGE1_PROMPT_CHANNELS:
            raise ValueError(
                "Sana-WM Stage-1 Gemma hidden size mismatch: expected "
                f"{SANA_WM_STAGE1_PROMPT_CHANNELS}, got {hidden_states.shape[-1]}."
            )
        # Return RAW Gemma hidden states: the transformer's ``attention_y_norm``
        # RMSNorm does the normalisation, so normalising here double-normalises.
        prompt_attention_mask = attention_mask.to(device=device, dtype=torch.float32)
        # An all-ones mask carries no information. Keep it ``None`` so
        # cross-attention takes the mask-free attention path instead of handing
        # every block, on every step, a mask the backend must reshape to no
        # effect. Evaluated once per prompt encode, so the device sync is paid
        # twice per request (cond + uncond), not once per denoising step.
        if bool(prompt_attention_mask.all()):
            prompt_attention_mask = None
        self._last_prompt_attention_mask = prompt_attention_mask
        return hidden_states.to(device=device, dtype=dtype)

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        """Single transformer forward.

        Overrides the mixin default, which assumes a tuple-returning transformer
        and takes ``result[0]``; SANA-WM's returns the noise prediction directly.
        """
        return self.transformer(**kwargs)

    def _decode_native_latents(
        self,
        latents: torch.Tensor,
        *,
        output_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if output_type == "latent":
            return latents
        self._ensure_vae(device=device, dtype=dtype)
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        # NVlabs always passes ``temb=None`` to ``vae.decode`` for this VAE
        # (``builder.py::vae_decode -> LTX2VAE_diffusers``); match that.
        denorm = self._vae_denormalize_latent(latents.to(getattr(self.vae, "dtype", dtype)))
        video = self.vae.decode(denorm, temb=None, return_dict=False)[0]
        # Post-process to the requested output_type (np/pil/pt). Let failures
        # surface: silently returning the raw decoder tensor would hand the
        # caller a wrong-format/range video with no error.
        from diffusers.video_processor import VideoProcessor

        processor = VideoProcessor(vae_scale_factor=getattr(self.vae, "spatial_compression_ratio", 32))
        return processor.postprocess_video(video, output_type=output_type)

    def _run_native_backend(
        self,
        *,
        prompt: dict[str, Any],
        payload: dict[str, Any],
        sampling_params: Any | None,
    ) -> DiffusionOutput:
        params = self._native_params(payload, sampling_params)
        latent_frames, latent_height, latent_width = resolve_video_latent_shape(
            params.height,
            params.width,
            params.num_frames,
            vae_spatial_compression_ratio=SANA_WM_VAE_SPATIAL_COMPRESSION,
            vae_temporal_compression_ratio=SANA_WM_VAE_TEMPORAL_COMPRESSION,
        )
        token_count = latent_frames * latent_height * latent_width
        extra_args = self._extra_args(sampling_params)
        max_tokens = int(extra_args.get("sana_wm_native_max_tokens", SANA_WM_NATIVE_MAX_TOKENS))
        if token_count > max_tokens:
            raise ValueError(
                "Sana-WM native latent token count exceeds the configured cap. "
                f"Requested latent tokens={token_count}, max={max_tokens}. Request a smaller "
                "`height`/`width`/`num_frames`, or raise the cap via "
                "`sana_wm_native_max_tokens`."
            )

        device, dtype = self._runtime_device_dtype()
        generator = torch.Generator(device=device)
        generator.manual_seed(params.seed)
        noise = torch.randn(
            (1, 128, latent_frames, latent_height, latent_width),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Shared flow-matching Euler scheduler. NVlabs ``LTXFlowEuler`` drives
        # diffusers' ``FlowMatchEulerDiscreteScheduler``; this is vLLM-Omni's
        # vendored copy of it, including the ``per_token_timesteps`` step branch.
        # The schedule applies the shift twice (once via the shifted ``sigma_min``
        # persisted by ``__init__``, once in ``set_timesteps``), which is what
        # reproduces the NVlabs timestep table (e.g. shift=9.8, N=3 ->
        # [1000, 909.0, 87.7]).
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=SANA_WM_NUM_TRAIN_TIMESTEPS,
            shift=self.sana_wm_config.inference_flow_shift,
        )
        scheduler.set_timesteps(params.num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        # ``normalize_sana_wm_payload`` guarantees the key is present but says
        # nothing about its type. Falling back to pure noise on an unusable
        # image would silently drop the conditioning and produce a
        # plausible-looking but wrong video, so this raises instead.
        import numpy as _np

        first_frame_image = (prompt.get("multi_modal_data") or {}).get("image")
        if not (hasattr(first_frame_image, "convert") or isinstance(first_frame_image, (_np.ndarray, torch.Tensor))):
            raise TypeError(
                "Sana-WM first-frame image must be a PIL Image, numpy ndarray, or "
                f"torch.Tensor; got {type(first_frame_image).__name__}."
            )
        # Per-frame timestep contract matching NVlabs ``LTXFlowEuler.sample``:
        # frame 0 is the clean VAE-encoded conditioning latent and its timestep
        # is held at 0 while the sampling loop drives the other frames from
        # noise.
        first_latent = self._vae_encode_first_frame(
            first_frame_image,
            height=params.height,
            width=params.width,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
        )
        # Place the CLEAN VAE-encoded first frame at frame 0 and rely on
        # per-frame timesteps + the `condition_mask` torch.where restore (below)
        # to keep it invariant across denoising steps.
        latents = torch.cat([first_latent, noise[:, :, 1:]], dim=2)

        prompt_embeds = self._native_prompt_embeds(
            prompt,
            device=device,
            dtype=dtype,
        )
        prompt_attention_mask = self._last_prompt_attention_mask

        # Under CFG, encode the negative prompt up front; each denoise step
        # then runs a two-branch transformer forward. The negative prompt comes
        # from the request (``prompt["negative_prompt"]``, set by
        # ``serving_video``); absent, it defaults to the empty string matching
        # NVlabs ``GenerationParams.negative_prompt = ""``.
        do_cfg = params.cfg_scale > 1.0
        if do_cfg:
            negative_prompt_obj = {"prompt": str(prompt.get("negative_prompt") or "")}
            negative_prompt_embeds = self._native_prompt_embeds(
                negative_prompt_obj,
                device=device,
                dtype=dtype,
            )
            negative_prompt_attention_mask = self._last_prompt_attention_mask
            # Restore the cond mask so later readers of
            # `_last_prompt_attention_mask` see the positive branch.
            self._last_prompt_attention_mask = prompt_attention_mask
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        camera = payload.get("camera") or {}
        condition = SanaWmCameraCondition(
            poses=camera.get("poses") if isinstance(camera, dict) else None,
            intrinsics=payload.get("intrinsics"),
            action=payload.get("action"),
            num_frames=params.num_frames,
            height=params.height,
            width=params.width,
            translation_speed=float(payload.get("translation_speed", 0.05)),
            rotation_speed_deg=float(payload.get("rotation_speed_deg", 1.2)),
        )
        camera_tensors = build_plucker_condition(condition)
        plucker = camera_tensors["chunk_plucker"].to(device=device, dtype=dtype)
        raymap = camera_tensors["raymap"].to(device=device, dtype=dtype)
        spatial_raymap = camera_tensors.get("spatial_raymap")
        if spatial_raymap is not None:
            spatial_raymap = spatial_raymap.to(device=device, dtype=dtype)

        # Build a per-frame condition mask matching NVlabs ``LTXFlowEuler``:
        # frame 0 is the conditioning frame at sigma=0 (preserved after
        # each ``scheduler.step``), all other frames carry the current
        # sampling sigma. We omit the legacy `add_noise_to_image_conditioning_latents`
        # motion-continuity term because the public SANA-WM config sets
        # ``condition_frame_info={0: 0.0}`` (image_cond_noise_scale=0).
        cam_batch = latents.shape[0]
        cam_frames = latents.shape[2]
        condition_mask = torch.zeros(cam_batch, 1, cam_frames, 1, 1, device=latents.device, dtype=latents.dtype)
        condition_mask[:, :, 0] = 1.0

        for timestep in timesteps:
            # (B, 1, F) per-frame timestep, frame 0 forced to 0. Stays fp32:
            # casting through the bf16 latent dtype would quantise it before
            # ``SanaWmTimestepEmbedder``'s sinusoidal embed.
            model_timestep = timestep.float().expand(cam_batch, 1, cam_frames).clone()
            model_timestep[:, :, 0] = 0.0
            positive_kwargs = {
                "hidden_states": latents,
                "timestep": model_timestep,
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": prompt_attention_mask,
                "plucker": plucker,
                "raymap": raymap,
                "spatial_raymap": spatial_raymap,
            }
            negative_kwargs = (
                {
                    **positive_kwargs,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "encoder_attention_mask": negative_prompt_attention_mask,
                }
                if do_cfg
                else None
            )
            # ``cfg_normalize=False`` keeps the plain NVlabs combine,
            # ``uncond + scale * (cond - uncond)``. Under cfg_parallel_size > 1
            # the mixin runs one branch per rank and all-gathers instead of
            # evaluating both branches serially.
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_cfg,
                params.cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize=False,
            )

            # The per-frame contract pairs with a per-token flow-matching Euler
            # step. Both step branches end at ``prev = sample + dt *
            # model_output``, but the scheduler defines dt as
            # ``sigma - sigma_next`` per-token against ``sigma_next - sigma``
            # standard, so the per-token call takes the sign-flipped prediction
            # (matching NVlabs ``LTXFlowEuler.sample``); passing the standard
            # sign here would integrate towards more noise.
            #
            # Broadcast the (B, 1, F) timestep to (B, 1, F, H, W) and flatten
            # F*H*W; frame-0 conditioning tokens are already 0. The flatten
            # order (F, H, W) matches ``pack_latents``.
            pt_t = (
                model_timestep.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(cam_batch, 1, cam_frames, latents.shape[3], latents.shape[4])
                .reshape(cam_batch, -1)
            )
            # The per-token branch consumes/returns the packed (B, N, C) layout
            # and keeps its result in fp32; cast back so the latent dtype stays
            # stable across steps.
            stepped_packed = scheduler.step(
                -pack_latents(noise_pred),
                timestep,
                pack_latents(latents),
                per_token_timesteps=pt_t,
                return_dict=False,
            )[0]
            stepped = unpack_latents(
                stepped_packed.to(latents.dtype),
                cam_frames,
                latents.shape[3],
                latents.shape[4],
            )

            # Match NVlabs LTXFlowEuler exactly. This is stricter than a plain
            # condition-frame restore: with bf16 latents, the `t=1000`
            # comparison rounds `1 - 1e-6` to `1`, so the first full-noise step
            # is discarded for generated tokens as well.
            tokens_to_denoise_mask = timestep / float(SANA_WM_NUM_TRAIN_TIMESTEPS) - 1e-6 < (1.0 - condition_mask)
            latents = torch.where(tokens_to_denoise_mask, stepped, latents)

        output_type = getattr(sampling_params, "output_type", None) or "np"
        output = self._decode_native_latents(latents, output_type=output_type, device=device, dtype=dtype)
        return DiffusionOutput(
            output=build_sana_wm_output_envelope(
                output=output,
                output_type=output_type,
                metadata={
                    "backend": "native_gdn",
                    "output_space": output_type,
                    "chi_prompt_applied": bool(self.sana_wm_config.chi_prompt),
                    "num_frames": params.num_frames,
                    "height": params.height,
                    "width": params.width,
                    "latent_tokens": token_count,
                    "sampling_steps": params.num_inference_steps,
                },
            ),
        )

    def forward(self, req: DiffusionRequestBatch, *args: Any, **kwargs: Any) -> DiffusionOutput:
        del args, kwargs
        if len(req.prompts) != 1:
            raise ValueError("Sana-WM native backend currently supports exactly one prompt per request.")

        prompt = req.prompts[0]
        if isinstance(prompt, str):
            raise ValueError("Sana-WM requires a mapping prompt with first-frame image and camera/action metadata.")
        # get_sana_wm_pre_process_func is the single canonical entry point: it
        # normalises the request and republishes the geometry onto
        # sampling_params. normalize_sana_wm_payload is idempotent, so this only
        # covers offline callers that bypass the preprocess hook.
        prompt = normalize_sana_wm_payload(prompt)
        payload = prompt["additional_information"]["sana_wm"]

        # The checkpoint is resolved once in __init__; this is a cached no-op
        # kept as a guard for pipelines constructed without od_config.model.
        if self.od_config is not None and self.od_config.model is not None:
            self.resolve_checkpoint()

        return self._run_native_backend(
            prompt=prompt,
            payload=payload,
            sampling_params=req.sampling_params,
        )

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        # Keys arrive pipeline-prefixed (``transformer.*``, per the component
        # source). AutoWeightsLoader streams them to the submodule and returns
        # the prefixed names the loader's strict-coverage check compares.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
