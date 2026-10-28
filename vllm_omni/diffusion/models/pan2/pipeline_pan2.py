# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN 2 video generation pipeline: text-to-video, and image-to-video from a first frame."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, TypeAlias, cast

import numpy as np
import PIL.Image
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Qwen3_5TextModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend, CacheDiTRequestSpec, RequestScopedCacheDiTRuntime
from vllm_omni.diffusion.data import DiffusionCacheConfig, DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.pan2.pan2_transformer import PAN2Transformer3DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest, resolve_video_num_frames
from vllm_omni.diffusion.utils.guardrail_utils import is_guardrails_enabled
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import GuardrailViolationError, OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

PAN2_SYSTEM_MESSAGE = "You are a helpful assistant. Generate the video with the following instructions."
# PAN 2 conditions on `hidden_states[-3]` of the text encoder, i.e. it skips its last two layers.
PAN2_TEXT_ENCODER_LAYERS_TO_SKIP = 2
PAN2_MAX_SEQUENCE_LENGTH = 1536
PAN2_DEFAULT_HEIGHT = 704
PAN2_DEFAULT_WIDTH = 1248
PAN2_DEFAULT_NUM_FRAMES = 313
PAN2_DEFAULT_FPS = 24
PAN2_VAE_SCALE_FACTOR_SPATIAL = 16
PAN2_VAE_SCALE_FACTOR_TEMPORAL = 4
PAN2_MAX_NUM_OUTPUTS = 10

# Latents, the frames of one video (PIL images), or the frames as arrays.
PAN2PostProcessOutput: TypeAlias = torch.Tensor | list[PIL.Image.Image] | np.ndarray | list[np.ndarray]


PAN2_GENERIC_CACHE_KEY = "pan2.generic"
PAN2_HIGH_CACHE_KEY = "pan2.high"


def pan2_high_quality_cache_config() -> DiffusionCacheConfig:
    """PAN 2's conservative Cache-DiT profile, selected by ``quality="high"``."""
    return DiffusionCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=4,
        residual_diff_threshold=0.06,
        max_continuous_cached_steps=1,
        enable_taylorseer=False,
        scm_steps_mask_policy=None,
    )


class PAN2QualityPolicy:
    """Resolve a request's ``quality`` into the Cache-DiT profile it should run with.

    ``high`` selects PAN 2's conservative profile and ``lossless`` runs without a cache. Omitted quality uses the
    server's Cache-DiT profile when the server started with ``cache_backend="cache_dit"``, and no cache otherwise.
    """

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self._od_config = od_config

    def resolve(self, *, quality: str | None, num_inference_steps: int) -> CacheDiTRequestSpec | None:
        if quality == "high":
            key, cache_config = PAN2_HIGH_CACHE_KEY, pan2_high_quality_cache_config()
        elif quality is None and self._od_config.cache_backend == "cache_dit":
            key, cache_config = PAN2_GENERIC_CACHE_KEY, self._od_config.cache_config
        else:
            return None
        return CacheDiTRequestSpec(
            installation_key=key, cache_config=cache_config, num_inference_steps=num_inference_steps
        )


# Guardrails: a thin adapter around the ``pan2_guardrail`` package's ``PAN2SafetyChecker``, which checks the prompt
# and each generated video. Either check blocks the request with a ``GuardrailViolationError``. They are on by
# default; ``--no-guardrails`` turns them off for the server and ``extra_args["guardrails"]`` per request, at the
# user's own risk.
try:
    from pan2_guardrail import PAN2SafetyChecker
except ImportError:

    class PAN2SafetyChecker:  # type: ignore[no-redef]
        # Raised at runtime (not import time) so guardrail-less inference keeps working without `pan2_guardrail`.
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> PAN2SafetyChecker:
            raise ValueError(
                "The PAN 2 guardrails are enabled but the pan2-guardrail package is not installed. Install it with "
                "`pip install git+https://github.com/ifm-ai/pan2-guardrail`, or disable the guardrails with "
                "--no-guardrails at your own risk."
            )


TextGuardrailFn = Callable[[str], None]
VideoGuardrailFn = Callable[[np.ndarray], None]

_text_guardrail: TextGuardrailFn | None = None
_video_guardrail: VideoGuardrailFn | None = None


@contextmanager
def _on_compute_device(module: nn.Module, offload_to_cpu: bool) -> Iterator[None]:
    if not offload_to_cpu:
        yield
        return
    module.to(current_omni_platform.device_type)
    try:
        yield
    finally:
        module.to("cpu")
        # Return the guard's cached blocks, so the offloaded guard does not keep its GPU memory between checks.
        current_omni_platform.empty_cache()


def _build_text_guardrail(checker: PAN2SafetyChecker, offload_to_cpu: bool) -> TextGuardrailFn:
    def text_guardrail(prompt: str) -> None:
        with _on_compute_device(checker.text_guard, offload_to_cpu):
            safe = checker.check_text_safety(prompt)
        if not safe:
            # PAN2SafetyChecker logs the reason at CRITICAL.
            raise GuardrailViolationError("Input was blocked by PAN 2 guardrails.")

    return text_guardrail


def _build_video_guardrail(checker: PAN2SafetyChecker, offload_to_cpu: bool) -> VideoGuardrailFn:
    def video_guardrail(frames: np.ndarray) -> None:
        with _on_compute_device(checker.video_guard, offload_to_cpu):
            result = checker.check_video_safety(frames)
        # `check_video_safety` returns the unchanged frames when they pass and None when the video is blocked.
        if result is None:
            raise GuardrailViolationError("The generated video was blocked by PAN 2 guardrails.")

    return video_guardrail


def _init_default_guardrails(offload_to_cpu: bool = False) -> None:
    global _text_guardrail, _video_guardrail
    if _text_guardrail is not None:
        return
    if _is_rank_zero():
        logger.info("Initializing PAN 2 guardrails (offload_to_cpu=%s)...", offload_to_cpu)

    # Raises ValueError when `pan2_guardrail` is not installed: the caller has opted in to guardrails.
    checker = PAN2SafetyChecker.from_pretrained()
    # The checker loads on CPU. With offloading, each guard moves to the compute device only for its check.
    if not offload_to_cpu:
        checker.to(current_omni_platform.device_type)

    _text_guardrail = _build_text_guardrail(checker, offload_to_cpu)
    _video_guardrail = _build_video_guardrail(checker, offload_to_cpu)
    if _is_rank_zero():
        logger.info("PAN 2 guardrails initialized.")


def _ensure_guardrails_initialized(od_config: OmniDiffusionConfig) -> None:
    if not is_guardrails_enabled(od_config):
        return
    model_config = od_config.model_config or {}
    _init_default_guardrails(offload_to_cpu=bool(model_config.get("offload_guardrail_models", False)))


def _check_text_safety(prompt: str) -> None:
    if _text_guardrail is not None:
        _text_guardrail(prompt)


def _video_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """Turn one decoded `[C, T, H, W]` video in `[-1, 1]` into the uint8 `[T, H, W, C]` frames serving delivers.

    Rounds as `VideoProcessor.postprocess_video` does, so the guardrail sees the delivered pixels. Converts one frame at
    a time to bound the float intermediates of long videos.
    """
    num_channels, num_frames, height, width = video.shape
    frames = np.empty((num_frames, height, width, num_channels), dtype=np.uint8)
    for i in range(num_frames):
        frame = (video[:, i].detach() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0)
        frames[i] = (frame.float() * 255).round().to(torch.uint8).cpu().numpy()
    return frames


def _check_video_safety(video: torch.Tensor) -> list[np.ndarray] | None:
    """Check each video of a decoded `[B, C, T, H, W]` batch; raises `GuardrailViolationError` when one is blocked.

    Returns the checked uint8 frames of each video, or None when the guardrails are not loaded.
    """
    if _video_guardrail is None:
        return None
    checked_frames = []
    for sample in video:
        frames = _video_to_uint8_frames(sample)
        _video_guardrail(frames)
        checked_frames.append(frames)
    return checked_frames


def get_pan2_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    # Eager-load the guardrail models at pipeline build time when the server-level gate is on. Per-request overrides
    # only decide whether the loaded models are invoked.
    if is_guardrails_enabled(od_config):
        _ensure_guardrails_initialized(od_config)

    patch_size = od_config.tf_model_config.get("patch_size") or (1, 2, 2)
    spatial_multiple = PAN2_VAE_SCALE_FACTOR_SPATIAL * patch_size[1]

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        # Reject unsupported requests here, in the engine process: an OmniClientError raised in `forward` on the
        # workers reaches the client as a 500 under the multi-process executor, which drops its 4xx status.
        _check_pan2_request(request.sampling_params, request.prompt, spatial_multiple)
        if is_guardrails_enabled(od_config, request.sampling_params):
            prompt = request.prompt
            _check_text_safety(prompt if isinstance(prompt, str) else prompt.get("prompt") or "")
        return request

    return pre_process_func


def get_pan2_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., PAN2PostProcessOutput]:
    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "pil",
        sampling_params: OmniDiffusionSamplingParams | None = None,
    ) -> PAN2PostProcessOutput:
        if sampling_params is not None and sampling_params.output_type is not None:
            output_type = sampling_params.output_type
        if output_type == "latent":
            return video
        checked_frames = _check_video_safety(video) if is_guardrails_enabled(od_config, sampling_params) else None
        if checked_frames is not None and output_type == "pil":
            # Reuse the uint8 frames the guardrail checked; they are the pixels `postprocess_video` would deliver.
            result = [[PIL.Image.fromarray(frame) for frame in frames] for frames in checked_frames]
        else:
            result = video_processor.postprocess_video(video, output_type=output_type)
        # postprocess_video returns a batch of frame lists; a single video is returned as its frames.
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return result

    return post_process_func


# The negative prompt PAN 2 was released with. Its `time_range_sec` fields are filled in per request to span the
# whole clip, see `pan2_default_negative_prompt`.
PAN2_DEFAULT_NEGATIVE_PROMPT: dict[str, Any] = {
    "video_style": "low quality, blur, softness, reduced detail, noise, compression, clipped highlights, "
    "blurred sidebar, black border, letterboxing, watermark, subtitle, AIGC text, oily waxy "
    "plastic texture, oversaturation or color cast, color clipping or chroma bleeding, flicker, "
    "jitter, ghosting, motion smear, choppy or laggy motion, slow motion, Ken Burns effect, "
    "unmotivated slow zoom over a frozen scene, pan across a still image, global frame scaling "
    "or drifting, simulated movement from a static photograph, repeated micro-motion loop, "
    "discontinuous shot, scene cut, hard cut, crossfade, fade-in, fade-out, fading transition, "
    "scene replacement, spontaneous object or person pop-in or disappearance, duplicated or "
    "missing people or props, identity or count drift, rigid-object deformation or morphing, "
    "bending or melting rigid structures, structural shape or component drift",
    "lighting": "overexposure, clipped highlights, underexposure, harsh lighting, overly deep skin shadows, "
    "brightness fading, excessive lighting saturation, chroma-clipped highlights, neon-like skin "
    "tones, colored spill overwhelming local material colors, oversaturated bloom, unnatural neon "
    "color casts, hue drift, chroma noise, bad white balance, color pulsing, flickering light, "
    "unnatural lighting",
    "perspective_view": "warped geometry, distorted perspective, inconsistent spatial relationships, "
    "unmotivated layout changes, camera teleport or jump, viewpoint reset, wrong depth or "
    "occlusion",
    "subjects_lifecycle": [
        {
            "subject_id": "<|Subject1|>",
            "display_name": "malformed inconsistent subject",
            "subject_attribute": "broken anatomy, distorted body proportions, reversed pelvis, "
            "twisted torso, deformed arms or legs, reversed joints, fused "
            "feet or toes, warped or collapsing faces, transparent or "
            "floating head and body, missing body parts, detached or "
            "disembodied body parts, floating hands or limbs, detached "
            "articulated parts, disconnected joints or supports, hands "
            "disconnected from wrists or forearms, discontinuous "
            "limb-to-torso connections, blurred or fused fingers, missing "
            "thumbs, extra or missing fingers, malformed wrists, extra "
            "limbs, duplicated hands, mirrored face, identity drift, wrong "
            "relative sizes, human-object fusion, unintended body-material "
            "fusion, loss of separate material boundaries, opaque white "
            "nail polish, gray nail polish, chalky opaque fingernails, "
            "artificial acrylic nails",
        }
    ],
    "init_frame_visual_description": "blur or softness, reduced detail, noise or compression artifacts, clipped "
    "highlights, blurred sidebar, black border or letterboxing, oily waxy "
    "plastic surfaces, oversaturation or color cast, exaggerated local "
    "contrast, exposure fading, bad white balance, smeared dense textures, "
    "blurred or distorted face, transparent or missing body parts, detached or "
    "disembodied body parts, floating hands or limbs, detached articulated "
    "parts, disconnected joints or supports, disconnected wrists or forearms, "
    "discontinuous limb-to-torso connections, twisted torso, fused or missing "
    "fingers, missing thumbs, malformed wrists, duplicated limbs, deformed "
    "arms or legs, reversed knees, fused toes, warped geometry, deformed or "
    "asymmetrically warped rigid objects, bent rigid structures, incorrect "
    "rigid part geometry, topology, proportions, or component count, wrong "
    "occlusion, floating or hovering objects, unsupported objects, object or "
    "human-object fusion, unintended fusion of distinct surfaces or layers, "
    "loss of separate material boundaries, merged objects, hand-object "
    "penetration, object-object penetration, penetration, collision, "
    "impossible contact, wrong sizes.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 10",
            "event_description": "blur, softness, reduced detail, compression artifacts, clipped "
            "highlights, flicker, jitter, ghosting, motion smear, choppy or laggy "
            "motion, slow motion, Ken Burns effect, unmotivated slow zoom over a "
            "frozen scene, pan across a still image, global frame scaling or drifting, "
            "simulated movement from a static photograph, repeated micro-motion loop, "
            "oily oversaturated or color-pulsing surfaces, shimmering surface detail, "
            "fingers fusing or changing count, hands smearing, facial features "
            "merging, reversed joints, upper body rotating independently of attached "
            "hips and feet, corkscrewed waist, limbs changing length or attachment "
            "after occlusion, transparent body or head, detached or disembodied body "
            "parts, floating hands or limbs, detached articulated parts, disconnected "
            "joints or supports, hands disconnected from wrists or forearms, body "
            "parts moving independently, vanishing limb-to-torso connections, merged "
            "people or limbs, limbs fused into torso, merged or fused feet and toes, "
            "floating or hovering objects, unsupported objects, object or human-object "
            "fusion, unintended fusion of distinct surfaces or layers, loss of "
            "separate material boundaries, merged objects, hand-object penetration, "
            "object-object penetration, penetration or clipping, objects colliding or "
            "morphing, rigid objects tearing, fragmenting, fusing, bending, melting or "
            "changing dimensions, topology, silhouette, component count or layout, "
            "subjects changing identity, splitting or duplicating, objects appearing "
            "or disappearing without entry, exit or occlusion, changed identity or "
            "count after occlusion, state changes inconsistent with the causing "
            "action, broken physics, uncaused deformation or motion, state changes "
            "preceding the causing interaction, loss of coupling during contact, pose "
            "jumps, discontinuous shot, camera jump, hard scene cut, crossfade, "
            "fade-in, fade-out, fading transition, scene replacement or viewpoint "
            "reset.",
            "camera_motion": [
                {"time_range_sec": "0 to 10", "camera_motion": ["camera_jump", "hard_cut", "viewpoint_reset"]}
            ],
            "contained_actions": [
                {
                    "time_range_sec": "0 to 10",
                    "action": "<|Subject1|> fingers fusing during grip changes, palms "
                    "merging with props, wrists collapsing, crossed limbs fusing, "
                    "intersecting or swapping attachment, duplicated hands or "
                    "feet, detached or disembodied hands or limbs, hands "
                    "disconnected from wrists or forearms, discontinuous "
                    "limb-to-torso connections, merged or fused feet and toes, "
                    "unstable contact, unintended grip slip, hand-object "
                    "penetration, body-object penetration, passage through solid "
                    "surfaces instead of openings, object-object penetration, "
                    "merged or fused objects, rigid objects fragmenting, fusing, "
                    "smearing or changing dimensions during motion, unintended "
                    "fusion of distinct surfaces or layers, loss of separate "
                    "material boundaries, floating or hovering objects, "
                    "unsupported objects, objects duplicating, splitting or "
                    "disappearing during interaction or occlusion, unchanged "
                    "action result, pose jump",
                }
            ],
        }
    ],
    "quality": {
        "video_quality": "low resolution, blur, softness, reduced detail, noise, compression, clipped "
        "highlights, blurred sidebar, black border, letterboxing, oily waxy plastic "
        "surfaces, greasy highlights, oversaturated colors, faded or washed-out "
        "colors, exposure fading, smeared faces, hands or held objects, surface detail "
        "smearing or swimming, material or existing markings changing without a "
        "physical cause, color or shape morphing, object pop-in or pop-out, sudden "
        "disappearance, missing persistent objects, object duplication or count drift, "
        "rigid-object warping or deformation, changing rigid proportions, topology, "
        "silhouette, or component layout",
        "text_logos_overlays": "watermark, subtitle, captions, AIGC text or logo",
        "artifacts": [
            "watermark",
            "template_or_text_dominated",
            "talking_head",
            "visual_quality_issue",
            "edited_or_discontinuous",
            "video_game",
            "animation",
            "ai_generated_content",
            "black_border",
            "ken_burn",
            "time_lapse",
            "blurred_sidebar",
            "hard_scene_cuts",
        ],
    },
}


def pan2_default_negative_prompt(num_frames: int, fps: float) -> str:
    """PAN 2's default negative prompt as compact one-line JSON, with every time range spanning the clip."""
    span = f"0 to {(num_frames - 1) / fps:g}"

    def fill(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: span if key == "time_range_sec" else fill(value) for key, value in node.items()}
        if isinstance(node, list):
            return [fill(item) for item in node]
        return node

    return json.dumps(fill(PAN2_DEFAULT_NEGATIVE_PROMPT), ensure_ascii=False, separators=(",", ":"))


def _resolve_pan2_num_outputs(sampling_params: OmniDiffusionSamplingParams) -> int:
    value = sampling_params.num_outputs_per_prompt
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise OmniClientError(f"PAN 2 num_outputs_per_prompt must be an integer in [1, {PAN2_MAX_NUM_OUTPUTS}]")
    if not 1 <= value <= PAN2_MAX_NUM_OUTPUTS:
        raise OmniClientError(f"PAN 2 num_outputs_per_prompt must be in [1, {PAN2_MAX_NUM_OUTPUTS}], got {value}")
    return value


def _check_pan2_request(
    sampling_params: OmniDiffusionSamplingParams,
    prompt: OmniPromptType,
    spatial_multiple: int = PAN2_VAE_SCALE_FACTOR_SPATIAL * 2,
) -> None:
    """Reject the inputs PAN 2 does not support."""
    height, width, num_frames = sampling_params.height, sampling_params.width, sampling_params.num_frames
    if (height is not None and height % spatial_multiple != 0) or (width is not None and width % spatial_multiple != 0):
        raise OmniClientError(
            f"PAN 2 `height` and `width` must be divisible by {spatial_multiple}, got {height} and {width}."
        )
    if num_frames is not None and (num_frames - 1) % PAN2_VAE_SCALE_FACTOR_TEMPORAL != 0:
        raise OmniClientError(
            f"PAN 2 `num_frames - 1` must be divisible by {PAN2_VAE_SCALE_FACTOR_TEMPORAL}, got {num_frames}."
        )
    if sampling_params.sigmas is not None or sampling_params.timesteps is not None:
        raise OmniClientError(
            "PAN 2 uses its own flow-matching schedule; custom `sigmas`/`timesteps` are not supported."
        )
    flow_shift = (sampling_params.extra_args or {}).get("flow_shift")
    if flow_shift is not None and (
        isinstance(flow_shift, bool)
        or not isinstance(flow_shift, (int, float))
        or not math.isfinite(flow_shift)
        or flow_shift <= 0
    ):
        raise OmniClientError(f"PAN 2 `flow_shift` must be a positive number, got {flow_shift!r}.")
    multi_modal_data = (prompt.get("multi_modal_data") or {}) if isinstance(prompt, dict) else {}
    if multi_modal_data.get("video") is not None:
        raise OmniClientError("PAN 2 does not accept video input; pass a single first-frame image for image-to-video.")
    image = multi_modal_data.get("image")
    if isinstance(image, list) and len(image) != 1:
        raise OmniClientError(f"PAN 2 conditions on a single first frame, got {len(image)} images.")
    _resolve_pan2_num_outputs(sampling_params)


class PAN2Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportsComponentDiscovery,
):
    """PAN 2 video generator. A request with an image in ``multi_modal_data`` runs image-to-video from that frame."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    # Distributed layerwise offload: the text refiner streams with the main blocks, and the Qwen3.5 text encoder
    # streams its layers.
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        block_attrs={"transformer": ("context_refiner", "transformer_blocks")},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("layers",)},
        encoder_dlo_weight_replication=frozenset({"text_encoder"}),
    )
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        parallel_config = od_config.parallel_config
        if parallel_config.ring_degree > 1 or parallel_config.allgather_degree > 1:
            raise NotImplementedError(
                "PAN 2 does not support ring or all-gather sequence parallel: its padded joint video/text sequence "
                "needs an attention mask they do not carry. Use Ulysses (--usp) and CFG parallelism instead."
            )
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        revision = od_config.revision
        local_files_only = os.path.exists(model)
        subfolders = ["tokenizer", "text_encoder", "vae", "scheduler"]
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only, revision=revision)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only, revision=revision
        )
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3_5TextModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            revision=revision,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            revision=revision,
            torch_dtype=dtype,
        ).to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only, revision=revision
        )
        if od_config.flow_shift is not None:
            # The scheduler exposes `shift` as a read-only property.
            self.scheduler._shift = od_config.flow_shift
        # The flow shift of the requests that do not set their own `flow_shift`.
        self.default_flow_shift = self.scheduler.shift

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, PAN2Transformer3DModel)
        self.transformer = PAN2Transformer3DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )
        # Cache-DiT tells the CFG branches apart by counting transformer calls in pairs, which only holds when the
        # branches run as two calls on the same rank. CFG parallelism makes one call per step on each rank.
        self.sequential_cfg_branches = od_config.parallel_config.cfg_parallel_size == 1
        self.transformer._cache_dit_adapter_config = dataclasses.replace(
            self.transformer._cache_dit_adapter_config, has_separate_cfg=self.sequential_cfg_branches
        )
        self._quality_policy = PAN2QualityPolicy(od_config)
        self._cache_dit_runtime = RequestScopedCacheDiTRuntime(self)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.num_channels_latents = self.vae.config.z_dim
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self._guidance_scale: float | None = None
        self._image_conditioned = False
        self._num_timesteps: int | None = None
        self._current_timestep: torch.Tensor | None = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self) -> float | None:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int | None:
        return self._num_timesteps

    @property
    def current_timestep(self) -> torch.Tensor | None:
        return self._current_timestep

    def encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        # An empty prompt is encoded as a single space, as in training.
        messages = [{"role": "system", "content": PAN2_SYSTEM_MESSAGE}, {"role": "user", "content": prompt or " "}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True
        ).input_ids

        # PAN 2 conditions on the tokens after the `<|im_start|>user\n` header: the prompt and the assistant header.
        header_ids = self.tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        crop_start = next(
            (i + len(header_ids) for i in range(len(input_ids)) if input_ids[i : i + len(header_ids)] == header_ids),
            None,
        )
        if crop_start is None:
            raise ValueError("The PAN 2 tokenizer's chat template has no `<|im_start|>user\\n` header.")
        input_ids = torch.tensor([input_ids[: crop_start + PAN2_MAX_SEQUENCE_LENGTH]], device=device)

        prompt_embeds = self.text_encoder(input_ids=input_ids, output_hidden_states=True).hidden_states[
            -(PAN2_TEXT_ENCODER_LAYERS_TO_SKIP + 1)
        ]
        return prompt_embeds[:, crop_start:]

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        # The latents stay in float32 across the denoising loop and are cast to the transformer dtype per step.
        if latents is not None:
            if latents.shape[0] != batch_size:
                raise ValueError(f"`latents` must have batch size {batch_size}, got {latents.shape[0]}.")
            return latents.to(device=device, dtype=torch.float32)
        shape = (
            batch_size,
            self.num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

    def prepare_condition_latents(
        self, latents: torch.Tensor, image: PIL.Image.Image | None, height: int, width: int
    ) -> torch.Tensor:
        """Conditioning latents and mask stacked along channels; image-to-video fills the first latent frame.

        The image is encoded once and shared by every video of the batch.
        """
        batch_size, num_channels, num_frames, latent_height, latent_width = latents.shape
        condition_latents = latents.new_zeros(batch_size, num_channels + 1, num_frames, latent_height, latent_width)
        if image is None:
            return condition_latents

        # Scale the image to cover the canvas, rounding its size, and center crop it, as PAN 2 was trained.
        scale = max(width / image.width, height / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), resample=PIL.Image.LANCZOS)
        left, top = round((image.width - width) / 2), round((image.height - height) / 2)
        image = image.crop((left, top, left + width, top + height))
        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.unsqueeze(2).to(device=latents.device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(image_latents)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(image_latents)
        image_latents = (image_latents - latents_mean) / latents_std

        condition_latents[:, :num_channels, :1] = image_latents
        condition_latents[:, num_channels:, :1] = 1.0
        return condition_latents

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        """Take over the Cache-DiT backend the runner enabled at startup, so each request can switch it."""
        self._cache_dit_runtime.adopt(backend, installation_key=PAN2_GENERIC_CACHE_KEY)

    def is_cache_dit_enabled(self) -> bool:
        return self._cache_dit_runtime.is_enabled

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor,
        negative_noise_pred: torch.Tensor,
        true_cfg_scale: float,
        cfg_normalize: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        noise_pred = super().combine_cfg_noise(positive_noise_pred, negative_noise_pred, true_cfg_scale, cfg_normalize)
        if self._image_conditioned:
            # The image-conditioned first latent frame takes the conditional prediction, without guidance.
            noise_pred[:, :, :1] = positive_noise_pred[0][:, :, :1]
        return noise_pred

    @contextmanager
    def _resident_dit_layers_on_device(self) -> Iterator[None]:
        """Keep the resident DiT layers of distributed layerwise offload on the device while denoising."""
        controller = getattr(self, "_dlo_residency_controller", None)
        if controller is not None:
            controller.load_resident_layers()
        try:
            yield
        finally:
            if controller is not None:
                controller.offload_resident_layers()

    def forward(
        self,
        req: DiffusionRequestBatch,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = PAN2_DEFAULT_HEIGHT,
        width: int = PAN2_DEFAULT_WIDTH,
        num_frames: int = PAN2_DEFAULT_NUM_FRAMES,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("PAN 2 takes a single prompt per request.")
        sampling_params = req.sampling_params
        prompt_data = req.prompts[0]
        _check_pan2_request(sampling_params, prompt_data)
        num_outputs = _resolve_pan2_num_outputs(sampling_params)
        if isinstance(prompt_data, str):
            prompt, negative_prompt, multi_modal_data = prompt_data, None, {}
        else:
            prompt = prompt_data.get("prompt")
            negative_prompt = prompt_data.get("negative_prompt")
            multi_modal_data = prompt_data.get("multi_modal_data") or {}

        image = multi_modal_data.get("image")
        if isinstance(image, list):
            image = image[0]
        if isinstance(image, str):
            image = PIL.Image.open(image)
        if image is not None:
            image = cast(PIL.Image.Image, image).convert("RGB")
        self._image_conditioned = image is not None

        output_type = sampling_params.output_type or output_type
        height = sampling_params.height or height
        width = sampling_params.width or width
        num_frames = resolve_video_num_frames(
            sampling_params.num_frames, default_num_frames=num_frames, is_dummy_run=req.is_dummy_run()
        )
        num_steps = sampling_params.num_inference_steps or num_inference_steps
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale
        self._guidance_scale = guidance_scale
        do_cfg = guidance_scale > 1.0

        cache_dit_spec = self._quality_policy.resolve(quality=sampling_params.quality, num_inference_steps=num_steps)
        if cache_dit_spec is not None and self.sequential_cfg_branches and not do_cfg:
            logger.warning_once(
                "PAN 2 runs without Cache-DiT for requests without CFG when CFG branches run sequentially."
            )
            cache_dit_spec = None
        self._cache_dit_runtime.prepare(cache_dit_spec)

        multiple = self.vae_scale_factor_spatial * self.transformer.patch_size[1]
        if height % multiple != 0 or width % multiple != 0:
            raise ValueError(f"`height` and `width` must be divisible by {multiple}, got {height} and {width}.")
        if (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            raise ValueError(
                f"`num_frames - 1` must be divisible by {self.vae_scale_factor_temporal}, got {num_frames}."
            )

        device = self.device
        dtype = self.transformer.x_embedder.weight.dtype

        if generator is None:
            # One generator per video: a single request generator draws the videos' noise in turn.
            generator = req.collate_request_generators(num_outputs, None)
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=sampling_params.generator_device or device).manual_seed(
                sampling_params.seed
            )

        # The videos share one prompt, so their text embeddings are repeated without padding.
        prompt_embeds = self.encode_prompt(prompt, device).to(dtype).repeat_interleave(num_outputs, dim=0)
        negative_prompt_embeds = None
        if do_cfg:
            # Without a negative prompt, PAN 2 uses its default one, filled in for this clip.
            if not negative_prompt:
                negative_prompt = pan2_default_negative_prompt(
                    num_frames, sampling_params.resolved_frame_rate or PAN2_DEFAULT_FPS
                )
            negative_prompt_embeds = (
                self.encode_prompt(negative_prompt, device).to(dtype).repeat_interleave(num_outputs, dim=0)
            )

        latents = self.prepare_latents(
            num_outputs, height, width, num_frames, device, generator, sampling_params.latents
        )
        condition_latents = self.prepare_condition_latents(latents, image, height, width).to(dtype)

        # PAN 2 samples on uniformly spaced sigmas from 1 down to (excluding) 0; the scheduler applies the shift, the
        # request's `flow_shift` if it sets one.
        flow_shift = (sampling_params.extra_args or {}).get("flow_shift")
        self.scheduler._shift = float(flow_shift) if flow_shift is not None else self.default_flow_shift
        sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        with self._resident_dit_layers_on_device(), self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t

                latent_model_input = torch.cat([latents.to(dtype), condition_latents], dim=1)
                timestep = t.expand(latent_model_input.shape[0])

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                }
                negative_kwargs = None
                if do_cfg:
                    negative_kwargs = {**positive_kwargs, "encoder_hidden_states": negative_prompt_embeds}

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=sampling_params.cfg_normalize,
                )
                latents = self.scheduler_step_maybe_with_cfg(noise_pred.float(), t, latents, do_true_cfg=do_cfg)
                pbar.update()

        self._current_timestep = None
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        if output_type == "latent":
            output = latents
        else:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
            latents = (latents * latents_std + latents_mean).to(self.vae.dtype)
            # Decode one video at a time, so the VAE peak memory does not grow with the number of videos.
            output = torch.cat(
                [self.vae.decode(video_latents, return_dict=False)[0] for video_latents in latents.split(1)]
            )

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
