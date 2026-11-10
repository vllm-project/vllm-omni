# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request MixFusion opt-in plumbing (framework, model-agnostic).

The engine resolves request opt-in before admission. Only requests carrying
model-resolved packing metadata receive a relaxed step-batch key. The runner
follows the scheduler output rather than interpreting the request opt-in.
The opt-in is gated on the pipeline's ``supports_mixfusion`` capability and
may be overridden per request through ``sampling.extra_args``.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from vllm_omni.diffusion.mixfusion.plan import MixFusionImageRequest

if TYPE_CHECKING:
    from vllm_omni.diffusion.mixfusion.scheduling import MixFusionRequestInfo
    from vllm_omni.diffusion.request import OmniDiffusionRequest

# Defaults for requests that do not carry per-request MixFusion extras.
DEFAULT_MIN_CHUNK_TOKENS = 256
DEFAULT_MAX_CHUNKS = 128

#: Step-batch key fields MixFusion relaxes so mixed resolutions share a step.
#: Dense admission retains them: equal-area images can have different RoPE.
RELAXED_STEP_KEY_FIELDS = ("height", "width", "resolution")


@dataclass(frozen=True)
class MixFusionImageRequestConfig:
    """Resolved plan values sent from workers to the host scheduler."""

    vae_scale_factor: int
    default_sample_size: int
    channels: int
    patch_size: int
    default_true_cfg_scale: float


def resolve_image_request_config(
    plan: MixFusionImageRequest, pipeline: object, transformer: object
) -> MixFusionImageRequestConfig:
    def resolve(source: str | int, owner: object) -> int:
        value = attrgetter(source)(owner) if isinstance(source, str) else source
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"MixFusion image layout requires a positive integer for {source!r}, got {value!r}")
        return value

    return MixFusionImageRequestConfig(
        resolve(plan.vae_scale_factor, pipeline),
        resolve(plan.default_sample_size, pipeline),
        resolve(plan.channels, transformer),
        resolve(plan.patch_size, transformer),
        plan.default_true_cfg_scale,
    )


def image_token_grid(height: int, width: int, vae_scale_factor: int, patch_size: int) -> tuple[int, int]:
    return int(height) // (vae_scale_factor * patch_size), int(width) // (vae_scale_factor * patch_size)


def prepare_image_request(
    request: OmniDiffusionRequest,
    config: MixFusionImageRequestConfig,
    min_chunk_tokens: int,
    max_chunks: int,
) -> MixFusionRequestInfo | None:
    """Interpret the declared image grid using the common request schema."""
    from vllm_omni.diffusion.mixfusion.scheduling import MixFusionRequestInfo

    sampling, prompt = request.sampling_params, request.prompt
    if not isinstance(prompt, str):
        if not isinstance(prompt, dict) or set(prompt) - {"prompt", "negative_prompt"}:
            return None
    if sampling.latents is not None:
        return None
    height = sampling.height or config.default_sample_size * config.vae_scale_factor
    width = sampling.width or config.default_sample_size * config.vae_scale_factor
    h, w = image_token_grid(height, width, config.vae_scale_factor, config.patch_size)
    num_images = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
    negative = isinstance(prompt, dict) and prompt.get("negative_prompt") is not None
    true_cfg = (sampling.true_cfg_scale or config.default_true_cfg_scale) > 1 and negative
    return MixFusionRequestInfo(
        (num_images, h * w, config.channels), min_chunk_tokens, max_chunks, condition_key=(true_cfg,)
    )


def mixfusion_requested(sampling: Any, od_config: Any, *, supported: bool) -> bool:
    """Whether a request with these sampling params opts into MixFusion.

    The per-request ``enable_mixfusion`` flag takes precedence over the
    server-level default. On a pipeline without the capability it is a no-op
    (a server-level flag there is rejected at engine startup).
    """
    if not supported:
        return False
    extra_args = sampling.extra_args or {}
    return bool(extra_args.get("enable_mixfusion", getattr(od_config, "enable_mixfusion", False)))


def has_per_prompt_inputs(prompt: Any) -> bool:
    """Whether a prompt carries per-request inputs that cannot share a batch key."""
    return isinstance(prompt, dict) and bool(prompt.get("additional_information"))


def relax_step_batch_key(request: Any, values: dict, od_config: Any, *, supported: bool) -> dict | None:
    """Drop resolution from a step-batch key when ``request`` opts in.

    Only prepared, packable requests may relax the shape key. Returns
    ``None`` when the request cannot share a key at all (per-prompt inputs).
    """
    info = getattr(request, "mixfusion_info", None)
    if info is None:
        if mixfusion_requested(request.sampling_params, od_config, supported=supported) and has_per_prompt_inputs(
            request.prompt
        ):
            return None
        return values
    for name in RELAXED_STEP_KEY_FIELDS:
        values[name] = None
    values["mixfusion_key"] = (info.shape[2], info.condition_key)
    return values


def request_batch_key_allowed(request: Any, od_config: Any, *, supported: bool) -> bool:
    """Whether a request-mode batch key may be built for ``request``.

    Request-mode batches resolve one height/width for the whole batch, so
    resolution stays in their key; only per-prompt inputs make a MixFusion
    request unbatchable.
    """
    return not (
        mixfusion_requested(request.sampling_params, od_config, supported=supported)
        and has_per_prompt_inputs(request.prompt)
    )
