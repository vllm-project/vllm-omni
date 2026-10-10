# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN2 guardrail hooks for vllm-omni.

Thin adapter around the ``pan2_guardrail`` package's ``PAN2SafetyChecker``: Qwen3Guard-Gen-4B on the prompt, and the
Cosmos-Guardrail1 video content safety filter on 16 frames of the generated video. Either check blocks the request
with a ``GuardrailViolationError``.

Enabled by default. Disable server-wide with ``--no-guardrails`` (which sets
``od_config.model_config["guardrails"] = False``); per-request overrides ride on
``sampling_params.extra_args["guardrails"]``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.progress_bar import _is_rank_zero
from vllm_omni.errors import GuardrailViolationError
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)

try:
    from pan2_guardrail import PAN2SafetyChecker
except ImportError:

    class PAN2SafetyChecker:  # type: ignore[no-redef]
        # Raised at runtime (not import time) so guardrail-less inference keeps working without `pan2_guardrail`.
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> PAN2SafetyChecker:
            raise ValueError(
                "The PAN2 guardrails are enabled but the pan2-guardrail package is not installed. Install it with "
                "`pip install git+https://github.com/MBZUAI-IFM/PAN2-guardrail`, or disable the guardrails with "
                "--no-guardrails. Its video check downloads the gated "
                "nvidia/Cosmos-Guardrail1 weights, whose license must be accepted on the Hugging Face Hub."
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


def _build_text_guardrail(checker: PAN2SafetyChecker, offload_to_cpu: bool) -> TextGuardrailFn:
    def text_guardrail(prompt: str) -> None:
        with _on_compute_device(checker.text_guard, offload_to_cpu):
            safe = checker.check_text_safety(prompt)
        if not safe:
            # PAN2SafetyChecker logs the reason at CRITICAL.
            raise GuardrailViolationError("Input was blocked by PAN2 guardrails.")

    return text_guardrail


def _build_video_guardrail(checker: PAN2SafetyChecker, offload_to_cpu: bool) -> VideoGuardrailFn:
    def video_guardrail(frames: np.ndarray) -> None:
        with _on_compute_device(checker.video_guard, offload_to_cpu):
            result = checker.check_video_safety(frames)
        # `check_video_safety` returns the unchanged frames when they pass and None when the video is blocked.
        if result is None:
            raise GuardrailViolationError("The generated video was blocked by PAN2 guardrails.")

    return video_guardrail


def _init_default_guardrails(offload_to_cpu: bool = False) -> None:
    global _text_guardrail, _video_guardrail
    if _text_guardrail is not None:
        return
    if _is_rank_zero():
        logger.info("Initializing PAN2 guardrails (offload_to_cpu=%s)...", offload_to_cpu)

    # Raises ValueError when `pan2_guardrail` is not installed: the caller has opted in to guardrails.
    checker = PAN2SafetyChecker.from_pretrained()
    # The checker loads on CPU. With offloading, each guard moves to the compute device only for its check.
    if not offload_to_cpu:
        checker.to(current_omni_platform.device_type)

    _text_guardrail = _build_text_guardrail(checker, offload_to_cpu)
    _video_guardrail = _build_video_guardrail(checker, offload_to_cpu)
    if _is_rank_zero():
        logger.info("PAN2 guardrails initialized.")


def ensure_initialized(od_config: OmniDiffusionConfig) -> None:
    if not is_guardrails_enabled(od_config):
        return
    model_config = od_config.model_config or {}
    _init_default_guardrails(offload_to_cpu=bool(model_config.get("offload_guardrail_models", False)))


def check_text_safety(prompt: str) -> None:
    if _text_guardrail is not None:
        _text_guardrail(prompt)


def video_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
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


def check_video_safety(video: torch.Tensor) -> list[np.ndarray] | None:
    """Check each video of a decoded `[B, C, T, H, W]` batch; raises `GuardrailViolationError` when one is blocked.

    Returns the checked uint8 frames of each video, or None when the guardrails are not loaded.
    """
    if _video_guardrail is None:
        return None
    checked_frames = []
    for sample in video:
        frames = video_to_uint8_frames(sample)
        _video_guardrail(frames)
        checked_frames.append(frames)
    return checked_frames


def is_guardrails_enabled(
    od_config: OmniDiffusionConfig,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> bool:
    """Resolve the active guardrail gate.

    Server-level ``od_config.model_config["guardrails"]`` decides whether the guardrail models are loaded at all
    (eager load at pipeline build time). When that is False, no per-request override can turn checks back on.

    When the server gate is on, ``sampling_params.extra_args["guardrails"]`` may override on a per-request basis:
    ``False`` skips the checks for that request, anything else (or missing) keeps them.
    """
    model_config = od_config.model_config or {}
    if not bool(model_config.get("guardrails", True)):
        return False
    if sampling_params is not None:
        per_request = (sampling_params.extra_args or {}).get("guardrails")
        if per_request is not None:
            return bool(per_request)
    return True
