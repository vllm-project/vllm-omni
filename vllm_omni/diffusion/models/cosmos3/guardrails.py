# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 guardrail hooks for vllm-omni.

Thin adapter around the ``cosmos_guardrail`` package's ``CosmosSafetyChecker``
(Blocklist + Qwen3Guard for text, RetinaFace face-blur for video).

Enabled by default. Disable server-wide with ``--cosmos3-no-guardrails`` (which
sets ``od_config.model_config["guardrails"] = False``); per-request overrides
ride on ``sampling_params.extra_args["guardrails"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.models.progress_bar import _is_rank_zero
from vllm_omni.errors import GuardrailViolationError

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


try:
    from cosmos_guardrail import CosmosSafetyChecker

    _COSMOS_GUARDRAIL_AVAILABLE = True
except ImportError:
    _COSMOS_GUARDRAIL_AVAILABLE = False

    class CosmosSafetyChecker:  # type: ignore[no-redef]
        # Raised at runtime (not import time) so guardrail-less inference
        # continues to work when ``cosmos_guardrail`` is not installed and
        # ``model_config["guardrails"]`` is False.
        def __init__(self, *args, **kwargs):
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement]"
                "(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
                f"Please install cosmos-guardrail package to enable safety checks."
            )


TextGuardrailFn = Callable[[str], None]
VideoGuardrailFn = Callable[[np.ndarray], np.ndarray]

_text_guardrail: TextGuardrailFn | None = None
_video_guardrail: VideoGuardrailFn | None = None


# ---------------------------------------------------------------------------
# Default guardrail builders
# ---------------------------------------------------------------------------
def _nn_models(runner: Any) -> list[torch.nn.Module]:
    return [m for m in runner.models if isinstance(m, torch.nn.Module)]


def _build_text_guardrail(checker: Any) -> TextGuardrailFn:
    def text_guardrail(prompt: str) -> None:
        if not checker.check_text_safety(prompt):
            # CosmosSafetyChecker logs the specific reason at CRITICAL.
            raise GuardrailViolationError("Input was blocked by Cosmos3 guardrails.")

    return text_guardrail


def _build_video_guardrail(checker: Any, offload_to_cpu: bool) -> VideoGuardrailFn:
    video_models = _nn_models(checker.video_guardrail)
    compute_device = "cuda"

    def video_guardrail(frames: np.ndarray) -> np.ndarray:
        if offload_to_cpu:
            for m in video_models:
                m.to(compute_device)
        try:
            result = checker.check_video_safety(frames)
        finally:
            if offload_to_cpu:
                for m in video_models:
                    m.to("cpu")
        # ``check_video_safety`` returns ``None`` when the content safety
        # filter blocks the frames. The face-blur postprocessor (the only
        # video module enabled by default) does not block, so in practice
        # ``result`` is always an ndarray here.
        return result if result is not None else frames

    return video_guardrail


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def _init_default_guardrails(offload_to_cpu: bool = False) -> None:
    global _text_guardrail, _video_guardrail
    if _text_guardrail is not None:
        return
    if _is_rank_zero():
        logger.info("Initializing Cosmos3 guardrails (offload_to_cpu=%s)...", offload_to_cpu)

    # Instantiation raises ValueError when ``cosmos_guardrail`` is not
    # installed - this is the right moment to fail loudly because the
    # caller has opted in to guardrails.
    checker = CosmosSafetyChecker()

    # Place text models on their resting device permanently. Video models
    # idle on CPU when offload is on and move to GPU per-call (handled in
    # the video guardrail closure).
    idle_device = "cpu" if offload_to_cpu else "cuda"
    for m in _nn_models(checker.text_guardrail):
        m.to(idle_device)
    for m in _nn_models(checker.video_guardrail):
        m.to(idle_device)

    _text_guardrail = _build_text_guardrail(checker)
    _video_guardrail = _build_video_guardrail(checker, offload_to_cpu)
    if _is_rank_zero():
        logger.info("Cosmos3 guardrails initialized.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ensure_initialized(od_config: OmniDiffusionConfig) -> None:
    if not is_guardrails_enabled(od_config):
        return
    model_config = od_config.model_config or {}
    _init_default_guardrails(offload_to_cpu=bool(model_config.get("offload_guardrail_models", False)))


def check_text_safety(prompt: str) -> None:
    if _text_guardrail is not None:
        _text_guardrail(prompt)


def check_video_safety(video_tensor: torch.Tensor) -> torch.Tensor:
    if _video_guardrail is None:
        return video_tensor

    v = video_tensor.detach().cpu().float()
    if v.dim() == 5:
        v = v[0]
    v = v.clamp(-1, 1) * 0.5 + 0.5
    frames_np = (v.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)

    frames_np = _video_guardrail(frames_np)

    # Convert back to [-1, 1] to match the VAE output range.
    result = torch.from_numpy(frames_np.copy()).float() / 127.5 - 1.0
    result = result.permute(3, 0, 1, 2)
    if video_tensor.dim() == 5:
        result = result.unsqueeze(0)
    return result.to(video_tensor.device)


def is_guardrails_enabled(
    od_config: OmniDiffusionConfig,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> bool:
    """Resolve the active guardrail gate.

    Server-level ``od_config.model_config["guardrails"]`` decides whether the
    guardrail models are loaded at all (eager load at pipeline build time).
    When that is False, no per-request override can turn checks back on,
    because the singletons in this module are never populated.

    When the server gate is on, ``sampling_params.extra_args["guardrails"]``
    may override on a per-request basis: ``False`` skips the check for that
    request, anything else (or missing) keeps the default behavior.
    """
    model_config = od_config.model_config or {}
    if not bool(model_config.get("guardrails", True)):
        return False
    if sampling_params is not None:
        per_request = (sampling_params.extra_args or {}).get("guardrails")
        if per_request is not None:
            return bool(per_request)
    return True
