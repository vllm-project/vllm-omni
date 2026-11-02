# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep Bagel's custom image modality visible to vLLM's prefix-range builder."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec

_RUNNER_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_states.default",
)


def apply_img2img_mm_prefix_patch() -> None:
    from vllm.v1.attention.backends import utils
    from vllm.version import __version_tuple__

    original = getattr(utils, "compute_mm_prefix_ranges", None)
    if original is None:
        # vLLM < 0.33 builds these ranges inline and already accepts img2img.
        if __version_tuple__[:2] >= (0, 33):
            raise RuntimeError("vLLM >= 0.33 is missing compute_mm_prefix_ranges")
        return
    if not callable(original):
        raise RuntimeError("vLLM compute_mm_prefix_ranges must be callable")
    if getattr(original, "_omni_img2img_mm_prefix", False):
        return

    @wraps(original)
    def compute_mm_prefix_ranges(
        mm_features_per_req: Sequence[Sequence[MultiModalFeatureSpec]],
        sliding_window: int | None = None,
        clamp_sliding_window: bool = False,
        span_pad: int = 0,
    ) -> dict[int, list[tuple[int, int]]]:
        # Alias only the prefix builder's host-side views. Encoder batching,
        # kwargs, caches, and embedding modality tags keep their native keys.
        if any(feature.modality == "img2img" for features in mm_features_per_req for feature in features):
            mm_features_per_req = [
                [
                    replace(feature, modality="image") if feature.modality == "img2img" else feature
                    for feature in features
                ]
                for features in mm_features_per_req
            ]
        return original(mm_features_per_req, sliding_window, clamp_sliding_window, span_pad)

    # Validate previously imported aliases before changing any module binding.
    loaded_runners = []
    for name in _RUNNER_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        alias = getattr(module, "compute_mm_prefix_ranges", None)
        if alias is not None and alias is not original:
            raise RuntimeError(f"Conflicting compute_mm_prefix_ranges alias in {name}")
        if alias is original:
            loaded_runners.append(module)

    setattr(compute_mm_prefix_ranges, "_omni_img2img_mm_prefix", True)
    setattr(compute_mm_prefix_ranges, "_omni_mm_prefix_original", original)
    utils.compute_mm_prefix_ranges = compute_mm_prefix_ranges
    for module in loaded_runners:
        setattr(module, "compute_mm_prefix_ranges", compute_mm_prefix_ranges)
