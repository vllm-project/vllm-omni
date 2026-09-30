# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Enable VoxCPM2 LocDiT NPUGraph acceleration on Ascend."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_GRAPHS = 8
_PATCHED = False
_original_init = None


def _patched_init(self, *, vllm_config: Any, prefix: str = "") -> None:
    assert _original_init is not None
    _original_init(self, vllm_config=vllm_config, prefix=prefix)
    setup_voxcpm2_loc_dit_npu_graph(self)


def _get_npu_exact_graph_runner_cls():
    # Import graph tooling only in the engine worker. Importing it in vLLM's
    # short-lived model-inspection subprocess initializes the NPU runtime and
    # can make that subprocess abort during teardown.
    from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

    return NPUExactGraphRunner


def apply_voxcpm2_talker_patch() -> None:
    """Install the Ascend LocDiT adapter before VoxCPM2 is constructed."""
    global _PATCHED, _original_init
    if _PATCHED:
        return

    # The model defers resolving ``current_omni_platform`` until construction,
    # so importing it while NPUOmniPlatform is initialized is cycle-free.
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    _original_init = VoxCPM2TalkerForConditionalGeneration.__init__
    VoxCPM2TalkerForConditionalGeneration.__init__ = _patched_init  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for VoxCPM2 LocDiT")


def setup_voxcpm2_loc_dit_npu_graph(model: object) -> None:
    """Wrap VoxCPM2's tensor-only LocDiT estimator with NPUGraph."""
    wrapped_model = getattr(model, "module", None)
    talker = wrapped_model if wrapped_model is not None else model
    tts = getattr(talker, "tts", None)
    feat_decoder = getattr(tts, "feat_decoder", None)
    estimator = getattr(feat_decoder, "estimator", None)
    if estimator is None:
        raise TypeError("expected a VoxCPM2 talker with tts.feat_decoder.estimator")
    if getattr(estimator, "_voxcpm2_npu_graph_runner", None) is not None:
        return

    graph_runner = _get_npu_exact_graph_runner_cls()(
        max_graphs=_MAX_GRAPHS,
        component_name="VoxCPM2 LocDiT",
        disable_config_hint="disable the VoxCPM2 NPU model patch",
    )
    if not graph_runner.is_supported():
        logger.warning("VoxCPM2 LocDiT NPUGraph APIs are unavailable; using eager execution")
        return

    original_forward = estimator.forward

    def _forward(
        module,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        del module
        return graph_runner.run(
            "forward",
            (x, mu, t, cond, dt),
            (),
            lambda *inputs: (original_forward(*inputs),),
        )[0]

    estimator.forward = MethodType(_forward, estimator)
    estimator._voxcpm2_npu_graph_runner = graph_runner
    logger.info("VoxCPM2 LocDiT NPUGraph replay enabled (max_graphs=%d)", _MAX_GRAPHS)
