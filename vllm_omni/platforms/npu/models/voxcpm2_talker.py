# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Enable VoxCPM2 LocDiT NPUGraph acceleration on Ascend."""

from __future__ import annotations

from types import MethodType

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)

_MAX_GRAPHS = 8
_MODEL_CLASS_NAME = "VoxCPM2TalkerForConditionalGeneration"


def setup_voxcpm2_loc_dit_npu_graph(model: object) -> None:
    """Wrap VoxCPM2's tensor-only LocDiT estimator with NPUGraph."""
    if type(model).__name__ != _MODEL_CLASS_NAME:
        return

    tts = getattr(model, "tts", None)
    feat_decoder = getattr(tts, "feat_decoder", None)
    estimator = getattr(feat_decoder, "estimator", None)
    if estimator is None or getattr(estimator, "_voxcpm2_npu_graph_runner", None) is not None:
        return

    graph_runner = NPUExactGraphRunner(
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
