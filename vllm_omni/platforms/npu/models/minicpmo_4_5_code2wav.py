# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inject MiniCPM-o Code2Wav NPUGraph acceleration on Ascend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)

_PATCHED = False
_original_build_backend = None
_ENABLE_KEY = "code2wav_enable_npu_graph"
_MAX_GRAPHS_KEY = "code2wav_max_npu_graphs"


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _graph_config(model: object) -> dict[str, object]:
    config = getattr(getattr(model, "vllm_config", None), "additional_config", None)
    return dict(config) if isinstance(config, Mapping) else {}


def prepare_code2wav_graph_runtime() -> None:
    """Select graph-capturable ACLNN kernels before Token2Wav is loaded."""
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError(
            "MiniCPM-o Code2Wav NPUGraph capture is incompatible with "
            "ASCEND_LAUNCH_BLOCKING=1; unset it or set it to 0 before startup."
        )
    npu = torch.npu
    npu.config.allow_internal_format = False
    npu.set_compile_mode(jit_compile=False)
    logger.info("Configured MiniCPM-o Code2Wav NPUGraph runtime (allow_internal_format=False, jit_compile=False)")


def _flow_execution_context(device: torch.device, *, require_math: bool):
    if device.type != "npu":
        return nullcontext()
    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    return npu_token2wav_sdpa_context(require_math=require_math)


def _patched_build_backend(self) -> None:
    if self.backend is not None:
        return

    config = _graph_config(self)
    max_graphs = max(0, int(config.get(_MAX_GRAPHS_KEY, 32)))
    graph_enabled = max_graphs > 0 and _config_bool(config.get(_ENABLE_KEY), False)
    if graph_enabled:
        # NPUOmniPlatform enables internal format for quantized LLM kernels.
        # Code2Wav uses regular convolution kernels that must remain in the
        # graph-capturable ACLNN path.
        prepare_code2wav_graph_runtime()

    assert _original_build_backend is not None
    _original_build_backend(self)

    graph_runner = None
    if graph_enabled:
        graph_runner = NPUExactGraphRunner(
            max_graphs=max_graphs,
            component_name="MiniCPM-o Code2Wav",
            disable_config_hint=(
                "set platforms.npu.stages[stage_id=2].additional_config.code2wav_enable_npu_graph=false"
            ),
        )
        if self.backend.speech_window.device.type == "npu" and not graph_runner.is_supported():
            raise RuntimeError(
                "MiniCPM-o Code2Wav NPUGraph capture requires torch.npu "
                "NPUGraph, graph, is_current_stream_capturing, and synchronize APIs."
            )

    self.backend.configure_acceleration(
        graph_runner=graph_runner,
        flow_execution_context=lambda device: _flow_execution_context(
            device,
            require_math=graph_enabled,
        ),
    )
    if graph_enabled:
        logger.info(
            "MiniCPM-o Code2Wav NPUGraph replay enabled (max_graphs=%d)",
            max_graphs,
        )


def apply_minicpmo_4_5_code2wav_patch() -> None:
    """Patch the generic Code2Wav backend builder with Ascend acceleration."""
    global _PATCHED, _original_build_backend
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
        MiniCPMO45Code2Wav,
    )

    _original_build_backend = MiniCPMO45Code2Wav._build_backend
    MiniCPMO45Code2Wav._build_backend = _patched_build_backend  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for MiniCPM-o 4.5 Code2Wav")
