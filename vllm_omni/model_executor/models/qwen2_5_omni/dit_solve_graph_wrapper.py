# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph wrapper for the Qwen2.5-Omni DiT flow-matching solve.

The RK4 solve issues ~1400 kernels per model eval and runs 36 evals per
response (9 intervals x 4 evals x CFG batch 2 at the served ``steps=10``).
At streaming-chunk mel sizes the solve is launch-bound, so replaying the
whole solve as a single CUDA graph is substantially faster than eager
execution -- and bit-exact by construction, since replay runs the identical
kernels in the identical order. The solve became capturable when its loop
was made free of data-dependent control flow (no tensor-to-bool
conversions); blocking host syncs are not capture-legal.

Graphs are captured lazily per solve key (mel length, reference length,
code length, step count, guidance scale) and cached up to ``max_entries``.
Unseen shapes beyond the cap and capture failures fall back to the eager
solve. Shapes are matched exactly on purpose: the DiT has look-ahead
attention layers, so zero-padding the mel axis to a bucket size would
change outputs for real frames (not bit-exact).

Enable/disable with ``VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH`` (default on);
cache size with ``VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH_MAX_ENTRIES``
(default 8).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_ENV_ENABLED = "VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH"
_ENV_MAX_ENTRIES = "VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH_MAX_ENTRIES"
DEFAULT_MAX_ENTRIES = 8


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off")


def dit_solve_cudagraph_enabled() -> bool:
    return _env_flag(_ENV_ENABLED, True)


def dit_solve_cudagraph_max_entries() -> int:
    try:
        return max(1, int(os.getenv(_ENV_MAX_ENTRIES, DEFAULT_MAX_ENTRIES)))
    except ValueError:
        return DEFAULT_MAX_ENTRIES


@dataclass
class _SolveGraphEntry:
    graph: CUDAGraph
    static_conditioning: torch.Tensor
    static_reference_mel: torch.Tensor
    static_code: torch.Tensor
    static_y0: torch.Tensor
    static_time_points: torch.Tensor
    static_output: torch.Tensor


class DiTSolveCUDAGraphWrapper:
    """Capture and replay whole RK4 solves keyed on exact input shapes.

    ``build_ode_fn(conditioning, reference_mel, code, guidance_scale,
    initial_state)`` must return the ODE function closed over the *static*
    tensors it is given, and ``integrate_fn(ode_function, initial_value,
    time_points)`` runs the solver loop. Both are injected so this file does
    not import the model it wraps.
    """

    def __init__(
        self,
        build_ode_fn: Callable[..., Callable],
        integrate_fn: Callable[..., torch.Tensor],
        enabled: bool = True,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self._build_ode_fn = build_ode_fn
        self._integrate_fn = integrate_fn
        self.enabled = enabled
        self.max_entries = max_entries
        self.entries: dict[tuple, _SolveGraphEntry] = {}

    def solve(
        self,
        conditioning_vector: torch.Tensor,
        reference_mel: torch.Tensor,
        quantized_code: torch.Tensor,
        y0: torch.Tensor,
        time_points: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor | None:
        """Replay a captured solve for these inputs; ``None`` if not applicable.

        The caller falls back to the eager solve whenever ``None`` is
        returned (graphs disabled, non-CUDA input, non-unit batch, cache
        full, or capture failed).
        """
        if not self.enabled or y0.device.type != "cuda" or y0.shape[0] != 1 or torch.cuda.is_current_stream_capturing():
            return None
        key = (
            int(y0.shape[1]),
            int(reference_mel.shape[1]),
            int(quantized_code.shape[-1]),
            len(time_points),
            float(guidance_scale),
        )
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_entries:
                return None
            entry = self._capture(
                key, conditioning_vector, reference_mel, quantized_code, y0, time_points, guidance_scale
            )
            if entry is None:
                return None
        entry.static_conditioning.copy_(conditioning_vector)
        entry.static_reference_mel.copy_(reference_mel)
        entry.static_code.copy_(quantized_code)
        entry.static_y0.copy_(y0)
        entry.static_time_points.copy_(time_points)
        entry.graph.replay()
        return entry.static_output.clone()

    def _capture(
        self,
        key: tuple,
        conditioning_vector: torch.Tensor,
        reference_mel: torch.Tensor,
        quantized_code: torch.Tensor,
        y0: torch.Tensor,
        time_points: torch.Tensor,
        guidance_scale: float,
    ) -> _SolveGraphEntry | None:
        try:
            # Build the statics and capture as *normal* tensors (not inference
            # tensors) so later copy_/replay calls work under any autograd
            # context -- capture may be triggered from inside a caller's
            # ``torch.inference_mode()`` block, and in-place updates to an
            # inference tensor outside inference mode are not allowed.
            with torch.inference_mode(False), torch.no_grad():
                static_conditioning = conditioning_vector.detach().clone()
                static_reference_mel = reference_mel.detach().clone()
                static_code = quantized_code.detach().clone()
                static_y0 = y0.detach().clone()
                static_time_points = time_points.detach().clone()

                def run() -> torch.Tensor:
                    ode_function = self._build_ode_fn(
                        static_conditioning,
                        static_reference_mel,
                        static_code,
                        guidance_scale,
                        static_y0,
                    )
                    return self._integrate_fn(ode_function, static_y0, static_time_points)

                # Warmup so cuBLAS/cuDNN handles and workspaces exist before
                # capture (same procedure as the Qwen3-Omni decoder graphs).
                run()
                current_omni_platform.synchronize()

                capture_start = time.perf_counter()
                graph = CUDAGraph()
                with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                    static_output = run()
            entry = _SolveGraphEntry(
                graph=graph,
                static_conditioning=static_conditioning,
                static_reference_mel=static_reference_mel,
                static_code=static_code,
                static_y0=static_y0,
                static_time_points=static_time_points,
                static_output=static_output,
            )
            self.entries[key] = entry
            logger.info(
                "Captured Qwen2.5-Omni DiT solve CUDA graph for key %s (%d/%d entries, %.1f ms)",
                key,
                len(self.entries),
                self.max_entries,
                (time.perf_counter() - capture_start) * 1000.0,
            )
            return entry
        except Exception:
            logger.warning(
                "Failed to capture Qwen2.5-Omni DiT solve CUDA graph for key %s; using eager",
                key,
                exc_info=True,
            )
            return None
