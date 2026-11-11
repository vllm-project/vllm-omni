# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-agnostic MoE specialty CUDA-graph runner checks (#9865 / #9911).

Shared V1/V2 runners call these helpers so they do not import a specific
model package. Specialty capture/replay itself still lives under Mammoth.
"""

from __future__ import annotations


def moe_routing_specialize_enabled(runner: object) -> bool:
    return bool(getattr(getattr(runner, "compilation_config", None), "cudagraph_specialize_moe_routing", False))


def runner_supports_moe_routing_specialty(runner: object) -> bool:
    """True when this runner implements mask-keyed MoE capture/replay."""
    return bool(getattr(type(runner), "supports_moe_routing_specialty", False))


def ensure_moe_specialty_runner_compatible(runner: object) -> None:
    """Fail after model load if specialty graphs are enabled on a no-op runner.

    ``cudagraph_specialize_moe_routing`` is normally set during Mammoth model
    ``__init__`` (inside ``load_model``) when the stage uses a FULL cudagraph
    mode. Checking here — after that load — covers both V1 and V2 runners.
    """
    if not moe_routing_specialize_enabled(runner):
        return
    if runner_supports_moe_routing_specialty(runner):
        return
    raise RuntimeError(
        "cudagraph_specialize_moe_routing is enabled but "
        f"{type(runner).__name__} does not implement MoE specialty "
        "capture/replay. Use V1 MammothGPUARModelRunner for MammothModa2 "
        "FULL AR stages, or a cudagraph_mode without FULL."
    )
