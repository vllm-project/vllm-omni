# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared guardrail helpers for diffusion pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def is_guardrails_enabled(
    od_config: OmniDiffusionConfig,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> bool:
    """Resolve the active guardrail gate.

    Server-level ``od_config.model_config["guardrails"]`` decides whether the
    guardrail models are loaded at all (eager load at pipeline build time).
    When that is False, no per-request override can turn checks back on,
    because the guardrail models are never loaded.

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
