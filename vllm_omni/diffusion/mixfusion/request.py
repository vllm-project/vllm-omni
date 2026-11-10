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

from typing import Any

# Defaults for requests that do not carry per-request MixFusion extras.
DEFAULT_MIN_CHUNK_TOKENS = 256
DEFAULT_MAX_CHUNKS = 128

#: Step-batch key fields MixFusion relaxes so mixed resolutions share a step.
#: Dense admission retains them: equal-area images can have different RoPE.
RELAXED_STEP_KEY_FIELDS = ("height", "width", "resolution")


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
