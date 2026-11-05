# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request MixFusion opt-in plumbing (framework, model-agnostic).

The scheduler and the runner both ask whether a request opts into MixFusion;
this module is the single answer. The scheduler relaxes resolution out of
the step-batch key for such requests, and the runner stamps the ragged
latents switch and chunk-policy thresholds onto a newly admitted step
request, so neither a pipeline nor a scheduler class carries MixFusion logic.
The opt-in is gated on the pipeline's ``supports_mixfusion`` capability and
may be overridden per request through ``sampling.extra_args``.
"""

from __future__ import annotations

from typing import Any

# Defaults for requests that do not carry per-request MixFusion extras.
DEFAULT_MIN_CHUNK_TOKENS = 256
DEFAULT_MAX_CHUNKS = 128

#: Step-batch key fields MixFusion relaxes so mixed resolutions share a step.
#: A dense fallback must restore them: requests that differ here may share a
#: latent sequence length (equal area, different aspect ratio) but not RoPE.
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

    Step batches resolve shapes per request (packed or serial ragged). Returns
    ``None`` when the request cannot share a key at all (per-prompt inputs).
    """
    if not mixfusion_requested(request.sampling_params, od_config, supported=supported):
        return values
    if has_per_prompt_inputs(request.prompt):
        return None
    for name in RELAXED_STEP_KEY_FIELDS:
        values[name] = None
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


def mixfusion_request_extras(sampling: Any, od_config: Any, *, supported: bool) -> dict[str, object]:
    """Build the ``state.extra`` entries that opt a request into MixFusion.

    Returns an empty dict unless the request opts in on a capable pipeline.
    """
    if not mixfusion_requested(sampling, od_config, supported=supported):
        return {}
    extra_args = sampling.extra_args or {}
    return {
        "use_ragged_latents": True,
        "mixfusion_min_chunk_tokens": extra_args.get(
            "mixfusion_min_chunk_tokens", od_config.mixfusion_min_chunk_tokens
        ),
        "mixfusion_max_chunks": extra_args.get("mixfusion_max_chunks", od_config.mixfusion_max_chunks),
    }
