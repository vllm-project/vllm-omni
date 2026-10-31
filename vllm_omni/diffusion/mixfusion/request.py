# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request MixFusion opt-in plumbing (framework, model-agnostic).

The request-side switches (ragged latents and chunk-policy thresholds) are
stamped onto a newly admitted step request by the runner, so a pipeline never
carries MixFusion-specific logic. The opt-in is gated on the pipeline's
``supports_mixfusion`` capability and may be overridden per request through
``sampling.extra_args``.
"""

from __future__ import annotations

from typing import Any

# Defaults for requests that do not carry per-request MixFusion extras.
DEFAULT_MIN_CHUNK_TOKENS = 256
DEFAULT_MAX_CHUNKS = 128


def mixfusion_request_extras(sampling: Any, od_config: Any, *, supported: bool) -> dict[str, object]:
    """Build the ``state.extra`` entries that opt a request into MixFusion.

    Returns an empty dict unless the pipeline supports MixFusion and the
    request has not opted out. The per-request ``enable_mixfusion`` flag takes
    precedence over the server-level default, matching the historical
    pipeline-side behavior.
    """
    if not supported:
        return {}
    extra_args = getattr(sampling, "extra_args", {}) or {}
    if not bool(
        extra_args.get(
            "enable_mixfusion",
            bool(getattr(od_config, "enable_mixfusion", False)),
        )
    ):
        return {}
    return {
        "use_ragged_latents": True,
        "mixfusion_min_chunk_tokens": extra_args.get(
            "mixfusion_min_chunk_tokens",
            getattr(od_config, "mixfusion_min_chunk_tokens", DEFAULT_MIN_CHUNK_TOKENS),
        ),
        "mixfusion_max_chunks": extra_args.get(
            "mixfusion_max_chunks",
            getattr(od_config, "mixfusion_max_chunks", DEFAULT_MAX_CHUNKS),
        ),
    }
