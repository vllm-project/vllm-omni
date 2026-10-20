# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single source of truth for the Talker K-step engagement contract.

The multi-frame Talker decode loop is armed from the engine-side
``speculative_config`` that stage_config.py injects for stage 1 (method
``ngram`` with ``num_speculative_tokens == K - 1``). The generic AR scheduler
and the model-side gate used to interpret that config independently and had
already drifted: a missing ``method`` defaulted to armed on the scheduler side
and disabled on the model side, and only one side enforced the frame ceiling.
This module is now the only interpreter; both consumers call
:func:`talker_kstep_frames` and cannot disagree about K.
"""

from typing import Any

# Hard ceiling on K codec frames per Talker decode step. Same TND ceiling as
# stage 0: at most 16 query positions per sequence.
MINICPMO_TALKER_FRAMES_MAX = 16


def talker_kstep_frames(model_stage: Any, speculative_config: Any) -> int:
    """K codec frames per Talker step, or 0 when the loop must not engage.

    ``method`` must be present and equal to ``"ngram"``: the engine injects it
    explicitly, so a config without it is not a K-step config and must not
    half-arm the pipeline. ``K = num_speculative_tokens + 1``; ``K == 1``
    degenerates to the ordinary single-frame path and reads as 0. A ``K`` above
    the ceiling raises -- it cannot be served, and must fail loudly at
    configuration time instead of being silently clamped.
    """
    if model_stage != "tts":
        return 0
    if isinstance(speculative_config, dict):
        method = speculative_config.get("method")
        num_spec = speculative_config.get("num_speculative_tokens", 0) or 0
    else:
        method = getattr(speculative_config, "method", None)
        num_spec = getattr(speculative_config, "num_speculative_tokens", 0) or 0
    if method != "ngram":
        return 0
    frames = int(num_spec) + 1
    if frames < 2:
        return 0
    if frames > MINICPMO_TALKER_FRAMES_MAX:
        raise ValueError(f"K-step frame count is capped at {MINICPMO_TALKER_FRAMES_MAX}, got {frames}")
    return frames
