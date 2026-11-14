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

# The Talker's vLLM-level vocabulary is two tokens wide: `compute_logits`
# returns one-hot stop rows ([0, -inf] / [-inf, 0]) and `sample` takes their
# argmax, so 0 means "keep going" and 1 is the stop token the deploy config
# names in `stop_token_ids`.
CONTINUE_TOKEN_ID = 0

# The id that same row emits to mean stop: the stop column wins the argmax,
# `parse_output` keeps ids below the two-wide vocab, and vLLM's `check_stop`
# only fires if this id is in the request's `stop_token_ids` -- which stage 1's
# pipeline constraints derive from this constant.
STOP_TOKEN_ID = 1

# Width of the Talker vLLM-level head: the two-wide continue/stop row.
# input_batch.vocab_size must report the same, because InputBatch stores
# top_k = vocab_size as its no-top-k sentinel and that holds only for 0 or 2.
STOP_ROW_WIDTH = 2


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
    if method not in ("ngram", "ngram_gpu"):
        return 0
    frames = int(num_spec) + 1
    if frames < 2:
        return 0
    if frames > MINICPMO_TALKER_FRAMES_MAX:
        raise ValueError(f"K-step frame count is capped at {MINICPMO_TALKER_FRAMES_MAX}, got {frames}")
    return frames
