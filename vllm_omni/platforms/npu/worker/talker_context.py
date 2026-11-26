# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eligibility for replaying a short Talker graph on a full-context worker."""

SHORT_CONTEXT = 512
MODEL_CONTEXT = 4096


def hybrid_enabled(runner):
    if runner.model_config.max_model_len != MODEL_CONTEXT:
        return False
    options = getattr(runner.vllm_config, "additional_config", None) or {}
    return options.get("minicpmo_talker_hybrid_context") is True


def scheduled_length(batch, rows, width):
    """Use the host length at the end of the reserved K-step, including speculative slots."""
    lengths = getattr(batch, "seq_lens_np", None)
    if lengths is None or len(lengths) < rows:
        return None
    lengths = lengths[:rows]
    if int(lengths.min()) < width:
        return None
    return int(lengths.max())
