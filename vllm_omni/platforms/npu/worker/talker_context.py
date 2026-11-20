# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eligibility for replaying a short Talker graph on a full-context worker."""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)
SHORT_CONTEXT = 512
MODEL_CONTEXT = 4096
HYBRID_MODE_FILE = os.getenv("MINICPMO_NPU_TALKER_HYBRID_CONTEXT_FILE")
TRACE_FILE = os.getenv("MINICPMO_C8_TRACE_FILE")


def hybrid_enabled(model_len):
    return model_len == MODEL_CONTEXT and HYBRID_MODE_FILE is not None and os.path.isfile(HYBRID_MODE_FILE)


def scheduled_length(batch, rows, width):
    """Use the host length at the end of the reserved K-step, including speculative slots."""
    lengths = getattr(batch, "seq_lens_np", None)
    if lengths is None or len(lengths) < rows:
        return None
    lengths = lengths[:rows]
    if int(lengths.min()) < width:
        return None
    return int(lengths.max())


def trace_context(batch, path, length, capacity=None):
    if TRACE_FILE and os.path.isfile(TRACE_FILE):
        logger.info("[talker-context] path=%s length=%s graph=%s requests=%s", path, length, capacity, batch.req_ids)
