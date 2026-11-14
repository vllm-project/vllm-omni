# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni fork of the vLLM V1 worker data model.

Upstream vLLM removed Model Runner V1 in aac1b2a4f3 (#63373) for the 0.34
release, deleting the V1 data model this package forks. Files are vendored
verbatim from the rebase assignment baseline
4ba787ec9959e566d3b587a796dff07673324da5 with import lines rewritten to the
live vLLM 0.34 modules or to siblings in this package; each module's header
records any non-import deviation. ``misc`` also carries small shared
utilities deleted upstream (grammar bitmask application, the cudagraph
capture flag) that Omni's MRv1 stage runners and MRv2 MTP capture use.
"""

from vllm_omni.worker.v1_compat.async_output import AsyncGPUModelRunnerOutput
from vllm_omni.worker.v1_compat.input_batch import CachedRequestState, InputBatch
from vllm_omni.worker.v1_compat.metadata import SamplingMetadata
from vllm_omni.worker.v1_compat.misc import (
    apply_grammar_bitmask,
    set_cudagraph_capturing_enabled,
    validate_cudagraph_capturing_enabled,
)
from vllm_omni.worker.v1_compat.ngram_proposer_gpu import (
    update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts,
)

__all__ = [
    "AsyncGPUModelRunnerOutput",
    "CachedRequestState",
    "InputBatch",
    "SamplingMetadata",
    "apply_grammar_bitmask",
    "set_cudagraph_capturing_enabled",
    "update_ngram_gpu_tensors_incremental",
    "update_scheduler_for_invalid_drafts",
    "validate_cudagraph_capturing_enabled",
]
