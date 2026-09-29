# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker output helpers.

Package seeded ahead of the async-output refactor (RFC #5450 series, G5-C2),
which relocates the remaining payload-copy helpers, snapshot types, and
`ExecuteModelState` here. Currently hosts the per-request multimodal payload
builders.
"""

from vllm_omni.worker.output.payload_build import (
    build_combined_prefix_cache_mm_payload,
    build_omni_mm_payload,
    unwrap_combined_payload_value,
)

__all__ = [
    "build_combined_prefix_cache_mm_payload",
    "build_omni_mm_payload",
    "unwrap_combined_payload_value",
]
