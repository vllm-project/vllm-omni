# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility imports for the shared model/runner output snapshot contract."""

from vllm_omni.model_executor.output_snapshot import (
    PackedOutputSnapshot,
    RequestOutputSnapshot,
    pack_output_snapshot,
)

__all__ = ["PackedOutputSnapshot", "RequestOutputSnapshot", "pack_output_snapshot"]
