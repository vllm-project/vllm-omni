# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations


def apply_patches() -> None:
    from vllm_omni.platforms.npu._310p.patch.patch_registry import (
        apply_qwen3_tts_patches,
    )

    apply_qwen3_tts_patches()
