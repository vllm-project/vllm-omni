# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_QWEN3_TTS_TALKER_PATCHED = False
_WORKER_PATCHED = False


def apply_patches() -> None:
    apply_worker_patch()


def apply_worker_patch() -> None:
    global _WORKER_PATCHED

    if _WORKER_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.patch_worker import (
        apply_patch,
    )

    apply_patch()
    _WORKER_PATCHED = True


def apply_qwen3_tts_talker_patches() -> None:
    global _QWEN3_TTS_TALKER_PATCHED

    if _QWEN3_TTS_TALKER_PATCHED:
        return

    # The talker owns the residual code predictor; Code2Wav does not need this.
    from vllm_omni.platforms.npu._310p.patch.patch_qwen3_tts_code_predictor import (
        apply_patch as apply_code_predictor_patch,
    )

    apply_code_predictor_patch()
    _QWEN3_TTS_TALKER_PATCHED = True
