# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_MODEL_PATCHED = False
_WORKER_PATCHED = False


def apply_patches() -> None:
    apply_qwen3_tts_worker_patch()


def apply_qwen3_tts_worker_patch() -> None:
    global _WORKER_PATCHED

    if _WORKER_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.patch_qwen3_tts_worker import (
        apply_patch,
    )

    apply_patch()
    _WORKER_PATCHED = True


def apply_qwen3_tts_model_patches() -> None:
    global _MODEL_PATCHED

    if _MODEL_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.patch_qwen3_tts_code_predictor import (
        apply_patch as apply_code_predictor_patch,
    )
    from vllm_omni.platforms.npu._310p.patch.patch_qwen3_tts_prompt_builder import (
        apply_patch as apply_prompt_builder_patch,
    )
    from vllm_omni.platforms.npu._310p.patch.patch_qwen3_tts_talker import (
        apply_patch as apply_talker_patch,
    )

    apply_code_predictor_patch()
    apply_prompt_builder_patch()
    apply_talker_patch()
    _MODEL_PATCHED = True
