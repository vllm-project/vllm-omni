# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS Local request state with Ascend attention metadata."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState


class NPUMossLocalModelState(MossLocalModelState, AscendModelState):
    """Local slot state whose attention metadata is the Ascend V2 one.

    Slots, code rows and audio buffers stay on ``MossLocalModelState``. The
    base chain is ``MossLocalModelState`` -> ``OmniModelState`` ->
    ``AscendModelState`` -> ``DefaultModelState``, so the Ascend setup runs
    during ``__init__`` and ``prepare_attn`` ends in the Ascend implementation.
    """

    uses_ascend_attn_metadata = True
