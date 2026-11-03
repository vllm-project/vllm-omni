# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend MRv2 model state for the Qwen3-TTS autoregressive Talker."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class NPUQwen3TTSTalkerModelState(OmniModelState, AscendModelState):
    """Combine Omni request lifecycle with Ascend V2 attention metadata."""

    uses_ascend_attn_metadata = True
