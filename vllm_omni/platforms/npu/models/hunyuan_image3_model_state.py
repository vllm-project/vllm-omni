# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage3 AR lifecycle and M-RoPE with Ascend MRv2 attention."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class NPUHunyuanImage3ModelState(OmniModelState, AscendModelState):
    """Use Ascend attention metadata while preserving Omni inputs and sampling."""

    uses_ascend_attn_metadata = True
