# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni request lifecycle with Ascend V2 attention metadata."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class NPUQwen3OmniModelState(OmniModelState, AscendModelState):
    """Keep Omni preprocessing and M-RoPE while using Ascend attention.

    Cooperative initialization runs AscendModelState before DefaultModelState.
    OmniModelState.prepare_attn delegates to AscendModelState, which consumes
    AscendInputBatch.attn_state and seq_lens_np rather than CUDA metadata.
    """

    uses_ascend_attn_metadata = True
