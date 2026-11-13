# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 request state with Ascend hybrid attention metadata."""

from typing import Any

from vllm_omni.model_executor.models.fish_omni.model_state import FishOmniModelState


class NPUFishOmniModelState(FishOmniModelState):
    """Drama 3 request state whose attention metadata is the Ascend V2 one.

    Request slots, frame rows and sampling stay on ``FishOmniModelState``.
    ``prepare_attn`` and the GDN align pre-copy come from the Ascend hybrid
    state, which reads ``attn_state`` and ``seq_lens_np`` off the Ascend batch.
    """

    uses_ascend_attn_metadata = True

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        # AscendMambaHybridModelState fields used by its preprocess_state.
        self._layerwise_mamba_copy = None
        self._layer_state_ranges = None

    def prepare_attn(self, *args: Any, **kwargs: Any):
        from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

        return AscendMambaHybridModelState.prepare_attn(self, *args, **kwargs)

    def preprocess_state(self, *args: Any, **kwargs: Any):
        from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

        return AscendMambaHybridModelState.preprocess_state(self, *args, **kwargs)
