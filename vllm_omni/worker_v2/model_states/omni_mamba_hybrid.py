# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni lifecycle for hybrid attention + Mamba / linear-attention models.

Upstream MRv2 gives hybrid models ``MambaHybridModelState`` for their
recurrent-state bookkeeping, while the Omni runner drives preprocess,
postprocess and intermediate buffers through ``OmniModelState``. Both extend
``DefaultModelState`` and chain their overrides through ``super()``, so the
method resolution order runs the Omni hooks around the hybrid state ones.
"""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class OmniMambaHybridModelState(OmniModelState, MambaHybridModelState):
    pass
