# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 request state with Ascend attention metadata."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.voxcpm2.model_state import VoxCPM2ModelState


class NPUVoxCPM2ModelState(VoxCPM2ModelState, AscendModelState):
    """VoxCPM2 audio state whose attention metadata is the Ascend V2 one.

    Slots, latents and audio buffers stay on ``VoxCPM2ModelState``. The base
    chain is ``VoxCPM2ModelState`` -> ``OmniModelState`` -> ``AscendModelState``
    -> ``DefaultModelState``, so the Ascend setup runs during ``__init__`` and
    ``prepare_attn`` ends in the Ascend implementation, which builds metadata
    from ``attn_state`` and ``seq_lens_np`` on the Ascend batch.

    Multiple inheritance works because VoxCPM2 is not a mamba hybrid. The
    sibling ``NPUFishOmniModelState`` needs ``AscendMambaHybridModelState``,
    which is not a base of ``FishOmniModelState``, so it forwards
    ``prepare_attn`` and ``preprocess_state`` explicitly instead.
    """

    uses_ascend_attn_metadata = True
