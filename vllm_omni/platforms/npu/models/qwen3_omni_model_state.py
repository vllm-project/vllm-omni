# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni request lifecycle with Ascend V2 attention metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


class NPUQwen3OmniModelState(OmniModelState, AscendModelState):
    """Keep Omni preprocessing and M-RoPE while using Ascend attention.

    Cooperative initialization runs AscendModelState before DefaultModelState.
    OmniModelState.prepare_attn delegates to AscendModelState, which consumes
    AscendInputBatch.attn_state and seq_lens_np rather than CUDA metadata.
    """

    uses_ascend_attn_metadata = True

    def postprocess_model_output(
        self,
        model_output: object,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        config = self.vllm_config
        if (
            self.model.model_stage == "thinker"
            and getattr(config.model_config, "async_chunk", False)
            and not config.cache_config.enable_prefix_caching
            and config.kv_transfer_config is None
            and config.speculative_config is None
            and isinstance(model_output, tuple)
            and len(model_output) == 2
            and isinstance(model_output[0], torch.Tensor)
            and input_batch.num_reqs > 0
            and all(
                info.get("omni_final_stage_id") == 0 and not info.get("omni_force_kv_transfer")
                for info in self.intermediate_buffer.gather(input_batch)
            )
        ):
            # The batch ends at the text output, so Talker conditioning has no consumer.
            text_hidden, _ = model_output
            return text_hidden.reshape(-1, text_hidden.shape[-1]), {}
        return super().postprocess_model_output(model_output, input_batch, req_states)
