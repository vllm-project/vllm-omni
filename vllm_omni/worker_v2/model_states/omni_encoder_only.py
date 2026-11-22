# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni lifecycle for encoder-only (BERT/RoBERTa/ModernBERT) pooling models.

Upstream MRv2 gives encoder-only models ``EncoderOnlyModelState`` so their
non-causal attention layers build metadata without a KV cache, while the Omni
runner drives preprocess, postprocess and intermediate buffers through
``OmniModelState``. Both extend ``DefaultModelState`` and chain their overrides
through ``super()``, so the method resolution order runs the Omni hooks around
the encoder-only state ones.
"""

from __future__ import annotations

from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class OmniEncoderOnlyModelState(OmniModelState, EncoderOnlyModelState):
    def remove_request(self, req_index_or_id: int | str) -> None:
        req_index = self._resolve_req_index(req_index_or_id)
        req_id = (
            self.intermediate_buffer.buffers[req_index].get("req_id")
            if req_index is not None
            else (req_index_or_id if isinstance(req_index_or_id, str) else None)
        )
        super().remove_request(req_index_or_id)
        if isinstance(req_id, str):
            EncoderOnlyModelState.remove_request(self, req_id)
