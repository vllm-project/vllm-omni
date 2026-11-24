# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 state for Higgs Audio V2 request binding and cleanup."""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

__all__ = ["HiggsAudioV2ModelState"]


class HiggsAudioV2ModelState(OmniModelState):
    """Keep model-owned audio history aligned with MRv2 request rows."""

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        num_reqs = input_batch.num_reqs
        self.model.prepare_runner_inputs(
            req_ids=input_batch.req_ids[:num_reqs],
            query_start_loc=input_batch.query_start_loc_np[: num_reqs + 1],
            input_ids=input_batch.input_ids,
            positions=input_batch.positions,
        )
        return inputs

    def on_requests_finished(self, req_ids: set[str]) -> None:
        super().on_requests_finished(req_ids)
        self.model.on_requests_finished(req_ids)
