# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super reasoning stage on Model Runner V2.

The reasoning model records each step's first input token and its M-RoPE
position (``prepare_runner_inputs``), so that when a request feeds back
``<|traj_future_start|>`` the KV handoff can send the expert its prefix length
and trigger position (``get_kv_transfer_metadata``). This state calls that hook
before every forward with the step's tokens and positions.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class Alpamayo2SuperModelState(OmniModelState):
    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        num_reqs = input_batch.num_reqs
        self.model.prepare_runner_inputs(
            input_ids=input_batch.input_ids,
            # M-RoPE positions of this step's tokens, (3, num_tokens).
            positions=inputs.get("positions", input_batch.positions),
            inputs_embeds=None,
            req_ids=input_batch.req_ids[:num_reqs],
            num_computed_tokens=input_batch.num_computed_tokens_np[:num_reqs],
            num_scheduled_tokens=input_batch.num_scheduled_tokens[:num_reqs],
            input_ids_buffer=input_batch.input_ids,
        )
        return inputs
