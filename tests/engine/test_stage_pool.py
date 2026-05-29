# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.stage_pool import StagePool


class _FakeDiffusionClient:
    stage_type = "diffusion"
    final_output = True

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.outputs: list[SimpleNamespace] = []

    async def add_request_async(self, request_id, request, params, **kwargs):
        self.submitted.append(request_id)

    def get_diffusion_output_nowait(self):
        return self.outputs.pop(0) if self.outputs else None


@pytest.mark.asyncio
async def test_stage_pool_routes_to_replica_with_less_inflight_work():
    clients = [_FakeDiffusionClient(), _FakeDiffusionClient()]
    pool = StagePool(1, clients)
    req_state = SimpleNamespace(sampling_params_list=[None, object()])

    assert await pool.submit_initial("r0", req_state, "prompt") == 0
    assert await pool.submit_initial("r1", req_state, "prompt") == 1

    clients[1].outputs.append(SimpleNamespace(request_id="r1", finished=True))
    assert pool.poll_diffusion_output(1).request_id == "r1"

    assert await pool.submit_initial("r2", req_state, "prompt") == 1
    assert clients[0].submitted == ["r0"]
    assert clients[1].submitted == ["r1", "r2"]
