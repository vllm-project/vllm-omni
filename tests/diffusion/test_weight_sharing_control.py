# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sharing RPCs preserve every TP rank result and drain requests before detach."""

import asyncio

import pytest

from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc, WorkerWrapperBase
from vllm_omni.distributed.weight_sharing.runtime import CONTROL_METHOD, RANK_RESULTS_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("remote_failed", [False, True])
def test_sharing_control_preserves_rank_results_and_remote_errors(mocker, remote_failed):
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 0
    proc.result_mq = object()
    proc.worker = mocker.Mock(spec=WorkerWrapperBase)
    proc.worker.execute_method.return_value = {"rank": 0, "receipt": b"receipt-0"}

    def gather(local):
        return [
            local,
            {
                "rank": 1,
                "ok": not remote_failed,
                "error_type": "RuntimeError",
                "error": "rank 1 import failed" if remote_failed else None,
                RANK_RESULTS_KEY: {"rank": 1, "receipt": b"receipt-1"},
            },
        ]

    mocker.patch.object(proc, "_gather_rpc_rank_statuses", side_effect=gather)
    envelope, reply = proc._execute_rpc(
        {
            "method": CONTROL_METHOD,
            "args": ("status", ()),
            "output_rank": 0,
            "exec_all_ranks": True,
            "collect_rank_status": True,
        }
    )
    assert reply
    if remote_failed:
        with pytest.raises(RuntimeError, match="rank 1 import failed"):
            MultiprocDiffusionExecutor._unwrap_rpc_result_envelope(envelope)
    else:
        result = MultiprocDiffusionExecutor._unwrap_rpc_result_envelope(envelope)
        assert result[RANK_RESULTS_KEY] == [
            {"rank": 0, "receipt": b"receipt-0"},
            {"rank": 1, "receipt": b"receipt-1"},
        ]


def test_quiesce_waits_for_active_diffusion_requests():
    async def run():
        proc = object.__new__(StageDiffusionProc)
        finish = asyncio.Event()
        request = asyncio.create_task(finish.wait())
        proc._active_tasks = {"request": request}
        quiesce = asyncio.create_task(proc._handle_collective_rpc("weight_sharing_quiesce", 1, (), {}))
        await asyncio.sleep(0)
        assert proc._sharing_quiesced and not quiesce.done()
        finish.set()
        assert await quiesce is True

    asyncio.run(run())


def test_request_drain_aborts_and_waits_without_sealing_diffusion_stage(mocker):
    async def run():
        proc = object.__new__(StageDiffusionProc)
        proc._engine = mocker.Mock()
        finish = asyncio.Event()
        request = asyncio.create_task(finish.wait())
        proc._active_tasks = {"request": request}
        drain = asyncio.create_task(proc._handle_collective_rpc("weight_sharing_drain", 1, (), {}))
        await asyncio.sleep(0)
        proc._engine.abort.assert_called_once_with("request")
        assert not proc._sharing_quiesced and not drain.done()
        finish.set()
        assert await drain is True
        assert not proc._sharing_quiesced

    asyncio.run(run())
