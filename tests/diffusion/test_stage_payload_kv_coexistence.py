# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("streaming", [False, True])
def test_subprocess_client_and_worker_preserve_both_transfer_contracts(streaming):
    """Exercise the IPC message and both process request builders without GPUs."""

    async def run():
        client = object.__new__(StageDiffusionClient)
        client._engine_dead = False
        client.stage_id = 1
        client.replica_id = 0
        client._encoder = MagicMock()
        client._encoder.encode.side_effect = lambda message: message
        client._request_socket = MagicMock()
        kv_params = {"remote_engine_id": "mooncake-producer", "remote_block_ids": [1, 2]}
        payload_sender = {"host": "10.0.0.1", "zmq_port": 50071}
        kv_sender = {0: {"host": "10.0.0.2", "zmq_port": 50171}}
        # Keep main's existing fifth positional argument bound to native KV.
        await client.add_request_async(
            "req-both",
            "prompt",
            OmniDiffusionSamplingParams(),
            kv_sender,
            kv_params,
            payload_sender_info=payload_sender,
        )
        message = client._request_socket.send.call_args.args[0]
        assert message["kv_transfer_params"] == kv_params
        assert message["payload_sender_info"] == payload_sender
        captured = []

        class Engine:
            async def step_streaming(self, request):
                captured.append(request)
                yield [OmniRequestOutput.from_diffusion(request_id=request.request_id, images=[])]

        proc = object.__new__(StageDiffusionProc)
        proc._engine = Engine()
        args = (message["request_id"], message["prompt"], message["sampling_params"], kv_sender, kv_params)
        if streaming:
            outputs = [
                output
                async for output in proc._process_streaming_request(
                    *args, payload_sender_info=message["payload_sender_info"]
                )
            ]
        else:
            outputs = [await proc._process_request(*args, payload_sender_info=message["payload_sender_info"])]
        assert outputs[0].request_id == "req-both"
        assert captured[0].kv_transfer_params == kv_params
        assert captured[0].kv_sender_info == kv_sender
        assert captured[0].payload_sender_info == payload_sender

    asyncio.run(run())
