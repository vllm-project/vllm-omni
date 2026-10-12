# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Response-owned codec updates remain independent before the generation step."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm import SamplingParams
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.lychee_fd.token2wav import (
    LycheeToken2WavForConditionalGeneration,
    LycheeToken2WavSessionStore,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class QueueScheduler(OmniGenerationScheduler):
    def __init__(self):
        self.requests = {}
        self.enqueued = []
        self.running = []
        self.kv_holding_waiting = []
        self.deferred_waiting = set()
        self.chunk_transfer_adapter = None
        self.connector = None
        self.log_stats = False
        self.spec_decode_metrics_level = "none"
        self.num_waiting_for_streaming_input = 0

    def _enqueue_waiting_request(self, request):
        self.enqueued.append(request)


class RecordingCore:
    sample_rate = 24000
    chunk_size = 25
    pre_lookahead_len = 3

    def __init__(self):
        self.calls = []

    def synthesize(self, tokens, state, *, final):
        self.calls.append((list(tokens), state.owner.response_number, final))
        state.stream_cache = {"owned": torch.ones(1)}
        return torch.ones(len(tokens))


def packet(ids, *, number, sequence, final):
    request = Request("queued-owner", ids, SamplingParams(max_tokens=1), pooling_params=None)
    request.resumable = True
    metadata = dict(
        session_id="queued-session",
        response_id=f"queued-owner:response:{number}",
        session_epoch=0,
        execution_epoch=0,
        response_number=number,
        request_id="queued-owner",
        chunk_seq=sequence,
        tick=number * 10 + sequence,
        final=final,
    )
    request.additional_information = serialize_additional_information({"lychee_t2w": metadata})
    return request


def test_generation_queue_keeps_multiple_response_metadata_and_exact_codec_delta_before_first_step():
    scheduler = QueueScheduler()
    packets = [
        packet([0, 1], number=1, sequence=0, final=True),
        packet(list(range(28)), number=2, sequence=0, final=False),
        packet([28], number=2, sequence=1, final=True),
    ]
    for item in packets:
        scheduler.add_request(item)
    resident = scheduler.requests["queued-owner"]
    assert len(resident.streaming_queue) == 2
    assert resident.prompt_token_ids == [0, 1]
    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    core = RecordingCore()
    wrapper.sessions = LycheeToken2WavSessionStore(core, "voice.wav")
    observed = []
    for index in range(3):
        # The runner installs each dequeued update's metadata in the runtime
        # buffer. Exercise its real wire decoder before calling the model.
        buffer = OmniIntermediateBuffer(1)
        buffer.add_request(
            0,
            SimpleNamespace(
                req_id=resident.request_id, additional_information=resident.additional_information, mm_features=[]
            ),
        )
        information = buffer.gather(SimpleNamespace(idx_mapping_np=[0]))
        result = wrapper.forward(
            torch.tensor(resident.prompt_token_ids),
            torch.arange(len(resident.prompt_token_ids)),
            runtime_additional_information=information,
        )
        output = result.multimodal_outputs
        observed.append(
            (
                resident.prompt_token_ids.copy(),
                output["chunk.lychee_t2w.response_number"].item(),
                output["chunk.lychee_t2w.chunk_seq"].item(),
                output["chunk.lychee_t2w.final"].item(),
            )
        )
        resident.num_computed_tokens = len(resident.prompt_token_ids)
        resident.status = RequestStatus.FINISHED_STOPPED
        assert scheduler._handle_stopped_request(resident) is False
        assert len(resident.streaming_queue) == max(0, 1 - index)
    assert observed == [([0, 1], 1, 0, True), (list(range(28)), 2, 0, False), ([28], 2, 1, True)]
    assert core.calls == [([0, 1], 1, True), (list(range(28)), 2, False), ([25, 26, 27, 28], 2, True)]
    assert resident.status is RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert not wrapper.sessions.states
