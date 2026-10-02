# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The batched Higgs async-chunk builder must emit exactly what the per-request one does."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
    talker2code2wav_async_chunk,
    talker2code2wav_async_chunk_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _manager():
    extra = {
        "codec_chunk_frames": 5,
        "codec_left_context_frames": 3,
        "codec_right_holdback_frames": 1,
        "initial_codec_chunk_frames": 2,
    }
    return SimpleNamespace(connector=SimpleNamespace(config={"extra": extra}), code_prompt_token_ids=defaultdict(list))


class _Request:
    def __init__(self, rid, finish_step):
        self.external_req_id = rid
        self.finish_step = finish_step
        self.step = 0

    def is_finished(self):
        return self.step >= self.finish_step


def _payload_items(payload):
    if payload is None:
        return None
    meta = payload.meta
    return (
        payload.codes.audio.tolist(),
        None if meta is None else (meta.left_context_size, meta.right_holdback_size, bool(meta.finished)),
    )


def test_batch_async_chunk_matches_per_request_builder():
    torch.manual_seed(0)
    single, batched = _manager(), _manager()
    finish = [23, 9, 31]
    requests_single = [_Request(f"r{i}", f) for i, f in enumerate(finish)]
    requests_batch = [_Request(f"r{i}", f) for i, f in enumerate(finish)]
    for step in range(32):
        outputs: list[dict | None] = []
        flags: list[bool] = []
        for i, request in enumerate(requests_single):
            if step > request.finish_step:
                outputs.append(None)
            elif step == 7 and i == 2:
                outputs.append(None)  # a step without an emitted row
            else:
                outputs.append({"codes": {"audio": torch.randint(0, 1026, (1, 8))}})
            flags.append(step == request.finish_step)
        for request in requests_single + requests_batch:
            request.step = step
        active = [i for i, request in enumerate(requests_single) if step <= request.finish_step]
        want = [
            _payload_items(talker2code2wav_async_chunk(single, outputs[i], requests_single[i], flags[i]))
            for i in active
        ]
        got = talker2code2wav_async_chunk_batch(
            batched,
            [outputs[i] for i in active],
            [requests_batch[i] for i in active],
            [flags[i] for i in active],
        )
        assert [_payload_items(p) for p in got] == want
    assert single.higgs_v3_emitted_frames == batched.higgs_v3_emitted_frames == {}
