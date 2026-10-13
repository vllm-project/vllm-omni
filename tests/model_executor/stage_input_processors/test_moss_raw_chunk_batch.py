# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The batched MOSS raw-codec processor returns exactly the scalar results."""

from __future__ import annotations

import random
from collections import defaultdict
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    talker2codec_raw_async_chunk,
    talker2codec_raw_async_chunk_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NQ = 12
_PAD = 1024


def _tm(extra):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _plain(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", value.dtype, tuple(value.shape), value.tolist())
    if isinstance(value, msgspec.Struct):
        return (type(value).__name__, {f: _plain(getattr(value, f)) for f in value.__struct_fields__})
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _output(rng, host):
    kind = rng.random()
    if kind < 0.08:
        return None
    if kind < 0.14:
        return {"codes": {"audio": torch.empty((0, _NQ), dtype=torch.long)}}
    rows = 2 if kind < 0.2 else 1
    frames = torch.randint(0, 1024, (rows, _NQ))
    for row in range(rows):
        if rng.random() < 0.15:
            frames[row] = _PAD
    if kind < 0.24:
        frames = frames[0]  # 1-D frame
    # Like the runner's request snapshot: views of one per-step host tensor.
    host.append(frames)
    return {"codes": {"audio": frames}}


def _publish(tm, req_id, payload):
    # The connector advances these counters when it enqueues a payload.
    if payload is not None:
        tm.put_req_chunk[req_id] += 1
        tm.ramp_chunk_count[req_id] += 1


@pytest.mark.parametrize(
    "extra",
    [
        {"codec_chunk_frames": 15},
        {"codec_chunk_frames": 15, "initial_codec_chunk_frames": 1},
        {"codec_chunk_frames": 4, "initial_codec_chunk_frames": 2},
        {"codec_chunk_frames": 15, "codec_chunk_ramp": [1, 4, 8]},
    ],
)
@pytest.mark.parametrize("direct_first", [False, True])
def test_batch_matches_scalar(extra, direct_first):
    rng = random.Random(0)
    torch.manual_seed(0)
    scalar_tm, batch_tm = _tm(dict(extra)), _tm(dict(extra))
    live = [f"r{i}" for i in range(12)]
    next_id = 12
    for _ in range(120):
        host: list[torch.Tensor] = []
        requests = [SimpleNamespace(external_req_id=req_id, request_id=f"internal-{req_id}") for req_id in live]
        outputs = [_output(rng, host) for _ in live]
        if direct_first:
            for output in outputs:
                if output is not None and rng.random() < 0.15:
                    output["meta"] = {"first_audio": torch.tensor(True)}
        finished = [rng.random() < 0.04 for _ in live]

        expected = [
            talker2codec_raw_async_chunk(scalar_tm, output, request, is_finished=done)
            for output, request, done in zip(outputs, requests, finished)
        ]
        got = talker2codec_raw_async_chunk_batch(
            transfer_manager=batch_tm, pooling_outputs=outputs, requests=requests, is_finished=finished
        )
        assert [_plain(p) for p in got] == [_plain(p) for p in expected]
        for req_id, a, b in zip(live, expected, got):
            _publish(scalar_tm, req_id, a)
            _publish(batch_tm, req_id, b)
        # Rows handed to the processor may be reused by the producer afterwards.
        for frames in host:
            frames.fill_(7)
        assert {k: [r.tolist() for r in v] for k, v in batch_tm.code_prompt_token_ids.items()} == {
            k: [r.tolist() for r in v] for k, v in scalar_tm.code_prompt_token_ids.items()
        }
        live = [req_id for req_id, done in zip(live, finished) if not done]
        while len(live) < 12:
            live.append(f"r{next_id}")
            next_id += 1


def test_batch_rejects_three_dimensional_frames():
    tm = _tm({"codec_chunk_frames": 15})
    request = SimpleNamespace(external_req_id="r0")
    with pytest.raises(ValueError, match="must be 2-D"):
        talker2codec_raw_async_chunk_batch(
            transfer_manager=tm,
            pooling_outputs=[{"codes": {"audio": torch.ones((1, 1, _NQ), dtype=torch.long)}}],
            requests=[request],
            is_finished=[False],
        )
