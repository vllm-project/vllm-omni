# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame conservation, delayed emission and owned CPU rows across interleaved requests."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("nq", [12, 32])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_owned_strided_rows_conserve_frames(nq, dtype, defer, direct):
    manager = SimpleNamespace(
        connector=SimpleNamespace(
            config={
                "codec_chunk_frames": 15,
                "initial_codec_chunk_frames": 1,
                "moss_defer_codec_prime": defer,
            }
        )
    )
    expected: dict[str, list[list[int]]] = {rid: [] for rid in ("a", "b")}
    received: dict[str, list[list[int]]] = {rid: [] for rid in expected}
    sizes: dict[str, list[int]] = {rid: [] for rid in expected}

    def collect(rid, payload):
        if payload is None:
            return
        frames = payload.meta.codec_chunk_frames
        if frames:
            received[rid].extend(payload.codes.audio.reshape(nq, frames).T.tolist())
            sizes[rid].append(frames)
            manager.put_req_chunk[rid] += 1
        assert bool(payload.meta.first_audio) == direct

    for step in range(34):
        for rid in expected:
            row = torch.arange(nq * 2, dtype=dtype)[::2].reshape(1, nq)
            row.add_(step)
            if step % 9 == 0:
                row.fill_(1024)
            else:
                expected[rid].extend(row.long().tolist())
            output = {"codes": {"audio": row}, "meta": {"first_audio": torch.tensor(direct)}}
            payload = talker2codec_raw_async_chunk(manager, output, SimpleNamespace(request_id=rid))
            row.fill_(0)  # The caller is allowed to recycle its D2H allocation.
            collect(rid, payload)
    for rid in expected:
        payload = talker2codec_raw_async_chunk(manager, None, SimpleNamespace(request_id=rid), True)
        collect(rid, payload)
        assert bool(payload.meta.finished)
        assert received[rid] == expected[rid]
        assert sizes[rid][0] == (15 if direct and defer else 1)
    assert not manager.request_payload and not manager.code_prompt_token_ids
