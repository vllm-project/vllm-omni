# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def manager(ramp=None):
    cfg = {"initial_codec_chunk_frames": 1, "codec_chunk_frames": 15}
    if ramp is not None:
        cfg["codec_chunk_ramp"] = ramp
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": cfg}),
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        put_req_chunk=defaultdict(int),
    )


@pytest.mark.parametrize("ramp,expected", [(None, [1, 15, 15, 9]), ([1, 2, 4, 8, 15], [1, 2, 4, 8, 15, 10])])
def test_order_and_final_flush(ramp, expected):
    tm = manager(ramp)
    req = SimpleNamespace(external_req_id="a")
    outputs = []
    for i in range(40):
        payload = talker2codec_raw_async_chunk(tm, {"codes": {"audio": torch.full((1, 4), i)}}, req, i == 39)
        if payload is not None:
            outputs.append(payload)
            tm.put_req_chunk["a"] += 1
    assert [x.meta.codec_chunk_frames for x in outputs] == expected
    actual = torch.cat([x.codes.audio.reshape(4, -1).T for x in outputs])
    assert torch.equal(actual, torch.arange(40).unsqueeze(1).expand(-1, 4))
    assert bool(outputs[-1].meta.finished)
    assert "a" not in tm.code_prompt_token_ids


def test_requests_progress_independently_and_empty_finish():
    tm = manager([1, 2, 4, 8, 15])
    for name in ("a", "b"):
        req = SimpleNamespace(external_req_id=name)
        first = talker2codec_raw_async_chunk(tm, {"codes": {"audio": torch.ones(1, 4, dtype=torch.long)}}, req)
        assert first.meta.codec_chunk_frames == 1
        tm.put_req_chunk[name] += 1
    req = SimpleNamespace(external_req_id="a")
    assert talker2codec_raw_async_chunk(tm, {"codes": {"audio": torch.ones(1, 4, dtype=torch.long)}}, req) is None
    final = talker2codec_raw_async_chunk(tm, None, req, True)
    assert final.meta.codec_chunk_frames == 1
    assert bool(final.meta.finished)
    final = talker2codec_raw_async_chunk(tm, None, SimpleNamespace(external_req_id="b"), True)
    assert final.meta.codec_chunk_frames == 0
    assert bool(final.meta.finished)


@pytest.mark.parametrize("ramp", [[1, 0, 15], [1], "invalid"])
def test_invalid_ramp_keeps_original_boundaries(ramp):
    test_order_and_final_flush(ramp, [1, 15, 15, 9])


@pytest.mark.parametrize(
    "ramp,expected,max_step",
    [(None, [1, 15], 15), ([1, 2, 4, 8, 15], [1, 2, 4, 8, 15], 15), ([1, 20, 15], [1, 15, 20], 20)],
)
def test_codec_captures_every_ramp_length(ramp, expected, max_step):
    from transformers import PretrainedConfig
    from vllm.config import CompilationConfig, ModelConfig, SchedulerConfig, VllmConfig

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder

    cfg = Mock(
        spec=VllmConfig,
        model_config=Mock(
            spec=ModelConfig,
            hf_config=PretrainedConfig(),
            async_chunk=True,
            enforce_eager=False,
            stage_connector_config=manager(ramp).connector.config,
        ),
        scheduler_config=Mock(spec=SchedulerConfig, max_num_seqs=8),
        compilation_config=Mock(spec=CompilationConfig, cudagraph_capture_sizes=[1, 2, 4, 8]),
    )
    codec = MossTTSCodecDecoder(vllm_config=cfg)
    assert codec._streaming_graph_frame_sizes == expected
    assert codec._stream_max_step_frames == max_step
