# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free talker EOS -> raw codec terminal contract.

Execute the production method ASTs with a deterministic local-code stub and
plain payload records. The scheduler's EOS decision is simulated; this is not
a full scheduler, connector transport, socket, or GPU-engine integration test.
"""

from collections import defaultdict
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .test_realtime_batched_gpu_history import bare, load_functions, run

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LOGITS_NS = load_functions("modeling_moss_tts_talker.py", {"compute_logits"}, "MossTTSRealtimeTalkerForGeneration")
LOGITS_NS.update(load_functions("modeling_moss_tts_talker.py", {"_iter_state_row_spans"}))
COMPUTE_LOGITS = LOGITS_NS["compute_logits"]

RAW_NS = load_functions(
    "../../stage_input_processors/moss_tts.py",
    {
        "_raw_request_id",
        "_init_raw_chunk_state",
        "_record_raw_first_audio",
        "_raw_new_frames",
        "_raw_frames_np",
        "_raw_chunk_config",
        "_raw_emit_chunk",
        "talker2codec_raw_async_chunk",
    },
)
RAW_NS.update(
    defaultdict=defaultdict,
    Mapping=Mapping,
    np=np,
    CodesStruct=SimpleNamespace,
    MetaStruct=SimpleNamespace,
    OmniPayloadStruct=SimpleNamespace,
    _MOSS_AUDIO_PAD_CODE=1024,
)
RAW_NS.update(load_functions("../../stage_input_processors/chunk_size_utils.py", {"parse_chunk_ramp"}))
PROCESS = RAW_NS["talker2codec_raw_async_chunk"]
BUILD_PAYLOAD = load_functions(
    "../../../distributed/omni_connectors/model_runner/omni_connector_payload_transport.py",
    {"_build_custom_process_payload"},
    "_OmniConnectorPayloadTransportMixin",
)["_build_custom_process_payload"]


@pytest.mark.parametrize("audio_frames", [0, 2, 3], ids=["no-audio", "pending-tail", "exact-chunk-boundary"])
@pytest.mark.parametrize("mixed_batch", [False, True], ids=["all-stopped", "other-request-still-active"])
def test_realtime_eos_flushes_raw_codec_without_control_frame(audio_frames, mixed_batch):
    talker, _ = bare()
    talker.text_vocab_size = 8
    talker.text_pad_id = 2
    talker.audio_eos_id = 3
    infos = [{} for _ in range(2 if mixed_batch else 1)]
    request = SimpleNamespace(request_id="r", external_req_id="r", finished=False)
    request.is_finished = lambda: request.finished
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 3}}),
        _custom_process_func=PROCESS,
        _custom_process_payload_kwarg="multimodal_output",
        _custom_process_supports_is_finished=True,
    )
    packets = []
    for step in range(audio_frames + 1):
        is_eos_step = step == audio_frames
        code = talker.AUDIO_EOS if is_eos_step else step + 1
        codes = [code, 100 + step] if mixed_batch else [code]
        output = run(talker, torch.tensor(codes, dtype=torch.float32).reshape(-1, 1), infos)
        logits = COMPUTE_LOGITS(talker, output)
        # Model the scheduler's stop-token decision, rather than passing a
        # hand-written finished=True directly to the chunk processor.
        request.finished = int(logits[0].argmax()) == talker.audio_eos_id
        assert request.finished is is_eos_step
        if mixed_batch:
            assert int(logits[1].argmax()) == talker.text_pad_id
        rows = output.multimodal_outputs.get("codes", {}).get("audio")
        payload = {"codes": {"audio": rows[0]}} if rows is not None else {}
        if is_eos_step:
            assert rows is None or rows[0].numel() == 0
        packet = BUILD_PAYLOAD(manager, "r", request, payload, propagate_errors=True)
        if packet is not None:
            packets.append(packet)
            manager.put_req_chunk["r"] += 1
            manager.ramp_chunk_count["r"] += 1

    terminal = packets[-1]
    assert bool(terminal.meta.finished) and bool(terminal.meta.stream_finished)
    assert all(not bool(packet.meta.finished) for packet in packets[:-1])
    assert terminal.meta.codec_chunk_frames == audio_frames % 3
    if audio_frames % 3 == 0:
        assert terminal.codes.audio.numel() == 0
    emitted = torch.cat([packet.codes.audio.reshape(4, -1).T for packet in packets])
    expected = torch.arange(1, audio_frames + 1).reshape(-1, 1).expand(-1, 4)
    torch.testing.assert_close(emitted, expected)
    assert not torch.any(emitted == talker.AUDIO_EOS)
    assert "r" not in manager.code_prompt_token_ids
    assert "r" not in manager.request_payload
