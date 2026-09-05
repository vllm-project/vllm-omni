from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_codec import (
    BreezeTTS2MimiCodec,
)
from vllm_omni.model_executor.stage_input_processors.breeze_tts_2 import (
    talker2codec_async_chunk,
)


class _TransferManager:
    def __init__(self, chunk_frames: int = 4):
        self.connector = SimpleNamespace(config={"extra": {"breeze_codec_chunk_frames": chunk_frames}})
        self.code_prompt_token_ids = defaultdict(list)


class _Request:
    def __init__(self, request_id: str, finished: bool = False):
        self.external_req_id = request_id
        self._finished = finished

    def is_finished(self):
        return self._finished


def test_async_processor_sends_only_unemitted_tail_and_flushes_finish():
    manager = _TransferManager(chunk_frames=4)
    request = _Request("req-1")

    for code0 in range(3):
        frame = torch.tensor([code0, code0 + 4, code0 + 8, code0 + 12], dtype=torch.long)
        assert talker2codec_async_chunk(manager, {"codes": {"audio": frame}}, request) is None

    payload = talker2codec_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([3, 7, 11, 15], dtype=torch.long)}},
        request,
    )
    assert payload is not None
    assert payload.codes.audio.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert bool(payload.meta.finished.item()) is False
    assert manager.code_prompt_token_ids["req-1"] == []

    assert talker2codec_async_chunk(manager, {"codes": {"audio": torch.arange(4)}}, request) is None
    payload = talker2codec_async_chunk(manager, None, _Request("req-1", finished=True))
    assert payload.codes.audio.tolist() == [0, 1, 2, 3]
    assert bool(payload.meta.finished.item()) is True
    assert payload.meta.codec_streaming is True


class _Decoder:
    def __init__(self):
        self.calls = []

    def batched_chunked_decode(self, codes, lengths, caches=None, **kwargs):
        self.calls.append((codes.shape, tuple(lengths), tuple(caches), kwargs))
        return [torch.ones(length * 1920, dtype=torch.float32) for length in lengths]


def test_stateful_codec_batches_request_chunks_and_releases_finished_state():
    decoder = _Decoder()
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec._async_chunk = True
    codec._num_codebooks = 4
    codec._codebook_size = 8
    codec._sample_rate = 24_000
    codec._audio_tokenizer = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    codec._decoder_state_cache = {}

    input_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    output = codec.forward(
        input_ids,
        runtime_additional_information=[
            {"meta": {"request_id": "live", "finished": False}},
            {"meta": {"request_id": "done", "finished": True}},
        ],
        seq_token_counts=[4, 4],
        request_ids=["scheduler-live", "scheduler-done"],
    )

    assert decoder.calls[0][0] == (2, 4, 1)
    assert [len(item) for item in output.multimodal_outputs["model_outputs"]] == [1920, 1920]
    assert set(codec._decoder_state_cache) == {"scheduler-live"}

    codec.on_requests_finished(["scheduler-live"])
    assert codec._decoder_state_cache == {}
