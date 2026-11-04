# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.lru_cache(maxsize=1)
def _fish_speech_regression_modules():
    from vllm_omni.model_executor.models.fish_speech import fish_speech_slow_ar as slow_ar_module
    from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import FishSpeechDACDecoder
    from vllm_omni.model_executor.models.fish_speech.fish_speech_slow_ar import (
        FishSpeechSlowARForConditionalGeneration,
    )

    return slow_ar_module, FishSpeechDACDecoder, FishSpeechSlowARForConditionalGeneration


class _FakeCodec:
    def decode(self, codes_bqf: torch.Tensor, feature_lengths: torch.Tensor):
        del codes_bqf, feature_lengths
        wav = torch.arange(100, dtype=torch.float32).view(1, 1, 100)
        audio_lengths = torch.tensor([100], dtype=torch.long)
        return wav, audio_lengths


class _RecordingCodec:
    def __init__(self):
        self.calls = []

    def decode(self, codes_bqf: torch.Tensor, feature_lengths: torch.Tensor):
        self.calls.append((tuple(codes_bqf.shape), tuple(int(x) for x in feature_lengths.tolist())))
        batch = codes_bqf.shape[0]
        wav_len = int(codes_bqf.shape[-1]) * 10
        wav = torch.arange(batch * wav_len, dtype=torch.float32).view(batch, 1, wav_len)
        audio_lengths = feature_lengths.to(torch.long) * 10
        return wav, audio_lengths


class _FakeOneArgCodec:
    def decode(self, codes_bqf: torch.Tensor):
        batch = codes_bqf.shape[0]
        return torch.arange(batch * 100, dtype=torch.float32).view(batch, 1, 100)


class _FakeFromIndicesCodec:
    def __init__(self):
        self.from_indices_calls = 0

    def from_indices(self, codes_bqf: torch.Tensor):
        self.from_indices_calls += 1
        batch = codes_bqf.shape[0]
        return torch.arange(batch * 100, dtype=torch.float32).view(batch, 1, 100)

    def decode(self, codes_bqf: torch.Tensor):
        del codes_bqf
        raise AssertionError("decode() should not be used for index tensors when from_indices() exists")


class _FakeTokenizer:
    def __init__(self, mapping, unk_token_id=-1):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._mapping.get(token, self.unk_token_id)


def test_dac_decoder_mixed_batch_empty_request_does_not_misalign_indices():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = _FakeCodec()
    decoder._codec_decode_takes_lengths = True
    decoder._num_codebooks = 10
    decoder._output_sample_rate = 44100
    decoder._hop_length = 512
    decoder._logged_codec_stats = False
    decoder._ensure_codec_loaded = lambda: None
    decoder._split_request_ids = lambda ids, seq_token_counts=None: [
        torch.empty((0,), dtype=torch.long),
        torch.arange(20, dtype=torch.long),
    ]

    out = decoder.forward(
        input_ids=torch.arange(20, dtype=torch.long),
        runtime_additional_information=[{}, {"meta": {"left_context_size": 1}}],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert audios[0].numel() == 0
    # 2 total frames with 1 frame of left context => proportional trim removes half the samples.
    assert audios[1].shape[0] == 50


def test_dac_decoder_uses_tensor_codes_from_runtime_info():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = _FakeCodec()
    decoder._codec_decode_takes_lengths = True
    decoder._num_codebooks = 2
    decoder._output_sample_rate = 44100
    decoder._hop_length = 512
    decoder._logged_codec_stats = False
    decoder._ensure_codec_loaded = lambda: None

    out = decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[
            {
                "codes": {"audio": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)},
                "meta": {"left_context_size": 1},
            }
        ],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 1
    # 2 total frames with 1 frame of left context => proportional trim removes half the samples.
    assert audios[0].shape[0] == 50


def test_dac_decoder_supports_codec_decode_without_lengths():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = _FakeOneArgCodec()
    decoder._codec_decode_takes_lengths = False
    decoder._num_codebooks = 10
    decoder._output_sample_rate = 44100
    decoder._hop_length = 512
    decoder._logged_codec_stats = False
    decoder._ensure_codec_loaded = lambda: None
    decoder._split_request_ids = lambda ids, seq_token_counts=None: [
        torch.arange(10, dtype=torch.long),
        torch.arange(20, dtype=torch.long),
    ]

    out = decoder.forward(
        input_ids=torch.arange(30, dtype=torch.long),
        runtime_additional_information=[{}, {}],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert audios[0].shape[0] == 50
    assert audios[1].shape[0] == 100


def test_dac_decoder_prefers_from_indices_for_one_arg_codec():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    codec = _FakeFromIndicesCodec()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = codec
    decoder._codec_decode_takes_lengths = False
    decoder._num_codebooks = 10
    decoder._output_sample_rate = 44100
    decoder._hop_length = 512
    decoder._logged_codec_stats = False
    decoder._ensure_codec_loaded = lambda: None
    decoder._split_request_ids = lambda ids, seq_token_counts=None: [
        torch.arange(10, dtype=torch.long),
        torch.arange(20, dtype=torch.long),
    ]

    out = decoder.forward(
        input_ids=torch.arange(30, dtype=torch.long),
        runtime_additional_information=[{}, {}],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert codec.from_indices_calls == 1
    assert len(audios) == 2
    assert audios[0].shape[0] == 50
    assert audios[1].shape[0] == 100


def test_dac_decoder_splits_groups_by_padded_frame_budget():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    codec = _RecordingCodec()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = codec
    decoder._codec_decode_takes_lengths = True
    decoder._num_codebooks = 2
    decoder._output_sample_rate = 44100
    decoder._hop_length = 512
    decoder._logged_codec_stats = False
    decoder._decode_batch_max_padded_frames = 100
    decoder._decode_batch_max_batch = 0
    decoder._ensure_codec_loaded = lambda: None
    decoder._split_request_ids = lambda ids, seq_token_counts=None: [
        torch.arange(2 * 46, dtype=torch.long),
        torch.arange(2 * 45, dtype=torch.long),
        torch.arange(2 * 47, dtype=torch.long),
    ]

    decoder.forward(input_ids=torch.arange(2 * (46 + 45 + 47), dtype=torch.long))

    assert codec.calls == [
        ((2, 2, 46), (45, 46)),
        ((1, 2, 47), (47,)),
    ]


def test_dac_decoder_dtype_can_be_configured():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="unused",
            stage_connector_config={"extra": {"fish_speech_dac_dtype": "fp16"}},
        )
    )

    decoder = FishSpeechDACDecoder(vllm_config=cfg)

    assert decoder._dac_dtype == torch.float16


def test_dac_codec_graphs_and_history_are_opt_in():
    """Without codec keys (Fish Speech S2 Pro's deploys) the module decoder runs; the
    stateful codec builds on the channels-last graphs, so it needs them."""
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()

    def build(extra):
        cfg = SimpleNamespace(model_config=SimpleNamespace(model="unused", stage_connector_config={"extra": extra}))
        return FishSpeechDACDecoder(vllm_config=cfg)

    default = build({})
    assert not default._cuda_graphs and not default._stateful
    both = build({"fish_speech_dac_cuda_graphs": True, "fish_speech_dac_stateful": True})
    assert both._cuda_graphs and both._stateful
    with pytest.raises(ValueError, match="fish_speech_dac_cuda_graphs"):
        build({"fish_speech_dac_stateful": True})


def test_structured_voice_clone_prefill_adds_full_codebooks_with_decode_scale(monkeypatch):
    slow_ar_module, _, FishSpeechSlowARForConditionalGeneration = _fish_speech_regression_modules()
    model = object.__new__(FishSpeechSlowARForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._num_codebooks = 2
    model._codebook_size = 8
    model._semantic_begin_id = 100
    model.model_path = "unused"

    hidden_size = 3
    text_embed = torch.nn.Embedding(256, hidden_size)
    codebook_embed = torch.nn.Embedding(model._num_codebooks * model._codebook_size, hidden_size)
    with torch.no_grad():
        text_embed.weight.zero_()
        text_embed.weight[20] = torch.tensor([1.0, 2.0, 3.0])
        text_embed.weight[21] = torch.tensor([4.0, 5.0, 6.0])
        codebook_embed.weight.zero_()
        codebook_embed.weight[1] = torch.tensor([10.0, 0.0, 0.0])
        codebook_embed.weight[10] = torch.tensor([0.0, 20.0, 0.0])
        codebook_embed.weight[3] = torch.tensor([30.0, 0.0, 0.0])
        codebook_embed.weight[12] = torch.tensor([0.0, 40.0, 0.0])

    model.embed_input_ids = lambda ids: text_embed(ids)
    model.codebook_embeddings = codebook_embed
    model._get_tokenizer = lambda: _FakeTokenizer({"<|audio_start|>": 10, "<|audio_end|>": 11})

    monkeypatch.setattr(
        slow_ar_module,
        "encode_reference_audio_codes",
        lambda *args, **kwargs: torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
    )
    monkeypatch.setattr(
        slow_ar_module,
        "build_fish_voice_clone_prompt_ids",
        lambda tokenizer, text, ref_text, semantic_token_ids: ([1, 10, 20, 21, 11, 2], None, None),
    )

    prefill = model._build_structured_voice_clone_prefill_embeds(
        {
            "ref_text": "ref",
            "text": "target",
            "ref_audio_wav": torch.tensor([0.0]),
            "ref_audio_sr": 16000,
        }
    )

    expected_0 = (torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([10.0, 20.0, 0.0])) / math.sqrt(3.0)
    expected_1 = (torch.tensor([4.0, 5.0, 6.0]) + torch.tensor([30.0, 40.0, 0.0])) / math.sqrt(3.0)
    assert torch.allclose(prefill[2].to(dtype=torch.float32), expected_0, atol=2e-2, rtol=0)
    assert torch.allclose(prefill[3].to(dtype=torch.float32), expected_1, atol=2e-2, rtol=0)


def _stream_decoder(slots: int):
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._stateful = True
    decoder._stream_slots = {}
    decoder._free_stream_slots = list(range(slots - 1, -1, -1))
    decoder._stream_positions = {}
    return decoder


def test_dac_decoder_stream_slots_follow_request_lifetime():
    decoder = _stream_decoder(2)
    claim = decoder._claim_stream_slots

    # First windows claim slots; a third stream finds the pool empty.
    slots, fresh, positions, release = claim(["a", "b", "c"], [True] * 3, [False] * 3, [6, 6, 6], 4096)
    assert slots == [0, 1, None] and fresh == [True, True, False] and positions == [0, 0, 0] and release == []
    # Later windows continue them; a finished window releases after its decode.
    slots, fresh, positions, release = claim(["a", "b", "c"], [False] * 3, [False, True, False], [8, 8, 8], 4096)
    assert slots == [0, 1, None] and fresh == [False, False, False] and positions == [6, 6, 0] and release == ["b"]
    for request_id in release:
        decoder._release_stream_slot(request_id)
    # A restarted stream gives its slot back and claims one afresh; an
    # aborted one is freed by the runner hook.
    slots, fresh, positions, _ = claim(["a"], [True], [False], [6], 4096)
    assert slots == [0] and fresh == [True] and positions == [0] and decoder._free_stream_slots == [1]
    decoder.on_requests_finished({"a", "unknown"})
    assert decoder._stream_slots == {} and sorted(decoder._free_stream_slots) == [0, 1]
    # No request id: decode without history.
    assert claim([None], [True], [False], [6], 4096)[0] == [None]
    # Real frames past the rotary table: the stream continues without history.
    claim(["d"], [True], [False], [4090], 4096)
    assert claim(["d"], [False], [False], [8], 4096)[0] == [None] and "d" not in decoder._stream_slots


@pytest.mark.parametrize("left_context", [0, 32])
def test_dac_stream_history_follows_the_drama3_chunker(left_context):
    """The real Drama 3 chunker into the codec's slot management: one slot per stream, fresh
    only at its first window, positions counting new frames -- also without left context."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.fish_omni import (
        SEMANTIC_BEGIN_ID,
        fish_omni_to_dac_decoder_async_chunk,
    )

    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"codec_left_context_frames": left_context}}),
        request_payload={},
    )
    decoder = _stream_decoder(4)

    def frame(index):
        row = torch.cat((torch.tensor([SEMANTIC_BEGIN_ID + index % 4096]), torch.arange(10)))
        return {"codes.audio": row.unsqueeze(0)}

    def run_stream(request_id, frames):
        request = SimpleNamespace(external_req_id=request_id, is_finished=lambda: False)
        claimed = []
        for index in range(frames):
            payload = fish_omni_to_dac_decoder_async_chunk(
                manager, frame(index), request, is_finished=index == frames - 1
            )
            if payload is None or payload.codes.audio.numel() == 0:
                continue
            meta = {"left_context_size": payload.meta.left_context_size, "chunk_seq": payload.meta.chunk_seq}
            context = int(meta["left_context_size"])
            new = int(payload.codes.audio.shape[1]) - context
            first = FishSpeechDACDecoder._meta_first_window([{"meta": meta}], 0, context)
            done = bool(payload.meta.finished)
            slots, fresh, positions, release = decoder._claim_stream_slots([request_id], [first], [done], [new], 4096)
            claimed.append((slots[0], fresh[0], positions[0], new))
            for owner in release:
                decoder._release_stream_slot(owner)
        return claimed

    windows = run_stream("r", 300)
    assert len(windows) > 3
    assert {slot for slot, *_ in windows} == {windows[0][0]} and windows[0][0] is not None
    assert [fresh for _, fresh, _, _ in windows] == [True] + [False] * (len(windows) - 1)
    starts = [position for _, _, position, _ in windows]
    assert starts == [sum(new for *_, new in windows[:i]) for i in range(len(windows))]
    assert sum(new for *_, new in windows) == 300
    # The stream's last window released its slot; the same id restarts afresh.
    assert "r" not in decoder._stream_slots
    manager.request_payload.clear()
    restarted = run_stream("r", 20)
    assert restarted[0][1] is True and restarted[0][2] == 0
