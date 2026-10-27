# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

from vllm_omni.model_executor.models.qwen3_omni.first_frame_decoder import Qwen3OmniFirstFrameDecoder
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
    plan_decode_groups,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_Q = 4
_CODEBOOK = 16


def _code2wav(quantizers=_Q) -> Qwen3OmniMoeCode2Wav:
    torch.manual_seed(0)
    config = Qwen3OmniMoeCode2WavConfig(
        codebook_size=_CODEBOOK,
        num_quantizers=quantizers,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        decoder_dim=32,
        sliding_window=8,
    )
    model = Qwen3OmniMoeCode2Wav(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    return model.eval()


def test_first_frame_pcm_equals_code2wav_streaming_chunk0():
    code2wav = _code2wav()
    decoder = Qwen3OmniFirstFrameDecoder(code2wav, sample_rate=24000)
    codes = torch.randint(0, _CODEBOOK, (3, _Q))
    with torch.inference_mode():
        for row in range(codes.shape[0]):
            pcm = decoder.decode(codes[row : row + 1])
            chunk0 = code2wav.chunked_decode_streaming(
                codes[row].reshape(1, _Q, 1), left_context_size=[0], seq_token_counts=[_Q]
            )[0]
            assert pcm.dtype == torch.float32
            assert torch.equal(pcm[0], chunk0.reshape(-1).float())
            assert 0 < pcm.shape[-1] < int(code2wav.total_upsample)
        batched = decoder.decode(codes)
    assert batched.shape == (3, pcm.shape[-1])


def test_first_frame_is_a_prefix_of_a_longer_first_chunk():
    code2wav = _code2wav()
    decoder = Qwen3OmniFirstFrameDecoder(code2wav, sample_rate=24000)
    codes = torch.randint(0, _CODEBOOK, (_Q, 4))
    with torch.inference_mode():
        first = decoder.decode(codes[:, :1].T)[0]
        chunk0 = code2wav.chunked_decode_streaming(
            codes.reshape(1, _Q, 4), left_context_size=[0], seq_token_counts=[4 * _Q]
        )[0]
    torch.testing.assert_close(chunk0.reshape(-1)[: first.numel()].float(), first)


def test_chunk_ramp_adds_exact_code2wav_graph_sizes(monkeypatch):
    import vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper as wrapper_module

    real = wrapper_module.CUDAGraphDecoderWrapper
    wrapper = MagicMock(compute_capture_sizes=real.compute_capture_sizes)
    monkeypatch.setattr(wrapper_module, "CUDAGraphDecoderWrapper", wrapper)
    extra = {"codec_chunk_frames": 25, "codec_left_context_frames": 25, "codec_chunk_ramp": [1, 2, 4, 8, 16, 25]}
    code2wav = SimpleNamespace(config=SimpleNamespace(num_quantizers=_Q))
    code2wav.enable_cudagraph = lambda **kw: Qwen3OmniMoeCode2Wav.enable_cudagraph(
        code2wav, device=torch.device("cuda"), **kw
    )
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.code2wav = code2wav
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(async_chunk=True, enforce_eager=False, stage_connector_config={"extra": extra})
    )
    for batches in ([], [2, 4, 8]):
        extra["codec_graph_batch_sizes"] = batches
        Qwen3OmniMoeForConditionalGeneration._maybe_enable_code2wav_cudagraph(model)
        args = wrapper.call_args.kwargs
        sizes = args["capture_sizes"]
        assert {1, 3, 7, 15, 31, 50} <= set(sizes)
        assert set(real.compute_capture_sizes(codec_chunk_frames=25, codec_left_context_frames=25)) <= set(sizes)
        assert set(args["extra_capture_shapes"]) == {
            (b, size) for b in batches for size in sizes if size <= 50 and b * size <= 325
        }
    assert code2wav._streaming_batch_sizes[50] == [1, 2, 4]
    extra.pop("codec_chunk_ramp")
    extra.pop("codec_graph_batch_sizes")
    Qwen3OmniMoeForConditionalGeneration._maybe_enable_code2wav_cudagraph(model)
    assert wrapper.call_args.kwargs["capture_sizes"] is None


_SIZES = [1, 2, 3, 4, 7, 8, 15, 16, 25, 31, 32, 50, 64]


def _bucket(length):
    return next((size for size in _SIZES if size >= length), None)


def test_decode_groups_keep_a_uniform_batch_whole_and_split_by_the_largest_graph():
    assert plan_decode_groups([50] * 6, _bucket, lambda size: [1, 2, 4, 8]) == [(list(range(6)), 50)]
    calls = plan_decode_groups([25] * 20, _bucket, lambda size: [1, 2, 4, 8, 16])
    assert [len(rows) for rows, _length in calls] == [16, 4]
    calls = plan_decode_groups([1] * 8 + [50] * 8, _bucket, lambda size: [1, 2, 4, 8] if size < 50 else [1, 2, 4])
    assert [(len(rows), length) for rows, length in calls] == [(8, 1), (4, 50), (4, 50)]
    assert plan_decode_groups([], _bucket, lambda size: [1, 2]) == []


@pytest.mark.parametrize("frames", [65, 129])
def test_decode_groups_beyond_largest_graph_use_eager(monkeypatch, frames):
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import CUDAGraphDecoderWrapper

    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.enabled = wrapper._warmed_up = True
    wrapper._bucket_sizes = [8, 32, 64]
    wrapper._compiled_shapes = {(1, 64), (8, 64)}
    wrapper._compiled_graphs = wrapper.graphs = dict.fromkeys(wrapper._compiled_shapes, MagicMock())
    wrapper.decoder = MagicMock(side_effect=lambda codes: codes.float() * 2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    groups = plan_decode_groups([frames, frames], wrapper._get_padded_size, lambda _: (1,))
    assert groups == [([0], frames), ([1], frames)]
    codes = torch.arange(2 * frames).reshape(2, 1, frames)
    for rows, width in groups:
        actual = wrapper.decode(codes[rows, :, :width])
        torch.testing.assert_close(actual, codes[rows].float() * 2, rtol=0, atol=0)
    assert wrapper.decoder.call_count == 2
    for graph in (*wrapper.graphs.values(), *wrapper._compiled_graphs.values()):
        graph.replay.assert_not_called()


class _PaddingGraphs:
    def __init__(self, model, batch_sizes):
        self.model = model
        self.batch_sizes = sorted(batch_sizes)
        self.calls = []

    def _get_padded_size(self, length):
        return _bucket(length)

    def decode(self, codes):
        rows, _q, frames = codes.shape
        size = _bucket(frames)
        batch = next(b for b in self.batch_sizes if b >= rows)
        self.calls.append((rows, frames))
        padded = torch.zeros(batch, codes.shape[1], size, dtype=codes.dtype)
        padded[:rows, :, :frames] = codes
        out = self.model(padded)
        return out[:rows, :, : out.shape[-1] - (size - frames) * int(self.model.total_upsample)].clone()


def test_grouped_streaming_decode_matches_one_padded_batch():
    code2wav = _code2wav()
    lengths = [1, 3, 50, 7, 31]
    lefts = [0, 1, 25, 3, 15]
    codes = torch.zeros(len(lengths), _Q, max(lengths), dtype=torch.long)
    for row, length in enumerate(lengths):
        codes[row, :, :length] = torch.randint(0, _CODEBOOK, (_Q, length))
    counts = [length * _Q for length in lengths]
    with torch.inference_mode():
        reference = code2wav.chunked_decode_streaming(codes, left_context_size=lefts, seq_token_counts=counts)
        graphs = _PaddingGraphs(code2wav, [1, 2, 4])
        code2wav._cudagraph_wrapper, code2wav._cudagraph_enabled = graphs, True
        code2wav._streaming_batch_sizes = {size: [1, 2, 4] for size in _SIZES}
        grouped = code2wav.chunked_decode_streaming(codes, left_context_size=lefts, seq_token_counts=counts)

    for want, got in zip(reference, grouped, strict=True):
        assert want.shape == got.shape
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("frames", "quantizers"), [(1, 16), (4, 16), (4, 4)])
def test_delivered_first_frame_is_removed_from_code2wav_output(frames, quantizers):
    from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY

    codec = _code2wav(quantizers)
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage, model.code2wav, model.code2wav_config = "code2wav", codec, codec.config
    model.generate_audio = lambda codes, left, counts: codec.chunked_decode_streaming(
        codes, left_context_size=left, seq_token_counts=counts
    )
    codes = torch.randint(0, _CODEBOOK, (2, quantizers, frames))
    with torch.inference_mode():
        reference = model.generate_audio(codes, [0, 0], [quantizers * frames] * 2)
        first = Qwen3OmniFirstFrameDecoder(codec, sample_rate=24000).decode(codes[0, :, :1].T)[0]
        out = model.forward(
            input_ids=codes.flatten(),
            positions=None,
            seq_token_counts=[quantizers * frames] * 2,
            runtime_additional_information=[
                {"meta": {"left_context_size": 0, "first_audio": True}},
                {"meta": {"left_context_size": 0, "first_audio": False}},
            ],
        ).multimodal_outputs
    torch.testing.assert_close(torch.cat([first, out["model_outputs"][0].flatten()]), reference[0].flatten())
    torch.testing.assert_close(out["model_outputs"][1], reference[1].reshape(1, -1))
    assert [bool(flag) for flag in out[FIRST_AUDIO_REQUIRED_KEY]] == [True, False]
