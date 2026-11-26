# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
from vllm_omni.model_executor.models.qwen3_tts.streaming_code2wav import StreamingCode2Wav
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import StreamingCodecDecoder

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_streaming_graphs_preserve_ring_state_and_owned_pcm():
    decoder = _make_decoder().to(device="cuda", dtype=torch.bfloat16)
    # Real tokenizer configs declare head_dim; the small CPU fixture omits it.
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    backend = StreamingCode2Wav(
        decoder, num_slots=2, max_batch_size=1, batch_sizes=[1], frame_sizes=[1, 25], capture=True
    )
    oracle = StreamingCodecDecoder(decoder, 2)
    caches = [{"skip_first_audio": True}, {}]
    positions = [0, 0]
    kept = None
    for ordinal, length in enumerate([1, 25, 25, 25, 25, 25, 25, 7]):
        codes = torch.randint(0, 32, (2, 2, length), device="cuda")
        terminal = ordinal == 7
        actual = backend.decode(
            codes,
            [length, length],
            request_ids=["0", "1"],
            caches=caches,
            terminal=[terminal, terminal],
            legacy_decoder=decoder,
            chunk_size=300,
            left_context_size=25,
        )
        for row in range(2):
            value = codes[row : row + 1].transpose(1, 2).to(torch.int32).contiguous()
            if terminal:
                value = torch.nn.functional.pad(value, (0, 0, 0, 25 - length))
            expected = oracle(
                value,
                torch.tensor([row], device="cuda", dtype=torch.int32),
                torch.tensor([positions[row]], device="cuda", dtype=torch.int32),
            )
            skip = int(row == 0 and positions[row] == 0)
            expected = expected[:, skip * oracle.spf : length * oracle.spf]
            torch.testing.assert_close(actual[row], expected, rtol=1e-5, atol=1e-6)
            positions[row] += length
        if ordinal == 1:
            kept = (actual[0], actual[0].clone())
    assert kept is not None
    torch.testing.assert_close(*kept, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_icl_graph_matches_eager_across_reordered_chunks():
    decoder = _make_decoder().to(device="cuda", dtype=torch.bfloat16)
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    for module in decoder.modules():
        if hasattr(module, "embedding_sum") and hasattr(module, "cluster_usage"):
            module.embedding_sum.uniform_(-1, 1)
            module.cluster_usage.fill_(1)
    graphed = StreamingCode2Wav(
        decoder, num_slots=2, max_batch_size=2, batch_sizes=[1, 2], frame_sizes=[1, 25], capture=True
    )
    eager = StreamingCode2Wav(
        decoder, num_slots=2, max_batch_size=2, batch_sizes=[1, 2], frame_sizes=[1, 25], capture=False
    )
    eager.graphs = {frames: eager.decoder for frames in graphed.graphs}
    torch.manual_seed(23)
    prefix_frames = 12
    lengths = (1, 25, 25, 25, 25)
    codes = torch.randint(
        0,
        decoder.config.codebook_size,
        (2, decoder.config.num_quantizers, prefix_frames + sum(lengths)),
        device="cuda",
    )
    graph_caches = [{"prefix_frames": prefix_frames} for _ in range(2)]
    eager_caches = [{"prefix_frames": prefix_frames} for _ in range(2)]
    cursor = prefix_frames
    for ordinal, length in enumerate(lengths):
        order = (0, 1) if ordinal % 2 == 0 else (1, 0)
        chunk = (
            codes[list(order), :, : cursor + length]
            if ordinal == 0
            else codes[list(order), :, cursor : cursor + length]
        )
        kwargs = {"terminal": [False, False], "legacy_decoder": decoder, "chunk_size": 300, "left_context_size": 25}
        graph_wavs = graphed.decode(
            chunk,
            [chunk.shape[-1]] * 2,
            request_ids=[str(index) for index in order],
            caches=[graph_caches[index] for index in order],
            **kwargs,
        )
        eager_wavs = eager.decode(
            chunk,
            [chunk.shape[-1]] * 2,
            request_ids=[str(index) for index in order],
            caches=[eager_caches[index] for index in order],
            **kwargs,
        )
        for graph_wav, eager_wav in zip(graph_wavs, eager_wavs, strict=True):
            torch.testing.assert_close(graph_wav, eager_wav, rtol=0, atol=0)
            assert graph_wav.abs().max() > 0
        cursor += length


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_initial_full_window_matches_split_priming_and_continuation():
    decoder = _make_decoder().to(device="cuda", dtype=torch.bfloat16)
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    for module in decoder.modules():
        if hasattr(module, "embedding_sum") and hasattr(module, "cluster_usage"):
            module.embedding_sum.uniform_(-1, 1)
            module.cluster_usage.fill_(1)
    # Exercise the real default window, including a prefix larger than max_frames.
    decoder.config.sliding_window = 72
    whole = StreamingCodecDecoder(decoder, 1)
    split = StreamingCodecDecoder(decoder, 1)
    codes = torch.randint(0, decoder.config.codebook_size, (1, 73, whole.nq), device="cuda", dtype=torch.int32)
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    pos = torch.zeros_like(slots)
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import StreamingDecodeGraphs

    graph = StreamingDecodeGraphs(whole, [1], frames=72, initial=True)
    actual = graph(codes[:, :72], slots, pos).clone()
    pieces = []
    for start in range(0, 72, split.max_frames):
        pieces.append(split(codes[:, start : min(start + split.max_frames, 72)], slots, pos + start))
    torch.testing.assert_close(actual, torch.cat(pieces, dim=1), rtol=0.02, atol=0.02)
    torch.testing.assert_close(
        whole(codes[:, 72:], slots, pos + 72), split(codes[:, 72:], slots, pos + 72), rtol=0.02, atol=0.02
    )
