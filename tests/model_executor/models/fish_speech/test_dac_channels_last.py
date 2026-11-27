# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Channels-last DAC decoding matches the module decoder (CUDA, random weights)."""

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)]

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _codec(dtype):
    from vllm_omni.model_executor.models.fish_speech.dac_utils import bake_weight_norm, build_dac_codec

    torch.manual_seed(0)
    codec = build_dac_codec()
    bake_weight_norm(codec)
    codec.encoder = None
    return codec.to("cuda", dtype).eval()


def _codes(batch, frames):
    return torch.cat(
        (torch.randint(0, 4096, (batch, 1, frames)), torch.randint(0, 1024, (batch, 9, frames))), dim=1
    ).cuda()


def _full_transformer(decoder):
    """Random weights leave the window transformer at 1% of the output (layer scale 1e-2),
    which hides attention errors: give its layers full weight."""
    for layer in decoder.transformer.layers:
        layer.attention_layer_scale.gamma.data.fill_(1.0)
        layer.ffn_layer_scale.gamma.data.fill_(1.0)


def _snr(actual, expected):
    return 10 * torch.log10(expected.pow(2).mean() / (actual - expected).pow(2).mean().clamp_min(1e-20))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
# 140 frames exceed the 128-frame attention window (masked path).
@pytest.mark.parametrize("frames", [21, 140])
def test_channels_last_decoder_matches_module_decoder(dtype, frames):
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder

    codec = _codec(dtype)
    codes = _codes(2, frames)
    with torch.inference_mode():
        expected = codec.decode(codes, torch.full((2,), frames, device="cuda"))[0].float()
        actual = ChannelsLastDACDecoder(codec)(codes).float()
    assert actual.shape == expected.shape
    snr = _snr(actual, expected)
    assert snr > 40, snr


@cuda
def test_dac_decode_graphs_match_eager_with_padding():
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder, DACDecodeGraphs

    codec = _codec(torch.float16)
    decoder = ChannelsLastDACDecoder(codec)
    graphs = DACDecodeGraphs(decoder, 10, [16, 32], max_padded_frames=96, max_batch=4)
    graphs.capture(torch.device("cuda"))
    assert graphs.batch_buckets == {16: [1, 2, 4], 32: [1, 2, 3]}
    # Fill the static inputs with other codes first: padding keeps stale codes.
    graphs(_codes(4, 16))
    for batch, frames in ((3, 13), (1, 32), (2, 16), (3, 30)):
        codes = _codes(batch, frames)
        with torch.inference_mode():
            expected = decoder(codes).float()
        actual = graphs(codes)
        assert actual.dtype == torch.float32 and actual.shape == expected.shape
        snr = _snr(actual, expected)
        assert snr > 50, (batch, frames, snr)
    # Past the buckets: eager.
    codes = _codes(1, 40)
    assert graphs(codes).shape == (1, 1, 40 * decoder.hop_length)
    groups = graphs.plan([13, 30, 16, 40, 5, 9, 2])
    assert all(trim == 0 for _, trim in groups)
    assert sorted(sorted(group) for group, _ in groups) == [[0, 2, 4, 5], [1], [3], [6]]


@cuda
def test_rows_with_fewer_frames_share_a_bucket_graph():
    """Windows of 13, 16 and 9 frames replay one 16-frame graph, padded at the end: each
    row's real frames decode as that window alone (causal convs and attention)."""
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder, DACDecodeGraphs

    decoder = ChannelsLastDACDecoder(_codec(torch.float16))
    _full_transformer(decoder)
    graphs = DACDecodeGraphs(decoder, 10, [16], max_padded_frames=64, max_batch=4)
    graphs.capture(torch.device("cuda"))
    torch.manual_seed(3)
    lengths = (13, 16, 9)
    windows = [_codes(1, n) for n in lengths]
    batch = torch.zeros((len(lengths), 10, 16), dtype=torch.long, device="cuda")
    for row, codes in enumerate(windows):
        batch[row, :, : codes.shape[-1]] = codes[0]
    with torch.inference_mode():
        expected = [decoder(codes)[0, 0].float() for codes in windows]
    audio = graphs(batch)
    assert audio.shape[-1] == 16 * decoder.hop_length
    for row, (frames, ref) in enumerate(zip(lengths, expected, strict=True)):
        got = audio[row, 0, : frames * decoder.hop_length]
        assert _snr(got, ref) > 50, (frames, _snr(got, ref))


@cuda
def test_conv_context_trim_keeps_new_frames_exact():
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import (
        ChannelsLastDACDecoder,
        DACDecodeGraphs,
        streaming_decode_shapes,
    )

    decoder = ChannelsLastDACDecoder(_codec(torch.float32))
    # Upsample ConvNeXts (3 + 1.5 frames), the first conv (1.5) and the residual
    # units of the 688 Hz block (2.4) dominate the conv stack's ~9.1 frames.
    receptive = decoder.receptive_frames
    assert receptive == 10
    buckets, trims = streaming_decode_shapes((10, 10, 12, 32), 64, 32, receptive)
    assert trims == {8: {0}, 10: {0}, 16: {0}, 20: {0}, 24: {10}, 32: {10}, 48: {22}, 64: {22}, 80: {22}, 96: {22}}
    hop = decoder.hop_length
    codes = _codes(2, 60)  # 28 new frames after 32 of context
    trim = 32 - receptive
    with torch.inference_mode():
        full = decoder(codes).float()
        trimmed = decoder(codes, trim).float()
    assert trimmed.shape[-1] == (60 - trim) * hop
    # Past the receptive field the trimmed decode reproduces the full one.
    snr = _snr(trimmed[..., receptive * hop :], full[..., (trim + receptive) * hop :])
    assert snr > 60, snr
    graphs = DACDecodeGraphs(decoder, 10, [64], max_padded_frames=128, max_batch=2, trims={64: {trim}})
    graphs.capture(torch.device("cuda"))
    assert graphs.plan([60, 60], [32, 32]) == [([0, 1], trim)]
    assert graphs.plan([60], [0]) == [([0], 0)]  # no captured (64, 0) shape: eager
    assert _snr(graphs(codes, trim), trimmed) > 60


@cuda
def test_compiled_transformer_decode_matches_eager_in_graphs():
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder, DACDecodeGraphs

    codec = _codec(torch.float16)
    eager = ChannelsLastDACDecoder(codec)
    compiled = ChannelsLastDACDecoder(codec)
    compiled.compile_transformer()
    # One dynamic graph serves every captured (batch, frames, trim), batch 1 included.
    graphs = DACDecodeGraphs(compiled, 10, [14, 42], max_padded_frames=168, max_batch=4, trims={42: [0, 16]})
    graphs.capture(torch.device("cuda"))
    for batch, frames, trim in ((1, 14, 0), (3, 14, 0), (4, 42, 16), (2, 40, 16), (1, 42, 0)):
        codes = _codes(batch, frames)
        with torch.inference_mode():
            expected = eager(codes, trim).float()
        actual = graphs(codes, trim)
        assert actual.shape == expected.shape
        snr = _snr(actual, expected)
        assert snr > 35, (batch, frames, trim, snr)


def test_graph_plan_splits_requests_into_exact_batch_buckets():
    import types

    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import DACDecodeGraphs

    graphs = DACDecodeGraphs(
        types.SimpleNamespace(), 10, [16], max_padded_frames=2048, max_batch=16, receptive_frames=10
    )
    assert graphs.batch_buckets[16] == [1, 2, 4, 8, 12, 16]
    # Replay costs as capture measures them: here one padded row each.
    graphs.cost_ms = {(b, 16, 0): b + 1.0 for b in graphs.batch_buckets[16]}
    assert sorted(graphs.split(16, 5)) == [1, 4]
    assert graphs.split(16, 7) == [8]
    assert sorted(graphs.split(16, 13)) == [1, 12]
    assert graphs.split(16, 12) == [12]
    for count in range(1, 17):
        assert sum(graphs.split(16, count)) >= count
    groups = graphs.plan([10] * 5)
    assert sorted(len(group) for group, _ in groups) == [1, 4]


def test_graph_plan_uses_measured_replay_costs():
    import types

    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import DACDecodeGraphs

    graphs = DACDecodeGraphs(types.SimpleNamespace(), 10, [16], max_padded_frames=2048, max_batch=8, receptive_frames=0)
    sizes = graphs.batch_buckets[16]
    assert sizes == [1, 2, 4, 8]
    # A large fixed part per replay: pad 5 up to the batch-8 graph.
    graphs.cost_ms = {(b, 16, 0): 3.0 + 0.25 * b for b in sizes}
    assert graphs.split(16, 5, 0) == [8]
    # A small fixed part: 4 + 1 beats padding three rows.
    graphs.cost_ms.update({(b, 16, 4): 0.2 + 0.5 * b for b in sizes})
    assert sorted(graphs.split(16, 5, 4)) == [1, 4]


def _lockstep_stream_new_frames(decode, codes_list, slots, ramp=(6, 8, 12), chunk=16):
    """Decode requests window by window in one batch per step: each window carries its new
    frames only, with the stream rows (slot, fresh, new frames, position)."""
    outs: list[list[torch.Tensor]] = [[] for _ in codes_list]
    emitted = [0] * len(codes_list)
    step = 0
    while any(e < c.shape[-1] for e, c in zip(emitted, codes_list)):
        size = ramp[step] if step < len(ramp) else chunk
        active = [i for i, c in enumerate(codes_list) if emitted[i] < c.shape[-1]]
        news = [min(size, codes_list[i].shape[-1] - emitted[i]) for i in active]
        width = max(news)
        batch = torch.zeros(len(active), codes_list[0].shape[1], width, dtype=torch.long, device="cuda")
        for row, i in enumerate(active):
            batch[row, :, : news[row]] = codes_list[i][0, :, emitted[i] : emitted[i] + news[row]]
        rows = torch.tensor(
            [[slots[i] for i in active], [int(emitted[i] == 0) for i in active], news, [emitted[i] for i in active]],
            dtype=torch.long,
            device="cuda",
        )
        audio = decode(batch, rows)
        hop = audio.shape[-1] // width
        for row, i in enumerate(active):
            outs[i].append(audio[row, 0, : news[row] * hop].float())
            emitted[i] += news[row]
        step += 1
    return [torch.cat(o) for o in outs]


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_stream_with_transformer_kv_matches_whole_utterance(dtype):
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import (
        ChannelsLastDACDecoder,
        DACDecodeGraphs,
        DACStreamState,
        streaming_decode_shapes,
    )

    codec = _codec(dtype)
    decoder = ChannelsLastDACDecoder(codec)
    assert decoder.transformer.window_size == 128
    _full_transformer(decoder)
    torch.manual_seed(5)
    # 200 frames run past the transformer window, so the key ring wraps.
    codes_list = [_codes(1, 200), _codes(1, 45)]
    with torch.inference_mode():
        expected = [decoder(c)[0, 0].float() for c in codes_list]
    floor = 60 if dtype == torch.float32 else 30
    stream = DACStreamState(4)

    def direct(batch, rows):
        stream.bind(rows, batch.shape[-1])
        decoder.stream = stream
        try:
            with torch.inference_mode():
                return decoder(batch, 0)
        finally:
            decoder.stream = None
            stream.unbind()

    for got, ref in zip(_lockstep_stream_new_frames(direct, codes_list, slots=[2, 0]), expected):
        assert got.shape == ref.shape
        assert _snr(got, ref) > floor, _snr(got, ref)

    ramp, chunk = (6, 8, 12), 16
    buckets, trims = streaming_decode_shapes(ramp, chunk, 0, receptive_frames=0)
    graphs = DACDecodeGraphs(
        decoder,
        10,
        buckets,
        max_padded_frames=256,
        max_batch=4,
        trims=trims,
        receptive_frames=0,
        stream=DACStreamState(4),
    )
    graphs.capture(torch.device("cuda"))
    assert graphs.stream.kv_tensors
    for slots in ([1, 3], [3, 1]):
        streamed = _lockstep_stream_new_frames(lambda b, r: graphs(b, 0, r), codes_list, slots=slots)
        for got, ref in zip(streamed, expected):
            assert _snr(got, ref) > floor, _snr(got, ref)


def _direct_stream(decoder, stream):
    def direct(batch, rows):
        stream.bind(rows, batch.shape[-1])
        decoder.stream = stream
        try:
            with torch.inference_mode():
                return decoder(batch, 0)
        finally:
            decoder.stream = None
            stream.unbind()

    return direct


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])  # gather path / slot-ring path
def test_stream_with_transformer_kv_windows_longer_than_the_attention_window(dtype):
    """A 192-frame window (past the 128-frame attention window) still matches the whole utterance."""
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder, DACStreamState

    decoder = ChannelsLastDACDecoder(_codec(dtype))
    _full_transformer(decoder)
    torch.manual_seed(6)
    codes = _codes(1, 400)
    with torch.inference_mode():
        expected = decoder(codes)[0, 0].float()
    stream = DACStreamState(2, max_window_frames=192)
    (got,) = _lockstep_stream_new_frames(_direct_stream(decoder, stream), [codes], slots=[0], ramp=(6,), chunk=192)
    floor = 60 if dtype == torch.float32 else 30
    assert _snr(got, expected) > floor, _snr(got, expected)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_stream_padding_rows_near_the_rotary_table_end(dtype):
    """A row whose real frames end at the rotary table's last row, padded to a wider batch:
    its padding frames must not index past the table, and the other row is unaffected."""
    from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder, DACStreamState

    decoder = ChannelsLastDACDecoder(_codec(dtype))
    limit = int(decoder.transformer.freqs_cis.shape[0])
    torch.manual_seed(7)
    late, fresh = _codes(1, 30), _codes(1, 32)
    batch = torch.zeros(2, 10, 32, dtype=torch.long, device="cuda")
    batch[0, :, :30] = late[0]
    batch[1] = fresh[0]
    rows = torch.tensor([[0, 1], [0, 1], [30, 32], [limit - 30, 0]], dtype=torch.long, device="cuda")
    stream = DACStreamState(2)
    audio = _direct_stream(decoder, stream)(batch, rows)
    torch.accelerator.synchronize()
    assert torch.isfinite(audio).all()
    alone = _direct_stream(decoder, DACStreamState(2))(fresh, rows[:, 1:].clone())
    floor = 60 if dtype == torch.float32 else 30
    assert _snr(audio[1, 0].float(), alone[0, 0].float()) > floor
