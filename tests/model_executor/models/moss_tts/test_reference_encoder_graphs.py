# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-encoder CUDA graphs reproduce the tokenizer's eager encode."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.moss_tts.reference_encoder_graphs import (
    MossReferenceEncoderGraphs,
    _cached_reference_rope,
    _reference_rope_factors,
    _validate_unmasked_reference_encoder,
    _windowed_attention,
    install_windowed_attention,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

FRAME, N_VQ = 8, 3


class _Stage(nn.Module):
    """Causal per-frame stage: frames never mix, like a causal encoder layer."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x, lengths):
        return torch.tanh(self.proj(x)), lengths


class _Quantizer(nn.Module):
    def forward(self, hidden, lengths, n_q):
        # hidden (B, frames, D): codes (n_q, B, frames) from per-frame argmax slices.
        codes = torch.stack([hidden[..., q::n_q].argmax(-1) for q in range(n_q)])
        return hidden, codes, lengths


class _Tokenizer(nn.Module):
    number_channels = 2
    downsample_rate = FRAME
    sampling_rate = 64

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([_Stage(FRAME * 2, 16), _Stage(16, 12)])
        self.quantizer = _Quantizer()

    def _flatten_channels_for_codec(self, audio, lengths):
        batch, channels, samples = audio.shape
        frames = samples // FRAME
        values = audio.reshape(batch, channels, frames, FRAME).permute(0, 2, 1, 3).reshape(batch, frames, -1)
        return values, lengths // FRAME

    def _codec_inference_autocast(self):
        return nullcontext()

    def eager(self, wavs):
        out = []
        for wav in wavs:
            values, lengths = self._flatten_channels_for_codec(wav[None], torch.tensor([wav.shape[-1]]))
            for module in self.encoder:
                values, lengths = module(values, lengths)
            _, codes, _ = self.quantizer(values, lengths, N_VQ)
            out.append(codes[:, 0, : wav.shape[-1] // FRAME].transpose(0, 1).long())
        return out


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda", torch.accelerator.current_device_index())


def test_cached_rope_reuses_longest_table_without_modifying_inputs(cuda):
    from types import SimpleNamespace

    module = SimpleNamespace(rope=SimpleNamespace(max_period=10000.0))
    cache: dict[tuple[int, float, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
    torch.manual_seed(17)
    # Prime once; all subsequent sequences and batches use an immutable prefix.
    q = torch.randn(1, 3, 128, 32, device=cuda, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    _cached_reference_rope(module, q, k, cache)
    stored = next(iter(cache.values()))
    for batch, length in [(1, 17), (4, 64), (2, 128)]:
        q = torch.randn(batch, 3, length, 32, device=cuda, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        before = (q.clone(), k.clone())
        got = _cached_reference_rope(module, q, k, cache)
        cos, sin = _reference_rope_factors(length, 32, 10000.0, cuda)
        torch.testing.assert_close(stored[0][:length], cos, rtol=0, atol=0)
        torch.testing.assert_close(stored[1][:length], sin, rtol=0, atol=0)
        for value, source, saved in zip(got, (q, k), before):
            real, imag = source.float().reshape(batch, 3, length, 16, 2).unbind(-1)
            expected = torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).to(source.dtype)
            torch.testing.assert_close(value, expected.reshape_as(source), rtol=0, atol=0)
            torch.testing.assert_close(source, saved, rtol=0, atol=0)
        assert len(cache) == 1
        assert next(iter(cache.values())) is stored


def test_cached_rope_distinguishes_period_and_head_width(cuda):
    from types import SimpleNamespace

    cache: dict[tuple[int, float, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
    for width, period in [(32, 10000.0), (64, 10000.0), (32, 1000.0)]:
        module = SimpleNamespace(rope=SimpleNamespace(max_period=period))
        q = torch.randn(1, 2, 32, width, device=cuda, dtype=torch.bfloat16)
        _cached_reference_rope(module, q, q, cache)
    assert len(cache) == 3


@pytest.mark.parametrize("batched_transfer", [False, True])
def test_graph_codes_match_eager_per_clip_and_fall_back_outside_buckets(cuda, batched_transfer):
    torch.manual_seed(0)
    tokenizer = _Tokenizer().to(cuda)
    # Buckets of 2 s and 4 s at 64 Hz: 128 and 256 samples.
    graphs = MossReferenceEncoderGraphs(
        tokenizer, n_vq=N_VQ, batch_sizes=(1, 2, 4), bucket_seconds=(2.0, 4.0), batched_transfer=batched_transfer
    )
    graphs.capture()
    assert graphs.captured == [(1, 128), (1, 256), (2, 128), (2, 256), (4, 128), (4, 256)]
    assert graphs.max_batch == 4

    for count, frames in [(1, [5]), (3, [16, 3, 9]), (4, [32, 1, 20, 17]), (2, [16, 16])]:
        wavs = [torch.randn(2, n * FRAME) for n in frames]
        codes = graphs.encode(wavs)
        expected = tokenizer.eager([w.to(cuda) for w in wavs])
        assert len(codes) == count
        for got, want in zip(codes, expected):
            assert got.device.type == "cpu" and got.dtype == torch.long
            torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)

    # Longer than every bucket: left to the caller; the rest still use graphs.
    long_clip, short_clip = torch.randn(2, 33 * FRAME), torch.randn(2, 4 * FRAME)
    codes = graphs.encode([long_clip, short_clip])
    assert codes[0] is None
    torch.testing.assert_close(codes[1], tokenizer.eager([short_clip.to(cuda)])[0].cpu(), rtol=0, atol=0)

    # Wider than every batch: split, longest first, and returned in input order.
    wavs = [torch.randn(2, n * FRAME) for n in (3, 30, 7, 12, 1)]
    for got, want in zip(graphs.encode(wavs), tokenizer.eager([w.to(cuda) for w in wavs])):
        torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)

    # Result ownership includes the degenerate single-frame shape. Reusing
    # the pinned output slab must not overwrite a previous caller's codes.
    previous = graphs.encode([torch.randn(2, FRAME)])[0]
    snapshot = previous.clone()
    graphs.encode([torch.randn(2, 16 * FRAME) for _ in range(4)])
    torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)


def test_batch_audio_budget_limits_captured_graphs(cuda):
    tokenizer = _Tokenizer().to(cuda)
    graphs = MossReferenceEncoderGraphs(
        tokenizer, n_vq=N_VQ, batch_sizes=(1, 2, 4), bucket_seconds=(2.0, 4.0), max_batch_seconds=8.0
    )
    graphs.capture()
    # 4 x 4 s exceeds the 8 s budget; a single clip is always captured.
    assert graphs.captured == [(1, 128), (1, 256), (2, 128), (2, 256), (4, 128)]
    wavs = [torch.randn(2, 30 * FRAME) for _ in range(3)]
    for got, want in zip(graphs.encode(wavs), tokenizer.eager([w.to(cuda) for w in wavs])):
        torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)


def test_finer_singleton_buckets_preserve_multiclip_batching(cuda):
    tokenizer = _Tokenizer().to(cuda)
    graphs = MossReferenceEncoderGraphs(
        tokenizer,
        n_vq=N_VQ,
        batch_sizes=(1, 2, 4),
        bucket_seconds=(2.0, 4.0),
        singleton_bucket_seconds=(1.0, 3.0),
    )
    graphs.capture()
    assert (1, 64) in graphs.captured and (1, 192) in graphs.captured
    assert (2, 64) not in graphs.captured and (4, 192) not in graphs.captured
    replay = graphs._replay
    shapes = []

    def record(entry, wavs):
        shapes.append(tuple(entry.audio.shape))
        return replay(entry, wavs)

    graphs._replay = record
    one = torch.randn(2, 20 * FRAME)
    out = graphs.encode([one])
    assert shapes == [(1, 2, 192)]
    torch.testing.assert_close(out[0], tokenizer.eager([one.to(cuda)])[0].cpu(), rtol=0, atol=0)
    shapes.clear()
    wavs = [one, one[:, : 8 * FRAME], one[:, : 12 * FRAME]]
    out = graphs.encode(wavs)
    assert shapes == [(4, 2, 256)]
    for got, want in zip(out, tokenizer.eager([w.to(cuda) for w in wavs])):
        torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)


def test_compiled_capture_matches_eager(cuda):
    torch.manual_seed(2)
    tokenizer = _Tokenizer().to(cuda)
    graphs = MossReferenceEncoderGraphs(
        tokenizer, n_vq=N_VQ, batch_sizes=(1, 4), bucket_seconds=(2.0, 4.0), compile_core=True
    )
    graphs.capture()
    assert graphs.captured == [(1, 128), (1, 256), (4, 128), (4, 256)]
    wavs = [torch.randn(2, n * FRAME) for n in (5, 30, 12)]
    for got, want in zip(graphs.encode(wavs), tokenizer.eager([w.to(cuda) for w in wavs])):
        torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)


class _Attention(nn.Module):
    """The attention surface the tokenizer's MHA exposes to the windowed kernel."""

    def __init__(self, embed_dim=128, heads=2, context=5):
        super().__init__()
        self.embed_dim, self.num_heads, self.context, self.causal = embed_dim, heads, context, True
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

    def _project_qkv(self, x):
        b, t, _ = x.shape
        return self.in_proj(x).reshape(b, t, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)

    def _apply_dense_rope(self, q, k):
        return q, k

    def _forward_non_streaming_sdpa(self, x, input_lengths):
        # Reference semantics: causal, distance < context, padded rows zeroed.
        b, t, _ = x.shape
        q, k, v = self._project_qkv(x)
        pos = torch.arange(t, device=x.device)
        delta = pos.view(-1, 1) - pos.view(1, -1)
        mask = (delta >= 0) & (delta < self.context)
        out = F.scaled_dot_product_attention(q, k, v, mask[None, None])
        valid = (pos.view(1, t) < input_lengths.view(-1, 1)).view(b, 1, t, 1)
        out = torch.where(valid, out, torch.zeros((), device=out.device, dtype=out.dtype))
        return out.transpose(1, 2).reshape(b, t, self.embed_dim)


@pytest.mark.parametrize("fa_version", [2, 3])
def test_windowed_attention_matches_masked_sdpa(cuda, fa_version):
    if fa_version == 3 and torch.cuda.get_device_capability(cuda)[0] != 9:
        pytest.skip("FA3 requires Hopper")
    torch.manual_seed(1)
    model = nn.Module()
    model.encoder = nn.ModuleList([_Attention().to(cuda, torch.bfloat16)])
    attention = model.encoder[0]
    x = torch.randn(3, 40, 128, device=cuda, dtype=torch.bfloat16)
    lengths = torch.tensor([40, 17, 3], device=cuda)
    expected = attention._forward_non_streaming_sdpa(x, lengths)
    assert install_windowed_attention(model, fa_version=fa_version) == 1
    actual = attention._forward_non_streaming_sdpa(x, lengths)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert actual[1, 17:].count_nonzero() == 0 and actual[2, 3:].count_nonzero() == 0
    # Non-bf16 inputs keep the tokenizer's own SDPA path.
    x32 = x.float()
    attention.float()
    torch.testing.assert_close(
        attention._forward_non_streaming_sdpa(x32, lengths), expected.float(), rtol=3e-2, atol=3e-2
    )


@pytest.mark.parametrize("fa_version", [2, 3])
def test_unmasked_causal_queries_preserve_valid_prefix_and_ignore_future_padding(cuda, fa_version):
    if fa_version == 3 and torch.cuda.get_device_capability(cuda)[0] != 9:
        pytest.skip("FA3 requires Hopper")
    torch.manual_seed(44)
    attention = _Attention().to(cuda, torch.bfloat16)
    x = torch.randn(4, 40, 128, device=cuda, dtype=torch.bfloat16)
    lengths = torch.tensor([40, 17, 1, 0], device=cuda)
    kwargs = dict(sdpa=attention._forward_non_streaming_sdpa, fa_version=fa_version)
    with torch.no_grad():
        masked = _windowed_attention(attention, x, lengths, **kwargs)
        unmasked = _windowed_attention(attention, x, lengths, skip_padded_query_mask=True, **kwargs)
        poisoned = x.clone()
        for row, length in enumerate([40, 17, 1, 0]):
            poisoned[row, length:] = torch.randn_like(poisoned[row, length:]) * 30
        changed = _windowed_attention(attention, poisoned, lengths, skip_padded_query_mask=True, **kwargs)
        for row, length in enumerate([40, 17, 1, 0]):
            torch.testing.assert_close(unmasked[row, :length], masked[row, :length], rtol=0, atol=0)
            torch.testing.assert_close(changed[row, :length], masked[row, :length], rtol=0, atol=0)
            assert masked[row, length:].count_nonzero() == 0
        assert unmasked[1, 17:].count_nonzero() > 0
        # The FP32 fallback retains its original masking contract.
        attention.float()
        got = _windowed_attention(attention, x.float(), lengths, skip_padded_query_mask=True, **kwargs)
        expected = attention._forward_non_streaming_sdpa(x.float(), lengths)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_unmasked_queries_reject_unknown_encoder_before_patching():
    tokenizer = _Tokenizer()
    with pytest.raises(ValueError, match="patch-downsample"):
        install_windowed_attention(tokenizer, skip_padded_query_mask=True)


@pytest.mark.parametrize("bad_part", ["upsample", "noncausal", "quantizer", "convolution"])
def test_unmasked_queries_reject_structures_that_can_read_invalid_rows(bad_part):
    from types import SimpleNamespace

    patch = type("MossAudioTokenizerPatchedPretransform", (), {})()
    patch.is_downsample = bad_part != "upsample"
    projected = type("MossAudioTokenizerProjectedTransformer", (), {})()
    projected.transformer = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(causal=bad_part != "noncausal"))]
    )
    quantizer_cls = type(
        "MossAudioTokenizerResidualVQ" if bad_part != "quantizer" else "UnknownQuantizer", (nn.Module,), {}
    )
    quantizer = quantizer_cls()
    quantizer.proj = nn.Conv1d(2, 2, 3 if bad_part == "convolution" else 1)
    tokenizer = SimpleNamespace(encoder=[patch, projected], quantizer=quantizer)
    with pytest.raises(ValueError):
        _validate_unmasked_reference_encoder(tokenizer)
