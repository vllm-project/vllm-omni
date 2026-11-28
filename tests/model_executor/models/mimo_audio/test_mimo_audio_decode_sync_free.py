# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the sync-free mimo_audio stage-1 decode path.

Covers:
  - ``extract_audio_code_tensor_device``: on-device extraction must be bit-exact
    against the CPU reference ``extract_audio_code_tensor`` across eostm /
    partial-group / no-audio edge cases (CPU and CUDA tensors).
  - ``codec_graph_capture_frames`` / ``_extend_codec_graph_buckets``: the
    chunk-derived exact bucket sizes and their wiring into the CUDA-graph
    decoder wrapper (the steady-state async-chunk payload of 280 code frames
    exceeds the wrapper's largest default bucket of 256 and would otherwise
    silently fall back to eager decode every chunk).
  - ``CUDAGraphMiMoDecoderWrapper.extend_capture_sizes``: new buckets are
    captured, idempotent, and replay bit-stable vs eager ``decode_fixed``.
"""

import functools
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = pytest.mark.core_model

_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")

_GROUP = 4
_AC = 8
_GROUP_WIDTH = _GROUP * (_AC + 1)  # 36 flat tokens per talker group


@functools.lru_cache(maxsize=1)
def _mimo_code2wav_deps():
    """Defer mimo code2wav (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import (
        AudioStreamerConfig,
        MiMoAudioToken2WavForConditionalGenerationVLLM,
        codec_graph_capture_frames,
        extract_audio_code_tensor,
        extract_audio_code_tensor_device,
    )

    return (
        AudioStreamerConfig,
        MiMoAudioToken2WavForConditionalGenerationVLLM,
        codec_graph_capture_frames,
        extract_audio_code_tensor,
        extract_audio_code_tensor_device,
    )


def _codes_ns(empty: int = 555, eostm: int = 666):
    return SimpleNamespace(empty=empty, eostm=eostm)


def _make_flat_codes(
    num_groups: int,
    *,
    eostm_group: int | None = None,
    text_ids: dict[int, int] | None = None,
    empty_id: int = 555,
    device=None,
) -> torch.Tensor:
    """Flat [n, 4, 9] group layout the processor emits (text slot = empty)."""
    g = torch.randint(1, 50, (num_groups, _GROUP, _AC + 1), device=device, dtype=torch.long)
    g[:, :, 0] = empty_id
    if eostm_group is not None:
        g[eostm_group, 0, 0] = _codes_ns().eostm
    for group_idx, token in (text_ids or {}).items():
        g[group_idx, 0, 0] = token
    return g.reshape(-1)


# ---------------------------------------------------------------------------
# On-device extraction parity (CPU tensors; the op set is device-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("label", "flat"),
    [
        ("normal", _make_flat_codes(70)),
        ("single_group", _make_flat_codes(1)),
        ("eostm_first", _make_flat_codes(70, eostm_group=0)),
        ("eostm_mid", _make_flat_codes(70, eostm_group=35)),
        ("eostm_last", _make_flat_codes(70, eostm_group=69)),
        (
            "partial_trailing_group",
            torch.cat([_make_flat_codes(70), torch.randint(1, 50, (5,))]),
        ),
        ("no_audio_after_text_tokens", _make_flat_codes(70, text_ids={i: 42 for i in range(70)})),
        (
            "text_tokens_mixed",
            _make_flat_codes(70, text_ids={3: 42, 17: 43}),
        ),
        ("empty_tensor", torch.zeros(0, dtype=torch.long)),
        ("stray_tokens_only", torch.randint(1, 50, (7,))),
    ],
)
def test_extract_device_matches_cpu_reference(label, flat):
    """Device extraction must reproduce the reference loop bit-exactly."""
    _, _, _, ref_fn, dev_fn = _mimo_code2wav_deps()
    ref = ref_fn(flat.cpu(), _GROUP, _AC, _codes_ns())
    new = dev_fn(flat, _GROUP, _AC, _codes_ns())
    if ref is None or new is None:
        assert ref is None and new is None, (label, ref, new)
    else:
        assert torch.equal(ref, new), label


@pytest.mark.cpu
def test_extract_device_keeps_gpu_residency():
    """Extraction output stays on the input device (no hidden host round trip)."""
    _, _, _, _, dev_fn = _mimo_code2wav_deps()
    out = dev_fn(_make_flat_codes(70), _GROUP, _AC, _codes_ns())
    assert out is not None and out.is_cpu and out.shape == (_AC, 70 * _GROUP)


@pytest.mark.cpu
def test_extract_device_dummy_tensor_is_none():
    """A pad-only dummy group extracts to None (caller skips the request)."""
    from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import TALKER_CODEC_PAD_TOKEN_ID

    _, _, _, _, dev_fn = _mimo_code2wav_deps()
    t = torch.zeros(_GROUP_WIDTH, dtype=torch.long)
    t = t.view(_GROUP, _AC + 1)
    t[:, 0] = TALKER_CODEC_PAD_TOKEN_ID
    assert dev_fn(t.view(-1), _GROUP, _AC, _codes_ns()) is None


# ---------------------------------------------------------------------------
# Chunk-derived capture sizes (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_codec_graph_capture_frames_shipped_config():
    """30 chunk + 40 left-context groups x 4 = the 280-frame steady-state payload."""
    _, _, frames_fn, _, _ = _mimo_code2wav_deps()
    assert frames_fn(chunk_frames=30, left_context_frames=40, group_size=4) == [120, 280]


@pytest.mark.cpu
def test_codec_graph_capture_frames_edges():
    _, _, frames_fn, _, _ = _mimo_code2wav_deps()
    assert frames_fn(chunk_frames=10, left_context_frames=40, group_size=1) == [10, 50]
    assert frames_fn(chunk_frames=0, left_context_frames=0, group_size=4) == []


@pytest.mark.cpu
def test_extend_codec_graph_buckets_passes_chunk_frames(mocker: MockerFixture):
    """The model threads chunk-derived bucket sizes into the worker's wrapper."""
    _, model_cls, _, _, _ = _mimo_code2wav_deps()
    AudioStreamerConfig = _mimo_code2wav_deps()[0]

    wrapper = mocker.MagicMock()
    model = object.__new__(model_cls)
    model._tokenizer_service = SimpleNamespace(cuda_graph_wrapper=wrapper)
    model._codec_chunk_frames = 30
    model._codec_left_context_frames = 40
    model.streamer_config = AudioStreamerConfig(group_size=_GROUP, audio_channels=_AC)

    model._extend_codec_graph_buckets()

    wrapper.extend_capture_sizes.assert_called_once_with([120, 280])


@pytest.mark.cpu
def test_extend_codec_graph_buckets_noop_without_wrapper(mocker: MockerFixture):
    """CPU tokenizer device (wrapper is None) or env-gated graphs: no-op."""
    _, model_cls, _, _, _ = _mimo_code2wav_deps()
    AudioStreamerConfig = _mimo_code2wav_deps()[0]

    model = object.__new__(model_cls)
    model._tokenizer_service = SimpleNamespace(cuda_graph_wrapper=None)
    model._codec_chunk_frames = 30
    model._codec_left_context_frames = 40
    model.streamer_config = AudioStreamerConfig(group_size=_GROUP, audio_channels=_AC)

    model._extend_codec_graph_buckets()  # must not raise


# ---------------------------------------------------------------------------
# Wrapper bucket extension + on-device extraction (CUDA)
# ---------------------------------------------------------------------------


def _small_tokenizer():
    """Small random-weight tokenizer so graph capture stays fast in tests."""
    from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioTokenizerConfig
    from vllm_omni.model_executor.models.mimo_audio.modeling_audio_tokenizer import MiMoAudioTokenizer

    cfg = MiMoAudioTokenizerConfig(
        avg_pooler=2,
        d_model=64,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_layers=1,
        decoder_attention_heads=4,
        decoder_ffn_dim=128,
        vocoder_dim=64,
        vocoder_intermediate_dim=128,
        vocoder_num_layers=2,
        vocoder_attention_heads=2,
        vocoder_attn_window_size=[8, 4],
        num_quantizers=8,
    )
    torch.manual_seed(0)
    return MiMoAudioTokenizer(cfg).to("cuda", torch.bfloat16).eval()


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_extract_device_parity_on_cuda():
    _, _, _, ref_fn, dev_fn = _mimo_code2wav_deps()
    flat = _make_flat_codes(70, eostm_group=35, device="cuda")
    ref = ref_fn(flat.cpu(), _GROUP, _AC, _codes_ns())
    new = dev_fn(flat, _GROUP, _AC, _codes_ns())
    assert new is not None and new.is_cuda
    assert torch.equal(ref, new.cpu())


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_wrapper_extend_capture_sizes():
    """Extended buckets are captured once and replay bit-stably."""
    from vllm_omni.model_executor.models.mimo_audio.cuda_graph_decoder_wrapper import (
        CUDAGraphMiMoDecoderWrapper,
    )
    from vllm_omni.model_executor.models.mimo_audio.modeling_audio_tokenizer import TransformerVocos

    tok = _small_tokenizer()
    device = torch.device("cuda")
    wrapper = CUDAGraphMiMoDecoderWrapper(tok, capture_sizes=[4, 8])
    wrapper.warmup(device)
    assert wrapper.is_ready
    assert wrapper.capture_sizes == [4, 8]

    wrapper.extend_capture_sizes([17, 40])
    assert 17 in wrapper.capture_sizes and 40 in wrapper.capture_sizes
    assert wrapper.capture_sizes == sorted(wrapper.capture_sizes)
    assert (8, 17) in wrapper.graphs and (8, 40) in wrapper.graphs

    n_graphs = len(wrapper.graphs)
    wrapper.extend_capture_sizes([17, 40])  # idempotent
    assert len(wrapper.graphs) == n_graphs
    wrapper.extend_capture_sizes([0, -3])  # invalid sizes ignored
    assert len(wrapper.graphs) == n_graphs

    # Exact-bucket replay must be bit-stable vs eager decode_fixed.
    codes = torch.randint(0, 1024, (8, 17), device=device, dtype=torch.long)
    voc_len = tok._compute_vocoder_seq_len(17)
    mask = TransformerVocos.build_vocoder_attn_mask(voc_len, voc_len, (8, 4), device, torch.bfloat16)
    with torch.no_grad():
        ref = tok.decode_fixed(codes, 17, vocoder_attn_mask=mask)
        rep = wrapper.decode(codes)
    assert torch.equal(ref, rep)

    # Sub-bucket length pads up to the extended bucket and slices back.
    short = codes[:, :10]
    wav = wrapper.decode(short)
    assert wav.shape[-1] == tok._compute_vocoder_seq_len(10) * tok.config.hop_length
