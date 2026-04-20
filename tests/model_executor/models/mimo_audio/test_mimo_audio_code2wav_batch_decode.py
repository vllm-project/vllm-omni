# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import TALKER_CODEC_PAD_TOKEN_ID
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import (
    AudioStreamerConfig,
    MiMoAudioToken2WavForConditionalGenerationVLLM,
    flat_codec_group_element_count,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_GROUP = 4
_AC = 8
_GROUP_WIDTH = flat_codec_group_element_count(_GROUP, _AC)
_FTP = 2 * 2 * 240  # frames_per_token from mocked tokenizer.config


def _codes_ns(empty: int = 555, eostm: int = 666):
    return SimpleNamespace(empty=empty, eostm=eostm)


def _make_dummy_code_tensor() -> torch.Tensor:
    """Pad-only talker dummy; matches _check_dummy_code_tensor."""
    t = torch.zeros(flat_codec_group_element_count(_GROUP, _AC), dtype=torch.long)
    t = t.view(_GROUP, _AC + 1)
    t[:, 0] = TALKER_CODEC_PAD_TOKEN_ID
    return t.view(-1)


def _make_valid_flat_codes(num_groups: int = 1, *, empty_id: int = 555) -> torch.Tensor:
    """Flat layout that extract_audio_code_tensor accepts (text col == empty)."""
    parts = []
    for _ in range(num_groups):
        g = torch.zeros(_GROUP, _AC + 1, dtype=torch.long)
        g[0, 0] = empty_id
        g[:, 1:] = torch.randint(1, 50, (_GROUP, _AC))
        parts.append(g.reshape(-1))
    return torch.cat(parts)


def _make_invalid_flat_immediate_eostm(eostm_id: int = 666) -> torch.Tensor:
    g = torch.zeros(_GROUP, _AC + 1, dtype=torch.long)
    g[0, 0] = eostm_id
    return g.reshape(-1)


def _minimal_model(mocker: MockerFixture):
    """Avoid __init__ (HF tokenizer paths); only fields used by _batch_decode_waveforms."""
    model = object.__new__(MiMoAudioToken2WavForConditionalGenerationVLLM)
    model.device = torch.device("cpu")
    model.config = SimpleNamespace(group_size=_GROUP, audio_channels=_AC)
    model.streamer_config = AudioStreamerConfig(group_size=_GROUP, audio_channels=_AC)
    model.codes = _codes_ns()

    decode_vq = mocker.Mock(
        side_effect=lambda audio_codes: torch.ones(
            audio_codes.shape[1],
            7,
            dtype=torch.float32,
            device=audio_codes.device,
        )
    )
    decoder = mocker.Mock()

    audio_tok = SimpleNamespace(
        encoder=SimpleNamespace(decode_vq=decode_vq),
        decoder=decoder,
        config=SimpleNamespace(avg_pooler=2, stride_size=2, hop_length=240),
    )
    model._tokenizer_service = SimpleNamespace(audio_tokenizer=audio_tok)
    return model, audio_tok


def test_batch_decode_waveforms_empty_input_list(mocker: MockerFixture):
    """Empty input list returns a single zero-length float32 tensor on model device."""
    model, _ = _minimal_model(mocker)
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, [])
    assert len(out) == 1
    assert out[0].dtype == torch.float32
    assert out[0].numel() == 0
    assert out[0].device == model.device


def test_batch_decode_waveforms_single_vs_multiple_decoder_shapes(mocker: MockerFixture):
    """Single and multi-request batches produce correctly shaped packed hidden states and trimmed waveforms."""
    model, audio_tok = _minimal_model(mocker)
    decoder = audio_tok.decoder

    # Single valid request: decoder output rank-3 for double squeeze path
    flat = _make_valid_flat_codes(1)
    decoder.return_value = torch.ones(1, 1, 4 * _FTP + 100, dtype=torch.float32)
    out1 = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, [flat])
    decoder.assert_called_once()
    packed_hs, input_lengths = decoder.call_args[0]
    assert packed_hs.shape == (4, 7)  # T from decode_vq mock
    assert input_lengths.tolist() == [4]
    assert out1[0].shape == (4 * _FTP,)
    assert out1[0].dtype == torch.float32

    decoder.reset_mock()
    a = _make_valid_flat_codes(1)
    b = _make_valid_flat_codes(2)
    decoder.return_value = torch.ones(2, 1, 8 * _FTP + 100, dtype=torch.float32)
    out2 = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, [a, b])
    decoder.assert_called_once()
    packed_hs2, input_lengths2 = decoder.call_args[0]
    assert packed_hs2.shape == (4 + 8, 7)  # 1 group -> T=4; 2 groups -> T=8
    assert input_lengths2.tolist() == [4, 8]
    assert len(out2) == 2
    assert out2[0].shape == (4 * _FTP,)
    assert out2[1].shape == (8 * _FTP,)


def test_batch_decode_waveforms_mixed_valid_invalid_requests(mocker: MockerFixture):
    """Mixed valid and invalid requests: invalid slots get empty tensors, valid slots get decoded waveforms."""
    model, audio_tok = _minimal_model(mocker)
    valid_a = _make_valid_flat_codes(1)
    valid_b = _make_valid_flat_codes(1)
    dummy = _make_dummy_code_tensor()
    bad_extract = _make_invalid_flat_immediate_eostm(eostm_id=model.codes.eostm)
    too_short = torch.tensor([1, 2, 3], dtype=torch.long)

    audio_tok.decoder.return_value = torch.ones(2, 1, 5000, dtype=torch.float32)

    inputs = [
        None,
        torch.tensor([], dtype=torch.long),
        dummy,
        bad_extract,
        too_short,
        valid_a,
        valid_b,
    ]
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, inputs)

    assert len(out) == len(inputs)
    for i in range(5):
        assert out[i].numel() == 0, f"index {i} should be empty"
    assert out[5].shape == (4 * _FTP,)
    assert out[6].shape == (4 * _FTP,)
    audio_tok.decoder.assert_called_once()
    packed_hs, input_lengths = audio_tok.decoder.call_args[0]
    assert packed_hs.shape[0] == 8
    assert input_lengths.tolist() == [4, 4]


def test_batch_decode_waveforms_all_invalid_returns_per_request_empty(mocker: MockerFixture):
    """All-invalid batch skips decoder entirely and returns empty tensors for every slot."""
    model, audio_tok = _minimal_model(mocker)
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(
        model,
        [None, _make_dummy_code_tensor(), torch.tensor([], dtype=torch.long)],
    )
    assert len(out) == 3
    assert all(t.numel() == 0 for t in out)
    audio_tok.decoder.assert_not_called()


def test_batch_decode_waveforms_output_shape_trim_when_decoder_returns_extra_samples(mocker: MockerFixture):
    """Decoder output longer than valid_len is trimmed to the exact expected waveform length."""
    model, audio_tok = _minimal_model(mocker)
    flat = _make_valid_flat_codes(1)
    # Longer than valid_len so branch wav = wav[:valid_len] runs
    audio_tok.decoder.return_value = torch.ones(1, 1, 10_000, dtype=torch.float32)
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, [flat])
    assert out[0].dim() == 1
    assert out[0].numel() == 4 * _FTP
    assert out[0].dtype == torch.float32


def test_batch_decode_waveforms_multi_request_trims_each_row_when_decoder_returns_extra(mocker: MockerFixture):
    """Else-branch split: per-request wav[:valid_len] when decoder pads each batch row."""
    model, audio_tok = _minimal_model(mocker)
    a = _make_valid_flat_codes(1)
    b = _make_valid_flat_codes(2)
    audio_tok.decoder.return_value = torch.ones(2, 1, 10_000, dtype=torch.float32)
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, [a, b])
    assert len(out) == 2
    assert out[0].shape == (4 * _FTP,)
    assert out[1].shape == (8 * _FTP,)
    assert out[0].dtype == torch.float32
    assert out[1].dtype == torch.float32


def test_batch_decode_waveforms_valid_only_at_edges_maps_to_correct_indices(mocker: MockerFixture):
    """Tensor packing order must match valid_indices when invalid requests are in the middle."""
    model, audio_tok = _minimal_model(mocker)
    first = _make_valid_flat_codes(1)
    last = _make_valid_flat_codes(2)
    inputs = [
        first,
        None,
        _make_dummy_code_tensor(),
        last,
    ]
    audio_tok.decoder.return_value = torch.ones(2, 1, 10_000, dtype=torch.float32)
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, inputs)
    assert len(out) == 4
    assert out[0].numel() == 4 * _FTP
    assert out[1].numel() == 0
    assert out[2].numel() == 0
    assert out[3].numel() == 8 * _FTP
    packed_hs, input_lengths = audio_tok.decoder.call_args[0]
    assert packed_hs.shape[0] == 12
    assert input_lengths.tolist() == [4, 8]


def test_batch_decode_waveforms_output_shapes_1d_float32_for_all_slots(mocker: MockerFixture):
    """Every slot is a 1-D float32 vector (empty or waveform), matching downstream expectations."""
    model, audio_tok = _minimal_model(mocker)
    inputs = [_make_valid_flat_codes(1), None, _make_valid_flat_codes(1)]
    audio_tok.decoder.return_value = torch.ones(2, 1, 5000, dtype=torch.float32)
    out = MiMoAudioToken2WavForConditionalGenerationVLLM._batch_decode_waveforms(model, inputs)
    assert len(out) == 3
    for i, t in enumerate(out):
        assert t.dim() == 1, f"slot {i}"
        assert t.dtype == torch.float32, f"slot {i}"
    assert out[0].numel() == 4 * _FTP
    assert out[1].numel() == 0
    assert out[2].numel() == 4 * _FTP
