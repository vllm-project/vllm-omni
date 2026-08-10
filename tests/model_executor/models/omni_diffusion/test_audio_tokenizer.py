# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionAudioTokenizer audio normalization and helper methods."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.model_executor.models.omni_diffusion.audio_tokenizer import (
    OmniDiffusionAudioEncodingMode,
    OmniDiffusionAudioTokenizer,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_tokenizer_base_data() -> OmniDiffusionTokenizerBaseData:
    """Build a TokenizerBaseData with deterministic token IDs for testing."""
    tok = MagicMock()

    def _encode(tokens, add_special_tokens):
        ids = []
        for t in tokens:
            for special in OmniDiffusionModelSpecialTokens:
                if special.value == t:
                    ids.append([100 + list(OmniDiffusionModelSpecialTokens).index(special)])
                    break
            else:
                ids.append([999])
        return SimpleNamespace(input_ids=ids)

    tok.side_effect = _encode
    return OmniDiffusionTokenizerBaseData(tok)


def _make_audio_tokenizer() -> OmniDiffusionAudioTokenizer:
    """Create an audio tokenizer without loading real models (lazy loading)."""
    return OmniDiffusionAudioTokenizer(
        sensevoice_path="/fake/sensevoice",
        flow_path="/fake/flow",
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Audio tokenizer: construction
# ---------------------------------------------------------------------------


class TestAudioTokenizerConstruction:
    def test_init_stores_paths_and_device(self) -> None:
        tok = _make_audio_tokenizer()
        assert tok.sensevoice_path == "/fake/sensevoice"
        assert tok.flow_path == "/fake/flow"
        assert tok.device == torch.device("cpu")

    def test_weights_not_loaded_on_init(self) -> None:
        tok = _make_audio_tokenizer()
        assert not hasattr(tok, "audio_decoder")
        assert not hasattr(tok, "sensevoice_kwargs")

    def test_common_resample_buffer_is_prebuilt(self) -> None:
        tok = _make_audio_tokenizer()
        assert len(tok._common_resample_buffer) > 0
        for sr in (8000, 22050, 24000, 32000, 44100, 48000):
            assert sr in tok._common_resample_buffer


class TestAudioDecode:
    def test_decode_uses_unconditioned_prompt_inputs(self) -> None:
        tok = _make_audio_tokenizer()
        decoder = MagicMock()
        decoder.token2wav.return_value = (
            torch.ones((1, 1, 32), dtype=torch.bfloat16),
            None,
        )
        tok.audio_decoder = decoder

        result = tok.decode([1, 2, 3], option_steps=7)

        assert result.shape == (32,)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"
        kwargs = decoder.token2wav.call_args.kwargs
        assert kwargs["finalize"] is True
        assert kwargs["option_steps"] == 7
        assert kwargs["prompt_token"].shape == (1, 0)
        assert kwargs["prompt_feat"].shape == (1, 0, 80)
        assert decoder.token2wav.call_args.args[0].tolist() == [[1, 2, 3]]

    def test_decode_requires_flow_path(self) -> None:
        tok = OmniDiffusionAudioTokenizer(
            sensevoice_path=None,
            flow_path=None,
            device="cpu",
        )
        with pytest.raises(ValueError, match="flow_path"):
            tok.decode([1, 2, 3])


# ---------------------------------------------------------------------------
# Audio normalization
# ---------------------------------------------------------------------------


class TestAudioNormalization:
    def test_normalize_mono_waveform(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(16000)
        result = tok._normalize_audio_waveform(audio)
        assert result.ndim == 1
        assert result.dtype == torch.float32

    def test_normalize_stereo_to_mono(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(2, 16000)
        result = tok._normalize_audio_waveform(audio)
        assert result.ndim == 1
        assert result.shape[0] == 16000

    def test_normalize_clips_large_values(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.tensor([500.0, -300.0, 100.0])
        result = tok._normalize_audio_waveform(audio)
        assert result.abs().max() <= 1.0

    def test_normalize_rejects_empty_audio(self) -> None:
        tok = _make_audio_tokenizer()
        with pytest.raises(ValueError, match="non-empty"):
            tok._normalize_audio_waveform(torch.zeros(0))

    def test_normalize_rejects_wrong_ndim(self) -> None:
        tok = _make_audio_tokenizer()
        with pytest.raises(ValueError, match="shape"):
            tok._normalize_audio_waveform(torch.randn(3, 2, 16000))


# ---------------------------------------------------------------------------
# Normalize audio tensors (request packing)
# ---------------------------------------------------------------------------


class TestNormalizeAudioTensors:
    def test_single_mono_tensor_returns_list_of_one(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(16000)
        result = tok._normalize_audio_tensors(audio)
        assert len(result) == 1
        assert result[0] is audio

    def test_single_stereo_tensor_returns_list_of_one(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(2, 16000)
        result = tok._normalize_audio_tensors(audio)
        assert len(result) == 1

    def test_batched_tensor_returns_list(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(3, 2, 16000)
        result = tok._normalize_audio_tensors(audio)
        assert len(result) == 3
        for item in result:
            assert item.ndim == 2

    def test_sequence_of_tensors_passes_through(self) -> None:
        tok = _make_audio_tokenizer()
        audios = [torch.randn(16000), torch.randn(2, 8000)]
        result = tok._normalize_audio_tensors(audios)
        assert len(result) == 2

    def test_rejects_wrong_ndim_tensor(self) -> None:
        tok = _make_audio_tokenizer()
        with pytest.raises(ValueError, match="shape"):
            tok._normalize_audio_tensors(torch.randn(4, 3, 2, 16000))

    def test_rejects_invalid_type(self) -> None:
        tok = _make_audio_tokenizer()
        with pytest.raises(TypeError, match="tensor or sequence"):
            tok._normalize_audio_tensors("not_audio")

    def test_rejects_sequence_with_non_tensor_item(self) -> None:
        tok = _make_audio_tokenizer()
        with pytest.raises(TypeError, match="torch.Tensor"):
            tok._normalize_audio_tensors([torch.randn(16000), "not_a_tensor"])


# ---------------------------------------------------------------------------
# Normalize audio sample rates
# ---------------------------------------------------------------------------


class TestNormalizeAudioSampleRates:
    def test_none_defaults_to_input_rate(self) -> None:
        tok = _make_audio_tokenizer()
        rates = tok._normalize_audio_sample_rates(None, audio_count=3)
        assert rates == [OMNI_DIFFUSION_INPUT_SAMPLE_RATE] * 3

    def test_single_int_broadcasts(self) -> None:
        tok = _make_audio_tokenizer()
        rates = tok._normalize_audio_sample_rates(44100, audio_count=2)
        assert rates == [44100] * 2

    def test_scalar_tensor_broadcasts(self) -> None:
        tok = _make_audio_tokenizer()
        rates = tok._normalize_audio_sample_rates(torch.tensor(22050), audio_count=2)
        assert rates == [torch.tensor(22050)] * 2

    def test_sequence_passes_through(self) -> None:
        tok = _make_audio_tokenizer()
        rates = tok._normalize_audio_sample_rates([16000, 44100], audio_count=2)
        assert rates == [16000, 44100]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:
    def test_passthrough_when_rate_already_matches(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(16000)
        result = tok._resample_to_input_sample_rate(audio, OMNI_DIFFUSION_INPUT_SAMPLE_RATE)
        assert result is audio

    def test_resample_uses_common_buffer(self) -> None:
        tok = _make_audio_tokenizer()
        audio = torch.randn(44100)
        # Only check it doesn't crash — actual resampling needs torchaudio.
        # Pre-cached common rate resampler should exist.
        result = tok._resample_to_input_sample_rate(audio, 44100)
        assert result.ndim == 1
        # The length should change roughly by the ratio.
        expected_len = int(round(audio.shape[0] * OMNI_DIFFUSION_INPUT_SAMPLE_RATE / 44100))
        assert abs(result.shape[0] - expected_len) <= 2

    def test_uncommon_rates_use_bounded_dynamic_cache(self) -> None:
        tok = _make_audio_tokenizer()

        first = tok._get_resampler(12345)
        second = tok._get_resampler(12345)

        assert first is second


# ---------------------------------------------------------------------------
# prepare_contiguous_audio_inputs
# ---------------------------------------------------------------------------


class TestPrepareContiguousAudioInputs:
    def test_replaces_audio_placeholder(self) -> None:
        tok = _make_audio_tokenizer()
        base_data = _make_tokenizer_base_data()
        aud_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)

        # We need to mock encode() since it requires SenseVoice model loading.
        with patch.object(tok, "encode", return_value=torch.randn(50, 80)):
            input_ids = [10, aud_tag_id, 20]
            new_ids, audios, audio_indices = tok.prepare_contiguous_audio_inputs(
                input_ids=input_ids,
                omni_audios=torch.randn(16000),
                omni_audio_sample_rates=16000,
                tokenizer_base_data=base_data,
            )
            assert aud_tag_id not in new_ids
            assert len(audios) == 1
            assert len(audio_indices) == 1

    def test_preserves_non_placeholder_tokens(self) -> None:
        tok = _make_audio_tokenizer()
        base_data = _make_tokenizer_base_data()
        aud_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)

        with patch.object(tok, "encode", return_value=torch.randn(50, 80)):
            input_ids = [10, aud_tag_id, 20, 30]
            new_ids, _, _ = tok.prepare_contiguous_audio_inputs(
                input_ids=input_ids,
                omni_audios=torch.randn(16000),
                omni_audio_sample_rates=16000,
                tokenizer_base_data=base_data,
            )
            # The tokens before the placeholder should be preserved.
            assert new_ids[0] == 10
            # The tokens after should be at the end.
            assert new_ids[-1] == 30

    def test_raises_on_audio_placeholder_mismatch(self) -> None:
        tok = _make_audio_tokenizer()
        base_data = _make_tokenizer_base_data()
        aud_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)

        with patch.object(tok, "encode", return_value=torch.randn(50, 80)):
            # 2 placeholders but only 1 audio tensor.
            input_ids = [aud_tag_id, aud_tag_id]
            with pytest.raises(ValueError, match="placeholder"):
                tok.prepare_contiguous_audio_inputs(
                    input_ids=input_ids,
                    omni_audios=torch.randn(16000),
                    omni_audio_sample_rates=16000,
                    tokenizer_base_data=base_data,
                )

    def test_raises_on_sample_rate_mismatch(self) -> None:
        tok = _make_audio_tokenizer()
        base_data = _make_tokenizer_base_data()
        aud_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)

        with patch.object(tok, "encode", return_value=torch.randn(50, 80)):
            input_ids = [aud_tag_id]
            # 1 audio tensor but 2 sample rates.
            with pytest.raises(ValueError, match="sample rate"):
                tok.prepare_contiguous_audio_inputs(
                    input_ids=input_ids,
                    omni_audios=torch.randn(16000),
                    omni_audio_sample_rates=[16000, 44100],
                    tokenizer_base_data=base_data,
                )

    def test_audio_indices_have_correct_shape(self) -> None:
        tok = _make_audio_tokenizer()
        base_data = _make_tokenizer_base_data()
        aud_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)

        with patch.object(tok, "encode", return_value=torch.randn(50, 80)):
            input_ids = [aud_tag_id]
            _, _, audio_indices = tok.prepare_contiguous_audio_inputs(
                input_ids=input_ids,
                omni_audios=torch.randn(16000),
                omni_audio_sample_rates=16000,
                tokenizer_base_data=base_data,
            )
            assert len(audio_indices) == 1
            # Shape: [coordinate, batch, audio_token_length].
            assert audio_indices[0].ndim == 3
            assert audio_indices[0].shape[:2] == (2, 1)


# ---------------------------------------------------------------------------
# Audio encoding mode enum
# ---------------------------------------------------------------------------


class TestAudioEncodingMode:
    def test_discrete_not_implemented(self) -> None:
        tok = _make_audio_tokenizer()
        with pytest.raises(NotImplementedError, match="Discrete"):
            tok.encode(
                torch.randn(16000),
                16000,
                mode=OmniDiffusionAudioEncodingMode.DISCRETE,
            )

    def test_contiguous_is_default(self) -> None:
        mode = inspect.signature(OmniDiffusionAudioTokenizer.encode).parameters["mode"]
        assert mode.default is OmniDiffusionAudioEncodingMode.CONTIGUOUS
