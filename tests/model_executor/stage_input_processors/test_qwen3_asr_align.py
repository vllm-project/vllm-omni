# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Qwen3-ASR forced-aligner stage input processor."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vllm_omni.model_executor.stage_input_processors import qwen3_asr_align as proc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "r0"):
    return SimpleNamespace(request_id=request_id, outputs=[SimpleNamespace(text=text)])


def _raw(transcript: str) -> str:
    """Qwen3-ASR's wire format, which the endpoint strips before returning."""
    return f"language English<asr_text>{transcript}"


class TestExtractText:
    def test_strips_the_asr_scaffolding(self):
        """Regression: the scaffolding was being aligned as spoken words.

        Segmenting the raw generation yields "language", "English", "asr" and
        "text" as leading words, which shifts every subsequent timestamp onto
        the wrong word -- silently, since the count still looks plausible.
        """
        got = proc._extract_text(_source_output(_raw("hello world")))
        assert got == "hello world"
        assert "asr_text" not in got
        assert not got.startswith("language")

    def test_empty_when_no_outputs(self):
        assert proc._extract_text(SimpleNamespace(outputs=[])) == ""


class TestAttachAlignerAudio:
    def test_attaches_to_mapping_prompt(self):
        prompt: dict = {}
        wave = np.zeros(16000, dtype=np.float32)
        proc.attach_aligner_audio(prompt, wave, 16000)
        carried = prompt["additional_information"][proc.ALIGNER_AUDIO_KEY]
        assert carried[1] == 16000
        assert len(carried[0]) == 16000

    def test_preserves_existing_additional_information(self):
        prompt = {"additional_information": {"global_request_id": ["abc"]}}
        proc.attach_aligner_audio(prompt, np.zeros(8, dtype=np.float32), 16000)
        assert prompt["additional_information"]["global_request_id"] == ["abc"]
        assert proc.ALIGNER_AUDIO_KEY in prompt["additional_information"]

    def test_attaches_to_attribute_prompt(self):
        prompt = SimpleNamespace()
        proc.attach_aligner_audio(prompt, np.zeros(4, dtype=np.float32), 16000)
        assert proc.ALIGNER_AUDIO_KEY in prompt.additional_information


class TestAudioFromPrompt:
    def test_prefers_the_carried_waveform(self):
        wave = np.arange(4, dtype=np.float32)
        src = {"additional_information": {proc.ALIGNER_AUDIO_KEY: (wave, 16000)}}
        got = proc._audio_from_prompt(src)
        assert got is not None and got[1] == 16000

    def test_unwraps_single_element_list(self):
        """additional_information values are list-wrapped on the way through."""
        wave = np.arange(4, dtype=np.float32)
        src = {"additional_information": {proc.ALIGNER_AUDIO_KEY: [(wave, 16000)]}}
        got = proc._audio_from_prompt(src)
        assert got is not None and got[1] == 16000

    def test_falls_back_to_raw_multimodal_data(self):
        wave = np.arange(4, dtype=np.float32)
        got = proc._audio_from_prompt({"multi_modal_data": {"audio": (wave, 22050)}})
        assert got is not None and got[1] == 22050

    def test_none_when_no_audio_anywhere(self):
        assert proc._audio_from_prompt({}) is None


class TestAudioSpanEnd:
    def test_reads_offset_plus_length(self):
        src = {"mm_placeholders": {"audio": [SimpleNamespace(offset=5, length=100)]}}
        assert proc._audio_span_end(src) == 105

    def test_accepts_mapping_shaped_ranges(self):
        src = {"mm_placeholders": {"audio": [{"offset": 2, "length": 8}]}}
        assert proc._audio_span_end(src) == 10

    def test_none_without_placeholders(self):
        assert proc._audio_span_end({}) is None


class TestAsr2Aligner:
    """The processor pairs stage 0's transcript with the request's audio."""

    @staticmethod
    def _prompt_with_audio(rid="r0"):
        return {
            "request_id": rid,
            "additional_information": {
                proc.ALIGNER_AUDIO_KEY: (np.zeros(16000, dtype=np.float32), 16000)
            },
        }

    def test_skips_when_transcript_is_empty(self):
        out = proc.asr2aligner([_source_output("")], self._prompt_with_audio())
        assert out == []

    def test_skips_when_audio_is_missing(self):
        out = proc.asr2aligner([_source_output(_raw("hello"))], {"request_id": "r0"})
        assert out == []

    def test_waveform_fallback_carries_words_and_audio(self):
        """Without a rendered prompt there is nothing to splice, so the
        processor hands over the waveform and lets the stage re-encode."""
        out = proc.asr2aligner([_source_output(_raw("hello world"))], self._prompt_with_audio())
        assert len(out) == 1
        assert "multi_modal_data" in out[0]
        assert out[0]["additional_information"]["aligner_words"] == ["hello", "world"]

    def test_words_exclude_the_asr_scaffolding(self):
        out = proc.asr2aligner([_source_output(_raw("hello world"))], self._prompt_with_audio())
        words = out[0]["additional_information"]["aligner_words"]
        assert "language" not in words and "asr" not in words

    def test_prefers_token_splicing_when_available(self):
        """With a rendered prompt the processor reuses stage 0's audio tokens
        so the forwarded mm_features stay valid."""
        spliced = {"prompt_token_ids": [1, 2, 3], "additional_information": {"aligner_words": ["hi"]}}
        with patch.object(proc, "_tokens_input", return_value=spliced) as m:
            out = proc.asr2aligner([_source_output(_raw("hi"))], self._prompt_with_audio())
        assert m.called
        assert out == [spliced]
        assert "multi_modal_data" not in out[0]
