"""Tests for speaker metadata (created_at, consent, ref_text, ...) round-tripping
through the safetensors header.

These cover the helpers that back ``_restore_uploaded_speakers`` — the logic
that rebuilds ``uploaded_speakers`` on server start by reading the
``.safetensors`` file written during upload.
"""

import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSpeakerMetadataRoundTrip:
    def test_str_only_header(self):
        """The safetensors metadata dict must be ``dict[str, str]`` — every value
        serialized as a string (ints and None handled)."""
        data = {
            "name": "Alice",
            "voice_name_lower": "alice",
            "consent": "xxx",
            "created_at": 1712345678,
            "sample_rate": 24000,
            "embedding_source": "audio",
            "ref_text": None,  # None values must be dropped
            "file_path": "/tmp/should/be/stripped.safetensors",  # re-derived, not persisted
        }
        header = OmniOpenAIServingSpeech._speaker_metadata_to_header(data)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in header.items())
        assert "ref_text" not in header  # None stripped
        assert "file_path" not in header  # never persisted

    def test_int_fields_coerce_back(self):
        """Int fields (created_at, file_size, sample_rate, embedding_dim) survive
        the string round-trip with their type preserved."""
        data = {
            "name": "Bob",
            "voice_name_lower": "bob",
            "consent": "yyy",
            "created_at": 1712345678,
            "sample_rate": 44100,
            "file_size": 12345,
            "embedding_dim": 1024,
            "embedding_source": "direct",
        }
        header = OmniOpenAIServingSpeech._speaker_metadata_to_header(data)
        back = OmniOpenAIServingSpeech._speaker_metadata_from_header(header, "/some/path.safetensors")
        assert isinstance(back["created_at"], int)
        assert back["created_at"] == 1712345678
        assert isinstance(back["sample_rate"], int)
        assert back["sample_rate"] == 44100
        assert isinstance(back["file_size"], int)
        assert isinstance(back["embedding_dim"], int)

    def test_file_path_reinjected_on_load(self):
        """file_path is not persisted in the header; restore derives it from the
        actual file location on disk."""
        data = {
            "name": "Carol",
            "voice_name_lower": "carol",
            "consent": "zzz",
            "created_at": 1234,
            "embedding_source": "audio",
        }
        header = OmniOpenAIServingSpeech._speaker_metadata_to_header(data)
        back = OmniOpenAIServingSpeech._speaker_metadata_from_header(header, "/real/path.safetensors")
        assert back["file_path"] == "/real/path.safetensors"

    def test_string_fields_preserved(self):
        """ref_text, consent, speaker_description must survive unchanged."""
        data = {
            "name": "Dave",
            "voice_name_lower": "dave",
            "consent": "consent-id-42",
            "created_at": 1,
            "ref_text": "Hello. This is a transcript with punctuation!",
            "speaker_description": "A warm baritone voice.",
            "embedding_source": "audio",
        }
        header = OmniOpenAIServingSpeech._speaker_metadata_to_header(data)
        back = OmniOpenAIServingSpeech._speaker_metadata_from_header(header, "/x.safetensors")
        assert back["ref_text"] == data["ref_text"]
        assert back["speaker_description"] == data["speaker_description"]
        assert back["consent"] == data["consent"]

    def test_malformed_int_is_left_as_string(self):
        """If an int field somehow contains non-numeric text (manual edit,
        corruption), the loader does not crash; it leaves the field as-is."""
        header = {
            "name": "Eve",
            "voice_name_lower": "eve",
            "consent": "x",
            "created_at": "not-a-number",
            "embedding_source": "audio",
        }
        back = OmniOpenAIServingSpeech._speaker_metadata_from_header(header, "/p.safetensors")
        # Preserved as string rather than raising.
        assert back["created_at"] == "not-a-number"
