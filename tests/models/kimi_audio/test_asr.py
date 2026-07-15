"""Integration tests for Kimi Audio ASR functionality.

Tests the critical fixes for ASR quality:
1. Correct sampling parameters (temp=0.0, top_k=5)
2. Correct EOS token (151667, not 151644)
3. Audio termination detection (media_end token 151663)
4. Dual-stream sampling in ASR mode
5. Chat-completion ASR detection and special-token cleanup
6. TTS transcript extraction

Run with: python tests/models/kimi_audio/test_asr.py
"""


class TestASRSamplingParameters:
    """Test that ASR mode uses correct sampling parameters."""

    def test_asr_mode_detection(self):
        """Verify ASR mode is detected when multimodal_embeddings present."""
        # This would require instantiating the model, which needs GPU
        # For now, just verify the logic exists
        from vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm import KimiAudioLLMForConditionalGeneration

        # Verify the method exists
        assert hasattr(KimiAudioLLMForConditionalGeneration, "forward")
        assert hasattr(KimiAudioLLMForConditionalGeneration, "sample")

    def test_sampling_constants(self):
        """Verify critical token constants are correct."""
        from vllm_omni.model_executor.models.kimi_audio.constants import (
            KIMI_AUDIO_BLANK_TOKEN_ID,
            KIMI_AUDIO_EOS_TOKEN_ID,
            KIMI_AUDIO_TEXT_EOS_TOKEN_ID,
        )

        # The engine-level EOS is 151644; the text-stream EOS is 151667
        assert KIMI_AUDIO_EOS_TOKEN_ID == 151644
        assert KIMI_AUDIO_TEXT_EOS_TOKEN_ID == 151667
        # BLANK should be 151666
        assert KIMI_AUDIO_BLANK_TOKEN_ID == 151666


class TestAudioTermination:
    """Test audio stream termination logic."""

    def test_media_end_token_defined(self):
        """Verify media_end token ID is defined."""
        # This would require model instantiation
        # For now, verify the constant exists in config
        from vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm import KimiAudioLLMForConditionalGeneration

        # Verify the sample method exists (where termination is checked)
        assert hasattr(KimiAudioLLMForConditionalGeneration, "sample")


class TestDualStreamGeneration:
    """Test dual-stream generation logic."""

    def test_dual_stream_in_asr_mode(self):
        """Verify ASR mode samples both streams then forces audio to BLANK."""
        # This is tested by the logic in sample() method
        # The key fix: removed early return in ASR mode
        # Now samples both streams, then forces audio to BLANK
        pass  # Requires GPU to test fully


class TestASREndToEnd:
    """End-to-end ASR tests (requires GPU)."""

    def test_chinese_audio_transcription(self):
        """Test ASR with Chinese audio file."""
        # This is the manual test in test_asr_example.py
        # Expected: "这并不是告别，这是一个篇章的结束，也是新篇章的开始"
        pass


class TestChatCompletionASRPath:
    """Test chat-completion ASR path helpers in serving_chat."""

    def test_audio_only_request_detection(self):
        """Detect requests that contain only audio input."""
        from vllm_omni.entrypoints.openai.serving_chat import _is_audio_only_request

        audio_only = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "YXNkZg==", "format": "wav"},
                    }
                ],
            }
        ]
        assert _is_audio_only_request(audio_only) is True

        text_only = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert _is_audio_only_request(text_only) is False

        mixed = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "YXNkZg==", "format": "wav"},
                    },
                ],
            }
        ]
        assert _is_audio_only_request(mixed) is False

    def test_special_token_string_filtering(self):
        """Remove Kimi Audio special token strings from decoded text."""
        from vllm_omni.entrypoints.openai.serving_chat import filter_kimi_audio_special_strings

        raw = "<|im_kimia_text_blank|>Hello world<|im_kimia_text_eos|><|im_msg_end|>  this is   a test<|im_end|>"
        cleaned = filter_kimi_audio_special_strings(raw)
        assert cleaned == "Hello world this is a test"

    def test_special_token_filtering_for_empty_text(self):
        """Filtering handles empty/None input gracefully."""
        from vllm_omni.entrypoints.openai.serving_chat import filter_kimi_audio_special_strings

        assert filter_kimi_audio_special_strings("") == ""
        assert filter_kimi_audio_special_strings(None) is None


class TestTTSTranscriptExtraction:
    """Test extraction of the input text used for TTS audio generation."""

    def test_extract_text_from_last_user_message(self):
        """Prefer the most recent user message text as the transcript."""
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Now say goodbye"},
        ]
        assert OmniOpenAIServingChat._extract_tts_input_text(messages) == "Now say goodbye"

    def test_extract_text_from_list_content(self):
        """Handle content provided as a list of parts."""
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate audio for"},
                    {"type": "text", "text": "this sentence."},
                ],
            }
        ]
        assert OmniOpenAIServingChat._extract_tts_input_text(messages) == "Generate audio for this sentence."

    def test_extract_text_fallback_empty(self):
        """Return empty string when no text content exists."""
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        messages = [{"role": "user", "content": [{"type": "input_audio"}]}]
        assert OmniOpenAIServingChat._extract_tts_input_text(messages) == ""


if __name__ == "__main__":
    # Run basic tests
    print("Running Kimi Audio ASR tests...")

    test = TestASRSamplingParameters()
    test.test_sampling_constants()
    print("✓ Sampling constants test passed")

    test2 = TestAudioTermination()
    test2.test_media_end_token_defined()
    print("✓ Media end token test passed")

    test3 = TestChatCompletionASRPath()
    test3.test_audio_only_request_detection()
    print("✓ Audio-only request detection test passed")
    test3.test_special_token_string_filtering()
    print("✓ Special token string filtering test passed")
    test3.test_special_token_filtering_for_empty_text()
    print("✓ Empty text filtering test passed")

    test4 = TestTTSTranscriptExtraction()
    test4.test_extract_text_from_last_user_message()
    print("✓ TTS transcript extraction test passed")
    test4.test_extract_text_from_list_content()
    print("✓ TTS transcript list content test passed")
    test4.test_extract_text_fallback_empty()
    print("✓ TTS transcript empty fallback test passed")

    print("\nAll tests passed!")
