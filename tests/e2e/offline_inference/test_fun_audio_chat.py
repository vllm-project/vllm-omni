# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Fun-Audio-Chat processor and model components.
Run with: pytest tests/e2e/offline_inference/test_fun_audio_chat.py -v
"""

import os

import numpy as np
import pytest
import torch


def get_model_path() -> str | None:
    """Find the Fun-Audio-Chat model path if available."""
    # Try common locations
    candidates = [
        "./pretrained_models/Fun-Audio-Chat-8B",
        "../pretrained_models/Fun-Audio-Chat-8B",
        "../../pretrained_models/Fun-Audio-Chat-8B",
        "../../../pretrained_models/Fun-Audio-Chat-8B",
        os.path.expanduser("~/pretrained_models/Fun-Audio-Chat-8B"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    return None


@pytest.fixture
def model_path():
    """Get model path or skip test."""
    path = get_model_path()
    if path is None:
        pytest.skip("Fun-Audio-Chat-8B model not found")
    return path


@pytest.fixture
def dummy_audio():
    """Create dummy audio for testing."""
    # 5 seconds of audio at 16kHz
    sample_rate = 16000
    duration = 5
    audio = np.random.randn(sample_rate * duration).astype(np.float32)
    return audio


class TestFunAudioChatProcessor:
    """Test the Fun-Audio-Chat processor."""

    def test_processor_import(self):
        """Test that processor can be imported."""
        from vllm_omni.model_executor.models.fun_audio_chat.processing_fun_audio_chat import (
            FunAudioChatProcessor,
        )

        assert FunAudioChatProcessor is not None

    def test_processor_load(self, model_path):
        """Test that processor can be loaded from model path."""
        from vllm_omni.model_executor.models.fun_audio_chat.processing_fun_audio_chat import (
            FunAudioChatProcessor,
        )

        processor = FunAudioChatProcessor.from_pretrained(model_path, trust_remote_code=True)
        assert processor is not None
        assert hasattr(processor, "audio_token")
        assert processor.audio_token == "<|AUDIO|>"

    def test_processor_text_only(self, model_path):
        """Test processor with text-only input."""
        from vllm_omni.model_executor.models.fun_audio_chat.processing_fun_audio_chat import (
            FunAudioChatProcessor,
        )

        processor = FunAudioChatProcessor.from_pretrained(model_path, trust_remote_code=True)
        output = processor(text="Hello, how are you?", return_tensors="pt")
        assert "input_ids" in output
        assert isinstance(output["input_ids"], torch.Tensor)

    def test_processor_with_audio(self, model_path, dummy_audio):
        """Test processor with audio input."""
        from vllm_omni.model_executor.models.fun_audio_chat.processing_fun_audio_chat import (
            FunAudioChatProcessor,
        )

        processor = FunAudioChatProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Process audio with prompt containing audio token
        text = "<|AUDIO|> What is in this audio?"
        output = processor(
            text=text,
            audios=[dummy_audio],
            return_tensors="pt",
            padding=True,
        )

        assert "input_ids" in output
        # Should have audio-related outputs
        assert "input_features" in output or "speech_ids" in output, (
            f"Missing audio features. Keys: {list(output.keys())}"
        )

        # Check shapes
        if "input_features" in output:
            input_features = output["input_features"]
            print(f"input_features shape: {input_features.shape}")
            # Should be [batch, num_mel_bins, seq_len] or similar
            assert input_features.dim() == 3, f"Expected 3D tensor, got {input_features.dim()}D"

    def test_processor_multiple_audios(self, model_path, dummy_audio):
        """Test processor with multiple audio inputs."""
        from vllm_omni.model_executor.models.fun_audio_chat.processing_fun_audio_chat import (
            FunAudioChatProcessor,
        )

        processor = FunAudioChatProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Multiple audios (though Fun-Audio-Chat may only support 1)
        text = "<|AUDIO|> What is in this audio?"
        output = processor(
            text=text,
            audios=[dummy_audio, dummy_audio],
            return_tensors="pt",
            padding=True,
        )

        print(f"Output keys: {list(output.keys())}")
        if "input_features" in output:
            print(f"input_features shape: {output['input_features'].shape}")
        if "speech_ids" in output:
            print(f"speech_ids shape: {output['speech_ids'].shape}")


class TestFunAudioChatFieldConfig:
    """Test field configuration for batching."""

    def test_field_config_structure(self):
        """Test that field config is properly defined."""
        from vllm_omni.model_executor.models.fun_audio_chat.fun_audio_chat import (
            _funaudiochat_field_config,
        )

        # Simulate processor output
        hf_inputs = {
            "input_features": torch.randn(1, 128, 3000),
            "feature_attention_mask": torch.ones(1, 3000),
            "speech_ids": torch.randint(0, 1000, (1, 100)),
            "speech_attention_mask": torch.ones(1, 100),
        }

        config = _funaudiochat_field_config(hf_inputs)

        assert "input_features" in config
        assert "feature_attention_mask" in config
        assert "speech_ids" in config
        assert "speech_attention_mask" in config


class TestTensorShapes:
    """Test tensor shape handling for profiling."""

    def test_batched_audio_stacking(self, model_path, dummy_audio):
        """Test that stacked audio tensors maintain correct dimensions."""
        from vllm_omni.model_executor.models.fun_audio_chat.processing_fun_audio_chat import (
            FunAudioChatProcessor,
        )

        processor = FunAudioChatProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Process single audio
        text = "<|AUDIO|> What is in this audio?"
        output = processor(
            text=text,
            audios=[dummy_audio],
            return_tensors="pt",
            padding=True,
        )

        if "input_features" in output:
            single_features = output["input_features"]
            print(f"Single audio features shape: {single_features.shape}")

            # Simulate what vLLM does during profiling (stacking 10 items)
            # This should NOT create a 4D tensor
            stacked = torch.stack([single_features[0]] * 10)
            print(f"Stacked (10 items) shape: {stacked.shape}")

            # Stacked should be 3D: [10, num_mel_bins, seq_len]
            assert stacked.dim() == 3, f"Stacked tensor should be 3D, got {stacked.dim()}D: {stacked.shape}"


if __name__ == "__main__":
    # Run basic tests
    print("Testing Fun-Audio-Chat components...")

    model_path = get_model_path()
    if model_path:
        print(f"Found model at: {model_path}")
    else:
        print("Model not found. Some tests will be skipped.")

    pytest.main([__file__, "-v", "-x"])
