#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for Fun-Audio-Chat S2T inference with vLLM-Omni.

This script tests Speech-to-Text inference using the Fun-Audio-Chat model.
It verifies that the model can be loaded and process audio inputs.

Usage:
    python test_s2t.py [--model-path MODEL_PATH] [--audio-path AUDIO_PATH]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_model_import():
    """Test that model classes can be imported."""
    print("=" * 60)
    print("Test 1: Import model classes")
    print("=" * 60)

    try:
        from vllm_omni.model_executor.models.fun_audio_chat import (
            FunAudioChatAudioEncoder,
            FunAudioChatDiscreteEncoder,
            FunAudioChatForConditionalGeneration,
        )

        # Use the imports to avoid F401 unused import warning
        assert FunAudioChatAudioEncoder is not None
        assert FunAudioChatDiscreteEncoder is not None
        assert FunAudioChatForConditionalGeneration is not None

        print("✓ FunAudioChatAudioEncoder imported")
        print("✓ FunAudioChatDiscreteEncoder imported")
        print("✓ FunAudioChatForConditionalGeneration imported")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_registry():
    """Test that model is registered in the registry."""
    print("\n" + "=" * 60)
    print("Test 2: Check model registry")
    print("=" * 60)

    try:
        from vllm_omni.model_executor.models.registry import OmniModelRegistry

        # Check if FunAudioChat is registered
        if "FunAudioChatForConditionalGeneration" in OmniModelRegistry.models:
            print("✓ FunAudioChatForConditionalGeneration is registered")
            return True
        else:
            print("✗ FunAudioChatForConditionalGeneration not found in registry")
            return False
    except ImportError as e:
        print(f"✗ Registry import failed: {e}")
        return False


def test_audio_encoder_standalone():
    """Test audio encoder modules can be instantiated."""
    print("\n" + "=" * 60)
    print("Test 3: Instantiate audio encoders")
    print("=" * 60)

    try:
        import torch

        from vllm_omni.model_executor.models.fun_audio_chat.audio_encoder import (
            FunAudioChatAudioEncoder,
            FunAudioChatDiscreteEncoder,
        )

        # Create a mock config
        class MockAudioConfig:
            num_mel_bins = 128
            d_model = 1280
            encoder_layers = 2  # Use fewer layers for testing
            encoder_attention_heads = 20
            encoder_ffn_dim = 5120
            output_dim = 3584
            n_window = 100
            max_source_positions = 1500
            dropout = 0.0
            attention_dropout = 0.0
            activation_function = "gelu"
            scale_embedding = False
            codebook_size = 6565
            group_size = 5
            pad_token_id = 0
            continuous_features_mode = "add"

        config = MockAudioConfig()

        # Test continuous audio encoder
        print("Creating FunAudioChatAudioEncoder...")
        continuous_encoder = FunAudioChatAudioEncoder(config)
        print(f"  - Encoder layers: {len(continuous_encoder.layers)}")
        print(f"  - Output dim: {continuous_encoder.output_dim}")
        print("✓ FunAudioChatAudioEncoder created")

        # Test discrete audio encoder
        print("Creating FunAudioChatDiscreteEncoder...")
        discrete_encoder = FunAudioChatDiscreteEncoder(config)
        print(f"  - Codebook size: {discrete_encoder.codebook_size}")
        print(f"  - Group size: {discrete_encoder.group_size}")
        print("✓ FunAudioChatDiscreteEncoder created")

        # Test forward pass with dummy data
        print("Testing forward pass with dummy data...")
        batch_size = 2
        seq_len = 100
        num_mel_bins = 128

        # Continuous encoder forward
        dummy_mel = torch.randn(batch_size, num_mel_bins, seq_len)
        feature_lens = torch.tensor([seq_len, seq_len // 2])
        output = continuous_encoder(dummy_mel, feature_lens=feature_lens)
        print(f"  - Continuous encoder output shape: {output.shape}")
        print("✓ Continuous encoder forward pass succeeded")

        # Discrete encoder forward
        dummy_tokens = torch.randint(0, config.codebook_size, (batch_size, 50))
        output = discrete_encoder(dummy_tokens)
        print(f"  - Discrete encoder output shape: {output.shape}")
        print("✓ Discrete encoder forward pass succeeded")

        return True
    except Exception as e:
        print(f"✗ Audio encoder test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_hf_processor(model_path: str):
    """Test loading HuggingFace processor."""
    print("\n" + "=" * 60)
    print("Test 4: Load HuggingFace processor")
    print("=" * 60)

    try:
        from transformers import AutoProcessor

        print(f"Loading processor from {model_path}...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Processor loaded: {type(processor).__name__}")

        # Test tokenization - processor may be tokenizer directly or have tokenizer attribute
        tokenizer = getattr(processor, "tokenizer", processor)
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization works: {tokens['input_ids'].shape}")

        return True
    except Exception as e:
        print(f"✗ Processor test failed: {e}")
        return False


def test_hf_config(model_path: str):
    """Test loading HuggingFace config."""
    print("\n" + "=" * 60)
    print("Test 5: Load HuggingFace config")
    print("=" * 60)

    try:
        import json

        from huggingface_hub import hf_hub_download

        print(f"Loading config from {model_path}...")

        # Download and load config.json directly
        config_path = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        print(f"✓ Config loaded from: {config_path}")

        # Print key config values
        print(f"  - Model type: {config_dict.get('model_type', 'N/A')}")
        print(f"  - Audio token index: {config_dict.get('audio_token_index', 'N/A')}")

        text_config = config_dict.get("text_config", {})
        if text_config:
            print(f"  - Text model type: {text_config.get('model_type', 'N/A')}")
            print(f"  - Vocab size: {text_config.get('vocab_size', 'N/A')}")
            print(f"  - Hidden size: {text_config.get('hidden_size', 'N/A')}")

        audio_config = config_dict.get("audio_config", {})
        if audio_config:
            print(f"  - Audio encoder layers: {audio_config.get('encoder_layers', 'N/A')}")
            print(f"  - Audio d_model: {audio_config.get('d_model', 'N/A')}")
            print(f"  - Group size: {audio_config.get('group_size', 'N/A')}")
            print(f"  - Codebook size: {audio_config.get('codebook_size', 'N/A')}")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Fun-Audio-Chat S2T inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="FunAudioLLM/Fun-Audio-Chat-8B",
        help="Path to the Fun-Audio-Chat model",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to test audio file (optional)",
    )
    parser.add_argument(
        "--skip-hf-tests",
        action="store_true",
        help="Skip tests that require downloading from HuggingFace",
    )
    args = parser.parse_args()

    print("Fun-Audio-Chat S2T Test Suite")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Audio path: {args.audio_path or 'None (skipping audio tests)'}")
    print()

    results = {}

    # Test 1: Import
    results["import"] = test_model_import()

    # Test 2: Registry
    results["registry"] = test_registry()

    # Test 3: Audio encoders
    results["audio_encoders"] = test_audio_encoder_standalone()

    # Test 4 & 5: HuggingFace tests (optional)
    if not args.skip_hf_tests:
        results["hf_processor"] = test_hf_processor(args.model_path)
        results["hf_config"] = test_hf_config(args.model_path)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed. ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
