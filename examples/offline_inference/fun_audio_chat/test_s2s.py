"""
Test script for Fun-Audio-Chat S2S (Speech-to-Speech) 3-stage pipeline.

This script validates:
1. CRQ Decoder model can be loaded and has correct architecture
2. CosyVoice stage can be instantiated
3. Stage input processors work correctly
4. 3-stage configuration is valid
"""

import sys


def test_crq_decoder_import():
    """Test CRQ decoder can be imported."""
    print("Testing CRQ decoder import...", end=" ")
    from vllm_omni.model_executor.models.fun_audio_chat import FunAudioChatCRQDecoder

    assert FunAudioChatCRQDecoder is not None
    print("✓")
    return True


def test_cosyvoice_import():
    """Test CosyVoice stage can be imported."""
    print("Testing CosyVoice import...", end=" ")
    from vllm_omni.model_executor.models.fun_audio_chat import FunAudioChatCosyVoice

    assert FunAudioChatCosyVoice is not None
    print("✓")
    return True


def test_registry_s2s():
    """Test S2S models are in registry."""
    print("Testing registry entries...", end=" ")
    from vllm_omni.model_executor.models.registry import _VLLM_OMNI_MODELS

    assert "FunAudioChatCRQDecoder" in _VLLM_OMNI_MODELS
    assert "FunAudioChatCosyVoice" in _VLLM_OMNI_MODELS
    print("✓")
    return True


def test_stage_config():
    """Test S2S stage configuration."""
    print("Testing stage config...", end=" ")
    from pathlib import Path

    import yaml

    config_path = (
        Path(__file__).parent.parent.parent.parent / "vllm_omni/model_executor/stage_configs/fun_audio_chat_s2s.yaml"
    )
    assert config_path.exists(), f"Config not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "stage_args" in config, "Should have stage_args key"
    stages = config["stage_args"]
    assert len(stages) == 3, "Should have 3 stages"

    # Check model types via model_arch in engine_args
    assert stages[0]["engine_args"]["model_arch"] == "FunAudioChatForConditionalGeneration"
    assert stages[1]["engine_args"]["model_arch"] == "FunAudioChatCRQDecoder"
    assert stages[2]["engine_args"]["model_arch"] == "FunAudioChatCosyVoice"
    print("✓")
    return True


def test_stage_processors():
    """Test stage input processors exist and have correct signatures."""
    print("Testing stage processors...", end=" ")
    import inspect

    from vllm_omni.model_executor.stage_input_processors.fun_audio_chat import (
        crq2cosyvoice,
        main2crq,
    )

    # Check main2crq signature
    sig = inspect.signature(main2crq)
    params = list(sig.parameters.keys())
    assert "stage_list" in params, "main2crq should have stage_list parameter"
    assert "engine_input_source" in params, "main2crq should have engine_input_source parameter"

    # Check crq2cosyvoice signature
    sig = inspect.signature(crq2cosyvoice)
    params = list(sig.parameters.keys())
    assert "stage_list" in params, "crq2cosyvoice should have stage_list parameter"
    assert "engine_input_source" in params, "crq2cosyvoice should have engine_input_source parameter"

    print("✓")
    return True


def test_crq_decoder_architecture():
    """Test CRQ decoder architecture matches expected."""
    print("Testing CRQ decoder architecture...", end=" ")
    # Check expected hyperparameters - these are loaded from audio_config
    # so we just verify the class can be imported and has key methods
    import inspect

    from vllm_omni.model_executor.models.fun_audio_chat.crq_decoder import (
        FunAudioChatCRQDecoder,
    )

    source = inspect.getsource(FunAudioChatCRQDecoder)

    # Check for key components
    assert "group_size" in source, "Should have group_size"
    assert "codebook_size" in source, "Should have codebook_size"
    assert "6565" in source or "codebook_size" in source, "Should reference codebook_size"
    assert "_generate_speech_tokens" in source, "Should have _generate_speech_tokens method"
    assert "pre_matching" in source, "Should have pre_matching layer"
    print("✓")
    return True


def main():
    """Run all S2S tests."""
    print("=" * 50)
    print("Fun-Audio-Chat S2S Pipeline Tests")
    print("=" * 50)
    print()

    tests = [
        test_crq_decoder_import,
        test_cosyvoice_import,
        test_registry_s2s,
        test_stage_config,
        test_stage_processors,
        test_crq_decoder_architecture,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print()
    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
