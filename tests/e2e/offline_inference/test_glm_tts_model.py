import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.outputs import OmniRequestOutput

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# GLM-TTS model from HuggingFace
# Note: For CI testing, a smaller random-weights model could be created
models = ["zai-org/GLM-TTS"]


@pytest.mark.skip(reason="GLM-TTS model requires full download; enable when model available locally")
@pytest.mark.parametrize("model_name", models)
def test_glm_tts_model(model_name: str):
    """Test GLM-TTS text-to-speech generation."""
    m = Omni(model=model_name)

    # GLM-TTS parameters
    audio_duration_s = 5.0  # 5 second audio

    # Generate speech tokens placeholder (in production, LLM generates these)
    # For testing, we provide mock tokens
    speech_tokens = torch.randint(0, 10000, (1, 100), dtype=torch.long)
    speaker_embedding = torch.randn(1, 192)

    outputs = m.generate(
        "Hello, this is a test of GLM-TTS text to speech synthesis.",
        num_inference_steps=8,  # Minimal steps for speed
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=1,
        extra={
            "audio_duration_s": audio_duration_s,
            "speech_tokens": speech_tokens,
            "speaker_embedding": speaker_embedding,
        },
    )

    # Verify output structure
    assert outputs is not None
    first_output = outputs[0]
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output[0]
    assert isinstance(req_out, OmniRequestOutput)
    assert hasattr(req_out, "images") and len(req_out.images) >= 1

    # For TTS, the "images" field contains audio numpy arrays
    audio = req_out.images[0]
    assert isinstance(audio, np.ndarray)
    # audio shape: (batch, channels, samples) or (channels, samples)
    assert audio.ndim >= 2


@pytest.mark.skip(reason="Unit test for pipeline components")
def test_glm_tts_dit_model():
    """Test GLM-TTS DiT model forward pass."""
    from vllm_omni.diffusion.models.glm_tts.glm_tts_dit import GLMTTSDiTModel

    # Create model with default config
    model = GLMTTSDiTModel(
        hidden_size=256,  # Small for testing
        num_attention_heads=4,
        num_hidden_layers=2,
        head_dim=64,
        mel_dim=80,
        speech_token_dim=128,
        speech_token_vocab_size=1000,
        speaker_embed_dim=64,
    ).cuda()

    # Test forward pass
    batch_size = 2
    seq_len = 100
    token_len = 50

    noisy_mel = torch.randn(batch_size, seq_len, 80).cuda()
    timestep = torch.rand(batch_size).cuda()
    speech_tokens = torch.randint(0, 1000, (batch_size, token_len)).cuda()
    speaker_embedding = torch.randn(batch_size, 64).cuda()

    output = model(
        noisy_mel=noisy_mel,
        timestep=timestep,
        speech_tokens=speech_tokens,
        speaker_embedding=speaker_embedding,
    )

    assert output.shape == (batch_size, seq_len, 80)


@pytest.mark.skip(reason="Unit test for stage input processor")
def test_glm_tts_stage_input_processor():
    """Test GLM-TTS stage input processor."""
    from vllm_omni.model_executor.stage_input_processors.glm_tts import (
        GLM_TTS_AUDIO_TOKEN_END,
        GLM_TTS_AUDIO_TOKEN_START,
        extract_speech_tokens,
    )

    # Test speech token extraction
    # Create mock token IDs with some audio tokens
    token_ids = [
        100,  # Non-audio token
        GLM_TTS_AUDIO_TOKEN_START,  # First audio token (should become 0)
        GLM_TTS_AUDIO_TOKEN_START + 100,  # Audio token (should become 100)
        200,  # Non-audio token
        GLM_TTS_AUDIO_TOKEN_END,  # Last audio token
    ]

    speech_tokens = extract_speech_tokens(token_ids)

    # Should have 3 audio tokens (normalized to 0-based)
    assert len(speech_tokens) == 3
    assert speech_tokens[0] == 0  # First audio token
    assert speech_tokens[1] == 100
    assert speech_tokens[2] == GLM_TTS_AUDIO_TOKEN_END - GLM_TTS_AUDIO_TOKEN_START


@pytest.mark.skip(reason="Integration test for two-stage pipeline")
def test_glm_tts_two_stage_pipeline():
    """Test GLM-TTS two-stage pipeline (LLM + DiT)."""
    # This test requires:
    # 1. GLM-TTS LLM model weights
    # 2. GLM-TTS DiT model weights
    # 3. Stage config file

    from vllm_omni import Omni

    # Load with two-stage config
    m = Omni(
        model="zai-org/GLM-TTS",
        stage_config="vllm_omni/model_executor/stage_configs/glm_tts.yaml",
    )

    outputs = m.generate(
        "Hello, this is a test of GLM-TTS.",
        num_inference_steps=8,
        seed=42,
    )

    assert outputs is not None
    first_output = outputs[0]
    assert hasattr(first_output, "request_output")

    # Audio should be in images field (X2S pattern)
    req_out = first_output.request_output[0]
    audio = req_out.images[0]
    assert isinstance(audio, np.ndarray)
