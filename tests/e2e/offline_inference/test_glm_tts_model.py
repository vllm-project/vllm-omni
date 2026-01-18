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
    sample_rate = 22050  # GLM-TTS default sample rate

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


@pytest.mark.skip(reason="Unit test for flow sampler")
def test_glm_tts_flow_sampler():
    """Test GLM-TTS flow matching sampler."""
    from vllm_omni.diffusion.models.glm_tts.glm_tts_dit import GLMTTSDiTModel
    from vllm_omni.diffusion.models.glm_tts.pipeline_glm_tts import GLMTTSFlowSampler

    # Create small model for testing
    model = GLMTTSDiTModel(
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        head_dim=64,
        mel_dim=80,
        speech_token_dim=128,
        speech_token_vocab_size=1000,
        speaker_embed_dim=64,
    ).cuda()

    sampler = GLMTTSFlowSampler(num_steps=4, cfg_scale=1.0)

    batch_size = 1
    seq_len = 50
    token_len = 25

    speech_tokens = torch.randint(0, 1000, (batch_size, token_len)).cuda()
    speaker_embedding = torch.randn(batch_size, 64).cuda()

    mel = sampler.sample(
        model=model,
        shape=(batch_size, seq_len, 80),
        speech_tokens=speech_tokens,
        speaker_embedding=speaker_embedding,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    assert mel.shape == (batch_size, seq_len, 80)
