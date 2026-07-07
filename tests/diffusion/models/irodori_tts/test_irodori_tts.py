# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from tests.helpers.media import get_asset_path
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.irodori_tts.irodori_tts_transformer import IrodoriTTSTransformer, ModelConfig
from vllm_omni.diffusion.models.irodori_tts.pipeline_irodori_tts import IrodoriTTSPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def test_irodori_transformer_forward():
    # Initialize a lightweight configuration for fast unit testing
    cfg = ModelConfig(
        model_dim=128,  # small model dim
        num_layers=2,  # lightweight layers
        num_heads=4,  # divisible heads
        text_dim=64,
        text_layers=2,
        text_heads=2,
        speaker_dim=64,
        speaker_layers=2,
        speaker_heads=2,
        use_speaker_condition=True,
        use_duration_predictor=True,
        duration_aux_dim=14,
        duration_hidden_dim=64,
        duration_layers=2,
    )

    model = IrodoriTTSTransformer(cfg=cfg).eval()

    # Define shape parameters
    batch_size = 2
    latent_seq_len = 32
    text_seq_len = 16
    ref_seq_len = 24

    # Generate mock inputs
    x_t = torch.randn(batch_size, latent_seq_len, cfg.patched_latent_dim)
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)
    text_input_ids = torch.randint(0, cfg.text_vocab_size, (batch_size, text_seq_len))
    text_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool)

    ref_latent = torch.randn(batch_size, ref_seq_len, cfg.patched_latent_dim)
    ref_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool)

    duration_features = torch.randn(batch_size, cfg.duration_aux_dim)
    duration_has_speaker = torch.ones(batch_size, dtype=torch.bool)
    duration_has_caption = torch.zeros(batch_size, dtype=torch.bool)

    # Test 1: Standard RF Forward
    with torch.no_grad():
        v_pred = model(
            x_t=x_t,
            t=t,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
        )
    assert v_pred.shape == x_t.shape, f"Expected shape {x_t.shape}, got {v_pred.shape}"

    # Test 2: Forward with Duration Predictor
    with torch.no_grad():
        v_pred_dur, duration_pred = model(
            x_t=x_t,
            t=t,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            duration_features=duration_features,
            duration_has_speaker=duration_has_speaker,
            duration_has_caption=duration_has_caption,
        )
    assert v_pred_dur.shape == x_t.shape, f"Expected shape {x_t.shape}, got {v_pred_dur.shape}"
    assert duration_pred.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {duration_pred.shape}"

    # Test 3: Duration-only Prediction
    with torch.no_grad():
        duration_pred_only = model(
            x_t=None,
            t=None,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            duration_features=duration_features,
            duration_has_speaker=duration_has_speaker,
            duration_has_caption=duration_has_caption,
            duration_only=True,
        )
    assert duration_pred_only.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {duration_pred_only.shape}"


class MockTFModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.quant_config = None


def test_irodori_pipeline_forward():
    # Initialize a lightweight configuration for fast integration testing
    tf_model_config = MockTFModelConfig(
        latent_dim=32,  # matches DACVAE 32-dim latent space
        model_dim=128,
        num_layers=2,
        num_heads=4,
        text_dim=64,
        text_layers=2,
        text_heads=2,
        speaker_dim=64,
        speaker_layers=2,
        speaker_heads=2,
        use_speaker_condition=True,
        use_duration_predictor=True,
        duration_aux_dim=14,
        duration_hidden_dim=64,
        duration_layers=2,
    )

    od_config = OmniDiffusionConfig(
        model="Aratako/Irodori-TTS-500M-v3",
        tf_model_config=tf_model_config,
        dtype=torch.float32,  # use float32 for stable CPU testing if executed on CPU
    )

    print("Instantiating IrodoriTTSPipeline...")
    pipeline = IrodoriTTSPipeline(od_config=od_config).eval()

    # Define simple Japanese text prompt for testing
    prompt = "こんにちは、私はGeorgeです。"

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=2,  # 2 steps for super-fast test execution
        seed=42,
        extra_args={
            "ref_wav": str(get_asset_path("qwen3_tts/clone_2.wav")),
            "duration_scale": 1.0,
            "cfg_scale_text": 1.5,
            "cfg_scale_speaker": 2.0,
        },
    )

    req = OmniDiffusionRequest(
        request_id="test-pipeline-req",
        prompts=[prompt],
        sampling_params=sampling_params,
    )

    print("Executing pipeline.forward()...")
    with torch.no_grad():
        output = pipeline(req)

    print(f"Pipeline output waveform shape: {tuple(output.output.shape)}")

    # Assert mono waveform output has 3 dimensions (B, 1, samples)
    assert output.output.ndim == 3, f"Expected 3D waveform tensor, got shape {output.output.shape}"
    assert output.output.shape[0] == 1, f"Expected batch size 1, got {output.output.shape[0]}"
    assert output.output.shape[1] == 1, f"Expected mono audio channel dimension 1, got {output.output.shape[1]}"
    assert output.output.shape[2] > 0, "Expected generated samples count > 0"

    # Test post-processing function
    from vllm_omni.diffusion.models.irodori_tts.pipeline_irodori_tts import get_irodori_tts_post_process_func

    print("Testing post-processing function...")
    post_process_func = get_irodori_tts_post_process_func(od_config)
    audio_np = post_process_func(output.output)
    print(f"Post-processed numpy audio shape: {audio_np.shape}")
    assert audio_np.ndim == 3
    assert audio_np.shape[0] == 1
    assert audio_np.shape[1] == 1

    print("Pipeline end-to-end forward pass and post-processing validated successfully!")
