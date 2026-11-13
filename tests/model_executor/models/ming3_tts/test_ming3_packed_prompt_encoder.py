# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE, AudioVAEConfig

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")

HOP = 16


def _encoder() -> torch.nn.Module:
    backbone = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,  # the aggregator reuses this config with 4 layers
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "sliding_window": 8,
        "use_sliding_window": True,
        "max_window_layers": 0,
        "vocab_size": 1,
        "_attn_implementation": "sdpa",
    }
    config = AudioVAEConfig(
        sample_rate=1600,
        enc_kwargs={"backbone": backbone, "input_dim": HOP, "hop_size": HOP, "latent_dim": 8},
        patch_size=4,
    )
    torch.manual_seed(0)
    return AudioVAE(config, build_decoder=False).eval().encoder


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_packed_prompt_encoder_matches_transformers():
    from vllm_omni.model_executor.models.ming3_tts.packed_prompt_encoder import PackedPromptEncoder

    # Cross the sliding window in both encoders and leave an incomplete patch.
    frames = 37
    encoder = _encoder().cuda()
    waveform = torch.randn(1, frames * HOP, device="cuda")
    with torch.inference_mode():
        reference = encoder(waveform)[0]
        encoder.to(torch.bfloat16)
        transformers_bf16 = encoder(waveform.bfloat16())[0].float()
        packed = PackedPromptEncoder(encoder, max_frames=64)(waveform.bfloat16()).float()

    assert packed.shape == reference.shape == (1, -(-frames // 4), 16)
    # The fused path differs from the fp32 reference no more than transformers' own bf16 path does.
    tolerance = 2 * (transformers_bf16 - reference).abs().max().item() + 1e-3
    assert (packed - reference).abs().max().item() <= tolerance


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("graphs", [False, True])
def test_packed_clips_match_single_clip_encoding(graphs):
    from vllm_omni.model_executor.models.ming3_tts.packed_prompt_encoder import PackedPromptEncoder

    encoder = _encoder().cuda().to(torch.bfloat16)
    packed = PackedPromptEncoder(encoder, max_frames=64)
    if graphs:
        packed.capture_graphs()
    torch.manual_seed(1)
    clips = [torch.randn(1, frames * HOP, device="cuda", dtype=torch.bfloat16) for frames in (37, 3, 8, 21)]
    with torch.inference_mode():
        expected = [packed(clip) for clip in clips]
        actual = packed.encode_clips([clip.cpu() for clip in clips])
        # Cover host and device inputs in successive calls. Replaying the same
        # graph for reordered clips must not overwrite the first result.
        reordered = packed.encode_clips(clips[::-1])
    # Clips share GEMMs but not attention: a leak across clips would move whole rows.
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)
    for got, want in zip(reordered, expected[::-1], strict=True):
        torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)


def _wrapper():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.ming3_tts.ming3_tts import Ming3TTSForConditionalGeneration

    wrapper = object.__new__(Ming3TTSForConditionalGeneration)
    wrapper._prompt_encoder_graphs = None
    wrapper._prompt_projection_graphs = None
    wrapper._prefetched_prompt_outputs = {}
    wrapper._prefetched_speakers = {}
    wrapper._prompt_request_id = None
    wrapper._prompt_cache_key = None
    # Projection stand-in: row-independent, like the Aggregator.
    wrapper.__dict__["model"] = SimpleNamespace(linear_proj_audio=lambda patches: patches.sum(-1, keepdim=True) * 2)
    return wrapper


@pytest.mark.cpu
def test_reference_encoder_output_is_cached_by_content_key():
    from types import SimpleNamespace

    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    wrapper = _wrapper()
    wrapper._speaker_cache = SpeakerEmbeddingCache()
    wrapper.__dict__["vllm_config"] = SimpleNamespace(model_config=SimpleNamespace(model="/model"))
    info = {"prompt_waveform": torch.zeros(1, 64), "speaker_audio_keys": ["clip-a"]}
    key = wrapper._prompt_encoder_cache_key(info)
    assert wrapper._prompt_encoder_cache_key({**info, "speaker_audio_keys": [None]}) is None
    assert wrapper._prompt_encoder_cache_key({"speaker_audio_keys": ["clip-a"]}) is None

    output = torch.randn(1, 4, 8)
    wrapper._prefetched_prompt_outputs["a"] = (64, output)
    waveform = torch.zeros(1, 64)
    wrapper._prompt_request_id = "b"
    assert wrapper._prompt_encoder_forward(None, waveform) is None
    wrapper._prompt_request_id, wrapper._prompt_cache_key = "a", key
    assert wrapper._prompt_encoder_forward(None, waveform) is output
    assert "a" not in wrapper._prefetched_prompt_outputs
    # A later request with the same reference takes a copy from the cache.
    wrapper._prompt_request_id = "b"
    cached = wrapper._prompt_encoder_forward(None, waveform)
    assert cached is not output and torch.equal(cached, output)
    # A different prompt length under the key does not use the entry.
    assert wrapper._prompt_encoder_forward(None, torch.zeros(1, 32)) is None
