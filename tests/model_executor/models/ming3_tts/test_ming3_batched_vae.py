# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE, AudioVAEConfig
from vllm_omni.model_executor.models.ming3_tts.batched_vae_decoder import BatchedStreamingVAEDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")

# Chunk schedules cross the sliding window at different steps per request.
_SCHEDULES = [[4, 8, 16, 5], [1, 2, 4, 8, 3], [6], [3, 3, 3, 3, 3, 3]]
_STARTS = [0, 1, 0, 2]


def _vae(*, packed: bool = False) -> AudioVAE:
    # Packed attention needs 64-dimensional heads; the CPU reference can be smaller.
    backbone = {
        "hidden_size": 128 if packed else 32,
        "intermediate_size": 256 if packed else 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2 if packed else 4,
        "num_key_value_heads": 1 if packed else 2,
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
        dec_kwargs={"backbone": backbone, "output_dim": 16, "latent_dim": 8},
        patch_size=4,
    )
    torch.manual_seed(0)
    vae = AudioVAE(config, build_encoder=False).eval()
    vae.decoder.upsampling.frame_holdback = True
    return vae


def _per_request(vae: AudioVAE, latents: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    past, state, outputs, start = None, (None, None, None), [], 0
    for index, size in enumerate(sizes):
        waveform, state, past = vae.decode(
            latents[start : start + size].reshape(1, -1, latents.shape[-1]),
            past_key_values=past,
            use_cache=True,
            stream_state=state,
            last_chunk=index == len(sizes) - 1,
        )
        outputs.append(waveform.reshape(-1))
        start += size
    return torch.cat(outputs)


def _batched_streams(decoder: BatchedStreamingVAEDecoder, latents: list[torch.Tensor]) -> list[torch.Tensor]:
    """Feed every schedule's chunks step by step, staggered by ``_STARTS``."""
    cursor, chunk = [0] * 4, [0] * 4
    outputs: list[list[torch.Tensor]] = [[] for _ in _SCHEDULES]
    step = 0
    while any(chunk[r] < len(_SCHEDULES[r]) for r in range(4)):
        items, owners = [], []
        for r, sizes in enumerate(_SCHEDULES):
            if step >= _STARTS[r] and chunk[r] < len(sizes):
                size = sizes[chunk[r]]
                items.append((f"req-{r}", latents[r][cursor[r] : cursor[r] + size], chunk[r] == len(sizes) - 1))
                owners.append(r)
                cursor[r] += size
                chunk[r] += 1
        for r, waveform in zip(owners, decoder.decode(items)):
            outputs[r].append(waveform)
        step += 1
    return [torch.cat(output) for output in outputs]


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=[cuda, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]),
    ],
)
def test_batched_decode_matches_per_request_streams(device):
    vae = _vae().to(device)
    torch.manual_seed(1)
    latents = [torch.randn(sum(sizes), 1, 8, device=device) for sizes in _SCHEDULES]
    expected = [_per_request(vae, latent, sizes) for latent, sizes in zip(latents, _SCHEDULES)]

    decoder = BatchedStreamingVAEDecoder(vae.decoder, max_streams=4)
    if device == "cuda":
        decoder.capture_graphs(batch_sizes=(1, 2, 4), lengths=(8, 16, 32, 64))
    buckets = []
    pick = decoder._graph_bucket

    def record_bucket(batch, length):
        bucket = pick(batch, length)
        buckets.append(bucket)
        return bucket

    decoder._graph_bucket = record_bucket
    for actual, reference in zip(_batched_streams(decoder, latents), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    # Replays covered padded batches and lengths, and the scratch slot stayed private.
    if device == "cuda":
        assert any(bucket is not None for bucket in buckets)
    assert sorted(decoder._free) == [0, 1, 2, 3]


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("graphs", [False, True])
def test_packed_decode_matches_per_request_streams(graphs):
    vae = _vae(packed=True).cuda()
    torch.manual_seed(1)
    latents = [torch.randn(sum(sizes), 1, 8, device="cuda") for sizes in _SCHEDULES]
    with torch.inference_mode():
        reference = [_per_request(vae, latent, sizes) for latent, sizes in zip(latents, _SCHEDULES)]
        vae.to(torch.bfloat16)
        latents = [latent.bfloat16() for latent in latents]
        transformers_bf16 = [_per_request(vae, latent, sizes).float() for latent, sizes in zip(latents, _SCHEDULES)]

    # A two-block ring of 16 keys: chunks over 25 decoder tokens take several passes.
    decoder = BatchedStreamingVAEDecoder(vae.decoder, max_streams=4, kv_block_size=16, ring_blocks=2)
    assert decoder._packed is not None
    if graphs:
        decoder.capture_graphs()
    for actual, want, baseline in zip(_batched_streams(decoder, latents), reference, transformers_bf16, strict=True):
        assert actual.shape == want.shape
        # The fused path differs from the fp32 reference no more than transformers' own bf16 path does.
        tolerance = 2 * (baseline - want).abs().max().item() + 1e-3
        assert (actual - want).abs().max().item() <= tolerance
    assert sorted(decoder._free) == [0, 1, 2, 3]


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [3, 8])
def test_short_flow_attention_matches_sdpa(tokens):
    from vllm_omni.model_executor.models.common.ming.dit_blocks import _short_attention

    torch.manual_seed(tokens)
    query, key, value = (torch.randn(5, 4, tokens, 16) for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(_short_attention(query, key, value), expected, rtol=1e-5, atol=1e-5)


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("shape", [(5, 3, 16), (7, 16)])
def test_feed_forward_gelu_epilogue_matches_modules(shape):
    from vllm_omni.model_executor.models.common.ming.dit_blocks import FeedForward

    torch.manual_seed(0)
    ff = FeedForward(16, mult=4, approximate="tanh").eval().cuda()
    ff.fused_gelu = True
    x = torch.randn(*shape, device="cuda")
    with torch.inference_mode():
        torch.testing.assert_close(ff(x), ff.ff(x), rtol=1e-4, atol=1e-5)
