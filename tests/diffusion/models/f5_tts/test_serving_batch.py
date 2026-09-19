# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for F5-TTS request batching and CUDA Graph mask replay."""

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.f5_tts.cuda_graph_dit_wrapper import (
    F5TTSDiTCUDAGraphWrapper,
    _GraphKey,
)
from vllm_omni.diffusion.models.f5_tts.f5_tts_transformer import F5TTSDiTModel
from vllm_omni.diffusion.models.f5_tts.pipeline_f5_tts import F5TTSPipeline
from vllm_omni.diffusion.models.f5_tts.text_utils import (
    pad_and_batch,
    pad_and_batch_items,
    quantize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_quantize() -> None:
    assert quantize(10, 64) == 64
    assert quantize(64, 64) == 64
    assert quantize(65, 64) == 128


def test_pad_and_batch_items_single() -> None:
    cond_mel = torch.randn(20, 100)
    text_token_ids = [1, 2, 3, 4, 5]
    total_mel_len = 50

    cond_audio, cond_text, seq_len, effective_lens, cond_mel_lens, total_mel_lens = pad_and_batch_items(
        [(cond_mel, text_token_ids, total_mel_len)],
        pad_multiple=64,
    )

    assert seq_len == 64
    assert cond_audio.shape == (1, 64, 100)
    assert cond_text.shape == (1, 64)
    # B=1 extends effective length to quantized seq_len for parity with upstream F5
    assert effective_lens == [64]
    assert cond_mel_lens == [20]
    assert total_mel_lens == [50]
    assert torch.equal(cond_audio[0, :20], cond_mel)
    assert torch.all(cond_audio[0, 20:] == 0)
    assert cond_text[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert torch.all(cond_text[0, 5:] == -1)

    # Compare with pad_and_batch wrapper
    ca, ct, sl, cml = pad_and_batch(cond_mel, text_token_ids, total_mel_len, pad_multiple=64)
    assert torch.equal(ca, cond_audio)
    assert torch.equal(ct, cond_text)
    assert sl == seq_len
    assert cml == 20


def test_pad_and_batch_items_multi() -> None:
    mel1 = torch.randn(30, 100)
    text1 = [1, 2, 3]
    tot1 = 45

    mel2 = torch.randn(50, 100)
    text2 = [10, 20, 30, 40, 50, 60]
    tot2 = 80

    items = [(mel1, text1, tot1), (mel2, text2, tot2)]
    cond_audio, cond_text, seq_len, effective_lens, cond_mel_lens, total_mel_lens = pad_and_batch_items(
        items,
        pad_multiple=64,
    )

    # max_total is 80 -> quantizes to 128
    assert seq_len == 128
    assert cond_audio.shape == (2, 128, 100)
    assert cond_text.shape == (2, 128)
    assert effective_lens == [45, 80]
    assert cond_mel_lens == [30, 50]
    assert total_mel_lens == [45, 80]

    # Verify item 0
    assert torch.equal(cond_audio[0, :30], mel1)
    assert torch.all(cond_audio[0, 30:] == 0)
    assert cond_text[0, :3].tolist() == text1
    assert torch.all(cond_text[0, 3:] == -1)

    # Verify item 1
    assert torch.equal(cond_audio[1, :50], mel2)
    assert torch.all(cond_audio[1, 50:] == 0)
    assert cond_text[1, :6].tolist() == text2
    assert torch.all(cond_text[1, 6:] == -1)


def test_pipeline_supports_request_batch() -> None:
    assert F5TTSPipeline.supports_request_batch is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA Graph tests")
def test_cuda_graph_wrapper_with_mask() -> None:
    device = torch.device("cuda:0")
    dim = 64
    heads = 2
    dim_head = 32
    depth = 2

    od_config = OmniDiffusionConfig(
        model="SWivid/F5-TTS/F5TTS_v1_Base",
    )

    model = F5TTSDiTModel(
        od_config=od_config,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        ff_mult=2,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=64,
        conv_layers=1,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()

    wrapper = F5TTSDiTCUDAGraphWrapper(model, max_graphs=4)

    batch = 2
    seq_len = 128
    mel_dim = 100

    noisy_audio = torch.randn(batch, seq_len, mel_dim, device=device, dtype=torch.bfloat16)
    cond_audio = torch.randn(batch, seq_len, mel_dim, device=device, dtype=torch.bfloat16)
    cond_text = torch.randint(0, 100, (batch, seq_len), device=device, dtype=torch.long)
    timestep = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)

    # Create 2D attention padding mask: sample 0 has len 80, sample 1 has len 120
    sample_lens = torch.tensor([80, 120], device=device)
    seq_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = seq_pos < sample_lens.unsqueeze(1)  # [2, 128] bool

    # Eager reference output
    with torch.no_grad():
        eager_out = wrapper._run_eager(
            noisy_audio=noisy_audio,
            cond_audio=cond_audio,
            cond_text=cond_text,
            timestep=timestep,
            mask=mask,
        )

    # CUDA Graph call (will capture on first call, replay on subsequent calls)
    graph_out1 = wrapper(
        noisy_audio=noisy_audio,
        cond_audio=cond_audio,
        cond_text=cond_text,
        timestep=timestep,
        mask=mask,
    )

    # Check that graph was indeed captured
    expected_key = _GraphKey(
        device_index=noisy_audio.get_device(),
        audio_dtype=noisy_audio.dtype,
        text_dtype=cond_text.dtype,
        timestep_dtype=timestep.dtype,
        batch=batch,
        seq_len=seq_len,
        mel_dim=mel_dim,
        has_mask=True,
    )
    assert expected_key in wrapper.graphs
    assert wrapper.graphs[expected_key].graph is not None

    # Replay call
    graph_out2 = wrapper(
        noisy_audio=noisy_audio,
        cond_audio=cond_audio,
        cond_text=cond_text,
        timestep=timestep,
        mask=mask,
    )

    # Compare eager and graph replay outputs
    torch.testing.assert_close(graph_out1, eager_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(graph_out2, eager_out, rtol=1e-2, atol=1e-2)
