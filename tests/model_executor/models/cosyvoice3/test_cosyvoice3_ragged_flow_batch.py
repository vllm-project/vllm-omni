# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-request streaming flow batching over different speaker references."""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PRE_LOOKAHEAD = 3


@pytest.fixture
def torch_sdpa():
    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield


@pytest.fixture
def position_noise(monkeypatch):
    """Make the flow's initial noise a function of mel position only, so a row
    draws the same noise whether it runs alone or padded inside a batch."""
    original_randn = torch.randn

    def randn(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        if len(shape) == 3 and shape[1] == 80:
            channels = torch.arange(shape[1], dtype=torch.float32).view(1, -1, 1)
            positions = torch.arange(shape[2], dtype=torch.float32).view(1, 1, -1)
            noise = torch.sin(channels * 0.17 + positions * 0.31).expand(shape[0], -1, -1)
            return noise.to(device=kwargs.get("device"), dtype=kwargs.get("dtype", torch.float32)).clone()
        return original_randn(*size, **kwargs)

    monkeypatch.setattr(torch, "randn", randn)


def _tiny_code2wav():
    from omegaconf import DictConfig

    from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        CausalConditionalCFM,
        CausalMaskedDiffWithDiT,
    )
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    torch.manual_seed(0)
    estimator = DiT(
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        dropout=0.0,
        ff_mult=2,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
    )
    decoder = CausalConditionalCFM(
        in_channels=80,
        cfm_params=DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        ),
        n_spks=1,
        spk_emb_dim=80,
        estimator=estimator,
    )
    flow_model = CausalMaskedDiffWithDiT(
        input_size=80,
        output_size=80,
        spk_embed_dim=192,
        vocab_size=64,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=PRE_LOOKAHEAD,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=80, pre_lookahead_len=PRE_LOOKAHEAD),
        decoder=decoder,
    ).eval()

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = flow_model

    def return_mel(self, feat, *, cache_state=None, finalize=False):
        return feat, None

    model._stream_hift_from_feat = types.MethodType(return_mel, model)
    return model


def _item(prompt: int, generated: int, *, offset: int, finalize: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        "token": torch.randint(0, 64, (1, generated), generator=generator, dtype=torch.int32),
        "prompt_token": torch.randint(0, 64, (1, prompt), generator=generator, dtype=torch.int32),
        "prompt_feat": torch.randn(1, 2 * prompt, 80, generator=generator),
        "embedding": torch.randn(1, 192, generator=generator),
        "token_offset_tokens": offset,
        "finalize": finalize,
        "cache_state": None,
    }


@pytest.mark.parametrize("finalize", [False, True])
@torch.inference_mode()
def test_mixed_reference_batch_matches_independent_flow(torch_sdpa, position_noise, finalize):
    """Requests with different reference lengths share one flow call, and each
    row must still get its own conditioning, lookahead context and crop point.
    Unequal prompt + token lengths also cover the lookahead slice, which used to
    hand a shorter row its own padding as context."""
    model = _tiny_code2wav()
    shapes = [(7, 31, 0), (13, 17, 3), (3, 44, 0), (10, 31, 5)]
    items = [
        _item(prompt, generated, offset=offset, finalize=finalize, seed=index)
        for index, (prompt, generated, offset) in enumerate(shapes)
    ]

    expected = [model.forward_streaming_batch([item], n_timesteps=2)[0][0] for item in items]
    for ordered, reference in [(items, expected), (items[::-1], expected[::-1])]:
        actual = model.forward_streaming_batch(ordered, n_timesteps=2)
        for (mel, _), ref_mel in zip(actual, reference, strict=True):
            assert mel.shape == ref_mel.shape
            torch.testing.assert_close(mel, ref_mel, rtol=1e-4, atol=1e-4)


def test_different_reference_lengths_share_one_flow_call():
    """Grouping on the reference length made every distinct speaker its own
    group of one, so the batched estimator never saw a batched shape."""
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    model = object.__new__(CosyVoice3Code2Wav)
    model.flow_model = SimpleNamespace(pre_lookahead_len=PRE_LOOKAHEAD, token_mel_ratio=2)
    calls = []

    def fake_forward_mel(self, **kwargs):
        calls.append(kwargs)
        return torch.zeros(kwargs["token"].shape[0], 80, 2 * int(kwargs["token"].shape[1]))

    def fake_stream_hift(self, feat, *, cache_state=None, finalize=False):
        return feat, None

    model._forward_mel = types.MethodType(fake_forward_mel, model)
    model._stream_hift_from_feat = types.MethodType(fake_stream_hift, model)
    prompts = [25, 37, 25, 112]
    items = [_item(prompt, 28, offset=0, finalize=False, seed=index) for index, prompt in enumerate(prompts)]
    items.append(_item(9, 20, offset=0, finalize=True, seed=9))

    results = model.forward_streaming_batch(items)

    streaming = [call for call in calls if not call["finalize"]]
    assert len(calls) == 2 and len(streaming) == 1
    call = streaming[0]
    assert call["prompt_token"].shape == (4, max(prompts))
    assert call["prompt_token_lens"].tolist() == prompts
    assert call["prompt_feat_lens"].tolist() == [2 * prompt for prompt in prompts]
    assert all(results[index][0].shape[-1] == 2 * (28 - PRE_LOOKAHEAD) for index in range(len(prompts)))
