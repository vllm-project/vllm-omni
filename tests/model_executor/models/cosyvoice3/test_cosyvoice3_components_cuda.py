# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only CosyVoice3 component tests."""

import types

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_code2wav_streaming_batch_matches_ragged_flow_numerics(monkeypatch):
    """A padded flow batch must match individual flow calls on valid mels."""
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
        depth=1,
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
    flow_model = (
        CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            vocab_size=64,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=2,
            pre_lookahead_len=1,
            pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=80, pre_lookahead_len=1),
            decoder=decoder,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = flow_model

    def return_mel(self, feat, *, cache_state=None, finalize=False):
        return feat, None

    model._stream_hift_from_feat = types.MethodType(return_mel, model)

    original_randn = torch.randn

    def length_consistent_randn(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        if len(shape) == 3 and shape[1] == 80:
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", torch.float32)
            channels = torch.arange(shape[1], device=device, dtype=torch.float32).view(1, -1, 1)
            positions = torch.arange(shape[2], device=device, dtype=torch.float32).view(1, 1, -1)
            noise = torch.sin(channels * 0.17 + positions * 0.31)
            return noise.expand(shape[0], -1, -1).to(dtype=dtype).clone()
        return original_randn(*size, **kwargs)

    monkeypatch.setattr(torch, "randn", length_consistent_randn)
    common = {
        "prompt_token": torch.tensor([[7, 8]], dtype=torch.int32),
        "prompt_feat": torch.linspace(-0.5, 0.5, 4 * 80).reshape(1, 4, 80),
        "embedding": torch.linspace(-1.0, 1.0, 192).reshape(1, 192),
        "finalize": False,
    }
    items = [
        {**common, "token": torch.tensor([[1, 2, 3]], dtype=torch.int32)},
        {**common, "token": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)},
    ]

    batched = model.forward_streaming_batch(items, n_timesteps=2)
    individual = [
        model.forward_streaming(
            token=item["token"],
            prompt_token=item["prompt_token"],
            prompt_feat=item["prompt_feat"],
            embedding=item["embedding"],
            n_timesteps=2,
        )
        for item in items
    ]

    for (batched_mel, _), (individual_mel, _) in zip(batched, individual):
        assert batched_mel.shape == individual_mel.shape
        rel_mean = (batched_mel - individual_mel).abs().mean() / individual_mel.abs().mean().clamp_min(1e-6)
        assert rel_mean.item() < 0.05
