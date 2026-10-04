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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_causal_hift_f0_predictor_cuda_parity_and_fallback(monkeypatch):
    """Test GPU-resident F0 predictor on CUDA with streaming/finalize, TF32 restoration,
    parity against CPU reference, and COSYVOICE3_F0_ON_CPU fallback."""
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
        CausalConvRNNF0Predictor,
        CausalHiFTGenerator,
    )

    torch.manual_seed(42)
    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16)
    hift = CausalHiFTGenerator(
        in_channels=80,
        base_channels=32,
        nb_harmonics=8,
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    ).eval()

    with torch.no_grad():
        # Force voiced F0 above nsf_voiced_threshold (10.0)
        hift.f0_predictor.classifier.bias.fill_(150.0)

    # 1. Compute CPU reference for both streaming (finalize=False) and finalize=True
    speech_feat_cpu = torch.randn(1, 80, 50, dtype=torch.float32)
    with torch.no_grad():
        f0_ref_streaming = hift.f0_predictor(speech_feat_cpu, finalize=False)
        f0_ref_finalize = hift.f0_predictor(speech_feat_cpu, finalize=True)

    # Move HiFT to CUDA
    hift = hift.to("cuda")
    speech_feat_cuda = speech_feat_cpu.to("cuda")

    # Set TF32 to True initially to verify `finally` restoration
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.delenv("COSYVOICE3_F0_ON_CPU", raising=False)

    # 2. Test CUDA streaming execution (finalize=False)
    wav_stream, s_stream, phase_acc = hift.inference(speech_feat_cuda, finalize=False)
    assert wav_stream.device.type == "cuda"
    assert s_stream.device.type == "cuda"
    assert next(hift.f0_predictor.parameters()).device.type == "cuda"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True

    # Check streaming F0 parity against CPU reference under explicit tolerances
    with torch.no_grad():
        f0_cuda_streaming = hift.f0_predictor(speech_feat_cuda, finalize=False)
    torch.testing.assert_close(f0_cuda_streaming.cpu(), f0_ref_streaming, atol=1e-3, rtol=1e-3)
    cos_sim_stream = torch.nn.functional.cosine_similarity(
        f0_cuda_streaming.cpu().flatten(1), f0_ref_streaming.flatten(1), dim=1
    ).item()
    assert cos_sim_stream > 0.9999

    # 3. Test CUDA finalize execution (finalize=True)
    wav_final, s_final, _ = hift.inference(speech_feat_cuda, finalize=True, phase_acc=phase_acc)
    assert wav_final.device.type == "cuda"
    assert s_final.device.type == "cuda"
    assert next(hift.f0_predictor.parameters()).device.type == "cuda"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True

    # Check finalize F0 parity against CPU reference under explicit tolerances
    with torch.no_grad():
        f0_cuda_finalize = hift.f0_predictor(speech_feat_cuda, finalize=True)
    torch.testing.assert_close(f0_cuda_finalize.cpu(), f0_ref_finalize, atol=1e-3, rtol=1e-3)
    cos_sim_final = torch.nn.functional.cosine_similarity(
        f0_cuda_finalize.cpu().flatten(1), f0_ref_finalize.flatten(1), dim=1
    ).item()
    assert cos_sim_final > 0.9999

    # 4. Test COSYVOICE3_F0_ON_CPU=1 fallback path
    monkeypatch.setenv("COSYVOICE3_F0_ON_CPU", "1")
    wav_fallback, s_fallback, _ = hift.inference(speech_feat_cuda, finalize=True, phase_acc=phase_acc)
    assert wav_fallback.device.type == "cuda"
    assert s_fallback.device.type == "cuda"
    # Predictor should have been moved to CPU
    assert next(hift.f0_predictor.parameters()).device.type == "cpu"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True

    # Fallback output should match within numerical tolerance
    torch.testing.assert_close(s_final, s_fallback, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(wav_final, wav_fallback, atol=1e-2, rtol=1e-2)
