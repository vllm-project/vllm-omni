# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8-weight chain of the fused residual-codebook predictor (code_predictor_fp8, sm_89)."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.qwen3_tts.test_fused_code_predictor_pdl import GROUPS, VOCAB, _predictor

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability() != (8, 9),
        reason="requires CUDA sm_89",
    ),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_fp8_predictor_matches_bf16_path_and_replays_in_graphs():
    from vllm_omni.model_executor.models.qwen3_tts import fused_code_predictor as fcp

    dev = "cuda"
    pred = _predictor(dev)
    fp8 = fcp.FusedCodePredictor(pred, 128, fp8_weights=True)
    assert fp8._w8, "sm_89 must take the FP8 chain when requested"
    ref = fcp.FusedCodePredictor(_predictor(dev), 128)
    assert not ref._w8

    for B in sorted({1, 2, 5, 16, 17, 33, 64, 65, 128}):
        gen = torch.Generator(device=dev).manual_seed(B)
        code0 = torch.randint(0, VOCAB, (B, 1), device=dev, generator=gen)
        emb = torch.randn(B, 1, 2048, device=dev, generator=gen).to(torch.bfloat16)
        hid = torch.randn(B, 1, 2048, device=dev, generator=gen).to(torch.bfloat16)
        # greedy: constant Gumbel noise, no top-k
        half = torch.full((B, GROUPS - 1, VOCAB), 0.5, device=dev)
        got = fp8(code0, emb, hid, 1.0, 0, half)
        want = ref(code0, emb, hid, 1.0, 0, half)
        assert torch.equal(got[:, 0], code0[:, 0])
        # FP8 weights round differently, so near-ties may flip.
        assert (got[:, 1] == want[:, 1]).float().mean().item() >= 0.7
        assert torch.equal(fp8(code0, emb, hid, 1.0, 0, half), got), "the chain must be deterministic"

        u = torch.rand(B, GROUPS - 1, VOCAB, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)
        eager = fp8(code0, emb, hid, 1 / 0.9, 50, u)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fp8(code0, emb, hid, 1 / 0.9, 50, u)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(captured, eager)
