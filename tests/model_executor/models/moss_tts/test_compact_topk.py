# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for MOSS-TTS compact top-k sampling."""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import modeling_moss_tts_local

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_compact_topk_only_samples_retained_candidates() -> None:
    logits = torch.randn(32, 1024, generator=torch.Generator().manual_seed(7))

    sampled = modeling_moss_tts_local._sample_token(
        logits,
        temperature=1.7,
        top_k=25,
        top_p=0.8,
        do_sample=True,
        generator=torch.Generator().manual_seed(42),
    )

    retained = logits.topk(25, dim=-1).indices
    assert torch.all((retained == sampled.unsqueeze(-1)).any(dim=-1))


def test_compact_topk_tiny_nucleus_selects_argmax() -> None:
    logits = torch.randn(32, 1024, generator=torch.Generator().manual_seed(17))

    sampled = modeling_moss_tts_local._sample_token(
        logits,
        temperature=1.7,
        top_k=25,
        top_p=1e-9,
        do_sample=True,
        generator=torch.Generator().manual_seed(42),
    )

    torch.testing.assert_close(sampled, logits.argmax(dim=-1))
