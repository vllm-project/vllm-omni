# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_omni.model_executor.models.minimind_o.minimind_omni_talker import (
    MiniMindOmniTalkerForConditionalGeneration,
)


def test_sample_codebook_logits_batch_respects_top_k():
    logits = torch.zeros(1, 100)
    logits[0, 42] = 10.0
    logits[0, 7] = 9.0
    # Instance method; sampling does not use self when logits are non-empty.
    codes = MiniMindOmniTalkerForConditionalGeneration._sample_codebook_logits_batch(
        object(),
        [logits],
        temperature=0.2,
        top_k=50,
    )
    assert codes.shape == (1, 1)
    assert int(codes[0, 0]) in (7, 42)
