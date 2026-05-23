# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_omni.model_executor.models.minimind_o.minimind_omni_talker import (
    MiniMindOmniTalkerForConditionalGeneration,
)


def test_sample_codebook_logits_respects_top_k():
    logits = torch.zeros(100)
    logits[42] = 10.0
    logits[7] = 9.0
    # Instance method; sampling does not use self.
    code = MiniMindOmniTalkerForConditionalGeneration._sample_codebook_logits(
        object(),
        logits,
        temperature=0.2,
        top_k=50,
    )
    assert code in (7, 42)
