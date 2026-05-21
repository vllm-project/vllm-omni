# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_omni.model_executor.models.minimind_o.minimind_o_talker import (
    MiniMindOTalkerForConditionalGeneration,
)


def test_sample_mimi_layer_logits_respects_top_k():
    logits = torch.zeros(100)
    logits[42] = 10.0
    logits[7] = 9.0
    code = MiniMindOTalkerForConditionalGeneration._sample_mimi_layer_logits(
        logits,
        [],
        temperature=0.2,
        top_k=50,
    )
    assert code in (7, 42)
