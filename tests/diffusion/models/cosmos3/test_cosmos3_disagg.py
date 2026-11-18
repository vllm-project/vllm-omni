# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Cosmos3 tower-disaggregated K/V handoff."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_disagg import (
    KV_KEY,
    _ReplayLanguageModel,
    fingerprint_text_ids,
)
from vllm_omni.model_executor.stage_input_processors.cosmos3 import reasoner2generator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_disagg_topology_is_registered() -> None:
    stages = OMNI_PIPELINES["cosmos3_omni_disagg"].stages
    assert [s.model_arch for s in stages] == ["Cosmos3ReasonerPipeline", "Cosmos3GeneratorPipeline"]


def test_reasoner2generator_forwards_prompt_and_kv() -> None:
    kv = {"fp": [(torch.zeros(1), torch.zeros(1))]}
    output = SimpleNamespace(multimodal_output={"trajectory": {KV_KEY: kv}}, outputs=[])

    result = reasoner2generator([output], prompt={"prompt": "a red car", "negative_prompt": "blurry"})

    assert result == {
        "prompt": "a red car",
        "modalities": ["image"],
        "extra": {KV_KEY: kv},
        "negative_prompt": "blurry",
    }
    assert reasoner2generator([SimpleNamespace(multimodal_output={}, outputs=[])], prompt={"prompt": "x"}) is None


def test_replay_returns_this_ranks_kv_heads() -> None:
    text_ids = torch.tensor([[1, 2, 3]])
    k, v = torch.randn(1, 3, 4, 2), torch.randn(1, 3, 4, 2)
    replay = _ReplayLanguageModel(nn.Identity(), kv_head_offset=2, num_kv_heads_local=2)
    replay.table, replay.dtype = {fingerprint_text_ids(text_ids): [(k, v)]}, torch.float32

    [(k_out, v_out)] = replay(text_ids, None)

    assert torch.equal(k_out, k[:, :, 2:4]) and torch.equal(v_out, v[:, :, 2:4])
    with pytest.raises(RuntimeError, match="no reasoner K/V"):
        replay(torch.tensor([[9]]), None)
