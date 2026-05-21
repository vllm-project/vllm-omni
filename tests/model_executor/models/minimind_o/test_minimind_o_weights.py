# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import pytest

from vllm_omni.model_executor.models.minimind_o.minimind_o import MiniMindOForConditionalGeneration


def _fake_keys():
    return [
        ("model.embed_tokens.weight", None),
        ("lm_head.weight", None),
        ("audio_proj.mlp.0.weight", None),
        ("vision_proj.mlp.0.weight", None),
        ("talker.layers.0.self_attn.q_proj.weight", None),
        ("talker.lm_head.base.weight", None),
    ]


def test_partition_flat_hf_weights():
    thinker, talker, code2wav, other = MiniMindOForConditionalGeneration._partition_omni_weights(_fake_keys())
    assert len(other) == 0
    assert len(code2wav) == 0
    assert {k for k, _ in thinker} == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "audio_proj.mlp.0.weight",
        "vision_proj.mlp.0.weight",
    }
    assert all(k.startswith("talker.") for k, _ in talker)


@pytest.mark.parametrize(
    "mapper_prefix,expected",
    [
        ("model.", "language_model.model."),
        ("talker.layers.", "language_model.model.layers."),
        ("talker.lm_head.", "mtp_head."),
    ],
)
def test_thinker_talker_weight_mapper_expectations(mapper_prefix, expected):
    from vllm_omni.model_executor.models.minimind_o.minimind_o_thinker import (
        MiniMindOThinkerForConditionalGeneration,
    )
    from vllm_omni.model_executor.models.minimind_o.minimind_o_talker import (
        MiniMindOTalkerForConditionalGeneration,
    )

    cls = MiniMindOThinkerForConditionalGeneration if mapper_prefix == "model." else MiniMindOTalkerForConditionalGeneration
    mapped = cls.hf_to_vllm_mapper.orig_to_new_prefix.get(mapper_prefix)
    assert mapped == expected


def test_hf_checkpoint_prefixes_match_mappers():
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import hf_hub_download
    import torch

    p = hf_hub_download("jingyaogong/minimind-3o", "pytorch_model.bin")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    prefixes = Counter(k.split(".")[0] for k in sd)
    assert prefixes["model"] > 0
    assert prefixes["talker"] > 0
    assert prefixes["audio_proj"] > 0
    assert prefixes["vision_proj"] > 0
