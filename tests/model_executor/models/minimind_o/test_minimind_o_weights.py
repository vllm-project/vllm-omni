# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import pytest

from vllm_omni.model_executor.models.minimind_o.minimind_omni_talker import (
    MiniMindOmniTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.minimind_o.minimind_omni_thinker import (
    MiniMindOmniThinkerForConditionalGeneration,
)


def _split_flat_hf_keys(keys: list[tuple[str, object]]):
    """Expected layout for jingyaogong/minimind-3o pytorch_model.bin."""
    thinker_prefixes = ("model.", "lm_head.", "audio_proj.", "vision_proj.")
    thinker, talker, other = [], [], []
    for k, v in keys:
        if k.startswith("talker."):
            talker.append((k, v))
        elif k.startswith(thinker_prefixes):
            thinker.append((k, v))
        else:
            other.append((k, v))
    return thinker, talker, other


def test_partition_flat_hf_weights():
    keys = [
        ("model.embed_tokens.weight", None),
        ("lm_head.weight", None),
        ("audio_proj.mlp.0.weight", None),
        ("vision_proj.mlp.0.weight", None),
        ("talker.layers.0.self_attn.q_proj.weight", None),
        ("talker.lm_head.base.weight", None),
    ]
    thinker, talker, other = _split_flat_hf_keys(keys)
    assert len(other) == 0
    assert {k for k, _ in thinker} == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "audio_proj.mlp.0.weight",
        "vision_proj.mlp.0.weight",
    }
    assert all(k.startswith("talker.") for k, _ in talker)


@pytest.mark.parametrize(
    "cls,mapper_prefix,expected",
    [
        (MiniMindOmniThinkerForConditionalGeneration, "model.", "language_model.model."),
        (MiniMindOmniThinkerForConditionalGeneration, "lm_head.", "language_model.lm_head."),
        (MiniMindOmniTalkerForConditionalGeneration, "talker.", ""),
    ],
)
def test_hf_to_vllm_mapper_prefixes(cls, mapper_prefix, expected):
    mapped = cls.hf_to_vllm_mapper.orig_to_new_prefix.get(mapper_prefix)
    assert mapped == expected


def test_hf_checkpoint_prefixes_match_mappers():
    pytest.importorskip("huggingface_hub")
    import torch
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("jingyaogong/minimind-3o", "pytorch_model.bin")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    prefixes = Counter(k.split(".")[0] for k in sd)
    assert prefixes["model"] > 0
    assert prefixes["talker"] > 0
    assert prefixes["audio_proj"] > 0
    assert prefixes["vision_proj"] > 0
