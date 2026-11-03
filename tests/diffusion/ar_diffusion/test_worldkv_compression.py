# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent dense scoring oracles and storage/lifecycle compression tests."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.ar_diffusion.test_worldkv import worldkv
from tests.diffusion.ar_diffusion.test_worldkv_retrieval import append as append_retrieval
from tests.diffusion.ar_diffusion.test_worldkv_retrieval import state as retrieval_state

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def rows(count=12, dtype=torch.float64):
    key = torch.randn(1, count, 2, 8, generator=torch.Generator().manual_seed(71), dtype=dtype)
    return worldkv.MemoryRows(key, key + 20, torch.arange(count) // 4, torch.arange(count))


@pytest.mark.parametrize("score", ["pairwise_mean", "centroid"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_world_selection_equals_dense_oracle_and_owns_exact_storage(score, ratio, dtype):
    source = rows(dtype=dtype)
    output = worldkv.world_compress(source, 4, ratio, score)
    flat = source.key[0].double().flatten(1)
    if score == "pairwise_mean":
        reference = (F.normalize(flat[4:], dim=-1) @ F.normalize(flat[:4], dim=-1).T).mean(-1)
    else:
        reference = F.cosine_similarity(flat[4:], flat[:4].mean(0)[None], dim=-1)
    selected = torch.argsort(reference, stable=True)[: math.ceil(8 * ratio)].sort().values + 4
    indices = torch.cat((torch.arange(4), selected))
    assert torch.equal(output.key, source.key[:, indices])
    assert torch.equal(output.value, source.value[:, indices])
    assert torch.equal(output.positions, source.positions[indices])
    assert output.key.untyped_storage().nbytes() == output.key.numel() * output.key.element_size()
    assert output.value.untyped_storage().nbytes() == output.value.numel() * output.value.element_size()


def test_paper_and_centroid_scores_are_distinct_not_silently_interchanged():
    source = rows()
    # Unequal anchor norms distinguish mean(normalize(K)) from normalize(mean(K)).
    source.key[:, 0] *= 100
    paper = worldkv.world_compress(source, 4, 0.25, "pairwise_mean")
    code = worldkv.world_compress(source, 4, 0.25, "centroid")
    assert not torch.equal(paper.key, code.key)


def test_zero_keys_ties_preserve_anchor_and_stable_original_order():
    source = worldkv.MemoryRows(
        torch.zeros(1, 12, 1, 4), torch.arange(48.0).view(1, 12, 1, 4), torch.arange(12) // 4, torch.arange(12)
    )
    output = worldkv.world_compress(source, 4, 0.25, "pairwise_mean")
    assert torch.equal(output.value, source.value[:, :6])
    assert torch.isfinite(output.key).all()


def test_compressed_bank_uses_half_kv_and_per_layer_selection():
    plain = retrieval_state()
    compressed = retrieval_state(world_keep_ratio=0.0)
    for start in range(0, 8, 2):
        append_retrieval(plain, start)
        append_retrieval(compressed, start)
    assert compressed.bank_nbytes < plain.bank_nbytes
    for start, entry in compressed.bank.items():
        for layer, r in enumerate(entry.layers):
            assert r.key.shape[1] == 2
            assert torch.equal(r.key, plain.bank[start].layers[layer].key[:, :2])


@pytest.mark.parametrize(
    "config",
    [
        {"world_keep_ratio": float("nan")},
        {"world_keep_ratio": 2},
        {"world_keep_ratio": 0.25},
        {"world_score": "unknown"},
    ],
)
def test_invalid_compression_config(config):
    with pytest.raises(ValueError):
        worldkv.WorldKVConfig(**config)
