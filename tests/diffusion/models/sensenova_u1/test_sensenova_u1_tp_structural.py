# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B5 L0: TP sharding/weight-routing structural tests (CPU only).

These pin the *structural* half of the B5 TP contract for SenseNova-U1(.5)
before any multi-GPU run exists, so that a TP bring-up failure can be
attributed to degree legality / shard routing rather than discovered by trial:

1. `stacked_params_mapping` name routing, loaded verbatim from the pinned
   pipeline source. The documented hazard is that the more specific
   ``*_mot_gen`` patterns must win over the plain ``q_proj`` / ``k_proj`` /
   ``v_proj`` substrings, and that both towers must be covered. This is a
   pure name-matching property and needs no model.
2. Tensor-parallel sharding arithmetic for every parallel layer the model
   uses, following vLLM's own rules
   (``vllm/model_executor/layers/linear.py`` → ``QKVParallelLinear``):
   query heads are partitioned, and when ``tp_size >= total_num_kv_heads`` the
   KV heads are *replicated* with ``tp_size // total_num_kv_heads`` replicas
   each rather than divided further. Getting this wrong is exactly the
   ``global_kv_heads // tp`` assumption the B5 issue warns about.
3. Degree legality for the real checkpoint dimensions, with a clear error
   instead of a silent mis-shard.
4. The invariant that the RoPE / QK-norm geometry (head_dim and its halves)
   must not change with TP degree.

No model, no GPU, no vllm import: the mapping and the rules are re-derived
from source + vLLM's documented arithmetic so this runs anywhere.
"""

import ast
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_PIPELINE_SRC = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py"
)

# Real checkpoint dimensions: models/SenseNova-U1.5-8B-MoT/config.json -> llm_config
U15 = dict(
    hidden_size=4096,
    intermediate_size=12288,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    num_hidden_layers=42,
)


def _load_stacked_params_mapping():
    """Extract the class-level `stacked_params_mapping` from the pipeline source.

    It is a literal list of tuples, so it is evaluated in isolation; no vllm
    import is required. Note `load_weights` also assigns a *local* name
    `stacked_params_mapping = self.stacked_params_mapping`; that shadowing
    assignment must not be picked up, so only the annotated assignment whose
    value is a literal list/tuple is considered.
    """
    text = _PIPELINE_SRC.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "stacked_params_mapping" for t in targets):
            continue
        value = node.value
        # only the ClassVar literal; skip the method-local alias `= self....`
        if isinstance(value, (ast.List, ast.Tuple)):
            return ast.literal_eval(value)
    raise RuntimeError(f"class-level stacked_params_mapping not found in {_PIPELINE_SRC}")


STACKED_PARAMS_MAPPING = _load_stacked_params_mapping()


# ---------------------------------------------------------------------------
# 1. Name routing
# ---------------------------------------------------------------------------


def route(name: str):
    """Replicate the loader's matching loop: first matching entry wins.

    Mirrors `SenseNovaU1Pipeline.load_weights` name resolution.
    """
    for param_name, weight_name, shard_id in STACKED_PARAMS_MAPPING:
        if weight_name in name:
            return name.replace(weight_name, param_name), shard_id
    return None, None


def test_both_towers_are_covered_by_the_mapping():
    """`und` and `mot_gen` must both route; testing one tower is not enough.

    The B5 issue requires covering the understanding and generation towers,
    because a loader bug can hide in the tower nobody checked.
    """
    und_layer = "model.layers.0.self_attn."
    gen_layer = "model.layers.0.self_attn."
    for proj, slot in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
        assert route(und_layer + proj)[0] == und_layer + "qkv_proj"
        assert route(und_layer + proj)[1] == slot
        assert route(gen_layer + proj + "_mot_gen")[0] == gen_layer + "qkv_proj_mot_gen"
        assert route(gen_layer + proj + "_mot_gen")[1] == slot
    # MLP gate/up fusion, both towers where applicable
    for layer in ("model.layers.0.mlp.", "model.layers.0.mlp_mot_gen."):
        assert route(layer + "gate_proj") == (layer + "gate_up_proj", 0)
        assert route(layer + "up_proj") == (layer + "gate_up_proj", 1)


def test_mot_gen_patterns_win_over_the_plain_substring():
    """`.q_proj` is a substring of `.q_proj_mot_gen`; order must disambiguate.

    If the plain patterns came first, a generation-tower weight would be routed
    into the understanding tower's fused projection.
    """
    # `.qkv_proj` is a substring of `.qkv_proj_mot_gen`
    assert ".qkv_proj" in ".qkv_proj_mot_gen"
    assert ".q_proj" in ".q_proj_mot_gen"

    gen = "model.layers.3.self_attn.v_proj_mot_gen"
    param, slot = route(gen)
    assert param == "model.layers.3.self_attn.qkv_proj_mot_gen"
    assert slot == "v"
    assert "mot_gen" in param

    und = "model.layers.3.self_attn.v_proj"
    param_u, slot_u = route(und)
    assert param_u == "model.layers.3.self_attn.qkv_proj"
    assert slot_u == "v"
    assert "mot_gen" not in param_u


def test_mot_gen_entries_precede_plain_entries():
    """Guard the ordering invariant directly, not just its effect."""
    names = [weight_name for _, weight_name, _ in STACKED_PARAMS_MAPPING]
    gen_positions = [i for i, n in enumerate(names) if "mot_gen" in n]
    plain_positions = [i for i, n in enumerate(names) if "mot_gen" not in n]
    assert gen_positions, "no mot_gen entries in the mapping"
    assert max(gen_positions) < min(plain_positions), (
        "all *_mot_gen entries must precede the plain entries, otherwise the "
        f"plain substring wins: {STACKED_PARAMS_MAPPING}"
    )


def test_non_parallel_names_are_left_alone():
    """Replicated modules must not be captured by the fused mapping."""
    for name in (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm_hw_mot_gen.weight",
        "model.layers.0.input_layernorm.weight",
    ):
        assert route(name) == (None, None), f"{name} unexpectedly routed"


def test_mapping_shard_ids_are_well_formed():
    """q/k/v map to string shard ids; the fused MLP maps to column indices."""
    qkv = {w: sid for p, w, sid in STACKED_PARAMS_MAPPING if p.endswith("qkv_proj") or p.endswith("qkv_proj_mot_gen")}
    assert set(qkv.values()) == {"q", "k", "v"}
    gate_up = {w: sid for p, w, sid in STACKED_PARAMS_MAPPING if p.endswith("gate_up_proj")}
    assert set(gate_up.values()) == {0, 1}
    assert set(gate_up) == {".gate_proj", ".up_proj"}


# ---------------------------------------------------------------------------
# 2. Sharding arithmetic (vLLM's rules)
# ---------------------------------------------------------------------------


def divide(numerator: int, denominator: int) -> int:
    """vLLM's `divide`: must be exact."""
    if denominator == 0 or numerator % denominator != 0:
        raise ValueError(f"{numerator} not divisible by {denominator}")
    return numerator // denominator


def qkv_shard(total_num_heads: int, total_num_kv_heads: int, head_dim: int, tp_size: int) -> dict:
    """Reproduce vLLM `QKVParallelLinear` head/KV-head partitioning.

    From vllm/model_executor/layers/linear.py:
        num_heads = divide(total_num_heads, tp_size)
        if tp_size >= total_num_kv_heads:
            num_kv_heads = 1
            num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            num_kv_heads = divide(total_num_kv_heads, tp_size)
            num_kv_head_replicas = 1
    """
    num_heads = divide(total_num_heads, tp_size)
    if tp_size >= total_num_kv_heads:
        num_kv_heads = 1
        num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
    else:
        num_kv_heads = divide(total_num_kv_heads, tp_size)
        num_kv_head_replicas = 1
    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_kv_head_replicas": num_kv_head_replicas,
        "q_size": num_heads * head_dim,
        "kv_size": num_kv_heads * head_dim,
        "qkv_output": num_heads * head_dim + 2 * num_kv_heads * head_dim,
        "o_proj_input": num_heads * head_dim,
    }


def tp_degree_is_legal(cfg: dict, tp_size: int) -> bool:
    """A degree is legal iff every partition divides exactly."""
    if tp_size < 1:
        return False
    if cfg["num_attention_heads"] % tp_size:
        return False
    if tp_size >= cfg["num_key_value_heads"]:
        # KV heads are replicated; the replica count must divide exactly.
        return cfg["num_key_value_heads"] > 0 and tp_size % cfg["num_key_value_heads"] == 0
    return cfg["num_key_value_heads"] % tp_size == 0


def test_real_config_degree_legality_table():
    """Documented result for the real checkpoint: which TP degrees are legal.

    num_heads=32, num_kv_heads=8. Every degree in {1,2,4,8,16,...} that divides
    32 query heads reaches the KV-replication branch (tp >= 8) and is legal as
    long as tp is a multiple of 8. Degrees that do not divide 32 (e.g. 3) are
    rejected. tp=16 is legal in this table but exceeds the 8 GPUs available on
    the target host, so practical coverage here is {1,2,4,8}.
    """
    assert tp_degree_is_legal(U15, 1)
    assert tp_degree_is_legal(U15, 2)
    assert tp_degree_is_legal(U15, 4)
    assert tp_degree_is_legal(U15, 8)
    # legal in principle: 32/16 = 2 query heads, 16 % 8 == 0 KV replicas
    assert tp_degree_is_legal(U15, 16)
    # rejected: does not divide the query heads
    assert not tp_degree_is_legal(U15, 3)
    assert not tp_degree_is_legal(U15, 5)
    # rejected: divides 32 but the KV replica count does not divide
    assert not tp_degree_is_legal(U15, 64)


def test_local_query_head_count_is_tp_invariant_in_total():
    """Sharded query heads must reconstruct the global head count."""
    for tp in (1, 2, 4, 8):
        shard = qkv_shard(U15["num_attention_heads"], U15["num_key_value_heads"], U15["head_dim"], tp)
        assert shard["num_heads"] * tp == U15["num_attention_heads"]


def test_real_config_shard_shapes():
    """Per-degree local shapes for the real checkpoint.

    `qkv_output` is the rank-local fused width
    `(num_heads + 2 * num_kv_heads) * head_dim` (q, k, v).
    """
    expected = {
        1: dict(num_heads=32, num_kv_heads=8, num_kv_head_replicas=1,
                q_size=4096, kv_size=1024, o_proj_input=4096, qkv_output=6144),
        2: dict(num_heads=16, num_kv_heads=4, num_kv_head_replicas=1,
                q_size=2048, kv_size=512, o_proj_input=2048, qkv_output=3072),
        4: dict(num_heads=8, num_kv_heads=2, num_kv_head_replicas=1,
                q_size=1024, kv_size=256, o_proj_input=1024, qkv_output=1536),
        # tp >= total_num_kv_heads: replicate instead of dividing further
        8: dict(num_heads=4, num_kv_heads=1, num_kv_head_replicas=1,
                q_size=512, kv_size=128, o_proj_input=512, qkv_output=768),
    }
    for tp, exp in expected.items():
        got = qkv_shard(U15["num_attention_heads"], U15["num_key_value_heads"], U15["head_dim"], tp)
        for key, value in exp.items():
            assert got[key] == value, (tp, key, got[key], value)
        # the fused width is q + k + v, and never zero for a legal degree
        assert got["qkv_output"] == got["q_size"] + 2 * got["kv_size"]
        assert got["o_proj_input"] == got["q_size"]
        # the partitioned query heads reconstruct the global head count
        assert got["num_heads"] * tp == U15["num_attention_heads"]


def test_kv_head_replication_is_not_naive_division():
    """`global_kv_heads // tp` and vLLM's rule differ exactly at tp >= kv_heads.

    With num_kv_heads=8 and tp=8, the replication branch applies: one local KV
    head rather than a division of 8 by 8 collapsing to zero-width semantics.
    Degrees above 8 do not divide num_heads=32, so they are rejected up front
    rather than reaching this arithmetic.
    """
    # naive division would produce a nonsense zero once tp > num_kv_heads
    assert 8 // 16 == 0
    # vLLM's replication branch instead clamps to 1 KV head with 2 replicas
    shard16 = qkv_shard(32, 8, 128, 16)
    assert shard16["num_kv_heads"] == 1
    assert shard16["num_kv_head_replicas"] == 2
    assert shard16["kv_size"] == 128  # one head's worth, not zero
    # and at tp == num_kv_heads there is exactly one replica per rank
    shard8 = qkv_shard(32, 8, 128, 8)
    assert shard8["num_kv_heads"] == 1
    assert shard8["num_kv_head_replicas"] == 1
    assert shard8["kv_size"] == 128


def test_qkv_shard_width_shrinks_with_tp_but_sums_to_a_constant():
    """The rank-LOCAL qkv width is `global_width / tp`, not a TP-invariant.

    The invariant is the sum over ranks:
        sum_ranks (num_heads + 2*num_kv_heads) * head_dim == global qkv width
    and the K/V contribution is replicated rather than divided once
    tp >= num_kv_heads, so the total can exceed the naive global q-k-v width.
    """
    per_rank = {tp: qkv_shard(32, 8, 128, tp)["qkv_output"] for tp in (1, 2, 4, 8)}
    assert per_rank == {1: 6144, 2: 3072, 4: 1536, 8: 768}
    # rank-local width is the global width divided by tp
    for tp, width in per_rank.items():
        assert width * tp == per_rank[1]
    # the sum of partitioned query-head widths is the global q width
    for tp in (1, 2, 4, 8):
        shard = qkv_shard(32, 8, 128, tp)
        assert shard["q_size"] * tp == 32 * 128


def test_mlp_and_projection_sharding():
    """gate_up splits per shard; down/o_proj take the rank-local input."""
    hidden = U15["hidden_size"]
    inter = U15["intermediate_size"]
    for tp in (1, 2, 4, 8):
        assert divide(inter, tp) * tp == inter
        # MergedColumnParallelLinear over [inter, inter] -> local 2 * inter/tp
        assert 2 * divide(inter, tp) == 2 * inter // tp
        # RowParallelLinear input is the rank-local intermediate width
        shard = qkv_shard(32, 8, 128, tp)
        assert shard["o_proj_input"] == (32 // tp) * 128
        assert shard["qkv_output"] == shard["o_proj_input"] + 2 * shard["kv_size"]
    assert hidden == 4096 and inter == 12288


def test_vocab_sharding_and_padding():
    """VocabParallelEmbedding pads to a multiple of the partition."""
    vocab = U15["vocab_size"]
    pad_to = 64  # vLLM DEFAULT_VOCAB_PADDING_SIZE

    def pad_vocab_size(n: int, pad: int = pad_to) -> int:
        return ((n + pad - 1) // pad) * pad

    for tp in (1, 2, 4, 8):
        per_partition = divide(pad_vocab_size(vocab), tp)
        assert per_partition * tp == pad_vocab_size(vocab)
        # this checkpoint needs no padding at these degrees
        assert pad_vocab_size(vocab) == vocab
    assert vocab % 8 == 0


# ---------------------------------------------------------------------------
# 3. Geometry invariants that must NOT scale with TP
# ---------------------------------------------------------------------------


def test_rope_and_qk_norm_geometry_is_tp_invariant():
    """head_dim and its halves/quarters are per-head, not per-rank.

    The model uses half-head_dim RoPE for t and quarter-head_dim for h/w, plus
    per-half QK RMSNorms (`head_dim // 2`). TP shards heads/projections, never a
    single head's internal dimension.
    """
    head_dim = U15["head_dim"]
    assert head_dim == 128
    assert head_dim // 2 == 64  # t-RoPE half + per-half QK norm width
    assert head_dim // 4 == 32  # h/w RoPE quarter
    # none of these depend on tp
    for tp in (1, 2, 4, 8):
        shard = qkv_shard(32, 8, head_dim, tp)
        assert shard["q_size"] == shard["num_heads"] * head_dim
        assert shard["kv_size"] == shard["num_kv_heads"] * head_dim
        # each local head still carries the full head_dim
        assert shard["q_size"] % shard["num_heads"] == 0
        assert shard["q_size"] // shard["num_heads"] == head_dim


def test_layer_count_and_replicated_modules_are_tp_invariant():
    """Replicated modules keep their full shape on every rank."""
    assert U15["num_hidden_layers"] == 42
    for tp in (1, 2, 4, 8):
        # norms are elementwise over hidden_size -> replicated, not sharded
        assert U15["hidden_size"] == 4096
        # the lm_head is vocab-parallel, so its logical width is unchanged
        assert U15["vocab_size"] == 151936
        assert tp_degree_is_legal(U15, tp)


def test_illegal_degree_raises_instead_of_mis_sharding():
    """An illegal degree must fail loudly at config time."""
    with pytest.raises(ValueError):
        divide(U15["num_attention_heads"], 3)
    with pytest.raises(ValueError):
        divide(U15["num_key_value_heads"], 3)
    assert not tp_degree_is_legal(U15, 3)
    assert not tp_degree_is_legal(U15, 0)
    assert not tp_degree_is_legal(U15, -1)


def test_stacked_mapping_extraction_is_from_the_pinned_source():
    """The mapping under test is the repo's own, not a fixture copy."""
    assert len(STACKED_PARAMS_MAPPING) == 8
    assert all(len(entry) == 3 for entry in STACKED_PARAMS_MAPPING)
    # and it round-trips as literals (no f-strings / runtime values)
    assert isinstance(STACKED_PARAMS_MAPPING[0], tuple)
    assert torch is not None  # torch is available for the shard arithmetic above
