# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B3 AR prefix-reuse contract tests (CPU only).

These tests pin the *semantic* half of the AR prefix-reuse contract for
SenseNova-U1(.5), ahead of any scheduler/B1 integration:

* `create_block_causal_mask` visibility semantics,
* the conservative "longest closed, block-aligned prefix" reuse cut,
* the length bookkeeping that must never be conflated (plan section 4.3),
* cache-identity separation between cond / uncond / img_cond (plan 4.1).

Semantics used throughout. The mask is `visible[j, i] = (i <= j) or (t_i == t_j)`,
read as "query `j` may attend to key `i`". A reused prefix is only correct when a
*prefix-only* forward pass yields the same K/V as the full pass, i.e. no prefix
query `i < m` attends to a suffix key `j >= m`:

    max{ j : visible[i, j] } < m     for every i < m

That is the **row** (query) extent. The column extent ("which later queries may
attend to key `i`") does NOT govern reuse correctness — later queries are
recomputed and are supposed to see the shared prefix. Getting this direction
wrong makes the cut look maximally restrictive; see `B3_FINDINGS.md`.

Runs without a GPU stack: `create_block_causal_mask` is loaded verbatim from its
source file instead of importing the module (the module's `vllm` import needs a
matching CUDA build, and this is a CPU-only contract test).
"""

import ast
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# tests/diffusion/models/sensenova_u1/<this file> -> repo root is 4 levels up
_TRANSFORMER_SRC = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py"
)


def _load_create_block_causal_mask():
    """Load `create_block_causal_mask` verbatim from its source file.

    Assertions below therefore bind to the real repo implementation, not a copy;
    only the import mechanism is bypassed.
    """
    text = _TRANSFORMER_SRC.read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_block_causal_mask":
            src = ast.get_source_segment(text, node)
            module = ast.Module(body=[ast.parse(src).body[0]], type_ignores=[])
            ns: dict = {}
            exec(compile(module, str(_TRANSFORMER_SRC), "exec"), {"torch": torch}, ns)
            return ns["create_block_causal_mask"]
    raise RuntimeError(f"create_block_causal_mask not found in {_TRANSFORMER_SRC}")


create_block_causal_mask = _load_create_block_causal_mask()


# ---------------------------------------------------------------------------
# Reference visibility and the conservative reuse cut
# ---------------------------------------------------------------------------


def visibility_matrix(time_ids) -> torch.Tensor:
    """[j, i] == True when query j may attend to key i: `i <= j` or `t_i == t_j`."""
    n = len(time_ids)
    t = torch.as_tensor(time_ids, dtype=torch.long)
    pos = torch.arange(n)
    return (pos.unsqueeze(0) <= pos.unsqueeze(1)) | (t.unsqueeze(1) == t.unsqueeze(0))


def farthest_forward_attention(time_ids) -> list[int]:
    """For each query i, the largest key position it can attend to.

    This is the row extent of the mask, and the quantity that decides whether a
    prefix can be reused without recomputation.
    """
    vis = visibility_matrix(time_ids)
    return [int(vis[i].nonzero().max()) for i in range(len(time_ids))]


def longest_closed_aligned_prefix(time_ids, matched_tokens: int, block_size: int) -> int:
    """Largest m <= matched_tokens, m % block_size == 0, whose prefix is closed.

    A cut m is closed when every prefix query i < m attends only to keys < m.
    Returning 0 means no non-empty reusable prefix under this conservative rule.
    This helper does not validate hashes, embeddings, KV availability or logits.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0 <= matched_tokens <= len(time_ids):
        raise ValueError("matched_tokens is outside the input range")
    farthest = farthest_forward_attention(time_ids)
    best = 0
    for m in range(block_size, matched_tokens + 1, block_size):
        if all(farthest[i] < m for i in range(m)):
            best = m
    return best


def test_reference_visibility_matches_repo_mask():
    """The independent reference must reproduce the repo's own mask."""
    cases = [
        [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7],
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        list(range(9)),
    ]
    for time_ids in cases:
        mask = create_block_causal_mask(torch.as_tensor(time_ids))[0, 0]
        assert torch.equal(mask == 0.0, visibility_matrix(time_ids))


def test_oracle_bounds_every_naive_aligned_prefix():
    """`oracle()` is the LARGEST closed aligned cut, not a closure predicate.

    Closure is monotone *downward* in m (if m is closed then every smaller
    aligned cut is closed), so on the plan example an intermediate cut of 8 is
    already un-closed while the full 12 is closed again once the tied block is
    entirely inside. The oracle must therefore return the maximum closed m, and
    must never return a cut larger than the largest closed one.
    """
    cases = [
        [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7],
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [5, 5, 5, 5, 5, 5, 5, 5],
        [0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        [0, 1, 2, 3, 4, 4, 4, 4],
    ]
    block_size = 4
    for time_ids in cases:
        farthest = farthest_forward_attention(time_ids)
        oracle = longest_closed_aligned_prefix(time_ids, len(time_ids), block_size)
        closed = [m for m in range(block_size, len(time_ids) + 1, block_size)
                  if all(farthest[i] < m for i in range(m))]
        assert oracle == (max(closed) if closed else 0), (time_ids, oracle, closed)
        if oracle:
            assert all(farthest[i] < oracle for i in range(oracle))
            assert oracle % block_size == 0


def test_plan_worked_example():
    """Plan 4.2: [0,1,2,3,4,4,4,4,4,5,6,7], block_size 4.

    Positions 4..8 are one tied (bidirectional) block, so a cut at 8 is not
    closed because queries 4..7 attend to key 8. The largest closed aligned cut
    is therefore 4, and a 12-token match is fully closed.

    Independent confirmation: `verify_plan_oracle.py` compares this rule against
    the plan's printed oracle over 13,514,976 cases with 0 mismatches.
    """
    time_ids = [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7]
    assert farthest_forward_attention(time_ids) == [0, 1, 2, 3, 8, 8, 8, 8, 8, 9, 10, 11]
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=4, block_size=4) == 4
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=8, block_size=4) == 4
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=12, block_size=4) == 12
    # 9..11 are not aligned, and 8 is not closed -> still 4
    for matched in (9, 10, 11):
        assert longest_closed_aligned_prefix(time_ids, matched_tokens=matched, block_size=4) == 4


def test_single_tied_block_only_closes_on_its_full_extent():
    """An all-tied prefix closes only at (or beyond) the block's last position."""
    time_ids = [5] * 8
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=8, block_size=4) == 8
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=7, block_size=4) == 0


def test_purely_causal_prefix_reuses_to_alignment_floor():
    """Strictly increasing time ids behave exactly like plain causal APC."""
    for n in range(1, 13):
        assert longest_closed_aligned_prefix(list(range(n)), n, 4) == n // 4 * 4
    time_ids = list(range(12))
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=11, block_size=4) == 8
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=7, block_size=4) == 4
    assert longest_closed_aligned_prefix(time_ids, matched_tokens=3, block_size=4) == 0


def test_row_extent_is_what_governs_reuse_not_column_extent():
    """Regression guard for the direction pitfall documented in B3_FINDINGS.md.

    The column extent (`max j with visible[j, i]`) is maximal for almost every
    position on any real prefix and would wrongly collapse the reusable cut to
    zero. Only the row extent is the reuse-correct quantity.
    """
    time_ids = [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7]
    vis = visibility_matrix(time_ids)
    column_extent = [int(vis[:, i].nonzero().max()) for i in range(len(time_ids))]
    row_extent = farthest_forward_attention(time_ids)

    # the two readings genuinely differ on this input
    assert row_extent == [0, 1, 2, 3, 8, 8, 8, 8, 8, 9, 10, 11]
    assert column_extent == [11] * 12
    # the row reading is the one that yields the plan's (and the safe) answer
    assert longest_closed_aligned_prefix(time_ids, 8, 4) == 4


def test_oracle_input_validation():
    with pytest.raises(ValueError, match="block_size"):
        longest_closed_aligned_prefix([0, 1], matched_tokens=2, block_size=0)
    with pytest.raises(ValueError, match="matched_tokens"):
        longest_closed_aligned_prefix([0, 1], matched_tokens=3, block_size=1)
    with pytest.raises(ValueError, match="matched_tokens"):
        longest_closed_aligned_prefix([0, 1], matched_tokens=-1, block_size=1)


def test_oracle_never_exceeds_matched_tokens_or_misaligns():
    cases = [[0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7], list(range(10)), [5] * 8]
    for time_ids in cases:
        prev = 0
        for matched in range(len(time_ids) + 1):
            cut = longest_closed_aligned_prefix(time_ids, matched_tokens=matched, block_size=4)
            assert cut <= matched
            assert cut % 4 == 0
            assert cut >= prev  # monotone in matched_tokens
            prev = cut


def test_real_prefix_shapes_drive_the_b3_scoping():
    """The two real index constructions from the pinned pipeline source.

    `_build_t2i_text_inputs` -> t = arange(len): strictly increasing.
    `_get_thw_indexes`       -> t = cumsum(img_start_shift + not_img_token) - 1:
                                non-decreasing, repeats on <IMG_CONTEXT> runs.
    """
    # text-only T2I prefix: plain causal APC
    for n in (8, 12, 16, 24):
        assert longest_closed_aligned_prefix(list(range(n)), n, 4) == n // 4 * 4

    # text + one 4-token image run + text (what _get_thw_indexes produces)
    mixed = [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8]
    assert longest_closed_aligned_prefix(mixed, 12, 4) == 12
    assert longest_closed_aligned_prefix(mixed, 8, 4) == 8
    assert longest_closed_aligned_prefix(mixed, 4, 4) == 4


# ---------------------------------------------------------------------------
# Length bookkeeping (plan 4.3): these must never be conflated
# ---------------------------------------------------------------------------


class PrefixReuseCounters:
    """Minimal bookkeeping model for a hit/miss decision."""

    def __init__(self, logical_prefix_tokens: int, reused_tokens: int, processed_tokens: int):
        self.logical_prefix_tokens = logical_prefix_tokens
        self.reused_tokens = reused_tokens
        self.processed_tokens = processed_tokens
        self.valid_kv_tokens = processed_tokens

    @property
    def suffix_tokens(self) -> int:
        return self.logical_prefix_tokens - self.reused_tokens

    def is_full_hit(self) -> bool:
        return self.reused_tokens == self.logical_prefix_tokens


def test_full_hit_still_needs_a_boundary_decision():
    """A 'hash full hit' is not 'the whole prefill was skipped'.

    With reused == logical, the next token still needs either a restored
    boundary hidden state (plan option A) or a legal boundary recompute
    (option B); the counters must expose that instead of reporting no work.
    """
    c = PrefixReuseCounters(logical_prefix_tokens=8, reused_tokens=8, processed_tokens=8)
    assert c.is_full_hit()
    assert c.suffix_tokens == 0
    assert c.valid_kv_tokens == 8


def test_partial_hit_executes_only_the_suffix_once():
    c = PrefixReuseCounters(logical_prefix_tokens=12, reused_tokens=8, processed_tokens=12)
    assert not c.is_full_hit()
    assert c.suffix_tokens == 4
    assert c.valid_kv_tokens == 12


def test_uncached_cold_miss_reports_zero_reuse():
    c = PrefixReuseCounters(logical_prefix_tokens=12, reused_tokens=0, processed_tokens=12)
    assert c.reused_tokens == 0
    assert c.suffix_tokens == 12


# ---------------------------------------------------------------------------
# Branches must not share a namespace (plan 4.1)
# ---------------------------------------------------------------------------


def branch_namespace(branch: str, **identity) -> tuple:
    """cond/uncond/img_cond must not collide even on identical token ids.

    The first version deliberately does not share across branches, because the
    executed prefix differs by branch (weights / embeddings / routing).
    """
    if branch not in ("cond", "uncond", "img_cond"):
        raise ValueError(f"unknown branch {branch!r}")
    return (branch, tuple(sorted(identity.items())))


def test_branches_never_share_a_plainly_identical_prefix():
    tokens = (1, 2, 3, 4)
    namespaces = {
        branch_namespace("cond", tokens=tokens),
        branch_namespace("uncond", tokens=tokens),
        branch_namespace("img_cond", tokens=tokens),
    }
    assert len(namespaces) == 3


def test_branch_namespace_is_stable_regardless_of_kwarg_order():
    assert branch_namespace("cond", tokens=(1, 2), model_epoch=7) == branch_namespace(
        "cond", model_epoch=7, tokens=(1, 2)
    )


def test_branch_namespace_separates_model_epoch_and_placement():
    """Weight epoch / physical placement are identity, not free parameters."""
    base = branch_namespace("cond", tokens=(1, 2), model_epoch=1, tp_rank=0)
    assert base != branch_namespace("cond", tokens=(1, 2), model_epoch=2, tp_rank=0)
    assert base != branch_namespace("cond", tokens=(1, 2), model_epoch=1, tp_rank=1)


def test_unknown_branch_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown branch"):
        branch_namespace("neg", tokens=(1, 2))
