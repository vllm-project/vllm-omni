# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B3 prefix-reuse decision: hit/miss parity and identity tests (CPU only).

Covers the phase-3 hook described in `B3_B1_CONTRACT.md`:
`ARDiffusionRequestAdapter.block_hashes` + `enable_caching` in
`vllm_omni/experimental/ar_diffusion/kv_cache/manager.py`.

The helpers here are production candidates, kept alongside their tests so the
cut and identity rules can be reviewed before anything is wired into the cache
manager. Nothing in this file enables caching; the open window/eviction
questions are recorded in the contract note.

Semantics: the mask is `visible[j, i] = (i <= j) or (t_i == t_j)`, read as
"query j may attend to key i". A cut is reusable only if every prefix query
`i < m` attends only to keys `< m` (the row extent).
"""

import ast
import hashlib
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TRANSFORMER_SRC = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py"
)

BRANCHES = ("cond", "uncond", "img_cond")


def _load_create_block_causal_mask():
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
# production-candidate helpers
# ---------------------------------------------------------------------------


def farthest_forward_attention(time_ids) -> list[int]:
    """For each query i, the largest key position it may attend to."""
    n = len(time_ids)
    t = torch.as_tensor(time_ids, dtype=torch.long)
    pos = torch.arange(n)
    vis = (pos.unsqueeze(0) <= pos.unsqueeze(1)) | (t.unsqueeze(1) == t.unsqueeze(0))
    return [int(vis[i].nonzero().max()) for i in range(n)]


def reusable_prefix_tokens(time_ids, matched_tokens: int, chunk_size: int) -> int:
    """Largest chunk-aligned, dependency-closed reusable prefix length.

    `matched_tokens` is what the cache manager believes it matched; the result is
    what may actually be reused. Not `floor(matched/chunk)*chunk` - see
    `test_chunk_floor_is_not_generally_safe`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= matched_tokens <= len(time_ids):
        raise ValueError("matched_tokens is outside the input range")
    farthest = farthest_forward_attention(time_ids)
    best = 0
    for m in range(chunk_size, matched_tokens + 1, chunk_size):
        if all(farthest[i] < m for i in range(m)):
            best = m
    return best


def canonical_block_hash(
    branch: str,
    token_ids,
    block_index: int,
    chunk_size: int,
    *,
    model_epoch: int,
    tp_rank: int,
    parent_hash: str = "",
) -> str:
    """One canonical hash for exactly one cacheable full block.

    Identity covers what the block's computation depends on: branch, this
    block's token ids, the ancestor hash, and the model/placement epoch.
    """
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    start = block_index * chunk_size
    end = start + chunk_size
    if start < 0 or end > len(token_ids):
        raise ValueError("block is not a full block of token_ids")
    payload = "|".join(
        [
            "v1",
            branch,
            f"epoch={model_epoch}",
            f"rank={tp_rank}",
            f"parent={parent_hash}",
            "tokens=" + ",".join(str(int(t)) for t in token_ids[start:end]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def prefix_block_hashes(
    branch: str, token_ids, reusable_tokens: int, chunk_size: int,
    *, model_epoch: int, tp_rank: int,
) -> list[str]:
    """Ancestor-chained hashes for a reusable prefix, one per full block."""
    if reusable_tokens % chunk_size:
        raise ValueError("reusable_tokens must be chunk aligned")
    hashes: list[str] = []
    parent = ""
    for block_index in range(reusable_tokens // chunk_size):
        parent = canonical_block_hash(
            branch, token_ids, block_index, chunk_size,
            model_epoch=model_epoch, tp_rank=tp_rank, parent_hash=parent,
        )
        hashes.append(parent)
    return hashes


def _thw_t(ids, img_start=100, img_ctx=99) -> list[int]:
    """Reproduce `_get_thw_indexes`' time index from the pinned pipeline source."""
    a = torch.as_tensor(ids, dtype=torch.long)
    shift = torch.cat([torch.zeros(1, dtype=torch.long), (a == img_start).long()])[:-1]
    return ((shift + (a != img_ctx).long()).cumsum(0) - 1).tolist()


class FakeKVPool:
    """Deterministic stand-in for a paged KV pool so parity is CPU-testable.

    If any (block hash, position) is missing the pool reports a miss - the
    behaviour a real manager must have once the window has evicted a block.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self._data: dict[tuple[str, int], int] = {}

    def publish(self, hashes, values) -> None:
        for block_index, h in enumerate(hashes):
            for off in range(self.chunk_size):
                self._data[(h, block_index * self.chunk_size + off)] = \
                    values[block_index * self.chunk_size + off]

    def serve(self, hashes) -> tuple[str, list[int]]:
        out: list[int] = []
        for block_index, h in enumerate(hashes):
            for off in range(self.chunk_size):
                key = (h, block_index * self.chunk_size + off)
                if key not in self._data:
                    return "miss", []
                out.append(self._data[key])
        return "hit", out

    def evict_blocks(self, num_blocks: int) -> None:
        for key in [k for k in self._data if k[1] < num_blocks * self.chunk_size]:
            del self._data[key]


def _prefix_values(token_ids) -> list[int]:
    return [int(t) * 31 + i for i, t in enumerate(token_ids)]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_closure_matches_the_repo_mask_exhaustively():
    """The helper's closure notion must equal the mask-derived one."""
    checked = 0
    for ln in range(1, 7):
        for t in range(3):
            ids = [t] * ln
            mask = create_block_causal_mask(torch.as_tensor(ids))[0, 0] == 0.0
            far_mask = [int(mask[i].nonzero().max()) for i in range(ln)]
            far_helper = farthest_forward_attention(ids)
            assert far_mask == far_helper, ids
            for m in range(ln + 1):
                assert (all(far_mask[i] < m for i in range(m))
                        == all(far_helper[i] < m for i in range(m)))
                checked += 1
    assert checked > 0


def test_text_only_prefix_behaves_as_plain_causal_apc():
    """`t = arange(len)` (the T2I text prefix) -> plain causal alignment floor."""
    for n in (16, 32, 48, 64, 100):
        assert reusable_prefix_tokens(list(range(n)), n, 16) == n // 16 * 16


def test_chunk_floor_is_not_generally_safe():
    """A tied image run straddling a chunk boundary makes the floor unsafe.

    Real T2I index shape from `_get_thw_indexes`: 9 text tokens, an 8-token
    `<IMG_CONTEXT>` run, then 6 text tokens. The run's last member is position
    17, so at chunk_size=16 the naive floor (16) is open and the cut retreats.
    """
    ids = list(range(1, 10)) + [100] + [99] * 8 + list(range(10, 16))
    t = _thw_t(ids)
    n = len(t)
    assert farthest_forward_attention(t)[10] == 17
    assert (n // 16) * 16 == 16
    assert reusable_prefix_tokens(t, n, 16) == 0
    # with a chunk size that happens to respect the run, the floor is closed
    assert reusable_prefix_tokens(t, n, 8) == 24


def test_cold_miss_partial_hit_and_full_hit_parity():
    chunk = 16
    token_ids = list(range(64))
    values = _prefix_values(token_ids)
    hashes = prefix_block_hashes("cond", token_ids, 64, chunk, model_epoch=1, tp_rank=0)
    assert len(hashes) == 4

    pool = FakeKVPool(chunk)
    assert pool.serve(hashes)[0] == "miss"  # cold

    pool.publish(hashes, values)
    status, served = pool.serve(hashes)  # full hit
    assert status == "hit"
    assert served == values[: len(served)] == values

    # partial: only the first two blocks published
    partial = FakeKVPool(chunk)
    partial.publish(hashes[:2], values)
    assert partial.serve(hashes[:2])[0] == "hit"
    assert partial.serve(hashes)[0] == "miss"


def test_full_hit_leaves_zero_suffix_but_a_boundary_decision():
    """reused == logical is not 'nothing left to do' (plan 4.3)."""
    token_ids = list(range(64))
    reusable = reusable_prefix_tokens(token_ids, 64, 16)
    assert reusable == len(token_ids)
    assert len(token_ids) - reusable == 0


def test_ancestor_hash_separates_differing_prefixes():
    chunk = 16
    a = prefix_block_hashes("cond", list(range(64)), 64, chunk, model_epoch=1, tp_rank=0)
    b = prefix_block_hashes(
        "cond", [99] * 16 + list(range(16, 64)), 64, chunk, model_epoch=1, tp_rank=0
    )
    assert a[0] != b[0]
    assert all(x != y for x, y in zip(a, b))


def test_branches_never_share_a_prefix_identity():
    chunk = 16
    token_ids = list(range(64))
    all_hashes = {
        br: prefix_block_hashes(br, token_ids, 64, chunk, model_epoch=1, tp_rank=0)
        for br in BRANCHES
    }
    flat = [h for hs in all_hashes.values() for h in hs]
    assert len(set(flat)) == len(flat) == 12
    with pytest.raises(ValueError, match="unknown branch"):
        canonical_block_hash("neg", token_ids, 0, chunk, model_epoch=1, tp_rank=0)


def test_epoch_and_rank_are_part_of_the_identity():
    chunk = 16
    token_ids = list(range(64))
    base = prefix_block_hashes("cond", token_ids, 64, chunk, model_epoch=1, tp_rank=0)
    assert base[0] != prefix_block_hashes(
        "cond", token_ids, 64, chunk, model_epoch=2, tp_rank=0)[0]
    assert base[0] != prefix_block_hashes(
        "cond", token_ids, 64, chunk, model_epoch=1, tp_rank=1)[0]


def test_evicting_a_published_block_must_miss_not_silently_hit():
    """The eviction hazard: a sliding window drops the prefix it published."""
    chunk = 16
    token_ids = list(range(64))
    values = _prefix_values(token_ids)
    hashes = prefix_block_hashes("cond", token_ids, 64, chunk, model_epoch=1, tp_rank=0)

    pool = FakeKVPool(chunk)
    pool.publish(hashes, values)
    assert pool.serve(hashes)[0] == "hit"

    pool.evict_blocks(1)  # drop the oldest published block
    assert pool.serve(hashes)[0] == "miss", (
        "a hit here would serve KV the window already evicted"
    )


def test_input_validation():
    token_ids = list(range(64))
    with pytest.raises(ValueError, match="chunk_size"):
        reusable_prefix_tokens(token_ids, 16, 0)
    with pytest.raises(ValueError, match="matched_tokens"):
        reusable_prefix_tokens(token_ids, -1, 16)
    with pytest.raises(ValueError, match="chunk aligned"):
        prefix_block_hashes("cond", token_ids, 63, 16, model_epoch=1, tp_rank=0)
    with pytest.raises(ValueError, match="full block"):
        canonical_block_hash("cond", token_ids, 4, 16, model_epoch=1, tp_rank=0)
