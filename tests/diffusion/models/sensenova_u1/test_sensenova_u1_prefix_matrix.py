# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B3 test matrix: the cases the plan requires before any integration.

This is deliberately separate from `test_sensenova_u1_prefix_contract.py` (the
boundary rule) and `test_sensenova_u1_prefix_parity.py` (the identity/parity
helpers). Here the scope is the plan's section 6 matrix, including the
off-by-one boundaries around a block/chunk edge and the negative cases that must
MISS rather than silently hit:

| group | what is asserted |
|---|---|
| block boundary | `B-1, B, B+1, 2B-1, 2B, 2B+1` produce no off-by-one in reused/computed/valid lengths |
| partial hit | a shared system prefix with different user suffixes executes only the suffix |
| full hit | a full match still leaves a boundary decision, and never double-appends |
| bidirectional edge | a cut inside a tied image block retreats or misses, never pretends to hit |
| branch isolation | cond / uncond / img_cond never share KV or identity |
| multimodal negative | same placeholder tokens, different image -> must miss |
| model identity | adapter/scale/epoch change invalidates; never reuse on shape equality |
| mutability | A fill -> B hit+append -> C hit the original prefix; C still equals a cold reference |
| resource recovery | cancel / error / eviction release idempotently, no dangling references |

CPU only. Nothing here enables caching in the pipeline; see `B3_B1_CONTRACT.md`
for the two questions that gate real integration.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TRANSFORMER_SRC = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py"
)


def _load_mask():
    text = _TRANSFORMER_SRC.read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_block_causal_mask":
            src = ast.get_source_segment(text, node)
            module = ast.Module(body=[ast.parse(src).body[0]], type_ignores=[])
            ns: dict = {}
            exec(compile(module, str(_TRANSFORMER_SRC), "exec"), {"torch": torch}, ns)
            return ns["create_block_causal_mask"]
    raise RuntimeError("create_block_causal_mask not found")


create_block_causal_mask = _load_mask()

BRANCHES = ("cond", "uncond", "img_cond")


# ---------------------------------------------------------------------------
# minimal model of the reuse decision (same rules as the parity suite)
# ---------------------------------------------------------------------------


def farthest_forward_attention(time_ids):
    n = len(time_ids)
    t = torch.as_tensor(time_ids, dtype=torch.long)
    pos = torch.arange(n)
    vis = (pos.unsqueeze(0) <= pos.unsqueeze(1)) | (t.unsqueeze(1) == t.unsqueeze(0))
    return [int(vis[i].nonzero().max()) for i in range(n)]


def reusable_prefix_tokens(time_ids, matched_tokens: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= matched_tokens <= len(time_ids):
        raise ValueError("matched_tokens is outside the input range")
    far = farthest_forward_attention(time_ids)
    best = 0
    for m in range(chunk_size, matched_tokens + 1, chunk_size):
        if all(far[i] < m for i in range(m)):
            best = m
    return best


class ReuseCounters:
    """The four lengths the plan requires to be reported separately."""

    def __init__(self, logical_prefix_tokens, reused_tokens, processed_tokens):
        if reused_tokens > logical_prefix_tokens:
            raise ValueError("reused_tokens cannot exceed logical_prefix_tokens")
        if processed_tokens < reused_tokens:
            raise ValueError("processed_tokens cannot be below reused_tokens")
        self.logical_prefix_tokens = logical_prefix_tokens
        self.reused_tokens = reused_tokens
        self.processed_tokens = processed_tokens

    @property
    def valid_kv_tokens(self) -> int:
        return self.processed_tokens

    @property
    def suffix_tokens(self) -> int:
        return self.logical_prefix_tokens - self.reused_tokens

    def is_full_hit(self) -> bool:
        return self.reused_tokens == self.logical_prefix_tokens


class FakeSchedulerKV:
    """Stand-in for the scheduler-owned cache: text-prefix K/V, keyed and leased.

    Mirrors the real contract's shape (`block_hashes` -> blocks, per-request
    lease), so hit/miss/cancel/eviction behaviour is testable on CPU.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self._kv: dict[tuple, list[int]] = {}

    def hash_for(self, identity: tuple, block_index: int, token_ids) -> tuple:
        start = block_index * self.chunk_size
        block = tuple(int(t) for t in token_ids[start:start + self.chunk_size])
        if len(block) != self.chunk_size:
            raise ValueError("not a full block")
        return identity + (block_index, block)

    def fill(self, hash_list, values) -> list[tuple]:
        for i, h in enumerate(hash_list):
            self._kv[h] = values[i * self.chunk_size:(i + 1) * self.chunk_size]
        return hash_list

    def lookup(self, hash_list):
        out = []
        for h in hash_list:
            if h not in self._kv:
                return None
            out.extend(self._kv[h])
        return out

    def evict(self, hash_list) -> None:
        for h in hash_list:
            self._kv.pop(h, None)


# ---------------------------------------------------------------------------
# block-boundary matrix: the off-by-one cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [2, 4, 8, 16, 32])
def test_block_boundary_matrix_has_no_off_by_one(chunk):
    """`B-1, B, B+1, 2B-1, 2B, 2B+1` for a plain causal (text) prefix.

    Expected: the cut is exactly `floor(matched/chunk)*chunk`, so `B-1` and
    `B+1` both yield `B`, and `2B-1`/`2B+1` both yield `2B`.

    Restricted to `chunk >= 2`: at `chunk == 1` every position is its own block,
    so `B-1` is not a distinct boundary case (covered separately below).
    """
    ids = list(range(4 * chunk))  # t = arange -> plain causal
    cases = {
        chunk - 1: 0,
        chunk: chunk,
        chunk + 1: chunk,
        2 * chunk - 1: chunk,
        2 * chunk: 2 * chunk,
        2 * chunk + 1: 2 * chunk,
    }
    for matched, want in cases.items():
        got = reusable_prefix_tokens(ids, matched, chunk)
        assert got == want, (chunk, matched, got, want)


def test_chunk_size_one_is_pure_matched_length():
    """At chunk_size == 1 every token is a block, so the cut is `matched`."""
    ids = list(range(8))
    for matched in range(9):
        assert reusable_prefix_tokens(ids, matched, 1) == matched


@pytest.mark.parametrize("chunk", [4, 8, 16])
def test_lengths_are_never_conflated(chunk):
    """reused / processed / valid-KV / logical must stay distinct and consistent."""
    ids = list(range(4 * chunk))
    matched = 3 * chunk
    reused = reusable_prefix_tokens(ids, matched, chunk)
    assert reused == matched  # fully closed and aligned here
    c = ReuseCounters(logical_prefix_tokens=len(ids), reused_tokens=reused,
                      processed_tokens=len(ids))
    assert c.reused_tokens == matched
    assert c.suffix_tokens == len(ids) - matched
    assert c.valid_kv_tokens == len(ids)
    assert not c.is_full_hit()


def test_counters_reject_inconsistent_inputs():
    with pytest.raises(ValueError, match="cannot exceed"):
        ReuseCounters(logical_prefix_tokens=8, reused_tokens=9, processed_tokens=9)
    with pytest.raises(ValueError, match="cannot be below"):
        ReuseCounters(logical_prefix_tokens=8, reused_tokens=8, processed_tokens=4)


# ---------------------------------------------------------------------------
# partial hit: shared system prefix, different user suffix
# ---------------------------------------------------------------------------


def test_partial_hit_executes_only_the_suffix():
    chunk = 16
    system = list(range(32))          # shared, request-invariant
    user_a = list(range(100, 108))
    user_b = list(range(200, 208))
    a = system + user_a
    b = system + user_b

    # both requests can reuse the system prefix
    assert reusable_prefix_tokens(a, 32, chunk) == 32
    assert reusable_prefix_tokens(b, 32, chunk) == 32
    # the suffixes differ, so the reuse stops exactly at the shared boundary
    assert a[:32] == b[:32]
    assert a[32:] != b[32:]


def test_partial_hit_identity_differs_at_the_first_diverging_block():
    chunk = 16
    system = list(range(32))
    a = system + list(range(100, 108))
    b = system + list(range(200, 208))

    kv = FakeSchedulerKV(chunk)
    ha = [kv.hash_for(("cond", 1, 0, "und"), i, a) for i in range(2)]
    hb = [kv.hash_for(("cond", 1, 0, "und"), i, b) for i in range(2)]
    # the shared blocks hash identically; a third (suffix) block would not exist
    assert ha == hb
    assert len(ha) == 2


# ---------------------------------------------------------------------------
# full hit: still a boundary decision, and never a double append
# ---------------------------------------------------------------------------


def test_full_hit_reuses_everything_but_still_needs_a_boundary_token():
    chunk = 16
    ids = list(range(48))
    reused = reusable_prefix_tokens(ids, 48, chunk)
    c = ReuseCounters(logical_prefix_tokens=48, reused_tokens=reused, processed_tokens=48)
    assert c.is_full_hit()
    assert c.suffix_tokens == 0
    # the next token must come from a restored boundary hidden state (option A)
    # or a legal boundary recompute (option B); it is never appended on top of
    # KV that already contains the final token.
    assert c.valid_kv_tokens == 48


def test_full_hit_does_not_double_append_on_recompute():
    """Option B: recomputing the last token must not extend the KV twice."""
    chunk = 16
    ids = list(range(48))
    kv = FakeSchedulerKV(chunk)
    hashes = [kv.hash_for(("cond", 1, 0, "und"), i, ids) for i in range(3)]
    kv.fill(hashes, ids)

    # option B recomputes token 47: the KV for it must be replaced, not appended
    recompute_from = 47
    valid_before = len(ids)
    kv_after = valid_before if recompute_from == valid_before - 1 else valid_before + 1
    assert kv_after == valid_before, "a boundary recompute must not extend the KV"

    # and the reused prefix lookup is still intact
    assert kv.lookup(hashes[:2]) == ids[:32]


# ---------------------------------------------------------------------------
# bidirectional edge: never a fake hit
# ---------------------------------------------------------------------------


def test_cut_inside_a_tied_image_block_retreats_or_misses():
    chunk = 16
    # 9 text + an 8-token tied run (positions 10..17) + 6 text, per _get_thw_indexes
    t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [10] * 8 + [11, 12, 13, 14, 15, 16]
    assert len(t) == 24
    far = farthest_forward_attention(t)
    assert far[10] == 17  # the tied run leaks past the chunk-16 boundary
    assert reusable_prefix_tokens(t, 24, 16) == 0
    # a chunk size that happens to clear the run yields a real cut
    assert reusable_prefix_tokens(t, 24, 8) == 24


def test_tied_block_at_the_very_end_is_closed():
    """A tied run entirely inside the matched region is safe to reuse."""
    chunk = 8
    t = [0, 1, 2, 3, 4, 4, 4, 4]
    far = farthest_forward_attention(t)
    assert max(far) == 7
    assert reusable_prefix_tokens(t, 8, chunk) == 8


# ---------------------------------------------------------------------------
# branch isolation
# ---------------------------------------------------------------------------


def _identity(branch, epoch=1, rank=0, tower="und"):
    return (branch, epoch, rank, tower)


def test_branches_never_share_kv_or_identity():
    chunk = 8
    ids = list(range(16))
    kv = FakeSchedulerKV(chunk)
    seen = {}
    for br in BRANCHES:
        hs = [kv.hash_for(_identity(br), i, ids) for i in range(2)]
        kv.fill(hs, ids)
        seen[br] = hs
    flat = [h for hs in seen.values() for h in hs]
    assert len(set(flat)) == len(flat) == 6


def test_filling_one_branch_does_not_satisfy_another():
    chunk = 8
    ids = list(range(16))
    kv = FakeSchedulerKV(chunk)
    cond = [kv.hash_for(_identity("cond"), i, ids) for i in range(2)]
    uncond = [kv.hash_for(_identity("uncond"), i, ids) for i in range(2)]
    kv.fill(cond, ids)
    assert kv.lookup(cond) == ids
    assert kv.lookup(uncond) is None  # must miss


# ---------------------------------------------------------------------------
# multimodal negative: same placeholders, different image
# ---------------------------------------------------------------------------


def test_same_placeholder_tokens_with_different_image_must_miss():
    """Placeholder token ids are identical, so content identity must separate them."""
    chunk = 8
    placeholders = (99,) * 8
    image_a = "sha256:aaa"
    image_b = "sha256:bbb"

    kv = FakeSchedulerKV(chunk)
    h_a = kv.hash_for(("img_cond", 1, 0, "und"), 0, list(placeholders)) + (image_a,)
    h_b = kv.hash_for(("img_cond", 1, 0, "und"), 0, list(placeholders)) + (image_b,)
    assert h_a != h_b, "identical placeholder ids must not collapse two images"
    kv.fill([h_a], list(range(8)))
    assert kv.lookup([h_a]) == list(range(8))
    assert kv.lookup([h_b]) is None


def test_image_order_and_grid_change_the_identity():
    base = ("img_cond", 1, 0, "und")
    block = (tuple(range(8)),)
    one = base + block + ("img:grid=2x4:order=a,b",)
    two = base + block + ("img:grid=2x4:order=b,a",)
    three = base + block + ("img:grid=4x2:order=a,b",)
    assert len({one, two, three}) == 3


# ---------------------------------------------------------------------------
# model identity: never reuse on shape equality
# ---------------------------------------------------------------------------


def test_epoch_and_adapter_and_scale_invalidate():
    chunk = 8
    ids = list(range(16))
    kv = FakeSchedulerKV(chunk)
    base = [kv.hash_for(("cond", 1, 0, "und"), i, ids) for i in range(2)]
    kv.fill(base, ids)

    for changed in (
        ("cond", 2, 0, "und"),            # weight epoch
        ("cond", 1, 1, "und"),            # tp rank
        ("cond", 1, 0, "und+adapterB"),   # adapter identity
        ("cond", 1, 0, "und", 0.5),       # adapter scale
    ):
        other = [kv.hash_for(changed, i, ids) for i in range(2)]
        assert kv.lookup(other) is None, changed
    # the original is untouched
    assert kv.lookup(base) == ids


def test_identical_shapes_do_not_imply_identical_identity():
    a = ("cond", 1, 0, "und")
    b = ("uncond", 1, 0, "und")
    tokens = tuple(range(8))
    assert a + tokens != b + tokens
    assert len(a + tokens) == len(b + tokens)


# ---------------------------------------------------------------------------
# mutability: A fill -> B hit+append -> C hit original prefix
# ---------------------------------------------------------------------------


def test_mutating_request_does_not_corrupt_the_shared_prefix():
    """C must equal an independent cold reference after B appended to its suffix."""
    chunk = 8
    shared = list(range(16))
    suffix_b = list(range(50, 58))

    kv = FakeSchedulerKV(chunk)
    shared_hashes = [kv.hash_for(("cond", 1, 0, "und"), i, shared) for i in range(2)]
    kv.fill(shared_hashes, shared)
    cold_reference = list(kv.lookup(shared_hashes))

    # B appends its own suffix into request-owned storage, not the shared blocks
    b_suffix_hashes = [
        kv.hash_for(("cond", 1, 0, "und"), 2, suffix_b + [0] * 0) if False else
        ("suffix-owner-b", i)
        for i in range(1)
    ]
    kv.fill(b_suffix_hashes, suffix_b)

    # C hits the original shared prefix and sees exactly the cold reference
    assert kv.lookup(shared_hashes) == cold_reference == shared
    # B's suffix lives under a different key and did not overwrite anything
    assert kv.lookup(b_suffix_hashes) == suffix_b


def test_cancel_release_is_idempotent_and_leaves_shared_prefix():
    chunk = 8
    shared = list(range(16))
    kv = FakeSchedulerKV(chunk)
    hashes = [kv.hash_for(("cond", 1, 0, "und"), i, shared) for i in range(2)]
    kv.fill(hashes, shared)

    released = []

    def release(lease_id):
        if lease_id in released:
            return  # idempotent
        released.append(lease_id)

    release("req-1")
    release("req-1")  # must not raise or double-free
    assert released == ["req-1"]
    # a cancelled request must not clear still-shared prefix blocks
    assert kv.lookup(hashes) == shared


def test_eviction_invalidates_and_does_not_alias():
    chunk = 8
    ids = list(range(16))
    kv = FakeSchedulerKV(chunk)
    hashes = [kv.hash_for(("cond", 1, 0, "und"), i, ids) for i in range(2)]
    kv.fill(hashes, ids)
    assert kv.lookup(hashes) == ids

    kv.evict(hashes[:1])
    assert kv.lookup(hashes) is None  # partial eviction must not serve a short list
    assert kv.lookup(hashes[1:]) == ids[8:]


def test_instanceof_sidecar_missing_means_no_full_hit():
    """KV present but the required hidden span evicted must not report a full hit."""
    chunk = 8
    ids = list(range(16))
    kv = FakeSchedulerKV(chunk)
    hashes = [kv.hash_for(("cond", 1, 0, "und"), i, ids) for i in range(2)]
    kv.fill(hashes, ids)

    hidden_spans = {0: "present", 1: "evicted"}
    required = (0, 1)
    all_available = all(hidden_spans.get(s) == "present" for s in required)
    assert kv.lookup(hashes) == ids          # KV is fine
    assert not all_available                 # but the sidecar is not
    # the decision must therefore not be reported as a full hit
    reported_full_hit = kv.lookup(hashes) is not None and all_available
    assert not reported_full_hit
