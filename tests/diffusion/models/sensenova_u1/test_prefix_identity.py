# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the B3 canonical block-hash computation (CPU only).

`vllm_omni/diffusion/models/sensenova_u1/prefix_identity.py` is the hash-production
step that `DiffusionKVRequest.block_hashes` documents but nothing populates yet.
It is pure and imports no `vllm`, so these tests exercise it directly.

Properties that matter for cache correctness: a hash must be stable across
processes, and every layer of the identity (branch, epoch, rank topology,
ancestor, block tokens) must actually change it. A layer that silently fails to
separate two computations lets one request reuse another's KV.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_MODULE_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/sensenova_u1/prefix_identity.py"


def _load():
    """Load the module by path without importing the `vllm_omni` package.

    The module must be registered in `sys.modules` before execution: dataclasses
    resolve `cls.__module__` through it, and a missing entry raises
    `AttributeError: 'NoneType' object has no attribute '__dict__'`.
    """
    name = "prefix_identity_under_test"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


pi = _load()
BLOCK = 4


def _ident(**kw):
    base = dict(branch="cond", model_epoch=1, tp_rank=0, tp_size=1)
    base.update(kw)
    return pi.PrefixCacheIdentity(**base)


# ---------------------------------------------------------------------------
# identity validation
# ---------------------------------------------------------------------------


def test_unknown_branch_is_rejected():
    with pytest.raises(ValueError, match="unknown branch"):
        _ident(branch="negative")


def test_rank_topology_is_validated():
    with pytest.raises(ValueError, match="tp_size"):
        _ident(tp_size=0)
    with pytest.raises(ValueError, match="tp_rank"):
        _ident(tp_rank=2, tp_size=2)
    with pytest.raises(ValueError, match="model_epoch"):
        _ident(model_epoch=-1)


# ---------------------------------------------------------------------------
# determinism across processes (python's hash() must not be used)
# ---------------------------------------------------------------------------


def test_hashes_are_stable_across_processes():
    """A cache identity is worthless if it differs per process."""
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('m', r'{_MODULE_PATH}');"
        "m=importlib.util.module_from_spec(spec);sys.modules['m']=m;spec.loader.exec_module(m);"
        "i=m.PrefixCacheIdentity(branch='cond',model_epoch=1,tp_rank=0,tp_size=1);"
        "print(m.canonical_block_hash(i,list(range(8)),0,4).hex())"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        for _ in range(2)
    }
    assert len(outs) == 1, f"hash is not stable across processes: {outs}"
    assert outs.pop() == pi.canonical_block_hash(_ident(), list(range(8)), 0, BLOCK).hex()


def test_same_inputs_give_same_hash():
    ids = list(range(16))
    a = pi.canonical_block_hash(_ident(), ids, 1, BLOCK, parent_hash=b"p")
    b = pi.canonical_block_hash(_ident(), ids, 1, BLOCK, parent_hash=b"p")
    assert a == b


# ---------------------------------------------------------------------------
# every identity layer must separate
# ---------------------------------------------------------------------------


def test_branch_separates():
    ids = list(range(8))
    h = {b: pi.canonical_block_hash(_ident(branch=b), ids, 0, BLOCK) for b in ("cond", "uncond", "img_cond")}
    assert len(set(h.values())) == 3


def test_model_epoch_separates():
    ids = list(range(8))
    assert pi.canonical_block_hash(_ident(model_epoch=1), ids, 0, BLOCK) != pi.canonical_block_hash(
        _ident(model_epoch=2), ids, 0, BLOCK
    )


def test_rank_and_degree_separate():
    ids = list(range(8))
    base = pi.canonical_block_hash(_ident(tp_rank=0, tp_size=1), ids, 0, BLOCK)
    other_rank = pi.canonical_block_hash(_ident(tp_rank=1, tp_size=2), ids, 0, BLOCK)
    other_degree = pi.canonical_block_hash(_ident(tp_rank=0, tp_size=4), ids, 0, BLOCK)
    assert len({base, other_rank, other_degree}) == 3


def test_extra_identity_separates():
    ids = list(range(8))
    a = pi.canonical_block_hash(_ident(extra=("adapter=A",)), ids, 0, BLOCK)
    b = pi.canonical_block_hash(_ident(extra=("adapter=B",)), ids, 0, BLOCK)
    assert a != b


def test_parent_hash_separates_shared_suffix():
    ids = list(range(16))
    a = pi.canonical_block_hash(_ident(), ids, 1, BLOCK, parent_hash=b"ancestor-A")
    b = pi.canonical_block_hash(_ident(), ids, 1, BLOCK, parent_hash=b"ancestor-B")
    assert a != b


def test_block_tokens_separate():
    """Same identity and index, different tokens -> different hash.

    (Both inputs must use the same `parent_hash`, otherwise the parent term
    differs too and the test would not isolate the token block.)
    """
    a = pi.canonical_block_hash(_ident(), list(range(8)), 0, BLOCK, parent_hash=b"p")
    b = pi.canonical_block_hash(_ident(), [7] * 8, 0, BLOCK, parent_hash=b"p")
    assert a != b


def test_hashable_and_frozen():
    ident = _ident()
    assert ident in {ident}
    with pytest.raises(Exception):
        ident.branch = "uncond"  # frozen dataclass


# ---------------------------------------------------------------------------
# chaining
# ---------------------------------------------------------------------------


def test_ancestor_chaining_makes_every_later_block_differ():
    ids_a = list(range(16))
    ids_b = [99] * 4 + list(range(4, 16))
    ha = pi.prefix_block_hashes(_ident(), ids_a, BLOCK, 16)
    hb = pi.prefix_block_hashes(_ident(), ids_b, BLOCK, 16)
    assert len(ha) == len(hb) == 4
    assert all(x != y for x, y in zip(ha, hb)), "a shared suffix must not collide"
    # identical prefixes still agree
    assert ha == pi.prefix_block_hashes(_ident(), list(ids_a), BLOCK, 16)


def test_shared_prefix_hashes_are_prefix_equal():
    ids = list(range(16))
    four = pi.prefix_block_hashes(_ident(), ids, BLOCK, 4)
    sixteen = pi.prefix_block_hashes(_ident(), ids, BLOCK, 16)
    assert sixteen[: len(four)] == four


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------


def test_partial_block_has_no_canonical_identity():
    with pytest.raises(ValueError, match="not a full block"):
        pi.canonical_block_hash(_ident(), list(range(6)), 1, BLOCK)


def test_unaligned_reusable_length_is_rejected():
    ids = list(range(16))
    with pytest.raises(ValueError, match="not a multiple"):
        pi.prefix_block_hashes(_ident(), ids, BLOCK, 6)
    with pytest.raises(ValueError, match="non-negative"):
        pi.prefix_block_hashes(_ident(), ids, BLOCK, -4)


def test_zero_reusable_tokens_yields_no_hashes():
    assert pi.prefix_block_hashes(_ident(), list(range(16)), BLOCK, 0) == []


def test_unsupported_payload_is_rejected():
    with pytest.raises(TypeError, match="unsupported value"):
        pi.stable_hash_function({"a": 1})


# ---------------------------------------------------------------------------
# the reuse cap (verified against real vLLM; see b3_native_manager.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_tokens", "block", "tokens", "blocks"),
    [
        (8, 4, 4, 1),
        (12, 4, 8, 2),
        (16, 4, 12, 3),
        (20, 4, 16, 4),
        (24, 4, 20, 5),
        (32, 4, 28, 7),
        (4, 4, 0, 0),
        (3, 4, 0, 0),
        (1, 4, 0, 0),
    ],
)
def test_reuse_cap_matches_native_vllm_behaviour(num_tokens, block, tokens, blocks):
    """Native vLLM never reports the request's own final block as reusable."""
    assert pi.truncate_to_reusable_blocks(num_tokens, block) == tokens
    assert pi.reusable_block_count(num_tokens, block) == blocks


def test_hashes_fit_blocks():
    assert pi.hashes_fit_blocks([b"1", b"2", b"3"], 16, BLOCK)
    assert not pi.hashes_fit_blocks([b"1", b"2", b"3", b"4"], 16, BLOCK)
