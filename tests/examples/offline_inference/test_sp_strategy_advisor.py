"""Regression tests for the SP strategy advisor example.

The advisor recommends a sequence-parallel attention strategy from the
attention shape, so the rules it encodes are worth pinning: the legality
constraints, the crossover between Ulysses and AllGather-KV, and the BAGEL
shape whose recommendation was confirmed on hardware
(.claude/skills/diffusion-perf-opt/references/sp-strategy-selection.md).
"""

import sys
from pathlib import Path

import pytest

_examples_dir = str(Path(__file__).parent.parent.parent.parent / "examples" / "offline_inference" / "diffusion")
sys.path.insert(0, _examples_dir)
from sp_strategy_advisor import comm_volume, legality, recommend  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# seq_len=4096, 32 heads over 4 KV heads, sp_degree=4, non-causal.
BAGEL = dict(num_heads=32, num_kv_heads=4, sp_degree=4, seq_len=4096)


def test_bagel_shape_matches_the_hardware_result():
    """AllGather-KV measured fastest on 4x A800; the rule must agree."""
    assert "recommended: allgather_kv" in recommend(**BAGEL)


def test_bagel_volume_ratio_is_the_closed_form():
    volumes = comm_volume(num_heads=32, num_kv_heads=4, sp_degree=4)
    # A/U = N * H_kv / (H + H_kv) = 4*4/36
    assert volumes["allgather_kv"] / volumes["ulysses"] == pytest.approx(4 * 4 / 36)
    # Ring moves the same bytes as AllGather-KV; it loses on hops, not volume.
    assert volumes["ring"] == pytest.approx(volumes["allgather_kv"])


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "sp_degree", "expected"),
    [
        (32, 32, 4, "ulysses"),  # MHA: group size 1, never beats the all-to-all
        (32, 16, 4, "ulysses"),  # group 2 <= sp_degree-1 = 3
        (32, 8, 4, "allgather_kv"),  # group 4 > 3, just past the crossover
        (32, 1, 4, "allgather_kv"),  # MQA: gathering KV is nearly free
        (32, 8, 8, "ulysses"),  # same model, wider SP: group 4 <= 7 flips it back
    ],
)
def test_crossover_follows_group_size_versus_sp_degree(num_heads, num_kv_heads, sp_degree, expected):
    report = recommend(num_heads=num_heads, num_kv_heads=num_kv_heads, sp_degree=sp_degree)
    assert f"recommended: {expected}" in report


def test_causal_attention_rules_out_allgather_kv():
    reasons = legality(
        num_heads=32,
        num_kv_heads=4,
        seq_len=4096,
        sp_degree=4,
        causal=True,
        attention_mask=False,
        ulysses_replicates_kv=False,
    )
    assert reasons["allgather_kv"] is not None
    assert reasons["ulysses"] is None
    assert "recommended: ulysses" in recommend(**BAGEL, causal=True)


def test_strict_ulysses_needs_kv_heads_divisible_by_sp_degree():
    reasons = legality(
        num_heads=32,
        num_kv_heads=4,
        seq_len=4096,
        sp_degree=8,
        causal=False,
        attention_mask=False,
        ulysses_replicates_kv=False,
    )
    assert reasons["ulysses"] is not None and "num_kv_heads" in reasons["ulysses"]
    # The KV-replicating variant lifts exactly that constraint.
    relaxed = legality(
        num_heads=32,
        num_kv_heads=4,
        seq_len=4096,
        sp_degree=8,
        causal=False,
        attention_mask=False,
        ulysses_replicates_kv=True,
    )
    assert relaxed["ulysses"] is None


def test_ring_is_only_recommended_when_it_is_the_only_legal_option():
    # Causal rules out AllGather-KV; an indivisible head count rules out Ulysses.
    report = recommend(num_heads=30, num_kv_heads=6, sp_degree=4, causal=True)
    assert "recommended: ring" in report
    assert "sequential hops" in report


def test_no_legal_strategy_is_reported_rather_than_guessed():
    report = recommend(num_heads=30, num_kv_heads=6, sp_degree=4, seq_len=4095, causal=True)
    assert "No legal strategy" in report


def test_seq_len_is_optional_and_skips_the_divisibility_checks():
    report = recommend(num_heads=32, num_kv_heads=4, sp_degree=4)
    assert "seq_len" not in report.splitlines()[0]
    assert "not divisible" not in report


def test_rejects_shapes_that_are_not_grouped_attention():
    with pytest.raises(ValueError, match="divisible"):
        recommend(num_heads=32, num_kv_heads=5, sp_degree=4)
