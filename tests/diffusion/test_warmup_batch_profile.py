# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm_omni.diffusion.data import DiffusionBatchingConfig
from vllm_omni.diffusion.warmup_batch_profile import (
    aggregate_step_records,
    build_profile_cache_key,
    choose_sweet_spot_batch,
    generate_sparse_ladder,
    load_cached_profile,
    save_cached_profile,
    should_stop_early,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# generate_sparse_ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b_max", "expected"),
    [
        (128, [1, 2, 4, 8, 16, 32, 64, 96, 128]),
        (20, [1, 2, 4, 8, 15, 16, 20]),
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
    ],
)
def test_sparse_ladder_powers_of_two_with_three_quarter_and_max(b_max: int, expected: list[int]) -> None:
    # Powers of two up to b_max, plus round(0.75 * b_max), plus b_max itself.
    assert generate_sparse_ladder(b_max) == expected


# ---------------------------------------------------------------------------
# aggregate_step_records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([(4, 0, 100.0), (4, 1, 110.0), (4, 2, 120.0)], {4: 115.0}),
        # The first step is compile/lazy-init noise and often the slowest
        # sample; discarding must follow step order, not discard by value.
        ([(4, 0, 1000.0), (4, 1, 10.0), (4, 2, 12.0)], {4: 11.0}),
        ([(2, 0, 50.0), (2, 1, 60.0), (4, 0, 200.0), (4, 1, 220.0)], {2: 60.0, 4: 220.0}),
        ([], {}),
        ([(3, 0, 10.0)], {}),
    ],
    ids=["discard-and-median", "discard-by-step-order", "group-by-batch", "empty", "all-discarded"],
)
def test_aggregate_groups_by_batch_and_medians_after_leading_discard(
    records: list[tuple[int, int, float]], expected: dict[int, float]
) -> None:
    assert aggregate_step_records(records, discard_steps=1) == expected


# ---------------------------------------------------------------------------
# should_stop_early
# ---------------------------------------------------------------------------


def _point(b: int, t: float) -> dict[str, float]:
    return {"b": float(b), "t": t, "thr": b / t}


@pytest.mark.parametrize(
    ("prev", "cur", "max_step_ms", "min_gain", "expected"),
    [
        ((4, 10.0), (8, 30.0), 25.0, 0.05, (True, "budget")),
        # cur exceeds the budget AND throughput saturates: budget wins.
        ((4, 10.0), (8, 80.0), 50.0, 0.05, (True, "budget")),
        ((4, 10.0), (8, 22.0), None, 0.05, (True, "saturation")),
        ((4, 20.0), (8, 30.0), None, 0.05, (False, None)),
        ((4, 10.0), (8, 40.0), None, 0.0, (False, None)),
    ],
)
def test_stop_early_budget_then_saturation(
    prev: tuple[int, float],
    cur: tuple[int, float],
    max_step_ms: float | None,
    min_gain: float,
    expected: tuple[bool, str | None],
) -> None:
    stop, reason = should_stop_early(_point(*prev), _point(*cur), max_step_ms=max_step_ms, min_marginal_gain=min_gain)
    assert (stop, reason) == expected


# ---------------------------------------------------------------------------
# choose_sweet_spot_batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("points", "max_step_ms", "expected_b", "expected_method"),
    [
        ([(1, 5.0), (2, 8.0), (4, 14.0), (8, 40.0)], None, 4, "max_throughput"),
        # Equal throughput for b=2 and b=4: the smaller batch wins.
        ([(2, 20.0), (4, 40.0)], None, 2, "max_throughput"),
        # b=4 exceeds the step budget; its higher throughput is filtered out.
        ([(2, 8.0), (4, 30.0)], 10.0, 2, "max_throughput"),
        ([(2, 20.0), (4, 40.0)], 10.0, 2, "fallback_min_batch"),
    ],
)
def test_choose_sweet_spot_max_throughput_and_budget_floor(
    points: list[tuple[int, float]],
    max_step_ms: float | None,
    expected_b: int,
    expected_method: str,
) -> None:
    b, analysis = choose_sweet_spot_batch([_point(*p) for p in points], max_step_ms=max_step_ms, min_marginal_gain=0.05)
    assert b == expected_b
    assert analysis["method"] == expected_method


def test_choose_requires_points() -> None:
    with pytest.raises(ValueError):
        choose_sweet_spot_batch([], max_step_ms=None, min_marginal_gain=0.05)


# ---------------------------------------------------------------------------
# DiffusionBatchingConfig new fields
# ---------------------------------------------------------------------------


def test_config_new_fields_defaults() -> None:
    cfg = DiffusionBatchingConfig()
    assert cfg.warmup_auto_calibrate is False
    assert cfg.warmup_profile_max_batch == 128
    assert cfg.warmup_profile_num_steps == 3
    assert cfg.warmup_profile_discard_steps == 1
    assert cfg.warmup_reference_height == 512
    assert cfg.warmup_reference_width == 512
    assert cfg.warmup_profile_max_step_ms is None
    assert cfg.warmup_profile_min_marginal_gain == 0.05
    assert cfg.warmup_profile_timeout_s == 600.0
    assert cfg.warmup_profile_cache is True
    assert cfg.warmup_profile_refresh is False
    assert cfg.warmup_profile_cache_ttl_s == 0.0
    assert cfg.warmup_profile_cache_dir is None


def test_config_new_fields_from_dict() -> None:
    cfg = DiffusionBatchingConfig.from_dict(
        {
            "warmup_auto_calibrate": True,
            "warmup_profile_max_batch": 64,
            "warmup_profile_max_step_ms": 1500.0,
        }
    )
    assert cfg.warmup_auto_calibrate is True
    assert cfg.warmup_profile_max_batch == 64
    assert cfg.warmup_profile_max_step_ms == 1500.0
    assert cfg.policy == "auto"


def test_config_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown_field"):
        DiffusionBatchingConfig.from_dict({"unknown_field": 1})


def test_removed_compute_unit_budget_points_at_max_num_batched_tokens() -> None:
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        DiffusionBatchingConfig.from_dict({"compute_unit_budget": 10240})


# ---------------------------------------------------------------------------
# profile cache helpers
# ---------------------------------------------------------------------------


def test_cache_key_stable_and_sensitive() -> None:
    fields = {"model": "Qwen-Image", "gpu": "A100", "ref": "512x512"}
    assert build_profile_cache_key(fields) == build_profile_cache_key(dict(fields))
    other = dict(fields, gpu="H100")
    assert build_profile_cache_key(fields) != build_profile_cache_key(other)


def test_cache_save_load_roundtrip(tmp_path) -> None:
    key = build_profile_cache_key({"gpu": "A100"})
    payload = {"b_star": 24, "budget": 24576, "method": "max_throughput"}
    path = save_cached_profile(str(tmp_path), key, payload)
    assert path is not None

    loaded = load_cached_profile(str(tmp_path), key)
    assert loaded is not None
    assert loaded["b_star"] == 24
    assert loaded["budget"] == 24576
    assert loaded["format_version"] == 1
    assert "saved_at" in loaded
    assert payload.get("saved_at") is None  # input not mutated


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{not json",
        {"format_version": 999, "b_star": 24},
        {"format_version": 1, "b_star": 0},
        {"format_version": 1},
    ],
    ids=["missing", "corrupt", "wrong_version", "bad_bstar", "missing_bstar"],
)
def test_cache_load_rejects_unusable_payloads(tmp_path, content) -> None:
    key = "deadbeef"
    if content is not None:
        text = content if isinstance(content, str) else json.dumps(content)
        (tmp_path / f"{key}.json").write_text(text, encoding="utf-8")
    assert load_cached_profile(str(tmp_path), key) is None


def test_cache_ttl_expiry(tmp_path, monkeypatch) -> None:
    key = build_profile_cache_key({"gpu": "A100"})
    path = save_cached_profile(str(tmp_path), key, {"b_star": 8})
    assert path is not None
    assert load_cached_profile(str(tmp_path), key, ttl_s=0.0) is not None
    assert load_cached_profile(str(tmp_path), key, ttl_s=3600.0) is not None
    # Negative TTL disables the check entirely, like 0.
    assert load_cached_profile(str(tmp_path), key, ttl_s=-1.0) is not None

    monkeypatch.setattr("time.time", lambda: json.load(open(path))["saved_at"] + 7200)
    assert load_cached_profile(str(tmp_path), key, ttl_s=3600.0) is None
    assert load_cached_profile(str(tmp_path), key, ttl_s=0.0) is not None
