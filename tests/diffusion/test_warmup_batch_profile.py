# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.data import DiffusionBatchingConfig
from vllm_omni.diffusion.warmup_batch_profile import (
    aggregate_step_records,
    choose_sweet_spot_batch,
    generate_sparse_ladder,
    should_stop_early,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# generate_sparse_ladder
# ---------------------------------------------------------------------------


def test_ladder_powers_of_two_plus_edges():
    ladder = generate_sparse_ladder(128)
    # powers of two up to 128, plus round(0.75*128)=96, plus 128
    assert ladder == sorted({1, 2, 4, 8, 16, 32, 64, 96, 128})
    assert ladder == sorted(ladder)
    assert len(ladder) <= 10


def test_ladder_small_bmax():
    assert generate_sparse_ladder(1) == [1]
    assert generate_sparse_ladder(2) == [1, 2]
    assert generate_sparse_ladder(3) == [1, 2, 3]


def test_ladder_three_quarter_inclusion():
    # round(0.75*20)=15 must appear along with powers of two and 20
    ladder = generate_sparse_ladder(20)
    assert 15 in ladder
    assert 20 in ladder
    assert 16 in ladder


# ---------------------------------------------------------------------------
# aggregate_step_records
# ---------------------------------------------------------------------------


def test_aggregate_discard_and_median():
    # batch 4: steps 0..2 with ms; first discarded, median of remaining two
    records = [
        (4, 0, 100.0),
        (4, 1, 110.0),
        (4, 2, 120.0),
    ]
    out = aggregate_step_records(records, discard_steps=1)
    assert out == {4: 115.0}


def test_aggregate_groups_by_batch():
    records = [
        (2, 0, 50.0),
        (2, 1, 60.0),
        (4, 0, 200.0),
        (4, 1, 220.0),
    ]
    out = aggregate_step_records(records, discard_steps=1)
    assert out == {2: 60.0, 4: 220.0}


def test_aggregate_empty():
    assert aggregate_step_records([], 1) == {}


def test_aggregate_all_discarded():
    out = aggregate_step_records([(3, 0, 10.0)], discard_steps=1)
    assert out == {}


# ---------------------------------------------------------------------------
# should_stop_early
# ---------------------------------------------------------------------------


def _point(b, t):
    return {"b": float(b), "t": t, "thr": b / t}


def test_budget_stop():
    prev = _point(4, 10.0)
    cur = _point(8, 30.0)  # exceeds 25ms budget
    stop, reason = should_stop_early(prev, cur, max_step_ms=25.0, min_marginal_gain=0.05)
    assert stop and reason == "budget"


def test_budget_priority_over_saturation():
    # cur exceeds budget AND marginal gain is tiny -> budget wins
    prev = _point(4, 10.0)
    cur = _point(8, 80.0)  # thr 0.1 vs 0.4 -> big drop, but also over budget
    stop, reason = should_stop_early(prev, cur, max_step_ms=50.0, min_marginal_gain=0.05)
    assert reason == "budget"


def test_saturation_stop():
    prev = _point(4, 10.0)  # thr 0.4
    cur = _point(8, 22.0)  # thr ~0.36, gain ~ -8% < 5%
    stop, reason = should_stop_early(prev, cur, max_step_ms=None, min_marginal_gain=0.05)
    assert stop and reason == "saturation"


def test_no_stop_when_gain_large():
    prev = _point(4, 20.0)  # thr 0.2
    cur = _point(8, 30.0)  # thr 0.266, gain ~33%
    stop, reason = should_stop_early(prev, cur, max_step_ms=None, min_marginal_gain=0.05)
    assert not stop and reason is None


def test_saturation_disabled_when_zero():
    prev = _point(4, 10.0)
    cur = _point(8, 40.0)  # thr drops to 0.2, gain -50%
    stop, reason = should_stop_early(prev, cur, max_step_ms=None, min_marginal_gain=0.0)
    assert not stop and reason is None


# ---------------------------------------------------------------------------
# choose_sweet_spot_batch
# ---------------------------------------------------------------------------


def test_choose_max_throughput():
    points = [
        _point(1, 5.0),   # thr 0.2
        _point(2, 8.0),   # thr 0.25
        _point(4, 14.0),  # thr ~0.286  <- winner
        _point(8, 40.0),  # thr 0.2
    ]
    b, analysis = choose_sweet_spot_batch(points, max_step_ms=None, min_marginal_gain=0.05)
    assert b == 4
    assert analysis["method"] == "max_throughput"


def test_choose_tie_takes_smaller():
    # thr equal for b=2 and b=4
    points = [_point(2, 20.0), _point(4, 40.0)]
    b, _ = choose_sweet_spot_batch(points, max_step_ms=None, min_marginal_gain=0.05)
    assert b == 2


def test_choose_filters_budget_exceeded():
    points = [
        _point(2, 8.0),   # thr 0.25 within 10ms
        _point(4, 30.0),  # exceeds 10ms -> filtered; thr would be higher
    ]
    b, analysis = choose_sweet_spot_batch(points, max_step_ms=10.0, min_marginal_gain=0.05)
    assert b == 2


def test_choose_all_over_budget_falls_back_min():
    points = [_point(2, 20.0), _point(4, 40.0)]
    b, analysis = choose_sweet_spot_batch(points, max_step_ms=10.0, min_marginal_gain=0.05)
    assert b == 2
    assert analysis["method"] == "fallback_min_batch"


def test_choose_requires_points():
    with pytest.raises(ValueError):
        choose_sweet_spot_batch([], max_step_ms=None, min_marginal_gain=0.05)


# ---------------------------------------------------------------------------
# DiffusionBatchingConfig new fields
# ---------------------------------------------------------------------------


def test_config_new_fields_defaults():
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


def test_config_new_fields_from_dict():
    cfg = DiffusionBatchingConfig.from_dict(
        {
            "warmup_auto_calibrate": True,
            "warmup_profile_max_batch": 64,
            "warmup_profile_max_step_ms": 1500.0,
            "unknown_field": 123,  # ignored
        }
    )
    assert cfg.warmup_auto_calibrate is True
    assert cfg.warmup_profile_max_batch == 64
    assert cfg.warmup_profile_max_step_ms == 1500.0
    assert cfg.policy == "fixed"


# ---------------------------------------------------------------------------
# profile cache helpers
# ---------------------------------------------------------------------------

from vllm_omni.diffusion.warmup_batch_profile import (  # noqa: E402
    build_profile_cache_key,
    load_cached_profile,
    save_cached_profile,
)


def test_cache_key_stable_and_sensitive():
    fields = {"model": "Qwen-Image", "gpu": "A100", "ref": "512x512"}
    assert build_profile_cache_key(fields) == build_profile_cache_key(dict(fields))
    other = dict(fields, gpu="H100")
    assert build_profile_cache_key(fields) != build_profile_cache_key(other)


def test_cache_save_load_roundtrip(tmp_path):
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


def test_cache_load_missing_and_corrupt(tmp_path):
    key = build_profile_cache_key({"gpu": "A100"})
    assert load_cached_profile(str(tmp_path), key) is None

    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert load_cached_profile(str(tmp_path), key) is None


def test_cache_load_rejects_wrong_version_and_bad_bstar(tmp_path):
    key = "deadbeef"
    (tmp_path / f"{key}.json").write_text(
        '{"format_version": 999, "b_star": 24}', encoding="utf-8"
    )
    assert load_cached_profile(str(tmp_path), key) is None

    (tmp_path / f"{key}.json").write_text(
        '{"format_version": 1, "b_star": 0}', encoding="utf-8"
    )
    assert load_cached_profile(str(tmp_path), key) is None


def test_cache_ttl_expiry(tmp_path, monkeypatch):
    import json as _json

    key = build_profile_cache_key({"gpu": "A100"})
    path = save_cached_profile(str(tmp_path), key, {"b_star": 8})
    assert path is not None
    assert load_cached_profile(str(tmp_path), key, ttl_s=0.0) is not None
    assert load_cached_profile(str(tmp_path), key, ttl_s=3600.0) is not None
    # Negative TTL disables the check entirely, like 0.
    assert load_cached_profile(str(tmp_path), key, ttl_s=-1.0) is not None

    monkeypatch.setattr("time.time", lambda: _json.load(open(path))["saved_at"] + 7200)
    assert load_cached_profile(str(tmp_path), key, ttl_s=3600.0) is None
    assert load_cached_profile(str(tmp_path), key, ttl_s=0.0) is not None


def test_config_cache_fields_defaults():
    cfg = DiffusionBatchingConfig()
    assert cfg.warmup_profile_cache is True
    assert cfg.warmup_profile_refresh is False
    assert cfg.warmup_profile_cache_ttl_s == 0.0
    assert cfg.warmup_profile_cache_dir is None
