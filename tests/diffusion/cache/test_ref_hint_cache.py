# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for framework-side reference-hint scheduling and history."""

from vllm_omni.diffusion.cache.ref_hint_cache import RefHintCacheState


def test_refresh_schedule_k2():
    """K=2: refresh on even steps, reuse on odd steps."""
    st = RefHintCacheState(refresh_interval=2)
    b, r = st.begin_call(0)
    assert r is True  # step 0: 0 % 2 == 0 -> refresh
    st.store(b, 0, "h0")
    b, r = st.begin_call(1)
    assert r is False and st.history(b)[-1][1] == "h0"  # step 1: reuse
    b, r = st.begin_call(2)
    assert r is True  # step 2: refresh
    st.store(b, 2, "h2")
    b, r = st.begin_call(3)
    assert r is False and st.history(b)[-1][1] == "h2"  # step 3: reuse the fresher h2
    assert st.hits == 2 and st.misses == 2
    assert st.refreshes == 1


def test_branch_keying_two_forwards_per_step():
    """Two sequential forwards per step (cond then uncond) must not alias."""
    st = RefHintCacheState(refresh_interval=2)
    b0, r0 = st.begin_call(0)
    b1, r1 = st.begin_call(0)
    assert (b0, r0) == (0, True) and (b1, r1) == (1, True)
    st.store(b0, 0, "cond")
    st.store(b1, 0, "uncond")
    # next step reuses per branch, cond never gets uncond's hint
    b0, r0 = st.begin_call(1)
    b1, r1 = st.begin_call(1)
    assert (b0, r0) == (0, False) and (b1, r1) == (1, False)
    assert st.history(b0)[-1][1] == "cond"
    assert st.history(b1)[-1][1] == "uncond"


def test_full_reuse_large_k():
    """Large K = compute once at step 0, reuse for all later steps."""
    st = RefHintCacheState(refresh_interval=10_000)
    b, r = st.begin_call(0)
    assert r is True
    st.store(b, 0, "once")
    for s in range(1, 12):
        b, r = st.begin_call(s)
        assert r is False and st.history(b)[-1][1] == "once"
    assert st.misses == 1 and st.hits == 11


def test_reuse_retains_only_latest_fresh_value():
    st = RefHintCacheState[str](refresh_interval=2, strategy="reuse")
    branch, _ = st.begin_call(0)
    st.store(branch, 0, "h0")
    branch, _ = st.begin_call(2)
    st.store(branch, 2, "h2")

    assert st.history(branch) == ((2, "h2"),)


def test_first_use_of_a_branch_always_refreshes():
    """Even if step % K != 0, a branch never seen before must recompute (nothing to reuse)."""
    st = RefHintCacheState(refresh_interval=2)
    _, r = st.begin_call(1)  # step 1 is not a refresh step, but branch 0 not cached yet
    assert r is True


def test_step_none_is_safe_noop():
    """Unknown step (no forward context / warmup) -> always refresh, never cache."""
    st = RefHintCacheState(refresh_interval=2)
    b, r = st.begin_call(None)
    assert b is None and r is True
    st.store(b, None, "x")  # no-op store for branch None
    assert st.hits == 0
    b, r = st.begin_call(None)
    assert b is None and r is True  # still no reuse


def test_reset_clears_state():
    st = RefHintCacheState(refresh_interval=2)
    b, _ = st.begin_call(0)
    st.store(b, 0, "h")
    st.begin_call(1)
    st.reset()
    assert st._history == {} and st._last_step is None and st._call_idx == 0
    assert st.hits == 0 and st.misses == 0
    assert st.refreshes == 0


def test_refresh_interval_clamped_to_at_least_one():
    st = RefHintCacheState(refresh_interval=0)
    assert st.refresh_interval == 1
    # K=1 -> every step is a refresh step, never reuse
    branch, _ = st.begin_call(0)
    st.store(branch, 0, "h")
    _, r = st.begin_call(1)
    assert r is True


def test_forecast50_calibrates_twice_then_alternates():
    st = RefHintCacheState[str](refresh_interval=2, strategy="forecast50")
    branch, refresh = st.begin_call(0)
    assert refresh is True
    st.store(branch, 0, "h0")
    branch, refresh = st.begin_call(1)
    assert refresh is True
    st.store(branch, 1, "h1")
    branch, refresh = st.begin_call(2)
    assert refresh is False
    assert st.history(branch) == ((0, "h0"), (1, "h1"))
    branch, refresh = st.begin_call(3)
    assert refresh is True
    st.store(branch, 3, "h3")
    branch, refresh = st.begin_call(4)
    assert refresh is False
    assert st.history(branch) == ((1, "h1"), (3, "h3"))


def test_unknown_strategy_is_rejected():
    try:
        RefHintCacheState(strategy="unknown")
    except ValueError as exc:
        assert "Unsupported ref_hint strategy" in str(exc)
    else:
        raise AssertionError("unknown strategy was not rejected")


if __name__ == "__main__":  # allow running standalone without pytest
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
    sys.exit(0)
