# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import gc
import weakref

import pytest
import torch

from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig
from vllm_omni.diffusion.cache.aligncache.runtime import AlignCacheRuntime
from vllm_omni.diffusion.cache.aligncache.state import GenerationSignature, SchedulePlan, StepContext, storage_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def setup_runtime(mode="accelerate", **overrides):
    return AlignCacheRuntime(
        AlignCacheConfig.parse(
            {"align_cache": {"mode": mode, "warmup_steps": 2, "tail_steps": 1, "base_threshold": 0.065, **overrides}}
        )
    )


def bind(runtime, total=20, dtype=torch.float32):
    x = torch.ones(1, 1, 1, 1, 128, dtype=dtype)
    runtime.bind_schedule(
        SchedulePlan(tuple(float(total - i) for i in range(total)), "fake", "{}"),
        GenerationSignature(runtime.epoch, 1, (2, 3), 5.0, tuple(x.shape), dtype, x.device),
    )
    return x


def step(runtime, i, x, exact=None):
    if exact is None:

        def exact(b):
            return x * (0.2 if b == "cond" else 0.15) + 1

    return runtime.predict_pair(StepContext(i, runtime.plan.timesteps[i], 1, 5.0), x, (2, 3), exact)


def test_accelerate_counts_and_cleanup():
    runtime = setup_runtime()
    with runtime.request_context():
        base = bind(runtime)
        for i in range(20):
            outputs = step(runtime, i, base + i * 0.01)
            assert all(v.shape == base.shape and v.dtype == base.dtype for v in outputs)
        assert runtime.counts["fast_pairs"] > 0
        assert runtime.counts["model_forward_calls"] == 2 * runtime.counts["exact_pairs"]
        assert runtime.counts["fast_pairs"] + runtime.counts["exact_pairs"] == 20
        ref = weakref.ref(runtime.state.anchor.cond)
    assert runtime.state is None and runtime.plan is None and runtime.signature is None
    gc.collect()
    assert ref() is None
    assert runtime.last_summary["status"] == "completed"


@pytest.mark.parametrize(
    "overrides",
    [{"base_threshold": 0}, {"warmup_steps": 15, "tail_steps": 15}, {"max_cache_bytes": 1}, {"mode": "force_exact"}],
)
def test_all_exact_paths(overrides):
    runtime = setup_runtime(**overrides)
    with runtime.request_context():
        x = bind(runtime)
        for i in range(20):
            cond, uncond = step(runtime, i, x + i * 0.01)
            torch.testing.assert_close(cond, (x + i * 0.01) * 0.2 + 1, rtol=0, atol=0)
        assert runtime.counts["model_forward_calls"] == 40
        assert runtime.counts["fast_pairs"] == 0
        if overrides.get("max_cache_bytes") == 1:
            assert runtime.state is None and runtime.peak_cache_bytes == 0


def test_uncond_error_atomic_no_retry_and_recovery():
    runtime = setup_runtime()
    calls = []
    error = RuntimeError("uncond failed")
    with pytest.raises(RuntimeError) as caught:
        with runtime.request_context():
            x = bind(runtime)
            step(runtime, 0, x)
            before = runtime.state

            def fail(branch):
                calls.append(branch)
                if branch == "uncond":
                    assert runtime.state is before
                    raise error
                return x + 1

            step(runtime, 1, x, fail)
    assert caught.value is error and calls == ["cond", "uncond"]
    assert runtime.last_summary["completed_pairs"] == 1
    assert runtime.last_summary["model_forward_calls"] == 4
    assert runtime.state is None
    with runtime.request_context():
        x = bind(runtime)
        assert runtime.state is not None
        assert runtime.state.completed_pairs == 0
        step(runtime, 0, x)


def test_reentry_does_not_clear_active_state():
    runtime = setup_runtime()
    with runtime.request_context():
        bind(runtime)
        before = runtime.state
        with pytest.raises(RuntimeError, match="active"):
            runtime.reset()
        with pytest.raises(RuntimeError, match="active"):
            with runtime.request_context():
                pass
        assert runtime.state is before


def test_trace_native_outputs_rng_and_shadow_history():
    runtime = setup_runtime(mode="trace")
    rng = torch.random.get_rng_state()
    with runtime.request_context():
        x = bind(runtime)
        for i in range(20):
            current = x + i * 0.01
            cond, uncond = step(runtime, i, current)
            torch.testing.assert_close(cond, current * 0.2 + 1, rtol=0, atol=0)
        assert runtime.counts["shadow_fast_pairs"] > 0
        assert runtime.counts["model_forward_calls"] == 40
        assert runtime.counts["fast_pairs"] == 0
    assert torch.equal(rng, torch.random.get_rng_state())


def test_owned_inputs_and_outputs():
    runtime = setup_runtime()
    with runtime.request_context():
        x = bind(runtime, dtype=torch.bfloat16)
        cond, uncond = step(runtime, 0, x)
        expected = runtime.state.anchor.cond.clone()
        cond.zero_()
        uncond.zero_()
        x.zero_()
        assert torch.equal(expected, runtime.state.anchor.cond)
        assert runtime.state.previous_input.sum() > 0
        assert runtime.state.anchor.cond.dtype == torch.float32
        step(runtime, 1, x + 1.01)
        c, u = step(runtime, 2, x + 1.02)
        snapshot = runtime.state.cond.last.clone()
        c.zero_()
        u.zero_()
        assert torch.equal(snapshot, runtime.state.cond.last)


def test_storage_unique_views():
    x = torch.ones(128)
    assert storage_bytes(x, x[:32], {"x": x}) == 512


def test_schedule_and_index_validation():
    runtime = setup_runtime()
    with runtime.request_context():
        x = bind(runtime)
        with pytest.raises(RuntimeError, match="fresh"):
            bind(runtime)
        with pytest.raises(ValueError, match="out-of-order"):
            step(runtime, 1, x)
        step(runtime, 0, x)
        with pytest.raises(ValueError, match="out-of-order"):
            step(runtime, 0, x)


def test_invalid_prediction_falls_back_before_native_side_effect(monkeypatch):
    runtime = setup_runtime()
    with runtime.request_context():
        x = bind(runtime)
        step(runtime, 0, x)
        step(runtime, 1, x + 0.01)
        original = runtime._fast

        def invalid(tx, *args):
            cond, uncond = original(tx, *args)
            return cond * float("nan"), uncond

        monkeypatch.setattr(runtime, "_fast", invalid)
        calls = []

        def exact(branch):
            calls.append(branch)
            return x + 2

        cond, uncond = step(runtime, 2, x + 0.02, exact)
        assert calls == ["cond", "uncond"]
        assert torch.isfinite(cond).all() and torch.isfinite(uncond).all()
        assert runtime.reasons["invalid_prediction"] == 1
        assert runtime.state.cond.source == "exact"


def test_native_nonfinite_is_not_retried_or_cached():
    runtime = setup_runtime()
    with runtime.request_context():
        x = bind(runtime)
        calls = []

        def exact(branch):
            calls.append(branch)
            return x * float("nan") if branch == "cond" else x

        cond, _ = step(runtime, 0, x, exact)
        assert torch.isnan(cond).all() and calls == ["cond", "uncond"]
        assert runtime.state is None
        step(runtime, 1, x)
        assert runtime.counts["model_forward_calls"] == 4
        assert runtime.reasons["nonfinite_exact"] == 2


def test_draft_order_inside_transaction(monkeypatch):
    import vllm_omni.diffusion.cache.aligncache.runtime as module

    runtime = setup_runtime(base_threshold=0.00001)
    original = module.predict
    observed = []

    def inspect(state, x, branch, phase, config):
        observed.append((branch, state.cond.last.clone(), state.uncond.last.clone()))
        return original(state, x, branch, phase, config)

    with runtime.request_context():
        x = bind(runtime)
        step(runtime, 0, x)
        step(runtime, 1, x + 0.01)
        monkeypatch.setattr(module, "predict", inspect)
        old_cond = runtime.state.cond.last.clone()
        old_uncond = runtime.state.uncond.last.clone()
        step(runtime, 2, x + 0.02)
        assert [v[0] for v in observed] == ["cond", "uncond"]
        assert torch.equal(observed[0][1], old_cond)
        assert not torch.equal(observed[1][1], old_cond)
        assert torch.equal(observed[1][2], old_uncond)


def test_repeated_requests_bounded_memory_and_diagnostics(monkeypatch):
    import vllm_omni.diffusion.cache.aligncache.runtime as module

    runtime = setup_runtime(diagnostics="trace")
    peaks, anchors = [], []
    original_rng = torch.random.get_rng_state()
    for request_index in range(12):
        with runtime.request_context():
            x = bind(runtime)
            for i in range(20):
                step(runtime, i, x + i * 0.01)
            anchors.append(weakref.ref(runtime.state.anchor.cond))
            peaks.append(runtime.peak_cache_bytes)
            assert runtime.peak_cache_bytes <= runtime.estimate_capacity(x)
        assert len(runtime.trace) <= 256
        assert runtime.last_summary["active_cache_bytes"] == 0
    gc.collect()
    assert all(ref() is None for ref in anchors)
    assert max(peaks) == min(peaks)
    assert torch.equal(original_rng, torch.random.get_rng_state())

    def bad_log(*args, **kwargs):
        raise RuntimeError("logger failed")

    monkeypatch.setattr(module.logger, "info", bad_log)
    with runtime.request_context():
        step(runtime, 0, bind(runtime))
    error = ValueError("model error")
    with pytest.raises(ValueError) as caught:
        with runtime.request_context():
            raise error
    assert caught.value is error and runtime.state is None


def test_trace_shadow_matches_accelerate_given_identical_inputs():
    states = []
    for mode in ("trace", "accelerate"):
        runtime = setup_runtime(mode=mode, tail_steps=0)
        with runtime.request_context():
            x = bind(runtime)
            for i in range(20):
                step(runtime, i, x + i * 0.01)
            states.append(
                (
                    runtime.state.cond.last.clone(),
                    runtime.state.anchor.index,
                    dict(runtime.state.feedback_scale),
                    runtime.state.budget,
                )
            )
    assert torch.equal(states[0][0], states[1][0])
    assert states[0][1:] == states[1][1:]


@pytest.mark.parametrize("mutation", ["condition", "component", "shape", "dtype", "device", "guidance", "timestep"])
def test_identity_mutations_fail(mutation):
    from dataclasses import replace

    runtime = setup_runtime()
    with pytest.raises(ValueError, match="identity"):
        with runtime.request_context():
            x = bind(runtime)
            context = StepContext(0, 20.0, 1, 5.0)
            conditions = (2, 3)
            if mutation == "condition":
                conditions = (2, 4)
            if mutation == "component":
                context = replace(context, component=2)
            if mutation == "shape":
                x = x[..., :64]
            if mutation == "dtype":
                x = x.to(torch.bfloat16)
            if mutation == "device":
                x = x.to("meta")
            if mutation == "guidance":
                context = replace(context, guidance_scale=6)
            if mutation == "timestep":
                context = replace(context, timestep=19)
            runtime.predict_pair(context, x, conditions, lambda branch: pytest.fail("must fail before forward"))
    assert runtime.state is None
    with runtime.request_context():
        step(runtime, 0, bind(runtime))
        assert runtime.state is not None
        assert runtime.state.completed_pairs == 1


@pytest.mark.parametrize("where", ["cond", "scheduler", "decode", "cancel"])
def test_exit_failures_preserve_error_release_and_retry(where):
    runtime = setup_runtime()
    error = KeyboardInterrupt("cancel") if where == "cancel" else RuntimeError(where)
    with pytest.raises(type(error)) as caught:
        with runtime.request_context():
            x = bind(runtime)
            if where == "cond":

                def fail(branch):
                    raise error

                step(runtime, 0, x, fail)
            else:
                step(runtime, 0, x)
                raise error
    assert caught.value is error
    assert runtime.state is None and runtime.ledger is None
    assert runtime.last_summary["model_forward_calls"] == (1 if where == "cond" else 2)
    with runtime.request_context():
        step(runtime, 0, bind(runtime))


@pytest.mark.parametrize("mode", ["accelerate", "trace", "force_exact"])
@pytest.mark.parametrize("size", [1, 65, 1024])
def test_capacity_envelope_covers_liveness(mode, size):
    runtime = setup_runtime(mode=mode, base_threshold=0.004)
    with runtime.request_context():
        x = torch.ones(1, 1, 1, 1, size)
        total = 30
        runtime.bind_schedule(
            SchedulePlan(tuple(float(total - i) for i in range(total)), "fake", "{}"),
            GenerationSignature(runtime.epoch, 1, (2, 3), 5.0, tuple(x.shape), x.dtype, x.device),
        )
        for i in range(total):
            step(runtime, i, x + i * 0.008)
            assert runtime.peak_cache_bytes <= runtime.estimate_capacity(x, mode == "trace")


def test_missing_schedule_fails_before_forward():
    runtime = setup_runtime()
    with pytest.raises(RuntimeError, match="schedule"):
        with runtime.request_context():
            runtime.predict_pair(
                StepContext(0, 20, 1, 5), torch.ones(128), (2, 3), lambda branch: pytest.fail("unexpected forward")
            )
    assert runtime.state is None
    with runtime.request_context():
        step(runtime, 0, bind(runtime))
        assert runtime.state is not None
        assert runtime.state.completed_pairs == 1


@pytest.mark.parametrize("mode", ["accelerate", "trace", "force_exact"])
def test_capacity_fallback_rejects_mixed_pair_dtypes(mode):
    runtime = setup_runtime(mode=mode, max_cache_bytes=1)
    with pytest.raises(ValueError, match="dtype"):
        with runtime.request_context():
            x = bind(runtime)
            step(runtime, 0, x, lambda branch: x if branch == "cond" else x.to(torch.bfloat16))
    assert runtime.last_summary["completed_pairs"] == 0
    assert runtime.last_summary["model_forward_calls"] == 2


def test_capacity_fallback_preserves_native_dtype_identity():
    runtime = setup_runtime(max_cache_bytes=1)
    with pytest.raises(ValueError, match="dtype"):
        with runtime.request_context():
            x = bind(runtime)
            step(runtime, 0, x)
            step(runtime, 1, x, lambda branch: x.to(torch.bfloat16))
    assert runtime.last_summary["completed_pairs"] == 1


def test_request_a_b_a_matches_fresh_a_across_prompt_seed_shape_and_cfg():
    runtime = setup_runtime(base_threshold=0.003)

    def generate(runtime, prompt, seed, width, guidance):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.rand((1, 1, 1, 1, width), generator=generator) + 1
        values = []
        with runtime.request_context():
            runtime.bind_schedule(
                SchedulePlan(tuple(float(20 - i) for i in range(20)), "fake", "{}"),
                GenerationSignature(
                    runtime.epoch, 1, (id(prompt), seed), guidance, tuple(initial.shape), initial.dtype, initial.device
                ),
            )
            for index in range(20):
                x = initial + index * 0.01

                def exact(branch):
                    return x * (0.2 if branch == "cond" else 0.15) + len(prompt) * 0.1

                output = runtime.predict_pair(
                    StepContext(index, float(20 - index), 1, guidance), x, (id(prompt), seed), exact
                )
                values.append(tuple(value.clone() for value in output))
            feedback = dict(runtime.state.feedback_scale)
        return values, feedback, dict(runtime.last_summary["fallback_counts"])

    independent = generate(setup_runtime(base_threshold=0.003), "request A", 11, 128, 5.0)
    first = generate(runtime, "request A", 11, 128, 5.0)
    generate(runtime, "different request B", 29, 65, 7.5)
    second = generate(runtime, "request A", 11, 128, 5.0)
    for candidate in (first, second):
        for actual_pair, expected_pair in zip(candidate[0], independent[0]):
            for actual, expected in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert candidate[1:] == independent[1:]


def test_repeated_step_releases_state_and_allows_new_request():
    runtime = setup_runtime()
    with pytest.raises(ValueError, match="out-of-order"):
        with runtime.request_context():
            x = bind(runtime)
            step(runtime, 0, x)
            step(runtime, 0, x, lambda branch: pytest.fail("duplicate step must not run the model"))
    assert runtime.state is None and runtime.plan is None
    assert runtime.last_summary["model_forward_calls"] == 2
    with runtime.request_context():
        step(runtime, 0, bind(runtime))
        assert runtime.state is not None
        assert runtime.state.completed_pairs == 1
