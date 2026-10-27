# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig
from vllm_omni.diffusion.cache.aligncache.flow import FlowRuntime, sample_velocity

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def config(profile, mode="accelerate", **kwargs):
    return AlignCacheConfig.parse({"align_cache": {"profile": profile, "mode": mode, **kwargs}})


def combine(outputs):
    return outputs[0] if len(outputs) == 1 else outputs[1] + 5 * (outputs[0] - outputs[1])


def run(runtime, branches=2, fail=False):
    outputs = []
    ref: Callable[[], torch.Tensor | None]
    with runtime.request_context():
        runtime.bind([torch.linspace(1, 0, 31).tolist()] * branches, (1, 2))
        for i in range(30):
            xs = [torch.ones(1, 8) * (1 + i * 0.01) for _ in range(branches)]

            def exact(count):
                values = []
                if not runtime.qwen:
                    count()
                for j, x in enumerate(xs):
                    if runtime.qwen:
                        count()
                    if fail and i == 1 and j == 1:
                        raise RuntimeError("joint failure")
                    values.append(x * 0.2 + 1 + j * 0.1)
                return values

            outputs.append(runtime.evaluate(i, xs, (1, 2), exact, combine))
        if runtime.streams[0].anchors:
            ref = weakref.ref(runtime.streams[0].anchors[-1].output)
        else:

            def no_anchor():
                return None

            ref = no_anchor

    assert not runtime.streams and runtime.schedule is None
    return outputs, ref


@pytest.mark.parametrize("profile", ["qwen21_reference_v1", "minimax_h3_reference_v1"])
@pytest.mark.parametrize("mode", ["accelerate", "trace", "force_exact", "passthrough"])
def test_flow_modes_and_cleanup(profile, mode):
    runtime = FlowRuntime(config(profile, mode))
    outputs, ref = run(runtime)
    summary = runtime.last_summary
    assert summary["completed_evaluations"] == 30
    if mode == "accelerate":
        assert summary["reasons"]["actual_fast"] > 0
        assert summary["model_forward_calls"] < (60 if runtime.qwen else 30)
    else:
        assert summary["model_forward_calls"] == (60 if runtime.qwen else 30)
        for i, pair in enumerate(outputs):
            torch.testing.assert_close(pair[0], torch.ones(1, 8) * (1 + i * 0.01) * 0.2 + 1)
    assert ref() is None


def test_qwen_single_branch_and_no_predicted_anchors():
    runtime = FlowRuntime(config("qwen21_reference_v1"))
    outputs, _ = run(runtime, branches=1)
    assert all(len(row) == 1 for row in outputs)
    assert runtime.last_summary["model_forward_calls"] < 30
    with runtime.request_context():
        runtime.bind([[1, 0.9, 0.8, 0.7, 0]], (1, 2))
        for i in range(3):
            x = torch.ones(1, 8) * (1 + i * 0.01)
            runtime.evaluate(i, [x], (1, 2), lambda count: count() or (x + 1,), combine)
        assert all(anchor.index <= 2 for anchor in runtime.streams[0].anchors)


@pytest.mark.parametrize("profile", ["qwen21_reference_v1", "minimax_h3_reference_v1"])
def test_failures_no_partial_commit_or_retry(profile):
    runtime = FlowRuntime(config(profile))
    with pytest.raises(RuntimeError, match="joint failure"):
        run(runtime, fail=True)
    assert runtime.last_summary["completed_evaluations"] == 1
    assert runtime.last_summary["model_forward_calls"] == (4 if runtime.qwen else 2)
    run(runtime)


@pytest.mark.parametrize("profile", ["qwen21_reference_v1", "minimax_h3_reference_v1"])
def test_capacity_fallback_and_zero_threshold(profile):
    for kwargs in ({"max_cache_bytes": 1}, {"base_threshold": 0}):
        runtime = FlowRuntime(config(profile, **kwargs))
        run(runtime)
        assert runtime.last_summary["model_forward_calls"] == (60 if runtime.qwen else 30)
        assert not runtime.streams


def test_sampling_deterministic_no_rng():
    x = torch.arange(10000.0)
    before = torch.random.get_rng_state()
    a = sample_velocity(x, True)
    assert len(a) == 4096 and torch.equal(a, sample_velocity(x, True))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(sample_velocity(x, False), x[::64])


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("mode", ["force_exact", "trace", "accelerate"])
def test_native_qwen_loop_prefill_scheduler_and_release(cfg, mode):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tests.diffusion.cache.aligncache_helpers import definitions
    from vllm_omni.diffusion.cache.aligncache.adapters.flow import QwenFlowRuntime

    env = definitions(
        "vllm_omni/diffusion/models/qwen_image_21/cfg_parallel.py",
        {"QwenImage21CFGParallelMixin"},
        {"torch": torch, "QwenImage21KVCache": list},
        {"diffuse"},
    )
    pipeline = env["QwenImage21CFGParallelMixin"]()
    pipeline.interrupt = False
    released = []
    pipeline.transformer = SimpleNamespace(
        causal_condition=True, transformer_blocks=[object()], release_request_geometry=lambda: released.append(True)
    )
    pipeline.scheduler = SimpleNamespace(
        timesteps=torch.arange(30, 0, -1), sigmas=torch.linspace(1, 0, 31), set_begin_index=lambda i: None
    )
    pipeline.progress_bar = lambda **kw: nullcontext(SimpleNamespace(update=lambda: None))
    calls = []
    cache_snapshots = []

    def native_predict(**kw):
        branch = kw["cache_branch"]
        cache = kw["kv_cache"]
        if branch not in cache[0]:
            cache[0][branch] = {"key": torch.ones(1)}
        calls.append(branch)
        cache_snapshots.append(cache[0][branch]["key"])
        return kw["hidden_states"] * 0.2 + (1 if branch == "cond" else 0.9)

    pipeline.predict_noise = native_predict
    pipeline.combine_cfg_noise = lambda c, u, g, normalize: u + g * (c - u)
    updates = []

    def scheduler(output, t, latent, do_cfg):
        updates.append(int(t))
        return latent - output * 0.001

    pipeline.scheduler_step_maybe_with_cfg = scheduler
    runtime = QwenFlowRuntime(config("qwen21_reference_v1", mode))
    pipeline._aligncache_runtime = runtime
    original_bind = runtime.bind_qwen

    def bind(pipe, x, *args):
        return original_bind(pipe, SimpleNamespace(shape=x.shape, device=torch.device("cuda:0")), *args)

    runtime.bind_qwen = bind
    x = torch.ones(1, 16, 4)
    cond = torch.ones(1, 2, 4)
    uncond = torch.zeros_like(cond) if cfg else None
    with runtime.request_context():
        pipeline.diffuse(
            cond,
            None,
            uncond,
            None,
            torch.ones(1, 18, dtype=torch.bool),
            None,
            x,
            None,
            [],
            pipeline.scheduler.timesteps,
            cfg,
            5.0 if cfg else 1.0,
            # The native T2I call always passes this output-format flag.
            additional_transformer_kwargs={"return_dict": False},
        )
    assert released == [True] and len(updates) == 30
    assert calls[0] == "cond" and (not cfg or calls[1] == "uncond")
    assert len({id(v) for v in cache_snapshots}) == (2 if cfg else 1)
    assert len(calls) < (60 if cfg else 30) if mode == "accelerate" else len(calls) == (60 if cfg else 30)
    assert runtime.last_summary["model_forward_calls"] == len(calls)


@pytest.mark.parametrize("mode", [None, "force_exact", "trace", "accelerate"])
@pytest.mark.parametrize("failure", [None, "model", "solver", "cancel", "callback"])
def test_native_h3_loop_preserves_av_solver_steps_and_cleanup(mode, failure):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tests.diffusion.cache.aligncache_helpers import definitions

    steps, progress, kv_events = [], [], []

    class ReferenceKVState:
        def begin_step(self, step, *, sigma):
            kv_events.append(("begin", step))

        def end_step(self):
            kv_events.append(("end", None))

        def close(self):
            kv_events.append(("close", None))

    def cancel(**kwargs):
        if failure == "cancel":
            raise RuntimeError("injected cancel")

    def callback(*args):
        if failure == "callback":
            raise RuntimeError("injected callback")

    class Solver:
        def __init__(self, sigmas):
            self.sigmas = sigmas

        def step(self, x, x0, i):
            if failure == "solver":
                raise RuntimeError("injected solver")
            steps.append(i)
            return x + (x0 - x) * 0.01

    def prepare(**kw):
        return kw["initial_video_rows"].clone(), kw["initial_audio_rows"].clone(), None, None

    env = definitions(
        "vllm_omni/diffusion/models/minimax_h3/denoise_loop.py",
        {"minimax_h3_denoise_loop"},
        {
            "torch": torch,
            "nullcontext": nullcontext,
            "MINIMAX_H3_IMGVID_COND_TIMESTEP": 0.999,
            "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP": 1.0,
            "current_omni_platform": SimpleNamespace(is_npu=lambda: False),
            "minimax_h3_prepare_denoise_rows": prepare,
            "create_h3_sample_solver": lambda sampler, s: Solver(s),
            "check_request_cancellation": cancel,
            "minimax_h3_publish_denoise_progress": lambda *a: progress.append(a),
            "minimax_h3_prepare_edit_rows": lambda rows, *a, **kw: (rows, None),
            "minimax_h3_rf_v_to_x0": lambda x, v, t: x - (1 - t) * v,
        },
    )
    branch = SimpleNamespace(
        update_mask_dev=torch.ones(8, dtype=torch.bool),
        audio_update_mask_dev=torch.ones(8, dtype=torch.bool),
        locked_audio_rows=None,
        update_mask=torch.ones(8, dtype=torch.bool),
        static_kwargs={"preserved": True},
    )
    branch.forward_kwargs = lambda **kw: kw
    calls = []

    def model(**kw):
        calls.append(1)
        if failure == "model":
            raise RuntimeError("injected model")
        return kw["video_rows"] * 0.2 + 1, kw["audio_rows"] * 0.2 + 0.9

    model.create_reference_kv_state = lambda **kwargs: ReferenceKVState() if mode is None else None
    runtime = FlowRuntime(config("minimax_h3_reference_v1", mode)) if mode is not None else None
    outcome = pytest.raises(RuntimeError, match=f"injected {failure}") if failure else nullcontext()
    with outcome, runtime.request_context() if runtime else nullcontext():
        env["minimax_h3_denoise_loop"](
            model=model,
            positive=branch,
            initial_video_rows=torch.ones(8, 96),
            initial_audio_rows=torch.ones(8, 32),
            keyframe_cond_rows=None,
            sigmas_video=torch.linspace(1, 0, 31).tolist(),
            sigmas_audio=torch.linspace(1, 0, 31).tolist(),
            device=torch.device("cpu"),
            aligncache_runtime=runtime,
            on_step=callback,
        )
    assert progress[-1] == (None, None, None)
    assert branch.static_kwargs == {"preserved": True}
    if runtime is not None:
        assert not runtime.active and not runtime.streams
        assert kv_events == []
    else:
        assert kv_events[-1] == ("close", None)
        if failure is None:
            assert [event for event, _ in kv_events].count("begin") == 30
            assert [event for event, _ in kv_events].count("end") == 30
    if failure is None:
        assert steps == [i for i in range(30) for _ in range(2)]
        assert len(calls) < 30 if mode == "accelerate" else len(calls) == 30
        if runtime is not None:
            assert runtime.last_summary["model_forward_calls"] == len(calls)


@pytest.mark.parametrize(
    "overrides",
    [
        {"anchor": {"enabled": False}},
        {"predictor": {"online_scale": False}},
        {"feedback": {"scale_init": {"mid": 0.9}}},
    ],
)
def test_flow_rejects_unsupported_wan_knobs(overrides):
    with pytest.raises(ValueError, match="align_cache"):
        config("qwen21_reference_v1", **overrides)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "ref2va"),
        ("latent_refine", {}),
        ("latent_upscale", (512, 512)),
        ("continuation", (10, 2)),
        ("num_outputs", 2),
    ],
)
def test_h3_rejects_unsupported_passes_before_dit(field, value):
    from types import SimpleNamespace

    from tests.diffusion.cache.aligncache_helpers import definitions

    cls = definitions(
        "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py",
        {"MiniMaxH3Pipeline"},
        {"check_request_cancellation": lambda: None, "torch": torch},
        {"forward"},
    )["MiniMaxH3Pipeline"]
    pipeline = cls()
    pipeline._aligncache_runtime = object()
    context = dict(task="t2va", num_outputs=1)
    context[field] = value
    pipeline._prepare_request_inputs = lambda *args: context
    pipeline.diffuse = lambda **kw: pytest.fail("must reject before DiT")
    with pytest.raises(ValueError, match="pure T2VA"):
        pipeline.forward(SimpleNamespace(prompts=["test"], sampling_params=object()))


def test_qwen_rejects_editing_at_bind():
    from vllm_omni.diffusion.cache.aligncache.adapters.flow import QwenFlowRuntime

    runtime = QwenFlowRuntime(config("qwen21_reference_v1"))
    with runtime.request_context(), pytest.raises(ValueError, match="editing"):
        runtime.bind_qwen(None, None, None, None, None, False, 1.0, torch.ones(1), {})
    assert runtime.schedule is None


@pytest.mark.parametrize("extras", [{"return_dict": True}, {"return_dict": False, "extra": 1}, {"extra": 1}])
def test_qwen_rejects_extra_transformer_conditions_at_bind(extras):
    from vllm_omni.diffusion.cache.aligncache.adapters.flow import QwenFlowRuntime

    runtime = QwenFlowRuntime(config("qwen21_reference_v1"))
    with runtime.request_context(), pytest.raises(ValueError, match="additional conditions"):
        runtime.bind_qwen(None, None, None, None, None, False, 1.0, None, extras)
    assert runtime.schedule is None


def test_capacity_output_dtype_stays_bound_and_no_retry():
    runtime = FlowRuntime(config("qwen21_reference_v1", max_cache_bytes=1))
    with pytest.raises(ValueError, match="output contract"):
        with runtime.request_context():
            runtime.bind([[1, 0.5, 0]], (1,))
            x = torch.ones(1, 8)
            runtime.evaluate(0, [x], (1,), lambda count: count() or (x,), combine)
            runtime.evaluate(1, [x], (1,), lambda count: count() or (x.to(torch.bfloat16),), combine)
    assert runtime.last_summary["completed_evaluations"] == 1 and runtime.calls == 2


def test_fast_nonfinite_falls_back_before_actual_forward(monkeypatch):
    runtime = FlowRuntime(config("qwen21_reference_v1", warmup_steps=2))
    original = runtime._prediction

    def corrupt(state, phase):
        return original(state, phase) * float("nan")

    monkeypatch.setattr(runtime, "_prediction", corrupt)
    run(runtime, branches=1)
    assert runtime.last_summary["model_forward_calls"] == 30
    assert runtime.last_summary["reasons"]["invalid_prediction"] > 0


@pytest.mark.parametrize("corruption", ["shape", "dtype", "none"])
def test_invalid_fast_contract_falls_back_without_native_retry(monkeypatch, corruption):
    runtime = FlowRuntime(config("qwen21_reference_v1", warmup_steps=2))
    original = runtime._prediction

    def invalid_prediction(state, phase):
        value = original(state, phase)
        if corruption == "shape":
            return value[..., :1]
        if corruption == "dtype":
            return value.to(torch.bfloat16)
        return None

    monkeypatch.setattr(runtime, "_prediction", invalid_prediction)
    run(runtime, branches=1)
    assert runtime.last_summary["model_forward_calls"] == 30
    assert runtime.last_summary["reasons"]["invalid_prediction"] > 0


def test_nonfinite_native_output_is_not_reported_as_capacity_fallback():
    runtime = FlowRuntime(config("minimax_h3_reference_v1"))
    with runtime.request_context():
        runtime.bind([[1, 0.5, 0]] * 2, (1,))
        x = torch.ones(1, 8)
        for index in range(2):

            def exact(count):
                count()
                return (x * float("nan"), x) if index == 0 else (x, x)

            runtime.evaluate(index, [x, x], (1,), exact)
    assert runtime.last_summary["reasons"].get("capacity_exceeded", 0) == 0
    assert runtime.last_summary["reasons"]["nonfinite_exact"] == 2


@pytest.mark.parametrize("mutation", ["replacement", "lost_prefill", "branch_name"])
def test_qwen_prefix_identity_validated_before_model(mutation):
    from types import SimpleNamespace

    from vllm_omni.diffusion.cache.aligncache.adapters.flow import QwenFlowRuntime

    runtime = QwenFlowRuntime(config("qwen21_reference_v1"))
    pipeline = SimpleNamespace(transformer=object(), scheduler=object())
    x, condition = torch.ones(1, 16, 4), torch.ones(1, 2, 4)
    cache: list[dict[str, Any]] = [{}]
    calls = []

    def predict(**kwargs):
        calls.append(kwargs["cache_branch"])
        kwargs["kv_cache"][0][kwargs["cache_branch"]] = {"key": torch.ones(1)}
        return kwargs["hidden_states"] + 1

    pipeline.predict_noise = predict
    positive = dict(hidden_states=x, encoder_hidden_states=condition, cache_branch="cond", kv_cache=cache)
    identity = (id(pipeline.transformer), id(pipeline.scheduler), id(condition), id(None), False, 1.0, tuple(x.shape))
    with runtime.request_context():
        runtime.bind([[1, 0.5, 0]], identity)
        runtime.predict_qwen(pipeline, 0, x, positive, None, False, 1.0)
        if mutation == "replacement":
            positive["kv_cache"] = [{}]
        elif mutation == "lost_prefill":
            cache[0].clear()
        else:
            positive["cache_branch"] = "uncond"
        with pytest.raises(ValueError, match="prefix|branch"):
            runtime.predict_qwen(pipeline, 1, x, positive, None, False, 1.0)
        assert calls == ["cond"] and runtime.index == 1


@pytest.mark.parametrize("schedules", [[[1, 0]], [[1, 0]] * 3])
def test_h3_rejects_non_av_stream_count(schedules):
    runtime = FlowRuntime(config("minimax_h3_reference_v1"))
    with runtime.request_context(), pytest.raises(ValueError, match="stream"):
        runtime.bind(schedules, (1,))


def test_wan_rejects_unused_flow_configuration():
    with pytest.raises(ValueError, match="flow"):
        AlignCacheConfig.parse({"align_cache": {"flow": {"audio_threshold": 0}}})


def test_overflowed_fit_keeps_previous_scale():
    from vllm_omni.diffusion.cache.aligncache.flow import fit

    x = torch.full((8,), 1e30)
    assert fit(x, x, 1.03, 0.9, 1.1) == 1.03


def test_quantized_flow_weights_are_admitted_only_with_opt_in():
    from types import SimpleNamespace

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vllm_omni.diffusion.cache.aligncache.adapters.flow import validate_pipeline
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel

    def pipeline(dtypes, device="cuda"):
        transformer = object.__new__(QwenImage21Transformer2DModel)
        torch.nn.Module.__init__(transformer)
        transformer.parameters = lambda: iter([SimpleNamespace(dtype=d, device=torch.device(device)) for d in dtypes])
        result = object.__new__(QwenImage21Pipeline)
        torch.nn.Module.__init__(result)
        result.transformer = transformer
        result.scheduler = FlowMatchEulerDiscreteScheduler()
        return result

    quantized = pipeline((torch.float8_e4m3fn, torch.bfloat16, torch.float32))
    validate_pipeline(quantized, "qwen_flow_v1", quantized=True)
    with pytest.raises(ValueError, match="BF16"):
        validate_pipeline(quantized, "qwen_flow_v1")
    for dtypes in ((torch.float8_e4m3fn, torch.float32), (torch.bfloat16, torch.float16), (torch.bfloat16, torch.int8)):
        with pytest.raises(ValueError, match="online FP8"):
            validate_pipeline(pipeline(dtypes), "qwen_flow_v1", quantized=True)
    with pytest.raises(ValueError, match="CUDA"):
        validate_pipeline(pipeline((torch.bfloat16, torch.float8_e4m3fn), "cpu"), "qwen_flow_v1", quantized=True)
    validate_pipeline(pipeline((torch.bfloat16,)), "qwen_flow_v1")
