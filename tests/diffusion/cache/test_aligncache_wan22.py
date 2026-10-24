# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU Wan component routing contracts; these do not establish checkpoint/GPU compatibility."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig
from vllm_omni.diffusion.cache.aligncache.multicomponent import WanRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_pipeline(two_experts=True, out_channels=None):
    timesteps = torch.arange(40, 0, -1).float()
    pipeline = SimpleNamespace(
        transformer=object(),
        transformer_2=object() if two_experts else None,
        expand_timesteps=False,
        is_dmd=False,
        record_denoise_step=lambda *a: None,
        _sample_solver="unipc",
        scheduler=SimpleNamespace(timesteps=timesteps, num_inference_steps=40, config={}),
    )
    pipeline.calls = []

    def predict_noise(**kwargs):
        pipeline.calls.append(kwargs["current_model"])
        if (runtime := getattr(pipeline, "_aligncache_runtime", None)) is not None:
            runtime.record_forward(kwargs["current_model"])
        return kwargs["hidden_states"][:, :out_channels] * 0.2 + kwargs["encoder_hidden_states"].mean()

    pipeline.predict_noise = predict_noise
    pipeline.combine_cfg_noise = lambda c, u, g, normalize: u + g * (c - u)

    def native(cfg, guidance, positive, negative, cfg_normalize):
        return pipeline.combine_cfg_noise(predict_noise(**positive), predict_noise(**negative), guidance, cfg_normalize)

    pipeline.predict_noise_maybe_with_cfg = native
    return pipeline


def bind_cpu(runtime, pipeline, x, cond, uncond, boundary=21, guidance=(5.0, 3.0), conditions=()):
    # Exercise production plan building with CUDA admission metadata; execute only toy CPU tensors.
    metadata = SimpleNamespace(shape=x.shape, ndim=x.ndim, device=torch.device("cuda:0"))
    runtime.bind_schedule(
        pipeline, pipeline.scheduler.timesteps, metadata, cond, uncond, *guidance, boundary, torch.bfloat16, *conditions
    )
    for child in runtime.components.values():
        if child.signature is not None:
            child.signature = replace(child.signature, device=x.device, dtype=x.dtype)


def test_profiles_are_explicit_and_independent():
    config = AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}})
    assert config.policy == "wan22_feedback_v1" and config.base_threshold == 0.078 and config.warmup_steps == 9
    assert AlignCacheConfig.parse().warmup_steps == 12
    assert AlignCacheConfig.parse({"align_cache": {"profile": "wan21_14b_reference_v1"}}).policy == "wan21_feedback_v1"
    with pytest.raises(ValueError, match="profile"):
        AlignCacheConfig.parse({"align_cache": {"policy": "wan22_feedback_v1"}})
    for profile, components in (
        ("wan22_i2v_reference_v1", ["high", "low"]),
        ("wan22_ti2v_reference_v1", ["transformer"]),
    ):
        config = AlignCacheConfig.parse({"align_cache": {"profile": profile}})
        assert config.policy == "wan22_feedback_v1" and config.base_threshold == 0.065 and config.warmup_steps == 9
        assert list(WanRuntime(config).components) == components
    with pytest.raises(ValueError, match="profile"):
        AlignCacheConfig.parse({"align_cache": {"profile": "wan22_ti2v_reference_v1", "policy": "wan21_feedback_v1"}})


@pytest.mark.parametrize("mode", ["accelerate", "trace", "force_exact", "passthrough"])
def test_component_local_state_cfg_and_counts(mode):
    runtime = WanRuntime(
        AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1", "mode": mode, "warmup_steps": 2}})
    )
    pipeline = make_pipeline()
    pipeline._aligncache_runtime = runtime
    x, cond, uncond = torch.ones(1, 16, 1, 1, 8), torch.ones(1, 2), torch.full((1, 2), 0.9)
    with runtime.request_context():
        bind_cpu(runtime, pipeline, x, cond, uncond)
        high, low = runtime.components["high"], runtime.components["low"]
        assert high.plan.evaluations == low.plan.evaluations == 20
        assert runtime.routes[19] == "high" and runtime.routes[20] == "low"
        assert high.signature.guidance_scale == 5 and low.signature.guidance_scale == 3
        for i, t in enumerate(pipeline.scheduler.timesteps):
            is_high = float(t) >= 21
            model, guidance = (pipeline.transformer, 5.0) if is_high else (pipeline.transformer_2, 3.0)
            current = x + i * 0.002
            positive = dict(hidden_states=current, current_model=model, encoder_hidden_states=cond)
            negative = dict(hidden_states=current, current_model=model, encoder_hidden_states=uncond)
            before = len(pipeline.calls)
            result = runtime.predict_noise(pipeline, i, t, guidance, positive, negative)
            if i in (0, 20):
                assert len(pipeline.calls) - before == 2
            if mode != "accelerate":
                torch.testing.assert_close(result, (current * 0.2 + 0.9) + guidance * 0.1)
        assert high.state is not low.state
        if mode == "accelerate":
            assert high.counts["fast_pairs"] > 0 and low.counts["fast_pairs"] > 0
        assert high.counts["completed_pairs"] == low.counts["completed_pairs"] == 20
    assert all(child.state is None and child.plan is None for child in runtime.components.values())
    assert runtime.routes is None
    summary = runtime.last_summary
    assert summary["completed_pairs"] == 40
    assert summary["model_forward_calls"] == len(pipeline.calls) == 2 * summary["exact_pairs"]


@pytest.mark.parametrize("boundary,active", [(0, "high"), (100, "low")])
def test_empty_component_plan(boundary, active):
    runtime = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}}))
    pipeline = make_pipeline()
    with runtime.request_context():
        bind_cpu(runtime, pipeline, torch.ones(1, 16, 1, 1, 8), torch.ones(1, 2), torch.zeros(1, 2), boundary)
        assert runtime.components[active].plan.evaluations == 40
        assert runtime.components["low" if active == "high" else "high"].plan is None


def test_second_component_entry_failure_unwinds_first():
    runtime = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}}))
    error = RuntimeError("second component failed")

    @contextmanager
    def fail():
        raise error
        yield

    runtime.components["low"].request_context = fail
    with pytest.raises(RuntimeError) as caught:
        with runtime.request_context():
            pytest.fail("entry must fail")
    assert caught.value is error
    assert not runtime.active
    assert not runtime.components["high"].active and runtime.components["high"].state is None


@pytest.mark.parametrize("mode", ["accelerate", "force_exact", "trace", "passthrough"])
def test_actual_wan22_loop_preserves_native_routing(mode):
    from types import MethodType

    from tests.diffusion.cache.aligncache_helpers import definitions

    runtime = WanRuntime(
        AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1", "mode": mode, "warmup_steps": 2}})
    )
    pipeline = make_pipeline()
    pipeline._aligncache_runtime = runtime
    original_bind = runtime.bind_schedule

    def bind(*args):
        runtime.bind_schedule = original_bind
        try:
            bind_cpu(runtime, pipeline, args[2], args[3], args[4])
        finally:
            runtime.bind_schedule = bind

    runtime.bind_schedule = bind

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self):
            pass

    pipeline.progress_bar = lambda **kwargs: Progress()
    updates = []

    def scheduler(noise, t, latents, cfg):
        assert cfg
        updates.append(float(t))
        return latents - noise * 0.001

    pipeline.scheduler_step_maybe_with_cfg = scheduler
    cls = definitions(
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        {"Wan22Pipeline"},
        {"set_forward_context_denoise_step_idx": lambda i: None},
        {"diffuse"},
    )["Wan22Pipeline"]
    pipeline.diffuse = MethodType(cls.diffuse, pipeline)
    with runtime.request_context():
        pipeline.diffuse(
            torch.ones(1, 16, 1, 1, 8),
            pipeline.scheduler.timesteps,
            torch.ones(1, 2),
            torch.full((1, 2), 0.9),
            5.0,
            3.0,
            21.0,
            torch.float32,
            {},
        )
    assert len(updates) == 40
    assert runtime.last_summary["model_forward_calls"] == len(pipeline.calls)
    assert len(pipeline.calls) < 80 if mode == "accelerate" else len(pipeline.calls) == 80
    assert all(child.state is None for child in runtime.components.values())


@pytest.mark.parametrize("failure", [None, "scheduler", "binding"])
def test_actual_wan_loop_runs_wan21_with_one_component(failure):
    from types import MethodType

    from tests.diffusion.cache.aligncache_helpers import definitions

    cls = definitions(
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        {"Wan22Pipeline"},
        {"set_forward_context_denoise_step_idx": lambda i: None},
        {"diffuse"},
    )["Wan22Pipeline"]

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self):
            pass

    scheduler_error = RuntimeError("injected scheduler failure")
    results = {}
    modes = ("passthrough", "accelerate") if failure else (None, "passthrough", "force_exact", "trace", "accelerate")
    for mode in modes:
        pipeline = make_pipeline(two_experts=False)
        pipeline.progress_bar = lambda **kwargs: Progress()
        pipeline.diffuse = MethodType(cls.diffuse, pipeline)

        def native(do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize):
            assert do_true_cfg and cfg_normalize is False
            cond = pipeline.predict_noise(**positive_kwargs)
            return pipeline.combine_cfg_noise(cond, pipeline.predict_noise(**negative_kwargs), true_cfg_scale, False)

        def scheduler(noise, t, latents, cfg):
            if failure == "scheduler":
                raise scheduler_error
            if failure == "binding":
                pipeline.transformer = object()
            return latents - noise * 0.001

        pipeline.predict_noise_maybe_with_cfg = native
        pipeline.scheduler_step_maybe_with_cfg = scheduler
        runtime = None
        if mode is not None:
            runtime = WanRuntime(AlignCacheConfig.parse({"align_cache": {"mode": mode, "warmup_steps": 2}}))
            pipeline._aligncache_runtime = runtime
            original_bind = runtime.bind_schedule

            def bind(*args, runtime=runtime, original_bind=original_bind, pipeline=pipeline):
                runtime.bind_schedule = original_bind
                try:
                    bind_cpu(runtime, pipeline, args[2], args[3], args[4], args[7], args[5:7])
                finally:
                    runtime.bind_schedule = bind

            runtime.bind_schedule = bind
        context = runtime.request_context() if runtime else nullcontext()
        if failure:
            expected_error = pytest.raises(RuntimeError if failure == "scheduler" else ValueError)
        else:
            expected_error = nullcontext()
        with expected_error as caught, context:
            results[mode] = pipeline.diffuse(
                torch.ones(1, 16, 1, 1, 8),
                pipeline.scheduler.timesteps,
                torch.ones(1, 2),
                torch.full((1, 2), 0.9),
                5.0,
                5.0,
                875.0,
                torch.float32,
                {},
            )
        if failure:
            child = runtime.components["transformer"]
            assert len(pipeline.calls) == 2 and child.state is None and child.plan is None
            assert runtime.last_summary["completed_pairs"] == 1
            assert runtime.last_summary["status"] == "failed"
            if failure == "scheduler":
                assert caught.value is scheduler_error
            continue
        assert set(pipeline.calls) == {pipeline.transformer}
        assert len(pipeline.calls) < 80 if mode == "accelerate" else len(pipeline.calls) == 80
        if runtime is not None:
            assert list(runtime.components) == ["transformer"]
            assert runtime.last_summary["completed_pairs"] == 40
            assert runtime.last_summary["model_forward_calls"] == len(pipeline.calls)
    for mode in () if failure else ("passthrough", "force_exact", "trace"):
        torch.testing.assert_close(results[mode], results[None], rtol=0, atol=0)


def test_large_transformer_structural_admission_is_profile_specific():
    from vllm_omni.diffusion.cache.aligncache.adapters.wan21 import validate_transformer

    class Model:
        config = SimpleNamespace(
            in_channels=16,
            out_channels=16,
            num_layers=40,
            num_attention_heads=40,
            attention_head_dim=128,
            text_dim=4096,
            ffn_dim=13824,
            freq_dim=256,
            image_dim=None,
            added_kv_proj_dim=None,
            patch_size=(1, 2, 2),
        )

        def modules(self):
            return iter([self])

        def parameters(self):
            return iter([SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cuda:0"))])

    model = Model()
    validate_transformer(model, large=True)
    with pytest.raises(ValueError, match="num_layers"):
        validate_transformer(model)
    model.config.image_dim = 1280
    with pytest.raises(ValueError, match="image_dim"):
        validate_transformer(model, large=True)


class Progress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self):
        pass


def build_image_loop(profile, mode, failure=None):
    """Build the actual I2V-A14B or TI2V-5B denoise loop on CPU; ``mode=None`` is the native path."""
    from types import MethodType

    from tests.diffusion.cache.aligncache_helpers import definitions

    i2v = profile == "wan22_i2v_reference_v1"
    path, name = (
        ("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py", "Wan22I2VPipeline")
        if i2v
        else ("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py", "Wan22Pipeline")
    )
    cls = definitions(
        path, {name}, {"torch": torch, "set_forward_context_denoise_step_idx": lambda i: None}, {"diffuse"}
    )[name]
    pipeline = make_pipeline(two_experts=i2v, out_channels=16 if i2v else None)
    pipeline.expand_timesteps = not i2v
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2, 2))
    pipeline.progress_bar = lambda **kwargs: Progress()
    pipeline.diffuse = MethodType(cls.diffuse, pipeline)
    error = RuntimeError("injected scheduler failure")

    def native(do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize):
        assert do_true_cfg and cfg_normalize is False
        cond = pipeline.predict_noise(**positive_kwargs)
        return pipeline.combine_cfg_noise(cond, pipeline.predict_noise(**negative_kwargs), true_cfg_scale, False)

    pipeline.predict_noise_maybe_with_cfg = native

    def scheduler(noise, t, latents, cfg):
        assert cfg and noise.shape == latents.shape
        if failure == "scheduler":
            raise error
        return latents - noise * 0.001

    pipeline.scheduler_step_maybe_with_cfg = scheduler
    latents = torch.ones(1, 16, 2, 2, 8)
    condition = torch.full((1, 20, 2, 2, 8), 0.5) if i2v else torch.full_like(latents, 0.5)
    mask = torch.ones(1, 1, 2, 2, 8)
    mask[:, :, 0] = 0
    guidance, boundary = ((5.0, 3.0), 21.0) if i2v else ((5.0, 5.0), None)
    runtime = None
    if mode is not None:
        runtime = WanRuntime(
            AlignCacheConfig.parse({"align_cache": {"profile": profile, "mode": mode, "warmup_steps": 2}})
        )
        pipeline._aligncache_runtime = runtime
        original_bind = runtime.bind_schedule

        def bind(*args):
            runtime.bind_schedule = original_bind
            try:
                bind_cpu(runtime, pipeline, args[2], args[3], args[4], args[7], args[5:7], args[9:])
            finally:
                runtime.bind_schedule = bind

        runtime.bind_schedule = bind
    cond, uncond = torch.ones(1, 2), torch.full((1, 2), 0.9)
    if i2v:
        args = (latents, pipeline.scheduler.timesteps, cond, uncond, None, *guidance, boundary, torch.float32, {})
        kwargs = dict(condition=condition, first_frame_mask=mask)
    else:
        args = (latents, pipeline.scheduler.timesteps, cond, uncond, *guidance, boundary, torch.float32, {})
        kwargs = dict(latent_condition=condition, first_frame_mask=mask)

    def run():
        with runtime.request_context() if runtime else nullcontext():
            return pipeline.diffuse(*args, **kwargs)

    return pipeline, runtime, run


def routes(pipeline):
    return [model is pipeline.transformer for model in pipeline.calls]


@pytest.mark.parametrize("profile", ["wan22_i2v_reference_v1", "wan22_ti2v_reference_v1"])
def test_actual_image_conditioned_loops_match_native_exact_modes(profile):
    native_pipeline, _, run = build_image_loop(profile, None)
    native = run()
    assert len(native_pipeline.calls) == 80
    for mode in ("passthrough", "force_exact", "trace", "accelerate"):
        pipeline, runtime, run = build_image_loop(profile, mode)
        result = run()
        assert runtime.last_summary["status"] == "completed"
        assert runtime.last_summary["completed_pairs"] == 40
        assert runtime.last_summary["model_forward_calls"] == len(pipeline.calls)
        assert all(child.state is None for child in runtime.components.values())
        if mode == "accelerate":
            assert len(pipeline.calls) < 80
            continue
        assert routes(pipeline) == routes(native_pipeline)
        torch.testing.assert_close(result, native, rtol=0, atol=0)
    if profile == "wan22_i2v_reference_v1":
        assert set(native_pipeline.calls) == {native_pipeline.transformer, native_pipeline.transformer_2}
    else:
        assert set(native_pipeline.calls) == {native_pipeline.transformer}


@pytest.mark.parametrize("profile", ["wan22_i2v_reference_v1", "wan22_ti2v_reference_v1"])
def test_image_conditioned_loop_failure_unwinds(profile):
    pipeline, runtime, run = build_image_loop(profile, "accelerate", failure="scheduler")
    with pytest.raises(RuntimeError, match="injected scheduler failure"):
        run()
    assert len(pipeline.calls) == 2 and not runtime.active and runtime.routes is None
    assert runtime.last_summary["status"] == "failed" and runtime.last_summary["completed_pairs"] == 1
    assert all(child.state is None and child.plan is None for child in runtime.components.values())


def test_image_conditioned_admission_is_profile_specific():
    pipeline = make_pipeline()
    x, cond, uncond = torch.ones(1, 16, 1, 1, 8), torch.ones(1, 2), torch.zeros(1, 2)
    condition, mask = torch.ones(1, 20, 1, 1, 8), torch.ones(1, 1, 1, 1, 8)
    t2v = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}}))
    with t2v.request_context(), pytest.raises(ValueError, match="image-conditioned"):
        bind_cpu(t2v, pipeline, x, cond, uncond, conditions=(condition, mask))
    i2v = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_i2v_reference_v1"}}))
    with i2v.request_context(), pytest.raises(ValueError, match="I2V-A14B"):
        bind_cpu(i2v, pipeline, x, cond, uncond)
    with i2v.request_context():
        bind_cpu(i2v, pipeline, x, cond, uncond, conditions=(condition, mask))
        model = pipeline.transformer
        current = torch.cat([x, condition], dim=1)
        positive = dict(hidden_states=current, current_model=model, encoder_hidden_states=cond)
        negative = dict(hidden_states=current, current_model=model, encoder_hidden_states=uncond)
        t = pipeline.scheduler.timesteps[0]
        with pytest.raises(ValueError, match="noisy latent"):
            i2v.predict_noise(pipeline, 0, t, 5.0, positive, negative)
        with pytest.raises(ValueError, match="lead"):
            i2v.predict_noise(pipeline, 0, t, 5.0, positive, negative, latents=current)
    ti2v = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_ti2v_reference_v1"}}))
    single = make_pipeline(two_experts=False)
    with ti2v.request_context(), pytest.raises(ValueError, match="TI2V-5B"):
        bind_cpu(ti2v, single, x, cond, uncond, None, (5.0, 5.0))
    single.expand_timesteps = True
    with ti2v.request_context(), pytest.raises(ValueError, match="TI2V-5B"):
        bind_cpu(ti2v, single, x, cond, uncond, None, (5.0, 5.0), (x, None))
    with ti2v.request_context(), pytest.raises(ValueError, match="one guidance scale"):
        bind_cpu(ti2v, single, x, cond, uncond, None, (5.0, 3.0))
    with ti2v.request_context():
        bind_cpu(ti2v, single, x, cond, uncond, None, (5.0, 5.0))
        assert ti2v.components["transformer"].plan.evaluations == 40


def test_wan22_adapter_structural_allowlist(monkeypatch):
    import sys
    import types

    from vllm_omni.diffusion.cache.aligncache.adapters.wan22 import validate_pipeline

    class Config(dict):
        __getattr__ = dict.__getitem__

    class Scheduler:
        config = Config(
            prediction_type="flow_prediction",
            num_train_timesteps=1000,
            solver_order=2,
            solver_type="bh2",
            predict_x0=True,
            use_dynamic_shifting=False,
            thresholding=False,
            lower_order_final=True,
            timestep_spacing="linspace",
            steps_offset=0,
            final_sigmas_type="zero",
            shift=5.0,
        )
        solver_p = None
        disable_corrector = []

    class Transformer:
        def __init__(self, **overrides):
            self.config = SimpleNamespace(
                **{
                    "in_channels": 16,
                    "out_channels": 16,
                    "num_layers": 40,
                    "num_attention_heads": 40,
                    "attention_head_dim": 128,
                    "text_dim": 4096,
                    "ffn_dim": 13824,
                    "freq_dim": 256,
                    "image_dim": None,
                    "added_kv_proj_dim": None,
                    "patch_size": (1, 2, 2),
                    **overrides,
                }
            )

        def modules(self):
            return iter([self])

        def parameters(self):
            return iter([SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cuda:0"))])

    class Pipeline:
        has_transformer_2 = True
        expand_timesteps = False
        is_dmd = False

        def __init__(self, *transformers):
            self.transformer, self.transformer_2 = (*transformers, None)[:2]
            self.scheduler = Scheduler()

    class I2VPipeline(Pipeline):
        image_encoder = None

    for name, attr, value in (
        ("vllm_omni.diffusion.models.schedulers", "FlowUniPCMultistepScheduler", Scheduler),
        ("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2", "Wan22Pipeline", Pipeline),
        ("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v", "Wan22I2VPipeline", I2VPipeline),
        ("vllm_omni.diffusion.models.wan2_2.wan2_2_transformer", "WanTransformer3DModel", Transformer),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    validate_pipeline(Pipeline(Transformer(), Transformer()))
    with pytest.raises(ValueError, match="native A14B"):
        validate_pipeline(Pipeline(Transformer(), Transformer()), "wan22_i2v_reference_v1")

    i2v = I2VPipeline(Transformer(in_channels=36), Transformer(in_channels=36))
    validate_pipeline(i2v, "wan22_i2v_reference_v1")
    with pytest.raises(ValueError, match="native A14B"):
        validate_pipeline(i2v)
    i2v.image_encoder = object()
    with pytest.raises(ValueError, match="CLIP"):
        validate_pipeline(i2v, "wan22_i2v_reference_v1")
    i2v.image_encoder = None
    i2v.transformer_2 = i2v.transformer
    with pytest.raises(ValueError, match="distinct"):
        validate_pipeline(i2v, "wan22_i2v_reference_v1")
    i2v.transformer_2 = Transformer(in_channels=36, image_dim=1280)
    with pytest.raises(ValueError, match="image_dim"):
        validate_pipeline(i2v, "wan22_i2v_reference_v1")

    ti2v_structure = dict(in_channels=48, out_channels=48, num_layers=30, num_attention_heads=24, ffn_dim=14336)
    ti2v = Pipeline(Transformer(**ti2v_structure))
    ti2v.has_transformer_2 = False
    ti2v.expand_timesteps = True
    validate_pipeline(ti2v, "wan22_ti2v_reference_v1")
    ti2v.transformer = Transformer(**{**ti2v_structure, "num_layers": 40})
    with pytest.raises(ValueError, match="num_layers"):
        validate_pipeline(ti2v, "wan22_ti2v_reference_v1")
    ti2v.transformer_2 = Transformer(**ti2v_structure)
    with pytest.raises(ValueError, match="one native transformer"):
        validate_pipeline(ti2v, "wan22_ti2v_reference_v1")
    ti2v.expand_timesteps = False
    with pytest.raises(ValueError, match="expand-timesteps"):
        validate_pipeline(ti2v, "wan22_ti2v_reference_v1")
