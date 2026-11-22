# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed import cfg_parallel
from vllm_omni.diffusion.guidance import (
    FlowMatchingPredictionAdapter,
    SkipLayerGuidanceConfig,
    SkipLayerGuidanceMixin,
    prepare_skip_layer_guidance,
)
from vllm_omni.diffusion.guidance.skip_layer_guidance import combine_skip_layer_guidance

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


class ToyPipeline(SkipLayerGuidanceMixin):
    slg_prediction_adapter = FlowMatchingPredictionAdapter()

    def __init__(self):
        self.calls = []

    def predict_noise(self, x, condition, skip_layers=()):
        self.calls.append((condition, skip_layers))
        return x * (0.2 if skip_layers else 0.5) + condition


def inputs():
    x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2) / 8
    return x, {"x": x, "condition": 2.0}, {"x": x, "condition": -1.0}


def test_cfg_rejects_missing_negative_arguments_before_forward():
    pipeline = ToyPipeline()
    _, positive, _ = inputs()
    with pytest.raises(ValueError, match="CFG requires negative"):
        pipeline.predict_noise_maybe_with_cfg(True, 3, positive, None)
    assert not pipeline.calls


@pytest.mark.parametrize("layers", [[9, 7, 8, 7], "9, 7, 8, 7", (7, 8, 9)])
def test_config_normalizes_layers(layers):
    config = SkipLayerGuidanceConfig.from_extra_args({"slg_skip_layers": layers, "slg_scale": 3})
    assert config.skip_layers == (7, 8, 9)
    assert config.enabled
    assert config.batch_key == SkipLayerGuidanceConfig((9, 8, 7), 3).batch_key


@pytest.mark.parametrize(
    "extra",
    [
        {"slg_skip_layers": [-1]},
        {"slg_skip_layers": [1.5]},
        {"slg_skip_layers": [True]},
        {"slg_skip_layers": "1,,2"},
        {"slg_skip_layers": "1-3"},
        {"uncond_skip_layers": [True]},
        {"uncond_skip_layers": [-1]},
        {"slg_scale": -1},
        {"slg_scale": float("nan")},
        {"slg_scale": None},
        {"slg_rescaling_scale": float("inf")},
        {"slg_rescaling_scale": -0.1},
        {"slg_start_percent": -0.1},
        {"slg_end_percent": 1.1},
        {"slg_start_percent": 0.5, "slg_end_percent": 0.1},
    ],
)
def test_invalid_config_is_rejected(extra):
    with pytest.raises(ValueError, match="SLG|slg_|uncond_skip_layers"):
        SkipLayerGuidanceConfig.from_extra_args(extra)


def test_standalone_parameter_names_defaults_and_legacy_aliases():
    current = SkipLayerGuidanceConfig.from_extra_args(
        {"slg_layers": [22], "uncond_skip_layers": [24], "slg_scale": 3, "slg_start": 0.01, "slg_stop": 1}
    )
    legacy = SkipLayerGuidanceConfig.from_extra_args(
        {
            "slg_skip_layers": "22,22",
            "uncond_skip_layers": "24",
            "slg_scale": 3,
            "slg_start_percent": 0.01,
            "slg_end_percent": 1,
        }
    )
    assert current == legacy
    assert current.batch_key == legacy.batch_key
    defaults = SkipLayerGuidanceConfig.from_extra_args({"slg_layers": None, "uncond_skip_layers": None})
    assert defaults.start == 0 and defaults.stop == 1
    assert not defaults.enabled
    redundant = SkipLayerGuidanceConfig.from_extra_args(
        {"slg_layers": [22], "slg_skip_layers": "22,22", "slg_scale": 3, "slg_start": "0", "slg_start_percent": 0}
    )
    assert redundant.skip_layers == (22,)


@pytest.mark.parametrize(
    "extra",
    [
        {"slg_layers": [22], "slg_skip_layers": [24]},
        {"slg_start": 0.01, "slg_start_percent": 0.1},
        {"slg_stop": 1, "slg_end_percent": 0.15},
    ],
)
def test_conflicting_aliases_are_rejected(extra):
    with pytest.raises(ValueError, match="Conflicting"):
        SkipLayerGuidanceConfig.from_extra_args(extra)


@pytest.mark.parametrize("do_cfg", [False, True])
@pytest.mark.parametrize(
    "config,active",
    [
        (None, ()),
        (SkipLayerGuidanceConfig((1,), 0), (True, True)),
        (SkipLayerGuidanceConfig((), 3), (True, True)),
        (SkipLayerGuidanceConfig((1,), 3), (False, False)),
    ],
)
def test_disabled_slg_preserves_base_prediction_and_forward_count(do_cfg, config, active):
    pipeline = ToyPipeline()
    x, positive, negative = inputs()
    expected = pipeline.predict_noise_maybe_with_cfg(do_cfg, 4, positive, negative, cfg_normalize=False)
    pipeline.calls.clear()
    actual = pipeline.predict_noise_with_slg(
        do_cfg, 4, positive, negative, slg_config=config, active=active, sigmas=(0.9, 0.1), latents=x
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(pipeline.calls) == (2 if do_cfg else 1)
    assert all(not skip for _, skip in pipeline.calls)
    assert "skip_layers" not in positive


@pytest.mark.parametrize("do_cfg", [False, True])
@pytest.mark.parametrize("rescale", [0.0, 0.7, 2.0])
def test_slg_matches_denoised_reference_and_keeps_inactive_rows(do_cfg, rescale):
    pipeline = ToyPipeline()
    x, positive, negative = inputs()
    config = SkipLayerGuidanceConfig((1,), 3, rescaling_scale=rescale)
    pos = pipeline.predict_noise(**positive)
    neg = pipeline.predict_noise(**negative)
    skipped = pipeline.predict_noise(**positive, skip_layers=config.skip_layers)
    base = neg + 4 * (pos - neg) if do_cfg else pos
    pipeline.calls.clear()
    actual = pipeline.predict_noise_with_slg(
        do_cfg, 4, positive, negative, slg_config=config, active=(True, False), sigmas=(0.5, 0.0), latents=x
    )
    sigma = 0.5
    cond_x0 = x[0] - sigma * pos[0]
    skip_x0 = x[0] - sigma * skipped[0]
    base_x0 = x[0] - sigma * base[0] if do_cfg else skip_x0
    guided_x0 = base_x0 + config.scale * (cond_x0 - skip_x0)
    if rescale:
        factor = 1 + rescale * (cond_x0.std() / guided_x0.std() - 1)
        guided_x0 = guided_x0 * factor
    expected = (x[0] - guided_x0) / sigma
    torch.testing.assert_close(actual[0], expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], base[1], atol=0, rtol=0)
    assert len(pipeline.calls) == (3 if do_cfg else 2)


def test_rescaling_is_independent_per_sample_and_handles_constant_outputs():
    torch.manual_seed(3)
    predictions = [torch.randn(2, 3, 4, 4) for _ in range(3)]
    predictions[0][1] *= 100
    latents = torch.randn_like(predictions[0])
    config = SkipLayerGuidanceConfig((1,), 3, rescaling_scale=0.5)
    kwargs = dict(do_true_cfg=True, cfg_scale=4, config=config, prediction_adapter=FlowMatchingPredictionAdapter())
    combined = combine_skip_layer_guidance(
        predictions, active=(True, True), sigmas=(0.8, 0.2), latents=latents, **kwargs
    )
    for i, sigma in enumerate((0.8, 0.2)):
        single = combine_skip_layer_guidance(
            [p[i : i + 1] for p in predictions], active=(True,), sigmas=(sigma,), latents=latents[i : i + 1], **kwargs
        )
        torch.testing.assert_close(single[0], combined[i], atol=0, rtol=0)
    constant = combine_skip_layer_guidance(
        [torch.ones(2, 1, 1, 1)] * 3, active=(True, True), sigmas=(0.8, 0.2), latents=torch.zeros(2, 1, 1, 1), **kwargs
    )
    assert torch.isfinite(constant).all()


def test_prepare_validates_capability_and_uses_step_fraction_interval():
    config = SkipLayerGuidanceConfig((1,), 3, start=0.2, stop=0.8, uncond_skip_layers=(2,))
    validated: list[tuple[int, ...]] = []
    transformer = SimpleNamespace(supports_skip_layer_guidance=True, validate_skip_layers=validated.append)
    state = prepare_skip_layer_guidance(
        config,
        transformer=transformer,
        sigmas=torch.tensor([1.0, 0.9, 0.85, 0.8, 0.0]),
    )
    assert state.active_steps == (False, True, True, True, False)
    assert validated == [(1,), (2,)]
    with pytest.raises(ValueError, match="does not support"):
        prepare_skip_layer_guidance(config, transformer=object(), sigmas=torch.ones(1))


@pytest.mark.parametrize("cfg_scale", [0.0, 1.0, 5.0])
@pytest.mark.parametrize("mode", ["cond", "uncond", "both"])
@pytest.mark.parametrize("active", [(True, True), (True, False), (False, False)])
def test_modes_match_standalone_formulas_and_forward_counts(cfg_scale, mode, active):
    pipeline = ToyPipeline()
    x, positive, negative = inputs()
    config = SkipLayerGuidanceConfig(
        skip_layers=(22,) if mode != "uncond" else (),
        scale=3,
        uncond_skip_layers=(24,) if mode != "cond" else (),
    )
    pos = pipeline.predict_noise(**positive)
    neg = pipeline.predict_noise(**negative)
    cond_skip = pipeline.predict_noise(**positive, skip_layers=(22,))
    uncond_skip = pipeline.predict_noise(**negative, skip_layers=(24,))
    do_cfg = cfg_scale != 1
    base = neg + cfg_scale * (pos - neg) if do_cfg else pos
    expected = base.clone()
    for i, enabled in enumerate(active):
        if not enabled:
            continue
        if do_cfg:
            uncond = uncond_skip[i] if config.uncond_skip_layers else neg[i]
            expected[i] = uncond + cfg_scale * (pos[i] - uncond)
        if config.cond_enabled:
            if not do_cfg:
                expected[i] = cond_skip[i]
            expected[i] += config.scale * (pos[i] - cond_skip[i])
    pipeline.calls.clear()
    actual = pipeline.predict_noise_with_slg(
        do_cfg, cfg_scale, positive, negative, slg_config=config, active=active, sigmas=(0.9, 0.1), latents=x
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_calls = 2 if do_cfg else 1
    if any(active):
        expected_calls += int(config.cond_enabled)
        expected_calls += int(do_cfg and bool(config.uncond_skip_layers) and not all(active))
    assert len(pipeline.calls) == expected_calls
    if not do_cfg:
        assert all(condition == 2 for condition, _ in pipeline.calls)
    if mode == "uncond" and do_cfg and all(active):
        assert pipeline.calls == [(2, ()), (-1, (24,))]
    assert "skip_layers" not in positive and "skip_layers" not in negative


def test_scale_one_without_cfg_matches_normal_conditional():
    pipeline = ToyPipeline()
    x, positive, negative = inputs()
    expected = pipeline.predict_noise(**positive)
    actual = pipeline.predict_noise_with_slg(
        False,
        1,
        positive,
        negative,
        slg_config=SkipLayerGuidanceConfig((22,), 1),
        active=(True, True),
        sigmas=(0.9, 0.1),
        latents=x,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_uncond_skipping_remains_enabled_when_slg_scale_is_zero():
    config = SkipLayerGuidanceConfig((22,), 0, uncond_skip_layers=(24,))
    assert config.enabled and not config.cond_enabled
    assert config.batch_key == SkipLayerGuidanceConfig(uncond_skip_layers=(24,)).batch_key


@pytest.mark.parametrize(
    "active,sigmas,rescale,message",
    [
        ((), (), 0, "one value per output"),
        ((False,), (0.9,), 0, "one value per output"),
        ((True, False), (0.9,), 0, "one value per output"),
        ((True, False), (0.0, 0.9), 0.7, "positive sigma"),
    ],
)
def test_invalid_per_sample_context_is_rejected(active, sigmas, rescale, message):
    pipeline = ToyPipeline()
    x, positive, negative = inputs()
    with pytest.raises(ValueError, match=message):
        pipeline.predict_noise_with_slg(
            True,
            4,
            positive,
            negative,
            slg_config=SkipLayerGuidanceConfig((1,), 3, rescaling_scale=rescale),
            active=active,
            sigmas=sigmas,
            latents=x,
        )
    assert not pipeline.calls


@pytest.mark.parametrize("do_cfg", [False, True])
@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("mode", ["cond", "uncond", "both"])
@pytest.mark.parametrize("active", [(True, True), (True, False)])
def test_cfg_parallel_slg_combines_every_rank(monkeypatch, do_cfg, world_size, mode, active):
    pipeline = ToyPipeline()
    x, positive, negative = inputs()
    config = SkipLayerGuidanceConfig(
        (1,) if mode != "uncond" else (),
        3,
        rescaling_scale=0.7,
        uncond_skip_layers=(2,) if mode != "cond" else (),
    )
    kwargs = dict(slg_config=config, active=active, sigmas=(0.9, 0.1), latents=x)
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: 1)
    expected = pipeline.predict_noise_with_slg(do_cfg, 4, positive, negative, **kwargs)
    calls = pipeline.calls.copy()
    predictions = [pipeline.predict_noise(x=x, condition=condition, skip_layers=skip) for condition, skip in calls]
    assignments = cfg_parallel._dispatch_branches(len(predictions), world_size)
    max_slots = max(map(len, assignments))
    slots = []
    for assignment in assignments:
        local = [predictions[i] for i in assignment] or [predictions[0]]
        slots.append(local + [torch.zeros_like(predictions[0])] * (max_slots - len(local)))

    class Group:
        def __init__(self, rank):
            self.rank, self.slot = rank, 0

        def all_gather(self, prediction, separate_tensors):
            assert separate_tensors
            torch.testing.assert_close(prediction, slots[self.rank][self.slot], atol=0, rtol=0)
            result = [local[self.slot] for local in slots]
            self.slot += 1
            return result

    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: world_size)
    for rank in range(world_size):
        group = Group(rank)
        monkeypatch.setattr(cfg_parallel, "get_cfg_group", lambda: group)
        monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda: rank)
        actual = pipeline.predict_noise_with_slg(do_cfg, 4, positive, negative, **kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert group.slot == (max_slots if do_cfg or config.cond_enabled else 0)


def _distributed_slg_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))

    class Group:
        def all_gather(self, prediction, separate_tensors):
            assert separate_tensors
            result = [torch.empty_like(prediction) for _ in range(2)]
            dist.all_gather(result, prediction)
            return result

    cfg_parallel._get_cfg_world_size_or_one = lambda: 2
    cfg_parallel.get_classifier_free_guidance_rank = lambda: rank
    cfg_parallel.get_cfg_group = Group
    try:
        pipeline = ToyPipeline()
        x, positive, negative = inputs()
        for mode in ("cond", "uncond", "both"):
            config = SkipLayerGuidanceConfig(
                (1,) if mode != "uncond" else (),
                3,
                rescaling_scale=0.7,
                uncond_skip_layers=(2,) if mode != "cond" else (),
            )
            for do_cfg in (False, True):
                for active in ((True, True), (True, False), (False, False)):
                    actual = pipeline.predict_noise_with_slg(
                        do_cfg, 4, positive, negative, slg_config=config, active=active, sigmas=(0.9, 0.1), latents=x
                    )
                    cfg_parallel._get_cfg_world_size_or_one = lambda: 1
                    expected = pipeline.predict_noise_with_slg(
                        do_cfg, 4, positive, negative, slg_config=config, active=active, sigmas=(0.9, 0.1), latents=x
                    )
                    cfg_parallel._get_cfg_world_size_or_one = lambda: 2
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_slg_with_real_two_process_cfg_communication(tmp_path):
    mp.spawn(_distributed_slg_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
