# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.guidance import SkipLayerGuidanceConfig, prepare_skip_layer_guidance
from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageDenoiseBody, MOSSImageTransformer
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    get_mossimage_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


class Transformer(nn.Module):
    latent_channels = 2
    supports_skip_layer_guidance = True

    def __init__(self):
        super().__init__()
        self.calls = []

    def validate_skip_layers(self, skip_layers):
        if any(index >= 42 for index in skip_layers):
            raise ValueError("MOSSImage SLG layer IDs must be in [0, 41].")

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        *,
        return_dict=False,
        _conditioning_cache=None,
        skip_layers=(),
    ):
        self.calls.append(skip_layers)
        condition = encoder_hidden_states.mean((1, 2)).reshape(-1, 1, 1, 1)
        time = torch.as_tensor(timestep).reshape(-1, 1, 1, 1) / 1000
        prediction = hidden_states * (0.2 if skip_layers else 0.5) + condition + time
        return (prediction,)


def pipeline():
    result = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(result)
    result.device = torch.device("cpu")
    result.transformer = Transformer()
    result.scheduler = MOSSImageScheduler(shift=5)
    result._interrupt = False

    def encode(prompts, max_sequence_length):
        embeds = torch.tensor([2.0 if prompt else -1.0 for prompt in prompts]).reshape(-1, 1, 1)
        return embeds, torch.ones(len(prompts), 1, dtype=torch.bool)

    result.encode_prompt = encode
    result._vae_decode = lambda latents: latents
    return result


def sampling(*, cfg=4.0, rescale=0.0, scale=3.0, outputs=1, seed=11, interval=(0.0, 1.0), mode="cond"):
    return OmniDiffusionSamplingParams(
        height=32,
        width=32,
        seed=seed,
        num_inference_steps=8,
        guidance_scale=cfg,
        num_outputs_per_prompt=outputs,
        extra_args={
            "slg_layers": [1] if mode != "uncond" else [],
            "uncond_skip_layers": [2] if mode != "cond" else [],
            "slg_scale": scale,
            "slg_rescaling_scale": rescale,
            "slg_start": interval[0],
            "slg_stop": interval[1],
            "flow_shift": 5.0,
        },
    )


def request(params, request_id="req"):
    return OmniDiffusionRequest(prompt="positive", sampling_params=params, request_id=request_id)


@pytest.mark.parametrize("cfg", [1.0, 4.0])
@pytest.mark.parametrize("rescale", [0.0, 0.7])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("mode", ["cond", "uncond", "both"])
@torch.no_grad()
def test_full_forward_and_step_execution_match(cfg, rescale, outputs, mode):
    full, step = pipeline(), pipeline()
    params = sampling(cfg=cfg, rescale=rescale, outputs=outputs, mode=mode)
    expected = full(DiffusionRequestBatch([request(copy.deepcopy(params))])).output
    state = StepRequestState(request_id="req", sampling=copy.deepcopy(params), prompt="positive")
    step.prepare_encode(state)
    batch = None
    while state.step_index < len(state.timesteps):
        batch = InputBatch.make_batch([state], cached_batch=batch)
        prediction = step.denoise_step(batch, states=[state])
        step.step_scheduler(state, prediction)
    actual = step.post_decode(state).output
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert full.transformer.calls == step.transformer.calls
    assert full.transformer.calls.count((1,)) == (len(state.timesteps) if mode != "uncond" else 0)
    assert full.transformer.calls.count((2,)) == (len(state.timesteps) if mode != "cond" and cfg != 1 else 0)


@torch.no_grad()
def test_disabled_slg_matches_unconfigured_pipeline_without_extra_forwards():
    disabled, unconfigured = pipeline(), pipeline()
    params = sampling(scale=0)
    baseline = copy.deepcopy(params)
    baseline.extra_args = {"flow_shift": 5.0}
    actual = disabled(DiffusionRequestBatch([request(params)])).output
    expected = unconfigured(DiffusionRequestBatch([request(baseline)])).output
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert disabled.transformer.calls == unconfigured.transformer.calls == [()] * 16


@pytest.mark.parametrize("cfg", [1.0, 4.0])
@torch.no_grad()
def test_request_batch_rescaling_is_independent(cfg):
    params = [sampling(cfg=cfg, rescale=0.7, outputs=2, seed=seed) for seed in (11, 30)]
    batched = pipeline()(DiffusionRequestBatch([request(p, str(i)) for i, p in enumerate(params)]))
    for i, param in enumerate(params):
        single = pipeline()(DiffusionRequestBatch([request(copy.deepcopy(param))]))
        torch.testing.assert_close(batched[i].output, single.output, atol=0, rtol=0)


@pytest.mark.parametrize("cfg", [1.0, 4.0])
@pytest.mark.parametrize("mode", ["cond", "uncond", "both"])
@torch.no_grad()
def test_mixed_step_activity_matches_isolated_requests_and_selected_order(cfg, mode):
    model = pipeline()
    states = []
    for i, step_index in enumerate((0, 2, 49)):
        params = sampling(cfg=cfg, rescale=0.7, interval=(0.01, 0.15), seed=i, mode=mode)
        params.num_inference_steps = 50
        state = StepRequestState(request_id=str(i), sampling=params, prompt="positive")
        model.prepare_encode(state)
        state.step_index = step_index
        states.append(state)
    expected = [model.denoise_step(InputBatch.make_batch([state])) for state in states]
    model.transformer.calls.clear()
    # Runner states may differ from the selected InputBatch order.
    batch = InputBatch.make_batch(states, idx_mapping=torch.tensor([2, 1, 0]))
    actual = model.denoise_step(batch, states=states)
    torch.testing.assert_close(actual, torch.cat([expected[2], expected[1], expected[0]]), atol=0, rtol=0)
    assert model.transformer.calls.count((1,)) == (1 if mode != "uncond" else 0)
    assert model.transformer.calls.count((2,)) == (1 if mode != "cond" and cfg != 1 else 0)
    assert len(model.transformer.calls) == (2 if cfg != 1 else 1) + int(mode != "uncond") + int(
        mode != "cond" and cfg != 1
    )


def test_preprocessor_normalizes_slg_and_separates_incompatible_requests():
    preprocess = get_mossimage_pre_process_func(None)
    params = sampling()
    first = preprocess(request(copy.deepcopy(params), "first"))
    same = copy.deepcopy(params)
    same.extra_args["slg_skip_layers"] = same.extra_args.pop("slg_layers")
    same.extra_args["slg_skip_layers"] = "1, 1"
    same.extra_args["slg_start_percent"] = same.extra_args.pop("slg_start")
    same.extra_args["slg_end_percent"] = same.extra_args.pop("slg_stop")
    second = preprocess(request(same, "second"))
    assert build_request_batch_sampling_params_key(first) == build_request_batch_sampling_params_key(second)
    different = copy.deepcopy(params)
    different.extra_args["slg_scale"] = 1
    third = preprocess(request(different, "third"))
    assert build_request_batch_sampling_params_key(first) != build_request_batch_sampling_params_key(third)
    alias = copy.deepcopy(params)
    alias.extra_args.pop("flow_shift")
    alias.extra_args["timestep_shift"] = 5
    assert preprocess(request(alias)).batch_compatibility_key == first.batch_compatibility_key
    for key, value in (("uncond_skip_layers", [2]), ("slg_start", 0.01), ("slg_stop", 0.15)):
        different = copy.deepcopy(params)
        different.extra_args[key] = value
        assert preprocess(request(different)).batch_compatibility_key != first.batch_compatibility_key


@pytest.mark.parametrize("stepwise", [False, True])
def test_incompatible_slg_configs_are_rejected_even_without_preprocessing(stepwise):
    model = pipeline()
    params = [sampling(scale=3), sampling(scale=2)]
    with pytest.raises(ValueError, match="identical SLG"):
        if stepwise:
            states = [StepRequestState(request_id=str(i), sampling=p, prompt="positive") for i, p in enumerate(params)]
            for state in states:
                model.prepare_encode(state)
            model.denoise_step(InputBatch.make_batch(states))
        else:
            model(DiffusionRequestBatch([request(p, str(i)) for i, p in enumerate(params)]))


def test_default_schedule_matches_release_shift():
    default = MOSSImageScheduler()
    reference = MOSSImageScheduler(shift=5.0)
    default.set_timesteps(50)
    reference.set_timesteps(50)
    torch.testing.assert_close(default.sigmas, reference.sigmas, rtol=0, atol=0)
    torch.testing.assert_close(default.timesteps, reference.timesteps, rtol=0, atol=0)


@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0])
def test_step_interval_uses_actual_schedule_length_independent_of_shift(shift):
    scheduler = MOSSImageScheduler(shift=3)
    scheduler.set_timesteps(50, shift=shift)
    config = SkipLayerGuidanceConfig((1,), 3, start=0.01, stop=0.15)
    state = prepare_skip_layer_guidance(config, transformer=Transformer(), sigmas=scheduler.sigmas[:-1])
    assert state.active_steps == tuple(0.01 <= index / 50 < 0.15 for index in range(50))
    assert [i for i, active in enumerate(state.active_steps) if active] == list(range(1, 8))


def test_step_interval_uses_snapped_schedule_after_duplicate_removal():
    scheduler = MOSSImageScheduler(num_train_timesteps=10, shift=5)
    scheduler.set_timesteps(50)
    assert len(scheduler.timesteps) < 50
    config = SkipLayerGuidanceConfig((22,), 3, start=0.2, stop=0.6)
    state = prepare_skip_layer_guidance(config, transformer=Transformer(), sigmas=scheduler.sigmas[:-1])
    count = len(scheduler.timesteps)
    assert state.active_steps == tuple(0.2 <= index / count < 0.6 for index in range(count))


@pytest.mark.parametrize("mode", ["cond", "uncond"])
def test_out_of_range_layers_are_rejected_even_with_zero_slg_scale(mode):
    params = sampling(scale=0, mode=mode)
    params.extra_args["slg_layers" if mode == "cond" else "uncond_skip_layers"] = [42]
    with pytest.raises(ValueError, match="layer IDs"):
        pipeline()(DiffusionRequestBatch([request(params)]))


def test_skip_replaces_complete_blocks_with_identity():
    calls = []

    class Block(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x, rope, modulation, key_mask):
            calls.append(self.index)
            return x * 2 + self.index

    body = MOSSImageDenoiseBody(nn.ModuleList([Block(i) for i in range(3)]))
    x = torch.ones(1, 2, 3)
    result = body(x, None, None, skip_layers=(1,))
    torch.testing.assert_close(result, (x * 2) * 2 + 2)
    assert calls == [0, 2]
    calls.clear()
    assert body(x, None, None, skip_layers=(0, 1, 2)) is x
    assert not calls
    shell = MOSSImageTransformer.__new__(MOSSImageTransformer)
    nn.Module.__init__(shell)
    shell.layers = body.layers
    for layers in ((3,), (-1,), (True,), (1.5,)):
        with pytest.raises(ValueError, match="layer IDs"):
            MOSSImageTransformer.validate_skip_layers(shell, layers)


def test_skip_branch_reuses_conditioning_cache_without_changing_text_refinement(monkeypatch):
    from tests.diffusion.test_mossimage_conditioning_cache import shell
    from vllm_omni.diffusion.models.mossimage import mossimage_transformer as model

    monkeypatch.setattr(model, "get_sequence_parallel_world_size", lambda: 1)
    transformer = shell()
    transformer.validate_skip_layers = lambda indices: None
    skipped_calls = []

    def denoise(x, rope, modulation, key_mask, skip_layers=()):
        skipped_calls.append(skip_layers)
        return x if skip_layers else x + x.mean(1, keepdim=True)

    transformer.denoise_body = denoise
    cache: dict[str, object] = {}
    with torch.no_grad():
        kwargs = dict(
            hidden_states=torch.randn(1, 128, 2, 2),
            timestep=1.0,
            encoder_hidden_states=torch.randn(1, 3, 128),
            _conditioning_cache=cache,
        )
        regular = MOSSImageTransformer.forward(transformer, **kwargs).sample
        skipped = MOSSImageTransformer.forward(transformer, **kwargs, skip_layers=(1,)).sample
    assert skipped_calls == [(), (1,)]
    assert transformer.text_embedder.calls == 1
    assert not torch.equal(regular, skipped)
