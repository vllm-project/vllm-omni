# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.models.ming_flash_omni import ming_zimage_transformer, pipeline_ming_imagegen
from vllm_omni.diffusion.models.ming_flash_omni.ming_zimage_transformer import (
    MingZImageTransformer2DModel,
)
from vllm_omni.diffusion.models.z_image import pipeline_z_image
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import thinker2imagegen

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, hidden):
        self.calls.append(hidden.detach().clone())
        return hidden + 10

    @staticmethod
    def zero_negative(value):
        return torch.zeros_like(value)


class _StepConditionEncoder(_ConditionEncoder):
    pass


def _pipeline(monkeypatch):
    pipe = object.__new__(pipeline_ming_imagegen.MingImagePipeline)
    nn.Module.__init__(pipe)
    pipe.register_parameter("_probe", nn.Parameter(torch.zeros(1)))
    pipe.image_gen_config = SimpleNamespace(
        img_gen_scales=[2],
        thinker_hidden_size=3,
        default_height=16,
        default_width=16,
        num_inference_steps=2,
        guidance_scale=2.0,
    )
    pipe.condition_encoder = _ConditionEncoder()
    pipe.byte5 = None
    pipe._dtype = torch.float32
    pipe.device = torch.device("cpu")

    captured = {}

    def fake_forward(_self, z_req):
        captured["request_ids"] = z_req.request_ids
        captured["sampling"] = z_req.sampling_params_list
        captured["positive"] = [x.detach().clone() for x in pipe._pending_prompt_embeds]
        captured["negative"] = [x.detach().clone() for x in pipe._pending_negative_prompt_embeds]
        captured["generator_values"] = [
            torch.rand(1, generator=item.generator).item() for item in z_req.sampling_params_list
        ]
        return DiffusionOutput(
            output=torch.arange(z_req.num_reqs * 4, dtype=torch.float32).reshape(z_req.num_reqs, 1, 2, 2)
        )

    monkeypatch.setattr(ZImagePipeline, "forward", fake_forward)
    return pipe, captured


def _request(request_id, hidden, *, seed, negative=None, reference=None):
    extra = {"thinker_hidden_states": hidden}
    if negative is not None:
        extra["negative_thinker_hidden_states"] = negative
    if reference is not None:
        extra["reference_image"] = reference
    return OmniDiffusionRequest(
        prompt={"prompt": "", "extra": extra},
        sampling_params=OmniDiffusionSamplingParams(seed=seed, height=16, width=16, num_inference_steps=2),
        request_id=request_id,
    )


def test_ming_pipeline_batches_request_local_conditions_and_preserves_order(monkeypatch):
    pipe, captured = _pipeline(monkeypatch)
    assert pipe.supports_request_batch is True
    req_a = _request("A", torch.full((2, 3), 1.0), seed=111, negative=torch.full((2, 3), 7.0))
    req_b = _request("B", torch.full((2, 3), 2.0), seed=222)

    outputs = pipe.forward(DiffusionRequestBatch([req_a, req_b]))

    assert [item.output.flatten()[0].item() for item in outputs] == [0.0, 4.0]
    assert captured["request_ids"] == ["A", "B"]
    assert captured["positive"][0][0, 0].item() == 11.0
    assert captured["positive"][1][0, 0].item() == 12.0
    assert captured["negative"][0][0, 0].item() == 17.0
    assert torch.count_nonzero(captured["negative"][1]) == 0
    assert [g.initial_seed() for g in (s.generator for s in captured["sampling"])] == [111, 222]


def test_ming_seed_isolation_matches_single_request_execution(monkeypatch):
    pipe, batch_capture = _pipeline(monkeypatch)
    req_a = _request("A", torch.ones((2, 3)), seed=111)
    req_b = _request("B", torch.ones((2, 3)), seed=222)
    batch = DiffusionRequestBatch([req_a, req_b])
    pipe.forward(batch)
    _pipeline_single, single_capture = _pipeline(monkeypatch)
    _pipeline_single.forward(DiffusionRequestBatch([_request("A", torch.ones((2, 3)), seed=111)]))
    assert batch_capture["generator_values"][0] == single_capture["generator_values"][0]
    assert batch_capture["generator_values"][0] != batch_capture["generator_values"][1]


def test_ming_preserves_explicit_request_generators(monkeypatch):
    pipe, capture = _pipeline(monkeypatch)
    generator = torch.Generator().manual_seed(987)
    request = OmniDiffusionRequest(
        prompt={"prompt": "", "extra": {"thinker_hidden_states": torch.ones((2, 3))}},
        sampling_params=OmniDiffusionSamplingParams(
            generator=generator,
            seed=123,
            height=16,
            width=16,
            num_inference_steps=2,
        ),
        request_id="generator-request",
    )

    pipe.forward(DiffusionRequestBatch([request]))

    assert capture["sampling"][0].generator is generator


def test_ming_step_preserves_explicit_generator_over_sampling_seed(monkeypatch):
    pipe = _step_pipeline(monkeypatch)
    generator = torch.Generator().manual_seed(987)
    state = StepRequestState(
        request_id="generator-request",
        sampling=OmniDiffusionSamplingParams(
            generator=generator,
            seed=123,
            height=4,
            width=4,
            num_inference_steps=2,
        ),
        prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3)}},
    )

    pipe.prepare_encode(state)

    assert state.sampling.generator is generator


def test_ming_reference_latents_are_indexed_per_request(monkeypatch):
    captured = {}
    context = SimpleNamespace(ref_latent=torch.tensor([[[[1.0]]], [[[2.0]]]]))
    monkeypatch.setattr(ming_zimage_transformer, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(ming_zimage_transformer, "get_forward_context", lambda: context)

    def fake_parent(_self, x, t, cap_feats, patch_size=2, f_patch_size=1):
        captured["x"] = x
        return x, {}

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", fake_parent)
    transformer = object.__new__(MingZImageTransformer2DModel)
    x = [torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)]
    transformer.forward(x, torch.ones(2), [torch.zeros(1, 1), torch.zeros(1, 1)])

    assert captured["x"][0][0, 1, 0, 0].item() == 1.0
    assert captured["x"][1][0, 1, 0, 0].item() == 2.0


def test_ming_preprocessor_marks_wave_compatibility():
    pre = pipeline_ming_imagegen.get_ming_image_pre_process_func(SimpleNamespace())
    req = _request("A", torch.ones((2, 3)), seed=111)
    processed = pre(req)
    assert processed.batch_compatibility_key[0] == "ming_image"


class _StepScheduler:
    order = 1

    def __init__(self):
        self.config = {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
        self.timesteps = None

    def set_timesteps(self, steps, device=None, **kwargs):
        del kwargs
        self.timesteps = torch.arange(steps, 0, -1, device=device, dtype=torch.float32)

    def set_begin_index(self, index):
        self.begin_index = index

    def scale_noise(self, latents, timestep, noise):
        del timestep
        return latents + noise

    def step(self, noise_pred, timestep, latents, **kwargs):
        del timestep, kwargs
        return (latents - noise_pred,)


def _step_pipeline(monkeypatch):
    pipe = object.__new__(pipeline_ming_imagegen.MingImagePipeline)
    nn.Module.__init__(pipe)
    pipe.register_parameter("_probe", nn.Parameter(torch.zeros(1)))
    pipe.device = torch.device("cpu")
    pipe._execution_device = pipe.device
    pipe._dtype = torch.float32
    pipe.vae_scale_factor = 1
    pipe.image_gen_config = SimpleNamespace(
        img_gen_scales=[2],
        thinker_hidden_size=3,
        default_height=4,
        default_width=4,
        num_inference_steps=3,
        guidance_scale=2.0,
    )
    pipe.transformer = SimpleNamespace(in_channels=1)
    pipe.scheduler = _StepScheduler()
    pipe.vae = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0),
        decode=lambda latents, return_dict=False: (latents,),
    )
    pipe.condition_encoder = _StepConditionEncoder()
    pipe.byte5 = None
    monkeypatch.setattr(
        pipe,
        "_encode_reference_image",
        lambda ref, height, width: torch.full((1, 1, 2, 2), float(torch.as_tensor(ref).flatten()[0])),
    )

    pipe.seen_step_conditions = []

    def fake_predict(**kwargs):
        pipe.seen_step_conditions.append(
            {
                "request_ids": [item for item in kwargs["positive_kwargs"]["cap_feats"]],
                "negative": None
                if kwargs["negative_kwargs"] is None
                else [item for item in kwargs["negative_kwargs"]["cap_feats"]],
                "x_shapes": [tuple(item.shape) for item in kwargs["positive_kwargs"]["x"]],
            }
        )
        return torch.zeros((len(kwargs["positive_kwargs"]["x"]), 1, 1, 4, 4))

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", fake_predict)
    return pipe


def test_ming_step_lifecycle_runs_one_step_per_tick(monkeypatch):
    pipe = _step_pipeline(monkeypatch)
    states = [
        StepRequestState(
            request_id="A",
            sampling=OmniDiffusionSamplingParams(seed=111, height=4, width=4, num_inference_steps=3),
            prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3)}},
        ),
        StepRequestState(
            request_id="B",
            sampling=OmniDiffusionSamplingParams(seed=222, height=4, width=4, num_inference_steps=3),
            prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3) * 2}},
        ),
    ]
    for state in states:
        pipe.prepare_encode(state)
    assert [state.step_index for state in states] == [0, 0]
    batch = InputBatch.make_batch(states)
    for _ in range(3):
        prediction = pipe.denoise_step(batch, states=states)
        assert pipe.seen_step_conditions[-1]["x_shapes"] == [(1, 1, 4, 4), (1, 1, 4, 4)]
        assert prediction.shape[0] == 2
        for index, state in enumerate(states):
            pipe.step_scheduler(state, prediction[index : index + 1])
        if not all(state.denoise_completed for state in states):
            batch = InputBatch.make_batch(states, cached_batch=batch)
    assert [state.step_index for state in states] == [3, 3]
    assert all(state.scheduler is not states[0].scheduler for state in states[1:])
    states[0].extra["ming_output_type"] = "latent"
    assert pipe.post_decode(states[0]).output is states[0].latents


def test_ming_step_batch_uses_each_request_timestep(monkeypatch):
    pipe = _step_pipeline(monkeypatch)
    states = []
    for request_id, step_index in (("A", 3), ("B", 0), ("C", 2)):
        state = StepRequestState(
            request_id=request_id,
            sampling=OmniDiffusionSamplingParams(height=4, width=4, num_inference_steps=4),
            prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3)}},
        )
        pipe.prepare_encode(state)
        state.step_index = step_index
        states.append(state)
    batch = InputBatch.make_batch(states)
    seen = {}

    def capture_predict(**kwargs):
        seen["timesteps"] = kwargs["positive_kwargs"]["t"].clone()
        return torch.zeros((3, 1, 1, 4, 4))

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", capture_predict)
    pipe.denoise_step(batch, states=states)
    torch.testing.assert_close(seen["timesteps"], torch.tensor([0.999, 0.996, 0.998]), atol=1e-6, rtol=0)


def test_ming_step_preprocessor_isolates_reference_requests():
    pre = pipeline_ming_imagegen.get_ming_image_pre_process_func(SimpleNamespace())
    ref_a = _request("A", torch.ones((2, 3)), seed=111, reference=torch.zeros(1))
    ref_b = _request("B", torch.ones((2, 3)), seed=222, reference=torch.ones(1))
    assert pre(ref_a).batch_compatibility_key != pre(ref_b).batch_compatibility_key


def test_ming_step_preprocessor_isolates_cfg_truncation():
    pre = pipeline_ming_imagegen.get_ming_image_pre_process_func(SimpleNamespace())
    req_a = _request("A", torch.ones((2, 3)), seed=111)
    req_b = _request("B", torch.ones((2, 3)), seed=222)
    req_a.sampling_params.extra_args = {"cfg_truncation": 0.5}
    req_b.sampling_params.extra_args = {"cfg_truncation": 1.0}
    assert pre(req_a).batch_compatibility_key != pre(req_b).batch_compatibility_key


def test_ming_step_consumes_cfg_truncation_per_timestep(monkeypatch):
    pipe = _step_pipeline(monkeypatch)
    states = []
    for request_id in ("A", "B"):
        state = StepRequestState(
            request_id=request_id,
            sampling=OmniDiffusionSamplingParams(height=4, width=4, num_inference_steps=3),
            prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3)}},
        )
        pipe.prepare_encode(state)
        state.extra["ming_cfg_truncation"] = 0.9985
        states.append(state)
    states[1].step_index = 2

    calls = []
    original = pipe.predict_noise_maybe_with_cfg

    def capture(**kwargs):
        calls.append((kwargs["do_true_cfg"], len(kwargs["positive_kwargs"]["x"])))
        return original(**kwargs)

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", capture)
    pipe.denoise_step(InputBatch.make_batch(states), states=states)

    assert calls == [(False, 1), (True, 1)]


def test_zimage_diffuse_marks_each_cfg_transformer_forward(monkeypatch):
    pipe = object.__new__(ZImagePipeline)
    pipe._uses_cudagraph_trees = True
    pipe._interrupt = False
    pipe.od_config = SimpleNamespace(dtype=torch.float32)
    pipe.transformer = lambda *args, **kwargs: ([torch.zeros(1, 1, 1, 2, 2)], {})
    pipe.scheduler_step_maybe_with_cfg = lambda noise, timestep, latents, apply_cfg: latents
    marker_calls = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marker_calls.append(True))

    pipe.diffuse(
        [torch.zeros(1, 1)],
        [torch.zeros(1, 1)],
        torch.zeros(1, 1, 2, 2),
        torch.tensor([900.0, 500.0]),
        do_true_cfg=True,
        true_cfg_scale=2.0,
    )

    assert marker_calls == [True, True, True, True]


def test_zimage_diffuse_preserves_five_dimensional_latents(monkeypatch):
    pipe = object.__new__(ZImagePipeline)
    pipe._uses_cudagraph_trees = False
    pipe._interrupt = False
    pipe.od_config = SimpleNamespace(dtype=torch.float32)
    captured: dict[str, Any] = {}

    def capture_predict(**kwargs):
        captured["x"] = kwargs["positive_kwargs"]["x"]
        return torch.zeros((1, 2, 3, 4, 5))

    pipe.predict_noise_maybe_with_cfg = capture_predict
    pipe.scheduler_step_maybe_with_cfg = lambda noise, timestep, latents, apply_cfg: latents

    latents = torch.zeros(1, 2, 3, 4, 5)
    result = pipe.diffuse(
        [torch.zeros(1, 1)],
        [torch.zeros(1, 1)],
        latents,
        torch.tensor([900.0]),
        do_true_cfg=False,
        true_cfg_scale=0.0,
    )

    assert captured["x"][0].shape == (2, 3, 4, 5)
    assert result.shape == latents.shape


def test_zimage_forward_accepts_multiple_requests_without_single_batch_assert(monkeypatch):
    pipe = object.__new__(ZImagePipeline)
    pipe._execution_device = torch.device("cpu")
    pipe.vae_scale_factor = 1
    pipe.transformer = SimpleNamespace(in_channels=1)
    pipe.scheduler = SimpleNamespace(config={}, sigma_min=0.0)
    captured: dict[str, Any] = {}

    pipe.encode_prompt = lambda **kwargs: ([torch.zeros(1, 1)] * 2, [torch.zeros(1, 1)] * 2)

    def fake_prepare_latents(batch_size, *args, **kwargs):
        return torch.zeros((batch_size, 1, 3, 8, 10))

    pipe.prepare_latents = fake_prepare_latents

    def fake_retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
        captured["mu"] = kwargs["mu"]
        return torch.tensor([1.0], device=device), 1

    monkeypatch.setattr(pipeline_z_image, "retrieve_timesteps", fake_retrieve_timesteps)

    def fake_diffuse(**kwargs):
        captured["latents"] = kwargs["latents"]
        return kwargs["latents"]

    pipe.diffuse = fake_diffuse
    params = OmniDiffusionSamplingParams(
        height=16,
        width=16,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    )
    requests = [
        OmniDiffusionRequest(prompt={"prompt": ""}, sampling_params=params, request_id=request_id)
        for request_id in ("A", "B")
    ]

    output = pipe.forward(DiffusionRequestBatch(requests))

    assert output.output.shape == (2, 1, 3, 8, 10)
    assert captured["latents"].shape == (2, 1, 3, 8, 10)
    assert captured["mu"] == pipeline_z_image.calculate_shift(20, 256, 4096, 0.5, 1.15)


def test_ming_step_denoise_scopes_reference_latents_in_active_request_order(monkeypatch):
    pipe = _step_pipeline(monkeypatch)
    states = [
        StepRequestState(
            request_id=request_id,
            sampling=OmniDiffusionSamplingParams(height=4, width=4, num_inference_steps=3),
            prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3), "reference_image": torch.full((1,), value)}},
        )
        for request_id, value in (("A", 1.0), ("B", 2.0))
    ]
    for state in states:
        pipe.prepare_encode(state)

    captured: list[torch.Tensor | None] = []
    monkeypatch.setattr(
        pipeline_ming_imagegen,
        "set_forward_context_ref_latent",
        lambda value: captured.append(None if value is None else value.clone()),
    )
    pipe.denoise_step(InputBatch.make_batch(states), states=states)

    assert captured[0] is not None
    torch.testing.assert_close(captured[0][:, 0, 0, 0], torch.tensor([1.0, 2.0]))
    assert captured[-1] is None


def test_ming_step_batch_runs_through_real_scheduler_and_runner(monkeypatch):
    """Exercise the closest CPU-only production path without loading a checkpoint.

    The scheduler, runner, request states, InputBatch construction, Ming hooks,
    completion bookkeeping, and output routing are real.  Only the heavyweight
    DIT/VAE kernels are replaced by the deterministic test pipeline above.
    """
    pipe = _step_pipeline(monkeypatch)
    pipe.supports_step_execution = True

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(
            ir_op_priority=SimpleNamespace(set_priority=lambda *args, **kwargs: nullcontext())
        ),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
    )
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=SimpleNamespace(use_hsdp=False),
        streaming_output=False,
    )
    runner.device = torch.device("cpu")
    runner.pipeline = pipe
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda *args, **kwargs: None,
    )

    # The CI CPU platform advertises availability but has no CUDA memory API.
    # Disable only that optional metric so the real runner path can execute.
    platform_path = "vllm_omni.diffusion.worker.diffusion_model_runner.current_omni_platform"
    monkeypatch.setattr(f"{platform_path}.is_available", lambda: False)
    monkeypatch.setattr(f"{platform_path}.max_memory_reserved", lambda: 0)
    monkeypatch.setattr(f"{platform_path}.max_memory_allocated", lambda: 0)

    pre = pipeline_ming_imagegen.get_ming_image_pre_process_func(SimpleNamespace())
    requests = []
    for request_id, hidden, seed in (
        ("A", torch.ones(257, 3), 111),
        ("B", torch.ones(257, 3) * 2, 222),
    ):
        # Feed a production-shaped thinker output through the real stage input
        # processor; this is the production producer of prompt.extra hidden state.
        thinker_output = SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=[157157] * 256 + [157159],
            outputs=[SimpleNamespace(multimodal_output={"final_hidden_states": hidden})],
        )
        thinker_outputs = [thinker_output]
        if request_id == "A":
            thinker_outputs.append(
                SimpleNamespace(
                    request_id="A__cfg_text",
                    prompt_token_ids=[157157] * 256 + [157159],
                    outputs=[SimpleNamespace(multimodal_output={"final_hidden_states": torch.full((257, 3), 7.0)})],
                )
            )
        imagegen_prompt = thinker2imagegen(thinker_outputs, prompt={"prompt": ""})[0]
        request = OmniDiffusionRequest(
            prompt=imagegen_prompt,
            sampling_params=OmniDiffusionSamplingParams(
                seed=seed,
                height=4,
                width=4,
                num_inference_steps=2,
            ),
            request_id=request_id,
        )
        requests.append(pre(request))
    scheduler = StepScheduler()
    scheduler.initialize(
        SimpleNamespace(
            max_num_seqs=2,
            omni_kv_config=None,
            diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        )
    )
    scheduler.add_request(requests[0])

    first = scheduler.schedule()
    assert first.scheduled_request_ids == ["A"]
    first_output = runner.execute_stepwise(first)
    assert first_output.request_ids == ["A"]
    assert all(not item.finished for item in first_output.runner_outputs)
    torch.testing.assert_close(pipe.seen_step_conditions[0]["request_ids"][0], torch.full((256, 3), 11.0))
    torch.testing.assert_close(pipe.seen_step_conditions[0]["negative"][0], torch.full((256, 3), 17.0))
    scheduler.update_from_output(first, first_output)

    # Admit B while A is already running; the next real wave must contain
    # newly admitted B plus cached A.  BaseScheduler emits new rows first;
    # request ids, rather than list position, are the routing contract.
    scheduler.add_request(requests[1])
    second = scheduler.schedule()
    assert second.scheduled_request_ids == ["B", "A"]
    assert second.scheduled_cached_reqs.request_ids == ["A"]
    second_output = runner.execute_stepwise(second)
    assert second_output.request_ids == ["B", "A"]
    torch.testing.assert_close(pipe.seen_step_conditions[1]["request_ids"][0], torch.full((256, 3), 12.0))
    torch.testing.assert_close(pipe.seen_step_conditions[1]["request_ids"][1], torch.full((256, 3), 11.0))
    torch.testing.assert_close(pipe.seen_step_conditions[1]["negative"][0], torch.zeros((256, 3)))
    torch.testing.assert_close(pipe.seen_step_conditions[1]["negative"][1], torch.full((256, 3), 17.0))
    assert second_output["A"].finished is True
    assert second_output["B"].finished is False
    scheduler.update_from_output(second, second_output)

    third = scheduler.schedule()
    assert third.scheduled_request_ids == ["B"]
    third_output = runner.execute_stepwise(third)
    assert third_output.request_ids == ["B"]
    assert third_output["B"].finished is True
    scheduler.update_from_output(third, third_output)

    assert not scheduler.has_requests()
    assert runner.state_cache == {}
    assert third_output["B"].result.output.shape == (1, 1, 4, 4)


def test_ming_step_scheduler_matches_deterministic_reference_recurrence(monkeypatch):
    """The scheduler hook applies the same deterministic latent recurrence each tick."""
    pipe = _step_pipeline(monkeypatch)
    state = StepRequestState(
        request_id="A",
        sampling=OmniDiffusionSamplingParams(seed=111, height=4, width=4, num_inference_steps=3),
        prompt={"extra": {"thinker_hidden_states": torch.ones(2, 3)}},
    )
    pipe.prepare_encode(state)
    initial = state.latents.clone()
    for _ in range(3):
        batch = InputBatch.make_batch([state])
        prediction = pipe.denoise_step(batch, states=[state])
        pipe.step_scheduler(state, prediction)

    expected = initial.clone()
    scheduler = _StepScheduler()
    scheduler.set_timesteps(3, device=torch.device("cpu"))
    assert scheduler.timesteps is not None
    for timestep in scheduler.timesteps:
        expected = scheduler.step(torch.zeros_like(expected), timestep, expected, return_dict=False)[0]

    torch.testing.assert_close(state.latents, expected)
