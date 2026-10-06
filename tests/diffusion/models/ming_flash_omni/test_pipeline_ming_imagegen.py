# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent Ming contracts: bridge -> scheduler -> runner -> public output.

Inputs come from actual stage/request types and production batch builders.
Expected results come from the Euler ODE, Z-Image CFG and runner/API contracts.
Small real HF Qwen2, VAE and FlowMatch components need no model downloads.
The analytic DiT isolates numerical wiring; real DiT coverage is in
test_ming_imagegen_model_contract.py and must also pass before acceptance.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import nn
from transformers import ByT5Tokenizer, Qwen2Config, Qwen2Model, T5Config, T5EncoderModel
from vllm.config import get_current_vllm_config
from vllm.outputs import CompletionOutput

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_flash_omni import pipeline_ming_imagegen
from vllm_omni.diffusion.models.ming_flash_omni.byte5_encoder import MingByT5Encoder
from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import MingConditionEncoder
from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen import (
    MingImagePipeline,
    get_ming_image_post_process_func,
    get_ming_image_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import thinker2imagegen
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class AnalyticDiT(nn.Module):
    """Controlled model boundary, not a substitute for real DiT shape tests.

    Reference equation: f(x,c,t,r) = x/8 + mean(c)/1000 + t/4 + r/2.
    Nonzero input-dependent predictions constrain sign, routing and stale state.
    Production CFG, predict_noise and scheduler adapters are untouched.
    """

    in_channels = 4

    def __init__(self):
        super().__init__()
        self.calls = []
        self.grad_modes = []

    def forward(self, x, t, cap_feats):
        self.grad_modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        ref = get_forward_context().ref_latent
        self.calls.append((t.clone(), [a.clone() for a in x], ref))
        output = []
        for i, (sample, condition) in enumerate(zip(x, cap_feats, strict=True)):
            assert sample.shape == (4, 1, 8, 8), "DiT frame axis must precede spatial axes"
            value = sample / 8 + condition.mean() / 1000 + t[i] / 4
            if ref is not None:
                value = value + ref[i].unsqueeze(1) / 2
            output.append(value)
        return output, {}


def stage_output(request_id, hidden):
    """Producer: actual stage result type and the image query-token layout."""
    completion = CompletionOutput(index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None)
    completion.multimodal_output = {"final_hidden_states": hidden}
    return OmniRequestOutput(
        request_id=request_id,
        prompt="",
        prompt_token_ids=[157157] * 256 + [157159],
        outputs=[completion],
        finished=True,
    )


def request(request_id="A", *, seed=11, reference=None, negative=False, **overrides):
    """Real bridge: one thinker hidden row per prompt token, then query slicing."""
    hidden = torch.arange(257 * 4, dtype=torch.float32).reshape(257, 4) / 1000
    hidden = hidden + (0.2 if request_id == "B" else 0)
    outputs = [stage_output(request_id, hidden)]
    if negative:
        outputs.append(stage_output(request_id + "__cfg_text", -hidden))
    original: dict[str, object] = {"prompt": "paint a landscape", "modalities": ["image"]}
    if reference is not None:
        original["multi_modal_data"] = {"image": reference}
    prompt = thinker2imagegen(outputs, prompt=original)[0]
    values = dict(seed=seed, height=16, width=16, num_inference_steps=3, output_type="latent")
    values.update(overrides)
    return OmniDiffusionRequest(
        prompt=prompt, request_id=request_id, sampling_params=OmniDiffusionSamplingParams(**values)
    )


@pytest.fixture
def pipeline(tmp_path):
    """Real stage components; bypass only checkpoint IO and the DiT kernel."""
    cfg = MingImageGenConfig(
        thinker_hidden_size=4,
        diffusion_c_input_dim=4,
        img_gen_scales=[16],
        default_height=16,
        default_width=16,
        num_inference_steps=3,
        guidance_scale=2.0,
        vae_subfolder="custom_vae",
    )
    root = tmp_path / "model"
    (root / "custom_vae").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"image_gen_config": cfg.to_dict()}), encoding="utf-8")
    (root / "custom_vae" / "config.json").write_text(json.dumps({"block_out_channels": [8, 16]}), encoding="utf-8")
    od_config = SimpleNamespace(
        model=str(root),
        revision=None,
        dtype=torch.float32,
        tf_model_config=cfg,
        step_execution=True,
        model_class_name="MingImagePipeline",
        streaming_output=False,
        max_num_seqs=2,
        omni_kv_config=None,
        cache_backend=None,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=SimpleNamespace(use_hsdp=False, sequence_parallel_size=1),
    )
    pipe = object.__new__(MingImagePipeline)
    nn.Module.__init__(pipe)
    pipe.od_config, pipe.image_gen_config = od_config, cfg
    pipe.device = pipe._execution_device = torch.device("cpu")
    pipe._dtype, pipe._interrupt, pipe.byte5 = torch.float32, False, None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        connector = Qwen2Model(
            Qwen2Config(
                vocab_size=16,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                attn_implementation="eager",
            )
        ).eval()
        pipe.condition_encoder = MingConditionEncoder(cfg, thinker_hidden_size=4)
        pipe.condition_encoder.connector = connector
        pipe.condition_encoder.eval()
        pipe.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8, 16),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=16,
            scaling_factor=0.5,
            shift_factor=0.25,
        ).eval()
    pipe.vae_scale_factor = 2
    pipe.image_processor = VaeImageProcessor(vae_scale_factor=4, do_convert_rgb=True)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    pipe.transformer = AnalyticDiT()
    return pipe


def prepared(pipe, req):
    req = get_ming_image_pre_process_func(pipe.od_config)(req)
    state = StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
    with torch.inference_mode():
        pipe.prepare_encode(state)
    return state


def oracle_prediction(state, *, scale=2.0, threshold=1.0, normalization=0.0):
    """Independent analytic-model equation and vendor Z-Image CFG convention."""
    t = float((1000 - state.current_timestep.float()) / 1000)
    reference = state.extra["ming_reference_latent"]
    base = state.latents / 8 + t / 4
    if reference is not None:
        base = base + reference / 2
    pos = base + state.prompt_embeds.mean() / 1000
    if scale <= 0 or (threshold is not None and t > threshold):
        return -pos
    neg = base + state.negative_prompt_embeds.mean() / 1000
    guided = pos + scale * (pos - neg)
    limit = normalization
    if limit > 0:
        max_norm, norm = torch.linalg.vector_norm(pos) * limit, torch.linalg.vector_norm(guided)
        if norm > max_norm:
            guided = guided * max_norm / norm
    return -guided


def assert_euler_tick(pipe, states, *, scale=2.0, threshold=1.0, normalization=0.0):
    # Expected source: Euler ODE, never scheduler.step() or implementation_under_test().
    predictions = [oracle_prediction(s, scale=scale, threshold=threshold, normalization=normalization) for s in states]
    expected = [
        s.latents + (s.scheduler.sigmas[s.step_index + 1] - s.scheduler.sigmas[s.step_index]) * p
        for s, p in zip(states, predictions, strict=True)
    ]
    batch = InputBatch.make_batch(states)
    with torch.inference_mode():
        prediction = pipe.denoise_step(batch)
        torch.testing.assert_close(prediction, torch.cat(predictions))
        for i, s in enumerate(states):
            old_index = s.step_index
            pipe.step_scheduler(s, prediction[i : i + 1])
            assert s.step_index == old_index + 1
            assert s.scheduler.step_index == s.step_index
            assert s.latents.dtype == torch.float32
            torch.testing.assert_close(s.latents, expected[i], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("normalize", [False, True, 2.0])
def test_nonzero_euler_cfg_and_interleaved_cursors(pipeline, normalize):
    """Scenario: B joins after A advances, CFG and non-CFG rows share a wave.

    Input source: bridge -> prepare_encode -> actual InputBatch.
    Why valid: compatible continuous-batch requests may be at different ticks.
    Expected source: Euler equation and inclusive CFG progress threshold.
    Regression: wrong sign/axis, reversed truncation, shared cursor, reordered rows.
    """
    a = prepared(pipeline, request(negative=True, cfg_normalize=normalize, extra_args={"cfg_truncation": 0.3}))
    b = prepared(pipeline, request("B", cfg_normalize=normalize, extra_args={"cfg_truncation": 0.3}))
    assert a.scheduler is not b.scheduler
    contract = {"threshold": 0.3, "normalization": float(normalize)}
    with set_forward_context(omni_diffusion_config=pipeline.od_config):
        assert_euler_tick(pipeline, [a], **contract)
        assert_euler_tick(pipeline, [b, a], **contract)
        assert_euler_tick(pipeline, [a, b], **contract)
        assert a.denoise_completed and not b.denoise_completed
        assert_euler_tick(pipeline, [b], **contract)
    assert b.denoise_completed


@pytest.mark.parametrize("reference", [False, True])
def test_b1_batch_matches_b2_and_single_request(pipeline, reference):
    """Regression: B1 multi-request sampling accessor must not assert.

    Input: same bridge requests/seeds in B1 wave, singleton and B2.
    Expected source: deterministic per-request ODE independent of admission.
    The separate Euler test pins correctness without depending on B1/B2 agreement.
    """
    image = Image.new("RGB", (16, 16), color=(100, 20, 50)) if reference else None

    def make():
        return [request("A", reference=image, negative=True), request("B", seed=22, reference=image)]

    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        batch = pipeline.forward(DiffusionRequestBatch(make()))
        singles = [pipeline.forward(DiffusionRequestBatch([r]))[0] for r in make()]
        states = [prepared(pipeline, r) for r in make()]
        for _ in range(3):
            assert_euler_tick(pipeline, states)
        for i, s in enumerate(states):
            torch.testing.assert_close(batch[i].output, singles[i].output, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(batch[i].output, pipeline.post_decode(s).output, rtol=1e-5, atol=1e-5)
            assert torch.isfinite(batch[i].output).all()


@pytest.mark.parametrize("strength", [0, 1e-8, 0.6, 1])
def test_reference_is_conditioning_not_strength_initialization(pipeline, strength, caplog):
    """Input: bridge PIL reference. Expected: vendor random-latent + extra frame.

    Regression: reference must not shorten schedule or replace seeded initial noise.
    """
    reference = Image.new("RGB", (16, 16), color="red")
    ref = prepared(pipeline, request(reference=reference, strength=strength))
    plain = prepared(pipeline, request())
    assert ref.extra["ming_reference_latent"].shape == (1, 4, 8, 8)
    assert torch.count_nonzero(ref.extra["ming_reference_latent"]) > 0
    torch.testing.assert_close(ref.latents, plain.latents)
    torch.testing.assert_close(ref.timesteps, plain.timesteps)
    assert ref.total_steps == 3 and ref.scheduler.begin_index == 0
    assert "ignores strength" in caplog.text


def test_generator_precedence_and_seed_isolation(pipeline):
    """API generator wins unless extra seed overrides; expected: independent torch RNG.

    Regression: runner-created generators must not be replaced or shared across seeds.
    """
    gen = torch.Generator().manual_seed(987)
    a = prepared(pipeline, request(generator=gen, seed=11))
    expected = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(987))
    torch.testing.assert_close(a.latents, expected)
    b = prepared(pipeline, request("B", generator=gen, extra_args={"seed": 123}))
    expected_b = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(b.latents, expected_b)
    torch.testing.assert_close(prepared(pipeline, request()).latents, prepared(pipeline, request()).latents)
    assert not torch.equal(a.latents, prepared(pipeline, request(seed=22)).latents)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("offset", [-1e-5, 0.0, 1e-5])
def test_cfg_threshold_is_inclusive_at_each_schedule_position(pipeline, index, offset):
    """Schedule-produced noise levels -> CFG at and before the declared threshold.

    Expected source: vendor progress=(1000-timestep)/1000, inclusive comparison.
    Regression: reversed time direction and accidentally exclusive threshold.
    """
    state = prepared(pipeline, request(negative=True))
    # Advancing through real scheduler steps is a legal lifecycle, not forged indices.
    with set_forward_context(omni_diffusion_config=pipeline.od_config):
        for _ in range(index):
            assert_euler_tick(pipeline, [state])
        progress = float((1000 - state.current_timestep.float()) / 1000)
        threshold = progress + offset
        state.extra["ming_cfg_truncation"] = threshold
        actual = pipeline.denoise_step(InputBatch.make_batch([state]))
        expected = oracle_prediction(state, threshold=threshold)
        torch.testing.assert_close(actual, expected)


def test_real_admission_runner_and_api_consumer(pipeline, monkeypatch):
    """Scenario: A starts, B joins, each finishes and retires separately.

    Input source: real stage output -> bridge -> preprocessor -> StepScheduler.
    Why valid: actual NewRequestData/CachedRequestData reach the actual runner.
    Expected source: runner tick/retirement and OmniRequestOutput API contracts.
    Regression: stale batch, misrouting, false completion, latent-as-PIL conversion.
    Only CPU memory metrics are disabled. The analytic DiT isolates GPU kernels.
    """
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config, runner.od_config = get_current_vllm_config(), pipeline.od_config
    runner.device, runner.pipeline = pipeline.device, pipeline
    runner.cache_backend = runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace(receive_multi_kv_cache_distributed=lambda *a, **kw: None)
    platform = "vllm_omni.diffusion.worker.diffusion_model_runner.current_omni_platform"
    monkeypatch.setattr(platform + ".is_available", lambda: False)
    monkeypatch.setattr(platform + ".max_memory_reserved", lambda: 0)
    monkeypatch.setattr(platform + ".max_memory_allocated", lambda: 0)
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    requests = {r.request_id: pre(r) for r in [request("A", negative=True), request("B", seed=22)]}
    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        baselines = {rid: pipeline.forward(DiffusionRequestBatch([req]))[0].output for rid, req in requests.items()}
    scheduler = StepScheduler()
    scheduler.initialize(pipeline.od_config)
    scheduler.add_request(requests["A"])
    outputs, waves = [], []
    for tick in range(4):
        if tick == 1:
            scheduler.add_request(requests["B"])
        scheduled = scheduler.schedule()
        waves.append(scheduled.scheduled_request_ids)
        result = runner.execute_stepwise(scheduled)
        assert result.request_ids == scheduled.scheduled_request_ids
        for row in result.runner_outputs:
            if row.result is not None:
                assert row.result.error is None
            if row.finished:
                torch.testing.assert_close(row.result.output, baselines[row.request_id], rtol=1e-5, atol=1e-5)
                outputs.append(row)
        scheduler.update_from_output(scheduled, result)
    assert waves[0] == ["A"] and waves[1] == ["B", "A"] and waves[-1] == ["B"]
    assert set(waves[2]) == {"A", "B"}
    assert [r.request_id for r in outputs] == ["A", "B"]
    assert not scheduler.has_requests() and not runner.state_cache
    assert all(not grad and inference for grad, inference in pipeline.transformer.grad_modes)
    engine = make_engine(pipeline)
    for row in outputs:
        (api,) = engine.postprocess_output(requests[row.request_id], row.result)
        assert api.request_id == row.request_id and api.finished
        assert api.images == [] and api.final_output_type == "latents"
        torch.testing.assert_close(api.latents, baselines[row.request_id])


def make_engine(pipeline):
    engine = object.__new__(DiffusionEngine)
    engine.od_config = pipeline.od_config
    engine.post_process_func = get_ming_image_post_process_func(pipeline.od_config)
    engine._post_process_accepts_sampling_params = True
    return engine


@pytest.mark.parametrize("output_type", ["pil", "pt", "np", "latent"])
def test_decode_through_engine_honors_api_output_type(pipeline, output_type):
    """Input: runner state. Expected: VAE inverse scale and API type/range.

    Regression: latent-as-PIL, unconditional PIL, wrong axes or double decode.
    """
    req = request(output_type=output_type)
    state = prepared(pipeline, req)
    with torch.inference_mode():
        raw = pipeline.post_decode(state)
        (api,) = make_engine(pipeline).postprocess_output(req, raw)
        if output_type == "latent":
            assert api.images == [] and api.latents is state.latents
        else:
            independent_decode = pipeline.vae.decode(state.latents / 0.5 + 0.25, return_dict=False)[0]
            torch.testing.assert_close(raw.output, independent_decode)
            pixels = (independent_decode / 2 + 0.5).clamp(0, 1)
            if output_type == "pil":
                assert isinstance(api.images[0], Image.Image) and api.images[0].size == (16, 16)
            elif output_type == "pt":
                assert isinstance(api.images[0], torch.Tensor) and api.images[0].shape == (1, 3, 16, 16)
                torch.testing.assert_close(api.images[0], pixels)
            else:
                assert isinstance(api.images[0], np.ndarray) and api.images[0].shape == (1, 16, 16, 3)
                np.testing.assert_allclose(api.images[0], pixels.permute(0, 2, 3, 1).numpy(), rtol=1e-5, atol=1e-5)


def test_effective_defaults_are_materialized_before_scheduler(pipeline):
    """API omission -> checkpoint defaults, available before scheduler admission.

    Regression: request constructor's substituted CFG=1 must not override default CFG=2.
    """
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    req = pre(request(height=None, width=None, num_inference_steps=None, guidance_scale=None))
    assert req.sampling_params.height == req.sampling_params.width == 16
    assert req.sampling_params.num_inference_steps == 3
    assert req.sampling_params.guidance_scale == 2
    assert pre(request(guidance_scale=0)).sampling_params.guidance_scale == 0
    override = pre(request(guidance_scale=0, extra_args={"cfg": 4, "steps": 2, "height": 32}))
    assert override.sampling_params.guidance_scale == 4
    assert override.sampling_params.num_inference_steps == 2 and override.sampling_params.height == 32
    assert prepared(pipeline, override).total_steps == 2


def test_equivalent_hidden_dtypes_and_decoded_types_share_key(pipeline):
    """Producer payloads normalize before gather; expected: equal execution keys.

    Regression: raw hidden dtype/final formatting unnecessarily splits compatible requests.
    """
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    a, b = request(output_type="pt"), request("B", output_type="pil")
    b.prompt["extra"]["thinker_hidden_states"] = b.prompt["extra"]["thinker_hidden_states"].double()
    assert pre(a).batch_compatibility_key == pre(b).batch_compatibility_key
    assert prepared(pipeline, b).prompt_embeds.dtype == torch.float32
    b.sampling_params.output_type = "latent"
    assert pre(a).batch_compatibility_key != pre(b).batch_compatibility_key


@pytest.mark.parametrize(
    "field,value",
    [
        ("height", 0),
        ("width", 3.5),
        ("steps", -1),
        ("steps", float("inf")),
        ("cfg", float("nan")),
        ("cfg", -1),
        ("cfg_truncation", float("inf")),
    ],
)
def test_invalid_sampling_rejected_before_admission(pipeline, field, value):
    """API extra_args -> invalid values must fail before scheduler state exists."""
    with pytest.raises(ValueError):
        get_ming_image_pre_process_func(pipeline.od_config)(request(extra_args={field: value}))


@pytest.mark.parametrize("count", [0, -1, 2, True, 1.5, "bad", float("inf")])
def test_step_output_count_is_not_silently_clamped(pipeline, count):
    """Regression: STEP_BATCH supports exactly one output, never clamps invalid counts."""
    req = request(num_outputs_per_prompt=count)
    with pytest.raises(ValueError, match="num_outputs_per_prompt"):
        get_ming_image_pre_process_func(pipeline.od_config)(req)


def test_b1_multiple_outputs_and_mixed_explicit_latents(pipeline):
    """API n=2 and per-request optional latents -> independent preparation before collation.

    Regression: tensor+None collation crash, incorrect output count and routing.
    """
    a = request(num_outputs_per_prompt=2, latents=torch.full((2, 4, 8, 8), 0.5), max_sequence_length=64)
    b = request("B", num_outputs_per_prompt=2, max_sequence_length=128)
    pipeline.od_config.step_execution = False
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    assert pre(a).batch_compatibility_key == pre(b).batch_compatibility_key
    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        batch = pipeline.forward(DiffusionRequestBatch([a, b]))
        single = pipeline.forward(DiffusionRequestBatch([a]))
    assert len(batch) == 2 and all(o.output.shape == (2, 4, 8, 8) for o in batch)
    torch.testing.assert_close(batch[0].output, single[0].output)


def test_b1_custom_sigmas_are_in_compatibility_key(pipeline):
    """Reviewer regression: distinct custom schedules must never share a B1 wave."""
    pipeline.od_config.step_execution = False
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    a, b = pre(request(sigmas=[1.0, 0.5, 0.1])), pre(request(sigmas=[1.0, 0.7, 0.1]))
    assert a.batch_compatibility_key != b.batch_compatibility_key
    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        assert pipeline.forward(DiffusionRequestBatch([a]))[0].output.shape == (1, 4, 8, 8)
    pipeline.od_config.step_execution = True
    with pytest.raises(ValueError, match="custom sigmas"):
        get_ming_image_pre_process_func(pipeline.od_config)(a)


def test_missing_hidden_warns_and_malformed_hidden_fails(pipeline, caplog):
    """Bridge missing/faulty payload -> explicit fallback warning or shape error."""
    req = request()
    req.prompt["extra"].pop("thinker_hidden_states")
    assert prepared(pipeline, req).prompt_embeds.shape == (1, 256, 4)
    assert "A" in caplog.text and "using zeros" in caplog.text
    req.prompt["extra"]["thinker_hidden_states"] = torch.ones(256, 3)
    with pytest.raises(ValueError, match="invalid shape"):
        prepared(pipeline, req)


def test_no_cfg_does_not_require_negative_and_batch_order_is_checked(pipeline):
    """Regression: unused absent negatives are legal; conflicting state order is illegal."""
    a, b = prepared(pipeline, request(guidance_scale=0)), prepared(pipeline, request("B", guidance_scale=0))
    batch = InputBatch.make_batch([a, b])
    batch.negative_prompt_embeds = None
    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        expected = torch.cat([oracle_prediction(a, scale=0), oracle_prediction(b, scale=0)])
        torch.testing.assert_close(pipeline.denoise_step(batch), expected)
        with pytest.raises(ValueError, match="request order"):
            pipeline.denoise_step(batch, states=[b, a])
        with pytest.raises(ValueError, match="empty batch"):
            pipeline.denoise_step(batch, states=[])
        a.extra.pop("ming_guidance_scale")
        with pytest.raises(ValueError, match="A.*ming_guidance_scale"):
            pipeline.denoise_step(batch)


def test_reference_context_restored_on_predictor_failure(pipeline, monkeypatch):
    """Nested active context -> caller's reference and scheduler survive failed prediction."""
    state = prepared(pipeline, request())
    sentinel = torch.ones(1, 4, 8, 8)

    def fail(*args, **kwargs):
        raise RuntimeError("injected DiT failure")

    monkeypatch.setattr(pipeline.transformer, "forward", fail)
    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        get_forward_context().ref_latent = sentinel
        with pytest.raises(RuntimeError, match="injected"):
            pipeline.denoise_step(InputBatch.make_batch([state]))
        assert get_forward_context().ref_latent is sentinel
        original = pipeline.scheduler
        with pytest.raises(RuntimeError, match="injected"):
            pipeline.forward(DiffusionRequestBatch([request()]))
        assert pipeline.scheduler is original and get_forward_context().ref_latent is sentinel


def test_mixed_reference_batch_and_missing_cfg_negative_are_rejected(pipeline):
    """Real prepared states + invalid batch combination -> explicit invariant failure.

    Regression: first-row-only reference inference and silent missing negative conditions.
    """
    ref = prepared(pipeline, request(reference=Image.new("RGB", (16, 16), "red")))
    plain = prepared(pipeline, request("B"))
    with set_forward_context(omni_diffusion_config=pipeline.od_config):
        with pytest.raises(ValueError, match="cannot mix"):
            pipeline.denoise_step(InputBatch.make_batch([ref, plain]))
        batch = InputBatch.make_batch([plain])
        batch.negative_prompt_embeds = None
        with pytest.raises(ValueError, match="negative prompt embeddings are missing"):
            pipeline.denoise_step(batch)


def test_scheduler_begin_index_capability_has_clear_error():
    """Invalid scheduler capability -> request-specific error rather than silent skip."""
    with pytest.raises(RuntimeError, match="bad-scheduler.*set_begin_index"):
        MingImagePipeline._set_scheduler_begin_index(object(), 0, "bad-scheduler")


def test_invalid_prediction_geometry_fails_at_model_boundary(pipeline, monkeypatch):
    """Regression: batch-only checks must not accept extra-frame predictions."""
    state = prepared(pipeline, request(guidance_scale=0))
    monkeypatch.setattr(pipeline.transformer, "forward", lambda **kw: ([torch.ones(4, 2, 8, 8)], {}))
    with set_forward_context(omni_diffusion_config=pipeline.od_config), torch.inference_mode():
        with pytest.raises(ValueError, match="prediction must have shape"):
            pipeline.denoise_step(InputBatch.make_batch([state]))


def test_scheduler_failure_does_not_advance_request_index(pipeline, monkeypatch):
    """Prepared state + scheduler failure -> no successful-tick bookkeeping."""
    state = prepared(pipeline, request())

    def fail(*args, **kwargs):
        raise RuntimeError("injected scheduler failure")

    monkeypatch.setattr(state.scheduler, "step", fail)
    initial = state.latents.clone()
    with pytest.raises(RuntimeError, match="injected"):
        pipeline.step_scheduler(state, torch.ones_like(initial))
    assert state.step_index == 0
    torch.testing.assert_close(state.latents, initial)


class _ByT5Projection(nn.Module):
    """Identity boundary for tokenizer/padding only; no mapper correctness claim."""

    def forward(self, hidden, mask):
        return hidden


def test_actual_byt5_tokenizer_lengths_gather_safely(pipeline):
    """Equal glyph count/different UTF-8 bytes -> fixed-length encoding and zero padding.

    Input source: genuine ByT5 tokenizer and HF T5 encoder with real InputBatch.
    Expected source: max_length padding/mask contract, independent of text byte length.
    Regression: conflating raw byte length with output condition length.
    Mapper is an identity here; its real GPU implementation is tested separately.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        t5 = T5EncoderModel(
            T5Config(
                vocab_size=384,
                d_model=4,
                d_ff=8,
                d_kv=4,
                num_layers=1,
                num_heads=1,
                dropout_rate=0,
            )
        ).eval()
    tokenizer = ByT5Tokenizer()
    pipeline.byte5 = MingByT5Encoder(tokenizer, t5, _ByT5Projection(), max_length=32)
    (Path(pipeline.od_config.model) / "byt5").mkdir()
    a, b = request(extra_args={"byte5_text": ["A"]}), request("B", extra_args={"byte5_text": ["中文中文"]})
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    assert pre(a).batch_compatibility_key == pre(b).batch_compatibility_key
    states = [prepared(pipeline, r) for r in [a, b]]
    assert InputBatch.make_batch(states).prompt_embeds.shape == (2, 288, 4)
    texts = ['Text "A". ', 'Text "中文中文". ']
    tokens = tokenizer(texts, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    features = pipeline.byte5(texts)
    assert tokens.attention_mask[0].sum() != tokens.attention_mask[1].sum()
    assert torch.count_nonzero(features[tokens.attention_mask == 0]) == 0
    with set_forward_context(omni_diffusion_config=pipeline.od_config):
        assert_euler_tick(pipeline, states)


def test_missing_byt5_warns_without_splitting_equal_shapes(pipeline, caplog):
    """Known local absent ByT5 -> ignored glyphs warn but do not split equal shapes."""
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    a, b = request(), request("B", extra_args={"byte5_text": ["A", "B"]})
    assert pre(a).batch_compatibility_key == pre(b).batch_compatibility_key
    prepared(pipeline, b)
    assert "no ByT5 encoder" in caplog.text


@pytest.mark.parametrize("value", ["bad", -1.0, float("nan"), float("inf")])
def test_invalid_cfg_normalization_fails_at_admission(pipeline, value):
    """API cfg_normalize -> finite/nonnegative contract, before batch construction."""
    with pytest.raises(ValueError, match="cfg_normalize"):
        get_ming_image_pre_process_func(pipeline.od_config)(request(cfg_normalize=value))


def test_missing_and_remote_config_paths_are_explicit(pipeline, monkeypatch, caplog):
    """Downloader boundary produces a local root; VAE subfolder must come from checkpoint.

    Expected source: downloader returns the resolved root and config JSON owns
    vae_subfolder. Mock only network IO; use the actual parser and image processor.
    Regression: HF identifier treated as filesystem path or bad JSON silently hidden.
    """
    local = pipeline.od_config.model
    pipeline.od_config.tf_model_config = None
    pipeline.od_config.model = "test-org/ming-checkpoint"
    calls = []

    def resolve(model, revision, patterns):
        calls.append((model, revision))
        return local

    monkeypatch.setattr(pipeline_ming_imagegen, "download_weights_from_hf_specific", resolve)
    post = get_ming_image_post_process_func(pipeline.od_config)
    assert calls == [("test-org/ming-checkpoint", None)]
    pixels = post(torch.zeros(1, 3, 16, 16), OmniDiffusionSamplingParams(output_type="pt"))
    torch.testing.assert_close(pixels["payload"]["image"], torch.full((1, 3, 16, 16), 0.5))
    (Path(local) / "custom_vae" / "config.json").unlink()
    get_ming_image_post_process_func(pipeline.od_config)
    assert "is missing" in caplog.text


@pytest.mark.parametrize("shape", [(4, 8, 8), (1, 4, 7, 8), (2, 4, 8, 8)])
def test_explicit_latent_geometry_matches_public_batch_layout(pipeline, shape):
    """API latents require [n,C,H/scale,W/scale]; malformed geometry must fail early."""
    with pytest.raises(ValueError, match="latents shape"):
        prepared(pipeline, request(latents=torch.ones(shape)))


@pytest.mark.parametrize("bad_config", ["{", "[]", '{"block_out_channels": []}'])
def test_postprocess_bad_vae_config_is_not_silently_defaulted(pipeline, bad_config):
    """Checkpoint corruption -> error, never silent scale=8."""
    path = Path(pipeline.od_config.model) / "custom_vae" / "config.json"
    path.write_text(bad_config, encoding="utf-8")
    with pytest.raises(ValueError, match="Ming VAE config"):
        get_ming_image_post_process_func(pipeline.od_config)


@pytest.mark.parametrize("mutation", ["sign", "cfg_formula", "frame_axis", "shared_scheduler"])
def test_critical_contracts_kill_deliberate_mutations(pipeline, monkeypatch, mutation):
    """Same bridge inputs and independent Euler oracle must catch concrete wrong behavior.

    Regression: tests must constrain behavior, not just execute code successfully.
    """
    a, b = prepared(pipeline, request(negative=True)), prepared(pipeline, request("B"))
    if mutation == "sign":
        original = pipeline.denoise_step
        monkeypatch.setattr(pipeline, "denoise_step", lambda *a, **kw: -original(*a, **kw))
    elif mutation == "cfg_formula":
        monkeypatch.setattr(pipeline, "combine_cfg_noise", lambda p, n, scale, *a, **kw: p[0])
    elif mutation == "frame_axis":
        original = pipeline._build_denoise_kwargs

        def wrong_axis(x, *args):
            return original([sample.transpose(1, 2) for sample in x], *args)

        monkeypatch.setattr(pipeline, "_build_denoise_kwargs", wrong_axis)
    else:
        with set_forward_context(omni_diffusion_config=pipeline.od_config):
            assert_euler_tick(pipeline, [a])
        b.scheduler = a.scheduler
    with set_forward_context(omni_diffusion_config=pipeline.od_config):
        with pytest.raises((AssertionError, ValueError, IndexError)):
            assert_euler_tick(pipeline, [b, a])
