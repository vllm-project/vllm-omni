# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for CFG packing and request plumbing, without loading model weights.

The fake transformer checks branch conditions and loop semantics. Real-model
attention/RoPE parity and image quality need separate GPU qualification.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as pipeline_module
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _pack_cfg_conditions,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("positive_length,negative_length", [(3, 0), (0, 3), (2, 5), (5, 2), (0, 0), (3, 3)])
def test_pack_conditions_preserves_branch_order_and_right_padding(positive_length, negative_length):
    positive = torch.arange(positive_length * 2, dtype=torch.bfloat16).reshape(1, positive_length, 2) + 10
    negative = torch.arange(negative_length * 2, dtype=torch.bfloat16).reshape(1, negative_length, 2) - 20
    positive_mask = torch.ones((1, positive_length), dtype=torch.bool)
    negative_mask = torch.ones((1, negative_length), dtype=torch.bool)
    # Preserve existing right padding, including nonzero values under its mask.
    if negative_length > 1:
        negative_mask[:, -1] = False
    before = [tensor.clone() for tensor in (positive, positive_mask, negative, negative_mask)]

    packed, mask = _pack_cfg_conditions(positive, positive_mask, negative, negative_mask)

    assert packed.shape == (2, max(positive_length, negative_length), 2)
    assert packed.dtype == positive.dtype
    assert mask.dtype == torch.bool
    for row, (embeds, original_mask) in enumerate(((positive, positive_mask), (negative, negative_mask))):
        length = embeds.shape[1]
        assert torch.equal(packed[row, :length], embeds[0])
        assert torch.equal(mask[row, :length], original_mask[0])
        assert torch.count_nonzero(packed[row, length:]) == 0
        assert not mask[row, length:].any()
    for actual, original in zip((positive, positive_mask, negative, negative_mask), before):
        assert torch.equal(actual, original)


@pytest.mark.parametrize("invalid_row", [0, 1])
def test_pack_conditions_rejects_mask_holes(invalid_row):
    conditions = [torch.ones(1, 3, 2), torch.ones(1, 3, 2)]
    masks = [torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)]
    masks[invalid_row][0, 1] = False
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        _pack_cfg_conditions(conditions[0], masks[0], conditions[1], masks[1])


def test_pack_conditions_rejects_multiple_requests():
    with pytest.raises(ValueError, match="single-request"):
        _pack_cfg_conditions(
            torch.ones(2, 3, 2),
            torch.ones(2, 3, dtype=torch.bool),
            torch.empty(1, 0, 2),
            torch.empty(1, 0, dtype=torch.bool),
        )


class _Transformer(nn.Module):
    def __init__(self, *, nested=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(in_channels=1)
        self.time_caption_embed = SimpleNamespace(image_embedder=object() if nested else None)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        )
        hidden = kwargs["hidden_states"]
        context = kwargs["text_hidden_states"]
        mask = kwargs["text_attention_mask"]
        conditioning = (context * mask.unsqueeze(-1)).sum(dim=(1, 2))
        if kwargs.get("ar_image_hidden_states") is not None:
            conditioning = conditioning + kwargs["ar_image_hidden_states"].sum(dim=(1, 2))
        return hidden * 0.125 + conditioning[:, None, None, None] + kwargs["timestep"][:, None, None, None]


class _ImageRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, image, padding_mask):
        self.calls += 1
        assert not padding_mask.any()
        # Make the refined output distinguishable from raw AR image conditions.
        return image + 100


class _Scheduler:
    def __init__(self):
        self.calls = []

    def set_timesteps(self, *, num_inference_steps, device, num_tokens):
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.float32)

    def step(self, prediction, timestep, latents, *, return_dict):
        assert prediction.shape == latents.shape
        assert latents.shape[0] == 1
        self.calls.append((prediction.clone(), latents.clone()))
        return (latents - prediction * 0.125,)


class _VAE(nn.Module):
    config = SimpleNamespace(scaling_factor=None, shift_factor=None)

    def decode(self, latents, *, return_dict):
        return (latents,)


def _default_parallel_config():
    return SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=None,
        ulysses_degree=1,
        ring_degree=1,
        allgather_degree=1,
    )


@pytest.fixture
def pipeline_factory(monkeypatch):
    def build(
        *,
        refined=False,
        model_type="mammothmoda2_qwen2_5_vl",
        nested=False,
        enforce_eager=True,
        cache_backend="none",
        cache_strategy="none",
        parallel_config=None,
    ):
        pipeline = MammothModa2DiTPipeline.__new__(MammothModa2DiTPipeline)
        nn.Module.__init__(pipeline)
        pipeline.config = SimpleNamespace(
            llm_config=SimpleNamespace(model_type=model_type, gen_vocab_start_index=100),
            image_token_id=900,
            video_token_id=901,
            vision_start_token_id=902,
            vision_end_token_id=903,
        )
        pipeline.od_config = SimpleNamespace(
            enforce_eager=enforce_eager,
            cache_backend=cache_backend,
            cache_strategy=cache_strategy,
            parallel_config=parallel_config or _default_parallel_config(),
        )
        pipeline.gen_transformer = _Transformer(nested=nested)
        pipeline.gen_image_condition_refiner = _ImageRefiner() if refined else None
        pipeline.gen_vae = _VAE()
        pipeline.gen_freqs_cis = object()
        scheduler = _Scheduler()
        monkeypatch.setattr(pipeline_module, "FlowMatchEulerDiscreteScheduler", lambda: scheduler)
        monkeypatch.setattr(
            pipeline_module, "randn_tensor", lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs.get("dtype"))
        )
        return pipeline, scheduler

    return build


def _batch(
    *,
    request_id: str = "req-cfg",
    guidance: float = 4.0,
    cfg_range: tuple[float, float] = (0.0, 1.0),
    extra_args: dict | None = None,
    num_inference_steps: int = 4,
) -> DiffusionRequestBatch:
    # Text tokens [10, 11] (< answer_start_index=2, not visual/gen) -> text_cond.
    # Answer tokens [100, 101] (>= gen_vocab_start_index=100) -> image_cond.
    prompt = {
        "prompt": "",
        "height": 16,
        "width": 16,
        "additional_information": {
            "full_hidden_states": torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]),
            "full_token_ids": [10, 11, 100],
            "answer_start_index": 2,
        },
    }
    merged_extra_args = {"cfg_range": list(cfg_range)}
    if extra_args:
        merged_extra_args.update(extra_args)
    sampling = OmniDiffusionSamplingParams(
        height=16,
        width=16,
        guidance_scale=guidance,
        num_inference_steps=num_inference_steps,
        extra_args=merged_extra_args,
    )
    return DiffusionRequestBatch([OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=request_id)])


@pytest.mark.parametrize(
    "guidance,cfg_range,packed_batch_sizes,sequential_calls",
    [
        (4.0, (0.0, 1.0), [2, 2, 2, 2], 8),
        (4.0, (0.25, 0.5), [1, 2, 2, 1], 6),
        (4.0, (0.0, 0.0), [2, 1, 1, 1], 5),
        (4.0, (0.5, 0.5), [1, 1, 2, 1], 5),
        # i / N never reaches 1, including on the last step.
        (4.0, (1.0, 1.0), [1, 1, 1, 1], 4),
        (1.0, (0.0, 1.0), [1, 1, 1, 1], 4),
        (0.5, (0.0, 1.0), [1, 1, 1, 1], 4),
    ],
)
def test_packed_loop_matches_sequential_and_preserves_cfg_boundaries(
    pipeline_factory, guidance, cfg_range, packed_batch_sizes, sequential_calls
):
    # The negative branch is always the empty unconditional prompt produced by
    # forward() itself; there is no external negative-condition input anymore.
    sequential, sequential_scheduler = pipeline_factory()
    reference = sequential.forward(_batch(guidance=guidance, cfg_range=cfg_range)).output
    packed, packed_scheduler = pipeline_factory()
    result = packed.forward(
        _batch(guidance=guidance, cfg_range=cfg_range, extra_args={"cfg_execution_mode": "packed"})
    ).output

    assert len(sequential.gen_transformer.calls) == sequential_calls
    assert all(call["hidden_states"].shape[0] == 1 for call in sequential.gen_transformer.calls)
    assert [call["hidden_states"].shape[0] for call in packed.gen_transformer.calls] == packed_batch_sizes
    assert len(sequential_scheduler.calls) == len(packed_scheduler.calls) == 4
    for reference_step, packed_step in zip(sequential_scheduler.calls, packed_scheduler.calls):
        torch.testing.assert_close(packed_step[0], reference_step[0], rtol=0, atol=0)
        torch.testing.assert_close(packed_step[1], reference_step[1], rtol=0, atol=0)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for call in packed.gen_transformer.calls:
        if call["hidden_states"].shape[0] == 2:
            assert torch.equal(call["hidden_states"][0], call["hidden_states"][1])
            assert torch.equal(call["timestep"][0], call["timestep"][1])


def test_packed_loop_preserves_refined_positive_and_pads_empty_negative(pipeline_factory):
    sequential, _ = pipeline_factory(refined=True)
    reference = sequential.forward(_batch()).output
    pipeline, _ = pipeline_factory(refined=True)
    result = pipeline.forward(_batch(extra_args={"cfg_execution_mode": "packed"})).output

    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert pipeline.gen_image_condition_refiner.calls == 1
    expected_positive = torch.cat([torch.tensor([[2.0, 3.0], [5.0, 7.0]]), torch.tensor([[111.0, 113.0]])], dim=0)
    for call in pipeline.gen_transformer.calls:
        assert call.get("ar_image_hidden_states") is None
        assert call.get("ar_image_attention_mask") is None
        context, mask = call["text_hidden_states"], call["text_attention_mask"]
        assert torch.equal(context[0, :3], expected_positive)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        if context.shape[0] == 2:
            # The negative row is always the empty unconditional prompt, padded
            # with zeros to the positive row's length.
            assert torch.count_nonzero(context[1]) == 0
            assert not mask[1].any()


def test_packed_static_conditions_are_built_once(pipeline_factory, monkeypatch):
    calls = []

    def track_pack(*args):
        calls.append(args)
        return _pack_cfg_conditions(*args)

    monkeypatch.setattr(pipeline_module, "_pack_cfg_conditions", track_pack)
    pipeline, _ = pipeline_factory()
    pipeline.forward(_batch(extra_args={"cfg_execution_mode": "packed"}))
    assert len(calls) == 1
    conditions = [call["text_hidden_states"] for call in pipeline.gen_transformer.calls]
    assert all(torch.equal(condition, conditions[0]) for condition in conditions)


@pytest.mark.parametrize(
    "guidance,cfg_range",
    [(4.0, (1.0, 1.0)), (4.0, (0.1, 0.2)), (1.0, (0.0, 1.0)), (0.5, (0.0, 1.0))],
)
def test_no_packing_when_no_step_uses_cfg(pipeline_factory, monkeypatch, guidance, cfg_range):
    def unexpected_pack(*args):
        pytest.fail("Conditions must not be packed when no denoising step uses CFG")

    monkeypatch.setattr(pipeline_module, "_pack_cfg_conditions", unexpected_pack)
    sequential, _ = pipeline_factory()
    reference = sequential.forward(_batch(guidance=guidance, cfg_range=cfg_range)).output
    packed, scheduler = pipeline_factory()
    result = packed.forward(
        _batch(guidance=guidance, cfg_range=cfg_range, extra_args={"cfg_execution_mode": "packed"})
    ).output
    assert len(packed.gen_transformer.calls) == len(scheduler.calls) == 4
    assert all(call["hidden_states"].shape[0] == 1 for call in packed.gen_transformer.calls)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_extra_body_mode_reaches_forward_and_overrides_legacy_sampling_knobs(pipeline_factory):
    params = OmniDiffusionSamplingParams(guidance_scale=1.0, height=16, width=16, num_inference_steps=4)
    apply_declared_extra_args(
        params,
        get_extra_body_params("MammothModa2ForConditionalGeneration"),
        {"cfg_execution_mode": "packed", "text_guidance_scale": 4.0, "cfg_range": [0.0, 0.0]},
    )
    pipeline, scheduler = pipeline_factory()
    batch = _batch(guidance=1.0)
    batch.requests[0].sampling_params = params
    pipeline.forward(batch)
    assert [call["hidden_states"].shape[0] for call in pipeline.gen_transformer.calls] == [2, 1, 1, 1]
    # Positive sum = 41, null prediction = 0, guidance = 4, first latent/timestep = 0.
    assert torch.equal(scheduler.calls[0][0], torch.full_like(scheduler.calls[0][0], 164.0))


@pytest.mark.parametrize("mode", ["typo", True, 2])
def test_invalid_execution_mode_fails_before_transformer(pipeline_factory, mode):
    pipeline, _ = pipeline_factory()
    with pytest.raises(ValueError, match="cfg_execution_mode"):
        pipeline.forward(_batch(extra_args={"cfg_execution_mode": mode}))
    assert pipeline.gen_transformer.calls == []


@pytest.mark.parametrize(
    "model_type,nested",
    [("mammothmoda2_qwen3_vl", True), ("mammothmoda2_qwen3_vl", False), ("mammothmoda2_qwen2_5_vl", True)],
)
def test_packed_rejects_unqualified_model_variants(pipeline_factory, model_type, nested):
    pipeline, _ = pipeline_factory(model_type=model_type, nested=nested)
    with pytest.raises(NotImplementedError, match="Preview only"):
        pipeline.forward(_batch(extra_args={"cfg_execution_mode": "packed"}))
    assert pipeline.gen_transformer.calls == []


def test_packed_rejects_non_eager_execution(pipeline_factory):
    pipeline, _ = pipeline_factory(enforce_eager=False)
    with pytest.raises(NotImplementedError, match="eager execution"):
        pipeline.forward(_batch(extra_args={"cfg_execution_mode": "packed"}))
    assert pipeline.gen_transformer.calls == []


@pytest.mark.parametrize("cache_backend,cache_strategy", [("tea_cache", "none"), ("none", "legacy")])
def test_packed_rejects_diffusion_caching(pipeline_factory, cache_backend, cache_strategy):
    pipeline, _ = pipeline_factory(cache_backend=cache_backend, cache_strategy=cache_strategy)
    with pytest.raises(NotImplementedError, match="caching disabled"):
        pipeline.forward(_batch(extra_args={"cfg_execution_mode": "packed"}))
    assert pipeline.gen_transformer.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"pipeline_parallel_size": 2},
        {"tensor_parallel_size": 2},
        {"sequence_parallel_size": 2},
        {"ulysses_degree": 2},
        {"ring_degree": 2},
        {"allgather_degree": 2},
    ],
)
def test_packed_rejects_multi_device_parallelism(pipeline_factory, overrides):
    parallel_config = _default_parallel_config()
    for key, value in overrides.items():
        setattr(parallel_config, key, value)
    pipeline, _ = pipeline_factory(parallel_config=parallel_config)
    with pytest.raises(NotImplementedError, match="single-device execution"):
        pipeline.forward(_batch(extra_args={"cfg_execution_mode": "packed"}))
    assert pipeline.gen_transformer.calls == []


def test_dev_default_sequential_keeps_image_conditioning_on_positive_only(pipeline_factory):
    pipeline, _ = pipeline_factory(model_type="mammothmoda2_qwen3_vl", nested=True)
    pipeline.forward(_batch())
    assert len(pipeline.gen_transformer.calls) == 8
    for index, call in enumerate(pipeline.gen_transformer.calls):
        assert call["hidden_states"].shape[0] == 1
        if index % 2 == 0:
            assert call["ar_image_hidden_states"] is not None
        else:
            assert call.get("ar_image_hidden_states") is None
