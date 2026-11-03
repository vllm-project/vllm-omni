# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for Nucleus step execution and expert-choice routing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn

import vllm_omni.diffusion.models.nucleus_image.nucleus_image_transformer as transformer_module
import vllm_omni.diffusion.models.nucleus_image.pipeline_nucleus_image as pipeline_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import _func_accepts_parameter
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.models.nucleus_image.nucleus_image_transformer import NucleusMoELayer, SwiGLUExperts
from vllm_omni.diffusion.models.nucleus_image.pipeline_nucleus_image import NucleusMoEImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def pipeline():
    pipe = NucleusMoEImagePipeline.__new__(NucleusMoEImagePipeline)
    nn.Module.__init__(pipe)
    pipe.vae_scale_factor = 8
    pipe.default_sample_size = 128
    pipe.default_max_sequence_length = 1024
    pipe.default_return_index = -8
    pipe._interrupt = False
    pipe._attention_kwargs = None
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()
    pipe._text_kv_request_ids = ()
    pipe.transformer = Mock()
    pipe.scheduler.set_timesteps(2)

    def prepare_context(**kwargs):
        return {
            "prompt_embeds": torch.ones(1, 2, 4),
            "prompt_embeds_mask": torch.ones(1, 2, dtype=torch.bool),
            "negative_prompt_embeds": torch.zeros(1, 2, 4),
            "negative_prompt_embeds_mask": torch.ones(1, 2, dtype=torch.bool),
            "do_true_cfg": kwargs["guidance_scale"] > 1,
            "true_cfg_scale": kwargs["guidance_scale"],
            "latents": torch.ones(1, 1, 4),
            "timesteps": pipe.scheduler.timesteps,
            "img_shapes": [[(1, 1, 1)]],
        }

    pipe._prepare_generation_context = Mock(side_effect=prepare_context)
    return pipe


@pytest.mark.parametrize("guidance_scale", [None, 1.0, 2.5])
def test_prepare_encode_uses_runner_state_and_cfg(pipeline, guidance_scale):
    sampling = OmniDiffusionSamplingParams(height=16, width=16, num_inference_steps=2)
    sampling.guidance_scale_provided = guidance_scale is not None
    if guidance_scale is not None:
        sampling.guidance_scale = guidance_scale
    state = StepRequestState(
        request_id="nucleus",
        sampling=sampling,
        prompt={"prompt": "a cat", "negative_prompt": "blurry"},
    )
    pipeline.prepare_encode(state)
    batch = InputBatch.make_batch([state])
    expected_scale = 4.0 if guidance_scale is None else guidance_scale
    assert pipeline._prepare_generation_context.call_args.kwargs["prompt"] == ["a cat"]
    assert pipeline._prepare_generation_context.call_args.kwargs["negative_prompt"] == ["blurry"]
    assert pipeline._prepare_generation_context.call_args.kwargs["guidance_scale"] == expected_scale
    assert batch.do_true_cfg == (expected_scale > 1)
    assert batch.true_cfg_scale == expected_scale
    assert batch.cfg_normalize
    assert state.scheduler is not pipeline.scheduler


def test_post_decode_keeps_prepared_size_and_output_type(pipeline):
    state = StepRequestState(
        request_id="small-image",
        sampling=OmniDiffusionSamplingParams(height=1, width=17, output_type="latent"),
        prompt="a cat",
    )
    pipeline.prepare_encode(state)
    pipeline._decode_latents = Mock(return_value=DiffusionOutput(output=state.latents))
    pipeline.post_decode(state)
    pipeline._decode_latents.assert_called_once_with(state.latents, 16, 16, "latent")


def test_denoise_step_consumes_input_batch(pipeline):
    state = StepRequestState(
        request_id="nucleus",
        sampling=OmniDiffusionSamplingParams(true_cfg_scale=2.5, cfg_normalize=True),
        latents=torch.ones(1, 1, 4),
        timesteps=pipeline.scheduler.timesteps,
        scheduler=pipeline.scheduler,
        do_true_cfg=True,
        prompt_embeds=torch.ones(1, 2, 4),
        prompt_embeds_mask=torch.ones(1, 2, dtype=torch.bool),
        negative_prompt_embeds=torch.zeros(1, 2, 4),
        negative_prompt_embeds_mask=torch.ones(1, 2, dtype=torch.bool),
        img_shapes=[[(1, 1, 1)]],
    )
    positive = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    negative = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    pipeline._predict_noise = Mock(side_effect=[positive, negative])
    combined = negative + 2.5 * (positive - negative)
    expected = -combined * (positive.norm(dim=-1, keepdim=True) / combined.norm(dim=-1, keepdim=True))
    noise = pipeline.denoise_step(InputBatch.make_batch([state]))
    torch.testing.assert_close(noise, expected)
    pipeline.step_scheduler(state, noise)
    assert state.step_index == 1
    assert state.latents.shape == (1, 1, 4)


@pytest.mark.parametrize("output_type", [None, "latent"])
def test_forward_uses_sampling_output_type(pipeline, output_type):
    sampling = OmniDiffusionSamplingParams(
        height=16, width=16, num_inference_steps=2, guidance_scale=1.0, output_type=output_type
    )
    request = OmniDiffusionRequest(request_id="output-type", prompt="a cat", sampling_params=sampling)
    pipeline._predict_noise = Mock(return_value=torch.zeros(1, 1, 4))
    pipeline._decode_latents = Mock(return_value=DiffusionOutput(output=torch.ones(1, 1, 4)))
    pipeline.forward(DiffusionRequestBatch([request]))
    assert pipeline._decode_latents.call_args.args[-1] == (output_type or "pil")


@pytest.mark.parametrize("output_type", [None, "latent"])
def test_postprocess_preserves_requested_latents(tmp_path, monkeypatch, output_type):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text("{}")
    processor = Mock()
    monkeypatch.setattr(pipeline_module, "VaeImageProcessor", Mock(return_value=processor))
    postprocess = pipeline_module.get_nucleus_image_post_process_func(SimpleNamespace(model=str(tmp_path)))
    sampling = OmniDiffusionSamplingParams(output_type=output_type)
    latents = torch.ones(1, 4, 64)
    kwargs = {"sampling_params": sampling} if _func_accepts_parameter(postprocess, "sampling_params") else {}
    result = postprocess(latents, **kwargs)
    if output_type == "latent":
        assert result is latents
        processor.postprocess.assert_not_called()
    else:
        processor.postprocess.assert_called_once_with(latents)
        assert result is processor.postprocess.return_value


def test_denoise_invalidates_text_kv_when_batch_order_changes(pipeline):
    block = transformer_module.NucleusMoEImageTransformerBlock.__new__(
        transformer_module.NucleusMoEImageTransformerBlock
    )
    nn.Module.__init__(block)
    projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
    block.encoder_proj = Mock(wraps=projection)
    block.attn = SimpleNamespace(
        add_k_proj=nn.Identity(), add_v_proj=nn.Identity(), norm_added_k=nn.Identity(), num_kv_heads=1, head_dim=2
    )
    block._text_kv_cache = {}
    pipeline.transformer = SimpleNamespace(reset_text_kv_cache=block.reset_text_kv_cache)

    def predict_noise(latents, img_shapes, embeds, mask, timestep, attention_kwargs):
        _, values = block._get_cached_text_kv(embeds.clone(), cache_source=embeds)
        return values.reshape(embeds.shape[0], 1, 4)

    pipeline._predict_noise = Mock(side_effect=predict_noise)
    states = [
        StepRequestState(
            request_id=name,
            sampling=OmniDiffusionSamplingParams(),
            prompt_embeds=torch.full((1, 2, 2), value),
            latents=torch.zeros(1, 1, 4),
            timesteps=torch.ones(2),
            img_shapes=[(1, 1, 1)],
        )
        for name, value in (("A", 1.0), ("B", 3.0))
    ]
    # A is running when B joins: new states precede cached states in the runner.
    batch = InputBatch.make_batch([states[1], states[0]])
    source_ptr = batch.prompt_embeds.data_ptr()
    with torch.no_grad():
        noise = pipeline.denoise_step(batch)
        torch.testing.assert_close(noise, -batch.prompt_embeds.reshape(2, 1, 4))
        pipeline.denoise_step(batch)
        assert block.encoder_proj.call_count == 1

        # The next tick has only cached states, in the scheduler's A, B order.
        batch = InputBatch.make_batch(states, cached_batch=batch)
        assert batch.prompt_embeds.data_ptr() == source_ptr
        noise = pipeline.denoise_step(batch)
        torch.testing.assert_close(noise, -batch.prompt_embeds.reshape(2, 1, 4))
        assert block.encoder_proj.call_count == 2

        pipeline.denoise_step(batch)
        assert block.encoder_proj.call_count == 2


def test_text_kv_cache_uses_prompt_identity_before_normalization():
    # Model allocator reuse: positive and negative RMSNorm outputs can have
    # the same address even though their persistent source embeddings differ.
    positive = torch.tensor([[[1.0, 2.0]]])
    negative = torch.tensor([[[3.0, 4.0]]])
    scratch = torch.empty_like(positive)
    projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
    block = transformer_module.NucleusMoEImageTransformerBlock.__new__(
        transformer_module.NucleusMoEImageTransformerBlock
    )
    nn.Module.__init__(block)
    block.encoder_proj = Mock(wraps=projection)
    block.attn = SimpleNamespace(
        add_k_proj=nn.Identity(),
        add_v_proj=nn.Identity(),
        norm_added_k=nn.Identity(),
        num_kv_heads=1,
        head_dim=2,
    )
    block._text_kv_cache = {}
    block.forward = Mock(side_effect=lambda **kwargs: kwargs["hidden_states"])
    model = transformer_module.NucleusMoEImageTransformer2DModel.__new__(
        transformer_module.NucleusMoEImageTransformer2DModel
    )
    nn.Module.__init__(model)
    freqs = torch.ones(1, 1, dtype=torch.complex64)
    model.image_rope_prepare = Mock(return_value=(torch.ones(1, 1, 2), freqs, freqs))
    model.time_text_embed = Mock(return_value=torch.zeros(1, 2))
    model.txt_norm = Mock(side_effect=lambda source: scratch.copy_(source))
    model.transformer_blocks = nn.ModuleList([block])
    model.norm_out = Mock(side_effect=lambda hidden, temb: hidden)
    model.proj_out = nn.Identity()
    with torch.no_grad():
        keys = []
        for source in (positive, negative, positive):
            model(
                hidden_states=torch.ones(1, 1, 2),
                img_shapes=[(1, 1, 1)],
                encoder_hidden_states=source,
                timestep=torch.ones(1),
            )
            keys.append(block.forward.call_args.kwargs["cached_txt_key"])
    torch.testing.assert_close(keys[0], positive.unsqueeze(2))
    torch.testing.assert_close(keys[1], negative.unsqueeze(2))
    assert keys[2] is keys[0]
    assert block.encoder_proj.call_count == 2


@pytest.mark.parametrize("sequence_length", [8, 7])
@pytest.mark.parametrize("use_sigmoid", [False, True])
def test_sp_expert_choice_matches_unsharded_tokens(monkeypatch, sequence_length, use_sigmoid):
    layer = NucleusMoELayer.__new__(NucleusMoELayer)
    nn.Module.__init__(layer)
    layer.num_experts = 2
    layer.capacity_factor = 1.0
    layer.use_sigmoid = use_sigmoid
    layer.route_scale = 2.5
    layer.gate = nn.Identity()
    layer.experts = SwiGLUExperts(hidden_size=2, moe_intermediate_dim=3, num_experts=2)
    layer.shared_expert = nn.Linear(2, 2, bias=False)
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    hidden = torch.randn(2, sequence_length, 2, generator=generator)
    # Both experts select unevenly across ranks; local top-k cannot reproduce this.
    logits = torch.tensor(
        [[8.0, 0.0], [7.0, 0.0], [6.0, 0.0], [5.0, 0.0], [0.0, 8.0], [0.0, 7.0], [0.0, 6.0], [0.0, 5.0]]
    )[:sequence_length]
    logits = torch.stack([logits, logits.flip(0)])
    with torch.no_grad():
        reference = layer(hidden, logits)
        shard_length = (sequence_length + 1) // 2
        padding = 2 * shard_length - sequence_length
        padded_hidden = torch.cat([hidden, hidden.new_zeros(2, padding, 2)], dim=1)
        # Padding has high affinity to ensure it never displaces a valid token.
        padded_logits = torch.cat([logits, logits.new_full((2, padding, 2), 100.0)], dim=1)
        scores = padded_logits.sigmoid() if use_sigmoid else padded_logits.softmax(dim=-1)
        for rank in range(2):
            group = SimpleNamespace(world_size=2, rank_in_group=rank, all_gather=Mock(return_value=scores))
            monkeypatch.setattr(transformer_module, "get_sp_group", lambda: group)
            context = ForwardContext(
                sp_plan_hooks_applied=True,
                _sp_shard_depth=1,
                sp_original_seq_len=sequence_length,
                sp_padding_size=padding,
            )
            start = rank * shard_length
            with override_forward_context(context):
                output = layer(
                    padded_hidden[:, start : start + shard_length], padded_logits[:, start : start + shard_length]
                )
            valid_length = min(shard_length, sequence_length - start)
            torch.testing.assert_close(output[:, :valid_length], reference[:, start : start + valid_length])
