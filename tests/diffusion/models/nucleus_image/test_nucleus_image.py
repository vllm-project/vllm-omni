# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for Nucleus pipeline, text KV, RoPE and sequence parallelism."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn

import vllm_omni.diffusion.models.nucleus_image.nucleus_image_transformer as transformer_module
import vllm_omni.diffusion.models.nucleus_image.pipeline_nucleus_image as pipeline_module
from vllm_omni.diffusion.attention.backends import ring_pytorch_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import _func_accepts_parameter
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.models.nucleus_image.nucleus_image_transformer import (
    NucleusMoEImageTransformerBlock,
    NucleusMoELayer,
    SwiGLUExperts,
)
from vllm_omni.diffusion.models.nucleus_image.pipeline_nucleus_image import NucleusMoEImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def text_kv_block(mocker):
    block = NucleusMoEImageTransformerBlock.__new__(NucleusMoEImageTransformerBlock)
    nn.Module.__init__(block)
    projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
    block.encoder_proj = mocker.Mock(wraps=projection)
    block.attn = SimpleNamespace(
        add_k_proj=nn.Identity(),
        add_v_proj=nn.Identity(),
        norm_added_k=nn.Identity(),
        num_kv_heads=1,
        head_dim=2,
    )
    block._text_kv_cache = {}
    return block


def _generation_context(timesteps, *, guidance_scale=4.0, sequence_length=2):
    return {
        "prompt_embeds": torch.ones(1, sequence_length, 4),
        "prompt_embeds_mask": torch.ones(1, sequence_length, dtype=torch.bool),
        "negative_prompt_embeds": torch.zeros(1, sequence_length, 4),
        "negative_prompt_embeds_mask": torch.ones(1, sequence_length, dtype=torch.bool),
        "do_true_cfg": guidance_scale > 1,
        "true_cfg_scale": guidance_scale,
        "latents": torch.ones(1, 1, 4),
        "timesteps": timesteps,
        "img_shapes": [[(1, 1, 1)]],
    }


@pytest.fixture
def pipeline(mocker):
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
    pipe.transformer = mocker.Mock()
    pipe.scheduler.set_timesteps(2)

    pipe._prepare_generation_context = mocker.Mock(
        side_effect=lambda **kwargs: _generation_context(
            pipe.scheduler.timesteps, guidance_scale=kwargs["guidance_scale"]
        )
    )
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


def test_post_decode_keeps_prepared_size_and_output_type(mocker, pipeline):
    state = StepRequestState(
        request_id="small-image",
        sampling=OmniDiffusionSamplingParams(height=1, width=17, output_type="latent"),
        prompt="a cat",
    )
    pipeline.prepare_encode(state)
    pipeline._decode_latents = mocker.Mock(return_value=DiffusionOutput(output=state.latents))
    pipeline.post_decode(state)
    pipeline._decode_latents.assert_called_once_with(state.latents, 16, 16, "latent")


def test_denoise_step_consumes_input_batch(mocker, pipeline):
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
    pipeline._predict_noise = mocker.Mock(side_effect=[positive, negative])
    combined = negative + 2.5 * (positive - negative)
    expected = -combined * (positive.norm(dim=-1, keepdim=True) / combined.norm(dim=-1, keepdim=True))
    noise = pipeline.denoise_step(InputBatch.make_batch([state]))
    torch.testing.assert_close(noise, expected)
    pipeline.step_scheduler(state, noise)
    assert state.step_index == 1
    assert state.latents.shape == (1, 1, 4)


@pytest.mark.parametrize("output_type", [None, "latent"])
def test_forward_uses_sampling_output_type(mocker, pipeline, output_type):
    sampling = OmniDiffusionSamplingParams(
        height=16, width=16, num_inference_steps=2, guidance_scale=1.0, output_type=output_type
    )
    request = OmniDiffusionRequest(request_id="output-type", prompt="a cat", sampling_params=sampling)
    pipeline._predict_noise = mocker.Mock(return_value=torch.zeros(1, 1, 4))
    pipeline._decode_latents = mocker.Mock(return_value=DiffusionOutput(output=torch.ones(1, 1, 4)))
    pipeline.forward(DiffusionRequestBatch([request]))
    assert pipeline._decode_latents.call_args.args[-1] == (output_type or "pil")


@pytest.mark.parametrize("output_type", [None, "latent"])
def test_postprocess_preserves_requested_latents(mocker, tmp_path, monkeypatch, output_type):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text("{}")
    processor = mocker.Mock()
    monkeypatch.setattr(pipeline_module, "VaeImageProcessor", mocker.Mock(return_value=processor))
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


def test_denoise_invalidates_text_kv_when_batch_order_changes(mocker, pipeline, text_kv_block):
    block = text_kv_block
    pipeline.transformer = SimpleNamespace(reset_text_kv_cache=block.reset_text_kv_cache)

    def predict_noise(latents, img_shapes, embeds, mask, timestep, attention_kwargs):
        _, values = block._get_cached_text_kv(embeds.clone(), cache_source=embeds)
        return values.reshape(embeds.shape[0], 1, 4)

    pipeline._predict_noise = mocker.Mock(side_effect=predict_noise)
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


@pytest.mark.parametrize("field", ["prompt_embeds", "negative_prompt_embeds"])
@pytest.mark.parametrize("both_maskless", [False, True])
def test_prepare_encode_keeps_masks_for_variable_length_batches(mocker, pipeline, field, both_maskless):
    lengths = [8, 16 if both_maskless else 10]
    contexts = []
    for row, length in enumerate(lengths):
        padded_length = (length + 7) // 8 * 8
        context = _generation_context(pipeline.scheduler.timesteps, sequence_length=8)
        context[field] = torch.full((1, padded_length, 4), float(row + 1))
        context[field + "_mask"] = None if length == padded_length else torch.arange(padded_length)[None, :] < length
        contexts.append(context)
    pipeline._prepare_generation_context = mocker.Mock(side_effect=contexts)
    states = [
        StepRequestState(
            request_id=str(row),
            sampling=OmniDiffusionSamplingParams(height=16, width=16),
            prompt="a cat",
        )
        for row in range(2)
    ]
    for state in states:
        pipeline.prepare_encode(state)
    batch = InputBatch.make_batch(states)
    mask = getattr(batch, field + "_mask")
    assert mask.dtype == torch.bool
    assert mask.sum(1).tolist() == lengths
    torch.testing.assert_close(getattr(batch, field)[0, : lengths[0]], torch.ones(lengths[0], 4))
    torch.testing.assert_close(getattr(batch, field)[1, : lengths[1]], torch.full((lengths[1], 4), 2.0))


def test_text_kv_cache_uses_prompt_identity_before_normalization(mocker, text_kv_block):
    # Model allocator reuse: positive and negative RMSNorm outputs can have
    # the same address even though their persistent source embeddings differ.
    positive = torch.tensor([[[1.0, 2.0]]])
    negative = torch.tensor([[[3.0, 4.0]]])
    scratch = torch.empty_like(positive)
    block = text_kv_block
    block.forward = mocker.Mock(side_effect=lambda **kwargs: kwargs["hidden_states"])
    model = transformer_module.NucleusMoEImageTransformer2DModel.__new__(
        transformer_module.NucleusMoEImageTransformer2DModel
    )
    nn.Module.__init__(model)
    freqs = torch.ones(1, 1, dtype=torch.complex64)
    model.image_rope_prepare = mocker.Mock(return_value=(torch.ones(1, 1, 2), freqs, freqs))
    model.time_text_embed = mocker.Mock(return_value=torch.zeros(1, 2))
    model.txt_norm = mocker.Mock(side_effect=lambda source: scratch.copy_(source))
    model.transformer_blocks = nn.ModuleList([block])
    model.norm_out = mocker.Mock(side_effect=lambda hidden, temb: hidden)
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


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
def test_text_kv_matches_image_kv_sharding(monkeypatch, tp_size):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config

    monkeypatch.setattr(transformer_module, "Attention", lambda **kwargs: nn.Identity())
    head_dim, kv_heads, input_dim = 128, 4, 8
    weights = torch.arange(kv_heads * head_dim * input_dim, dtype=torch.float32).reshape(-1, input_dim) / 1000
    encoder = torch.arange(2 * input_dim, dtype=torch.float32).reshape(1, 2, input_dim) / 10
    for rank in range(tp_size):
        monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(world_size=tp_size, rank_in_group=rank))
        with set_current_vllm_config(VllmConfig()):
            attention = transformer_module.NucleusMoECrossAttention(
                dim=16 * head_dim,
                num_heads=16,
                head_dim=head_dim,
                num_kv_heads=kv_heads,
                joint_attention_dim=input_dim,
            )
        replicas = max(tp_size // kv_heads, 1)
        local_heads = max(kv_heads // tp_size, 1)
        shard_start = rank // replicas * local_heads * head_dim
        shard_size = local_heads * head_dim
        expected_weights = []
        for projection, full_weight in [(attention.add_k_proj, weights), (attention.add_v_proj, weights.flip(0))]:
            projection.weight.weight_loader(projection.weight, full_weight)
            expected_weights.append(full_weight[shard_start : shard_start + shard_size])
        block = transformer_module.NucleusMoEImageTransformerBlock.__new__(
            transformer_module.NucleusMoEImageTransformerBlock
        )
        nn.Module.__init__(block)
        block.attn = attention
        block.encoder_proj = nn.Identity()
        key, value = block._get_cached_text_kv(encoder)
        assert key.shape == value.shape == (1, 2, local_heads, head_dim)
        for projection, expected in zip([attention.add_k_proj, attention.add_v_proj], expected_weights):
            torch.testing.assert_close(projection.weight, expected)
        expected_key = torch.nn.functional.linear(encoder, expected_weights[0]).unflatten(-1, (local_heads, head_dim))
        expected_value = torch.nn.functional.linear(encoder, expected_weights[1]).unflatten(-1, (local_heads, head_dim))
        torch.testing.assert_close(key, attention.norm_added_k.forward_native(expected_key))
        torch.testing.assert_close(value, expected_value)


@pytest.mark.parametrize("scale_rope", [False, True])
def test_rope_grows_for_long_text_without_changing_existing_positions(scale_rope):
    rope = transformer_module.NucleusMoEEmbedRope(10000, [16, 56, 56], scale_rope=scale_rope)
    device = torch.device("cpu")
    old_pos, old_neg = rope.pos_freqs.clone(), rope.neg_freqs.clone()
    old_image, old_text = rope((1, 16, 16), [32], device)
    cached_image = rope._freq_cache[(1, 16, 16, 0)]
    offset = 8 if scale_rope else 16
    for length in [4096, 8192]:
        image, text = rope((1, 16, 16), [length], device)
        assert text.shape == (length, 64)
        expected = torch.cat(
            [rope._rope_params(torch.arange(offset, offset + length), dim, rope.theta) for dim in rope.axes_dim],
            dim=1,
        )
        torch.testing.assert_close(text, expected, rtol=0, atol=0)
        rotated = transformer_module.apply_rotary_emb_nucleus(torch.ones(1, length, 1, 128), text, use_real=False)
        assert rotated.shape == (1, length, 1, 128)
        assert torch.isfinite(rotated).all()
        torch.testing.assert_close(image, old_image, rtol=0, atol=0)
        torch.testing.assert_close(rope.pos_freqs[:4096], old_pos, rtol=0, atol=0)
        torch.testing.assert_close(rope.neg_freqs[-4096:], old_neg, rtol=0, atol=0)
        assert rope._freq_cache[(1, 16, 16, 0)] is cached_image
    image, text = rope((1, 16, 16), [32], device)
    torch.testing.assert_close(image, old_image, rtol=0, atol=0)
    torch.testing.assert_close(text, old_text, rtol=0, atol=0)


@pytest.mark.parametrize("scale_rope", [False, True])
def test_rope_grows_before_computing_large_image_frequencies(scale_rope):
    rope = transformer_module.NucleusMoEEmbedRope(10000, [16, 56, 56], scale_rope=scale_rope)
    image, text = rope((1, 1, 8200), [32], torch.device("cpu"))
    assert image.shape == (8200, 64)
    assert text.shape == (32, 64)
    assert torch.isfinite(image).all()


def test_rope_cache_does_not_outlive_module():
    rope = transformer_module.NucleusMoEEmbedRope(10000, [16, 56, 56], scale_rope=True)
    freqs = rope._compute_video_freqs(1, 2, 2)
    assert rope._compute_video_freqs(1, 2, 2) is freqs
    refs = [weakref.ref(value) for value in (rope, rope.pos_freqs, rope.neg_freqs, freqs)]

    del rope, freqs
    gc.collect()

    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("sequence_length", [8, 7])
@pytest.mark.parametrize("use_sigmoid", [False, True])
def test_sp_expert_choice_matches_unsharded_tokens(mocker, monkeypatch, sequence_length, use_sigmoid):
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
            group = SimpleNamespace(world_size=2, rank_in_group=rank, all_gather=mocker.Mock(return_value=scores))
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


@pytest.fixture
def mask_transformer(mocker):
    # Isolate the real forward's mask handling from model weights and device kernels.
    hidden = torch.zeros(1, 4, 8)
    block = mocker.Mock(return_value=hidden)
    block._get_cached_text_kv.return_value = (None, None)
    model = SimpleNamespace(
        image_rope_prepare=mocker.Mock(return_value=(hidden, torch.ones(4, 4), torch.ones(2, 4))),
        time_text_embed=mocker.Mock(return_value=torch.zeros(1, 8)),
        txt_norm=nn.Identity(),
        transformer_blocks=[block],
        norm_out=lambda x, temb: x,
        proj_out=nn.Identity(),
        od_config=SimpleNamespace(
            parallel_config=SimpleNamespace(sequence_parallel_size=2, ring_degree=1, mask_sp_padding=True)
        ),
    )
    return model, hidden, block


def _forward_mask_transformer(model, hidden):
    return transformer_module.NucleusMoEImageTransformer2DModel.forward(
        model, hidden, (1, 1, 7), torch.ones(1, 2, 8), timestep=torch.ones(1), return_dict=False
    )


@pytest.mark.parametrize("ring_degree", [1, 2])
@pytest.mark.parametrize("mask_sp_padding", [False, True])
def test_sp_padding_respects_mask_config(mocker, mask_transformer, ring_degree, mask_sp_padding):
    model, hidden, block = mask_transformer
    model.od_config.parallel_config.ring_degree = ring_degree
    model.od_config.parallel_config.mask_sp_padding = mask_sp_padding
    with override_forward_context(ForwardContext(sp_original_seq_len=7, sp_padding_size=1)):
        _forward_mask_transformer(model, hidden)
    mask = block.call_args.kwargs["hidden_states_mask"]
    if mask_sp_padding and ring_degree == 1:
        torch.testing.assert_close(mask, torch.tensor([[True] * 7 + [False]]))
    else:
        assert mask is None
        if ring_degree > 1:
            runner = mocker.Mock()
            runner.run_attention.return_value = hidden
            attention = SimpleNamespace(
                attn_backend=None,
                attention=SimpleNamespace(skip=None),
                ring_runner=runner,
                softmax_scale=1.0,
                causal=False,
            )
            output = Attention._run_ring_attention(attention, hidden, hidden, hidden, AttentionMetadata(attn_mask=mask))
            assert output is hidden


def test_sp_padding_propagates_context_errors(mask_transformer, monkeypatch):
    model, hidden, block = mask_transformer

    def broken_context():
        raise RuntimeError("invalid forward context")

    monkeypatch.setattr(transformer_module, "get_forward_context", broken_context)
    with override_forward_context(ForwardContext()), pytest.raises(RuntimeError, match="invalid forward context"):
        _forward_mask_transformer(model, hidden)
    block.assert_not_called()


def test_sp_padding_without_forward_context(mask_transformer):
    model, hidden, block = mask_transformer
    with override_forward_context(None):
        _forward_mask_transformer(model, hidden)
    assert block.call_args.kwargs["hidden_states_mask"] is None


@pytest.mark.parametrize("rank", [0, 1])
def test_ulysses_padding_mask_matches_gathered_keys(monkeypatch, rank):
    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(1, 8, 2, 4, generator=generator) for _ in range(3)]
    # A strong padded key/value must not contribute to any valid output.
    k[:, 7] = 1000
    v[:, 7] = 1000
    text_k, text_v = [torch.randn(1, 2, 2, 4, generator=generator) for _ in range(2)]
    group = SimpleNamespace(ulysses_group=None, ulysses_world_size=2, ulysses_rank=rank, ring_world_size=1)
    strategy = UlyssesParallelAttention(group, scatter_idx=2, gather_idx=1, use_sync=False)
    metadata = AttentionMetadata(
        attn_mask=torch.tensor([[True] * 7 + [False]]),
        joint_query=torch.zeros_like(text_k),
        joint_key=text_k,
        joint_value=text_v,
        joint_attn_mask=torch.ones(1, 2, dtype=torch.bool),
        joint_strategy="rear",
    )
    # Emulate only the collective; execute the real post-AllToAll metadata handling.
    gathered = iter([tensor[:, :, rank : rank + 1] for tensor in (q, k, v)])
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.parallel.ulysses.SeqAllToAll4D.apply", lambda *args: next(gathered)
    )
    with override_forward_context(None):
        gathered_q, gathered_k, gathered_v, metadata, _ = strategy.pre_attention(
            q[:, rank * 4 : (rank + 1) * 4],
            k[:, rank * 4 : (rank + 1) * 4],
            v[:, rank * 4 : (rank + 1) * 4],
            metadata,
        )
    assert metadata.attn_mask.shape[1] == gathered_k.shape[1] == 10
    output = torch.nn.functional.scaled_dot_product_attention(
        gathered_q.transpose(1, 2),
        gathered_k.transpose(1, 2),
        gathered_v.transpose(1, 2),
        attn_mask=metadata.attn_mask[:, None, None, :],
    )
    valid_k = torch.cat([k[:, :7, rank : rank + 1], text_k[:, :, rank : rank + 1]], dim=1)
    valid_v = torch.cat([v[:, :7, rank : rank + 1], text_v[:, :, rank : rank + 1]], dim=1)
    reference = torch.nn.functional.scaled_dot_product_attention(
        gathered_q.transpose(1, 2), valid_k.transpose(1, 2), valid_v.transpose(1, 2)
    )
    torch.testing.assert_close(output, reference)


@pytest.mark.parametrize(
    "text_mask",
    [None, [[True, True, True]], [[True, False, False]], [[True, False, False], [False, True, True]]],
)
@pytest.mark.parametrize("image_padding", [False, True])
def test_ring_attention_excludes_text_padding(monkeypatch, text_mask, image_padding):
    # Exercise Nucleus metadata, Ring dispatch and the Ring loop; substitute only
    # communication and the CUDA-only local kernel for a CPU reference kernel.
    monkeypatch.setattr(ring_pytorch_attn, "RingComm", lambda group: SimpleNamespace(rank=0, world_size=1))

    def local_kernel(query, key, value, softmax_scale=None, **kwargs):
        scores = query.transpose(1, 2) @ key.transpose(1, 2).transpose(-1, -2)
        scores = scores * (softmax_scale or query.shape[-1] ** -0.5)
        output = (scores.softmax(-1) @ value.transpose(1, 2)).transpose(1, 2)
        return output, scores.logsumexp(-1)

    monkeypatch.setattr(ring_pytorch_attn, "pytorch_attn_forward", local_kernel)
    strategy = RingParallelAttention(SimpleNamespace(ring_group=None), attn_backend_pref="sdpa")

    def attention(query, key, value, metadata):
        query, key, value, metadata, context = strategy.pre_attention(query, key, value, metadata)
        output = strategy.run_attention(query, key, value, metadata)
        return strategy.post_attention(output, context)

    batch_size = 1 if text_mask is None else len(text_mask)
    mask = torch.ones(batch_size, 3, dtype=torch.bool) if text_mask is None else torch.tensor(text_mask)
    hidden = torch.ones(batch_size, 2, 2)
    text_key = torch.zeros(batch_size, 3, 1, 2)
    text_value = torch.full_like(text_key, 1000.0)
    image_values = hidden.clone()
    if image_padding:
        hidden[:, 1:] = 0
        image_values[:, 1:] = 1000
    for row in range(batch_size):
        text_value[row, mask[row]] = 2.0 + row
    model = SimpleNamespace(
        num_heads=1,
        head_dim=2,
        num_kv_heads=1,
        num_kv_groups=1,
        to_qkv=lambda x: (torch.cat([x, x, image_values], dim=-1), None),
        norm_q=nn.Identity(),
        norm_k=nn.Identity(),
        parallel_config=SimpleNamespace(sequence_parallel_size=2, ring_degree=2, mask_sp_padding=True),
        attn=attention,
        to_out=nn.Identity(),
    )
    context = ForwardContext(
        omni_diffusion_config=SimpleNamespace(parallel_config=model.parallel_config),
        sp_plan_hooks_applied=True,
        _sp_shard_depth=1,
        sp_original_seq_len=1 if image_padding else None,
        sp_padding_size=int(image_padding),
    )
    with override_forward_context(context):
        output = transformer_module.NucleusMoECrossAttention.forward(
            model,
            hidden,
            encoder_hidden_states_mask=None if text_mask is None else mask,
            cached_txt_key=text_key,
            cached_txt_value=text_value,
        )
    reference = []
    image_length = 1 if image_padding else hidden.shape[1]
    for row in range(batch_size):
        valid_key = torch.cat([hidden[row : row + 1, :image_length], text_key[row : row + 1, mask[row], 0]], dim=1)
        valid_value = torch.cat(
            [image_values[row : row + 1, :image_length], text_value[row : row + 1, mask[row], 0]], dim=1
        )
        reference.append(
            torch.nn.functional.scaled_dot_product_attention(
                hidden[row : row + 1, None], valid_key[:, None], valid_value[:, None]
            )[:, 0]
        )
    torch.testing.assert_close(output, torch.cat(reference))


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 7, 63])
@pytest.mark.parametrize("hooks_applied", [False, True])
def test_ring_image_padding_preserves_sp_metadata_and_gathers_original_length(
    mocker, monkeypatch, degree, length, hooks_applied
):
    from vllm_omni.diffusion.distributed import parallel_state, sp_sharding
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig, SequenceParallelOutput
    from vllm_omni.diffusion.hooks import sequence_parallel

    config = SequenceParallelConfig(ring_degree=degree)
    hidden = torch.randn(2, length, 8)
    rope = torch.ones(length, 4, dtype=torch.complex64)
    text_rope = torch.ones(3, 4, dtype=torch.complex64)
    prepare = transformer_module.NucleusMoEImageRopePrepare(nn.Identity(), mocker.Mock(return_value=(rope, text_rope)))
    padded_length = (length + degree - 1) // degree * degree
    shard_length = padded_length // degree
    for rank in range(degree):
        ctx = ForwardContext(
            omni_diffusion_config=SimpleNamespace(parallel_config=config), sp_plan_hooks_applied=hooks_applied
        )
        monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: degree)
        monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda: rank)
        monkeypatch.setattr(sp_sharding, "get_sequence_parallel_world_size", lambda: degree)
        monkeypatch.setattr(sp_sharding, "get_sequence_parallel_rank", lambda: rank)
        split = sequence_parallel.SequenceParallelSplitHook(
            transformer_module.NucleusMoEImageTransformer2DModel._sp_plan["image_rope_prepare"], config
        )
        gather = sequence_parallel.SequenceParallelGatherHook(SequenceParallelOutput(gather_dim=1), config)
        with override_forward_context(ctx):
            padded, padded_rope, unchanged_text = prepare(hidden, (1, 1, length), [3])
            assert padded.shape[1] == padded_rope.shape[0] == padded_length
            torch.testing.assert_close(padded[:, :length], hidden)
            torch.testing.assert_close(padded_rope[:length], rope)
            assert torch.count_nonzero(padded[:, length:]) == 0
            assert unchanged_text is text_rope
            assert ctx.sp_original_seq_len == length
            assert ctx.sp_padding_size == padded_length - length
            local_image, local_rope, _ = split.post_forward(prepare, (padded, padded_rope, unchanged_text))
            assert ctx.sp_active and ctx.sp_rank_local_seq_lens_equal
            assert local_image.shape[1] == local_rope.shape[0] == shard_length
            assert ctx.sp_original_seq_len == length
            monkeypatch.setattr(sequence_parallel, "sp_gather", lambda *args, **kwargs: padded)
            output = gather.post_forward(nn.Identity(), local_image)
            torch.testing.assert_close(output, hidden)
            assert ctx._sp_shard_depth == 0 and not ctx._sp_equal_pad_stack
