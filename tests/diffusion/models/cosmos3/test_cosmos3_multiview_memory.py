# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import MaskItem, MultiviewLayout
from vllm_omni.diffusion.models.cosmos3.multiview_packing import pack_state, unpack_state
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer
from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_single_sensor_packing_avoids_redundant_copy(noncontiguous):
    tensor = torch.arange(96).reshape(2, 3, 4, 2, 2)
    if noncontiguous:
        tensor = tensor.transpose(2, 3)
    packed = pack_state([tensor])
    torch.testing.assert_close(unpack_state(packed, (tensor.shape[1:],))[0], tensor)
    if not noncontiguous:
        assert packed.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize("shard_input", [False, True])
@torch.inference_mode()
def test_packed_embeddings_and_caption_temporaries_released_before_later_layers(shard_input):
    model = object.__new__(Cosmos3MultiviewVFMTransformer)
    nn.Module.__init__(model)
    model.lidar_config = None
    model.latent_patch_size = 1
    model.temporal_compression_factor = 4
    model.base_fps = 30
    model.enable_fps_modulation = True
    model.temporal_modality_margin = 10
    model.cached_kv = model.cached_freqs_gen = None
    model._multiview_mask_cache, model._multiview_buffer_cache = {}, {}
    model._offload_context = lambda _: nullcontext()
    model.proj_in = nn.Identity()
    model._embed_timestep = lambda t, dtype: torch.zeros(1, 2, dtype=dtype)
    model._project_video_tokens = lambda hidden: hidden
    model.gen_sp_gather = nn.Identity()
    caption_refs, embedding_refs, visited = [], [], []

    class Language(nn.Module):
        def rotary_emb(self, dummy, position_ids):
            return torch.ones(1, 1, 2), torch.zeros(1, 1, 2)

        def forward(self, ids, freqs):
            keys = [torch.ones(1, ids.shape[1], 1, 2) for _ in range(4)]
            caption_refs.extend(weakref.ref(key) for key in keys)
            return [(keys[0], keys[1]), (keys[2], keys[3])]

    class Layer(nn.Module):
        def forward(self, hidden, **kwargs):
            assert all(ref() is None for ref in caption_refs)
            if visited or shard_input:
                assert embedding_refs[0]() is None
            visited.append(True)
            return hidden + 1

    embed = model._embed_packed_streams

    def record_embeddings(*args):
        output = embed(*args)
        embedding_refs.append(weakref.ref(output))
        return output

    def prepare(hidden, cos, sin):
        # Simulate the SP hook's allocation without distributed hardware.
        return (hidden.clone() if shard_input else hidden), cos, sin

    model._embed_packed_streams = record_embeddings
    model.gen_sp_prepare = prepare
    model.language_model = Language()
    model.gen_layers = nn.ModuleList([Layer(), Layer()])
    camera = torch.zeros(1, 2, 4, 1, 1)
    output = model(
        hidden_states=pack_state([camera]),
        timestep=torch.ones(1),
        text_ids=torch.ones(1, 5, dtype=torch.long),
        text_mask=torch.ones(1, 5, dtype=torch.long),
        caption_lengths=(2, 3),
        packed_shapes=(tuple(camera.shape[1:]),),
        multiview_layout=MultiviewLayout(items=(MaskItem((4, 1, 1), 2),)),
    )
    assert len(visited) == 2
    torch.testing.assert_close(output, torch.full_like(output, 2))
    torch.testing.assert_close(camera, torch.zeros_like(camera))


def _pipeline(scheduler, pipeline_cls=Cosmos3MultiviewPipeline):
    pipeline = object.__new__(pipeline_cls)
    nn.Module.__init__(pipeline)
    pipeline.scheduler = scheduler
    pipeline.transformer = SimpleNamespace(reset_cache=lambda: None)
    pipeline.progress_bar = lambda steps: steps
    pipeline._set_mixed_precision_step = lambda *args: None
    pipeline._reset_mixed_precision = lambda: None
    return pipeline


@pytest.mark.parametrize("scheduler_cls", [FlowUniPCMultistepScheduler, FlowMatchEulerDiscreteScheduler])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fractional_mask", [False, True])
@torch.inference_mode()
def test_compact_conditioning_matches_expanded_masks_and_preserves_solver_history(
    scheduler_cls, joint, dtype, fractional_mask
):
    generator = torch.Generator().manual_seed(42)
    camera = torch.randn(2, 3, 4, 2, 2, generator=generator, dtype=dtype)
    targets = [camera]
    if joint:
        targets.append(torch.randn(2, 2, 3, 1, 2, generator=generator, dtype=dtype))
    shapes = tuple(tuple(target.shape[1:]) for target in targets)
    mask = torch.tensor([0, 1, 0.25 if fractional_mask else 0, 1], dtype=dtype).reshape(1, 1, 4, 1, 1)
    condition = torch.randn(camera.shape, generator=generator, dtype=dtype)
    expanded_mask = pack_state([mask.expand_as(camera), *[torch.ones_like(t) for t in targets[1:]]])
    expanded_condition = pack_state([condition, *[torch.zeros_like(t) for t in targets[1:]]])
    actual, expected = pack_state(targets), pack_state(targets).clone()
    scheduler, reference = scheduler_cls(), scheduler_cls()
    scheduler.set_timesteps(5, device="cpu")
    reference.set_timesteps(5, device="cpu")
    pipeline = _pipeline(scheduler)
    for timestep in scheduler.timesteps:
        prediction = (actual * 0.125).sin()
        expected_prediction = (expected * 0.125).sin() * expanded_mask
        masked = pipeline._mask_transfer_noise(prediction, mask, {"packed_shapes": shapes})
        assert masked is prediction
        torch.testing.assert_close(masked, expected_prediction, rtol=0, atol=0)
        expected = reference.step(expected_prediction, timestep, expected, return_dict=False)[0]
        expected = expanded_mask * expected + (1 - expanded_mask) * expanded_condition
        actual = scheduler.step(masked, timestep, actual, return_dict=False)[0]
        history = (
            [t for t in [*scheduler.model_outputs, scheduler.last_sample] if t is not None]
            if isinstance(scheduler, FlowUniPCMultistepScheduler)
            else []
        )
        snapshots = [t.clone() for t in history]
        restored = pipeline._apply_transfer_condition(actual, mask, condition, {"packed_shapes": shapes})
        assert restored is actual
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for tensor, snapshot in zip(history, snapshots, strict=True):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("fail_step", [None, 1])
@pytest.mark.parametrize("text_cfg,control_cfg", [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)])
@pytest.mark.parametrize("pipeline_cls", [Cosmos3MultiviewPipeline, Cosmos3OmniDiffusersPipeline])
@torch.inference_mode()
def test_transfer_releases_predictions_and_completed_or_failed_scheduler_history(
    fail_step, text_cfg, control_cfg, pipeline_cls
):
    scheduler = FlowUniPCMultistepScheduler()
    pipeline = _pipeline(scheduler, pipeline_cls)
    original_step = scheduler.step
    noise_refs, history_refs = [], []

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        history_refs.extend(weakref.ref(t) for t in [*scheduler.model_outputs, scheduler.last_sample] if t is not None)
        return result

    scheduler.step = step
    condition = torch.full((1, 2, 4, 1, 1), 0.5)
    mask = torch.tensor([0, 1, 0, 1]).reshape(1, 1, 4, 1, 1)
    if pipeline_cls is Cosmos3OmniDiffusersPipeline:
        mask = mask.expand_as(condition).flatten(1)
        condition = condition.flatten(1)
    for _ in range(2):  # A completed or failed request must allow clean reuse.
        scheduler.set_timesteps(4, device="cpu")
        step_count = 0

        def predict(**kwargs):
            nonlocal step_count
            assert all(ref() is None for ref in noise_refs)
            if fail_step == step_count:
                raise RuntimeError("prediction failed")
            step_count += 1
            noise = kwargs["hidden_states"] * 0.125
            noise_refs.append(weakref.ref(noise))
            return noise

        pipeline.predict_noise = predict
        pipeline.predict_noise_with_multi_branch_cfg = lambda *, branches_kwargs, **kwargs: predict(
            **branches_kwargs[0]
        )
        with pytest.raises(RuntimeError, match="prediction failed") if fail_step is not None else nullcontext():
            output = pipeline.diffuse_transfer(
                latents=torch.ones(1, 8),
                timesteps=scheduler.timesteps,
                cond_ids=torch.ones(1, 2),
                cond_mask=torch.ones(1, 2),
                uncond_ids=torch.ones(1, 2),
                uncond_mask=torch.ones(1, 2),
                guidance_scale=text_cfg,
                control_guidance=control_cfg,
                control_guidance_interval=None,
                control_latents=[],
                shared_kwargs={"packed_shapes": ((2, 4, 1, 1),)},
                velocity_mask=mask,
                condition_latents=condition,
            )
            assert torch.isfinite(output).all()
            assert output.reshape(1, 2, 4, 1, 1)[:, :, ::2].eq(0.5).all()
        assert scheduler.model_outputs == [None] * scheduler.config.solver_order
        assert scheduler.last_sample is None
        assert all(ref() is None for ref in history_refs)
        assert pipeline._cosmos3_branch_caches is None
