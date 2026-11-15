# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-transformer parity across text padding and batch composition changes."""

import copy

import pytest
import torch
from vllm.model_executor.layers.utils import default_unquantized_gemm

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.mossimage import mossimage_transformer as model
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline, MOSSImageScheduler
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def make_transformer(config: OmniDiffusionConfig, device: torch.device, dtype: torch.dtype):
    with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        transformer = model.MOSSImageTransformer(
            od_config=config, dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=8, freq_dim=8, latent_channels=4
        )
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for name, parameter in transformer.named_parameters():
            if parameter.ndim > 1:
                torch.nn.init.normal_(parameter, std=0.15)
            else:
                parameter.fill_(0 if name.endswith("bias") else 1)
    return transformer.to(device=device, dtype=dtype).eval()


@pytest.fixture
def transformer(monkeypatch, mocker):
    # Keep real projections, attention, refinement, RoPE, and denoising blocks.
    # Only the single-rank TP environment and CPU GEMM dispatch are substituted.
    tp_group = mocker.Mock(rank_in_group=0, world_size=1)
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp_group)
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    monkeypatch.setattr(model, "get_sequence_parallel_world_size", lambda: 1)
    config = OmniDiffusionConfig(
        dtype=torch.float32,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    tiny = make_transformer(config, torch.device("cpu"), torch.float32)
    with set_forward_context(omni_diffusion_config=config):
        yield tiny


def make_inputs(lengths: tuple[int, ...]):
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn(len(lengths), 4, 2, 2, generator=generator)
    text = torch.randn(len(lengths), max(lengths), 8, generator=generator)
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    text = text.masked_fill(~mask[..., None], 0)
    return latents, text, mask


@pytest.mark.parametrize("lengths", [(3, 4), (4, 3), (3, 5)])
@pytest.mark.parametrize("use_cache", [False, True])
@torch.no_grad()
def test_unequal_lengths_match_individual_requests(transformer, lengths, use_cache):
    latents, text, mask = make_inputs(lengths)
    expected = torch.cat(
        [transformer(latents[i : i + 1], 500.0, text[i : i + 1, :length]).sample for i, length in enumerate(lengths)]
    )
    cache: dict | None = {} if use_cache else None
    for _ in range(2 if use_cache else 1):
        actual = transformer(latents, 500.0, text, mask, _conditioning_cache=cache).sample
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@torch.no_grad()
def test_sp_alignment_padding_keeps_unpadded_positions(transformer, monkeypatch):
    # Four image tokens + three text tokens require one extra slot for SP=2.
    latents, text, mask = make_inputs((3,))
    expected = transformer(latents, 500.0, text, mask).sample
    monkeypatch.setattr(model, "get_sequence_parallel_world_size", lambda: 2)
    actual = transformer(latents, 500.0, text, mask).sample
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@torch.no_grad()
def test_continuous_batch_joins_and_leaves_match_isolated_requests(transformer):
    pipeline = MOSSImagePipeline.__new__(MOSSImagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = transformer
    pipeline._interrupt = False
    lengths = (3, 4, 5)
    latents, text, mask = make_inputs(lengths)
    states = []
    for i, length in enumerate(lengths):
        scheduler = MOSSImageScheduler(num_train_timesteps=64)
        scheduler.set_timesteps(6)
        states.append(
            StepRequestState(
                request_id=str(i),
                sampling=OmniDiffusionSamplingParams(guidance_scale=1.0, true_cfg_scale=1.0),
                latents=latents[i : i + 1].clone(),
                prompt_embeds=text[i : i + 1, :length],
                prompt_embeds_mask=mask[i : i + 1, :length],
                timesteps=scheduler.timesteps,
                scheduler=scheduler,
                do_true_cfg=False,
            )
        )
    isolated = copy.deepcopy(states)
    cached_batch = None
    for active in ((0,), (0, 1), (1, 0, 2), (0, 2), (0,)):
        selected = [states[i] for i in active]
        cached_batch = InputBatch.make_batch(selected, cached_batch=cached_batch)
        actual = pipeline.denoise_step(cached_batch)
        for row, i in enumerate(active):
            expected = pipeline.denoise_step(InputBatch.make_batch([isolated[i]]))
            torch.testing.assert_close(actual[row : row + 1], expected, atol=2e-6, rtol=2e-5)
            pipeline.step_scheduler(states[i], actual[row : row + 1])
            pipeline.step_scheduler(isolated[i], expected)
            torch.testing.assert_close(states[i].latents, isolated[i].latents, atol=2e-6, rtol=2e-5)
