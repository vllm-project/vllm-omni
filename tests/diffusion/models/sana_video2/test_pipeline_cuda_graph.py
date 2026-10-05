# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.sana_video2.test_cuda_graph import tiny_model
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]


class ImageEncoder(nn.Module):
    config = SimpleNamespace(
        latent_channels=128, temporal_compression_ratio=8, spatial_compression_ratio=32, scaling_factor=1.0
    )
    dtype = torch.bfloat16

    def __init__(self):
        super().__init__()
        self.register_buffer("latents_mean", torch.zeros(128, device="cuda"))
        self.register_buffer("latents_std", torch.ones(128, device="cuda"))

    def encode(self, pixels):
        latent = pixels.new_full((1, 128, 1, pixels.shape[-2] // 32, pixels.shape[-1] // 32), 0.25)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


def components():
    model = tiny_model(torch.bfloat16, in_channels=128)
    return object(), nn.Identity(), ImageEncoder(), model


def pipeline():
    tokenizer, text, vae, transformer = components()
    return SanaVideo2Pipeline(tokenizer=tokenizer, text_encoder=text, vae=vae, transformer=transformer)


@pytest.mark.parametrize("task", ["t2v", "ti2v"])
@pytest.mark.parametrize("guidance", [1.0, 8.0])
def test_pipeline_graph_matches_eager_trajectory(task, guidance):
    model = pipeline()
    shape = dict(height=64, width=96, num_frames=9, task=task, guidance_scale=guidance)
    torch.manual_seed(7)
    kwargs = dict(
        height=64,
        width=96,
        num_frames=9,
        num_inference_steps=5,
        guidance_scale=guidance,
        latents=torch.randn(1, 128, 2, 2, 3, device="cuda"),
        prompt_embeds=torch.randn(1, 300, 12, device="cuda", dtype=torch.bfloat16),
        prompt_attention_mask=torch.arange(300, device="cuda")[None] < 231,
        negative_prompt_embeds=torch.randn(1, 300, 12, device="cuda", dtype=torch.bfloat16),
        negative_prompt_attention_mask=torch.arange(300, device="cuda")[None] < 71,
        image=Image.new("RGB", (96, 64)) if task == "ti2v" else None,
        output_type="latent",
    )
    expected, actual = [], []
    reference = model.generate(**kwargs, callback=lambda i, t, x: expected.append(x.clone()))
    model.prepare_cuda_graphs([shape])
    calls = []
    handle = model.transformer.x_embedder.register_forward_hook(lambda *args: calls.append(1))
    try:
        result = model.generate(**kwargs, callback=lambda i, t, x: actual.append(x.clone()))
    finally:
        handle.remove()
    assert calls == []
    assert torch.equal(result, reference)
    for result_step, reference_step in zip(actual, expected, strict=True):
        assert torch.equal(result_step, reference_step)
        if task == "ti2v":
            assert torch.equal(result_step[:, :, 0], torch.full_like(result_step[:, :, 0], 0.25))


@pytest.mark.parametrize("eager,configured", [(True, True), (False, True), (False, False)])
def test_configured_setup_honors_enforce_eager(monkeypatch, eager, configured):
    loaded = components()
    config = OmniDiffusionConfig(
        model_config={"cuda_graph_shapes": [dict(height=64, width=96, num_frames=9)]} if configured else {},
        enforce_eager=eager,
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=True,
    )
    monkeypatch.setattr(SanaVideo2Pipeline, "_load_components", lambda self, config: loaded)
    model = SanaVideo2Pipeline(od_config=config)
    model.setup_compile()
    runner = model._cuda_graph_runner
    assert (runner is None) == (eager or not configured)
    if runner is not None:
        assert len(runner.entries) == 1
    assert set(model.state_dict()) == set(
        SanaVideo2Pipeline(
            tokenizer=loaded[0], text_encoder=loaded[1], vae=loaded[2], transformer=loaded[3]
        ).state_dict()
    )


def test_prepare_replaces_previous_graphs_and_empty_shapes_restore_eager():
    model = pipeline()
    first = dict(height=64, width=96, num_frames=9)
    second = dict(height=96, width=64, num_frames=9)
    model.prepare_cuda_graphs([first])
    old_signature = next(iter(model._cuda_graph_runner.entries))
    model.prepare_cuda_graphs([second])
    assert old_signature not in model._cuda_graph_runner.entries
    model.prepare_cuda_graphs([])
    calls = []
    handle = model.transformer.x_embedder.register_forward_hook(lambda *args: calls.append(1))
    try:
        kwargs = dict(
            hidden_states=torch.randn(2, 128, 2, 2, 3, device="cuda"),
            timestep=torch.full((2,), 300.0, device="cuda"),
            encoder_hidden_states=torch.randn(2, 300, 12, device="cuda", dtype=torch.bfloat16),
            encoder_attention_mask=torch.ones(2, 300, device="cuda", dtype=torch.bool),
        )
        output = model._cuda_graph_runner(**kwargs)
        assert calls == [1]
        assert torch.equal(output, model.transformer(**kwargs))
        assert model._cuda_graph_runner.entries == {}
    finally:
        handle.remove()
