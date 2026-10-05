# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import (
    SanaVideo2Pipeline,
    validate_parallel_config,
)

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


def _config(degree=1, mode="strict", **overrides):
    parallel = dict(
        tensor_parallel_size=1,
        sequence_parallel_size=degree,
        ulysses_degree=degree,
        ulysses_mode=mode,
        ring_degree=1,
        allgather_degree=1,
        cfg_parallel_size=1,
        use_hsdp=False,
    )
    parallel.update(overrides)
    return SimpleNamespace(parallel_config=SimpleNamespace(**parallel), cache_backend=None)


@pytest.mark.parametrize(
    "degree,mode", [(1, "strict"), (2, "strict"), (2, "advanced_uaa"), (4, "advanced_uaa"), (8, "advanced_uaa")]
)
def test_accepts_pure_ulysses(degree, mode):
    validate_parallel_config(_config(degree, mode))


@pytest.mark.parametrize("degree", [4, 8])
def test_rejects_strict_nondivisible_anchor_heads(degree):
    with pytest.raises(ValueError, match="advanced_uaa"):
        validate_parallel_config(_config(degree))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_parallel_size": 4}, "tensor_parallel_size"),
        ({"cfg_parallel_size": 4}, "cfg_parallel_size"),
        ({"ring_degree": 2}, "ring_degree"),
        ({"allgather_degree": 2}, "allgather_degree"),
        ({"ulysses_degree": 1}, "ulysses_degree"),
        ({"sequence_parallel_size": 3, "ulysses_degree": 3}, "sequence_parallel_size"),
    ],
)
def test_rejects_unsupported_topology(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_parallel_config(_config(2, **overrides))


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.group = None
        self.calls = []

    def set_sequence_parallel(self, group):
        self.group = group

    def forward(self, x, timestep, embeddings, mask):
        self.calls.append((x.clone(), timestep.clone(), embeddings.clone(), mask.clone()))
        return torch.zeros_like(x)


def _components():
    vae = nn.Module()
    vae.config = SimpleNamespace(latent_channels=128, temporal_compression_ratio=8, spatial_compression_ratio=32)
    return object(), nn.Identity(), vae, _Transformer()


def test_pipeline_enables_model_sp_and_synchronizes_conditioning(monkeypatch):
    components = _components()
    broadcasts = []
    group = SimpleNamespace(broadcast=lambda x: broadcasts.append(x.clone()) or x)
    monkeypatch.setattr(SanaVideo2Pipeline, "_load_components", lambda *_: components)
    monkeypatch.setattr("vllm_omni.diffusion.distributed.parallel_state.get_sp_group", lambda: group)
    pipeline = SanaVideo2Pipeline(od_config=_config(2))
    assert components[-1].group is group
    latents = torch.arange(128 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 128, 2, 2, 3) / 1000
    positive, negative = torch.ones(1, 3, 4), -torch.ones(1, 3, 4)
    mask = torch.tensor([[True, True, False]])
    pipeline.generate(
        height=64,
        width=96,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=8,
        latents=latents,
        prompt_embeds=positive,
        negative_prompt_embeds=negative,
        prompt_attention_mask=mask,
        negative_prompt_attention_mask=mask,
        output_type="latent",
    )
    assert len(broadcasts) == 3
    assert any(torch.equal(x, latents) for x in broadcasts)
    assert any(torch.equal(x, torch.cat([negative, positive])) for x in broadcasts)
    assert any(torch.equal(x, torch.cat([mask, mask])) for x in broadcasts)
    first_x, _, first_embeddings, _ = components[-1].calls[0]
    torch.testing.assert_close(first_x[0], first_x[1], rtol=0, atol=0)
    torch.testing.assert_close(first_embeddings, torch.cat([negative, positive]), rtol=0, atol=0)


def test_single_rank_pipeline_keeps_native_transformer():
    tokenizer, text, vae, transformer = _components()
    SanaVideo2Pipeline(tokenizer=tokenizer, text_encoder=text, vae=vae, transformer=transformer)
    assert transformer.group is None


def test_forward_routes_warmup_and_consecutive_request_parameters(monkeypatch):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    tokenizer, text, vae, transformer = _components()
    pipeline = SanaVideo2Pipeline(tokenizer=tokenizer, text_encoder=text, vae=vae, transformer=transformer)
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1)

    monkeypatch.setattr(pipeline, "generate", generate)
    for dummy, height, width, frames in [(True, 512, 512, 9), (False, 64, 96, 9), (False, 96, 64, 17)]:
        request = SimpleNamespace(
            prompts=[{"prompt": "boat"}],
            is_dummy_run=lambda: dummy,
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=1 if dummy else 5,
                seed=42,
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0} if dummy else {},
            ),
        )
        pipeline.forward(request)
    assert [(c["height"], c["width"], c["num_frames"], c["num_inference_steps"]) for c in calls] == [
        (512, 512, 9, 2),
        (64, 96, 9, 5),
        (96, 64, 17, 5),
    ]
