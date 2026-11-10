# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qk_norm_matches_diffusers_rounding(monkeypatch, device, dtype):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    attention = transformer.QwenImage21Attention(dim=256, heads=2, dim_head=128).to(device=device, dtype=dtype)
    values = torch.randn(2, 17, 2, 128, generator=torch.Generator().manual_seed(42)).to(device=device, dtype=dtype)
    for norm in (attention.norm_q, attention.norm_k):
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 128, device=device, dtype=dtype))
        variance = values.float().square().mean(-1, keepdim=True)
        expected = (values * torch.rsqrt(variance + norm.eps)).to(dtype) * norm.weight
        torch.testing.assert_close(norm(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("meta", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)),
    ],
)
def test_rope_frequencies_do_not_depend_on_loader_device(device):
    reference = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    with torch.device(device):
        loaded = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    for expected, actual in zip(reference.freqs, loaded.freqs):
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("valid_lengths", [(13, 90), (10, 30)], ids=["positive", "negative"])
@pytest.mark.parametrize(
    "row_index",
    [
        pytest.param(
            0,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="Padded rows change image RoPE; tracked in diffusion_runtime.md per-request Qwen RoPE follow-up",
            ),
            id="padded-row",
        ),
        pytest.param(1, id="unpadded-control"),
    ],
)
def test_batched_rope_preserves_positions_from_each_requests_text_mask(valid_lengths, row_index):
    """Helper-level tracking check for positions derived from each text mask.

    This intentionally exercises ``QwenImage21Rope`` directly, rather than the
    full batched Transformer path. A row-aware RoPE fix will change this
    helper's signature; rewrite this strict xfail before removing it.
    """
    rope = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    device = torch.device("cpu")
    padded_length = max(valid_lengths)
    prompt_embeds_mask = torch.arange(padded_length).unsqueeze(0) < torch.tensor(valid_lengths).unsqueeze(1)
    target_tokens = 4
    image_shapes = [(1, 2, 2)]
    batched_image_mask = torch.cat(
        [torch.zeros(padded_length, dtype=torch.bool), torch.ones(target_tokens, dtype=torch.bool)]
    )
    batched_freqs = rope(image_shapes, batched_image_mask, device)

    # Standalone image positions advance only over this request's valid text.
    valid_length = int(prompt_embeds_mask[row_index].sum())
    standalone_image_mask = torch.cat(
        [torch.zeros(valid_length, dtype=torch.bool), torch.ones(target_tokens, dtype=torch.bool)]
    )
    standalone_freqs = rope(image_shapes, standalone_image_mask, device)
    torch.testing.assert_close(
        batched_freqs[:padded_length][prompt_embeds_mask[row_index]], standalone_freqs[:valid_length], rtol=0, atol=0
    )
    torch.testing.assert_close(batched_freqs[-target_tokens:], standalone_freqs[-target_tokens:], rtol=0, atol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_scheduler_matches_diffusers_on_latent_device():
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

    pipeline = object.__new__(QwenImage21Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda", 0)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    timesteps, _ = pipeline.prepare_timesteps(50, None, 4096)
    reference = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    reference.set_timesteps(sigmas=np.linspace(1.0, 1 / 50, 50), device=pipeline.device, mu=1.15)
    sample = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(42)).to(
        device=pipeline.device, dtype=torch.bfloat16
    )
    noise = sample.cos()
    actual = pipeline.scheduler.step(noise, timesteps[0], sample, return_dict=False)[0]
    expected = reference.step(noise, reference.timesteps[0], sample, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pipeline.scheduler.sigmas.device == sample.device
    assert timesteps.device == sample.device


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_timestep_frequencies_match_cpu_initialization():
    reference = transformer.QwenImage21TemporalTimesteps(timestep_dim=256)
    with torch.device("cuda"):
        loaded = transformer.QwenImage21TemporalTimesteps(timestep_dim=256)
    assert loaded.freqs.device.type == "cuda"
    torch.testing.assert_close(loaded.freqs.cpu(), reference.freqs, rtol=0, atol=0)
    timesteps = torch.tensor([0.0, 0.984375, 1.0], device="cuda")
    torch.testing.assert_close(loaded(timesteps), reference.to("cuda")(timesteps), rtol=0, atol=0)
