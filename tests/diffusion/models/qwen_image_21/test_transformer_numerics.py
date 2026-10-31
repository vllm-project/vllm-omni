# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "device,attention_device",
    [
        pytest.param("cpu", "cpu", marks=pytest.mark.cpu),
        pytest.param("cpu", "npu", marks=pytest.mark.cpu),
        pytest.param("cuda", "cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("attn_path", [None, "auto", "piecewise", "mask", "flex"])
def test_prefill_attention_metadata(monkeypatch, device, attention_device, attn_path):
    seen = []

    class CaptureBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(
                attn=SimpleNamespace(
                    attn_backend=SimpleNamespace(supports_piecewise_spans=True, get_name=lambda: "FLASH_ATTN")
                )
            )

        def forward(self, hidden_states, attn_metadata, **kwargs):
            seen.append(attn_metadata)
            return hidden_states

    class SequencePrepare:
        def __call__(self, hidden_states, text, layout, img_mask, is_decode):
            return text, hidden_states, text.new_zeros(2, 1), hidden_states.new_zeros(2, 1)

        @staticmethod
        def expand_image_pad_mask(mask):
            return mask

    # Exercise the real model forward and metadata selection without weights.
    model = object.__new__(transformer.QwenImage21Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.parallel_config = None
    model.causal_block = True
    model.causal_condition = False
    model.txt_in = torch.nn.Identity()
    model.sequence_prepare = SequencePrepare()
    model.time_text_embed = lambda timestep, hidden_states: timestep
    model.modulation = torch.nn.Identity()
    model.transformer_blocks = torch.nn.ModuleList([CaptureBlock()])
    model.norm_out = lambda hidden_states, *args: hidden_states
    model.proj_out = torch.nn.Identity()

    if attention_device == "npu":
        # Exercise NPU routing with CPU tensors and no model weights.
        resolve_attn_path = model._resolve_attn_path
        monkeypatch.setattr(
            model,
            "_resolve_attn_path",
            lambda hint, padding, sp_padding, device_type: resolve_attn_path(hint, padding, sp_padding, "npu"),
        )

    hidden = torch.ones(1, 2, 4, device=device)
    text = torch.zeros_like(hidden)
    inputs = dict(
        hidden_states=hidden,
        encoder_hidden_states=text,
        timestep=hidden.new_zeros(1),
        img_shapes=[[(1, 1, 2)]],
        img_mask=torch.tensor([[False, False, True, True]], device=device),
        attention_kwargs=None if attn_path is None else {"attn_path": attn_path},
        return_dict=False,
    )
    if attn_path == "flex" and attention_device != "npu":
        with pytest.raises(ValueError, match="unpadded NPU prefill"):
            model(**inputs)
        return
    output = model(**inputs)[0]
    assert len(seen) == 1

    for metadata in seen:
        if attn_path == "mask":
            assert metadata.full_attn_spans is None
            expected = torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
                device=device,
            )[None, None]
            torch.testing.assert_close(metadata.attn_mask, expected)
        else:
            assert metadata.full_attn_spans == [[(2, 4)]]
            assert metadata.attn_mask is None
        if attn_path == "flex":
            assert metadata.extra == {"npu_flex_attention": True, "_qwen_image21_flex_cache": {}}
        else:
            assert metadata.extra == {}
    torch.testing.assert_close(output, torch.cat([text, hidden], dim=1), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "hint,padding,sp_padding,supports_spans,backend_name,expected",
    [
        ("auto", False, False, True, "FLASH_ATTN", "piecewise"),
        ("auto", True, False, True, "FLASH_ATTN", "mask"),
        ("auto", False, True, True, "FLASH_ATTN", "mask"),
        ("auto", False, False, False, "SDPA", "mask"),
        ("mask", False, False, True, "FLASH_ATTN", "mask"),
        ("flex", False, False, True, "FLASH_ATTN", "flex"),
        ("flex", True, False, True, "FLASH_ATTN", None),
        ("flex", False, True, True, "FLASH_ATTN", None),
        ("flex", False, False, True, "SDPA", None),
        ("invalid", False, False, True, "FLASH_ATTN", None),
    ],
)
def test_npu_prefill_path_contract(hint, padding, sp_padding, supports_spans, backend_name, expected):
    model = object.__new__(transformer.QwenImage21Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.causal_block = True
    backend = SimpleNamespace(supports_piecewise_spans=supports_spans, get_name=lambda: backend_name)
    model.transformer_blocks = [SimpleNamespace(attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend)))]
    if expected is None:
        with pytest.raises(ValueError):
            model._resolve_attn_path(hint, padding, sp_padding, "npu")
    else:
        assert model._resolve_attn_path(hint, padding, sp_padding, "npu") == expected


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
