# SPDX-License-Identifier: Apache-2.0
"""Small CPU contract suite for Irodori loading and optimized step execution."""

import json
import types

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.irodori_tts import packed_attention
from vllm_omni.diffusion.models.irodori_tts.batching import IrodoriDenoiseBatch
from vllm_omni.diffusion.models.irodori_tts.config import ModelConfig, read_irodori_checkpoint_config
from vllm_omni.diffusion.models.irodori_tts.model import TextToLatentRFDiT
from vllm_omni.diffusion.models.irodori_tts.precision import (
    IEEE,
    REFERENCE_POLICY,
    TF32,
    TRAINED_POLICY,
)
from vllm_omni.diffusion.models.irodori_tts.sampler import (
    run_packed_euler_rf_cfg_step,
    run_packed_varlen_euler_rf_cfg_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sm120_unsafe_fa4_uses_flashinfer(monkeypatch):
    monkeypatch.setattr(packed_attention, "HAS_FLASHINFER", True)
    monkeypatch.setattr(packed_attention, "flash_attn_varlen_func", lambda: None)
    monkeypatch.setattr(packed_attention, "_has_unsafe_sm120_fa4", lambda _device: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    assert packed_attention.resolve_packed_attention_backend(torch.device("cuda"), torch.bfloat16) == "flashinfer"


def test_supported_flash_attention_remains_preferred(monkeypatch):
    monkeypatch.setattr(packed_attention, "HAS_FLASHINFER", True)
    monkeypatch.setattr(packed_attention, "flash_attn_varlen_func", lambda: None)
    monkeypatch.setattr(packed_attention, "_has_unsafe_sm120_fa4", lambda _device: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    assert packed_attention.resolve_packed_attention_backend(torch.device("cuda"), torch.bfloat16) == "flash-attn"


def test_checkpoint_metadata_and_precision_contract(tmp_path):
    model = ModelConfig(
        latent_dim=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        text_dim=8,
        text_layers=1,
        text_heads=2,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        duration_hidden_dim=8,
        duration_layers=1,
        duration_attention_heads=2,
    )
    config = {field: getattr(model, field) for field in model.__dataclass_fields__}
    path = tmp_path / "model.safetensors"
    save_file(
        {"weight": torch.zeros(1)},
        str(path),
        metadata={
            "config_json": json.dumps(config),
            "text_encoder_config_json": json.dumps({"model_type": "modernbert"}),
        },
    )
    assert read_irodori_checkpoint_config(path).model.model_dim == 8
    assert REFERENCE_POLICY.dit_matmul == IEEE
    assert TRAINED_POLICY.dit_matmul == TF32
    assert TRAINED_POLICY.condition_matmul == IEEE
    assert TRAINED_POLICY.attention_dtype is torch.bfloat16
    assert TextToLatentRFDiT._repeated_blocks == ["DiffusionBlock"]


class _PaddedDiT:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.row_counts = []

    def forward_with_encoded_conditions(self, *, x_t, t, text_state, **_):
        self.row_counts.append(x_t.shape[0])
        marker = text_state[:, :1, :1]
        return x_t * 2 + marker + t.reshape(-1, 1, 1)


def _padded_batch(*, refresh: bool, correction=None):
    requests, cfg_rows = 2, 3
    text = torch.arange(requests * cfg_rows, dtype=torch.float32).reshape(-1, 1, 1).expand(-1, 2, 2)
    latents = torch.arange(16, dtype=torch.float32).reshape(requests, 4, 2)
    return IrodoriDenoiseBatch(
        request_ids=("a", "b"),
        cfg_active=True,
        cfg_layout=("cond", "text", "speaker"),
        latents=latents,
        latent_mask=torch.ones((requests, 4), dtype=torch.bool),
        timesteps=torch.zeros(requests),
        dt=torch.full((requests,), 0.5),
        cfg_scales=torch.tensor([[2.0, 3.0], [2.0, 3.0]]),
        bundle=(text, torch.ones((requests * cfg_rows, 2), dtype=torch.bool), None, None, None, None),
        context_kv_cache=None,
        context_buckets=(2, 1, 1),
        cfg_refresh=refresh,
        cfg_correction=torch.zeros_like(latents) if refresh else correction,
    )


def test_padded_cfg_refresh_and_reuse_are_equivalent():
    refresh_model = _PaddedDiT()
    refresh = _padded_batch(refresh=True)
    refreshed_latents = run_packed_euler_rf_cfg_step(refresh_model, refresh)
    expected_correction = torch.full_like(refresh.latents, -8.0)
    torch.testing.assert_close(refresh.cfg_correction, expected_correction)
    assert refresh_model.row_counts == [6]

    reuse_model = _PaddedDiT()
    reuse = _padded_batch(refresh=False, correction=expected_correction)
    reused_latents = run_packed_euler_rf_cfg_step(reuse_model, reuse)
    torch.testing.assert_close(reused_latents, refreshed_latents)
    assert reuse_model.row_counts == [2]


class _VarlenDiT:
    dtype = torch.float32
    device = torch.device("cpu")
    packed_attention_dtype = torch.bfloat16

    def __init__(self):
        self.query_lengths = ()

    def supports_packed_varlen_attention(self):
        return True

    def forward_with_packed_conditions(self, *, x_t, query_lengths, **_):
        self.query_lengths = query_lengths
        rows, offset = [], 0
        for marker, length in enumerate(query_lengths):
            rows.append(x_t[:, offset : offset + length] * 2 + marker)
            offset += length
        return torch.cat(rows, dim=1)


def _varlen_state(*, cfg_active: bool, correction: float | None = None):
    latent = torch.ones((1, 2, 1))
    return types.SimpleNamespace(
        latents=latent,
        t_schedule=torch.tensor([0.8, 0.7]),
        current_timestep=torch.tensor(0.8),
        step_index=0,
        cfg_active=(cfg_active,),
        independent_names=("cond", "text", "speaker"),
        cfg_scales={"text": 2.0, "speaker": 3.0},
        cfg_guidance_mode="independent",
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_active=False,
        context_kv_cond=[],
        context_kv_cfg=[],
        valid_latent_lengths=(2,),
        latent_mask=None,
        cfg_correction=None if correction is None else torch.full_like(latent, correction),
    )


def test_varlen_batch_coalesces_cfg_modes(monkeypatch):
    states = [
        _varlen_state(cfg_active=True),
        _varlen_state(cfg_active=False),
        _varlen_state(cfg_active=True, correction=7.0),
    ]
    observed_modes = []

    def fake_context(states, *, modes: list[str], attention_dtype):
        del states, attention_dtype
        row_counts = [3 if mode == "cfg" else 1 for mode in modes]
        observed_modes.extend(modes)
        return [], tuple(1 for count in row_counts for _ in range(count))

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.irodori_tts.sampler.pack_irodori_batch_context",
        fake_context,
    )
    model = _VarlenDiT()
    result = run_packed_varlen_euler_rf_cfg_step(model, states, cfg_refreshes=[True, True, False])

    assert observed_modes == ["cfg", "cond", "cond"]
    assert model.query_lengths == (2, 2, 2, 2, 2)
    for updated, velocity in zip(result, (-6.0, 5.0, 13.0), strict=True):
        torch.testing.assert_close(updated, torch.ones_like(updated) + velocity * -0.1)
