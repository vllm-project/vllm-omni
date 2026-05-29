# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    create_transformer_from_config,
    load_transformer_config,
    retrieve_latents,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _LatentDist:
    def sample(self, generator):
        assert isinstance(generator, torch.Generator)
        return torch.tensor([1.0])

    def mode(self):
        return torch.tensor([2.0])


def test_retrieve_latents_supports_sample_mode_argmax_and_direct_latents() -> None:
    generator = torch.Generator(device="cpu")

    assert retrieve_latents(SimpleNamespace(latent_dist=_LatentDist()), generator).item() == 1.0
    assert retrieve_latents(SimpleNamespace(latent_dist=_LatentDist()), sample_mode="argmax").item() == 2.0
    torch.testing.assert_close(retrieve_latents(SimpleNamespace(latents=torch.tensor([3.0]))), torch.tensor([3.0]))


def test_retrieve_latents_rejects_unknown_encoder_output() -> None:
    with pytest.raises(AttributeError, match="Could not access latents"):
        retrieve_latents(SimpleNamespace())


def test_load_transformer_config_reads_local_subfolder_config(tmp_path) -> None:
    config_dir = tmp_path / "transformer_2"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({"patch_size": [1, 2, 2], "num_layers": 2}))

    assert load_transformer_config(str(tmp_path), "transformer_2") == {"patch_size": [1, 2, 2], "num_layers": 2}
    assert load_transformer_config(str(tmp_path), "missing") == {}


def test_create_transformer_from_config_maps_supported_keys(monkeypatch) -> None:
    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    transformer = create_transformer_from_config(
        {
            "patch_size": [1, 2, 2],
            "num_attention_heads": 8,
            "attention_head_dim": 128,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 4096,
            "vace_layers": [0],
            "ignored": "value",
        }
    )

    assert isinstance(transformer, FakeTransformer)
    assert captured == {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 8,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 4096,
    }
