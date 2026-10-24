# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native discovery and strict checkpoint contracts, without release weights."""

import pytest
import torch

from vllm_omni.diffusion.models.sana_video2.transformer_sana_video2 import (
    SanaVideo2TransformerConfig,
    SanaVideo2TransformerModel,
)
from vllm_omni.diffusion.utils.hf_utils import resolve_native_diffusion_model_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def tiny_model():
    return SanaVideo2TransformerModel(
        SanaVideo2TransformerConfig(
            hidden_size=32,
            depth=2,
            num_heads=4,
            in_channels=4,
            caption_channels=16,
            model_max_length=8,
            linear_head_dim=8,
            softmax_head_dim=16,
            attn_res_block_size=1,
        )
    )


def test_official_id_resolves_without_model_index():
    assert resolve_native_diffusion_model_class("Efficient-Large-Model/SANA-Video_2.0_5B_720p") == "SanaVideo2Pipeline"


def test_registered_class_is_native_pipeline():
    from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    name = resolve_native_diffusion_model_class("Efficient-Large-Model/SANA-Video_2.0_5B_720p")
    assert DiffusionModelRegistry._try_load_model_cls(name) is SanaVideo2Pipeline


def test_local_release_discovery_requires_both_files_and_5b_config(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  model: SanaVideo2_5B\n")
    assert resolve_native_diffusion_model_class(str(tmp_path)) is None
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/SANA_Video_2.0_5B_720p.pth").touch()
    assert resolve_native_diffusion_model_class(str(tmp_path)) == "SanaVideo2Pipeline"
    (tmp_path / "config.yaml").write_text("model:\n  model: SanaVideo2_14B\n")
    assert resolve_native_diffusion_model_class(str(tmp_path)) is None
    (tmp_path / "config.yaml").write_text("model: [broken")
    assert resolve_native_diffusion_model_class(str(tmp_path)) is None


def test_checkpoint_loads_every_parameter_and_only_documented_buffers(tiny_model, tmp_path):
    weights = {key: torch.full_like(value, 0.125) for key, value in tiny_model.state_dict().items()}
    weights["pos_embed"] = torch.zeros(1, 1, 32)
    weights["y_embedder.y_embedding"] = torch.zeros(8, 16)
    path = tmp_path / "release.pth"
    torch.save({"state_dict": weights}, path)
    report = tiny_model.load_checkpoint(path)
    assert set(report.loaded_keys) == set(tiny_model.state_dict())
    assert report.loaded_numel == sum(p.numel() for p in tiny_model.parameters())
    assert report.ignored_keys == ("pos_embed", "y_embedder.y_embedding")
    assert report.source_dtypes == ("torch.float32",)
    assert all(torch.equal(value, weights[key]) for key, value in tiny_model.state_dict().items())


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape", "duplicate", "integer", "ignored_shape"])
def test_invalid_checkpoint_is_rejected_before_mutation(tiny_model, defect):
    original = {key: value.clone() for key, value in tiny_model.state_dict().items()}
    weights = [(key, torch.zeros_like(value)) for key, value in original.items()]
    key, value = weights[0]
    if defect == "missing":
        weights.pop(0)
    elif defect == "unexpected":
        weights.append(("unknown.weight", value))
    elif defect == "shape":
        weights[0] = (key, torch.zeros(1))
    elif defect == "duplicate":
        weights.append(weights[0])
    elif defect == "integer":
        weights[0] = (key, value.long())
    else:
        weights.append(("pos_embed", torch.zeros(1)))
    with pytest.raises(ValueError, match="Invalid SANA-Video 2.0 checkpoint"):
        tiny_model.load_weights(weights)
    assert all(torch.equal(value, original[key]) for key, value in tiny_model.state_dict().items())
