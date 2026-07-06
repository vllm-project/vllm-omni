# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.diffusion.lora.utils import (
    convert_single_file_lora,
    find_single_file_lora,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Module layout mirroring the Qwen-Image transformer: plain projections,
# a diffusers ModuleList indirection (`to_out.0`), and MLP linears whose
# leaf names (`proj`, `2`) are how they appear in the pipeline module tree.
_MODULE_PATHS = [
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.img_mlp.net.0.proj",
    "transformer_blocks.0.img_mlp.net.2",
]
_EXPECTED_MODULES = {"to_q", "to_out", "proj", "2"}
_RANK = 4
_DIM = 16


def _kohya_state_dict(alpha: float | None = 2.0) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for path in _MODULE_PATHS:
        sd[f"{path}.lora_down.weight"] = torch.randn(_RANK, _DIM)
        sd[f"{path}.lora_up.weight"] = torch.randn(_DIM, _RANK)
        if alpha is not None:
            sd[f"{path}.alpha"] = torch.tensor(alpha)
    return sd


def test_convert_kohya_names_and_config():
    sd = _kohya_state_dict()
    config, tensors = convert_single_file_lora(sd, _EXPECTED_MODULES)

    assert config["r"] == _RANK
    # alpha is folded into lora_B, so the config keeps a 1.0 global scale.
    assert config["lora_alpha"] == config["r"]
    assert config["target_modules"] == [
        "attn.to_out",
        "attn.to_q",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
    ]
    # The config round-trips through vLLM's PEFT parsing.
    helper = PEFTHelper.from_dict(config)
    assert helper.vllm_lora_scaling_factor == 1.0

    # ModuleList indirection is folded onto the pipeline module path.
    assert "base_model.model.transformer_blocks.0.attn.to_out.lora_A.weight" in tensors
    # Plain paths are preserved.
    assert "base_model.model.transformer_blocks.0.img_mlp.net.0.proj.lora_B.weight" in tensors
    assert len(tensors) == 2 * len(_MODULE_PATHS)


def test_convert_folds_alpha_into_lora_b():
    sd = _kohya_state_dict(alpha=2.0)
    _, tensors = convert_single_file_lora(sd, _EXPECTED_MODULES)
    expected = sd["transformer_blocks.0.attn.to_q.lora_up.weight"] * (2.0 / _RANK)
    got = tensors["base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight"]
    torch.testing.assert_close(got, expected)


def test_convert_accepts_peft_style_names_and_prefixes():
    sd = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(_RANK, _DIM),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(_DIM, _RANK),
    }
    config, tensors = convert_single_file_lora(sd, _EXPECTED_MODULES)
    assert config["r"] == _RANK
    assert "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight" in tensors


def test_convert_rejects_unsupported_modules():
    sd = {
        "transformer_blocks.0.attn.unknown_proj.lora_down.weight": torch.randn(_RANK, _DIM),
        "transformer_blocks.0.attn.unknown_proj.lora_up.weight": torch.randn(_DIM, _RANK),
    }
    with pytest.raises(ValueError, match="do not match any supported LoRA module"):
        convert_single_file_lora(sd, _EXPECTED_MODULES)


def test_convert_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unrecognized key"):
        convert_single_file_lora({"transformer_blocks.0.attn.to_q.weird": torch.zeros(1)}, _EXPECTED_MODULES)


def test_find_single_file_lora(tmp_path):
    ckpt = tmp_path / "adapter.safetensors"
    save_file({"x": torch.zeros(1)}, str(ckpt))

    # Direct file path and a directory holding exactly one safetensors file.
    assert find_single_file_lora(str(ckpt)) == str(ckpt)
    assert find_single_file_lora(str(tmp_path)) == str(ckpt)

    # A PEFT directory keeps going through the PEFT loader.
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4}))
    assert find_single_file_lora(str(tmp_path)) is None

    # Ambiguous directories (multiple safetensors) are not treated as single-file.
    (tmp_path / "adapter_config.json").unlink()
    save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"))
    assert find_single_file_lora(str(tmp_path)) is None


def test_find_single_file_lora_case_insensitive(tmp_path):
    ckpt = tmp_path / "ADAPTER.SAFETENSORS"
    save_file({"x": torch.zeros(1)}, str(ckpt))
    assert find_single_file_lora(str(ckpt)) == str(ckpt)
    assert find_single_file_lora(str(tmp_path)) == str(ckpt)


def test_convert_mixed_ranks_uses_max_and_keeps_per_module_scale():
    sd = {
        "transformer_blocks.0.attn.to_q.lora_down.weight": torch.randn(4, _DIM),
        "transformer_blocks.0.attn.to_q.lora_up.weight": torch.randn(_DIM, 4),
        "transformer_blocks.0.attn.to_q.alpha": torch.tensor(2.0),
        "transformer_blocks.0.attn.to_k.lora_down.weight": torch.randn(8, _DIM),
        "transformer_blocks.0.attn.to_k.lora_up.weight": torch.randn(_DIM, 8),
        "transformer_blocks.0.attn.to_k.alpha": torch.tensor(2.0),
    }
    config, tensors = convert_single_file_lora(sd, _EXPECTED_MODULES | {"to_k"})
    # The synthesized config advertises the max rank...
    assert config["r"] == 8
    assert config["lora_alpha"] == 8
    # ...while each module keeps its exact alpha/rank scaling folded into lora_B.
    torch.testing.assert_close(
        tensors["base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight"],
        sd["transformer_blocks.0.attn.to_q.lora_up.weight"] * (2.0 / 4),
    )
    torch.testing.assert_close(
        tensors["base_model.model.transformer_blocks.0.attn.to_k.lora_B.weight"],
        sd["transformer_blocks.0.attn.to_k.lora_up.weight"] * (2.0 / 8),
    )
