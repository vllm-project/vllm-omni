# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.lora_runtime import DiffusionLoRADeployment
from vllm_omni.diffusion.models.minimax_h3.lora_runtime import (
    MiniMaxH3LoRALoader,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _write_tiny_h3_lora(path, *, alpha: str = "128", extra_metadata: dict[str, str] | None = None) -> None:
    rank = 128
    metadata = {"alpha": alpha, "key_format": "minimax-h3-diffusers"}
    metadata.update(extra_metadata or {})
    save_file(
        {
            "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight": torch.ones(rank, 2),
            "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight": torch.cat(
                (torch.ones(2, rank), torch.full((2, rank), 2.0)), dim=0
            ),
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(rank, 2),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(4, rank),
        },
        str(path),
        metadata=metadata,
    )


def test_h3_loader_normalizes_keys_alpha_and_ffn_layout(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0.safetensors"
    _write_tiny_h3_lora(path)
    loaded = MiniMaxH3LoRALoader(SimpleNamespace(partition="fl2va")).load(
        DiffusionLoRADeployment("turbo", str(path)),
        path,
    )
    updates = {update.logical_target: update for update in loaded.updates}
    assert set(updates) == {"blocks.0.mlp.fc1", "blocks.0.attn.to_q"}
    assert updates["blocks.0.mlp.fc1"].rank == 128
    assert updates["blocks.0.mlp.fc1"].intrinsic_scale == 1.0
    torch.testing.assert_close(
        updates["blocks.0.mlp.fc1"].lora_b,
        torch.cat((torch.full((2, 128), 2.0), torch.ones(2, 128)), dim=0),
    )


def test_h3_loader_rejects_non_v1_alpha_and_ref2va(tmp_path):
    path = tmp_path / "turbo.safetensors"
    _write_tiny_h3_lora(path, alpha="8")
    loader = MiniMaxH3LoRALoader(SimpleNamespace(partition="fl2va"))
    with pytest.raises(ValueError, match="requires alpha=128"):
        loader.load(DiffusionLoRADeployment("turbo", str(path)), path)

    with pytest.raises(ValueError, match="supports FL2VA only"):
        MiniMaxH3LoRALoader(SimpleNamespace(partition="ref2va"))


def test_h3_loader_rejects_unknown_and_ambiguous_formats(tmp_path):
    loader = MiniMaxH3LoRALoader(SimpleNamespace(partition="fl2va"))

    unknown = tmp_path / "unknown.safetensors"
    _write_tiny_h3_lora(unknown, extra_metadata={"key_format": "unknown"})
    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 LoRA metadata"):
        loader.load(DiffusionLoRADeployment("any-name", str(unknown)), unknown)

    ambiguous = tmp_path / "ambiguous.safetensors"
    _write_tiny_h3_lora(
        ambiguous,
        extra_metadata={"base_model": "minimax-h3-fl2va", "lora_rank": "128"},
    )
    with pytest.raises(ValueError, match="Ambiguous MiniMax-H3 LoRA format"):
        loader.load(DiffusionLoRADeployment("still-just-an-alias", str(ambiguous)), ambiguous)


def test_h3_loader_splits_native_fused_qkv(tmp_path):
    path = tmp_path / "native.safetensors"
    rank = 2
    lora_a = torch.arange(rank * 3, dtype=torch.float32).reshape(rank, 3)
    lora_b = torch.arange(12 * rank, dtype=torch.float32).reshape(12, rank)
    out_b = torch.ones(3, rank)
    save_file(
        {
            "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight": lora_a,
            "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight": lora_b,
            "diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_A.weight": lora_a.clone(),
            "diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_B.weight": out_b,
        },
        str(path),
        metadata={"base_model": "minimax-h3-fl2va", "lora_rank": str(rank)},
    )

    loaded = MiniMaxH3LoRALoader(SimpleNamespace(partition="fl2va")).load(
        DiffusionLoRADeployment("native", str(path)),
        path,
    )
    updates = {update.logical_target: update for update in loaded.updates}
    assert set(updates) == {
        "token_refiner.blocks.0.attn.to_q",
        "token_refiner.blocks.0.attn.to_k",
        "token_refiner.blocks.0.attn.to_v",
        "token_refiner.blocks.0.attn.out_proj",
    }
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        update = updates[f"token_refiner.blocks.0.attn.{name}"]
        torch.testing.assert_close(update.lora_a, lora_a)
        torch.testing.assert_close(update.lora_b, lora_b.chunk(3, dim=0)[index])
        assert update.intrinsic_scale == 1.0
