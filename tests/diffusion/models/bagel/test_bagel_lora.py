# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BAGEL LoRA support across Stage 0 (Thinker) and Stage 1 (DiT)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.diffusion.lora.conftest import (
    DummyBaseLayerWithLoRA,
    FakeLinearBase,
    fake_replace_submodule,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FakeLinearBase = FakeLinearBase


# ---------------------------------------------------------------------------
# Stage 0 (Thinker / AR) -- packed_modules_mapping on the AR model class
# ---------------------------------------------------------------------------


class TestStage0ThinkerLoRA:
    """Validate that OmniBagelForConditionalGeneration declares correct LoRA metadata."""

    def test_omni_bagel_supports_lora(self):
        from vllm_omni.model_executor.models.bagel.bagel import (
            OmniBagelForConditionalGeneration,
        )

        assert getattr(OmniBagelForConditionalGeneration, "supports_lora", False) is True

    def test_omni_bagel_packed_modules_mapping_complete(self):
        from vllm_omni.model_executor.models.bagel.bagel import (
            OmniBagelForConditionalGeneration,
        )

        mapping = OmniBagelForConditionalGeneration.packed_modules_mapping
        # Standard Qwen2 projections
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
        # MoE generation-mode projections
        assert mapping["qkv_proj_moe_gen"] == [
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
        ]
        assert mapping["mlp_moe_gen.gate_up_proj"] == [
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
        ]


# ---------------------------------------------------------------------------
# Stage 1 (DiT / Diffusion) -- DiffusionLoRAManager with bagel component
# ---------------------------------------------------------------------------


class TestStage1DiTLoRA:
    """Validate DiffusionLoRAManager discovers BAGEL's packed modules."""

    def test_diffusion_lora_manager_discovers_bagel_packed_modules(self):
        """Manager should derive packed→sublayer mapping from stacked_params_mapping."""
        pipeline = torch.nn.Module()
        pipeline.bagel = torch.nn.Module()

        # Simulate a submodule that exposes stacked_params_mapping
        # (as Bagel does after load_weights())
        language_model = torch.nn.Module()
        language_model.stacked_params_mapping = [
            (".qkv_proj_moe_gen", ".q_proj_moe_gen", "q"),
            (".qkv_proj_moe_gen", ".k_proj_moe_gen", "k"),
            (".qkv_proj_moe_gen", ".v_proj_moe_gen", "v"),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        pipeline.bagel.language_model = language_model

        manager = DiffusionLoRAManager(
            pipeline=pipeline,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_cached_adapters=1,
        )

        mapping = manager._packed_modules_mapping
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert mapping["qkv_proj_moe_gen"] == [
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
        ]
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]

    def test_diffusion_lora_manager_replaces_bagel_packed_layer_via_sublayer_target(self, monkeypatch):
        """Targeting sublayer 'q_proj' should replace the fused 'qkv_proj' under bagel."""
        import vllm_omni.diffusion.lora.manager as manager_mod

        monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", DummyBaseLayerWithLoRA)

        def _fake_from_layer_diffusion(*, layer, **_kwargs):
            return DummyBaseLayerWithLoRA(layer)

        replace_calls: list[str] = []

        monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
        monkeypatch.setattr(
            manager_mod,
            "replace_submodule",
            lambda root, name, sub: fake_replace_submodule(root, name, sub, replace_calls),
        )

        # Build pipeline with bagel component
        pipeline = torch.nn.Module()
        pipeline.bagel = torch.nn.Module()
        lm = torch.nn.Module()
        lm.stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        lm.attn = torch.nn.Module()
        lm.attn.qkv_proj = _FakeLinearBase()
        pipeline.bagel.language_model = lm

        manager = DiffusionLoRAManager(
            pipeline=pipeline,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_cached_adapters=1,
        )

        # Treat qkv_proj as 3-slice packed layer
        monkeypatch.setattr(manager, "_get_packed_modules_list", lambda _module: ["q", "k", "v"])

        # Target sublayer "q_proj" -- manager should replace the packed "qkv_proj"
        peft_helper = type("_PH", (), {"r": 1, "target_modules": ["q_proj"]})()
        manager._replace_layers_with_lora(peft_helper)

        assert "language_model.attn.qkv_proj" in replace_calls
        assert "bagel.language_model.attn.qkv_proj" in manager._lora_modules
        # Verify the module was actually replaced in the tree (not just recorded)
        assert isinstance(pipeline.bagel.language_model.attn.qkv_proj, DummyBaseLayerWithLoRA)


# ---------------------------------------------------------------------------
# Round-trip: synthetic checkpoint → set_active_adapter → verify weights
# ---------------------------------------------------------------------------


def _write_synthetic_lora(
    adapter_dir: Path,
    module_name: str,
    rank: int,
    in_dim: int,
    out_dim: int,
) -> str:
    """Write a minimal LoRA adapter (safetensors + config) to *adapter_dir*."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    lora_a = torch.ones((rank, in_dim), dtype=torch.float32)
    lora_b = torch.ones((out_dim, rank), dtype=torch.float32) * 2.0
    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": [module_name]}),
        encoding="utf-8",
    )
    return str(adapter_dir)


class TestBagelLoRARoundTrip:
    """End-to-end: synthetic checkpoint → load → activate → verify weights in fused layer."""

    def test_set_active_adapter_loads_and_activates_bagel_lora(self, tmp_path, monkeypatch):
        """Full round-trip through set_active_adapter for a bagel component module."""
        import vllm_omni.diffusion.lora.manager as manager_mod

        monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", DummyBaseLayerWithLoRA)

        # Build pipeline with bagel.language_model.foo (simple non-packed layer)
        pipeline = torch.nn.Module()
        pipeline.bagel = torch.nn.Module()
        lm = torch.nn.Module()
        lm.foo = _FakeLinearBase()
        pipeline.bagel.language_model = lm

        def _fake_from_layer(*, layer, **_kwargs):
            if isinstance(layer, FakeLinearBase):
                return DummyBaseLayerWithLoRA(layer)
            return layer

        monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer)
        monkeypatch.setattr(
            manager_mod,
            "replace_submodule",
            lambda root, name, sub: fake_replace_submodule(root, name, sub),
        )

        manager = DiffusionLoRAManager(
            pipeline=pipeline,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_cached_adapters=1,
        )

        # Write synthetic adapter targeting bagel.language_model.foo
        module_name = "bagel.language_model.foo"
        rank = 2
        in_dim = 4
        out_dim = 4
        lora_dir = _write_synthetic_lora(tmp_path / "lora", module_name, rank, in_dim, out_dim)

        lora_request = LoRARequest(
            lora_name="test_bagel",
            lora_int_id=42,
            lora_path=lora_dir,
        )

        # Full round-trip: load from disk → replace layer → activate weights
        manager.set_active_adapter(lora_request, lora_scale=0.5)

        # Verify the layer was replaced and weights were set
        replaced_layer = pipeline.bagel.language_model.foo
        assert isinstance(replaced_layer, DummyBaseLayerWithLoRA), "Layer should be wrapped with LoRA"
        assert len(replaced_layer.set_calls) == 1, "set_lora should have been called once"

        lora_a, lora_b = replaced_layer.set_calls[0]
        # A weights should be ones (as written)
        assert torch.all(lora_a == 1.0), f"lora_a should be all ones, got {lora_a}"
        # B weights should be 2.0 * scale(0.5) = 1.0
        assert torch.allclose(lora_b, torch.ones_like(lora_b)), f"lora_b should be 2.0 * 0.5 = 1.0, got {lora_b}"
