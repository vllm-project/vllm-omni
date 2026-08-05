# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for transformer quant-config auto-detection.

The loader path at pipeline_wan2_2.py carries two quantization contracts:

  create_transformer_from_config (~L137)
    - Reads quantization_config from config.json (injected by the merge scripts)
    - Auto-detects the quant method when no CLI quant_config is provided
    - Rejects method mismatches (CLI vs disk)
    - Upgrades online → offline when disk marks is_checkpoint_*_serialized=True
    - Rebuilds when the active ignored_layers differs from the disk value

  Wan22Pipeline._create_transformer (~L456)
    - Passes od_config.quantization_config (set by CLI or externally) to
      create_transformer_from_config for each transformer independently.
    - When od_config.quantization_config is None, each transformer auto-detects
      from its own config.json; od_config is NOT modified by this call.

All tests are pure-CPU and do not load model weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import create_transformer_from_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Minimum config that create_transformer_from_config accepts without raising.
_MIN_CFG: dict = {"patch_size": [1, 2, 2]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_transformer():
    """Return (FakeTransformer class, captured list).

    Each _FakeTransformer.__init__ call appends its **kwargs to captured,
    letting tests inspect what quant_config was passed per transformer.
    """
    captured: list[dict] = []

    class _FakeTransformer:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    return _FakeTransformer, captured


class _FakePipeline:
    """Minimal stand-in exposing only what _create_transformer needs from self."""

    def __init__(self, od_config: SimpleNamespace) -> None:
        self.od_config = od_config

    # Bind the real unbound method so the tests exercise production code.
    _create_transformer = wan22_module.Wan22Pipeline._create_transformer


# ---------------------------------------------------------------------------
# create_transformer_from_config — auto-detection
# ---------------------------------------------------------------------------


def test_create_transformer_detects_mxfp8_serialized_from_config_json(monkeypatch):
    """When config.json carries MXFP8 quant and no CLI quant_config is provided,
    the transformer must receive a DiffusionMXFP8Config with serialized=True."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    create_transformer_from_config(config)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8Config)
    assert qc.is_checkpoint_mxfp8_serialized is True


def test_create_transformer_detects_mxfp4_dualscale_from_config_json(monkeypatch):
    """config.json with mxfp4_dualscale + ignored_layers must produce
    a DiffusionMXFP4DualScaleMixedConfig with the correct ignored_layers and
    serialized flag."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp4_dualscale",
            "ignored_layers": ["blocks.0.attn1.to_q", "blocks.0.attn1.to_k"],
            "is_checkpoint_serialized": True,
        },
    }
    create_transformer_from_config(config)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP4DualScaleMixedConfig)
    assert set(qc.ignored_layers) == {"blocks.0.attn1.to_q", "blocks.0.attn1.to_k"}
    assert qc.is_checkpoint_serialized is True


def test_create_transformer_without_quantization_config_passes_no_quant(monkeypatch):
    """A plain BF16 config.json (no quantization_config key) must result in no
    quant_config being passed to WanTransformer3DModel."""
    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    create_transformer_from_config(_MIN_CFG)

    assert "quant_config" not in captured[0]


# ---------------------------------------------------------------------------
# create_transformer_from_config — method-mismatch guard
# ---------------------------------------------------------------------------


def test_create_transformer_rejects_method_mismatch(monkeypatch):
    """Passing a CLI quant_config whose get_name() differs from the config.json
    quant_method must raise ValueError immediately (prevents silent weight corruption).

    fp8 (vLLM built-in, get_name()=='fp8') vs disk 'mxfp8' triggers the guard.
    These are distinct methods; using the same method for both would not trigger it.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    FakeTransformer, _ = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    fp8_cli = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    with pytest.raises(ValueError, match="quant_method"):
        create_transformer_from_config(config, quant_config=fp8_cli)


# ---------------------------------------------------------------------------
# create_transformer_from_config — online → offline upgrade
# ---------------------------------------------------------------------------


def test_create_transformer_upgrades_to_serialized_when_disk_marks_it(monkeypatch):
    """CLI passes online (is_checkpoint_mxfp8_serialized=False) but config.json
    marks is_checkpoint_mxfp8_serialized=True → must switch to offline (serialized)
    so that pre-quantized FP8 tensors are loaded correctly."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    online_cli = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False)
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    create_transformer_from_config(config, quant_config=online_cli)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8Config)
    assert qc.is_checkpoint_mxfp8_serialized is True


# ---------------------------------------------------------------------------
# create_transformer_from_config — ignored_layers rebuild
# ---------------------------------------------------------------------------


def test_create_transformer_rebuilds_when_ignored_layers_differ(monkeypatch):
    """When the active quant_config has different ignored_layers than config.json,
    the config must be rebuilt from disk so per-layer BF16 routing is authoritative
    for this specific transformer."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    stale = DiffusionMXFP4DualScaleMixedConfig(
        is_checkpoint_serialized=True,
        ignored_layers=["blocks.0.attn1.to_q"],
    )
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp4_dualscale",
            "ignored_layers": ["blocks.0.attn1.to_q", "blocks.1.attn1.to_q"],
            "is_checkpoint_serialized": True,
        },
    }
    create_transformer_from_config(config, quant_config=stale)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP4DualScaleMixedConfig)
    assert set(qc.ignored_layers) == {"blocks.0.attn1.to_q", "blocks.1.attn1.to_q"}


def test_create_transformer_does_not_rebuild_when_ignored_layers_match(monkeypatch):
    """When the active quant_config already has the same ignored_layers,
    the same instance must be passed through unchanged (no unnecessary rebuild)."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    matching = DiffusionMXFP4DualScaleMixedConfig(
        is_checkpoint_serialized=True,
        ignored_layers=["blocks.0.attn1.to_q"],
    )
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp4_dualscale",
            "ignored_layers": ["blocks.0.attn1.to_q"],
            "is_checkpoint_serialized": True,
        },
    }
    create_transformer_from_config(config, quant_config=matching)

    assert captured[0].get("quant_config") is matching


# ---------------------------------------------------------------------------
# Wan22Pipeline._create_transformer — od_config passthrough
# ---------------------------------------------------------------------------


def test_pipeline_create_transformer_auto_detects_from_config_json(monkeypatch):
    """When od_config.quantization_config is None, _create_transformer must
    auto-detect the quant method from config.json and pass it to the transformer.
    od_config itself must remain unchanged (None)."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    od_config = SimpleNamespace(quantization_config=None)
    pipeline = _FakePipeline(od_config)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    pipeline._create_transformer(config)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8Config)
    assert qc.is_checkpoint_mxfp8_serialized is True
    # od_config must NOT be modified — each transformer auto-detects independently.
    assert od_config.quantization_config is None


def test_pipeline_create_transformer_does_not_overwrite_existing_od_config(monkeypatch):
    """If od_config.quantization_config is already set (e.g. via CLI), _create_transformer
    must pass it through to the transformer unchanged and leave od_config unmodified."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, _ = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    existing = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    od_config = SimpleNamespace(quantization_config=existing)
    pipeline = _FakePipeline(od_config)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    pipeline._create_transformer(config)

    assert od_config.quantization_config is existing


# ---------------------------------------------------------------------------
# Wan22Pipeline._create_transformer — cascade contracts
# ---------------------------------------------------------------------------


def test_pipeline_cascade_both_transformers_get_mxfp8_serialized_config(monkeypatch):
    """Cascade model (transformer + transformer_2) with MXFP8 checkpoint:
    - Each transformer auto-detects from its own config.json independently.
    Both must receive is_checkpoint_mxfp8_serialized=True."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    od_config = SimpleNamespace(quantization_config=None)
    pipeline = _FakePipeline(od_config)

    mxfp8_qc = {"quant_method": "mxfp8", "is_checkpoint_mxfp8_serialized": True}
    pipeline._create_transformer({**_MIN_CFG, "quantization_config": mxfp8_qc})
    pipeline._create_transformer({**_MIN_CFG, "quantization_config": mxfp8_qc})

    assert len(captured) == 2
    for i, kwargs in enumerate(captured):
        qc = kwargs.get("quant_config")
        assert isinstance(qc, DiffusionMXFP8Config), f"transformer[{i}]: expected DiffusionMXFP8Config, got {type(qc)}"
        assert qc.is_checkpoint_mxfp8_serialized is True, f"transformer[{i}]: expected serialized=True"


def test_pipeline_cascade_mxfp4_dualscale_each_transformer_gets_correct_ignored_layers(monkeypatch):
    """Cascade with mxfp4_dualscale where transformer and transformer_2 have
    different ignored_layers in their config.json.

    Expected outcome:
      transformer   → ignored_layers=["blocks.0.attn1.to_q"]  (auto-detected from its config.json)
      transformer_2 → ignored_layers=["blocks.0.attn1.to_q", "blocks.1.attn1.to_q"]
                      (auto-detected from its own config.json independently)
      od_config     → quantization_config remains None (not modified by _create_transformer)
    """
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    od_config = SimpleNamespace(quantization_config=None)
    pipeline = _FakePipeline(od_config)

    cfg1 = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp4_dualscale",
            "ignored_layers": ["blocks.0.attn1.to_q"],
            "is_checkpoint_serialized": True,
        },
    }
    cfg2 = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp4_dualscale",
            "ignored_layers": ["blocks.0.attn1.to_q", "blocks.1.attn1.to_q"],
            "is_checkpoint_serialized": True,
        },
    }

    pipeline._create_transformer(cfg1)
    pipeline._create_transformer(cfg2)

    assert len(captured) == 2
    qc1 = captured[0].get("quant_config")
    qc2 = captured[1].get("quant_config")

    assert isinstance(qc1, DiffusionMXFP4DualScaleMixedConfig)
    assert isinstance(qc2, DiffusionMXFP4DualScaleMixedConfig)
    assert set(qc1.ignored_layers) == {"blocks.0.attn1.to_q"}, f"transformer expected 1 layer, got {qc1.ignored_layers}"
    assert set(qc2.ignored_layers) == {
        "blocks.0.attn1.to_q",
        "blocks.1.attn1.to_q",
    }, f"transformer_2 expected 2 layers, got {qc2.ignored_layers}"

    # od_config must remain unchanged — _create_transformer does not modify it.
    assert od_config.quantization_config is None
