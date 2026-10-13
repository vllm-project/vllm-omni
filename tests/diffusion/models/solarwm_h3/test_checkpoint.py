# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

_PACKAGE = "_solarwm_h3_checkpoint_tests"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/solarwm_h3")]
sys.modules.setdefault(_PACKAGE, package)
checkpoint = importlib.import_module(f"{_PACKAGE}.checkpoint")
pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def tiny_contract(monkeypatch):
    dims = {
        "attn.to_q": (4, 6),
        "attn.to_k": (4, 6),
        "attn.to_v": (4, 6),
        "attn.to_out.0": (6, 4),
        "ff.net.0.proj": (4, 8),
        "ff.net.2": (4, 4),
    }
    monkeypatch.setattr(checkpoint, "TARGET_DIMS", dims)
    monkeypatch.setattr(checkpoint, "BLOCK_COUNTS", (("transformer_blocks", 2), ("token_refiner.refiner_blocks", 1)))
    monkeypatch.setattr(checkpoint, "LORA_RANK", 2)
    monkeypatch.setattr(checkpoint, "LORA_ALPHA", 2)
    generator = torch.Generator().manual_seed(11)
    tensors = {}
    for target, (in_dim, out_dim) in checkpoint._expected_targets().items():
        tensors[f"{target}.lora_A.weight"] = torch.randn(2, in_dim, generator=generator) / 10
        tensors[f"{target}.lora_B.weight"] = torch.randn(out_dim, 2, generator=generator) / 10
    return tensors


def _write_checkpoint(root, tensors):
    torch.save({"schema": "solarwm.minimax-h3-ema.v1", "shadow": tensors}, root / "ema.pt")
    manifest = {
        "schema": "solarwm.checkpoint.v2",
        "step": 3900,
        "contract": {
            "family": "minimax_h3",
            "stage": "stage2",
            "camera_translation_transform": "logd4",
            "parameterization": "peft-lora-r384-alpha384",
            "causal_mode": "self_gradient_forcing",
            "objective": "flow_matching",
            "objective_variant": "data_ward_velocity",
            "extras": {
                "chunk_latents": 5,
                "window_chunks": 6,
                "target_latents": 47,
                "rollout_latents": 50,
                "student_rope_mode": "sliding_local",
                "encoder_profile": {"pixel_frames": 158, "height": 768, "width": 1344},
            },
        },
        "files": [{"path": "ema.pt", "size": (root / "ema.pt").stat().st_size}],
    }
    raw = json.dumps(manifest).encode()
    (root / "checkpoint-manifest.json").write_bytes(raw)
    (root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "solarwm.checkpoint-complete.v2",
                "step": 3900,
                "manifest_digest": hashlib.blake2s(raw).hexdigest(),
            }
        )
    )


def test_published_target_topology_includes_refiner():
    assert len(checkpoint._expected_targets()) == 312
    assert "token_refiner.refiner_blocks.1.ff.net.0.proj" in checkpoint._expected_targets()


def test_loads_ema_shadow_and_canonicalizes_wrapped_peft_names(tmp_path, tiny_contract):
    wrapped = {
        "base_model.model._fsdp_wrapped_module." + key.replace(".weight", ".default.weight"): value
        for key, value in tiny_contract.items()
    }
    _write_checkpoint(tmp_path, wrapped)
    adapter = checkpoint.load_stage2_adapter(tmp_path, expected_step=3900)
    assert adapter.step == 3900 and adapter.rank == 2 and adapter.alpha == 2
    assert adapter.tensors.keys() == tiny_contract.keys()
    with pytest.raises(ValueError, match="expected=1200"):
        checkpoint.load_stage2_adapter(tmp_path, expected_step=1200)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "shape", "nan", "wrong_stage", "wrong_velocity", "digest", "truncated"]
)
def test_rejects_incompatible_or_incomplete_release(tmp_path, tiny_contract, mutation):
    values = dict(tiny_contract)
    first = next(iter(values))
    if mutation == "missing":
        del values[first]
    elif mutation == "duplicate":
        values["base_model.model." + first] = values[first]
    elif mutation == "shape":
        values[first] = values[first][:-1]
    elif mutation == "nan":
        values[first] = torch.full_like(values[first], float("nan"))
    _write_checkpoint(tmp_path, values)
    if mutation == "digest":
        with (tmp_path / "checkpoint-manifest.json").open("a") as handle:
            handle.write(" ")
    elif mutation in ("wrong_stage", "wrong_velocity"):
        before, after = (
            (b'"stage2"', b'"stage1"') if mutation == "wrong_stage" else (b'"data_ward_velocity"', b'"noise"')
        )
        raw = (tmp_path / "checkpoint-manifest.json").read_bytes().replace(before, after)
        (tmp_path / "checkpoint-manifest.json").write_bytes(raw)
        complete = json.loads((tmp_path / "COMPLETE.json").read_text())
        complete["manifest_digest"] = hashlib.blake2s(raw).hexdigest()
        (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    elif mutation == "truncated":
        with (tmp_path / "ema.pt").open("r+b") as handle:
            handle.truncate(20)
    with pytest.raises(ValueError):
        checkpoint.load_stage2_adapter(tmp_path)


class _Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(output_dim * input_dim).reshape(output_dim, input_dim).to(torch.bfloat16) / 100
        )

    def forward(self, value):
        return F.linear(value, self.weight), None


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj, self.attn.out_proj = _Linear(4, 18), _Linear(6, 4)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = _Linear(4, 8), _Linear(4, 4)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([_Block()])


def test_live_sidecars_keep_base_weights_and_obey_native_qkv_swiglu_order(tiny_contract):
    model = _Transformer()
    originals = {name: value.detach().clone() for name, value in model.named_parameters()}
    adapter = checkpoint.Stage2Adapter(tiny_contract, {}, 3900, rank=2, alpha=2)
    checkpoint.install_static_adapter(model, adapter)
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0, 0.2]], dtype=torch.bfloat16)

    def delta(target, *, swap=False):
        A, B = (tiny_contract[f"{target}.lora_{side}.weight"].to(torch.bfloat16) for side in ("A", "B"))
        if swap:
            up, gate = B.chunk(2)
            B = torch.cat((gate, up))
        return F.linear(F.linear(x, A), B)

    actual, bias = model.blocks[0].attn.qkv_proj(x)
    expected_delta = torch.cat([delta("transformer_blocks.0.attn.to_" + side) for side in ("q", "k", "v")], -1)
    torch.testing.assert_close(
        actual, F.linear(x, originals["blocks.0.attn.qkv_proj.weight"]) + expected_delta, rtol=0, atol=0
    )
    assert bias is None
    actual, _ = model.blocks[0].mlp.fc1(x)
    torch.testing.assert_close(
        actual,
        F.linear(x, originals["blocks.0.mlp.fc1.weight"]) + delta("transformer_blocks.0.ff.net.0.proj", swap=True),
        rtol=0,
        atol=0,
    )
    assert len(list(model.named_parameters())) == len(originals)
    assert any("solarwm_lora.A0" in name for name, _ in model.named_buffers())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="already adapted"):
        checkpoint.install_static_adapter(model, adapter)


def test_live_adapter_validation_does_not_partially_install(tiny_contract):
    model = _Transformer()
    model.token_refiner.blocks[-1].mlp.fc2.weight.data = model.token_refiner.blocks[-1].mlp.fc2.weight.float()
    with pytest.raises(ValueError, match="BF16 unsharded"):
        checkpoint.install_static_adapter(model, checkpoint.Stage2Adapter(tiny_contract, {}, 3900, rank=2, alpha=2))
    assert not any(hasattr(module, "solarwm_lora") for module in model.modules())
