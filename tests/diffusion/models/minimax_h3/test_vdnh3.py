# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3 import vdnh3
from vllm_omni.diffusion.models.minimax_h3.vdnh3 import (
    VDN_CHECKPOINT,
    VDNCheckpoint,
    VDNConfig,
    _checkpoint_dir,
    _delta_factors,
    _outside_window_state,
    _scan,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

TRANSFORM = {
    "type": "hybrid_attention",
    "version": 2,
    "config": {
        "anchor_frames": "both",
        "enable_softmax_gate": True,
        "linear_attention": {
            "a_fp32": True,
            "bridge": "alpha",
            "delta_rule": "vdn_solve",
            "enable_text_state": True,
            "linear_head_dim": 4,
            "short_conv": {"targets": ["k", "v"]},
        },
        "softmax_attention": {"chunk": 5, "radius": 1},
    },
}


def test_delta_factors_apply_the_vdn_solve_rule():
    torch.manual_seed(0)
    if torch.version.hip is not None and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required for the LAPACK-free solve-rule path")
    device = torch.device("cuda" if torch.version.hip is not None else "cpu")
    k = torch.nn.functional.normalize(torch.randn(3, 2, 6, 4, dtype=torch.float64, device=device), dim=-1)
    A = k.transpose(-1, -2) @ k
    B, state = (
        torch.randn(3, 2, 4, 4, dtype=torch.float64, device=device),
        torch.randn(3, 2, 4, 4, dtype=torch.float64, device=device),
    )
    alpha = torch.rand(3, 2, 4, dtype=torch.float64, device=device)
    transition, injection = _delta_factors(A, B, alpha)
    expected = (state * alpha.unsqueeze(-2) + B) @ torch.linalg.inv(
        torch.eye(4, dtype=torch.float64, device=device) + A
    )
    torch.testing.assert_close(state @ transition + injection, expected)


@pytest.mark.parametrize("with_text", [True, False])
def test_outside_window_state_matches_explicit_recurrence(with_text):
    torch.manual_seed(0)
    frames, heads, dim = 7, 2, 3
    transitions = torch.randn(frames, heads, dim, dim, dtype=torch.float64) * 0.5
    injections = torch.randn(frames, heads, dim, dim, dtype=torch.float64)
    alpha = torch.rand(frames, heads, dim, dtype=torch.float64) * 0.5 + 0.5
    text = torch.randn(heads, dim, dim, dtype=torch.float64) if with_text else None
    start = torch.zeros(heads, dim, dim, dtype=torch.float64) if text is None else text
    # Every frame lies inside its own window; some windows touch or pass the clip ends.
    bounds = [(-2, 1), (0, 2), (1, 4), (3, 6), (4, 8), (-1, 7), (5, 9)]
    prefix = _scan(transitions, injections, start, reverse=False)
    suffix = _scan(transitions, injections, start, reverse=True)
    got = _outside_window_state(prefix, suffix, alpha, bounds, text_state=text, bridge="alpha")

    for t, (lo, hi) in enumerate(bounds):
        left = start.clone()
        for f in range(max(lo, 0)):
            left = left @ transitions[f] + injections[f]
        right = start.clone()
        for f in reversed(range(min(hi, frames - 1) + 1, frames)):
            right = right @ transitions[f] + injections[f]
        if text is None:
            left = left if lo > 0 else torch.zeros_like(left)
            right = right if hi < frames - 1 else torch.zeros_like(right)
        # Bridge each side to t through the window's frames, per key channel.
        left = left * alpha[max(lo, 0) : t + 1].prod(0).unsqueeze(1)
        right = right * alpha[t : min(hi, frames - 1) + 1].prod(0).unsqueeze(1)
        torch.testing.assert_close(got[t], left + right)


def _write_artifact(root, *, delta_rule="vdn_solve", turbo_num_steps=8):
    transform = json.loads(json.dumps(TRANSFORM))
    transform["config"]["linear_attention"]["delta_rule"] = delta_rule
    lora = {"alpha": 2, "rank": 2, "alpha_pattern": {"transformer_blocks.0.adaln_proj.linear": 4}}
    (root / "linear_branch").mkdir(parents=True)
    (root / "model_spec.json").write_text(
        json.dumps({"base": {"class_name": "MiniMaxH3Transformer3DModel"}, "transforms": [transform]})
    )
    metadata = {} if turbo_num_steps is None else {"turbo_num_steps": turbo_num_steps}
    (root / "metadata.json").write_text(json.dumps({"kind": "weights", "metadata": metadata}))
    save_file(
        {
            "transformer_blocks.0.attn.linear_attention.norm.weight": torch.ones(4),
            "transformer_blocks.0.attn.softmax_gate.up.weight": torch.ones(2, 6),
        },
        root / "linear_branch" / "model.safetensors",
    )
    torch.manual_seed(0)
    factors: dict[str, list[torch.Tensor]] = {}
    for adapter, modules in {
        "default": ["transformer_blocks.0.attn.orig.to_k", "transformer_blocks.0.ff.net.0.proj"],
        "turbo": ["transformer_blocks.0.attn.orig.to_k", "transformer_blocks.0.adaln_proj.linear"],
    }.items():
        tensors: dict[str, torch.Tensor] = {}
        for module in modules:
            out = {"to_k": 8, "proj": 10}.get(module.rsplit(".", 1)[-1], 12)
            a, b = torch.randn(2, 6), torch.randn(out, 2)
            tensors |= {f"{module}.lora_A.{adapter}.weight": a, f"{module}.lora_B.{adapter}.weight": b}
            factors.setdefault(module, []).append(b @ a)
        (root / "adapters" / adapter).mkdir(parents=True)
        # Both release spellings of the adapter spec.
        spec = "adapter_spec.json" if adapter == "turbo" else "adapter_config.json"
        (root / "adapters" / adapter / spec).write_text(json.dumps({"config": lora}))
        save_file(tensors, root / "adapters" / adapter / "adapter_model.safetensors")
    return factors


def test_vdn_checkpoint_fuses_loras_into_native_layout(tmp_path):
    factors = _write_artifact(tmp_path)
    ckpt = VDNCheckpoint.from_path(tmp_path, head_dim=4)
    assert ckpt.num_inference_steps == 8
    assert ckpt.config == VDNConfig(
        chunk=5,
        radius=1,
        anchor_frames="both",
        enable_softmax_gate=True,
        enable_text_state=True,
        bridge="alpha",
        short_conv=("k", "v"),
    )
    qkv = torch.zeros(24, 6)  # 2 heads x (q, k, v) x head_dim 4, grouped per head
    fc1 = torch.zeros(10, 6)  # native [gate; up]
    adaln = torch.zeros(12, 6)
    # On a GPU host the fused weights come back on the device; compare on the CPU.
    fused = {
        name: weight.cpu()
        for name, weight in ckpt.apply(
            [
                ("blocks.0.attn.qkv_proj.weight", qkv),
                ("blocks.0.mlp.fc1.weight", fc1),
                ("blocks.0.adaln_proj.linear.weight", adaln),
                ("blocks.0.mlp.fc2.weight", torch.zeros(6, 5)),
            ]
        )
    }
    grouped = fused["blocks.0.attn.qkv_proj.weight"].view(2, 3, 4, 6)
    k_delta = sum(factors["transformer_blocks.0.attn.orig.to_k"]).view(2, 4, 6)
    torch.testing.assert_close(grouped[:, 1], k_delta)
    assert torch.all(grouped[:, 0] == 0) and torch.all(grouped[:, 2] == 0)
    up, gate = factors["transformer_blocks.0.ff.net.0.proj"][0].chunk(2)  # diffusers [up; gate]
    torch.testing.assert_close(fused["blocks.0.mlp.fc1.weight"], torch.cat([gate, up]))
    # alpha_pattern 4 over rank 2
    torch.testing.assert_close(
        fused["blocks.0.adaln_proj.linear.weight"], 2 * factors["transformer_blocks.0.adaln_proj.linear"][0]
    )
    assert torch.all(fused["blocks.0.mlp.fc2.weight"] == 0)
    branch = {"blocks.0.attn.vdn.linear_attention.norm.weight", "blocks.0.attn.vdn.softmax_gate.up.weight"}
    assert branch <= set(fused)
    ckpt.validate(set(fused), branch)


def test_vdn_checkpoint_validation(tmp_path):
    _write_artifact(tmp_path)
    ckpt = VDNCheckpoint.from_path(tmp_path, head_dim=4)
    streamed = dict(ckpt.apply([("blocks.0.attn.qkv_proj.weight", torch.zeros(24, 6))]))
    with pytest.raises(ValueError, match="never provided"):
        ckpt.validate(set(streamed), set())

    sampling = OmniDiffusionSamplingParams(num_inference_steps=50)
    with pytest.raises(OmniClientError, match="8-step student"):
        ckpt.check_request(sampling, "t2va")
    with pytest.raises(OmniClientError, match="serves"):
        ckpt.check_request(OmniDiffusionSamplingParams(num_inference_steps=8), "ref2va")
    with pytest.raises(OmniClientError, match="Euler"):
        ckpt.check_request(
            OmniDiffusionSamplingParams(num_inference_steps=8, extra_args={"sampler": "res_multistep"}), "t2va"
        )
    ckpt.check_request(OmniDiffusionSamplingParams(num_inference_steps=8, extra_args={"sampler": "euler"}), "t2va")


def test_vdn_checkpoint_detection(tmp_path):
    assert VDNCheckpoint.from_path(tmp_path, head_dim=4) is None
    _write_artifact(tmp_path / "unsupported", delta_rule="other")
    with pytest.raises(ValueError, match="delta_rule"):
        VDNCheckpoint.from_path(tmp_path / "unsupported", head_dim=4)
    # Only the distilled student is served.
    _write_artifact(tmp_path / "undistilled", turbo_num_steps=None)
    with pytest.raises(ValueError, match="8-step DMD"):
        VDNCheckpoint.from_path(tmp_path / "undistilled", head_dim=4)


def test_vdn_checkpoint_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = tmp_path / "release"
    _write_artifact(release / VDN_CHECKPOINT)
    assert _checkpoint_dir(str(release / VDN_CHECKPOINT), from_hub=False) == release / VDN_CHECKPOINT
    assert _checkpoint_dir(str(release), from_hub=False) == release / VDN_CHECKPOINT
    assert _checkpoint_dir("OpenVDN/vdn-minimax-h3", from_hub=False) is None

    prefetched = []
    monkeypatch.setattr(
        vdnh3, "prefetch_subfolders", lambda repo, subfolders, **_: prefetched.append((repo, subfolders))
    )
    monkeypatch.setattr(vdnh3.hf_api(), "snapshot_download", lambda **_: str(release))
    assert _checkpoint_dir("OpenVDN/vdn-minimax-h3", from_hub=True) == release / VDN_CHECKPOINT
    assert prefetched == [("OpenVDN/vdn-minimax-h3", [VDN_CHECKPOINT])]
