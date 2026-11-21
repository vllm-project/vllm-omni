#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fibo-2.0 transformer parity: the correctness oracle for the in-tree port.

Checks that :class:`Fibo2Transformer2DModel` (the vLLM-Omni port) reproduces
the diffusers ``BriaFibo2Transformer2DModel`` on the released
``briaai/Fibo-2.0-dev`` weights, using the fixed inputs and outputs captured
by the Diffusers baseline run (vllm-omni issue #9212, step 1):

* inputs  - first latent_model_input + prompt/negative prompt embeds, seed 42
* outputs - first-step noise predictions (cond / uncond), saved in fp32

The reference data lives in ``ref-data/run1/tensors`` of the baseline; the
weights in the HF cache downloaded for it. Both are needed, so this test runs
locally rather than in CI. Point ``FIBO2_HF_HOME`` at the baseline HF cache
and ``FIBO2_REF_DATA`` at the baseline ref-data directory (defaults below),
then::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
        tests/diffusion/models/bria_fibo2/test_fibo2_transformer_parity.py -v -s

Tolerances mirror the AuK parity oracle: bf16 kernels differ from diffusers'
SDPA in reduction order, so exact bitwise equality is not expected; agreement
within bf16 noise (atol 3e-2 on fp32 outputs, cosine > 0.999) is.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, safe_open

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

FIBO2_HF_HOME = Path(os.environ.get("FIBO2_HF_HOME", "/root/autodl-fs/fibo-baseline/hf-home"))
FIBO2_REF_DATA = Path(os.environ.get("FIBO2_REF_DATA", "/root/autodl-fs/fibo-baseline/ref-data"))

# Tolerances: the port runs end-to-end in bf16 (the deployment dtype), while
# the diffusers baseline ran the same bf16 weights through PyTorch SDPA and
# eager RMSNorm. Kernel reduction orders differ, so bitwise equality is not
# expected. Across all six sampled points (steps 0/15/29 x cond/uncond) the
# cosine sits at 0.99993 +- 0.000003 — a single noise signature — while the
# max deviation is an extreme-value statistic that fluctuates between 0.03
# and 0.13 absolute (0.67%-2.23% of each step's ~5-6 magnitude output scale)
# as the trajectory deepens. A logic error (RoPE planes, bundle indexing,
# modulation, masks) collapses the cosine far below this bar, so cosine is
# the primary oracle; the relative bound is set above the observed spread
# with margin.
FORWARD_COS = 0.9995
FORWARD_REL = 0.025


def _baseline_meta() -> dict:
    return json.loads((FIBO2_REF_DATA / "run1" / "meta.json").read_text())


def _snapshot_dir() -> Path:
    """The snapshot the baseline ran on, pinned by its meta.json model_sha."""
    sha = _baseline_meta()["model_sha"]
    snap = FIBO2_HF_HOME / "hub" / "models--briaai--Fibo-2.0-dev" / "snapshots" / sha
    if not snap.exists():
        pytest.skip(f"baseline revision {sha} not found under {FIBO2_HF_HOME}")
    return snap


@pytest.fixture(scope="module")
def resources():
    """Availability gate: runs before any env/model setup, so absent
    GPU/weights/reference data skip instead of erroring mid-fixture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    tdir = FIBO2_REF_DATA / "run1" / "tensors"
    for name in ("first_step.safetensors", "prompt_embeds.safetensors", "negative_prompt_embeds.safetensors"):
        if not (tdir / name).exists():
            pytest.skip(f"reference data {name} not found under {FIBO2_REF_DATA}")
    if not (FIBO2_REF_DATA / "run1" / "meta.json").exists():
        pytest.skip(f"baseline meta.json not found under {FIBO2_REF_DATA}")
    snap_root = FIBO2_HF_HOME / "hub" / "models--briaai--Fibo-2.0-dev" / "snapshots"
    if not snap_root.exists():
        pytest.skip(f"HF cache not found under {FIBO2_HF_HOME}")
    _snapshot_dir()  # skips unless the pinned revision is cached


@pytest.fixture(scope="module")
def vllm_env(resources):
    """Single-process distributed env so vLLM parallel layers initialize."""
    import vllm.distributed as dist
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        dist.init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29611")
        dist.initialize_model_parallel(tensor_model_parallel_size=1)
        try:
            yield
        finally:
            dist.destroy_model_parallel()
            dist.destroy_distributed_environment()


@pytest.fixture(scope="module")
def model(vllm_env):
    from vllm_omni.diffusion.models.bria_fibo2.fibo2_transformer import (
        Fibo2Transformer2DModel,
    )

    snap = _snapshot_dir()
    config = json.loads((snap / "transformer" / "config.json").read_text())
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Fibo2Transformer2DModel(**config)
    finally:
        torch.set_default_dtype(prev_dtype)
    model = model.to(torch.bfloat16).to("cuda")

    weights = []
    for shard in sorted((snap / "transformer").glob("diffusion_pytorch_model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                weights.append((k, f.get_tensor(k)))
    loaded = model.load_weights(weights)
    model.eval()

    all_params = {n for n, _ in model.named_parameters()}
    missing = all_params - loaded
    assert not missing, f"parameters never loaded from checkpoint: {sorted(missing)}"
    return model


def _first_inputs():
    tdir = FIBO2_REF_DATA / "run1" / "tensors"
    first = load_file(tdir / "first_step.safetensors")
    pos = load_file(tdir / "prompt_embeds.safetensors")
    neg = load_file(tdir / "negative_prompt_embeds.safetensors")
    meta = json.loads((FIBO2_REF_DATA / "run1" / "meta.json").read_text())
    # Baseline scheduler timesteps are in "1000 * sigma" units; the transformer
    # takes t in [0, 1].
    t0 = meta["timing_s"]["timesteps"][0] / 1000.0
    timestep = torch.full((1,), t0, dtype=torch.float32, device="cuda")
    return first, pos, neg, timestep


def _compare(name: str, out: torch.Tensor, ref: torch.Tensor, report: dict):
    out_f, ref_f = out.float().cpu(), ref.float().cpu()
    max_abs = (out_f - ref_f).abs().max().item()
    mean_abs = (out_f - ref_f).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0).item()
    ref_scale = ref_f.abs().max().item()
    rel = max_abs / ref_scale
    ok = (cos >= FORWARD_COS) and (rel <= FORWARD_REL)
    report[name] = {
        "pass": ok,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cosine": cos,
        "ref_abs_max": ref_scale,
        "rel_err": rel,
    }


@pytest.mark.parametrize("branch", ["cond", "uncond"])
def test_first_step_parity(model, branch):
    first, pos, neg, timestep = _first_inputs()
    embeds = pos["prompt_embeds"] if branch == "cond" else neg["negative_prompt_embeds"]
    ref = first[f"noise_pred_{branch}"]

    with torch.no_grad():
        out, _ = model(
            hidden_states=first["latent_model_input"].to("cuda", torch.bfloat16),
            timestep=timestep,
            encoder_hidden_states=embeds.to("cuda", torch.bfloat16),
            encoder_attention_mask=None,
            context_latents=None,
        )

    report: dict = {}
    _compare(branch, out, ref, report)
    print(json.dumps(report, indent=2))
    entry = report[branch]
    assert entry["pass"], (
        f"{branch} forward mismatch: cosine={entry['cosine']:.6f} (min {FORWARD_COS}), "
        f"rel_err={entry['rel_err']:.4f} (max {FORWARD_REL}), max_abs_diff={entry['max_abs_diff']:.4g}"
    )


@pytest.mark.parametrize("step", [15, 29])
@pytest.mark.parametrize("branch", ["cond", "uncond"])
def test_selected_steps_parity(model, step, branch):
    """Mid-trajectory and near-end steps: parity at step 0 alone does not
    prove the whole denoising interval matches. Reference data comes from
    the baseline's selected_steps capture."""
    tdir = FIBO2_REF_DATA / "run1" / "tensors"
    sel_path = tdir / "selected_steps.safetensors"
    if not sel_path.exists():
        pytest.skip("selected_steps.safetensors not in reference data (regenerate baseline)")
    sel = load_file(sel_path)
    prefix = f"step_{step:03d}_"
    keys = [k for k in sel if k.startswith(prefix)]
    assert keys, f"step {step} missing from selected_steps reference"

    if branch == "cond":
        embeds = load_file(tdir / "prompt_embeds.safetensors")["prompt_embeds"]
    else:
        embeds = load_file(tdir / "negative_prompt_embeds.safetensors")["negative_prompt_embeds"]

    with torch.no_grad():
        out, _ = model(
            hidden_states=sel[f"{prefix}latent_model_input"].to("cuda", torch.bfloat16),
            timestep=sel[f"{prefix}timestep"].to("cuda", torch.float32),
            encoder_hidden_states=embeds.to("cuda", torch.bfloat16),
            encoder_attention_mask=None,
            context_latents=None,
        )

    report: dict = {}
    _compare(f"step{step}_{branch}", out, sel[f"{prefix}noise_pred_{branch}"], report)
    print(json.dumps(report, indent=2))
    entry = report[f"step{step}_{branch}"]
    assert entry["pass"], (
        f"step {step} {branch} mismatch: cosine={entry['cosine']:.6f} "
        f"(min {FORWARD_COS}), rel_err={entry['rel_err']:.4f} (max {FORWARD_REL})"
    )


def test_masked_batch_consistency(model):
    """The padding-mask path: a padded batch must reproduce each row's
    unpadded forward. Row 0 is the baseline positive prompt (78 tokens);
    row 1 pads the 46-token null caption to 78 and crosses a gist tier
    (78 -> 256, 300 -> 320 gist tokens) via a synthetic long prompt.

    The reference is each row run alone through the same model, so this
    checks mask semantics (padding must not leak), not kernel parity.
    """
    tdir = FIBO2_REF_DATA / "run1" / "tensors"
    pos = load_file(tdir / "prompt_embeds.safetensors")["prompt_embeds"]  # (1,6,78,7680)
    neg = load_file(tdir / "negative_prompt_embeds.safetensors")["negative_prompt_embeds"]
    torch.manual_seed(1234)
    long_prompt = torch.randn(1, 6, 300, 7680).to(torch.bfloat16)

    text_len = 78
    neg_pad = torch.zeros(1, 6, text_len, 7680, dtype=torch.bfloat16)
    neg_pad[:, :, : neg.shape[2]] = neg.to(torch.bfloat16)
    mask = torch.zeros(2, text_len, dtype=torch.bool)
    mask[0] = True
    mask[1, : neg.shape[2]] = True

    latents = load_file(tdir / "first_step.safetensors")["latent_model_input"].to("cuda", torch.bfloat16)
    latents = latents.repeat(2, 1, 1, 1)
    timestep = torch.full((2,), 1.0, dtype=torch.float32, device="cuda")

    batch_embeds = torch.cat([pos.to("cuda", torch.bfloat16), neg_pad.to("cuda", torch.bfloat16)], dim=0)
    with torch.no_grad():
        out_batch, _ = model(latents, timestep, batch_embeds, encoder_attention_mask=mask.to("cuda"))

    # Row 1 alone, unpadded (its own length) must match the batched row 1.
    with torch.no_grad():
        out_single, _ = model(
            latents[1:2],
            timestep[:1],
            neg.to("cuda", torch.bfloat16),
            encoder_attention_mask=None,
        )
    d = (out_batch[1:2].float() - out_single.float()).abs()
    scale = out_single.float().abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_batch[1:2].flatten().float().cpu(), out_single.flatten().float().cpu(), dim=0
    ).item()
    print(
        json.dumps(
            {"padded_row_vs_single": {"max_abs_diff": d.max().item(), "cosine": cos, "rel": d.max().item() / scale}},
            indent=2,
        )
    )
    assert cos >= 0.999, f"padded row leaked into mask: cosine={cos:.6f}"
    assert d.max().item() / scale <= 0.02, f"padding changed row output: rel={d.max().item() / scale:.4f}"

    # Cross-tier batch: 78-token and 300-token prompts together exercise the
    # gist-padding branch (gist 256 vs 320 -> image first, padded gist last).
    pos_pad300 = torch.zeros(1, 6, 300, 7680, dtype=torch.bfloat16)
    pos_pad300[:, :, :78] = pos.to(torch.bfloat16)
    tier_embeds = torch.cat([pos_pad300, long_prompt], dim=0).to("cuda", torch.bfloat16)
    tier_mask = torch.zeros(2, 300, dtype=torch.bool)
    tier_mask[0, :78] = True
    tier_mask[1] = True
    with torch.no_grad():
        out_tier, _ = model(latents, timestep, tier_embeds, encoder_attention_mask=tier_mask.to("cuda"))
    with torch.no_grad():
        out_tier_single, _ = model(
            latents[0:1], timestep[:1], pos.to("cuda", torch.bfloat16), encoder_attention_mask=None
        )
    d0 = (out_tier[0:1].float() - out_tier_single.float()).abs()
    cos0 = torch.nn.functional.cosine_similarity(
        out_tier[0:1].flatten().float().cpu(), out_tier_single.flatten().float().cpu(), dim=0
    ).item()
    print(
        json.dumps(
            {
                "gist_padded_row_vs_single": {
                    "max_abs_diff": d0.max().item(),
                    "cosine": cos0,
                    "rel": d0.max().item() / out_tier_single.float().abs().max().item(),
                }
            },
            indent=2,
        )
    )
    assert cos0 >= 0.999, f"gist padding leaked: cosine={cos0:.6f}"


def test_gist_count():
    from vllm_omni.diffusion.models.bria_fibo2.fibo2_transformer import (
        compute_gist_count,
    )

    # Text shorter than gist_min_text_len maps to the floor: min_gist tokens.
    assert compute_gist_count([78], 256, 512, 64, 256, 2048) == [256]
    assert compute_gist_count([46], 256, 512, 64, 256, 2048) == [256]
    # At max_text_len the gist saturates.
    assert compute_gist_count([2048], 256, 512, 64, 256, 2048) == [512]
    # Mid-range maps linearly, rounded up to a gist_step multiple.
    assert compute_gist_count([1024], 256, 512, 64, 256, 2048) == [384]


@pytest.mark.parametrize("batch_kind", ["distinct", "mixed_len"])
def test_pipeline_batch_row_correspondence(model, vllm_env, batch_kind):
    """Pipeline-level batching oracle (no engine): rows in a batch must
    correspond to their own prompt/seed. `distinct` batches two different
    captions and seeds; `mixed_len` pads a short caption against the long
    baseline one, exercising the text padding mask. Each row is compared
    against the same request run alone through the same model.
    """
    import torch as _torch

    tdir = FIBO2_REF_DATA / "run1" / "tensors"
    pos = load_file(tdir / "prompt_embeds.safetensors")["prompt_embeds"]
    latents = load_file(tdir / "first_step.safetensors")["latent_model_input"].to("cuda", torch.bfloat16)
    timestep = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")

    torch.manual_seed(11)
    if batch_kind == "distinct":
        # Two different-length captions as the encoder would hand them over:
        # the short row padded to the longest, with a validity mask
        padded = torch.zeros(1, 6, 78, 7680, dtype=torch.bfloat16)
        padded[:, :, :46] = _torch.randn(1, 6, 46, 7680).to(torch.bfloat16)
        other = padded
        masks = torch.ones(2, 78, dtype=torch.bool)
        masks[1, 46:] = False
    else:
        other = torch.zeros(1, 6, 120, 7680, dtype=torch.bfloat16)
        other[:, :, :78] = pos  # short real prefix, long padding tail
        masks = torch.ones(2, 120, dtype=torch.bool)
        masks[0, :78] = True
        masks[0, 78:] = False
        masks[1] = True
        pos = torch.cat([pos, torch.zeros(1, 6, 42, 7680, dtype=torch.bfloat16)], dim=2)

    batch_embeds = torch.cat([pos.to("cuda"), other.to("cuda")], dim=0)

    with torch.no_grad():
        batch_out, _ = model(
            latents.repeat(2, 1, 1, 1),
            torch.full((2,), 1.0, dtype=torch.float32, device="cuda"),
            batch_embeds,
            encoder_attention_mask=masks.to("cuda") if masks is not None else None,
        )
        row0_single, _ = model(
            latents,
            timestep,
            pos.to("cuda"),
            encoder_attention_mask=(masks[0:1].to("cuda") if masks is not None else None),
        )
        row1_single, _ = model(
            latents,
            timestep,
            other.to("cuda"),
            encoder_attention_mask=(masks[1:2].to("cuda") if masks is not None else None),
        )

    for row, single, name in ((0, row0_single, "row0"), (1, row1_single, "row1")):
        a = batch_out[row : row + 1].float().cpu().flatten()
        b = single.float().cpu().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        max_abs = (batch_out[row : row + 1].float() - single.float()).abs().max().item()
        print(json.dumps({f"{name}_vs_single": {"cosine": cos, "max_abs_diff": max_abs}}))
        assert cos >= 0.999, f"{name} drifted from its single run: cosine={cos:.4f}"
