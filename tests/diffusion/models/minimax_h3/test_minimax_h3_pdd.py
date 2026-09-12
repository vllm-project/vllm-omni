# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only tests for the MiniMax-H3 PDD Ref2VA loader.

These verify the artifact parsing, plan math, head-bank fusion, and trunk
LoRA key remapping without requiring a GPU or an instantiated pipeline. Run:

    pytest tests/diffusion/models/minimax_h3/test_minimax_h3_pdd.py -q

The handful of tests that load a real PDD artifact skip automatically unless
``MINIMAX_H3_PDD_LORA_DIR`` points at a directory containing the released
``MiniMax-H3-{Ref2VA,FL2VA}-Acc-8Step.safetensors`` files.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3.pdd import (
    PDDAdapter,
    PDDConfig,
    PDDParallelHead,
    _build_pdd_plans,
    _parse_pdd_metadata,
    _validate_and_convert_tensors,
    load_minimax_h3_pdd_lora,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_PDD_LORA_DIR = Path(os.environ.get("MINIMAX_H3_PDD_LORA_DIR", "/nonexistent"))
PDD_CKPT = _PDD_LORA_DIR / "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"


# ---------------------------------------------------------------------------
# Plan math
# ---------------------------------------------------------------------------


def test_plan_matrix_shape_and_row_sum():
    vp, ap = _build_pdd_plans(32, 4, 12.0, 3.0)
    assert vp.shape == (8, 32) and ap.shape == (8, 32)
    assert torch.allclose(vp.sum(dim=1), torch.ones(8), atol=1e-6)
    assert torch.allclose(ap.sum(dim=1), torch.ones(8), atol=1e-6)
    # Plan is block-sparse: entries outside the current block must be zero.
    for k in range(8):
        start = k * 4
        outside = torch.cat([vp[k, :start], vp[k, start + 4 :]]) if start + 4 < 32 else vp[k, :start]
        if outside.numel() > 0:
            assert outside.abs().max().item() < 1e-8, f"block {k} leaks outside"
        outside_a = torch.cat([ap[k, :start], ap[k, start + 4 :]]) if start + 4 < 32 else ap[k, :start]
        if outside_a.numel() > 0:
            assert outside_a.abs().max().item() < 1e-8, f"audio block {k} leaks outside"


def test_sigma_block_boundaries_match_9point_schedule():
    """The 9-point (8-NFE) schedule must equal every 4th entry of the 32-step
    schedule -- this is the geometric condition that lets PDD's block-mean
    velocity fuse cleanly into one Euler step."""
    for shift in (12.0, 3.0):
        s33 = minimax_h3_time_shift_sigmas(num_steps=33, shift_scale=shift)
        s9 = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=shift)
        assert len(s33) == 33 and len(s9) == 9
        for i, (a, b) in enumerate(zip(s33[::4], s9)):
            assert abs(a - b) < 1e-6, f"shift={shift} boundary {i}: {a} vs {b}"


def test_plan_matches_reference_implementation():
    """Numerical equivalence to the released minimax_h3_pdd.py."""

    def _ref_shifted_sigma(shift, sigma):
        return shift * sigma / (1 + (shift - 1) * sigma)

    def _ref_grid(shift, num_steps):
        sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
        return 1.0 - _ref_shifted_sigma(shift, sigma)

    def _ref_plan(step_sizes, start, block):
        out = torch.zeros(1, step_sizes.shape[0])
        span = step_sizes[start : start + block].sum()
        out[0, start : start + block] = step_sizes[start : start + block] / span
        return out

    for shift in (12.0, 3.0):
        our_v_diff = _build_pdd_plans(32, 4, shift, 3.0 if shift == 12.0 else 12.0)[0]
        ref_grid = _ref_grid(shift, 32)
        ref_steps = ref_grid.diff()
        for k in range(8):
            ref_p = _ref_plan(ref_steps, k * 4, 4).float()
            assert torch.allclose(our_v_diff[k : k + 1], ref_p, atol=1e-6)


# ---------------------------------------------------------------------------
# PDDParallelHead
# ---------------------------------------------------------------------------


def test_parallel_head_default_plan_matches_source():
    torch.manual_seed(0)
    src = nn.Linear(5376, 96, bias=True).float()
    head = PDDParallelHead(src, 32)
    x = torch.randn(3, 5376)
    out, extra_bias = head(x)
    assert extra_bias is None
    assert torch.allclose(out, src(x), atol=1e-5)


def test_parallel_head_averaging_plan_is_manual_mean():
    torch.manual_seed(1)
    src = nn.Linear(5376, 32, bias=True).float()
    head = PDDParallelHead(src, 32)
    # Initialize the bank with distinct weights so averaging is observable.
    head.weight.data = torch.randn_like(head.weight)
    head.bias.data = torch.randn_like(head.bias)
    plan = torch.zeros(1, 32)
    plan[0, :4] = 0.25
    head.set_plan(plan)
    x = torch.randn(2, 5376)
    w_mean = head.weight[:4].mean(0)
    b_mean = head.bias[:4].mean(0)
    assert torch.allclose(head(x)[0], F.linear(x, w_mean, b_mean), atol=1e-5)


def test_parallel_head_reset_plan_restores_base_after_artifact_load():
    """Regression: deactivation used to leave a fused (non-identity) plan in
    place, so a later non-PDD request on the same DiT would silently keep
    running through the last PDD step's fused head instead of the base
    weight."""
    torch.manual_seed(2)
    src = nn.Linear(5376, 32, bias=True).float()
    head = PDDParallelHead(src, 32)
    # Loading an artifact replaces ALL bank entries, including head 0.
    head.weight.data.copy_(torch.randn_like(head.weight))
    head.bias.data.copy_(torch.randn_like(head.bias))
    plan = torch.zeros(1, 32)
    plan[0, 1] = 1.0
    head.set_plan(plan)
    x = torch.randn(2, 5376)
    assert not torch.allclose(head(x)[0], src(x), atol=1e-5)
    head.reset_plan()
    assert torch.allclose(head(x)[0], src(x), atol=1e-5)


def test_parallel_head_rejects_bad_plan_shape():
    src = nn.Linear(16, 4, bias=True).float()
    head = PDDParallelHead(src, 4)
    with pytest.raises(ValueError):
        head.set_plan(torch.zeros(1, 3))


def test_parallel_head_no_bias_supported():
    src = nn.Linear(16, 4, bias=False).float()
    head = PDDParallelHead(src, 4)
    assert head.bias is None
    x = torch.randn(2, 16)
    # Default plan is head 0; should match src(x).
    assert torch.allclose(head(x)[0], src(x), atol=1e-6)


def test_parallel_head_plan_buffer_shares_weight_device():
    """Regression: the plan buffer used to be created on the default (CPU)
    device while the bank lives wherever the source layer already was.  On
    GPU that made the fusing einsum a cross-device bmm, so every PDD request
    died with "mat2 is on cuda:N, different from other tensors on cpu"."""
    src = nn.Linear(16, 4, bias=True).float()
    head = PDDParallelHead(src, 4)
    assert head.plan.device == head.weight.device


def test_parallel_head_does_not_reshard_an_already_sharded_source():
    """Regression: ColumnParallelLinear hands us the *local* shard already
    ([out_local, in]), not the full [out, in].  A `>=` guard re-narrowed it at
    offset tp_rank*out_local and blew past the end of the tensor."""

    class FakeColumnParallel(nn.Module):
        """Mimics a tp=2 rank-1 ColumnParallelLinear: weight is pre-sharded."""

        def __init__(self, out_local: int, in_features: int) -> None:
            super().__init__()
            self.input_size_per_partition = in_features
            self.output_size_per_partition = out_local
            self.tp_rank = 1
            self.tp_size = 2
            self.gather_output = True
            self.weight = nn.Parameter(torch.randn(out_local, in_features).float())
            self.bias = nn.Parameter(torch.randn(out_local).float())

    src = FakeColumnParallel(out_local=48, in_features=5376)
    head = PDDParallelHead(src, 32)
    assert tuple(head.weight.shape) == (32, 48, 5376)
    assert tuple(head.bias.shape) == (32, 48)
    # Every copy is initialized from the local shard verbatim.
    assert torch.allclose(head.weight[0], src.weight)
    assert torch.allclose(head.bias[0], src.bias)


def test_install_heads_covers_every_dit_not_just_the_first():
    """Regression: PDDAdapter carried a single ``_heads_installed`` bool, so in
    the ``combined`` partition the second DiT (transformers_ref) silently kept
    its plain ColumnParallelLinear and load_head_bank then tried to copy a
    (32, out, in) bank into a 2-D weight."""

    def _make_dit():
        dit = nn.Module()
        dit.final_layer = nn.Module()
        dit.final_layer.video_out = nn.Linear(5376, 96, bias=True).float()
        dit.final_layer.audio_out = nn.Linear(5376, 32, bias=True).float()
        return dit

    cfg = PDDConfig()
    vp, ap = cfg.plans()
    adapter = PDDAdapter(config=cfg, lora_id=1, video_plans=vp, audio_plans=ap)
    transformer, transformers_ref = _make_dit(), _make_dit()
    for dit in (transformer, transformers_ref):
        adapter.install_heads(dit)
        assert isinstance(dit.final_layer.video_out, PDDParallelHead)
        assert isinstance(dit.final_layer.audio_out, PDDParallelHead)

    # And loading the bank into both must work (this is what used to raise).
    head_weights = {
        "video_out": torch.randn(cfg.num_steps, 96, 5376),
        "audio_out": torch.randn(cfg.num_steps, 32, 5376),
    }
    head_biases = {
        "video_out": torch.randn(cfg.num_steps, 96),
        "audio_out": torch.randn(cfg.num_steps, 32),
    }
    for dit in (transformer, transformers_ref):
        adapter.load_head_bank(dit, head_weights, head_biases)
        assert torch.allclose(dit.final_layer.video_out.weight, head_weights["video_out"])
        assert torch.allclose(dit.final_layer.audio_out.weight, head_weights["audio_out"])

    # Re-installing is a no-op rather than wrapping a head inside a head.
    adapter.install_heads(transformer)
    assert torch.allclose(transformer.final_layer.video_out.weight, head_weights["video_out"])


def test_disarm_restores_head0_plan_set_by_arm_step():
    """Regression: deactivation cleared PDDAdapter's own bookkeeping but never
    touched the installed heads' plan buffer, so a later request reusing this
    DiT without the adapter (no-LoRA, Turbo, or a different PDD artifact)
    would keep running through this adapter's last-armed per-step plan."""

    def _make_dit():
        dit = nn.Module()
        dit.final_layer = nn.Module()
        dit.final_layer.video_out = nn.Linear(5376, 96, bias=True).float()
        dit.final_layer.audio_out = nn.Linear(5376, 32, bias=True).float()
        return dit

    cfg = PDDConfig()
    vp, ap = cfg.plans()
    adapter = PDDAdapter(config=cfg, lora_id=1, video_plans=vp, audio_plans=ap)
    dit = _make_dit()
    adapter.install_heads(dit)
    adapter.arm_step(dit, 1)
    default_plan = torch.zeros(1, cfg.num_steps)
    default_plan[0, 0] = 1.0
    assert not torch.allclose(dit.final_layer.video_out.plan, default_plan)
    assert not torch.allclose(dit.final_layer.audio_out.plan, default_plan)

    adapter.disarm(dit)
    assert torch.allclose(dit.final_layer.video_out.plan, default_plan)
    assert torch.allclose(dit.final_layer.audio_out.plan, default_plan)


def test_diffuse_accepts_pdd_adapter_but_build_denoise_inputs_does_not():
    """Regression: ``forward`` passed ``pdd_adapter`` straight through to
    ``_build_denoise_inputs``, which has no such parameter.  ``diffuse`` owns
    the argument (it steers per-step head arming); the input builder must not
    see it."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _MINIMAX_H3_DENOISE_INPUT_KEYS,
        MiniMaxH3Pipeline,
    )

    diffuse_params = inspect.signature(MiniMaxH3Pipeline.diffuse).parameters
    build_params = inspect.signature(MiniMaxH3Pipeline._build_denoise_inputs).parameters
    assert "pdd_adapter" in diffuse_params
    assert "pdd_adapter" not in build_params
    assert "pdd_adapter" not in _MINIMAX_H3_DENOISE_INPUT_KEYS
    # Everything _denoise_kwargs selects must be a real builder argument.
    assert set(_MINIMAX_H3_DENOISE_INPUT_KEYS) <= set(build_params)


# ---------------------------------------------------------------------------
# Artifact parsing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_parse_pdd_metadata():
    from safetensors import safe_open

    with safe_open(PDD_CKPT, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
    cfg = _parse_pdd_metadata(md)
    assert cfg.num_steps == 32
    assert cfg.block_size == 4
    assert cfg.rank == 64
    assert abs(cfg.alpha - 64.0) < 1e-6
    assert cfg.nfe == 8
    assert cfg.sigma_points == 9


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_validate_and_convert_tensors_against_real_artifact():
    from safetensors import safe_open

    with safe_open(PDD_CKPT, framework="pt", device="cpu") as f:
        cfg = _parse_pdd_metadata(f.metadata() or {})
        trunk, hw, hb = _validate_and_convert_tensors(f, cfg)
    # 50 DiT blocks x 7 targets + 2 token-refiner blocks x 6 targets = 362 modules x 2 sides
    expected = (50 * 7 + 2 * 6) * 2
    assert len(trunk) == expected, f"trunk tensor count {len(trunk)} != {expected}"
    assert tuple(hw["video_out"].shape) == (32, 96, 5376)
    assert tuple(hw["audio_out"].shape) == (32, 32, 5376)
    assert tuple(hb["video_out"].shape) == (32, 96)
    assert tuple(hb["audio_out"].shape) == (32, 32)
    assert hw["video_out"].dtype == torch.float32
    # Spot-check adaln targets are remapped and shaped correctly.
    fc1_up = [v for k, v in trunk.items() if "ff.net.0.proj.lora_up" in k][0]
    ad_down = [v for k, v in trunk.items() if "adaln_proj.linear.lora_down" in k][0]
    ad_up = [v for k, v in trunk.items() if "adaln_proj.linear.lora_up" in k][0]
    assert tuple(fc1_up.shape) == (2 * 14336, 64)  # post gate/value swap
    assert tuple(ad_down.shape) == (64, 2688)  # time_embed_dim input
    assert tuple(ad_up.shape) == (96768, 64)  # 18*5376 block adaln output
    # Trunk keys end with .weight (required by vLLM's parser).
    assert all(k.endswith(".weight") for k in trunk.keys())


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_lora_model_from_tensors_accepts_pdd_trunk():
    """Full trunk LoRA round-trips through LoRAModel.from_lora_tensors with the
    PDD weights mapper -- catches key-suffix / fc1-pack / target-regex bugs."""
    from safetensors import safe_open
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    from vllm_omni.diffusion.models.minimax_h3.pdd import (
        _PDD_ALPHA,
        _PDD_RANK,
        _PDD_TRUNK_TARGET_PATTERN,
        _PDD_WEIGHTS_MAPPER,
        _pack_pdd_fc1,
    )

    with safe_open(PDD_CKPT, framework="pt", device="cpu") as f:
        cfg = _parse_pdd_metadata(f.metadata() or {})
        trunk, _hw, _hb = _validate_and_convert_tensors(f, cfg)
    ph = PEFTHelper.from_dict({"r": _PDD_RANK, "lora_alpha": _PDD_ALPHA, "target_modules": _PDD_TRUNK_TARGET_PATTERN})
    lm = LoRAModel.from_lora_tensors(
        lora_model_id=9001,
        tensors=trunk,
        peft_helper=ph,
        device="cpu",
        dtype=torch.bfloat16,
        weights_mapper=_PDD_WEIGHTS_MAPPER,
    )
    _pack_pdd_fc1(lm)
    # Expect 50*7 + 2*6 = 362 modules, with block adaln present and fc1 packed.
    assert len(lm.loras) == 362, len(lm.loras)
    assert any("adaln_proj.linear" in k for k in lm.loras), "adaln targets missing"
    assert any(k.endswith(".mlp.fc1") for k in lm.loras), "fc1 not packed"


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_load_minimax_h3_pdd_lora_refuses_non_ref2va():
    # "combined" is accepted (it is what the ref2va server actually reports
    # when the text encoder and both DiTs are colocated); anything else is not.
    req = LoRARequest(lora_int_id=9002, lora_name="pdd", lora_path=str(PDD_CKPT))
    with pytest.raises(ValueError, match="ref2va"):
        load_minimax_h3_pdd_lora(
            partition="fl2va",
            lora_request=req,
            lora_path=str(PDD_CKPT),
            dtype=torch.bfloat16,
        )


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_load_minimax_h3_pdd_lora_accepts_combined_partition():
    """Regression: the production ref2va deployment reports partition
    "combined", and the loader used to hard-reject anything but "ref2va",
    so the very first GPU request failed before touching the model."""
    req = LoRARequest(lora_int_id=9004, lora_name="pdd", lora_path=str(PDD_CKPT))
    loaded = load_minimax_h3_pdd_lora(
        partition="combined",
        lora_request=req,
        lora_path=str(PDD_CKPT),
        dtype=torch.bfloat16,
    )
    assert loaded is not None


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_load_minimax_h3_pdd_lora_returns_none_for_non_pdd_path(tmp_path):
    # A path that doesn't contain the PDD filename returns None (PEFT fallback).
    assert (
        load_minimax_h3_pdd_lora(
            partition="ref2va",
            lora_request=LoRARequest(lora_int_id=9003, lora_name="x", lora_path=str(tmp_path)),
            lora_path=str(tmp_path),
            dtype=torch.bfloat16,
        )
        is None
    )


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_load_minimax_h3_pdd_lora_full_return_shape():
    from vllm.lora.lora_model import LoRAModel

    req = LoRARequest(lora_int_id=9004, lora_name="pdd", lora_path=str(PDD_CKPT))
    loaded = load_minimax_h3_pdd_lora(
        partition="ref2va", lora_request=req, lora_path=str(PDD_CKPT), dtype=torch.bfloat16
    )
    assert loaded is not None
    lora_model, peft_helper, cfg, hw, hb = loaded
    assert isinstance(lora_model, LoRAModel)
    assert isinstance(cfg, PDDConfig)
    assert cfg.nfe == 8
    assert peft_helper.r == 64
    assert tuple(hw["video_out"].shape) == (32, 96, 5376)
    assert tuple(hb["audio_out"].shape) == (32, 32)


# ---------------------------------------------------------------------------
# Per-variant routing (Ref2VA vs FL2VA)
#
# Alibaba ships one artifact per partition. They are byte-different but
# structurally identical, so nothing in the file says which DiT it belongs to
# except its name -- and in a `combined` deployment `transformer` holds FL2VA
# while `transformers_ref` holds Ref2VA. A component-blind target pattern
# therefore bound the Ref2VA trunk delta onto the FL2VA DiT (while the head
# bank went to the right one), which no shape check can catch.
# ---------------------------------------------------------------------------

FL2VA_CKPT = PDD_CKPT.with_name("MiniMax-H3-FL2VA-Acc-8Step.safetensors")


def test_variant_registry_maps_each_release_to_its_own_dit():
    from vllm_omni.diffusion.models.minimax_h3.pdd import (
        _PDD_VARIANTS_BY_NAME,
    )

    ref2va = _PDD_VARIANTS_BY_NAME["ref2va"]
    fl2va = _PDD_VARIANTS_BY_NAME["fl2va"]
    # Combined: two DiTs resident, each artifact must pick its own.
    assert ref2va.dit_component("combined") == "transformers_ref"
    assert fl2va.dit_component("combined") == "transformer"
    # Single-partition deployments load their own DiT as `transformer`.
    assert ref2va.dit_component("ref2va") == "transformer"
    assert fl2va.dit_component("fl2va") == "transformer"
    # Task ownership is disjoint; t2va rides the FL2VA DiT.
    assert ref2va.tasks == frozenset({"ref2va"})
    assert fl2va.tasks == frozenset({"fl2va", "t2va"})
    assert not (ref2va.tasks & fl2va.tasks)


@pytest.mark.skipif(not (PDD_CKPT.is_file() and FL2VA_CKPT.is_file()), reason="PDD checkpoints not present")
@pytest.mark.parametrize(
    "ckpt,partition,expected_component",
    [
        (PDD_CKPT, "combined", "transformers_ref"),
        (PDD_CKPT, "ref2va", "transformer"),
        (FL2VA_CKPT, "combined", "transformer"),
        (FL2VA_CKPT, "fl2va", "transformer"),
    ],
)
def test_trunk_lora_keys_are_anchored_on_the_variants_own_dit(ckpt, partition, expected_component):
    """Every trunk key must match the variant's own component-anchored pattern,
    and none may match the other DiT's."""
    import re

    from vllm_omni.diffusion.models.minimax_h3.pdd import _trunk_target_pattern

    req = LoRARequest(lora_int_id=9101, lora_name="pdd", lora_path=str(ckpt))
    loaded = load_minimax_h3_pdd_lora(partition=partition, lora_request=req, lora_path=str(ckpt), dtype=torch.bfloat16)
    assert loaded is not None
    _lm, _ph, cfg, _hw, _hb = loaded
    assert cfg.dit_component == expected_component
    assert len(_lm.loras) == 362
    own = cfg.trunk_target_pattern
    other = _trunk_target_pattern("transformer" if expected_component == "transformers_ref" else "transformers_ref")
    assert all(re.search(own, name) for name in _lm.loras)
    assert not any(re.search(other, name) for name in _lm.loras)
    assert all(name.startswith(f"{expected_component}.") for name in _lm.loras)


@pytest.mark.skipif(not (PDD_CKPT.is_file() and FL2VA_CKPT.is_file()), reason="PDD checkpoints not present")
def test_the_two_releases_are_not_interchangeable():
    """Same shapes, different weights -- so only the name/metadata can route
    them, and the loader must not treat one as the other."""
    from safetensors import safe_open

    with safe_open(PDD_CKPT, framework="pt", device="cpu") as a:
        wa = a.get_tensor("proj_out.weight")
    with safe_open(FL2VA_CKPT, framework="pt", device="cpu") as b:
        wb = b.get_tensor("proj_out.weight")
    assert wa.shape == wb.shape
    assert not torch.equal(wa, wb)


@pytest.mark.skipif(not FL2VA_CKPT.is_file(), reason="FL2VA checkpoint not present")
def test_fl2va_artifact_takes_the_pdd_path_not_the_peft_fallback():
    """Regression: the loader used to recognise PDD by the *Ref2VA* filename,
    so the FL2VA artifact returned None and fell through to the generic PEFT
    loader -- dropping the head bank while the adapter still pinned 9 steps."""
    req = LoRARequest(lora_int_id=9102, lora_name="pdd", lora_path=str(FL2VA_CKPT))
    loaded = load_minimax_h3_pdd_lora(
        partition="combined", lora_request=req, lora_path=str(FL2VA_CKPT), dtype=torch.bfloat16
    )
    assert loaded is not None
    _lm, _ph, cfg, hw, hb = loaded
    assert cfg.variant == "fl2va"
    assert cfg.nfe == 8
    assert tuple(hw["video_out"].shape) == (32, 96, 5376)
    assert tuple(hb["audio_out"].shape) == (32, 32)


@pytest.mark.skipif(not PDD_CKPT.is_file(), reason="PDD checkpoint not present")
def test_ref2va_artifact_is_refused_on_an_fl2va_only_deployment():
    req = LoRARequest(lora_int_id=9103, lora_name="pdd", lora_path=str(PDD_CKPT))
    with pytest.raises(ValueError, match="partition holding the ref2va DiT"):
        load_minimax_h3_pdd_lora(partition="fl2va", lora_request=req, lora_path=str(PDD_CKPT), dtype=torch.bfloat16)


@pytest.mark.skipif(not FL2VA_CKPT.is_file(), reason="FL2VA checkpoint not present")
def test_fl2va_artifact_is_refused_on_a_ref2va_only_deployment():
    req = LoRARequest(lora_int_id=9104, lora_name="pdd", lora_path=str(FL2VA_CKPT))
    with pytest.raises(ValueError, match="partition holding the fl2va DiT"):
        load_minimax_h3_pdd_lora(partition="ref2va", lora_request=req, lora_path=str(FL2VA_CKPT), dtype=torch.bfloat16)


@pytest.mark.skipif(not (PDD_CKPT.is_file() and FL2VA_CKPT.is_file()), reason="PDD checkpoints not present")
def test_release_directory_holding_both_variants_is_refused():
    """Silently picking one by iteration order would accelerate half the
    traffic with the wrong distillation."""
    from vllm_omni.diffusion.models.minimax_h3.pdd import _select_pdd_file

    with pytest.raises(ValueError, match="point lora.path at a single file"):
        _select_pdd_file(PDD_CKPT.parent)


def test_pdd_named_file_without_a_head_bank_raises_instead_of_falling_back(tmp_path):
    """A file the caller believes is PDD but which has no per-step bank must be
    a hard error: the caller still pins 9 steps, and 9 undistilled steps is a
    wasted 3-minute request that returns garbage, not an error."""
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.minimax_h3.pdd import _PDD_VARIANTS_BY_NAME

    fake = tmp_path / _PDD_VARIANTS_BY_NAME["fl2va"].filename
    save_file({"transformer_blocks.0.attn.to_q.lora_down": torch.zeros(64, 5376)}, str(fake))
    with pytest.raises(ValueError, match="no per-step head bank"):
        load_minimax_h3_pdd_lora(
            partition="combined",
            lora_request=LoRARequest(lora_int_id=9105, lora_name="x", lora_path=str(fake)),
            lora_path=str(fake),
            dtype=torch.bfloat16,
        )


def test_unnamed_pdd_artifact_is_refused_rather_than_guessed(tmp_path):
    """A correctly-shaped head bank under an unknown name cannot be routed --
    the two releases are structurally identical, so guessing would be a 50%
    chance of binding to the wrong DiT."""
    from safetensors.torch import save_file

    fake = tmp_path / "some-pdd-artifact.safetensors"
    save_file(
        {
            "proj_out.weight": torch.zeros(32, 96, 5376, dtype=torch.bfloat16),
            "audio_proj_out.weight": torch.zeros(32, 32, 5376, dtype=torch.bfloat16),
        },
        str(fake),
    )
    with pytest.raises(ValueError, match="only signal for which DiT"):
        load_minimax_h3_pdd_lora(
            partition="combined",
            lora_request=LoRARequest(lora_int_id=9106, lora_name="x", lora_path=str(fake)),
            lora_path=str(fake),
            dtype=torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# Pipeline-side routing: the DiT the trunk binds to must be the DiT that
# actually runs the request.
# ---------------------------------------------------------------------------


def _fake_pipeline(partition: str):
    """Minimal stand-in exposing just what the two routing helpers touch."""
    from types import SimpleNamespace

    fake = SimpleNamespace(partition=partition, transformer=object())
    if partition in ("ref2va", "combined"):
        # A ref2va-only deployment loads its DiT as `transformer`; only the
        # combined deployment has a second module.
        if partition == "combined":
            fake.transformers_ref = object()
    return fake


@pytest.mark.parametrize(
    "variant_name,partition,task",
    [
        ("ref2va", "combined", "ref2va"),
        ("ref2va", "ref2va", "ref2va"),
        ("fl2va", "combined", "fl2va"),
        ("fl2va", "combined", "t2va"),
        ("fl2va", "fl2va", "fl2va"),
    ],
)
def test_trunk_binds_to_the_dit_that_serves_the_task(variant_name, partition, task):
    """The invariant the component-anchored pattern exists to hold:
    ``getattr(pipeline, cfg.dit_component)`` is exactly the module
    ``_transformer_for_task(task)`` hands to the denoise loop. Before the fix
    these diverged for ref2va on a combined server -- head bank on
    `transformers_ref`, trunk delta on `transformer`."""
    from vllm_omni.diffusion.models.minimax_h3.pdd import _PDD_VARIANTS_BY_NAME
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    variant = _PDD_VARIANTS_BY_NAME[variant_name]
    assert task in variant.tasks
    assert partition in variant.partitions
    fake = _fake_pipeline(partition)
    serving = MiniMaxH3Pipeline._transformer_for_task(fake, task)
    assert serving is getattr(fake, variant.dit_component(partition))


def test_validate_pdd_sampling_rejects_a_task_the_artifact_was_not_distilled_for():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pdd import _PDD_VARIANTS_BY_NAME
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    ref2va = _PDD_VARIANTS_BY_NAME["ref2va"]
    cfg = PDDConfig(
        variant=ref2va.name,
        tasks=ref2va.tasks,
        dit_component="transformers_ref",
        filename=ref2va.filename,
    )
    fake = SimpleNamespace(
        _pdd_adapters={7: {"cfg": cfg}},
        default_video_shift=12.0,
        default_audio_shift=3.0,
    )
    sampling = SimpleNamespace(
        lora_request=LoRARequest(lora_int_id=7, lora_name="pdd", lora_path=str(PDD_CKPT)),
        extra_args={},
        num_inference_steps=9,
        lora_scale=1.0,
    )
    for bad_task in ("fl2va", "t2va"):
        with pytest.raises(OmniClientError, match="serves \\['ref2va'\\]"):
            MiniMaxH3Pipeline._validate_pdd_sampling(fake, sampling, bad_task)
    # The task it *was* distilled for passes the whole schedule check.
    assert MiniMaxH3Pipeline._validate_pdd_sampling(fake, sampling, "ref2va") is cfg


def test_validate_pdd_sampling_rejects_a_non_unit_lora_scale():
    """Regression: load_head_bank installs the distilled heads at full
    strength regardless of the requested lora_scale, so a fractional scale
    (which does apply to the trunk LoRA delta) silently blended a scaled
    trunk with an unscaled head bank -- neither the trained PDD model nor
    the requested scale."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pdd import _PDD_VARIANTS_BY_NAME
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    ref2va = _PDD_VARIANTS_BY_NAME["ref2va"]
    cfg = PDDConfig(
        variant=ref2va.name,
        tasks=ref2va.tasks,
        dit_component="transformers_ref",
        filename=ref2va.filename,
    )
    fake = SimpleNamespace(
        _pdd_adapters={7: {"cfg": cfg}},
        default_video_shift=12.0,
        default_audio_shift=3.0,
    )
    sampling = SimpleNamespace(
        lora_request=LoRARequest(lora_int_id=7, lora_name="pdd", lora_path=str(PDD_CKPT)),
        extra_args={},
        num_inference_steps=9,
        lora_scale=0.5,
    )
    with pytest.raises(OmniClientError, match="lora_scale=1.0"):
        MiniMaxH3Pipeline._validate_pdd_sampling(fake, sampling, "ref2va")
