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

from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import (
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
        s33 = minimax_h3_time_shift_sigmas(num_steps=32, shift_scale=shift)
        s9 = minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=shift)
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


def test_parallel_head_base_and_explicit_adapter_selection():
    torch.manual_seed(2)
    source = nn.Linear(16, 4)
    head = PDDParallelHead(source, 32)
    plans, _ = _build_pdd_plans(32, 4, 12.0, 3.0)
    weights = torch.randn(32, 4, 16)
    biases = torch.randn(32, 4)
    head.load_bank(1, weights, biases, plans)
    head.load_bank(2, weights + 1, biases + 2, plans)
    x = torch.randn(3, 16)
    base = source(x)
    first = head(x, 3, 1)[0]
    expected = F.linear(x, torch.einsum("n,noi->oi", plans[3], weights), plans[3] @ biases)
    torch.testing.assert_close(first, expected)
    assert not torch.allclose(first, head(x, 3, 2)[0])
    # A -> B -> A and Base never depend on the last adapter or step.
    torch.testing.assert_close(head(x, 3, 1)[0], first)
    torch.testing.assert_close(head(x)[0], base)
    head.remove_bank(1)
    with pytest.raises(KeyError):
        head(x, 3, 1)
    head.load_bank(1, weights + 3, biases + 4, plans)
    assert not torch.allclose(head(x, 3, 1)[0], first)
    torch.testing.assert_close(head(x)[0], base)


@pytest.mark.parametrize("step", [None, -1, 8, True])
def test_parallel_head_rejects_invalid_explicit_step(step):
    head = PDDParallelHead(nn.Linear(3, 2), 32)
    plans, _ = _build_pdd_plans(32, 4, 12.0, 3.0)
    head.load_bank(1, torch.zeros(32, 2, 3), torch.zeros(32, 2), plans)
    with pytest.raises(ValueError, match="step"):
        head(torch.ones(1, 3), step, 1)


def test_parallel_head_no_bias_and_bank_device():
    source = nn.Linear(3, 2, bias=False)
    head = PDDParallelHead(source, 32)
    plans, _ = _build_pdd_plans(32, 4, 12.0, 3.0)
    head.load_bank(1, source.weight.detach()[None].repeat(32, 1, 1), torch.zeros(32, 2), plans)
    bank = head.banks["1"]
    assert bank.bias is None
    assert bank.plans.device == bank.weight.device == head.base_weight.device
    torch.testing.assert_close(head(torch.ones(1, 3), 0, 1)[0], source(torch.ones(1, 3)))


def test_parallel_head_keeps_existing_local_tp_shard():
    source = nn.Linear(3, 2)
    source.input_size_per_partition = 3
    source.output_size_per_partition = 2
    source.tp_rank, source.tp_size, source.gather_output = 1, 2, False
    head = PDDParallelHead(source, 32)
    torch.testing.assert_close(head.base_weight, source.weight)
    plans, _ = _build_pdd_plans(32, 4, 12.0, 3.0)
    weights, biases = torch.randn(32, 4, 3), torch.randn(32, 4)
    head.load_bank(1, weights, biases, plans)
    torch.testing.assert_close(head.banks["1"].weight, weights[:, 2:])
    torch.testing.assert_close(head.banks["1"].bias, biases[:, 2:])


def test_artifact_heads_cannot_replace_full_checkpoint_heads():
    from types import SimpleNamespace

    cfg = PDDConfig()
    adapter = PDDAdapter(cfg, 1, *cfg.plans())
    model = SimpleNamespace(arch=SimpleNamespace(pdd_steps=32))
    with pytest.raises(ValueError, match="checkpoint heads"):
        adapter.install_heads(model)


def test_full_checkpoint_grid_stays_independent():
    from vllm_omni.diffusion.models.minimax_h3.pdd import pdd_sigmas

    assert pdd_sigmas(12.0)[0] == pytest.approx(0.999)
    assert minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=12.0)[0] == 1.0


@pytest.mark.parametrize("step_execution", [False, True])
def test_full_checkpoint_still_rejects_request_adapters(step_execution):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3OmniCheckpointSpec
    from vllm_omni.errors import OmniClientError

    sampling = SimpleNamespace(
        num_inference_steps=8, guidance_scale=1.0, timesteps=None, sigmas=None, extra_args={}, lora_request=None
    )
    spec = FastH3OmniCheckpointSpec()
    spec.check_request(sampling, step_execution=step_execution)
    sampling.lora_request = object()
    with pytest.raises(OmniClientError, match="per-request LoRA"):
        spec.check_request(sampling, step_execution=step_execution)


@pytest.mark.parametrize("kind", ["base", "artifact", "full_checkpoint"])
def test_final_layer_selects_the_requested_head_family(kind):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3FinalLayer
    from vllm_omni.diffusion.models.minimax_h3.pdd import MiniMaxH3PDDOutput

    class AdaLN(nn.Module):
        def forward(self, value):
            return torch.zeros(1, 3), torch.zeros(1, 3)

    class TupleLinear(nn.Linear):
        def forward(self, value):
            return super().forward(value), None

    final = MiniMaxH3FinalLayer.__new__(MiniMaxH3FinalLayer)
    nn.Module.__init__(final)
    final.norm, final.adaln_proj = nn.Identity(), AdaLN()
    final.pdd_steps = 32 if kind == "full_checkpoint" else None
    for name in ("video_out", "audio_out"):
        if kind == "full_checkpoint":
            head = MiniMaxH3PDDOutput(3, 2, shift=12.0)
            nn.init.normal_(head.weight)
            nn.init.normal_(head.bias)
        else:
            head = TupleLinear(3, 2)
            if kind == "artifact":
                head = PDDParallelHead(head, 32)
                plans, _ = _build_pdd_plans(32, 4, 12.0, 3.0)
                head.load_bank(1, torch.randn(32, 2, 3), torch.randn(32, 2), plans)
        setattr(final, name, head)
    value = torch.randn(2, 3)
    step = 6 if kind != "base" else None
    adapter_id = 1 if kind == "artifact" else None
    expected = []
    for head in (final.video_out, final.audio_out):
        result = (
            head(value, step)
            if kind == "full_checkpoint"
            else (head(value, step, adapter_id)[0] if kind == "artifact" else head(value)[0])
        )
        expected.append(result)
    actual = final(
        value,
        t_emb=torch.empty(0),
        inverse_indices=torch.zeros(2, dtype=torch.long),
        pdd_step=step,
        pdd_adapter_id=adapter_id,
    )
    for found, wanted in zip(actual, expected):
        torch.testing.assert_close(found, wanted)


def test_diffuse_accepts_pdd_adapter_but_build_denoise_inputs_does_not():
    """Regression: ``forward`` passed ``pdd_adapter`` straight through to
    ``_build_denoise_inputs``, which has no such parameter.  ``diffuse`` owns
    the argument (it selects the request-owned adapter ID); the input builder must not
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

    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import (
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
    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import (
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

    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import _trunk_target_pattern

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
    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import _select_pdd_file

    with pytest.raises(ValueError, match="point lora.path at a single file"):
        _select_pdd_file(PDD_CKPT.parent)


def test_pdd_named_file_without_a_head_bank_raises_instead_of_falling_back(tmp_path):
    """A file the caller believes is PDD but which has no per-step bank must be
    a hard error: the caller still pins 8 steps, and 8 undistilled steps is a
    wasted 3-minute request that returns garbage, not an error."""
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import _PDD_VARIANTS_BY_NAME

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
    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import _PDD_VARIANTS_BY_NAME
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    variant = _PDD_VARIANTS_BY_NAME[variant_name]
    assert task in variant.tasks
    assert partition in variant.partitions
    fake = _fake_pipeline(partition)
    serving = MiniMaxH3Pipeline._transformer_for_task(fake, task)
    assert serving is getattr(fake, variant.dit_component(partition))


def test_validate_pdd_sampling_rejects_a_task_the_artifact_was_not_distilled_for():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import _PDD_VARIANTS_BY_NAME
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
        num_inference_steps=8,
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

    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import _PDD_VARIANTS_BY_NAME
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
        num_inference_steps=8,
        lora_scale=0.5,
    )
    with pytest.raises(OmniClientError, match="lora_scale=1.0"):
        MiniMaxH3Pipeline._validate_pdd_sampling(fake, sampling, "ref2va")


@pytest.mark.parametrize("evict", [False, True])
def test_manager_removal_releases_pdd_banks(evict):
    """Exercise explicit removal and the LRU entrypoint, not cleanup directly."""
    import weakref
    from collections import OrderedDict
    from types import SimpleNamespace

    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._ensure_pdd_bookkeeping()
    bank = torch.ones(4)
    bank_ref = weakref.ref(bank)
    pipeline._pdd_adapters[1] = {"head_weights": bank}
    pipeline._pdd_adapter_ids.add(1)
    del bank
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = pipeline
    manager.max_cached_adapters = 1
    manager._registered_adapters = {1: SimpleNamespace(id=1)}
    manager._active_adapter_id = None
    manager._suspended_adapter_id = None
    manager._fixed_adapter_request = None
    manager._merged_adapters = {}
    manager._adapter_scales = {1: 1.0}
    manager._adapter_access_order = OrderedDict([(1, 0)])
    manager._pinned_adapters = set()
    if evict:
        manager._evict_for_new_adapter()
    else:
        assert manager.remove_adapter(1)
    assert bank_ref() is None
    assert not pipeline._pdd_adapters
    assert not pipeline._pdd_adapter_ids
    assert not manager._registered_adapters


def test_manager_close_keeps_fixed_guard_and_releases_model_and_merged_banks():
    from collections import OrderedDict
    from types import SimpleNamespace

    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._ensure_pdd_bookkeeping()
    pipeline._pdd_adapters[1] = {"head_weights": torch.ones(1)}
    pipeline._pdd_adapter_ids.add(1)
    closed = []
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline, manager._closed = pipeline, False
    manager._registered_adapters, manager._adapter_scales = {1: object()}, {1: 1.0}
    manager._fixed_adapter_request = SimpleNamespace(lora_int_id=1)
    manager._active_adapter_id = manager._suspended_adapter_id = None
    manager._merged_adapters = {1: SimpleNamespace(close=lambda: closed.append(1))}
    manager._lora_modules = {}
    manager._adapter_access_order, manager._pinned_adapters = OrderedDict([(1, 0)]), {1}
    assert manager.remove_adapter(1) is False
    assert pipeline._pdd_adapters
    manager.close()
    assert closed == [1]
    assert not manager._registered_adapters
    assert not pipeline._pdd_adapters
    assert not pipeline._pdd_adapter_ids


@pytest.mark.parametrize("num_steps", [7, 9])
def test_validate_pdd_sampling_rejects_wrong_evaluation_count(num_steps):
    """The API counts evaluations; nine is the old sigma-boundary contract."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    cfg = PDDConfig()
    fake = SimpleNamespace(_pdd_adapters={7: {"cfg": cfg}}, default_video_shift=12.0, default_audio_shift=3.0)
    sampling = SimpleNamespace(
        lora_request=LoRARequest(lora_int_id=7, lora_name="pdd", lora_path=str(PDD_CKPT)),
        extra_args={},
        num_inference_steps=num_steps,
        lora_scale=1.0,
    )
    with pytest.raises(OmniClientError, match="num_inference_steps=8"):
        MiniMaxH3Pipeline._validate_pdd_sampling(fake, sampling)


@pytest.mark.parametrize("sampler", [None, "euler", "res_multistep", "unknown"])
def test_pdd_sampling_preserves_the_distilled_euler_contract(sampler):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pdd_artifact_lifecycle import MiniMaxH3PDDLifecycleMixin
    from vllm_omni.errors import OmniClientError

    cfg = PDDConfig()
    fake = SimpleNamespace(_pdd_adapters={7: {"cfg": cfg}}, default_video_shift=12.0, default_audio_shift=3.0)
    sampling = SimpleNamespace(
        lora_request=LoRARequest(lora_int_id=7, lora_name="pdd", lora_path=str(PDD_CKPT)),
        extra_args={"sampler": sampler},
        num_inference_steps=8,
        lora_scale=1.0,
    )
    if sampler in (None, "euler"):
        assert MiniMaxH3PDDLifecycleMixin._validate_pdd_sampling(fake, sampling, "ref2va") is cfg
    else:
        with pytest.raises(OmniClientError, match="sampler"):
            MiniMaxH3PDDLifecycleMixin._validate_pdd_sampling(fake, sampling, "ref2va")
