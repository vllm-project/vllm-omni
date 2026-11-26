# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@hardware_test(res={"rocm": "MI350"}, num_cards=1)
@torch.inference_mode()
def test_wide_fmha_two_segments(tmp_path):
    if not torch.version.hip or not torch.accelerator.is_available():
        pytest.skip("requires a gfx950 GPU")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("requires gfx950")

    from aiter import flash_attn_varlen_func

    from vllm_omni.diffusion.models.minimax_h3 import rocm_wide_fmha as wide

    asm = Path(wide.__file__).parent / "rocm_kernels" / "wide_first_gfx950.s"
    clang = shutil.which("amdclang") or "/opt/rocm/llvm/bin/clang"
    hsaco = tmp_path / "wide.hsaco"
    subprocess.run(
        [
            str(clang),
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "-x",
            "assembler",
            "-nostdlib",
            str(asm),
            "-o",
            str(hsaco),
        ],
        check=True,
    )

    device = torch.device("cuda", 0)
    rng = torch.Generator(device=device).manual_seed(2026)
    wire = torch.randn((37_760, 3, 14, 128), generator=rng, device=device, dtype=torch.bfloat16)
    q, k, v = wire.unbind(1)
    # A distinct suffix value makes accidental attention across the cut observable.
    v[-14:].add_(2)
    contiguous = tuple(t.contiguous() for t in (q, k, v))
    for cut in (37_746, 37_747, 37_748, 37_749, 37_750, 37_751, 37_753, 37_755, 37_757):
        bounds = torch.tensor([0, cut, len(wire)], dtype=torch.int32, device=device)
        reference = flash_attn_varlen_func(*contiguous, bounds, bounds, cut, cut, softmax_scale=128**-0.5, causal=False)
        result = wide.try_wide_fmha(q, k, v, cut, str(hsaco))
        assert result is not None and result.is_contiguous()
        torch.testing.assert_close(result, reference, atol=0.005, rtol=0.02)
        if cut == 37_746:
            # Upstream Ulysses returns contiguous inputs. Any mix of the
            # two supported layouts must feed identical BF16 operands.
            for inputs in (contiguous, (q, contiguous[1], v)):
                staged = wide.try_wide_fmha(*inputs, cut, str(hsaco))
                assert staged is not None and staged.is_contiguous()
                assert torch.equal(staged, result)


@hardware_test(res={"rocm": "MI350"}, num_cards=1)
@torch.inference_mode()
def test_workspace_limit_and_invalid_inputs_decline_before_staging(monkeypatch):
    if not torch.version.hip or not torch.accelerator.is_available():
        pytest.skip("requires a gfx950 GPU")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("requires gfx950")

    from vllm_omni.diffusion.models.minimax_h3 import rocm_wide_fmha as wide

    # Exercise all runtime guards with small real GPU views. The native module
    # and the QKV staging allocation must both remain untouched on fallback.
    dims = (8, 2, 8)
    monkeypatch.setattr(wide, "_SHAPE", dims)
    monkeypatch.setattr(wide, "_QKV_STRIDES", (48, 8, 1))
    monkeypatch.setattr(wide, "_workspaces", {})
    monkeypatch.setattr(wide, "_Workspace", lambda *args: pytest.fail("unexpected workspace allocation"))
    monkeypatch.setattr(torch, "stack", lambda *args, **kwargs: pytest.fail("unexpected staging allocation"))
    q = torch.zeros(dims, device="cuda", dtype=torch.bfloat16)
    misaligned = q.new_empty(q.numel() + 1)[1:].reshape_as(q)
    non_wire = q.new_empty((8, 4, 2, 8))[:, 0]
    for invalid in (misaligned, non_wire, torch._neg_view(q), q.detach().requires_grad_()):
        assert wide.try_wide_fmha(q, invalid, q, 37_746, "/unused") is None
    monkeypatch.setattr(wide, "_MAX_WORKSPACES_PER_DEVICE", 0)
    assert wide.try_wide_fmha(q, q, q, 37_746, "/unused") is None
