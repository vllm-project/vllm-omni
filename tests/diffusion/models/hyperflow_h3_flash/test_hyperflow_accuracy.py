# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Accuracy / contract checks vs official HyperFlow H3 Flash.

References:
- README t2va prompt: Video-Rebirth/hyperflow-h3-flash
- Schedule grids: hyperflow_h3_flash/schedule.py SIGMA_GRIDS
- Weight layout: INT8 ``qweight ≈ round(W R / wscale)``, low-rank ``W = up @ down``
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.hyperflow_h3_flash.int8_convrot import regular_hadamard
from vllm_omni.diffusion.models.hyperflow_h3_flash.prepare import (
    OFFICIAL_T2VA_PROMPT,
    resolve_hyperflow_sigma_positions,
)
from vllm_omni.diffusion.models.hyperflow_h3_flash.schedule import (
    AUDIO_SHIFT,
    DEFAULT_STEPS,
    SIGMA_GRIDS,
    VIDEO_SHIFT,
    sigma_grid,
)
from vllm_omni.diffusion.models.hyperflow_h3_flash.transformer import densify_int8, densify_lowrank
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Official upstream clone used during RFC work (optional).
_UPSTREAM = Path("/dataset/home/z00897794/hyperflow-h3-flash/src")


def _load_upstream_schedule():
    path = _UPSTREAM / "hyperflow_h3_flash" / "schedule.py"
    if not path.is_file():
        pytest.skip("official hyperflow-h3-flash source not available")
    spec = importlib.util.spec_from_file_location("hyperflow_upstream_schedule", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sigma_grids_match_official_upstream():
    upstream = _load_upstream_schedule()
    assert SIGMA_GRIDS == upstream.SIGMA_GRIDS
    assert DEFAULT_STEPS == upstream.DEFAULT_STEPS
    assert VIDEO_SHIFT == upstream.VIDEO_SHIFT
    assert AUDIO_SHIFT == upstream.AUDIO_SHIFT
    for steps in (4, 6, 8):
        assert sigma_grid(steps) == upstream.sigma_grid(steps)


def test_official_readme_prompt_sampling_contract():
    """Official CLI default: prompt + 6 steps, CFG off."""
    assert "ramen" in OFFICIAL_T2VA_PROMPT.lower()
    sampling = SimpleNamespace(
        num_inference_steps=None,
        guidance_scale=1.0,
        timesteps=None,
        sigmas=None,
        prompt=OFFICIAL_T2VA_PROMPT,
    )
    positions, steps = resolve_hyperflow_sigma_positions(sampling)
    assert steps == 6
    assert positions == SIGMA_GRIDS[6]

    for allowed in (4, 6, 8):
        sampling.num_inference_steps = allowed
        _, got = resolve_hyperflow_sigma_positions(sampling)
        assert got == allowed

    sampling.num_inference_steps = 5
    with pytest.raises(OmniClientError, match="4.*6.*8"):
        resolve_hyperflow_sigma_positions(sampling)

    sampling.num_inference_steps = 6
    sampling.guidance_scale = 1.5
    with pytest.raises(OmniClientError, match="CFG"):
        resolve_hyperflow_sigma_positions(sampling)


def test_lowrank_densify_matches_official_factorization():
    torch.manual_seed(0)
    down = torch.randn(8, 32)
    up = torch.randn(64, 8)
    expected = (up.float() @ down.float()).to(torch.bfloat16)
    actual = densify_lowrank(down, up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_int8_densify_recovers_weight_within_quantization_error():
    """Official store: ``qweight ≈ round(W R / wscale)``; densify inverts that map."""
    torch.manual_seed(0)
    group, out_features, in_features = 16, 32, 64
    rot = regular_hadamard(group)
    w = torch.randn(out_features, in_features, dtype=torch.float32)
    wr = (w.reshape(out_features, in_features // group, group) @ rot).reshape(out_features, in_features)
    wscale = wr.abs().amax(1).clamp_min(1e-12) / 127.0
    qweight = (wr / wscale[:, None]).round().clamp(-127, 127).to(torch.int8)

    recovered = densify_int8(qweight, wscale, group).float()
    # Reconstruction matches the quantized rotated matrix (bf16 store → small rounding).
    wr_hat = (qweight.float() * wscale[:, None]).reshape(out_features, in_features // group, group) @ rot
    wr_hat = wr_hat.reshape(out_features, in_features)
    torch.testing.assert_close(recovered, wr_hat, rtol=1e-2, atol=1e-2)
    # And close to the original weight within INT8 channel quantization.
    rel = (recovered - w).abs().amax() / w.abs().amax().clamp_min(1e-6)
    assert float(rel) < 0.05
