# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for real multi-request batching in the Irodori packed step.

``tests/diffusion/models/test_irodori_tts.py`` already checks that a packed
batch issues a single fused forward. These tests check the stronger property
the batching path actually depends on: request ``i``'s result inside an
N-request batch equals the result it gets on its own, with per-request latents,
valid lengths, timesteps, step sizes, and CFG scales all differing. That is
what catches a packing bug that crosses request rows, wrongly broadcasts a
per-request scale, or writes the wrong slice of the carried CFG correction.

The stub DiT is elementwise, so this covers the packing and Euler-update
plumbing, not attention leakage across requests. ``test_irodori_batching_e2e``
in this directory covers the real model end to end.
"""

import pytest
import torch

from vllm_omni.diffusion.models.irodori_tts.batching import IrodoriDenoiseBatch
from vllm_omni.diffusion.models.irodori_tts.sampler import run_packed_euler_rf_cfg_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_CFG_LAYOUT = ("cond", "text", "speaker")
_POOL_REQUESTS = 4
_POOL_LATENT_LEN = 4
_POOL_LATENT_DIM = 2
# Deliberately ragged: requests 1 and 2 are shorter than the padded window.
_POOL_VALID_LENGTHS = (4, 3, 2, 4)


class _PaddedDiT:
    """Elementwise stub that records how many rows each forward received."""

    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.row_counts = []

    def forward_with_encoded_conditions(self, *, x_t, t, text_state, **_):
        self.row_counts.append(x_t.shape[0])
        marker = text_state[:, :1, :1]
        return x_t * 2 + marker + t.reshape(-1, 1, 1)


def _request_pool():
    """Fixed per-request inputs, indexed identically by every batch shape."""
    latents = torch.arange(
        _POOL_REQUESTS * _POOL_LATENT_LEN * _POOL_LATENT_DIM,
        dtype=torch.float32,
    ).reshape(_POOL_REQUESTS, _POOL_LATENT_LEN, _POOL_LATENT_DIM)
    latent_mask = torch.zeros((_POOL_REQUESTS, _POOL_LATENT_LEN), dtype=torch.bool)
    for row, length in enumerate(_POOL_VALID_LENGTHS):
        latent_mask[row, :length] = True
    timesteps = torch.tensor([0.9, 0.7, 0.5, 0.3])
    dt = torch.tensor([-0.1, -0.2, -0.05, -0.4])
    cfg_scales = torch.tensor([[2.0, 3.0], [1.5, 0.5], [4.0, 1.0], [0.25, 2.0]])
    text = torch.arange(_POOL_REQUESTS * len(_CFG_LAYOUT), dtype=torch.float32).reshape(-1, 1, 1).expand(-1, 2, 2)
    return latents, latent_mask, timesteps, dt, cfg_scales, text


def _padded_batch_subset(indices, *, refresh: bool, corrections=None):
    """Build a padded batch holding exactly the pool requests in ``indices``."""
    latents, latent_mask, timesteps, dt, cfg_scales, text = _request_pool()
    cfg_rows = len(_CFG_LAYOUT)
    selected = torch.tensor(indices)
    # ``_pack_bundles`` lays condition rows out request-major.
    bundle_rows = torch.cat([torch.arange(i * cfg_rows, (i + 1) * cfg_rows) for i in indices])
    subset_latents = latents[selected].clone()
    if refresh:
        correction = torch.zeros_like(subset_latents)
    else:
        correction = torch.stack([corrections[i] for i in indices])
    return IrodoriDenoiseBatch(
        request_ids=tuple(f"r{i}" for i in indices),
        cfg_active=True,
        cfg_layout=_CFG_LAYOUT,
        latents=subset_latents,
        latent_mask=latent_mask[selected].clone(),
        timesteps=timesteps[selected].clone(),
        dt=dt[selected].clone(),
        cfg_scales=cfg_scales[selected].clone(),
        bundle=(
            text[bundle_rows].clone(),
            torch.ones((len(indices) * cfg_rows, 2), dtype=torch.bool),
            None,
            None,
            None,
            None,
        ),
        context_kv_cache=None,
        context_buckets=(2, 1, 1),
        cfg_refresh=refresh,
        cfg_correction=correction,
    )


def test_padded_refresh_batch_matches_serial_per_request():
    indices = list(range(_POOL_REQUESTS))
    batched = _padded_batch_subset(indices, refresh=True)
    batched_model = _PaddedDiT()
    batched_latents = run_packed_euler_rf_cfg_step(batched_model, batched)

    # One fused forward carrying every request's CFG rows — real batching.
    assert batched_model.row_counts == [_POOL_REQUESTS * len(_CFG_LAYOUT)]
    assert batched_latents.shape[0] == _POOL_REQUESTS

    for position, request in enumerate(indices):
        solo = _padded_batch_subset([request], refresh=True)
        solo_model = _PaddedDiT()
        solo_latents = run_packed_euler_rf_cfg_step(solo_model, solo)
        assert solo_model.row_counts == [len(_CFG_LAYOUT)]
        torch.testing.assert_close(batched_latents[position : position + 1], solo_latents)
        torch.testing.assert_close(
            batched.cfg_correction[position : position + 1],
            solo.cfg_correction,
        )


def test_padded_reuse_batch_matches_serial_per_request():
    corrections = {
        index: torch.full((_POOL_LATENT_LEN, _POOL_LATENT_DIM), float(index) - 1.5) for index in range(_POOL_REQUESTS)
    }
    indices = list(range(_POOL_REQUESTS))
    batched = _padded_batch_subset(indices, refresh=False, corrections=corrections)
    batched_model = _PaddedDiT()
    batched_latents = run_packed_euler_rf_cfg_step(batched_model, batched)

    # Reuse steps run the conditional branch only: one row per request.
    assert batched_model.row_counts == [_POOL_REQUESTS]

    for position, request in enumerate(indices):
        solo = _padded_batch_subset([request], refresh=False, corrections=corrections)
        solo_latents = run_packed_euler_rf_cfg_step(_PaddedDiT(), solo)
        torch.testing.assert_close(batched_latents[position : position + 1], solo_latents)


def test_padded_batch_respects_per_request_valid_lengths():
    """Padding rows must stay zero and must not perturb shorter requests."""
    batched = _padded_batch_subset(list(range(_POOL_REQUESTS)), refresh=True)
    latents = run_packed_euler_rf_cfg_step(_PaddedDiT(), batched)
    for row, length in enumerate(_POOL_VALID_LENGTHS):
        assert torch.count_nonzero(latents[row, length:]) == 0
