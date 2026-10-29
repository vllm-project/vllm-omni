# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CAE-FastWAM real-checkpoint parity on one GPU (no server).

Loads ``amd/CAE-FastWAM`` (or ``FASTWAM_MODEL``) with the Wan2.2-TI2V-5B-Diffusers
base (or ``FASTWAM_BASE_MODEL``) and checks the action chunk for a fixed
observation against the original implementation's chunk (see
``reference_observation.py``), plus seed determinism and suite conditioning.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from tests.diffusion.models.fastwam import reference_observation as ref
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.fastwam.config import resolve_config
from vllm_omni.diffusion.models.fastwam.policy import (
    _BASE_MODEL_PATTERNS,
    DEFAULT_BASE_MODEL,
    FastWAMPolicy,
    checkpoint_patterns,
    resolve_local_dir,
)
from vllm_omni.diffusion.models.fastwam.processing_fastwam import parse_observation
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]
# Hardware: MI325 is the closest ROCm selection label (no MI355X mark exists);
# the qualification target is MI355X. No CUDA SKU until someone runs it there.

MODEL = os.environ.get("FASTWAM_MODEL", "amd/CAE-FastWAM")
BASE_MODEL = os.environ.get("FASTWAM_BASE_MODEL", DEFAULT_BASE_MODEL)


@pytest.fixture(scope="module")
def policy():
    cfg = resolve_config({})
    return FastWAMPolicy(
        cfg=cfg,
        model_dir=resolve_local_dir(MODEL, checkpoint_patterns(cfg)),
        base_model_dir=resolve_local_dir(BASE_MODEL, _BASE_MODEL_PATTERNS),
        device=current_omni_platform.get_torch_device(0),
        dtype=torch.bfloat16,
    )


def _infer(policy, seed: int, **overrides) -> np.ndarray:
    robot_obs = {**ref.build_libero_reference_observation(seed=None), **overrides}
    obs = parse_observation(robot_obs, proprio_dim=8, num_cameras=2, default_suite=ref.SUITE)
    return policy.infer(obs, num_steps=10, generator=torch.Generator(device="cpu").manual_seed(seed))


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_chunk_matches_reference(policy):
    actions = _infer(policy, ref.SEED)
    assert actions.shape == (32, 7) and actions.dtype == np.float32
    rows = actions[list(ref.REFERENCE_ROWS)]
    np.testing.assert_allclose(rows[:, :6], ref.REFERENCE_CHUNK_ROWS[:, :6], atol=ref.GPU_ATOL, rtol=0.0)
    np.testing.assert_array_equal(rows[:, 6], ref.REFERENCE_CHUNK_ROWS[:, 6])


@contextlib.contextmanager
def _deterministic_kernels():
    """Bitwise-repeatable kernels for the duration of the block. By default one
    convolution of the Wan VAE encoder is not run-to-run deterministic on ROCm
    (MIOpen), so two calls with the same seed can differ by a few 1e-2."""
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[2], previous[3]


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_seed_determinism_and_suite_conditioning(policy):
    with _deterministic_kernels():
        first = _infer(policy, 3)
        np.testing.assert_array_equal(first, _infer(policy, 3))
    assert not np.allclose(first, _infer(policy, 4))
    assert not np.allclose(first, _infer(policy, 3, task_suite="libero_goal"))
