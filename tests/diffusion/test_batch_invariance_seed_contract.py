# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import vllm.envs as envs

from vllm_omni.diffusion.batch_invariance import (
    DIFFUSION_BATCH_INVARIANT_ENV,
    MAX_TORCH_MANUAL_SEED,
    MIN_TORCH_MANUAL_SEED,
    diffusion_batch_invariant_enabled,
    validate_batch_invariant_diffusion_seed,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _unset_diffusion_switch(monkeypatch):
    """Stop an inherited explicit switch from silently disabling the batch-invariant cases."""
    monkeypatch.delenv(DIFFUSION_BATCH_INVARIANT_ENV, raising=False)


@pytest.mark.parametrize("seed", [MIN_TORCH_MANUAL_SEED, -2, 0, 42, MAX_TORCH_MANUAL_SEED])
def test_batch_invariant_mode_accepts_full_torch_seed_range(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(seed=seed),
        request_id="request-test",
    )


@pytest.mark.parametrize(
    "seed",
    [None, True, False, 1.5, "1", MIN_TORCH_MANUAL_SEED - 1, MAX_TORCH_MANUAL_SEED + 1],
)
def test_batch_invariant_mode_rejects_missing_invalid_or_out_of_range_seed(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(ValueError, match="seed"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(seed=seed),
            request_id="request-test",
        )


def test_batch_invariant_mode_rejects_generator_input(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    generator = torch.Generator(device="cpu").manual_seed(7)

    with pytest.raises(ValueError, match="does not accept generator"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(seed=7, generator=generator),
            request_id="request-test",
        )


def test_feature_off_preserves_generator_and_missing_seed_compatibility(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(generator=torch.Generator(device="cpu")),
        request_id="request-test",
    )
    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(),
        request_id="request-test",
    )


def test_diffusion_switch_unset_follows_global_batch_invariant(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert diffusion_batch_invariant_enabled() is True

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert diffusion_batch_invariant_enabled() is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        (" On ", True),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        (" Off ", False),
    ],
)
def test_diffusion_switch_overrides_global_batch_invariant(monkeypatch, raw, expected):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", not expected)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    assert diffusion_batch_invariant_enabled() is expected


@pytest.mark.parametrize("raw", ["", "maybe", "2", "none"])
def test_diffusion_switch_rejects_unparsable_values(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    with pytest.raises(ValueError, match=DIFFUSION_BATCH_INVARIANT_ENV):
        diffusion_batch_invariant_enabled()


def test_seed_contract_follows_the_diffusion_switch_not_the_global_one(monkeypatch):
    """The switch must gate the seed validator itself, not just the helper."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "1")

    with pytest.raises(ValueError, match="seed"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(),
            request_id="request-test",
        )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "0")
    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(),
        request_id="request-test",
    )
