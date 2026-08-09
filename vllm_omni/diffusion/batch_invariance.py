# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Diffusion batch-invariance switch and seed contract."""

import os
from typing import TYPE_CHECKING

import vllm.envs as envs

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest

MIN_TORCH_MANUAL_SEED = -(2**63)
MAX_TORCH_MANUAL_SEED = 2**64 - 1

DIFFUSION_BATCH_INVARIANT_ENV = "VLLM_OMNI_DIFFUSION_BATCH_INVARIANT"
_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def diffusion_batch_invariant_enabled() -> bool:
    """Whether diffusion batch invariance is requested.

    Unset (the default) follows vLLM's global ``VLLM_BATCH_INVARIANT`` so mixed
    AR + diffusion pipelines keep a single source of truth. Setting it
    explicitly overrides the global switch in either direction, which lets a
    pipeline enable batch invariance for its LLM stage without subjecting the
    diffusion stage to the seed contract -- and lets non-CUDA platforms opt out
    of the diffusion-side hard requirement.
    """
    raw = os.environ.get(DIFFUSION_BATCH_INVARIANT_ENV)
    if raw is None:
        return bool(envs.VLLM_BATCH_INVARIANT)
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{DIFFUSION_BATCH_INVARIANT_ENV} must be one of {_TRUE_VALUES + _FALSE_VALUES}; got {raw!r}.")


def validate_batch_invariant_diffusion_seed(
    sampling_params: OmniDiffusionSamplingParams,
    *,
    request_id: str,
) -> None:
    """Require a portable diffusion RNG identity in batch-invariant mode."""
    if not diffusion_batch_invariant_enabled():
        return

    if sampling_params.generator is not None:
        raise ValueError(
            "Diffusion batch invariance requires one explicit integer seed and "
            f"does not accept generator input for diffusion request {request_id!r}."
        )

    seed = sampling_params.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "Diffusion batch invariance requires an explicit integer seed for "
            f"diffusion request {request_id!r}; got {seed!r}."
        )
    if not MIN_TORCH_MANUAL_SEED <= seed <= MAX_TORCH_MANUAL_SEED:
        raise ValueError(
            "Diffusion seed must be in the torch.Generator.manual_seed range "
            f"[{MIN_TORCH_MANUAL_SEED}, {MAX_TORCH_MANUAL_SEED}]; got {seed} "
            f"for request {request_id!r}."
        )


def validate_batch_invariant_diffusion_request(request: "OmniDiffusionRequest") -> None:
    """Require that the seed alone determines the RNG identity of a request.

    Only the RNG identity is checked, not the recipe: batch invariance does not
    reject configurations outside the validated tuple, it runs them (determinism
    then holds for the operators vLLM actually replaces). The checks here are the
    premise of "same seed => same output" itself, so they stay: caller-supplied
    ``latents`` mean the seed no longer decides the initial noise, and
    ``sigmas`` rewrite the schedule the seeded trajectory follows.
    """
    if not diffusion_batch_invariant_enabled():
        return

    params = request.sampling_params
    unsupported: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            unsupported.append(message)

    validate_batch_invariant_diffusion_seed(params, request_id=request.request_id)
    # Not redundant with the seed check above: the request may already carry a seed
    # that __post_init__ assigned as a fallback, which params alone cannot tell apart
    # from a caller-supplied one. Vacuous when reached from __post_init__, which runs
    # this before that fallback; load-bearing for callers that validate a request
    # object built earlier, or built while the switch was off.
    require(
        request.seed_was_explicit,
        "seed must be explicit when OmniDiffusionRequest is constructed",
    )
    require(params.generator_device is None, "generator_device must not be supplied")

    for field_name in ("latents", "sigmas"):
        require(getattr(params, field_name, None) is None, f"sampling_params.{field_name} must not be supplied")

    if unsupported:
        raise ValueError(
            "Diffusion batch invariance requires the seed to determine the RNG identity. "
            "Unsupported request: " + "; ".join(unsupported) + "."
        )
