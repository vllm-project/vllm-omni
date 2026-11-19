# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned residual sampling policy, without accelerator dependencies."""

from __future__ import annotations

import math
from collections.abc import Mapping

SUBTALKER_SAMPLING_PARAMS_KEY = "subtalker_sampling_params"
_SUPPORTED_PARAMS = frozenset({"do_sample", "temperature", "top_k", "top_p"})


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"subtalker_sampling_params.{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"subtalker_sampling_params.{name} must be a finite number")
    return number


def validate_subtalker_sampling_params(value: object) -> None:
    """Validate explicit residual sampling overrides at request admission.

    An empty object is valid: it selects the deployment's residual defaults
    without inheriting greedy sampling from the main Talker. The residual
    predictor currently supports top-k sampling only, so top_p must be 1.
    """
    if not isinstance(value, Mapping):
        raise ValueError("subtalker_sampling_params must be an object")
    unknown = set(value) - _SUPPORTED_PARAMS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"Unsupported subtalker_sampling_params keys: {names}")

    if "do_sample" in value:
        do_sample = value["do_sample"]
        if not isinstance(do_sample, bool):
            raise ValueError("subtalker_sampling_params.do_sample must be a boolean")
    if "temperature" in value:
        temperature = _finite_float(value["temperature"], "temperature")
        if temperature < 0:
            raise ValueError("subtalker_sampling_params.temperature must be non-negative")
    if "top_k" in value:
        top_k = value["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < -1:
            raise ValueError("subtalker_sampling_params.top_k must be -1 or a non-negative integer")
    if "top_p" in value:
        top_p = _finite_float(value["top_p"], "top_p")
        if top_p != 1.0:
            raise ValueError("subtalker_sampling_params.top_p must be 1.0; residual top-p sampling is unsupported")


def get_mtp_sampling_overrides(sampling_params: object) -> dict[str, bool | float | int] | None:
    """Resolve one request without changing the model's deployment defaults.

    Explicit residual overrides are validated at request admission. They take
    precedence, including an empty object, over the main Talker's temperature.
    A request-level temperature applies to both predictors; a deployment-only
    main temperature of zero also makes the residual predictor greedy. Other
    deployment settings stay independent.
    """
    extra_args = getattr(sampling_params, "extra_args", None)
    if isinstance(extra_args, Mapping) and SUBTALKER_SAMPLING_PARAMS_KEY in extra_args:
        return dict(extra_args[SUBTALKER_SAMPLING_PARAMS_KEY])
    temperature = getattr(sampling_params, "temperature", None)
    if isinstance(extra_args, Mapping) and extra_args.get("temperature") is not None and temperature is not None:
        return {"do_sample": temperature > 0, "temperature": float(temperature)}
    if temperature == 0:
        return {"do_sample": False, "temperature": 0.0}
    return None
