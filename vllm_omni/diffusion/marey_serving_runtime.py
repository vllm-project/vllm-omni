# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class MareyVAEInitializationError(RuntimeError):
    """Raised when Marey cannot serve decoded video because the VAE is unavailable."""


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_VALUES


def allow_raw_latent_output(
    *,
    default_output_type: str | None,
    model_config: Mapping[str, object] | None,
) -> bool:
    if model_config is not None and "allow_raw_latent_output" in model_config:
        return _to_bool(model_config["allow_raw_latent_output"])
    if (default_output_type or "").lower() == "latent":
        return True
    return _to_bool(os.environ.get("VLLM_OMNI_ALLOW_MAREY_RAW_LATENTS"))


def build_startup_missing_vae_error(reason: str | None) -> str:
    detail = reason or "VAE initialization did not return an instance."
    return (
        "Marey VAE initialization failed. "
        f"{detail} "
        "Decoded video output requires the OpenSora VAE runtime dependencies plus "
        "a valid moonvalley_ai checkout exposed through MOONVALLEY_AI_PATH and PYTHONPATH. "
        "If latent-only output is intentional, set "
        "model_config.allow_raw_latent_output=true, "
        "VLLM_OMNI_ALLOW_MAREY_RAW_LATENTS=1, or configure "
        "od_config.output_type='latent'."
    )


def build_request_missing_vae_error(reason: str | None, output_type: str) -> str:
    detail = reason or "VAE initialization did not return an instance."
    return (
        f"Marey cannot serve output_type={output_type!r} because the OpenSora VAE is unavailable. "
        f"{detail} "
        "Use output_type='latent' only if latent-only serving was enabled intentionally."
    )
