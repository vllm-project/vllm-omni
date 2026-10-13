# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight configuration discovery for the published SolarWM bundle."""

from __future__ import annotations


def solarwm_h3_transformer_config_subfolder(*, model: str | None, revision: str | None = None) -> str:
    """Locate H3's base config before the Solar pipeline loads its components."""
    if model is None:
        return "transformer"
    from vllm.transformers_utils.repo_utils import file_or_path_exists

    nested = "SolarWM-h3-33B-base/transformer"
    return nested if file_or_path_exists(model, f"{nested}/config.json", revision=revision) else "transformer"
