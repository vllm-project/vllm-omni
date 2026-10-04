# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility alias for :mod:`vllm_omni.diffusion.models.pi.pi05`."""

from __future__ import annotations

import importlib
import sys

_CANONICAL_PACKAGE = "vllm_omni.diffusion.models.pi.pi05"
_legacy_package = __name__
_canonical = importlib.import_module(_CANONICAL_PACKAGE)

for _submodule in ("config", "modeling_pi05", "pipeline_pi05", "processor_pi05"):
    sys.modules[f"{_legacy_package}.{_submodule}"] = importlib.import_module(f"{_CANONICAL_PACKAGE}.{_submodule}")

sys.modules[_legacy_package] = _canonical
