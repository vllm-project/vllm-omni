# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF AutoConfig registration for Vevo2.

The :class:`Vevo2Config` class itself lives next to the model code (see
``vllm_omni.model_executor.models.vevo2.configuration_vevo2``); this
shim re-exports it and calls :func:`AutoConfig.register` so the HF
loader stops rejecting ``model_type='vevo2'`` with::

    pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
      Value error, The checkpoint you are trying to load has model type
      `vevo2` but Transformers does not recognize this architecture...

vllm-omni's ``vllm_omni.transformers_utils.configs.__init__`` eagerly
imports this module on first access so the registration side-effect runs
before any ``ModelConfig`` is built.
"""

from __future__ import annotations

from transformers import AutoConfig

from vllm_omni.model_executor.models.vevo2.configuration_vevo2 import Vevo2Config

# Idempotent: AutoConfig.register raises if the type is already registered,
# so guard against re-import (e.g. test reloads).
try:
    AutoConfig.register("vevo2", Vevo2Config)
except ValueError:
    pass

__all__ = ["Vevo2Config"]
