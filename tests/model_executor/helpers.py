# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared helpers for ``tests/model_executor``.

:func:`bootstrap_vllm_layer_custom_op_modules` pins vLLM layer modules in
:data:`sys.modules` before Omni model shims import them.

In isolated ``pytest`` runs (e.g. only ``tests/model_executor/``) different test
files may pull the same vLLM layer stack in different order.  That can
re-execute ``@CustomOp.register`` / ``direct_register_custom_op`` and raise
``Duplicate op name`` for symbols such as ``fatrelu_and_mul``,
``maybe_calc_kv_scales``, or ``rms_norm``), or
``RuntimeError`` from :func:`torch.library.Library.define` (e.g.
``vllm::sequence_parallel_chunk_impl`` from :mod:`vllm.model_executor.models.utils`).
Eager, idempotent :func:`importlib.import_module` calls for these entrypoints
make subsequent imports a no-op so registration runs once per process.

``vllm::flashinfer_rotary_embedding`` is registered from
:mod:`vllm.model_executor.layers.rotary_embedding` (e.g. ``common.ApplyRotaryEmb``).
In recent vLLM, :mod:`vllm.utils.torch_utils` *also* defines the same op; pre-importing
``torch_utils`` first and then importing omni (which pulls ``rotary_embedding`` via
:file:`vllm_omni/patch.py`) runs :func:`torch.library.Library.define` twice and
raises a duplicate registration error.  Import the rotary package *without* a prior
dedicated ``torch_utils`` pre-import so a single path owns registration.
"""

from __future__ import annotations

import importlib
from typing import Final

# CustomOp- / :func:`direct_register_custom_op` entrypoints.  Do not import
# ``vllm.model_executor.model_loader`` (that path is part of the duplicate
# issue for ``MULTIMODAL_REGISTRY._get_model_cls``).
_VLLM_PREIMPORT_MODULES: Final[tuple[str, ...]] = (
    "vllm.model_executor.custom_op",
    # Single registration site for ``vllm::flashinfer_rotary_embedding`` (do not
    # pre-import ``vllm.utils.torch_utils`` first — it duplicates the op vs. rotary).
    "vllm.model_executor.layers.rotary_embedding",
    "vllm.model_executor.layers.activation",
    "vllm.model_executor.layers.layernorm",  # rms_norm, etc. (Qwen2.5-VL / Omni chain)
    "vllm.model_executor.layers.conv",
    "vllm.model_executor.layers.vocab_parallel_embedding",
    "vllm.model_executor.layers.attention",  # maybe_calc_kv_scales, etc.
    "vllm.model_executor.models.utils",  # sequence_parallel_chunk_impl, etc. (qwen2_vl import chain)
)


def bootstrap_vllm_layer_custom_op_modules() -> None:
    """Pre-import vLLM layer and model-util modules that register ops; safe to call multiple times."""
    for name in _VLLM_PREIMPORT_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            # Allow CPU-only or minimal vLLM dev installs: skip if a module is absent.
            pass
