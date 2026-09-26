# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reserved for model_executor-only hooks.

vLLM custom-op bootstrap runs in the root :file:`tests/conftest.py` (at import
time, before ``pytest_plugins``) so it always precedes any test module imports.
A subdirectory :func:`pytest_configure` hook can be too late when pytest
collects other packages first, which re-triggers
``vllm::flashinfer_rotary_embedding``-style duplicate registrations.
"""
