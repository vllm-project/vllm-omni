# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side libraries for vLLM-Omni serving APIs.

Modules under this package speak to a running vLLM-Omni server over the
network. They are for applications and tests; server runtime code
(``vllm_omni.engine``, ``vllm_omni.entrypoints``, ``vllm_omni.model_executor``)
must never import from here.
"""
