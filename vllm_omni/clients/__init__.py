# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side libraries for vLLM-Omni serving APIs.

``vllm_omni.clients.duplex`` speaks to a running vLLM-Omni server over the
network; ``vllm_omni.clients.inline_duplex`` drives an in-process
``DuplexOmni`` handed in by the caller with the same client API. Modules here
import no server runtime code at module load, and server runtime code
(``vllm_omni.engine``, ``vllm_omni.entrypoints``, ``vllm_omni.model_executor``)
must never import from here.
"""
