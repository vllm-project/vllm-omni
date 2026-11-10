# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Wire protocols vLLM-Omni speaks, independent of who serves or runs them.

A module under ``vllm_omni.protocol`` describes *what is on the wire*: how a
client payload is parsed and validated, and what shape a server payload has.
It never decides how a request is executed, which model answers it, or where
the session state lives. That keeps one protocol implementation usable by more
than one runtime (see ``vllm_omni.protocol.realtime``).

Dependency rule, asserted by ``tests/protocol/realtime/test_protocol_import_boundary.py``:
nothing here may import ``vllm_omni.engine``, ``vllm_omni.entrypoints``,
``vllm_omni.model_executor``, ``vllm_omni.worker`` or ``vllm_omni.clients``.
The arrows point inwards --- those packages import the protocol, never the
other way round.
"""
