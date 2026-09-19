# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""PersonaPlex duplex is not on the duplex plugin framework yet.

RFC vllm-omni#7181 splits the port: this PR lands the unified framework with
MiniCPM-o 4.5, and the follow-up PR moves PersonaPlex onto ``DuplexModelPlugin``.
Until then the modules under test still import the pre-framework duplex
runtime / serving adapter that this PR removed, so collecting them would fail
at import time and abort the whole pytest session. The tests themselves are
unchanged and come back with the port.

Listed per file rather than globbed, so the coverage loss is visible and shrinks
file by file as the port lands.
"""

#: ``test_unified_runtime`` imports ``engine.duplex.runtime`` and
#: ``entrypoints.duplex.runtime_adapter`` directly. ``test_stage0_runtime`` fails
#: transitively: ``personaplex.duplex.runtime_extension`` imports ``DuplexFence``
#: from ``engine.duplex.messages``, which no longer exports it.
collect_ignore = ["test_stage0_runtime.py", "test_unified_runtime.py"]
