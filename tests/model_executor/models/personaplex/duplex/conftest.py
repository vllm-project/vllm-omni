# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""PersonaPlex duplex is not on the duplex plugin framework yet.

RFC vllm-omni#7181 splits the port: this PR lands the unified framework with
MiniCPM-o 4.5, and the follow-up PR moves PersonaPlex onto ``DuplexModelPlugin``.
Until then the modules under test still import the pre-framework duplex
runtime / serving adapter that this PR removed, so collecting them would fail
at import time and abort the whole pytest session. The tests themselves are
unchanged and come back with the port.
"""

collect_ignore_glob = ["test_*.py"]
