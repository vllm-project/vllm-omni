#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wrapper for TTS accuracy evaluation with convenient defaults."""

import subprocess
import sys

if __name__ == "__main__":
    # Re-run this module's sibling script with all args passed through
    script = __file__.replace("run_", "")
    result = subprocess.run([sys.executable, script] + sys.argv[1:])
    sys.exit(result.returncode)
