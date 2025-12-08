# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pytest configuration for multi_stages_h100 tests.

This conftest imports fixtures from the parent multi_stages conftest.
"""

# Import the omni_runner fixture from the parent directory's conftest
# This makes the fixture available to tests in this directory
from ..multi_stages.conftest import omni_runner  # noqa: F401

# Explicitly expose the fixture for pytest discovery
__all__ = ["omni_runner"]
