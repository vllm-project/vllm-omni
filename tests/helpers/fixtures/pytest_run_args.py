# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pytest CLI options and the fixtures that expose them."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-level",
        action="store",
        default="core_model",
        choices=["core_model", "advanced_model", "full_model"],
        help="Test level to run: L2, L3, L4",
    )
    # Nightly stability always passes --run-slow (vLLM skip-gate). Register it
    # here when missing; ignore if vLLM or another plugin already added it.
    try:
        parser.addoption(
            "--run-slow",
            action="store_true",
            default=False,
            help="Run tests marked slow. Required by some vLLM pytest plugins; ignored otherwise.",
        )
    except ValueError:
        pass


@pytest.fixture(scope="session")
def run_level(request) -> str:
    """Session test level from ``--run-level`` (see CI five-level docs)."""
    return request.config.getoption("--run-level")
