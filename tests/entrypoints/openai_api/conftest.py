# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os

import pytest


@pytest.fixture(scope="module")
def safety_checker_module():
    """Load safety_checker.py directly without importing vllm_omni.__init__."""
    sc_path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "entrypoints",
        "openai",
        "safety_checker.py",
    )
    spec = importlib.util.spec_from_file_location("safety_checker", os.path.abspath(sc_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
