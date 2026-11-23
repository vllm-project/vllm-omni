# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Importing vLLM-Omni must not replace vLLM's XPU memory query.

vllm.platforms.xpu sets ``torch.accelerator.get_memory_info`` to its
``get_mem_info_wrapper`` at import time. vLLM's general-plugin load imports
vllm_omni (and through it the XPU platform module), so an override in
vllm_omni would change memory sizing in every vLLM process. The check runs
in a fresh interpreter so earlier imports or monkeypatches in the test
session cannot hide it.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model]

_PROBE = textwrap.dedent(
    """
    import torch
    import vllm.platforms.xpu as vllm_xpu

    import vllm_omni.platforms.xpu.platform  # noqa: F401
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    fn = torch.accelerator.get_memory_info
    print(f"{fn.__module__}.{fn.__qualname__}")
    assert fn is vllm_xpu.get_mem_info_wrapper, f"{fn.__module__}.{fn.__qualname__}"
    """
)


@hardware_test(res={"xpu": "B60"})
@pytest.mark.skipif(not torch.xpu.is_available(), reason="needs an XPU")
def test_omni_xpu_import_keeps_vllm_get_memory_info():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "vllm.platforms.xpu.get_mem_info_wrapper"
