# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate the streamed RainFusion execution guard used by NPU nightly."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TRACE_SCRIPT = Path("tests/e2e/online_serving/minimax_h3/rainfusion_trace.py")
MULTI_VIDEO = "RAINFUSION_ATTN multi-video active: sparsity=0.80"
SPARSE = "RainFusion uses MindIE-SD sparse attention, precision=bf16."
HYBRID = "RAINFUSION_ATTN hybrid active: dense_q_rows=128, sparse_q_rows=4096, kv_rows=12000"
REQUEST_BEGIN = "MINIMAX_H3_REF2VA_REQUEST_BEGIN"


@pytest.mark.parametrize(
    ("lines", "success"),
    [
        ([REQUEST_BEGIN, MULTI_VIDEO, SPARSE], True),
        ([REQUEST_BEGIN, MULTI_VIDEO, HYBRID], True),
        ([REQUEST_BEGIN, "RAINFUSION_ATTN staying dense: missing video layout"], False),
        ([REQUEST_BEGIN, MULTI_VIDEO], False),
        ([REQUEST_BEGIN, SPARSE], False),
        ([REQUEST_BEGIN, MULTI_VIDEO, HYBRID.replace("sparse_q_rows=4096", "sparse_q_rows=0")], False),
        ([MULTI_VIDEO, SPARSE, REQUEST_BEGIN], False),
    ],
)
def test_streamed_ref2va_sparse_trace(lines: list[str], success: bool) -> None:
    output = "\n".join(lines) + "\n"
    result = subprocess.run([sys.executable, str(TRACE_SCRIPT)], input=output, text=True, capture_output=True)
    assert result.stdout == output
    assert (result.returncode == 0) is success
