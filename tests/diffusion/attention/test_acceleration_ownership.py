# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "module",
    [
        "attention/layer.py",
        "attention/parallel/ulysses.py",
        "attention/backends/fastvideo_vsa.py",
        "attention/ops/sage_block_sparse_attention.py",
        "attention/ops/sage_quantization.py",
        "distributed/flashinfer_ulysses.py",
        "layers/fused_qk_norm_rope.py",
        "layers/indexed_modulation.py",
        "layers/mxfp8.py",
    ],
)
def test_shared_acceleration_does_not_depend_on_model_policy(module):
    root = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion"
    source = (root / module).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("vllm_omni.diffusion.models.")
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("vllm_omni.diffusion.models.") for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "VLLM_OMNI_H3_" not in node.value
            assert "VLLM_OMNI_MINIMAX_H3_" not in node.value
