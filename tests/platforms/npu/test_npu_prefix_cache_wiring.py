# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU prefix-cache wiring: save_outputs -> batch id -> merge_batch on CPU.

The NPU runner hands the batch id across execute_model()/sample_tokens() as the
LAST field of its positionally-packed ``ExecuteModelState``. This doesn't run in
CUDA CI, so a change can silently break it (`vllm_ascend` imports keep the
runner modules out of reach of a plain unit test).

AST characterization of the ``ExecuteModelState`` field order in the NPU
runner — the pack/unpack sites pass fields positionally, so a
reorder or drop of ``prefix_cache_batch_id`` corrupts the batch id without any
import-time error.
"""

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NPU_RUNNER = _REPO_ROOT / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"


def _named_tuple_fields(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    raise AssertionError(f"{class_name} not found in {path}")


def test_execute_model_state_keeps_prefix_cache_batch_id_last():
    """The NPU runner packs ExecuteModelState positionally: the batch id must stay
    the LAST field, and the NPU field order is pinned exactly — a reorder
    passes import and every CUDA test, then ships the wrong value as the
    batch id on hardware."""
    npu_fields = _named_tuple_fields(_NPU_RUNNER, "ExecuteModelState")
    assert npu_fields == [
        "scheduler_output",
        "logits",
        "spec_decode_metadata",
        "spec_decode_common_attn_metadata",
        "hidden_states",
        "sample_hidden_states",
        "aux_hidden_states",
        "attn_metadata",
        "positions",
        "ec_connector_output",
        "cudagraph_stats",
        "batch_desc",
        "multimodal_outputs",
        "prefix_cache_batch_id",
    ]
