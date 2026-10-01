# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise NPU generation state consumption without importing Ascend workers."""

import ast
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_ROOT = Path(__file__).resolve().parents[3]


def _sample_tokens():
    path = _ROOT / "vllm_omni/platforms/npu/worker/npu_generation_model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUGenerationModelRunner")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "torch": torch,
        "Mapping": Mapping,
        "cast": cast,
        "OmniModelRunnerOutput": SimpleNamespace,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["sample_tokens"]


def test_generation_consumes_extended_ar_state_and_clears_it():
    path = _ROOT / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExecuteModelState")
    fields = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    # Field names come from the runner source, so mypy cannot see a literal.
    state_type = namedtuple("ExecuteModelState", fields)  # type: ignore[misc]
    payload = torch.tensor([[1.0, 2.0]])
    stats = object()
    encoder_output = object()
    values = dict.fromkeys(fields)
    values.update(multimodal_outputs=payload, cudagraph_stats=stats, ec_connector_output=encoder_output)
    # The AR-only staging field must not become part of the generation payload.
    values["staged_hidden_states"] = torch.tensor([-100.0])
    runner = SimpleNamespace(
        execute_model_state=state_type(**values),
        kv_connector_output=None,
        ascend_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False))
        ),
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["r"], req_id_to_index={"r": 0}),
        _async_chunk=False,
        _should_accumulate_full_payload_output=lambda: False,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False)),
        supports_mm_inputs=True,
        get_omni_connector_output=lambda: None,
        speculative_config=None,
        dynamic_eplb=False,
        _finalize_dump_data=lambda: None,
        use_async_scheduling=False,
    )
    sample = _sample_tokens()
    for _ in range(2):
        runner.execute_model_state = state_type(**values)
        output = sample(runner, None)
        assert runner.execute_model_state is None
        assert output.cudagraph_stats is stats
        assert output.ec_connector_output is encoder_output
        assert output.req_ids == ["r"]
        torch.testing.assert_close(output.multimodal_outputs[0]["model_outputs"], payload[0])
    assert sample(runner, None) is None
