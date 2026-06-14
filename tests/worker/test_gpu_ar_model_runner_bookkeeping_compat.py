# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


def test_gpu_ar_model_runner_calls_bookkeeping_with_current_vllm_signature():
    runner = object.__new__(GPUARModelRunner)
    calls = []

    def bookkeeping_sync(*args):
        calls.append(args)
        return (0, None, [], {}, [], {}, [])

    runner._bookkeeping_sync = bookkeeping_sync
    runner._bookkeeping_accepts_spec_decode_metadata = False
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=3)
    sampler_output = SimpleNamespace()
    logits = torch.empty(1, 1)
    hidden_states = torch.empty(1, 1)

    result = runner._bookkeeping_sync_compat(
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        scheduler_output.total_num_scheduled_tokens,
        spec_decode_metadata=object(),
    )

    assert result == (0, None, [], {}, [], {}, [])
    assert calls == [
        (
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            3,
        )
    ]


if __name__ == "__main__":
    test_gpu_ar_model_runner_calls_bookkeeping_with_current_vllm_signature()


def test_bookkeeping_compat_does_not_reflect_per_call():
    import inspect

    source = inspect.getsource(GPUARModelRunner._bookkeeping_sync_compat)
    assert "inspect.signature" not in source


def test_sample_tokens_passes_spec_decode_metadata_to_bookkeeping_compat():
    import ast
    import inspect

    source = inspect.getsource(GPUARModelRunner.sample_tokens)
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_bookkeeping_sync_compat"
    ]
    assert len(calls) == 1
    assert any(isinstance(arg, ast.Name) and arg.id == "spec_decode_metadata" for arg in calls[0].args)
