# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vllm_ascend")

from vllm_omni.platforms.npu.worker import npu_omni_v2_model_runner, prefill_graph
from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner, _minicpmo_mixed_long_prefill
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("lengths", "prefilling", "enabled", "stage", "expected"),
    [
        ([39958], [True], True, "llm", False),
        ([39958, 39925], [True, True], True, "llm", False),
        ([39958, 39925], [True, False], True, "llm", True),
        ([8192, 39925], [True, False], True, "llm", False),
        ([39958, 39925], [True, False], False, "llm", False),
        ([39958, 39925], [True, False], True, "tts", False),
    ],
)
def test_only_active_long_prefill_with_a_decode_peer_uses_eager(lengths, prefilling, enabled, stage, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage, architectures=["MiniCPMO45OmniForConditionalGeneration"]),
        additional_config={"minicpmo_long_prefill_eager": enabled},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    batch = SimpleNamespace(
        has_prefill=any(prefilling),
        is_dummy=False,
        is_prefilling_np=np.array(prefilling),
        prefill_len_np=np.array(lengths),
    )
    assert _minicpmo_mixed_long_prefill(config, batch) is expected


@pytest.mark.parametrize("graph_hit", [False, True])
def test_kstep_talker_prefill_skips_compile_for_graph_reference_and_fallback(monkeypatch, graph_hit):
    runner = object.__new__(NPUOmniARModelRunner)
    runner.vllm_config = SimpleNamespace(additional_config={"minicpmo_talker_prefill_eager": True})
    runner._kstep_ran_last = True
    context = SimpleNamespace(skip_compiled=False)
    monkeypatch.setattr(npu_omni_v2_model_runner, "get_forward_context", lambda: context)

    def replay(*args):
        assert context.skip_compiled
        return "graph" if graph_hit else None

    def eager(*args, **kwargs):
        assert context.skip_compiled
        assert not graph_hit
        return "eager"

    monkeypatch.setattr(prefill_graph, "run_prefill", replay)
    monkeypatch.setattr(OmniGPUModelRunner, "_call_model", eager)
    assert runner._call_talker_prefill("active", {}, SimpleNamespace(is_dummy=False), None) == (
        "graph" if graph_hit else "eager"
    )
    assert not runner._kstep_ran_last
    assert not context.skip_compiled
