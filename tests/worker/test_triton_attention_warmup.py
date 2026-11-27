# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup warmup for the Triton unified-attention prefill kernel."""

from types import SimpleNamespace

import pytest

from vllm_omni.worker import triton_attention_warmup
from vllm_omni.worker.triton_attention_warmup import warm_up_triton_attention_prefill
from vllm_omni.worker_v2.warmup import is_warmup_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Backend:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        if self._name is None:
            raise NotImplementedError
        return self._name


def _worker(*backend_names, pooling=False):
    groups = [[SimpleNamespace(backend=_Backend(name)) for name in backend_names]]
    return SimpleNamespace(model_runner=SimpleNamespace(is_pooling_model=pooling, attn_groups=groups))


@pytest.fixture
def steps(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group", lambda: SimpleNamespace(world_size=1), raising=False
    )
    monkeypatch.setattr(triton_attention_warmup, "_run_lone_prefill", lambda worker, n: calls.append(("lone", n)))
    monkeypatch.setattr(
        "vllm.model_executor.warmup.attention_warmup.mixed_batch_attention_warmup",
        lambda worker, num_tokens: calls.append(("mixed", num_tokens)),
    )
    return calls


@pytest.mark.parametrize(
    ("backends", "pooling", "expected"),
    [
        (("TRITON_ATTN",), False, True),
        (("FLASH_ATTN", "TRITON_ATTN"), False, True),
        (("FLASH_ATTN",), False, False),
        ((None,), False, False),
        (("TRITON_ATTN",), True, False),
    ],
    ids=["triton", "mixed-groups", "flash-attn", "unnamed-backend", "pooling"],
)
def test_runs_lone_then_mixed_prefill_only_for_triton_attention(steps, backends, pooling, expected):
    assert warm_up_triton_attention_prefill(_worker(*backends, pooling=pooling)) is expected
    # The lone prefill (num_seqs == 1) and the mixed step (num_seqs > 1) both
    # exceed the 8-token query limit of the 3D kernel path.
    assert steps == ([("lone", 16), ("mixed", 16)] if expected else [])


def test_lone_prefill_request_is_an_omni_warmup_request():
    # Omni runners skip stage-input preprocessing only for warmup request ids.
    assert is_warmup_request(triton_attention_warmup.LONE_PREFILL_REQ_ID)
