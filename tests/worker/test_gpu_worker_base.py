from types import SimpleNamespace

import pytest

from vllm_omni.outputs import StagePostWarmupMemoryStats
from vllm_omni.worker.base import OmniGPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_stage_post_warmup_memory_stats_returns_native_rpc_payload():
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    model = SimpleNamespace(
        get_stage_post_warmup_memory_stats=lambda: StagePostWarmupMemoryStats(
            allocated_bytes=11,
            reserved_bytes=13,
        )
    )
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    assert worker.get_stage_post_warmup_memory_stats() == {
        "allocated_bytes": 11,
        "reserved_bytes": 13,
    }


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(),
        SimpleNamespace(get_stage_post_warmup_memory_stats=lambda: None),
    ],
)
def test_get_stage_post_warmup_memory_stats_omits_unsupported_model(model):
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    assert worker.get_stage_post_warmup_memory_stats() is None


def test_get_stage_post_warmup_memory_stats_omits_nonzero_rank():
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 1
    worker.model_runner = SimpleNamespace(get_model=lambda: pytest.fail("non-reporting rank must not read the model"))

    assert worker.get_stage_post_warmup_memory_stats() is None
