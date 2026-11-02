# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup warmup for prefill-sized fused-MoE Triton configs (#9069)."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_omni.worker import moe_prefill_warmup
from vllm_omni.worker.moe_prefill_warmup import (
    MOE_TUNING_BATCH_SIZES,
    model_has_fused_moe,
    moe_prefill_warmup_sizes,
    warm_up_moe_prefill_kernels,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingRunner:
    def __init__(self, model):
        self._model = model
        self.calls: list[tuple[int, dict]] = []

    def get_model(self):
        return self._model

    def _dummy_run(self, num_tokens, **kwargs):
        self.calls.append((num_tokens, kwargs))


def _worker(model, *, capture_sizes=(1, 2, 4, 5, 8), max_num_batched_tokens=32768, use_v2=False):
    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_runner=_RecordingRunner(model),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(capture_sizes)),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
    )


def _moe_model():
    return SimpleNamespace(modules=lambda: [object(), "moe-layer"])


@pytest.fixture
def moe_detection(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MOE_PREFILL_WARMUP", raising=False)
    monkeypatch.setattr(
        moe_prefill_warmup,
        "model_has_fused_moe",
        lambda model: "moe-layer" in list(model.modules()),
    )


def test_sizes_cover_each_tuning_bucket_above_graph_capture_with_both_parities():
    sizes = moe_prefill_warmup_sizes(max_cudagraph_size=8, max_num_tokens=32768)

    # The thinker's CI capture sizes stop at 8, so the first uncovered size is
    # the odd neighbour of bucket 8.
    assert sizes[0] == 9
    for bucket in MOE_TUNING_BATCH_SIZES:
        if bucket > 8:
            assert bucket in sizes
            assert bucket + 1 in sizes
    assert all(size > 8 for size in sizes)
    assert sizes == sorted(set(sizes))


def test_sizes_never_exceed_the_schedulable_batch():
    sizes = moe_prefill_warmup_sizes(max_cudagraph_size=8, max_num_tokens=300)

    assert sizes[-1] == 257
    assert all(size <= 300 for size in sizes)


def test_sizes_empty_when_graph_capture_already_covers_every_bucket():
    assert moe_prefill_warmup_sizes(max_cudagraph_size=8192, max_num_tokens=8192) == []


def test_warmup_runs_one_eager_dummy_batch_per_size(moe_detection):
    worker = _worker(_moe_model())

    sizes = warm_up_moe_prefill_kernels(worker)

    assert sizes == moe_prefill_warmup_sizes(8, 32768)
    assert [num_tokens for num_tokens, _ in worker.model_runner.calls] == sizes
    for _, kwargs in worker.model_runner.calls:
        assert kwargs == {"skip_eplb": True, "remove_lora": False}


def test_warmup_skips_models_without_fused_moe(moe_detection):
    worker = _worker(SimpleNamespace(modules=lambda: [object()]))

    assert warm_up_moe_prefill_kernels(worker) == []
    assert worker.model_runner.calls == []


@pytest.mark.parametrize("value", ["0", "false", "OFF"])
def test_warmup_can_be_disabled(moe_detection, monkeypatch, value):
    monkeypatch.setenv("VLLM_OMNI_MOE_PREFILL_WARMUP", value)
    worker = _worker(_moe_model())

    assert warm_up_moe_prefill_kernels(worker) == []
    assert worker.model_runner.calls == []


def test_warmup_skips_v2_model_runner(moe_detection):
    worker = _worker(_moe_model(), use_v2=True)

    assert warm_up_moe_prefill_kernels(worker) == []
    assert worker.model_runner.calls == []


class _FakeMoERunner:
    pass


@pytest.fixture
def fake_fused_moe_module(monkeypatch):
    module = ModuleType("vllm.model_executor.layers.fused_moe")
    module.MoERunner = _FakeMoERunner
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", module)


def test_model_without_modules_is_not_moe(fake_fused_moe_module):
    assert model_has_fused_moe(object()) is False


def test_model_with_moe_runner_modules_is_moe(fake_fused_moe_module):
    assert model_has_fused_moe(SimpleNamespace(modules=lambda: [object(), _FakeMoERunner()])) is True
    assert model_has_fused_moe(SimpleNamespace(modules=lambda: [object()])) is False


def test_missing_moe_runner_symbol_reports_unknown_with_a_warning(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", None)
    monkeypatch.setattr(moe_prefill_warmup.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    assert model_has_fused_moe(SimpleNamespace(modules=lambda: [])) is None
    assert any("MoERunner" in warning for warning in warnings)


def test_unknown_moe_status_skips_without_claiming_no_moe(monkeypatch):
    infos: list[str] = []
    monkeypatch.delenv("VLLM_OMNI_MOE_PREFILL_WARMUP", raising=False)
    monkeypatch.setattr(moe_prefill_warmup, "model_has_fused_moe", lambda model: None)
    monkeypatch.setattr(moe_prefill_warmup.logger, "info", lambda msg, *args: infos.append(msg % args))
    monkeypatch.setattr(moe_prefill_warmup.logger, "debug", lambda msg, *args: infos.append(msg % args))
    worker = _worker(_moe_model())

    assert warm_up_moe_prefill_kernels(worker) == []
    assert worker.model_runner.calls == []
    assert not any("no fused-MoE layers" in line for line in infos)
