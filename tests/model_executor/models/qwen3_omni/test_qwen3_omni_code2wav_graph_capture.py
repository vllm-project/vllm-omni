# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""When the Qwen3-Omni Code2Wav stage captures its CUDA graphs: from the
worker's ``capture_auxiliary_graphs`` hook after the first load, at once on a
weight reload, and while loading on NPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni as qwen3_omni
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(*, code2wav: bool = True) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.thinker = None
    model.talker = None
    model.code2wav = MagicMock(load_weights=MagicMock(return_value=set())) if code2wav else None
    model.model_stage = "code2wav" if code2wav else "thinker"
    return model


@pytest.fixture
def platform(monkeypatch):
    npu = MagicMock(return_value=False)
    monkeypatch.setattr(qwen3_omni.current_omni_platform, "is_npu", npu)
    return npu


@pytest.fixture
def enable(monkeypatch, platform):
    calls = MagicMock()
    monkeypatch.setattr(Qwen3OmniMoeForConditionalGeneration, "_maybe_enable_code2wav_cudagraph", calls)
    return calls


def test_load_defers_capture_to_the_warmup_hook_which_captures_once(enable):
    model = _model()

    model.load_weights(iter(()))
    model.code2wav.precompute_snake_caches.assert_called_once()
    enable.assert_not_called()

    model.capture_auxiliary_graphs()
    model.capture_auxiliary_graphs()
    enable.assert_called_once()


@pytest.mark.parametrize("code2wav", [True, False])
def test_warmup_hook_captures_nothing_without_loaded_code2wav_weights(enable, code2wav):
    model = _model(code2wav=code2wav)
    assert model.release_auxiliary_graphs is None
    model.capture_auxiliary_graphs()

    enable.assert_not_called()


@pytest.mark.parametrize("eager", [False, True])
def test_talker_first_frame_graphs_are_budgeted_released_and_recaptured(monkeypatch, eager):
    import vllm.v1.worker.gpu_worker as gpu_worker

    from vllm_omni.model_executor.models.qwen3_omni.first_frame_decoder import Qwen3OmniFirstFrameDecoder
    from vllm_omni.worker.base import OmniGPUWorkerBase

    model = _model(code2wav=False)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=eager))
    decoder = Qwen3OmniFirstFrameDecoder(MagicMock(), sample_rate=24000)
    model.first_frame_decoder = decoder
    capture = MagicMock(side_effect=lambda: decoder._graphs.update(dict.fromkeys((1, 2, 4, 8))))
    monkeypatch.setattr(decoder, "capture", capture)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1000 - 100 * len(decoder._graphs), 1000))
    worker = object.__new__(OmniGPUWorkerBase)
    worker.model_runner = MagicMock(model=model, adaptive_verification=None)

    assert worker._profile_auxiliary_graph_memory() == (0 if eager else 400)
    assert not decoder._graphs

    # Keep vLLM's real warmup, sizing and extension call order. Replace device
    # operations and distributed collectives at their boundaries.
    worker.device = torch.device("cpu")
    worker.compilation_config = SimpleNamespace(
        mode=gpu_worker.CompilationMode.NONE, compilation_time=0.0, encoder_compilation_time=0.0
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=worker.compilation_config, kernel_config=SimpleNamespace(enable_jit_warmup=False)
    )
    worker.model_config = SimpleNamespace(enforce_eager=eager, seed=0)
    worker.parallel_config = SimpleNamespace(data_parallel_index=0)
    worker.cache_config = SimpleNamespace(enable_extensible_kv_cache=True, kv_cache_memory_bytes=None)
    worker.init_snapshot = SimpleNamespace(free_memory=1000)
    worker.requested_memory = 1000
    worker.profiler = MagicMock()
    worker._kv_cache_config = object()
    for name in (
        "kernel_warmup",
        "warmup_kernels",
        "run_kv_cache_sizing_warmup",
        "set_random_seed",
        "freeze_gc_heap",
        "maybe_attach_gc_debug_callback",
        "enable_gpu_sync_check",
    ):
        monkeypatch.setattr(gpu_worker, name, MagicMock())
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", MagicMock())
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    monkeypatch.setattr(gpu_worker, "_min_across_ranks", lambda blocks: blocks)
    monkeypatch.setattr(
        gpu_worker, "measure_kv_cache_blocks", lambda *a, **kw: torch.accelerator.get_memory_info()[0] // 100
    )
    extend = MagicMock()
    monkeypatch.setattr(gpu_worker, "extend_kv_cache", extend)
    times = worker.compile_or_warm_up_model()
    worker.extend_kv_cache(times.num_kv_blocks)
    extend.assert_called_once_with(worker.model_runner, worker._kv_cache_config, 10 if eager else 6)
    assert len(decoder._graphs) == (0 if eager else 4)
    assert capture.call_count == (0 if eager else 2)


def test_weight_reload_after_warmup_recaptures_at_once(enable):
    model = _model()
    model.load_weights(iter(()))
    model.capture_auxiliary_graphs()

    model.load_weights(iter(()))

    assert enable.call_count == 2
    assert model.code2wav.precompute_snake_caches.call_count == 2


def test_npu_captures_while_loading(enable, platform):
    platform.return_value = True
    model = _model()

    model.load_weights(iter(()))
    enable.assert_called_once()

    model.capture_auxiliary_graphs()
    enable.assert_called_once()


def test_npu_weight_reload_recaptures_while_loading(enable, platform):
    platform.return_value = True
    model = _model()
    model.load_weights(iter(()))
    model.capture_auxiliary_graphs()

    model.load_weights(iter(()))

    assert enable.call_count == 2


def test_failed_capture_falls_back_to_eager_with_a_warning(enable, monkeypatch):
    enable.side_effect = RuntimeError("capture failed")
    warning = MagicMock()
    monkeypatch.setattr(qwen3_omni.logger, "warning", warning)
    model = _model()
    model.load_weights(iter(()))

    model.capture_auxiliary_graphs()

    assert "falling back to eager" in warning.call_args.args[0]


def test_failed_capture_raises_on_npu(enable, platform):
    enable.side_effect = RuntimeError("capture failed")
    platform.return_value = True

    with pytest.raises(RuntimeError, match="capture failed"):
        _model().load_weights(iter(()))


def test_failed_snake_precompute_skips_graph_capture(enable, monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr(qwen3_omni.logger, "warning", warning)
    model = _model()
    model.code2wav.precompute_snake_caches.side_effect = RuntimeError("precompute failed")

    model.load_weights(iter(()))
    model.capture_auxiliary_graphs()

    enable.assert_not_called()
    assert "not capturing" in warning.call_args.args[0]
