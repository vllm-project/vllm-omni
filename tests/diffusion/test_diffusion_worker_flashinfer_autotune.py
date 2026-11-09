# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the diffusion worker's opt-in FlashInfer autotune."""

import contextlib
import os
import sys
import types
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.worker import diffusion_worker as dw
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTuner:
    def __init__(self):
        self.is_tuning_mode = False
        self._dirty = False
        self.loaded: list[str] = []
        self.saved: list[str] = []

    def load_configs(self, path):
        self.loaded.append(path)
        return True

    def save_configs(self, path):
        self.saved.append(path)
        self._dirty = False


@pytest.fixture
def fake_flashinfer(monkeypatch):
    tuner = _FakeTuner()

    @contextlib.contextmanager
    def autotune(tune_mode=True, **kwargs):
        assert not kwargs, "the worker manages the cache file itself"
        tuner.is_tuning_mode = tune_mode
        try:
            yield
        finally:
            tuner.is_tuning_mode = False

    autotuner = types.ModuleType("flashinfer.autotuner")
    autotuner.AutoTuner = SimpleNamespace(get=lambda: tuner)
    autotuner.autotune = autotune
    package = types.ModuleType("flashinfer")
    package.autotuner = autotuner
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
    monkeypatch.setattr("vllm.utils.flashinfer.has_flashinfer", lambda: True)
    return tuner


def _od_config(enabled: bool, **overrides):
    values = dict(
        diffusion_flashinfer_autotune=enabled,
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        quantization_config="fp8",
        dtype="bfloat16",
        num_gpus=1,
        parallel_config="usp1",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_worker(od_config, runner_fn, rank=0) -> DiffusionWorker:
    worker = object.__new__(DiffusionWorker)
    worker.od_config = od_config
    worker.rank = rank
    worker.lora_manager = None
    worker.model_runner = SimpleNamespace(execute_model=runner_fn)
    return worker


def _cuda(mocker, is_cuda=True):
    platform = mocker.patch.object(dw, "current_omni_platform")
    platform.is_cuda.return_value = is_cuda
    return platform


def _params(height=480, width=832, num_frames=81, guidance_scale=4.0):
    return SimpleNamespace(
        lora_request=None,
        lora_scale=None,
        height=height,
        width=width,
        num_frames=num_frames,
        resolution=640,
        num_outputs_per_prompt=1,
        guidance_scale=guidance_scale,
    )


def _request(**shape):
    return SimpleNamespace(sampling_params=_params(**shape))


def test_disabled_runs_without_tuning(mocker, fake_flashinfer):
    _cuda(mocker)
    seen = []
    worker = _make_worker(_od_config(False), lambda req, **kw: seen.append(fake_flashinfer.is_tuning_mode) or "out")

    assert worker.execute_model(_request(), worker.od_config) == "out"
    assert seen == [False]
    assert fake_flashinfer.loaded == [] and fake_flashinfer.saved == []


def test_non_cuda_platform_is_a_noop(mocker, fake_flashinfer):
    _cuda(mocker, is_cuda=False)
    worker = _make_worker(_od_config(True), lambda req, **kw: fake_flashinfer.is_tuning_mode)

    assert worker.execute_model(_request(), worker.od_config) is False


def test_missing_flashinfer_is_a_noop(mocker, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.utils.flashinfer.has_flashinfer", lambda: False)
    worker = _make_worker(_od_config(True), lambda req, **kw: "out")

    assert isinstance(worker._flashinfer_autotune_context(), contextlib.nullcontext)
    assert worker.execute_model(_request(), worker.od_config) == "out"


def test_only_new_request_shapes_run_in_tune_mode(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    seen = []

    def runner(req, **kwargs):
        seen.append(fake_flashinfer.is_tuning_mode)
        fake_flashinfer._dirty = fake_flashinfer.is_tuning_mode  # tune mode profiles the new shape
        return "out"

    worker = _make_worker(_od_config(True), runner, rank=3)
    path = dw._flashinfer_autotune_cache_path(worker.od_config, 3)
    for shape in ({}, {}, {}, {"height": 720, "width": 1280}, {"guidance_scale": 1.0}, {}):
        assert worker.execute_model(_request(**shape), worker.od_config) == "out"

    # Tune mode costs host time on every FlashInfer call, so repeats of a tuned shape run in normal mode.
    assert seen == [True, False, False, True, True, False]
    assert fake_flashinfer.saved == [path, path, path]
    assert path.endswith("rank3.json") and path.startswith(str(tmp_path))
    assert fake_flashinfer.loaded == []  # no file existed before the first save


def test_step_requests_stay_in_tune_mode_until_finished(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    worker = _make_worker(_od_config(True), lambda req, **kw: "out")

    def step(new=(), ids=()):
        new_reqs = [SimpleNamespace(req=SimpleNamespace(request_id=i, sampling_params=_params())) for i in new]
        with worker._flashinfer_autotune_context(
            [r.req.sampling_params for r in new_reqs],
            request_ids=list(ids),
            new_request_ids=list(new),
        ):
            return fake_flashinfer.is_tuning_mode

    assert step(new=["a"], ids=["a"]) is True  # prefill of a new shape
    assert step(ids=["a"]) is True  # decode steps of the same request have other row counts
    assert step(new=["b"], ids=["a", "b"]) is True  # [S, S] is a new batch: twice the GEMM rows
    worker._finish_flashinfer_tuning(["a"])
    assert step(ids=["b"]) is True  # "b" keeps tuning its decode steps until it finishes
    worker._finish_flashinfer_tuning(["b"])
    assert step(new=["c"], ids=["c"]) is False  # repeat of the [S] batch
    assert step(ids=["c"]) is False
    assert step(new=["d"], ids=["c", "d"]) is False  # repeat of the [S, S] batch
    worker._finish_flashinfer_tuning(["c", "d"])
    assert worker._flashinfer_step_signatures == {}  # finished requests are forgotten


def test_batch_size_is_part_of_the_tuned_shape(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    seen = []
    worker = _make_worker(_od_config(True), None)
    worker.model_runner = SimpleNamespace(
        execute_model_batch=lambda sched, od: seen.append(fake_flashinfer.is_tuning_mode) or "out"
    )

    def batch(n, **shape):
        new_reqs = [SimpleNamespace(req=SimpleNamespace(sampling_params=_params(**shape))) for _ in range(n)]
        return SimpleNamespace(scheduled_new_reqs=new_reqs)

    # Wan's effective batch is requests x outputs per prompt: [S] then [S, S] are different GEMM row counts.
    for sched in (batch(1), batch(2), batch(2), batch(1), batch(2, height=720, width=1280)):
        assert worker.execute_model_batch(sched, worker.od_config) == "out"

    assert seen == [True, True, False, False, True]


def test_existing_cache_is_loaded_once(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    worker = _make_worker(_od_config(True), lambda req, **kw: "out")
    path = dw._flashinfer_autotune_cache_path(worker.od_config, 0)
    os.makedirs(os.path.dirname(path))
    open(path, "w").close()

    for _ in range(2):
        worker.execute_model(_request(), worker.od_config)

    assert fake_flashinfer.loaded == [path]
    assert fake_flashinfer.saved == []


def test_cache_path_is_keyed_on_layout_and_rank(tmp_path, monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    base = _od_config(True)
    path = dw._flashinfer_autotune_cache_path(base, 0)

    assert dw._flashinfer_autotune_cache_path(_od_config(True), 0) == path
    assert dw._flashinfer_autotune_cache_path(base, 1) != path
    assert dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=None), 0) != path
    assert dw._flashinfer_autotune_cache_path(_od_config(True, num_gpus=8, parallel_config="usp8"), 0) != path


class _QuantConfig:  # like DiffusionFp8Config: no __repr__, so str() embeds the object address
    def __init__(self, activation_scheme="dynamic"):
        self.activation_scheme = activation_scheme
        self.ignored_layers = ["img_mlp"]


def test_cache_path_is_stable_across_config_objects(tmp_path, monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    first = dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=_QuantConfig()), 0)
    second = dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=_QuantConfig()), 0)
    other = dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=_QuantConfig("static")), 0)

    assert first == second  # a restart builds a new config object and must find the same file
    assert first != other


class _OnlineQuantConfig:  # stands in for a vLLM QuantizationConfig that model loading mutates
    def __init__(self):
        self.ignored_layers = ["img_mlp"]

    def get_name(self):
        return "fp8"


def test_cache_path_ignores_state_added_by_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    config = _OnlineQuantConfig()
    before = dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=config), 0)
    config.packed_modules_mapping = {"qkv": object()}  # runtime state, differs per process
    config.loaded_layers = {f"blocks.{i}" for i in range(4)}

    assert dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=config), 0) == before
    other = _OnlineQuantConfig()
    other.ignored_layers = []
    assert dw._flashinfer_autotune_cache_path(_od_config(True, quantization_config=other), 0) != before


@pytest.mark.parametrize(
    ("setting", "capability_ok", "expected"),
    [
        (None, True, True),  # auto on SM100 / SM120
        (None, False, False),  # auto on SM90 and older: FlashInfer's heuristic already matches
        (False, True, False),  # explicit off wins on Blackwell
        (True, False, True),  # explicit on wins on older GPUs
    ],
)
def test_auto_default_turns_on_for_sm100_and_newer(mocker, fake_flashinfer, setting, capability_ok, expected):
    _cuda(mocker)
    capability = mocker.patch.object(dw, "_device_capability_at_least", return_value=capability_ok)
    worker = _make_worker(_od_config(setting), lambda req, **kw: fake_flashinfer.is_tuning_mode)

    assert worker.execute_model(_request(), worker.od_config) is expected
    if setting is None:
        capability.assert_called_with(100)
    else:
        capability.assert_not_called()


def test_auto_default_without_flashinfer_is_silent(mocker, monkeypatch):
    _cuda(mocker)
    mocker.patch.object(dw, "_device_capability_at_least", return_value=True)
    monkeypatch.setattr("vllm.utils.flashinfer.has_flashinfer", lambda: False)
    warn = mocker.patch.object(dw.logger, "warning_once")

    assert dw._flashinfer_autotune_wanted(_od_config(None)) is False
    warn.assert_not_called()
    assert dw._flashinfer_autotune_wanted(_od_config(True)) is False
    warn.assert_called_once()


def test_decision_is_made_once_per_worker(mocker, fake_flashinfer):
    _cuda(mocker)
    capability = mocker.patch.object(dw, "_device_capability_at_least", return_value=True)
    worker = _make_worker(_od_config(None), lambda req, **kw: "out")
    for shape in ({}, {"height": 720, "width": 1280}, {}):
        worker.execute_model(_request(**shape), worker.od_config)

    assert capability.call_count == 1
