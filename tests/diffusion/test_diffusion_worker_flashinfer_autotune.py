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
        self.reject_paths: set[str] = set()  # files FlashInfer refuses (tuned in another environment)
        self.loaded: list[str] = []
        self.saved: list[str] = []

    def load_configs(self, path):
        self.loaded.append(path)
        return path not in self.reject_paths

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

    def step(new=(), ids=(), **shape):
        new_reqs = [SimpleNamespace(req=SimpleNamespace(request_id=i, sampling_params=_params(**shape))) for i in new]
        with worker._flashinfer_autotune_context(
            [r.req for r in new_reqs],
            request_ids=list(ids),
            new_request_ids=list(new),
        ):
            return fake_flashinfer.is_tuning_mode

    assert step(new=["a"], ids=["a"]) is True  # prefill of a new shape
    assert step(ids=["a"]) is True  # decode steps of the same request have other row counts
    assert step(new=["b"], ids=["a", "b"]) is True  # [S, S] is a new batch: twice the GEMM rows
    worker._finish_flashinfer_tuning(["a"], completed=True)
    assert step(ids=["b"]) is True  # "b" keeps tuning its decode steps until it finishes
    worker._finish_flashinfer_tuning(["b"], completed=True)
    assert step(new=["c"], ids=["c"]) is False  # repeat of the [S] batch
    assert step(ids=["c"]) is False
    assert step(new=["d"], ids=["c", "d"]) is False  # repeat of the [S, S] batch
    worker._finish_flashinfer_tuning(["c", "d"], completed=True)
    assert worker._flashinfer_step_signatures == {}  # finished requests are forgotten


def test_cancelled_request_leaves_its_shape_untuned(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    worker = _make_worker(_od_config(True), lambda req, **kw: "out")

    def step(new=(), ids=()):
        new_reqs = [SimpleNamespace(req=SimpleNamespace(request_id=i, sampling_params=_params())) for i in new]
        with worker._flashinfer_autotune_context(
            [r.req for r in new_reqs], request_ids=list(ids), new_request_ids=list(new)
        ):
            return fake_flashinfer.is_tuning_mode

    assert step(new=["a"], ids=["a"]) is True  # prefill of a new shape
    worker._finish_flashinfer_tuning(["a"], completed=False)  # cancelled before its decode steps
    # The shape never finished a forward, so the retry must still tune the steps "a" never reached.
    assert step(new=["b"], ids=["b"]) is True
    worker._finish_flashinfer_tuning(["b"], completed=True)
    assert step(new=["c"], ids=["c"]) is False  # now covered

    # Same for a raising forward: release_step_requests is the abort/error path.
    worker.model_runner = SimpleNamespace(release_step_requests=lambda ids: None)
    worker._step_lora_state = {}
    assert step(new=["d"], ids=["d"]) is False  # composition already tuned by "b"
    assert step(new=["e"], ids=["d", "e"]) is True  # new composition [S, S]
    worker.release_step_requests(["d", "e"])
    assert step(new=["f", "g"], ids=["f", "g"]) is True  # [S, S] is untuned again


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


def _real_request(request_id="r0", *, condition=(), height=1328, width=1328):
    """A real request so the signature goes through the scheduler's batch-compatibility key."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    request = OmniDiffusionRequest(
        prompt={"prompt": "a cat"},
        sampling_params=OmniDiffusionSamplingParams(height=height, width=width, true_cfg_scale=4.0),
        request_id=request_id,
    )
    # What Qwen-Image-2.1's preprocessor records: the reference-image sizes that set the prefix rows.
    request.batch_compatibility_key = ("qwen_image_21", tuple(condition))
    return request


def test_condition_geometry_is_part_of_the_tuned_shape(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    seen = []
    worker = _make_worker(_od_config(True), lambda req, **kw: seen.append(fake_flashinfer.is_tuning_mode) or "out")

    # Same output geometry throughout; only the reference-image prefix changes.
    for condition in ((), (), ((512, 512),), ((512, 512),), ((1024, 1024), (512, 512)), ((512, 512),)):
        assert worker.execute_model(_real_request(condition=condition), worker.od_config) == "out"

    assert seen == [True, False, True, False, True, False]


def test_prompt_only_differences_share_a_signature(mocker, fake_flashinfer, tmp_path, monkeypatch):
    """Text rows are padded to the longest prompt of the batch, so they are not in the signature."""
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    first = _real_request()
    second = _real_request("r1")
    second.prompt = {"prompt": "a very long prompt " * 20}

    assert dw._flashinfer_tune_signature(first) == dw._flashinfer_tune_signature(second)


def test_rejected_cache_file_moves_this_environment_to_its_own_path(mocker, fake_flashinfer, tmp_path, monkeypatch):
    _cuda(mocker)
    monkeypatch.setattr("vllm.envs.VLLM_CACHE_ROOT", str(tmp_path))
    mocker.patch.object(dw, "_flashinfer_autotune_env_tag", return_value="deadbeef")

    def runner(req, **kwargs):
        fake_flashinfer._dirty = fake_flashinfer.is_tuning_mode
        return "out"

    worker = _make_worker(_od_config(True), runner)
    path = dw._flashinfer_autotune_cache_path(worker.od_config, 0)
    os.makedirs(os.path.dirname(path))
    open(path, "w").close()
    # FlashInfer refuses a file tuned under other GPU / library versions, and refuses to overwrite it.
    fake_flashinfer.reject_paths.add(path)
    env_path = dw._flashinfer_autotune_env_cache_path(path)

    worker.execute_model(_request(), worker.od_config)

    # Saving to the rejected file would be skipped by FlashInfer, so the fresh tactics go next to it.
    assert fake_flashinfer.loaded == [path] and fake_flashinfer.saved == [env_path]
    assert env_path == path.replace(".json", ".env-deadbeef.json")

    # The next worker loads that file instead of tuning again.
    open(env_path, "w").close()
    fake_flashinfer.loaded.clear()
    other = _make_worker(_od_config(True), runner, rank=0)
    other.execute_model(_request(), other.od_config)

    assert fake_flashinfer.loaded == [path, env_path]


def test_signature_uses_the_scheduler_batch_key():
    """Fields that only the scheduler key knows about must reach the signature."""
    base = _real_request()
    longer_text = _real_request()
    longer_text.sampling_params.max_sequence_length = 512
    cfg_off = _real_request()
    cfg_off.sampling_params.true_cfg_scale = 1.0

    assert dw._flashinfer_tune_signature(base) != dw._flashinfer_tune_signature(longer_text)
    assert dw._flashinfer_tune_signature(base) != dw._flashinfer_tune_signature(cfg_off)
