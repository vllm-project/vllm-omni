# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion batch-key shape relaxation (CPU).

``enable_mixfusion`` packs requests of different resolutions into one denoise step
batch, so ``height``/``width``/``resolution`` must stop splitting the step-batch
key — but only when the pipeline declares ``supports_mixfusion``. On any other
pipeline the request-level flag degrades to a no-op and resolution keeps grouping
the batch (a server-level flag is rejected at engine startup instead).
Request-mode batches still resolve one height/width for the whole batch from the
first request's sampling params, so they keep resolution in their key.
A prompt dict that carries ``additional_information`` cannot share a key at all,
which both schedulers express by returning ``None``.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

import vllm_omni.diffusion.registry as registry_module
from vllm_omni.diffusion.registry import supports_mixfusion
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler, _apply_mixfusion_shape_relaxation
from vllm_omni.diffusion.sched.interface import RequestBatchSamplingParamsKey, StepBatchSamplingParamsKey
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SHAPE: dict[str, Any] = {"height": 1024, "width": 768, "resolution": 1024}


class _MixfusionCapablePipeline:
    supports_mixfusion = True


class _PlainPipeline:
    pass


class _Scheduler(BaseScheduler):
    """Minimal concrete scheduler; only the step-batch key builder is exercised."""

    def update_from_output(self, sched_output, output) -> set[str]:
        return set()


def _request(prompt: Any, **sampling_overrides: Any) -> SimpleNamespace:
    sampling: dict[str, Any] = {field.name: field.default for field in fields(StepBatchSamplingParamsKey)}
    sampling.update({field.name: field.default for field in fields(RequestBatchSamplingParamsKey)})
    sampling.update(SHAPE, extra_args={}, lora_request=None)
    sampling.update(sampling_overrides)
    return SimpleNamespace(
        sampling_params=SimpleNamespace(**sampling),
        prompt=prompt,
        batch_compatibility_key=None,
        use_step_execution=True,
    )


def _initialized_scheduler(scheduler_cls, monkeypatch: pytest.MonkeyPatch, pipeline_cls: type | None):
    """Build a scheduler initialized against a stub pipeline class from the registry."""
    monkeypatch.setattr(
        registry_module.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: pipeline_cls,
    )
    scheduler = scheduler_cls()
    scheduler.initialize(SimpleNamespace(model_class_name="StubPipeline"))
    return scheduler


@pytest.mark.parametrize("scheduler_cls", [_Scheduler, RequestScheduler])
def test_shape_still_splits_the_key_without_mixfusion(scheduler_cls):
    request = _request("a cat")
    values = dict(SHAPE)

    assert _apply_mixfusion_shape_relaxation(request, values) is values
    key = scheduler_cls()._build_sampling_params_key(request)
    assert (key.height, key.width, key.resolution) == (1024, 768, 1024)


def test_mixfusion_drops_shape_from_the_step_key(monkeypatch: pytest.MonkeyPatch):
    request = _request("a cat", extra_args={"enable_mixfusion": True})
    values = dict(SHAPE)

    assert _apply_mixfusion_shape_relaxation(request, values) is values
    assert values == {"height": None, "width": None, "resolution": None}
    scheduler = _initialized_scheduler(_Scheduler, monkeypatch, _MixfusionCapablePipeline)
    key = scheduler._build_sampling_params_key(request)
    assert (key.height, key.width, key.resolution) == (None, None, None)
    # Other shape fields still participate, so mixfusion only relaxes resolution.
    assert key.num_frames == 1


def test_mixfusion_keeps_shape_in_the_request_mode_key(monkeypatch: pytest.MonkeyPatch):
    # The request-mode batch forward resolves one height/width for the whole
    # batch from the first request's sampling params, so resolution must keep
    # splitting the key until that path supports mixed shapes.
    request = _request("a cat", extra_args={"enable_mixfusion": True})

    scheduler = _initialized_scheduler(RequestScheduler, monkeypatch, _MixfusionCapablePipeline)
    key = scheduler._build_sampling_params_key(request)
    assert (key.height, key.width, key.resolution) == (1024, 768, 1024)


@pytest.mark.parametrize("scheduler_cls", [_Scheduler, RequestScheduler])
def test_mixfusion_leaves_per_prompt_requests_without_a_key(scheduler_cls, monkeypatch: pytest.MonkeyPatch):
    request = _request(
        {"prompt": "a cat", "additional_information": {"image": "x"}},
        extra_args={"enable_mixfusion": True},
    )

    assert _apply_mixfusion_shape_relaxation(request, dict(SHAPE)) is None
    scheduler = _initialized_scheduler(scheduler_cls, monkeypatch, _MixfusionCapablePipeline)
    assert scheduler._build_sampling_params_key(request) is None


def test_requests_without_mixfusion_keep_their_key_despite_additional_information():
    request = _request({"prompt": "a cat", "additional_information": {"image": "x"}})

    assert _Scheduler()._build_sampling_params_key(request) is not None


def test_empty_additional_information_still_batches(monkeypatch: pytest.MonkeyPatch):
    request = _request(
        [{"prompt": "a cat", "additional_information": {}}],
        extra_args={"enable_mixfusion": True},
    )

    scheduler = _initialized_scheduler(_Scheduler, monkeypatch, _MixfusionCapablePipeline)
    key = scheduler._build_sampling_params_key(request)
    assert key is not None
    assert key.height is None


@pytest.mark.parametrize(
    ("pipeline_cls", "expected"),
    [(_MixfusionCapablePipeline, True), (_PlainPipeline, False), (None, False)],
)
def test_supports_mixfusion_resolution(monkeypatch: pytest.MonkeyPatch, pipeline_cls, expected):
    # Supported pipeline opts in; a pipeline without the attribute — or an
    # unresolvable model_class_name — is treated as unsupported.
    monkeypatch.setattr(
        registry_module.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: pipeline_cls,
    )

    assert supports_mixfusion(SimpleNamespace(model_class_name="StubPipeline")) is expected


def test_supports_mixfusion_honors_custom_pipeline_class():
    od_config = SimpleNamespace(
        model_class_name="StubPipeline",
        custom_pipeline_args={"pipeline_class": _MixfusionCapablePipeline},
    )

    assert supports_mixfusion(od_config) is True


@pytest.mark.parametrize("scheduler_cls", [_Scheduler, RequestScheduler])
def test_request_level_mixfusion_degrades_on_unsupported_pipeline(scheduler_cls, monkeypatch: pytest.MonkeyPatch):
    # The pipeline lacks supports_mixfusion, so the request-level flag is a
    # no-op: resolution stays in the key and mixed-resolution requests simply
    # never co-batch, exactly as if MixFusion were off.
    scheduler = _initialized_scheduler(scheduler_cls, monkeypatch, _PlainPipeline)

    request = _request("a cat", extra_args={"enable_mixfusion": True})
    key = scheduler._build_sampling_params_key(request)
    assert (key.height, key.width, key.resolution) == (1024, 768, 1024)

    # Per-prompt requests keep their key too instead of becoming unbatchable.
    per_prompt = _request(
        {"prompt": "a cat", "additional_information": {"image": "x"}},
        extra_args={"enable_mixfusion": True},
    )
    assert scheduler._build_sampling_params_key(per_prompt) is not None
