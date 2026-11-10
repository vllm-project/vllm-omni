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
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionRequestInfo
from vllm_omni.diffusion.registry import supports_mixfusion
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
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


MIXFUSION = {"extra_args": {"enable_mixfusion": True}}
PER_PROMPT = {"prompt": "a cat", "additional_information": {"image": "x"}}


@pytest.mark.parametrize(
    ("scheduler_cls", "pipeline_cls", "prompt", "overrides", "expected"),
    [
        # Without MixFusion resolution splits both keys.
        (_Scheduler, _MixfusionCapablePipeline, "a cat", {}, (1024, 768, 1024)),
        (RequestScheduler, _MixfusionCapablePipeline, "a cat", {}, (1024, 768, 1024)),
        # Step batches resolve shapes per request, so MixFusion drops them.
        (_Scheduler, _MixfusionCapablePipeline, "a cat", MIXFUSION, (None, None, None)),
        (
            _Scheduler,
            _MixfusionCapablePipeline,
            [{"prompt": "a cat", "additional_information": {}}],
            MIXFUSION,
            (None, None, None),
        ),
        # Request-mode batches resolve one height/width from the first request.
        (RequestScheduler, _MixfusionCapablePipeline, "a cat", MIXFUSION, (1024, 768, 1024)),
        # Per-prompt inputs cannot share a key once MixFusion is requested ...
        (_Scheduler, _MixfusionCapablePipeline, PER_PROMPT, MIXFUSION, None),
        (RequestScheduler, _MixfusionCapablePipeline, PER_PROMPT, MIXFUSION, None),
        # ... but keep it without MixFusion or on an unsupported pipeline,
        # where the request flag degrades to a no-op.
        (_Scheduler, _MixfusionCapablePipeline, PER_PROMPT, {}, (1024, 768, 1024)),
        (_Scheduler, _PlainPipeline, "a cat", MIXFUSION, (1024, 768, 1024)),
        (_Scheduler, _PlainPipeline, PER_PROMPT, MIXFUSION, (1024, 768, 1024)),
        (RequestScheduler, _PlainPipeline, PER_PROMPT, MIXFUSION, (1024, 768, 1024)),
    ],
)
def test_batch_key_shape(monkeypatch, scheduler_cls, pipeline_cls, prompt, overrides, expected):
    scheduler = _initialized_scheduler(scheduler_cls, monkeypatch, pipeline_cls)
    request = _request(prompt, **overrides)
    if overrides == MIXFUSION and pipeline_cls is _MixfusionCapablePipeline and expected == (None, None, None):
        request.mixfusion_info = MixFusionRequestInfo((1, 3072, 64), 256, 128)
    key = scheduler._build_sampling_params_key(request)

    if expected is None:
        assert key is None
    else:
        assert (key.height, key.width, key.resolution) == expected
        # Only resolution is relaxed; other shape fields keep splitting.
        assert key.num_frames == 1


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
