# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterization of ``OmniGPUModelRunner.initialize_metadata_builders``.

Upstream ``FlashAttentionMetadataBuilder`` pre-allocates ``scheduler_metadata``
for only ``max_num_seqs + 1`` entries, but FA3 with split scheduling
(``max_num_splits > 1``) may need ``max_num_seqs * max_num_splits + 1`` during
CUDA-graph capture. ``OmniGPUModelRunner`` re-sizes the buffer up to that
requirement after upstream init — this test pins that Omni-specific workaround
exactly.

It drives the Omni runner's OWN method (with the upstream ``super()`` call
stubbed and a fake builder standing in), so a change to the resize is caught and
a failure is unambiguously an Omni-runner regression. Pure CPU — no model, no
CUDA, no flashinfer.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The class ``super().initialize_metadata_builders`` dispatches to.
_UPSTREAM_RUNNER = OmniGPUModelRunner.__mro__[1]


def _builder(*, prealloc: int | None, max_num_splits: int) -> SimpleNamespace:
    sm = None if prealloc is None else torch.zeros(prealloc, dtype=torch.int32)
    return SimpleNamespace(scheduler_metadata=sm, max_num_splits=max_num_splits)


def _make_runner(monkeypatch, *, max_num_seqs: int, builders: list) -> OmniGPUModelRunner:
    # Stub the upstream initialize_metadata_builders; our fake runner already
    # carries the attn_groups the Omni override iterates over.
    monkeypatch.setattr(_UPSTREAM_RUNNER, "initialize_metadata_builders", lambda self, *a, **k: None)
    runner = object.__new__(OmniGPUModelRunner)
    runner.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    runner.cache_config = SimpleNamespace(enable_prefix_caching=False)  # skip omni_prefix_cache branch
    runner.attn_groups = [[SimpleNamespace(metadata_builders=builders)]]
    return runner


def test_resizes_up_when_split_scheduling_needs_more(monkeypatch):
    max_num_seqs, splits, prealloc = 32, 4, 33  # 33 == upstream default (max_num_seqs + 1)
    builder = _builder(prealloc=prealloc, max_num_splits=splits)
    runner = _make_runner(monkeypatch, max_num_seqs=max_num_seqs, builders=[builder])

    OmniGPUModelRunner.initialize_metadata_builders(runner, None, None)

    assert builder.scheduler_metadata.shape[0] == max_num_seqs * splits + 1
    assert builder.scheduler_metadata.dtype == torch.int32


def test_no_resize_when_not_split_scheduling(monkeypatch):
    builder = _builder(prealloc=33, max_num_splits=1)  # max_num_splits <= 1 -> workaround skipped
    runner = _make_runner(monkeypatch, max_num_seqs=32, builders=[builder])
    original = builder.scheduler_metadata

    OmniGPUModelRunner.initialize_metadata_builders(runner, None, None)

    assert builder.scheduler_metadata is original  # untouched


def test_no_resize_when_prealloc_already_sufficient(monkeypatch):
    max_num_seqs, splits = 32, 4
    builder = _builder(prealloc=max_num_seqs * splits + 1, max_num_splits=splits)  # already big enough
    runner = _make_runner(monkeypatch, max_num_seqs=max_num_seqs, builders=[builder])
    original = builder.scheduler_metadata

    OmniGPUModelRunner.initialize_metadata_builders(runner, None, None)

    assert builder.scheduler_metadata is original


def test_no_crash_when_builder_has_no_scheduler_metadata(monkeypatch):
    builder = _builder(prealloc=None, max_num_splits=4)  # scheduler_metadata is None
    runner = _make_runner(monkeypatch, max_num_seqs=32, builders=[builder])

    OmniGPUModelRunner.initialize_metadata_builders(runner, None, None)

    assert builder.scheduler_metadata is None
