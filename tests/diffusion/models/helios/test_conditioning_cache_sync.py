# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the conditioning-cache comparison path.

Inference tensors (the default serving path) carry no version counter, so the
cache used to fall back to a full ``torch.equal`` against an owned snapshot on
every denoise step. On an accelerator that comparison returns a host bool and
therefore synchronises the device inside the denoise loop.
"""

import pytest
import torch

from vllm_omni.diffusion.models.helios.conditioning_cache import (
    HeliosConditioningCache,
    comparison_snapshot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _AcceleratorTensor:
    """Stand-in for an NPU-resident source."""

    device = type("_Device", (), {"type": "npu"})()

    def detach(self):  # pragma: no cover - must never be reached
        raise AssertionError("accelerator sources must not be snapshotted")


def test_versioned_sources_need_no_snapshot():
    assert comparison_snapshot(torch.zeros(2), version=7) is None


def test_accelerator_sources_are_not_copied():
    assert comparison_snapshot(_AcceleratorTensor(), version=None, immutable_accelerator_source=True) is None


def test_host_inference_tensor_is_still_snapshotted():
    with torch.inference_mode():
        source = torch.arange(4.0)
        snapshot = comparison_snapshot(source, version=None)
    assert snapshot is not None
    assert torch.equal(snapshot, source)
    assert snapshot.data_ptr() != source.data_ptr()


def _build(value):
    projected = value * 2
    key_value = value.new_zeros(1, value.shape[1], 2, 4)
    return projected, [(key_value, key_value)]


def test_cache_hit_and_host_mutation_detection():
    with torch.inference_mode():
        with HeliosConditioningCache() as cache:
            source = torch.ones(1, 4, 8)
            first, _ = cache.prepare(source, branch="positive", projection_revision=("r", 1), build=_build)
            hit, _ = cache.prepare(source, branch="positive", projection_revision=("r", 1), build=_build)
            assert hit is first

            source.add_(1.0)
            rebuilt, _ = cache.prepare(source, branch="positive", projection_revision=("r", 1), build=_build)
    assert rebuilt is not first
    assert torch.equal(rebuilt, torch.full((1, 4, 8), 4.0))
