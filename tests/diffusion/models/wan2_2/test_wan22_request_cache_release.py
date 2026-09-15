# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The Wan2.2 request-path cache release must not run on XPU.

Returning cached blocks to the driver between denoising and VAE decoding is
harmless elsewhere, but on XPU it moves the decoder's tensors to new device
addresses and the collective backend then maps new peer segments on every
request without releasing the old ones (10.3 GB of host memory per request on
a 4-rank service). These tests pin the platform split and make sure the
pipelines go through the shared helper instead of calling ``empty_cache``
directly.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.models.wan2_2 as wan22_pkg
from vllm_omni.diffusion.models.wan2_2 import device_cache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _platform(*, is_available: bool, is_xpu: bool):
    """A stand-in platform that records ``empty_cache`` calls without a device."""
    calls: list[str] = []
    platform = SimpleNamespace(
        is_available=lambda: is_available,
        is_xpu=lambda: is_xpu,
        empty_cache=lambda: calls.append("empty_cache"),
    )
    return platform, calls


@pytest.mark.parametrize(
    ("is_available", "is_xpu", "expected"),
    [
        (True, False, 1),  # every non-XPU accelerator keeps the previous behaviour
        (True, True, 0),  # XPU skips the request-path release
        (False, False, 0),  # no accelerator: nothing to release (previous behaviour)
    ],
)
def test_release_request_cache_platform_split(is_available, is_xpu, expected) -> None:
    platform, calls = _platform(is_available=is_available, is_xpu=is_xpu)

    device_cache.release_request_cache(platform)

    assert calls.count("empty_cache") == expected


def test_release_request_cache_does_not_probe_xpu_without_an_accelerator() -> None:
    """The pipelines' CPU tests stub the platform with ``is_available`` only."""
    platform = SimpleNamespace(is_available=lambda: False)

    device_cache.release_request_cache(platform)  # must not touch is_xpu / empty_cache


def test_wan22_pipelines_release_cache_through_the_helper() -> None:
    """A direct ``empty_cache()`` in a pipeline would bypass the XPU gate."""
    pkg_dir = Path(wan22_pkg.__file__).parent
    offenders = []
    for path in sorted(pkg_dir.glob("pipeline_wan2_2*.py")):
        text = path.read_text()
        if "current_omni_platform.empty_cache()" in text:
            offenders.append(path.name)
        assert "release_request_cache(current_omni_platform)" in text, (
            f"{path.name} does not release the cache through the shared helper"
        )
    assert offenders == [], f"request-path empty_cache called directly in: {offenders}"
