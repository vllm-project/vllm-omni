# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only fixtures for routing tests over the live Buildkite pipelines."""

import pytest

from tests.buildkite.source_dependency_helpers import PIPELINES, load_pipeline


@pytest.fixture(scope="module", params=PIPELINES)
def nightly_pipeline(request):
    """Provide each platform's pipeline without loading model/hardware fixtures."""
    platform = request.param
    path = PIPELINES[platform]
    return platform, path, load_pipeline(path)
