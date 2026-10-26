# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.interfaces.model_local_cudagraph import (
    SupportsModelLocalCUDAGraph,
    supports_model_local_cudagraph,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_capability_discovery_requires_both_declaration_and_provider() -> None:
    class Model(SupportsModelLocalCUDAGraph):
        supports_model_local_cudagraph = True

        def get_model_local_cudagraph_components(self):
            return ()

    assert supports_model_local_cudagraph(Model())
    assert not supports_model_local_cudagraph(object())
