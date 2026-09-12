# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    SupportsVocoderCUDAGraph,
    supports_vocoder_cudagraph,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_capability_discovery_requires_both_declaration_and_provider() -> None:
    class Model(SupportsVocoderCUDAGraph):
        supports_vocoder_cudagraph = True

        def get_vocoder_cudagraph_components(self):
            return ()

    assert supports_vocoder_cudagraph(Model())
    assert not supports_vocoder_cudagraph(object())
