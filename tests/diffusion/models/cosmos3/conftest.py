# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch


@pytest.fixture(autouse=True)
def _fresh_dynamo_cache():
    # Multiview tests compile FlexAttention at many geometries. Dynamo's
    # recompile limit is shared by every torch.compile(flex_attention) in the
    # process, so leftover cache entries would make a later model's
    # non-fullgraph compile fall back to eager FlexAttention, which
    # materializes the full fp32 score matrix and OOMs on L4.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
