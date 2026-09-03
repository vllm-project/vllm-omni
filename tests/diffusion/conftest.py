# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.fixture(autouse=True)
def restore_torch_default_dtype():
    """Keep process-global dtype changes from leaking between diffusion tests."""
    default_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)
