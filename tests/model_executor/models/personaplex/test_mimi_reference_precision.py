# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test-only codec reference precision lifecycle; no weights or GPU allocation."""

from contextlib import nullcontext

import pytest
import torch

from tests.model_executor.models.personaplex.test_mimi_prefix_flush import _mimi_reference_precision

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fail", [False, True])
def test_reference_precision_restores_after_exit(monkeypatch, fail):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    expected = pytest.raises(RuntimeError, match="codec setup failed") if fail else nullcontext()
    with expected:
        with _mimi_reference_precision("cuda"):
            assert torch.backends.cudnn.allow_tf32 is False
            if fail:
                raise RuntimeError("codec setup failed")
    assert torch.backends.cudnn.allow_tf32 is True
