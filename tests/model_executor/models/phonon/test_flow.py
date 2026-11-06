# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_path = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/phonon/flow.py"
_spec = importlib.util.spec_from_file_location("phonon_flow", _path)
_flow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flow)


def test_time_norm_and_stateless_flow():
    inputs = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
    norm = _flow.TimeRMSNorm(4)
    torch.testing.assert_close(norm(inputs), inputs / (inputs.square().mean(-1, keepdim=True) + 1e-5).sqrt())
    torch.manual_seed(41)
    head = _flow.PhononFlowHead(4, 8, 6, 2).eval()
    condition = torch.randn(2, 6)
    noise = torch.randn(2, 4)
    before = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
    with torch.inference_mode():
        actual = head(condition, noise, steps=2)
        repeated = head(condition, noise, steps=2)
        individual = torch.cat([head(c[None], n[None], steps=2) for c, n in zip(condition, noise)])
    torch.testing.assert_close(actual, repeated)
    torch.testing.assert_close(actual, individual)
    for name, tensor in head.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    with pytest.raises(ValueError, match="positive"):
        head(condition, noise, steps=0)
