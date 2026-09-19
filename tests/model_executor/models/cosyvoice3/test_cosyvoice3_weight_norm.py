# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized

from tests.model_executor.models.cosyvoice3.test_cosyvoice3_incremental_hift import (
    CONFIGS,
    _chunks,
    _incremental,
    _make_hift,
    _make_model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("config", CONFIGS)
def test_folded_hift_preserves_streaming_waveform(config):
    hift = _make_hift(config)
    model = _make_model(hift, window_len=64)
    chunks = _chunks()
    assert any(is_parametrized(module, "weight") for module in hift.modules())
    with torch.inference_mode():
        reference = _incremental(model, chunks)
        hift.remove_weight_norm()
        actual = _incremental(model, chunks)
    assert all(not is_parametrized(module, "weight") for module in hift.modules())
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_checkpoint_load_folds_hift_after_loading(tmp_path):
    hift = _make_hift()
    model = _make_model(hift, window_len=64)
    model.flow_model = nn.Linear(2, 2)
    torch.save(model.flow_model.state_dict(), tmp_path / "flow.pt")
    torch.save({"generator." + k: v for k, v in hift.state_dict().items()}, tmp_path / "hift.pt")
    expected = {}
    with torch.no_grad():
        for name, module in hift.named_modules():
            if is_parametrized(module, "weight"):
                expected[name] = module.weight.clone()
    model.load_weights(str(tmp_path), torch.device("cpu"))
    assert expected
    for name, weight in expected.items():
        module = hift.get_submodule(name)
        assert not is_parametrized(module, "weight")
        torch.testing.assert_close(module.weight, weight, rtol=0, atol=0)
