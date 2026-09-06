"""Unit tests for OmniModelStatePlugin interface correctness."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.worker_v2.model_states.plugin import OmniModelStatePlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ConcretePlugin(OmniModelStatePlugin):
    """Minimal concrete plugin for testing the ABC."""

    def __init__(self):
        self.added: list[tuple[int, Any]] = []
        self.removed: list[int] = []

    def on_add_request(self, req_index, new_req_data):
        self.added.append((req_index, new_req_data))

    def on_remove_request(self, req_index):
        self.removed.append(req_index)

    def postprocess(self, text_hidden, multimodal_outputs, input_batch, req_states):
        return text_hidden, multimodal_outputs


def test_concrete_plugin_instantiates():
    plugin = _ConcretePlugin()
    assert isinstance(plugin, OmniModelStatePlugin)


def test_on_add_request_called():
    plugin = _ConcretePlugin()
    plugin.on_add_request(0, SimpleNamespace(req_id="r1"))
    assert len(plugin.added) == 1
    assert plugin.added[0][0] == 0


def test_on_remove_request_called():
    plugin = _ConcretePlugin()
    plugin.on_remove_request(3)
    assert plugin.removed == [3]


def test_prepare_extra_inputs_default_empty():
    plugin = _ConcretePlugin()
    result = plugin.prepare_extra_inputs(SimpleNamespace(), SimpleNamespace())
    assert result == {}


def test_dummy_run_default_noop():
    plugin = _ConcretePlugin()
    plugin.dummy_run(None, 128)


def test_postprocess_returns_inputs():
    plugin = _ConcretePlugin()
    hidden = torch.randn(4, 8)
    mm = {"audio": torch.randn(4, 2)}
    out_h, out_mm = plugin.postprocess(hidden, mm, None, None)
    assert out_h is hidden
    assert out_mm is mm


def test_abc_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        OmniModelStatePlugin()
