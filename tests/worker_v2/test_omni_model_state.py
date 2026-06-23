"""Unit tests for OmniModelState core methods and plugin dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.model_states.plugin import OmniModelStatePlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------


class _DummyInputBatch:
    def __init__(self, indices):
        self.idx_mapping_np = indices
        self.num_reqs = len(indices)
        self.num_scheduled_tokens = [1] * len(indices)
        self.query_start_loc_np = list(range(len(indices)))


class _DummyReqState:
    pass


class _SpyPlugin(OmniModelStatePlugin):
    """Plugin that records calls for verification."""

    def __init__(self):
        self.add_calls: list = []
        self.remove_calls: list = []
        self.prepare_calls: list = []
        self.postprocess_calls: list = []

    def on_add_request(self, req_index, new_req_data):
        self.add_calls.append((req_index, new_req_data))

    def on_remove_request(self, req_index):
        self.remove_calls.append(req_index)

    def prepare_extra_inputs(self, input_batch, req_states):
        self.prepare_calls.append(True)
        return {"plugin_key": "plugin_value"}

    def postprocess(self, text_hidden, multimodal_outputs, input_batch, req_states):
        self.postprocess_calls.append(True)
        return text_hidden, multimodal_outputs


def _make_state(
    max_num_reqs=4, has_preprocess=False, has_postprocess=False, have_multimodal_outputs=False, plugins=None
):
    """Create an OmniModelState without calling real __init__."""
    state = object.__new__(OmniModelState)

    # Minimal model mock
    model = MagicMock()
    model.has_preprocess = has_preprocess
    model.has_postprocess = has_postprocess
    model.have_multimodal_outputs = have_multimodal_outputs
    model.preprocess_in_forward = False
    model.get_omni_plugins = MagicMock(return_value=[])
    state.model = model

    # scheduler_config mock
    state.scheduler_config = SimpleNamespace(max_num_seqs=max_num_reqs)

    # Skip DefaultModelState.__init__ side effects
    state.has_preprocess = has_preprocess
    state.has_postprocess = has_postprocess
    state.have_multimodal_outputs = have_multimodal_outputs
    state.plugins = plugins or []

    from vllm_omni.worker_v2.model_states.intermediate_buffer import (
        OmniIntermediateBuffer,
    )

    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
    state._static_inputs_embeds = None
    return state


def _make_new_req_data(req_id="r1"):
    return SimpleNamespace(
        req_id=req_id,
        mm_features=[],
    )


# ---------------------------------------------------------------
# add_request / remove_request
# ---------------------------------------------------------------


def test_add_request_populates_buffer():
    state = _make_state()
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    assert state.intermediate_buffer.buffers[0]["req_id"] == "r1"


def test_add_request_dispatches_to_plugins():
    plugin = _SpyPlugin()
    state = _make_state(plugins=[plugin])
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    assert len(plugin.add_calls) == 1
    assert plugin.add_calls[0][0] == 0


def test_add_request_unflattens_serialized_additional_information():
    state = _make_state()
    hidden = torch.randn(2, 4)
    embeds = torch.randn(3, 4)
    req = SimpleNamespace(
        req_id="r1",
        mm_features=[],
        additional_information=serialize_additional_information(
            {
                "hidden_states": {"output": hidden},
                "embed": {"prefill": embeds},
                "ids": {"prompt": [1, 2, 3], "output": [4]},
            }
        ),
    )

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    buf = state.intermediate_buffer.buffers[0]
    assert torch.equal(buf["hidden_states"]["output"], hidden)
    assert torch.equal(buf["embed"]["prefill"], embeds)
    assert buf["ids"]["prompt"] == [1, 2, 3]
    assert buf["ids"]["output"] == [4]


def test_remove_request_clears_buffer():
    state = _make_state()
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    state.remove_request(0)
    assert state.intermediate_buffer.buffers[0] == {}


def test_remove_request_dispatches_to_plugins():
    plugin = _SpyPlugin()
    state = _make_state(plugins=[plugin])
    state.remove_request(2)
    assert plugin.remove_calls == [2]


def test_intermediate_buffer_update_merges_nested_and_tuple_keys():
    state = _make_state()
    trailing = torch.randn(2, 4)
    last = torch.randn(4)
    codes = torch.ones(1, 16, dtype=torch.long)
    state.intermediate_buffer.buffers[0] = {
        "req_id": "r1",
        "hidden_states": {"trailing_text": trailing},
    }

    state.intermediate_buffer.update(
        0,
        {
            ("codes", "audio"): codes,
            "hidden_states": {"last": last},
        },
        {("codes", "audio"), ("hidden_states", "last")},
    )

    buf = state.intermediate_buffer.buffers[0]
    assert ("codes", "audio") not in buf
    assert torch.equal(buf["codes"]["audio"], codes)
    assert torch.equal(buf["hidden_states"]["trailing_text"], trailing)
    assert torch.equal(buf["hidden_states"]["last"], last)


# ---------------------------------------------------------------
# prepare_inputs
# ---------------------------------------------------------------


def test_prepare_inputs_injects_buffer_and_runtime_info():
    state = _make_state()
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    batch = _DummyInputBatch([0])

    with patch.object(type(state).__bases__[0], "prepare_inputs", return_value={}):
        result = state.prepare_inputs(batch, _DummyReqState())

    assert "model_intermediate_buffer" in result
    assert "runtime_additional_information" in result
    assert len(result["model_intermediate_buffer"]) == 1
    assert result["model_intermediate_buffer"][0]["req_id"] == "r1"


def test_prepare_inputs_merges_plugin_extra():
    plugin = _SpyPlugin()
    state = _make_state(plugins=[plugin])

    batch = _DummyInputBatch([])

    with patch.object(type(state).__bases__[0], "prepare_inputs", return_value={"base": True}):
        result = state.prepare_inputs(batch, _DummyReqState())

    assert result["base"] is True
    assert result["plugin_key"] == "plugin_value"


# ---------------------------------------------------------------
# prepare_dummy_inputs
# ---------------------------------------------------------------


def test_prepare_dummy_inputs_has_buffer_keys():
    state = _make_state()
    with patch.object(type(state).__bases__[0], "prepare_dummy_inputs", return_value={}):
        result = state.prepare_dummy_inputs(num_reqs=2, num_tokens=16)

    assert "model_intermediate_buffer" in result
    assert len(result["model_intermediate_buffer"]) == 2


# ---------------------------------------------------------------
# postprocess_model_output
# ---------------------------------------------------------------


def test_postprocess_omni_output_with_multimodal():
    state = _make_state(have_multimodal_outputs=True)
    hidden = torch.randn(4, 8)
    mm = {"audio": torch.randn(4, 2)}
    omni = OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    batch = _DummyInputBatch([0])
    th, mo = state.postprocess_model_output(omni, batch, _DummyReqState())
    assert th is hidden
    assert mo is mm


def test_postprocess_tuple_output():
    state = _make_state(have_multimodal_outputs=False)
    del state.model.make_omni_output
    hidden = torch.randn(4, 8)

    batch = _DummyInputBatch([0])
    th, mo = state.postprocess_model_output((hidden, {}), batch, _DummyReqState())
    assert th is hidden
    assert mo == {}


def test_postprocess_raw_tensor():
    state = _make_state()
    del state.model.make_omni_output
    hidden = torch.randn(4, 8)

    batch = _DummyInputBatch([0])
    th, mo = state.postprocess_model_output(hidden, batch, _DummyReqState())
    assert th is hidden
    assert mo == {}


def test_postprocess_calls_make_omni_output_when_not_omni():
    """When model output is not OmniOutput but model has make_omni_output, it should be called."""
    state = _make_state(have_multimodal_outputs=True)
    hidden = torch.randn(4, 8)
    mm = {"audio": torch.randn(4, 2)}
    expected_omni = OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    state.model.make_omni_output = MagicMock(return_value=expected_omni)

    batch = _DummyInputBatch([0])
    th, mo = state.postprocess_model_output((hidden, {}), batch, _DummyReqState())
    state.model.make_omni_output.assert_called_once()
    assert th is hidden


def test_postprocess_dispatches_to_plugins():
    plugin = _SpyPlugin()
    state = _make_state(plugins=[plugin])
    hidden = torch.randn(4, 8)

    batch = _DummyInputBatch([0])
    state.postprocess_model_output(hidden, batch, _DummyReqState())
    assert len(plugin.postprocess_calls) == 1


def test_run_postprocess_does_not_duplicate_hidden_states_kwarg():
    state = _make_state(has_postprocess=True)
    state.model.gpu_resident_buffer_keys = set()
    seen = {}
    trailing = torch.ones(1, 2)

    def postprocess(hidden_states, **info):
        seen["hidden_states"] = hidden_states
        seen["info"] = info
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    state.model.postprocess = postprocess
    state.intermediate_buffer.buffers[0] = {
        "req_id": "r1",
        "hidden_states": {"trailing_text": trailing},
        "meta": {"codec_streaming": True},
    }

    hidden = torch.randn(1, 2)
    state.run_postprocess(hidden, _DummyInputBatch([0]))

    assert torch.equal(seen["hidden_states"], hidden)
    assert "hidden_states" not in seen["info"]
    assert seen["info"]["meta"] == {"codec_streaming": True}
    assert torch.equal(state.intermediate_buffer.buffers[0]["hidden_states"]["trailing_text"], trailing)
    assert torch.equal(state.intermediate_buffer.buffers[0]["hidden_states"]["last"], hidden[-1])
