"""Unit tests for OmniModelState core methods and plugin dispatch."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states.omni_model_state import (
    OmniModelState,
    _make_safe_get_rope,
)
from vllm_omni.worker_v2.model_states.plugin import OmniModelStatePlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------


class _DummyInputBatch:
    def __init__(self, indices, *, num_computed_tokens_cpu=None):
        self.idx_mapping_np = indices
        self.num_reqs = len(indices)
        self.num_scheduled_tokens = [1] * len(indices)
        self.query_start_loc_np = list(range(len(indices)))
        if num_computed_tokens_cpu is not None:
            self.num_computed_tokens_cpu = np.array(num_computed_tokens_cpu, dtype=np.int32)


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
    model.gpu_resident_buffer_keys = set()
    model.talker_mtp_accepts_per_row_generators = False
    model.talker_mtp_accepts_req_infos = False
    model.talker_mtp_output_key = ("codes", "audio")
    model.talker_mtp_validity_key = None
    model.requires_native_model_intermediate_buffer = False
    model.preprocess_batch_mrv2 = None
    model.preprocess_decode_batch_mrv2 = None
    model.postprocess_batch_mrv2 = None
    model.get_omni_plugins = MagicMock(return_value=[])
    state.model = model

    # scheduler_config mock
    state.scheduler_config = SimpleNamespace(max_num_seqs=max_num_reqs)
    state.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))

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
    state._talker_mtp_generators = {}
    state._mtp_input_ids = None
    state._mtp_input_embeds = None
    state._mtp_hidden = None
    state._mtp_text_step = None
    state._mtp_offsets = None
    state._talker_mtp_runner = None
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


def test_add_request_initializes_declared_validity_for_upstream_warmup():
    state = _make_state()
    state.model.talker_mtp_validity_key = ("meta", "codec_frame_valid")
    req = _make_new_req_data("_warmup_0_")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    validity = state.intermediate_buffer.buffers[0]["meta"]["codec_frame_valid"]
    assert isinstance(validity, torch.Tensor)
    assert validity.dtype == torch.bool
    assert validity.item() is False


def test_add_request_does_not_mask_missing_validity_for_real_requests():
    state = _make_state()
    state.model.talker_mtp_validity_key = ("meta", "codec_frame_valid")
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    assert "meta" not in state.intermediate_buffer.buffers[0]


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


def test_remove_request_accepts_req_id():
    state = _make_state()
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    state.remove_request("r1")
    assert state.intermediate_buffer.buffers[0] == {}


def test_remove_request_ignores_unknown_req_id():
    state = _make_state()

    state.remove_request("missing")

    assert state.intermediate_buffer.buffers == [{}, {}, {}, {}]


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


def test_prepare_inputs_native_buffer_models_skip_runtime_info():
    state = _make_state()
    state.model.requires_native_model_intermediate_buffer = True
    req = _make_new_req_data("r1")

    with patch.object(type(state).__bases__[0], "add_request", return_value=None):
        state.add_request(0, req)

    batch = _DummyInputBatch([0])

    with patch.object(type(state).__bases__[0], "prepare_inputs", return_value={}):
        result = state.prepare_inputs(batch, _DummyReqState())

    assert "model_intermediate_buffer" in result
    assert "runtime_additional_information" not in result
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


def test_postprocess_native_buffer_models_skip_runtime_info():
    state = _make_state(have_multimodal_outputs=True)
    state.model.requires_native_model_intermediate_buffer = True
    hidden = torch.randn(4, 8)
    expected_omni = OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
    state.model.make_omni_output = MagicMock(return_value=expected_omni)
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}

    batch = _DummyInputBatch([0])
    state.postprocess_model_output((hidden, {}), batch, _DummyReqState())

    _, kwargs = state.model.make_omni_output.call_args
    assert "model_intermediate_buffer" in kwargs
    assert "runtime_additional_information" not in kwargs


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


def test_run_postprocess_batches_gpu_last_hidden_writeback():
    state = _make_state(max_num_reqs=2, has_postprocess=True)
    state.model.gpu_resident_buffer_keys = {("hidden_states", "last")}
    state.intermediate_buffer.buffers[0] = {"req_id": "r0"}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1"}
    state.model.postprocess = MagicMock(side_effect=AssertionError("batch hook must replace scalar postprocess"))
    calls = []

    def postprocess_batch(*, hidden_states, last_token_indices):
        calls.append((hidden_states, last_token_indices.detach().clone()))
        return (
            ("hidden_states", "last"),
            hidden_states.index_select(0, last_token_indices),
        )

    state.model.postprocess_batch_mrv2 = postprocess_batch
    batch = _DummyInputBatch([1, 0])
    batch.num_scheduled_tokens = [2, 2]
    batch.query_start_loc_np = [0, 2]
    batch.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    state.run_postprocess(hidden, batch)

    state.model.postprocess.assert_not_called()
    assert len(calls) == 1
    assert torch.equal(calls[0][1], torch.tensor([1, 3], dtype=torch.int32))
    assert torch.equal(state.intermediate_buffer.buffers[1]["hidden_states"]["last"], hidden[1])
    assert torch.equal(state.intermediate_buffer.buffers[0]["hidden_states"]["last"], hidden[3])
    first = state.intermediate_buffer.buffers[0]["hidden_states"]["last"]
    second = state.intermediate_buffer.buffers[1]["hidden_states"]["last"]
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def test_run_preprocess_passes_prefix_cache_prefill_boundary_metadata():
    state = _make_state(has_preprocess=True)
    seen = {}

    def preprocess(input_ids, input_embeds, **info):
        seen.update(info)
        assert info["_omni_num_computed_tokens"] == 9
        assert info["_omni_prompt_len"] == 10
        assert info["_omni_is_prefill"] is True
        return input_ids, input_embeds, {}

    state.model.preprocess = preprocess
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    model_inputs = {
        "input_ids": torch.tensor([123], dtype=torch.long),
        "inputs_embeds": torch.zeros(1, 4),
    }
    batch = _DummyInputBatch([0], num_computed_tokens_cpu=[9])
    req_states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array([10], dtype=np.int32)),
        num_computed_tokens=SimpleNamespace(np=np.array([9], dtype=np.int32)),
    )

    state.run_preprocess(batch, model_inputs, req_states)

    assert seen["_omni_num_computed_tokens"] == 9


def test_run_preprocess_skips_upstream_warmup_requests():
    state = _make_state(has_preprocess=True)
    state.model.preprocess = MagicMock(side_effect=AssertionError("warmup must bypass omni conditioning"))
    state.intermediate_buffer.buffers[0] = {"req_id": "_warmup_0_"}
    model_inputs = {
        "input_ids": torch.tensor([123], dtype=torch.long),
        "inputs_embeds": torch.zeros(1, 4),
    }

    state.run_preprocess(_DummyInputBatch([0]), model_inputs)

    state.model.preprocess.assert_not_called()


def test_run_preprocess_propagates_protocol_errors_for_real_requests():
    state = _make_state(has_preprocess=True)

    def preprocess(_input_ids, _input_embeds, **_info):
        raise RuntimeError("native MRv2 conditioning credit violated")

    state.model.preprocess = preprocess
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    model_inputs = {
        "input_ids": torch.tensor([123], dtype=torch.long),
        "inputs_embeds": torch.zeros(1, 4),
    }

    with pytest.raises(RuntimeError, match="conditioning credit violated"):
        state.run_preprocess(_DummyInputBatch([0]), model_inputs)


def test_run_preprocess_refreshes_static_decode_embeddings_from_input_ids():
    state = _make_state(has_preprocess=True)
    static_embeds = torch.zeros(1, 4)
    state._static_inputs_embeds = static_embeds
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    state.model.embed_input_ids = lambda input_ids: expected.to(input_ids.device)
    seen = {}

    def preprocess(input_ids, input_embeds, **_info):
        seen["input_embeds"] = input_embeds.detach().clone()
        return input_ids, input_embeds, {}

    state.model.preprocess = preprocess
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    model_inputs = {
        "input_ids": torch.tensor([1049], dtype=torch.long),
        "inputs_embeds": static_embeds[:1],
    }

    state.run_preprocess(_DummyInputBatch([0]), model_inputs)

    assert torch.equal(seen["input_embeds"], expected)
    assert torch.equal(static_embeds, expected)


def test_preprocess_writeback_skips_identity_results_only():
    original = torch.tensor([[1.0, 2.0]])

    assert OmniModelState._preprocess_result_needs_writeback(original, original) is False
    assert OmniModelState._preprocess_result_needs_writeback(original, original.view_as(original)) is True


def test_stage_preprocess_batches_declared_cpu_rows_once(monkeypatch):
    state = _make_state(max_num_reqs=2)
    state.model.batched_gpu_staging_keys = {("embed", "decode")}
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    state.intermediate_buffer.buffers[0] = {"req_id": "r0", "embed": {"decode": first}}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1", "embed": {"decode": second}}
    calls = []

    def fake_batch_move(tensors, device):
        calls.append((list(tensors), device))
        return [tensor.clone() for tensor in tensors]

    monkeypatch.setattr(state, "_batch_move_tensor_rows", fake_batch_move)

    state._stage_batched_preprocess_inputs([0, 1], torch.device("cuda"))

    assert len(calls) == 1
    assert calls[0][0][0] is first
    assert calls[0][0][1] is second
    assert calls[0][1] == torch.device("cuda")
    assert torch.equal(state.intermediate_buffer.buffers[0]["embed"]["decode"], first)
    assert torch.equal(state.intermediate_buffer.buffers[1]["embed"]["decode"], second)


def test_run_preprocess_batches_deferred_talker_text_projection_once():
    state = _make_state(max_num_reqs=2, has_preprocess=True)
    projected_inputs = []

    def project_talker_text_steps(text_steps):
        projected_inputs.append(text_steps.detach().clone())
        return text_steps + 100

    state.model.project_talker_text_steps = project_talker_text_steps

    def preprocess(input_ids, input_embeds, **info):
        assert info["_omni_defer_talker_text_projection"] is True
        req_offset = 10 if info["req_id"] == "r1" else 20
        return (
            input_ids,
            input_embeds,
            {
                "mtp_inputs": (
                    torch.full((1, 2), float(req_offset)),
                    torch.tensor([[float(req_offset + 1), float(req_offset + 2)]]),
                ),
                "mtp_text_step_requires_projection": True,
            },
        )

    state.model.preprocess = preprocess
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    state.intermediate_buffer.buffers[1] = {"req_id": "r2"}
    model_inputs = {
        "input_ids": torch.tensor([101, 202], dtype=torch.long),
        "inputs_embeds": torch.zeros(2, 2),
    }
    seen_batches = []
    state._run_batched_mtp = lambda batches, *_args: seen_batches.extend(batches)

    state.run_preprocess(_DummyInputBatch([0, 1]), model_inputs)

    assert len(projected_inputs) == 1
    assert torch.equal(projected_inputs[0], torch.tensor([[11.0, 12.0], [21.0, 22.0]]))
    assert torch.equal(seen_batches[0][2][1], torch.tensor([[111.0, 112.0]]))
    assert torch.equal(seen_batches[1][2][1], torch.tensor([[121.0, 122.0]]))


def test_run_preprocess_dispatches_decode_cohort_to_model_batch_hook():
    state = _make_state(max_num_reqs=2, has_preprocess=True)
    state.intermediate_buffer.buffers[0] = {"req_id": "r1", "meta": {"step": 1}}
    state.intermediate_buffer.buffers[1] = {"req_id": "r2", "meta": {"step": 2}}
    batch_calls = []

    def preprocess_decode_batch(*, input_ids, input_embeds, req_infos):
        batch_calls.append((input_ids.detach().clone(), input_embeds.detach().clone(), list(req_infos)))
        return (
            input_ids + 10,
            input_embeds + 20,
            torch.tensor([[31.0, 32.0], [41.0, 42.0]]),
            torch.tensor([[51.0, 52.0], [61.0, 62.0]]),
            [
                {"meta": {"step": 11}},
                {"meta": {"step": 12}},
            ],
        )

    state.model.preprocess_decode_batch_mrv2 = preprocess_decode_batch
    state.model.preprocess = MagicMock(side_effect=AssertionError("decode requests must use the batch hook"))
    seen_mtp_batches = []
    seen_prepacked_mtp = []

    def capture_mtp(batches, *_args, prepacked_mtp_inputs=None):
        seen_mtp_batches.extend(batches)
        seen_prepacked_mtp.append(prepacked_mtp_inputs)

    state._run_batched_mtp = capture_mtp
    model_inputs = {
        "input_ids": torch.tensor([101, 202], dtype=torch.long),
        "inputs_embeds": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    req_states = SimpleNamespace(prompt_len=np.array([1, 1], dtype=np.int32))
    input_batch = _DummyInputBatch([0, 1], num_computed_tokens_cpu=[1, 1])

    state.run_preprocess(input_batch, model_inputs, req_states)

    state.model.preprocess.assert_not_called()
    assert len(batch_calls) == 1
    assert torch.equal(batch_calls[0][0], torch.tensor([101, 202]))
    assert torch.equal(batch_calls[0][1], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert [info["req_id"] for info in batch_calls[0][2]] == ["r1", "r2"]
    assert torch.equal(model_inputs["input_ids"], torch.tensor([111, 212]))
    assert torch.equal(model_inputs["inputs_embeds"], torch.tensor([[21.0, 22.0], [23.0, 24.0]]))
    assert state.intermediate_buffer.buffers[0]["meta"]["step"] == 11
    assert state.intermediate_buffer.buffers[1]["meta"]["step"] == 12
    assert torch.equal(seen_mtp_batches[0][2][0], torch.tensor([[31.0, 32.0]]))
    assert torch.equal(seen_mtp_batches[0][2][1], torch.tensor([[51.0, 52.0]]))
    assert torch.equal(seen_mtp_batches[1][2][0], torch.tensor([[41.0, 42.0]]))
    assert torch.equal(seen_mtp_batches[1][2][1], torch.tensor([[61.0, 62.0]]))
    assert len(seen_prepacked_mtp) == 1
    assert torch.equal(seen_prepacked_mtp[0][0], torch.tensor([[31.0, 32.0], [41.0, 42.0]]))
    assert torch.equal(seen_prepacked_mtp[0][1], torch.tensor([[51.0, 52.0], [61.0, 62.0]]))


def test_run_preprocess_reuses_prefix_view_for_decode_rows_before_prefill():
    state = _make_state(max_num_reqs=2, has_preprocess=True)
    state.intermediate_buffer.buffers[0] = {"req_id": "decode"}
    state.intermediate_buffer.buffers[1] = {"req_id": "prefill"}
    batch_input_ptrs = []

    def preprocess_decode_batch(*, input_ids, input_embeds, req_infos):
        batch_input_ptrs.append((input_ids.data_ptr(), input_embeds.data_ptr()))
        return (
            input_ids,
            input_embeds,
            torch.tensor([[31.0, 32.0]]),
            torch.tensor([[41.0, 42.0]]),
            [{}],
        )

    state.model.preprocess_decode_batch_mrv2 = preprocess_decode_batch
    state.model.preprocess = lambda input_ids, input_embeds, **_info: (
        input_ids,
        input_embeds,
        {},
    )
    state._run_batched_mtp = lambda *_args, **_kwargs: None
    model_inputs = {
        "input_ids": torch.tensor([101, 201, 202, 203], dtype=torch.long),
        "inputs_embeds": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }
    req_states = SimpleNamespace(prompt_len=np.array([1, 3], dtype=np.int32))
    input_batch = _DummyInputBatch([0, 1], num_computed_tokens_cpu=[1, 0])
    input_batch.num_scheduled_tokens = [1, 3]
    input_batch.query_start_loc_np = [0, 1]

    state.run_preprocess(input_batch, model_inputs, req_states)

    assert batch_input_ptrs == [
        (
            model_inputs["input_ids"].data_ptr(),
            model_inputs["inputs_embeds"].data_ptr(),
        )
    ]


def test_run_preprocess_batches_native_prefill_admission_before_scalar_calls():
    state = _make_state(max_num_reqs=2, has_preprocess=True)
    state.intermediate_buffer.buffers[0] = {"req_id": "prefill"}
    state.intermediate_buffer.buffers[1] = {"req_id": "decode"}
    seen_batches = []
    state.model.preprocess_batch_mrv2 = lambda req_infos, device: seen_batches.append((list(req_infos), device))
    state.model.preprocess = lambda input_ids, input_embeds, **_info: (
        input_ids,
        input_embeds,
        {},
    )
    req_states = SimpleNamespace(prompt_len=np.array([4, 1], dtype=np.int32))
    input_batch = _DummyInputBatch([0, 1], num_computed_tokens_cpu=[0, 1])
    model_inputs = {
        "input_ids": torch.tensor([101, 202], dtype=torch.long),
        "inputs_embeds": torch.zeros(2, 2),
    }

    state.run_preprocess(input_batch, model_inputs, req_states)

    assert seen_batches == [([state.intermediate_buffer.buffers[0]], model_inputs["inputs_embeds"].device)]


def test_run_preprocess_scalar_sees_artifacts_written_by_prefill_batch_hook():
    state = _make_state(max_num_reqs=1, has_preprocess=True)
    state.intermediate_buffer.buffers[0] = {"req_id": "prefill"}
    seen_infos = []

    def preprocess_batch_mrv2(*, req_infos, device):
        req_infos[0]["batched_artifact"] = torch.tensor([7], device=device)

    def preprocess(input_ids, input_embeds, **info):
        seen_infos.append(info)
        return input_ids, input_embeds, {}

    state.model.preprocess_batch_mrv2 = preprocess_batch_mrv2
    state.model.preprocess = preprocess
    req_states = SimpleNamespace(prompt_len=np.array([1], dtype=np.int32))
    input_batch = _DummyInputBatch([0], num_computed_tokens_cpu=[0])
    model_inputs = {
        "input_ids": torch.tensor([101], dtype=torch.long),
        "inputs_embeds": torch.zeros(1, 2),
    }

    state.run_preprocess(input_batch, model_inputs, req_states)

    assert torch.equal(seen_infos[0]["batched_artifact"], torch.tensor([7]))
    assert seen_infos[0]["_omni_is_prefill"] is True
    assert seen_infos[0]["_omni_num_computed_tokens"] == 0
    assert seen_infos[0]["_omni_prompt_len"] == 1


def test_talker_mtp_capture_sizes_do_not_exceed_scheduler_capacity():
    state = _make_state(max_num_reqs=64)
    state.vllm_config.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[128, 120, 64, 32, 1])

    assert state._get_talker_mtp_capture_sizes() == [64, 32, 1]


def test_pack_talker_mtp_batch_fills_static_buffers_in_request_order():
    state = _make_state(max_num_reqs=3)
    state._mtp_input_ids = torch.zeros(3, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros(3, 2)
    state._mtp_hidden = torch.zeros(3, 2)
    state._mtp_text_step = torch.zeros(3, 2)
    input_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    input_embeds = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    )
    mtp_batches = [
        (0, 3, (torch.tensor([[31.0, 32.0]]), torch.tensor([[41.0, 42.0]]))),
        (1, 1, (torch.tensor([[51.0, 52.0]]), torch.tensor([[61.0, 62.0]]))),
    ]

    batch_ids, batch_embeds, batch_hidden, batch_text_step, offsets = state._pack_talker_mtp_batch(
        mtp_batches,
        input_ids,
        input_embeds,
    )

    assert torch.equal(offsets, torch.tensor([3, 1]))
    assert torch.equal(batch_ids, torch.tensor([40, 20]))
    assert torch.equal(batch_embeds, torch.tensor([[7.0, 8.0], [3.0, 4.0]]))
    assert torch.equal(batch_hidden, torch.tensor([[31.0, 32.0], [51.0, 52.0]]))
    assert torch.equal(batch_text_step, torch.tensor([[41.0, 42.0], [61.0, 62.0]]))


def test_talker_mtp_full_decode_cohort_uses_static_long_device_offsets(monkeypatch):
    state = _make_state(max_num_reqs=2)
    state._mtp_offsets = torch.zeros(2, dtype=torch.long)
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    input_batch = SimpleNamespace(num_reqs=2, query_start_loc=query_start_loc)
    mtp_batches = [
        (0, 0, (torch.ones(1, 2), torch.ones(1, 2))),
        (1, 3, (torch.ones(1, 2), torch.ones(1, 2))),
    ]

    monkeypatch.setattr(
        torch,
        "as_tensor",
        lambda *_args, **_kwargs: pytest.fail(
            "full decode cohorts must copy InputBatch.query_start_loc into the static buffer"
        ),
    )

    offsets = state._talker_mtp_batch_offsets(
        mtp_batches,
        input_batch,
        query_start_loc.device,
    )

    assert offsets.data_ptr() == state._mtp_offsets.data_ptr()
    assert offsets.dtype == torch.long
    assert offsets.tolist() == [0, 3]


def test_talker_mtp_decode_prefix_uses_static_device_offsets(monkeypatch):
    state = _make_state(max_num_reqs=3)
    state._mtp_offsets = torch.zeros(3, dtype=torch.long)
    query_start_loc = torch.tensor([0, 1, 2, 6], dtype=torch.int32)
    input_batch = SimpleNamespace(num_reqs=3, query_start_loc=query_start_loc)
    mtp_batches = [
        (0, 0, (torch.ones(1, 2), torch.ones(1, 2))),
        (1, 1, (torch.ones(1, 2), torch.ones(1, 2))),
    ]

    monkeypatch.setattr(
        torch,
        "as_tensor",
        lambda *_args, **_kwargs: pytest.fail("a contiguous decode prefix must reuse device query offsets"),
    )

    offsets = state._talker_mtp_batch_offsets(
        mtp_batches,
        input_batch,
        query_start_loc.device,
    )

    assert offsets.data_ptr() == state._mtp_offsets.data_ptr()
    assert offsets.tolist() == [0, 1]


def test_talker_mtp_mixed_cohort_keeps_explicit_offset_fallback():
    state = _make_state(max_num_reqs=3)
    input_batch = SimpleNamespace(
        num_reqs=3,
        query_start_loc=torch.tensor([0, 2, 5, 9], dtype=torch.int32),
    )
    mtp_batches = [
        (0, 0, (torch.ones(1, 2), torch.ones(1, 2))),
        (2, 5, (torch.ones(1, 2), torch.ones(1, 2))),
    ]

    offsets = state._talker_mtp_batch_offsets(
        mtp_batches,
        input_batch,
        input_batch.query_start_loc.device,
    )

    assert offsets.dtype == torch.long
    assert offsets.tolist() == [0, 5]


def test_pack_talker_mtp_batch_gathers_directly_into_static_buffers(monkeypatch):
    state = _make_state(max_num_reqs=2)
    state._mtp_input_ids = torch.zeros(2, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros(2, 2)
    state._mtp_hidden = torch.zeros(2, 2)
    state._mtp_text_step = torch.zeros(2, 2)
    input_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    input_embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mtp_batches = [
        (0, 2, (torch.tensor([[31.0, 32.0]]), torch.tensor([[41.0, 42.0]]))),
        (1, 0, (torch.tensor([[51.0, 52.0]]), torch.tensor([[61.0, 62.0]]))),
    ]
    index_select_outs = []
    cat_outs = []
    original_index_select = torch.index_select
    original_cat = torch.cat

    def record_index_select(input_tensor, dim, index, *, out=None):
        index_select_outs.append(out)
        return original_index_select(input_tensor, dim, index, out=out)

    def record_cat(tensors, dim=0, *, out=None):
        cat_outs.append(out)
        return original_cat(tensors, dim=dim, out=out)

    monkeypatch.setattr(torch, "index_select", record_index_select)
    monkeypatch.setattr(torch, "cat", record_cat)

    state._pack_talker_mtp_batch(mtp_batches, input_ids, input_embeds)

    assert [tensor.data_ptr() for tensor in index_select_outs] == [
        state._mtp_input_ids.data_ptr(),
        state._mtp_input_embeds.data_ptr(),
    ]
    assert [tensor.data_ptr() for tensor in cat_outs] == [
        state._mtp_hidden.data_ptr(),
        state._mtp_text_step.data_ptr(),
    ]


def test_pack_talker_mtp_batch_casts_runtime_int32_ids_to_static_int64() -> None:
    state = _make_state(max_num_reqs=1)
    state._mtp_input_ids = torch.zeros(1, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros(1, 2)
    state._mtp_hidden = torch.zeros(1, 2)
    state._mtp_text_step = torch.zeros(1, 2)

    batch_ids, *_ = state._pack_talker_mtp_batch(
        [(0, 1, (torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]])))],
        torch.tensor([10, 20], dtype=torch.int32),
        torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
    )

    assert batch_ids.dtype == torch.long
    assert batch_ids.tolist() == [20]


def test_remove_request_clears_talker_mtp_generator():
    state = _make_state()
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    state._talker_mtp_generators["r1"] = torch.Generator(device="cpu")

    state.remove_request(0)

    assert "r1" not in state._talker_mtp_generators


def test_init_talker_mtp_runner_respects_disable_graph_marker():
    state = object.__new__(OmniModelState)
    state.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True))
    )
    talker_mtp = MagicMock(name="talker_mtp")
    model = SimpleNamespace(
        talker_mtp=talker_mtp,
        talker=object(),
        talker_mtp_disable_graph=True,
    )

    runner = state._init_talker_mtp_runner(model)

    assert runner is talker_mtp


def test_talker_mtp_generator_ignores_vllm_sampling_seed():
    state = _make_state()
    sampling_params = SimpleNamespace(extra_args={}, seed=42)

    assert state._get_talker_mtp_generator("r1", sampling_params, torch.device("cpu")) is None


def test_run_batched_mtp_passes_sampling_kwargs_and_generators():
    state = _make_state(max_num_reqs=2)
    state.vllm_config.model_config.subtalker_sampling_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 32,
        "top_p": 0.95,
    }
    state.model.talker_mtp_accepts_per_row_generators = True

    embed_dim = 3
    state._mtp_input_ids = torch.zeros(2, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros((2, embed_dim))
    state._mtp_hidden = torch.zeros((2, embed_dim))
    state._mtp_text_step = torch.zeros((2, embed_dim))
    state.intermediate_buffer.buffers[0] = {
        "req_id": "r0",
        "sampling_params": SimpleNamespace(extra_args={"tts_local_seed": 11}, seed=None),
    }
    state.intermediate_buffer.buffers[1] = {
        "req_id": "r1",
        "sampling_params": SimpleNamespace(extra_args={"tts_local_seed": 22}, seed=None),
    }

    def talker_mtp(input_ids, input_embeds, last_hidden, text_step, **kwargs):
        assert input_ids.tolist() == [101, 202]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_k"] == 32
        assert kwargs["top_p"] == 0.95
        assert len(kwargs["generators"]) == 2
        assert all(isinstance(generator, torch.Generator) for generator in kwargs["generators"])
        return input_embeds + 10, torch.tensor([[1, 2, 3], [4, 5, 6]])

    state.model.talker_mtp = talker_mtp
    input_ids = torch.tensor([101, 202], dtype=torch.long)
    embeds = torch.zeros((2, embed_dim))
    batch = _DummyInputBatch([0, 1])
    mtp_batches = [
        (0, 0, (torch.ones(embed_dim), torch.ones(embed_dim) * 2)),
        (1, 1, (torch.ones(embed_dim) * 3, torch.ones(embed_dim) * 4)),
    ]

    with patch("vllm.forward_context.set_forward_context", return_value=nullcontext()):
        state._run_batched_mtp(mtp_batches, input_ids, embeds, batch, {("codes", "audio")})

    assert torch.equal(embeds, torch.full((2, embed_dim), 10.0))
    assert torch.equal(state.intermediate_buffer.buffers[0]["codes"]["audio"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(state.intermediate_buffer.buffers[1]["codes"]["audio"], torch.tensor([[4, 5, 6]]))


def test_run_batched_mtp_resolves_each_request_generator_once():
    state = _make_state(max_num_reqs=2)
    state.intermediate_buffer.buffers[0] = {"req_id": "r0"}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1"}
    state.model.talker_mtp = lambda _ids, input_embeds, _hidden, _step, **_kwargs: (
        input_embeds,
        None,
    )
    state._get_talker_mtp_generator = MagicMock(return_value=None)
    input_ids = torch.tensor([101, 202], dtype=torch.long)
    embeds = torch.zeros((2, 3))
    batch = _DummyInputBatch([0, 1])
    mtp_batches = [
        (0, 0, (torch.ones(3), torch.ones(3) * 2)),
        (1, 1, (torch.ones(3) * 3, torch.ones(3) * 4)),
    ]

    with patch("vllm.forward_context.set_forward_context", return_value=nullcontext()):
        state._run_batched_mtp(mtp_batches, input_ids, embeds, batch, set())

    assert state._get_talker_mtp_generator.call_count == 2


def test_run_batched_mtp_batches_gpu_codec_state_writeback():
    state = _make_state(max_num_reqs=2)
    state._mtp_input_ids = torch.zeros(2, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros((2, 3))
    state._mtp_hidden = torch.zeros((2, 3))
    state._mtp_text_step = torch.zeros((2, 3))
    state.intermediate_buffer.buffers[0] = {"req_id": "r0"}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1"}
    source_codes = torch.tensor([[1, 2, 3], [4, 5, 6]])
    state.model.talker_mtp = lambda input_ids, input_embeds, last_hidden, text_step, **kwargs: (
        input_embeds,
        source_codes,
    )
    state.model.talker_mtp_validity_key = ("meta", "codec_frame_valid")
    state.intermediate_buffer.update = MagicMock(
        side_effect=AssertionError("GPU codec rows must not use scalar update")
    )
    state.intermediate_buffer.update_gpu_tensor_rows = MagicMock()
    input_ids = torch.tensor([101, 202], dtype=torch.long)
    embeds = torch.zeros((2, 3))
    batch = _DummyInputBatch([0, 1])
    mtp_batches = [
        (0, 0, (torch.ones(3), torch.ones(3) * 2)),
        (1, 1, (torch.ones(3) * 3, torch.ones(3) * 4)),
    ]

    with patch("vllm.forward_context.set_forward_context", return_value=nullcontext()):
        state._run_batched_mtp(
            mtp_batches,
            input_ids,
            embeds,
            batch,
            {("codes", "audio"), ("meta", "codec_frame_valid")},
        )

    assert state.intermediate_buffer.update_gpu_tensor_rows.call_count == 2
    code_call, validity_call = state.intermediate_buffer.update_gpu_tensor_rows.call_args_list
    req_indices, key, values = code_call.args
    assert req_indices == [0, 1]
    assert key == ("codes", "audio")
    assert values.data_ptr() == source_codes.data_ptr()
    req_indices, key, values = validity_call.args
    assert req_indices == [0, 1]
    assert key == ("meta", "codec_frame_valid")
    assert torch.equal(values, torch.ones(2, dtype=torch.bool))
    assert validity_call.kwargs == {"keepdim": False}


def test_seeded_talker_mtp_bypasses_outer_graph_runner():
    state = _make_state(max_num_reqs=2)
    raw_calls = []

    def raw_talker_mtp(input_ids, input_embeds, last_hidden, text_step, **kwargs):
        raw_calls.append(kwargs)
        return input_embeds, input_ids.reshape(-1, 1)

    state.model.talker_mtp = raw_talker_mtp
    state._talker_mtp_runner = MagicMock(side_effect=AssertionError("seeded sampling must not replay the outer graph"))
    generators = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(22)]

    state._call_talker_mtp_runner(
        torch.tensor([101, 202]),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        generators=generators,
    )

    assert raw_calls == [{"generators": generators}]
    state._talker_mtp_runner.assert_not_called()


def test_run_batched_mtp_uses_dispatched_graph_descriptor():
    state = _make_state(max_num_reqs=4)

    class _FakeGraphRunner:
        def __call__(self, input_ids, input_embeds, last_hidden, text_step, **kwargs):
            assert input_ids.shape[0] == 4
            return input_embeds + 2, torch.arange(12, dtype=torch.long).reshape(4, 3)

    state._talker_mtp_runner = _FakeGraphRunner()
    embed_dim = 3
    state._mtp_input_ids = torch.zeros(4, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros((4, embed_dim))
    state._mtp_hidden = torch.zeros((4, embed_dim))
    state._mtp_text_step = torch.zeros((4, embed_dim))
    state.intermediate_buffer.buffers[0] = {"req_id": "r0"}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1"}
    input_ids = torch.tensor([101, 202], dtype=torch.long)
    embeds = torch.zeros((2, embed_dim))
    batch = _DummyInputBatch([0, 1])
    mtp_batches = [
        (0, 0, (torch.ones(embed_dim), torch.ones(embed_dim) * 2)),
        (1, 1, (torch.ones(embed_dim) * 3, torch.ones(embed_dim) * 4)),
    ]
    graph_desc = SimpleNamespace(cg_mode="FULL", num_tokens=4)
    set_ctx = MagicMock(return_value=nullcontext())

    with (
        patch(
            "vllm_omni.worker_v2.model_states.omni_model_state.current_omni_platform.get_graph_wrapper_cls",
            return_value=_FakeGraphRunner,
        ),
        patch("vllm.forward_context.set_forward_context", set_ctx),
    ):
        state._run_batched_mtp(
            mtp_batches,
            input_ids,
            embeds,
            batch,
            {("codes", "audio")},
            lambda bsz: graph_desc,
        )

    _, kwargs = set_ctx.call_args
    assert kwargs["batch_descriptor"] is graph_desc
    assert kwargs["num_tokens"] == 4
    assert torch.equal(embeds, torch.full((2, embed_dim), 2.0))
    assert torch.equal(state.intermediate_buffer.buffers[0]["codes"]["audio"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(state.intermediate_buffer.buffers[1]["codes"]["audio"], torch.tensor([[3, 4, 5]]))


def test_run_batched_mtp_uses_scalar_fallback_without_per_row_generators():
    state = _make_state(max_num_reqs=2)
    state.model.talker_mtp_accepts_per_row_generators = False
    state.intermediate_buffer.buffers[0] = {
        "req_id": "r0",
        "sampling_params": SimpleNamespace(extra_args={"tts_local_seed": 11}, seed=None),
    }
    state.intermediate_buffer.buffers[1] = {
        "req_id": "r1",
        "sampling_params": SimpleNamespace(extra_args={"tts_local_seed": 22}, seed=None),
    }

    call_batch_sizes = []

    def talker_mtp(input_ids, input_embeds, last_hidden, text_step, **kwargs):
        call_batch_sizes.append(int(input_ids.shape[0]))
        assert "generator" in kwargs
        return input_embeds + 1, input_ids.reshape(1, 1)

    state.model.talker_mtp = talker_mtp
    input_ids = torch.tensor([101, 202], dtype=torch.long)
    embeds = torch.zeros((2, 3))
    batch = _DummyInputBatch([0, 1])
    mtp_batches = [
        (0, 0, (torch.ones(3), torch.ones(3) * 2)),
        (1, 1, (torch.ones(3) * 3, torch.ones(3) * 4)),
    ]

    with patch("vllm.forward_context.set_forward_context", return_value=nullcontext()):
        state._run_batched_mtp(mtp_batches, input_ids, embeds, batch, {("codes", "audio")})

    assert call_batch_sizes == [1, 1]


def test_prepare_attn_delegates_complete_vllm_025_contract():
    state = _make_state()
    input_batch = object()
    block_tables = (object(),)
    slot_mappings = object()
    attn_groups = [[object()]]
    kv_cache_config = object()
    expected = {"attn": object()}

    with patch.object(
        type(state).__bases__[0],
        "prepare_attn",
        return_value=expected,
    ) as upstream:
        result = state.prepare_attn(
            input_batch,
            CUDAGraphMode.FULL,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=True,
        )

    assert result is expected
    upstream.assert_called_once_with(
        input_batch,
        CUDAGraphMode.FULL,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        True,
    )


def test_safe_get_rope_does_not_hide_signature_type_errors():
    def broken_get_rope(*_args, **_kwargs):
        raise TypeError("vLLM rope API drift")

    wrapped = _make_safe_get_rope(broken_get_rope)

    with pytest.raises(TypeError, match="rope API drift"):
        wrapped(SimpleNamespace(uses_mrope=False), object())


def test_request_state_num_computed_wins_when_input_batch_is_full():
    state = _make_state(max_num_reqs=2, has_preprocess=True)
    state.intermediate_buffer.buffers[1] = {"req_id": "r1"}
    state.model.preprocess.return_value = (
        torch.tensor([1]),
        torch.zeros(1, 2),
        {},
    )
    batch = _DummyInputBatch([1, 0], num_computed_tokens_cpu=[100, 200])
    batch.input_ids = torch.tensor([1, 2])
    batch.num_tokens = 2
    req_states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array([5, 5], dtype=np.int32)),
        num_computed_tokens=SimpleNamespace(np=np.array([3, 4], dtype=np.int32)),
    )

    state.run_preprocess(
        batch,
        {"input_ids": batch.input_ids, "inputs_embeds": torch.zeros(2, 2)},
        req_states,
    )

    info = state.model.preprocess.call_args.kwargs
    assert info["_omni_num_computed_tokens"] == 4
    assert info["_omni_is_prefill"] is True


def test_postprocess_model_output_propagates_make_output_failure():
    state = _make_state(have_multimodal_outputs=True)
    state.model.make_omni_output.side_effect = RuntimeError("broken payload")
    input_batch = _DummyInputBatch([0])

    with pytest.raises(RuntimeError, match="broken payload"):
        state.postprocess_model_output((torch.zeros(1, 2), []), input_batch, object())
