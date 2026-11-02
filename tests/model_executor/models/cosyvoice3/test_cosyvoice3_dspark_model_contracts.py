# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU model/real-Eagle-mixin contracts; no checkpoint or CUDA execution."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
from vllm_omni.model_executor.models.cosyvoice3.dspark import UNIFIED_MASK_TOKEN_ID, converted_config
from vllm_omni.worker_v2 import staged_writes
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_capture_layers_match_converted_checkpoint_conventions():
    class Decoder(torch.nn.Module, EagleModelMixin):
        pass

    decoder = Decoder()
    model = CosyVoice3Model.__new__(CosyVoice3Model)
    torch.nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.talker = SimpleNamespace(llm=SimpleNamespace(model=decoder))
    layers = model.get_eagle3_default_aux_hidden_state_layers()
    config = converted_config(
        dict(
            speculators_model_type="dspark",
            markov_head_type="vanilla",
            mask_token_id=UNIFIED_MASK_TOKEN_ID,
            aux_hidden_state_layer_ids=list(layers),
            transformer_layer_config=dict(hidden_size=896),
            markov_rank=256,
            block_size=8,
        ),
        6500,
    )
    model.set_aux_hidden_state_layers(tuple(config["eagle_aux_hidden_state_layer_ids"]))
    assert decoder.aux_hidden_state_layers == layers
    captured: list[torch.Tensor] = []
    for index in range(24):
        decoder._maybe_add_hidden_state(captured, index, torch.full((1, 2), index), None)
    assert [int(value[0, 0]) for value in captured] == config["eagle_aux_hidden_state_layer_ids"]
    assert config["target_layer_ids"] == [index - 1 for index in layers]
    assert config["dflash_config"]["target_layer_ids"] == config["target_layer_ids"]
    # Drafter layers must not opt into the distinct Omni-owned aux tuple slot.
    assert not getattr(model, "_returns_tuple", False)
    assert not getattr(model, "supports_mrv2_full_graph_aux_outputs", False)
    assert not hasattr(model, "_last_captured_layers")


def test_dspark_language_model_is_the_decoder_not_the_speech_head():
    model = CosyVoice3Model.__new__(CosyVoice3Model)
    torch.nn.Module.__init__(model)
    model._dspark_config = object()
    decoder = object()
    model.talker = SimpleNamespace(llm=decoder)
    assert model.get_language_model() is decoder


@pytest.mark.parametrize("cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("dummy_run", [False, True])
def test_talker_draft_tuple_is_unwrapped_before_logits_and_omni_output(cg_mode, dummy_run, monkeypatch):
    """Exercise real Talker methods with mocked attention/preparation and graph replay."""
    hidden = torch.ones(1, 2)
    aux = [hidden + index for index in (1, 2, 3)]
    embeds = torch.zeros_like(hidden)
    model = CosyVoice3Model.__new__(CosyVoice3Model)
    torch.nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model._dspark_config = object()
    model.config = SimpleNamespace(llm={"speech_token_size": 3, "eos_token_id": 3}, cosyvoice3_sampling_mode="standard")
    model.model = SimpleNamespace(llm=MagicMock(return_value=(hidden, aux)), llm_decoder=torch.nn.Linear(2, 203))

    runner = object.__new__(OmniGPUModelRunner)
    runner._staged_writes = staged_writes.StagedWrites()
    # Contract inputs: model, use_aux_hidden_state_outputs and graph tuple output.
    # The request/KV/EPLB state and step hooks below only bypass runner setup.
    runner.model = model
    runner.use_aux_hidden_state_outputs = True
    runner._configure_cudagraph_output_contract()
    runner.model_config = runner.vllm_config = SimpleNamespace()
    runner.req_states = SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.long)))
    runner.model_state = MagicMock()
    runner.model_state.prepare_inputs.return_value = {}
    runner._omni_data_plane = object()
    runner.supports_mm_inputs = False
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.aux_output_connector = None
    runner.eplb = MagicMock()
    runner.kv_connector = MagicMock()
    runner.kv_cache_config = object()
    runner.attn_groups = []
    runner.block_tables = SimpleNamespace(apply_staged_writes=MagicMock())
    for name in (
        "_prefix_cache_step_begin",
        "_prepare_native_data_plane",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
        "_sync_native_data_plane_payloads",
    ):
        monkeypatch.setattr(runner, name, MagicMock())

    input_batch = SimpleNamespace(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        num_tokens=1,
        num_tokens_after_padding=1,
        is_padding=None,
    )
    runner.gather_batch_req_state = MagicMock(return_value=(None if dummy_run else SimpleNamespace(num_tokens=1), 1))
    runner.prepare_inputs = MagicMock(return_value=input_batch)
    runner.prepare_attn = runner.prepare_dummy_attn = MagicMock(return_value=((), torch.empty(0)))
    monkeypatch.setattr(runner, "_make_dummy_input_batch", MagicMock(return_value=input_batch))
    monkeypatch.setattr(runner, "_prepare_mm_inputs", MagicMock(return_value=(input_batch.input_ids, embeds, None)))
    monkeypatch.setattr(runner, "_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("vllm_omni.worker_v2.omni_model_runner.build_slot_mappings_by_layer", lambda *args: {})
    batch_desc = SimpleNamespace(cg_mode=cg_mode, num_reqs=1, num_tokens=1, num_active_loras=0, max_query_len=1)
    runner._dispatch_batch_descriptor = MagicMock(return_value=(batch_desc, None))
    runner.cudagraph_manager = MagicMock()
    runner.cudagraph_manager.run_fullgraph.return_value = (hidden, aux)
    runner.cudagraph_manager.run_pw_graph.side_effect = lambda target, inputs: target(**inputs)
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"_dummy_req_0": 1}
    scheduled.total_num_scheduled_tokens = 1

    runner.execute_model(scheduled, dummy_run=dummy_run, valid_dummy_state_slots=True)

    state = runner.execute_model_state
    assert state.hidden_states is hidden
    assert state.aux_hidden_states is aux
    assert runner._last_aux_output is None
    if cg_mode == CUDAGraphMode.FULL:
        model.model.llm.assert_not_called()
    else:
        model.model.llm.assert_called_once()
    if dummy_run:
        runner.model_state.run_preprocess.assert_not_called()
        runner.model_state.run_postprocess.assert_not_called()
    else:
        runner.model_state.run_preprocess.assert_called_once()
        runner.model_state.run_postprocess.assert_called_once_with(hidden, input_batch)

    logits = model.compute_logits(state.hidden_states)
    output = model.make_omni_output(state.hidden_states)
    assert isinstance(logits, torch.Tensor) and logits.shape == (1, 203)
    assert torch.isfinite(logits).all()
    assert output.text_hidden_states is hidden
    assert output.multimodal_outputs == {}
