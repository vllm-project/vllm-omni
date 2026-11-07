# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder
from vllm_omni.model_executor.models.phonon.model_state import PhononModelState
from vllm_omni.model_executor.models.phonon.phonon_talker import PhononFlowLM, PhononLayer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_feedback_eos_output_ownership_and_slot_reuse(monkeypatch):
    config = {
        "flow_lm": {
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "max_period": 10000.0,
            "ldim": 4,
            "n_bins": 20,
            "flow_dim": 8,
            "flow_depth": 1,
        },
        "mimi": {
            "dimension": 8,
            "quantizer_dimension": 4,
            "quantizer_output_dimension": 8,
            "transformer_d_model": 8,
            "transformer_num_heads": 2,
            "transformer_num_layers": 1,
            "transformer_context": 3,
            "transformer_max_period": 10000.0,
            "transformer_dim_feedforward": 16,
            "transformer_layer_scale": 0.01,
            "ratios": [2],
            "sample_rate": 8,
            "frame_rate": 2,
            "n_filters": 4,
            "kernel_size": 3,
            "last_kernel_size": 3,
            "channels": 1,
            "n_residual_layers": 1,
            "residual_kernel_size": 3,
            "dilation_base": 2,
            "compress": 2,
        },
        "lsd_decode_steps": 1,
        "eos_threshold": -1000.0,
    }
    model = SimpleNamespace(
        phonon_config=config, flow_lm=PhononFlowLM(config), mimi=MimiStreamingDecoder(config["mimi"], 1)
    )
    buffers = [{}]
    intermediate = SimpleNamespace(buffers=buffers)

    def base_init(owner, *args):
        owner.model = model
        owner.scheduler_config = SimpleNamespace(max_num_seqs=1)
        owner.dtype = torch.float32
        owner._static_inputs_embeds = torch.zeros(16, 8)
        owner.intermediate_buffer = intermediate

    monkeypatch.setattr(OmniModelState, "__init__", base_init)
    monkeypatch.setattr(OmniModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(OmniModelState, "remove_request", lambda *args: None)
    cfg = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    state = PhononModelState(cfg, model, None, torch.device("cpu"))
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.array([0]),
        query_start_loc=torch.tensor([0, 1]),
        query_start_loc_np=np.array([0, 1]),
        num_scheduled_tokens=[1],
        num_computed_tokens_np=np.array([1]),
        num_tokens_after_padding=1,
    )
    buffers[0] = {"req_id": "first", "phonon_prompt_embeds": torch.ones(1, 8), "phonon_tail_frames": 1}
    state.add_request(0, SimpleNamespace(req_id="first"))
    assert state.tail_frames[0] == 1
    with torch.inference_mode():
        for step in range(2):
            state.run_preprocess(batch, {})
            state.run_postprocess(torch.ones(1, 8), batch)
            assert torch.equal(buffers[0]["next_embedding"], state.next_embeddings[0])
            _, marker = state.postprocess_model_output(torch.ones(1, 8), batch, None)
            assert marker
            output = state.prepare_streaming_audio_output(batch, None, marker)
            assert output.get_output()[0] is not None
            assert bool(model._continue[0]) == (step == 0)
        old_histories = [module.history.clone() for module in model.mimi.modules() if hasattr(module, "history")]
        old_audio = output.audio.clone()
        state.remove_request(0)
        torch.testing.assert_close(output.audio, old_audio)
        buffers[0] = {"req_id": "second", "phonon_prompt_embeds": torch.ones(1, 8)}
        state.add_request(0, SimpleNamespace(req_id="second"))
        assert state.mimi_positions[0] == 0
        assert state.remaining_tail[0] == -1
        assert state.tail_frames[0] == 3
        assert state.next_embeddings.count_nonzero() == 0
        assert state.active[0]
        histories = [module.history for module in model.mimi.modules() if hasattr(module, "history")]
        for actual, previous in zip(histories, old_histories, strict=True):
            torch.testing.assert_close(actual, previous)
    with pytest.raises(RuntimeError, match="preempted"):
        state.on_request_preempted("second", 0)


def test_official_loader_preserves_backend_buffers_and_requires_model_weights():
    lm = PhononFlowLM.__new__(PhononFlowLM)
    torch.nn.Module.__init__(lm)
    lm.input_linear = torch.nn.Linear(4, 8, bias=False)
    lm.transformer = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    backend = torch.nn.Module()
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        backend.register_buffer(name, torch.tensor(1.0))
    layer.self_attn.attention = backend
    lm.transformer.layers = torch.nn.ModuleList([layer])
    weights = {"flow_lm.input_linear.weight": torch.randn(8, 4)}
    lm.load_official_weights(weights)
    torch.testing.assert_close(lm.input_linear.weight, weights["flow_lm.input_linear.weight"])
    for buffer in backend.buffers():
        assert buffer.item() == 1.0
    with pytest.raises(RuntimeError, match="input_linear.weight"):
        lm.load_official_weights({})
    with pytest.raises(RuntimeError, match="Unexpected key"):
        lm.load_official_weights({**weights, "flow_lm.unknown.weight": torch.zeros(1)})


def test_bf16_ar_rope_preserves_dtype_in_eager_and_compiled_forward():
    config = {"d_model": 8, "num_heads": 2, "dim_feedforward": 16, "max_period": 10000.0}
    layer = PhononLayer(config).to(dtype=torch.bfloat16).eval()
    inputs = torch.randn(3, 8, dtype=torch.bfloat16)
    positions = torch.arange(4096, 4099)
    cache = [torch.empty(0, 2, 4, dtype=torch.bfloat16) for _ in range(2)]
    with torch.inference_mode():
        expected = layer(inputs, positions, cache)
        assert expected.dtype == torch.bfloat16
        assert all(tensor.dtype == torch.bfloat16 for tensor in cache)
        assert torch.isfinite(expected).all()
        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        fresh_cache = [torch.empty(0, 2, 4, dtype=torch.bfloat16) for _ in range(2)]
        actual = compiled(inputs, positions, fresh_cache)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("project_voice", [True, False])
def test_voice_uses_tensor_transport_without_python_list(monkeypatch, project_voice):
    import sentencepiece
    from transformers.utils import hub

    from vllm_omni.engine.serialization import (
        deserialize_additional_information,
        serialize_additional_information,
    )
    from vllm_omni.entrypoints.openai.tts_adapters.phonon import PhononAdapter
    from vllm_omni.model_executor.models.phonon import prompt as prompt_module

    shape = (1, 512, 125) if project_voice else (1, 125, 512)
    voice = torch.randn(shape)
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(
        sentencepiece, "SentencePieceProcessor", lambda **kwargs: SimpleNamespace(encode=lambda text: [2, 3])
    )
    monkeypatch.setattr(prompt_module, "load_voice", lambda *args: (voice, project_voice))
    config = SimpleNamespace(model="test/voice", hf_config=SimpleNamespace(to_dict=lambda: {}))
    adapter = PhononAdapter(SimpleNamespace(engine_client=SimpleNamespace(model_config=config)))
    prepared = adapter._build(SimpleNamespace(input="Hello world"))
    info = prepared.prompt["additional_information"]
    assert info["phonon_voice"] is voice
    payload = serialize_additional_information(info)
    entry = payload.entries["phonon_voice"]
    assert entry.tensor_data is not None
    assert entry.list_data is None
    restored = deserialize_additional_information(payload)
    assert isinstance(restored["phonon_voice"], torch.Tensor)
    torch.testing.assert_close(restored["phonon_voice"], voice)
    assert restored["phonon_project_voice"] == project_voice
    assert len(prepared.prompt["prompt_token_ids"]) == 128


def test_slot_reset_uses_current_stream_without_cross_stream_wait(monkeypatch):
    from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import MimiDecodeRunner

    calls = []
    runner = MimiDecodeRunner.__new__(MimiDecodeRunner)
    runner.decoder = SimpleNamespace(reset_slots=lambda slot: calls.append(slot))

    def unexpected_stream_operation(*args, **kwargs):
        pytest.fail("Slot reset must use the current stream directly")

    monkeypatch.setattr(torch.cuda, "stream", unexpected_stream_operation)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_stream_operation)
    runner.reset_slots(0)
    assert calls == [0]


def test_mimi_runner_dispatch_with_lightweight_decoder():
    from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import MimiDecodeRunner

    class FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))
            self.register_buffer("history", torch.zeros(2, 4))

        def forward(self, latent, slots, positions):
            previous = torch.where(positions[:, None] == 0, 0, self.history[slots])
            output = previous + latent.squeeze(-1) * self.scale
            self.history[slots] = output
            return output[:, None, :]

    decoder = FakeDecoder()
    runner = MimiDecodeRunner(decoder)
    with torch.inference_mode():
        first, event = runner.decode(torch.ones(1, 4, 1), torch.tensor([1]), torch.tensor([0]))
        assert event is None
        later, _ = runner.decode(torch.ones(1, 4, 1), torch.tensor([1]), torch.tensor([1]))
        torch.testing.assert_close(first, torch.ones(1, 1, 4))
        torch.testing.assert_close(later, torch.full((1, 1, 4), 2.0))
        reused, _ = runner.decode(torch.full((1, 4, 1), 3.0), torch.tensor([1]), torch.tensor([0]))
        torch.testing.assert_close(reused, torch.full((1, 1, 4), 3.0))
        assert decoder.history[0].count_nonzero() == 0
