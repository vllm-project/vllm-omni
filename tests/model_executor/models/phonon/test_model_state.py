# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder
from vllm_omni.model_executor.models.phonon.model_state import PhononModelState
from vllm_omni.model_executor.models.phonon.phonon_talker import PhononFlowLM
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_feedback_eos_output_ownership_and_slot_reuse(monkeypatch, async_scheduling):
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

    class FakeFlow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bos_emb = torch.nn.Parameter(torch.zeros(4))
            self.input_linear = torch.nn.Linear(4, 8, bias=False)
            self.register_buffer("emb_mean", torch.zeros(4))
            self.register_buffer("emb_std", torch.ones(4))

        def conditioning(self):
            return self.bos_emb.new_zeros(8)

        def sample(self, hidden, noise):
            return noise, hidden[:, :1]

    model = SimpleNamespace(phonon_config=config, flow_lm=FakeFlow(), mimi=MimiStreamingDecoder(config["mimi"], 1))
    buffers = [{}]
    intermediate = SimpleNamespace(buffers=buffers)

    def base_init(owner, *args):
        owner.model = model
        owner.scheduler_config = SimpleNamespace(max_num_seqs=1, async_scheduling=async_scheduling)
        owner.dtype = torch.float32
        owner._static_inputs_embeds = torch.zeros(16, 8)
        owner.intermediate_buffer = intermediate

    monkeypatch.setattr(OmniModelState, "__init__", base_init)
    monkeypatch.setattr(OmniModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(OmniModelState, "remove_request", lambda *args: None)
    cfg = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    state = PhononModelState(cfg, model, None, torch.device("cpu"))
    flow_flags = []
    original_sample = state.flow.sample

    def sample_with_flag_reference(*args):
        result = original_sample(*args)
        flow_flags.append(result[0])
        return result

    monkeypatch.setattr(state.flow, "sample", sample_with_flag_reference)
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
            _, marker = state.postprocess_model_output(torch.ones(1, 8), batch, None)
            assert marker
            output = state.prepare_streaming_audio_output(batch, None, marker)
            assert (output.valid is flow_flags[-1]) == (not async_scheduling)
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
