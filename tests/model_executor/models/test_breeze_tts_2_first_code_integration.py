# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check only the caller's grouping seam, with the existing weight-free fixture."""

import pytest
import torch
import torch.nn.functional as F

from tests.model_executor.models.test_breeze_tts_2 import _request_info, _small_talker
from vllm_omni.model_executor.models.breeze_tts_2.depth_decoder import sample_logits
from vllm_omni.model_executor.models.breeze_tts_2.prompt import CFG_UNCOND_SUFFIX

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@torch.inference_mode()
def test_first_code_groups_preserve_logits_request_order_cfg_and_eos(monkeypatch):
    """Grouping changes dispatch only; head/history arithmetic and routing stay exact."""
    model = _small_talker()
    cfg_a, cfg_b = "cfg_a" + CFG_UNCOND_SUFFIX, "cfg_b" + CFG_UNCOND_SUFFIX
    order = ["plain_a", "plain_b", cfg_a, "finished_a", "cfg_a", "cfg_b", cfg_b, "plain_c"]
    tuples = {"a": (0.9, 3, 1.0), "b": (0.7, 2, 0.8), "c": (0.0, 0, 1.0)}
    chosen = {"plain_a": 1, "plain_b": 2, "finished_a": 7, "cfg_a": 3, "cfg_b": 0, "plain_c": 2}
    hidden = {
        "plain_a": [1.0, -0.5],
        "plain_b": [-0.5, 1.0],
        cfg_a: [0.25, 0.5],
        "finished_a": [-1.0, -1.0],
        "cfg_a": [0.75, -0.5],
        "cfg_b": [0.5, 0.75],
        cfg_b: [-0.25, 0.5],
        "plain_c": [0.5, -0.75],
    }
    infos = {rid: _request_info(rid) for rid in order}
    for index, (rid, info) in enumerate(infos.items()):
        parent = rid.removesuffix(CFG_UNCOND_SUFFIX)
        parameters = tuples[parent[-1]]
        info["breeze_sampling"].update(
            temperature=parameters[0],
            top_k=parameters[1],
            top_p=parameters[2],
            repetition_penalty=2.0 if parent.startswith("cfg") else 1.25,
        )
        info["breeze_state"]["generator"].manual_seed(100 + index)
        info["breeze_state"]["history"] = torch.tensor([[1, 3, 1]])
        if parent.startswith("cfg"):
            info["breeze_prompt"]["guidance_scale"] = 4.0 if parent == "cfg_a" else 2.0
        if rid.endswith(CFG_UNCOND_SUFFIX):
            info["breeze_prompt"]["role"] = "uncond"
    original_states = {
        rid: {key: value.clone() for key, value in info["breeze_state"].items() if key in ("history", "current")}
        for rid, info in infos.items()
    }
    request_for_generator = {id(info["breeze_state"]["generator"]): rid for rid, info in infos.items()}

    class RecordingSampler:
        def __init__(self):
            self.calls = []

        def sample(self, logits, parameters, generators):
            ids = [request_for_generator[id(generator)] for generator in generators]
            self.calls.append((parameters, ids, [row.clone() for row in logits]))
            return [torch.tensor([chosen[rid]], dtype=torch.long) for rid in ids]

    recorder = RecordingSampler()
    model._first_code_sampler = recorder
    # Build the scalar pre-grouping reference before spying on actual head
    # calls. Both positive and negative repeated logits exercise the penalty.
    expected_logits = {}
    for rid in chosen:
        row = torch.tensor([hidden[rid]], dtype=torch.bfloat16).float()
        scores = F.linear(row, model.lm_head.weight)
        scale = infos[rid]["breeze_prompt"]["guidance_scale"]
        if scale != 1.0:
            uncond = F.linear(
                torch.tensor([hidden[rid + CFG_UNCOND_SUFFIX]], dtype=torch.bfloat16).float(), model.lm_head.weight
            )
            scores = uncond + scale * (scores - uncond)
        scores[:, model.codebook_size : model.config.eos_token_id] = -torch.inf
        history = infos[rid]["breeze_state"]["history"]
        selected = scores.gather(1, history)
        penalty = infos[rid]["breeze_sampling"]["repetition_penalty"]
        scores.scatter_(1, history, torch.where(selected < 0, selected * penalty, selected / penalty))
        expected_logits[rid] = scores

    actual_linear, head_inputs = F.linear, []

    def scalar_head(inputs, weight, bias=None):
        assert inputs.dtype == torch.float32 and inputs.shape == (1, model.hidden_size)
        assert weight is model.lm_head.weight
        head_inputs.append(inputs.clone())
        return actual_linear(inputs, weight, bias)

    # Prefill-like unequal spans must still select each physical row's last
    # hidden state, including a CFG companion preceding its parent.
    flat_hidden: list[list[float]] = []
    spans: list[tuple[int, int]] = []
    for index, rid in enumerate(order):
        start = len(flat_hidden)
        flat_hidden.extend([[-99.0, -99.0]] * (index % 3))
        flat_hidden.append(hidden[rid])
        spans.append((start, len(flat_hidden)))
    values = torch.tensor(flat_hidden, dtype=torch.bfloat16)
    with monkeypatch.context() as head_guard:
        head_guard.setattr(F, "linear", scalar_head)
        output = model.make_omni_output(
            values, model_intermediate_buffer=[infos[rid] for rid in order], request_token_spans=spans
        )
    assert output.text_hidden_states is values
    assert len(head_inputs) == len(chosen) + 2
    assert [(parameters, ids) for parameters, ids, _ in recorder.calls] == [
        (tuples["a"], ["plain_a", "finished_a", "cfg_a"]),
        (tuples["b"], ["plain_b", "cfg_b"]),
        (tuples["c"], ["plain_c"]),
    ]
    for _, ids, rows in recorder.calls:
        for rid, row in zip(ids, rows, strict=True):
            assert row.dtype == torch.float32 and row.shape == (1, model.config.vocab_size)
            torch.testing.assert_close(row, expected_logits[rid], atol=0, rtol=0)

    # Depth group creation must follow original conditioned order, not the
    # first-code sampler's grouped traversal (A,A,A,B,B,C).
    live_order = ["plain_a", "plain_b", "cfg_a", "cfg_b", "plain_c"]
    calls = model.depth_decoder.generate_frames.call_args_list
    assert len(calls) == len(live_order)
    for call, rid in zip(calls, live_order, strict=True):
        depth_hidden, first = call.args
        scale = infos[rid]["breeze_prompt"]["guidance_scale"]
        expected_hidden = [hidden[rid]] + ([hidden[rid + CFG_UNCOND_SUFFIX]] if scale != 1 else [])
        torch.testing.assert_close(depth_hidden, torch.tensor(expected_hidden, dtype=torch.bfloat16), atol=0, rtol=0)
        assert first.tolist() == [chosen[rid]]
        assert call.kwargs["generators"] == [infos[rid]["breeze_state"]["generator"]]
        assert call.kwargs["guidance_scale"] == scale
        assert tuple(call.kwargs[key] for key in ("temperature", "top_k", "top_p")) == tuples[rid[-1]]
    assert model.compute_logits(torch.zeros(len(order), 2)).argmax(-1).tolist() == [0, 0, 0, 7, 0, 0, 0, 0]
    for index, rid in enumerate(order):
        state = infos[rid]["breeze_state"]
        audio = output.multimodal_outputs["codes"]["audio"][index]
        if rid in live_order:
            assert state["history"].tolist() == [[1, 3, 1, chosen[rid]]]
            assert audio.tolist() == [[chosen[rid]] * 3]
        else:
            assert audio.numel() == 0
            torch.testing.assert_close(state["history"], original_states[rid]["history"], atol=0, rtol=0)
    torch.testing.assert_close(
        infos["finished_a"]["breeze_state"]["current"], original_states["finished_a"]["current"], atol=0, rtol=0
    )
    for parent, companion in (("cfg_a", cfg_a), ("cfg_b", cfg_b)):
        assert infos[parent]["breeze_state"]["current"] is infos[companion]["breeze_state"]["current"]


@torch.inference_mode()
def test_first_code_shared_generator_preserves_order_across_parameter_groups(monkeypatch):
    """A shared RNG stream must retain A,B,A request order across tuple groups."""
    model = _small_talker()
    infos = [_request_info(rid) for rid in ("first_a", "middle_b", "last_a")]
    parameters = [(0.9, 3, 1.0), (0.7, 2, 0.8), (0.9, 3, 1.0)]
    actual_generator = torch.Generator().manual_seed(2**63 + 19)
    expected_generator = torch.Generator().manual_seed(2**63 + 19)
    hidden = torch.tensor([[0.2, 0.0], [0.0, 0.2], [0.1, 0.2]], dtype=torch.bfloat16)
    expected = []
    for index, (info, values) in enumerate(zip(infos, parameters, strict=True)):
        info["breeze_sampling"].update(temperature=values[0], top_k=values[1], top_p=values[2])
        info["breeze_state"]["generator"] = actual_generator
        logits = F.linear(hidden[index : index + 1].float(), model.lm_head.weight)
        logits[:, model.codebook_size : model.config.eos_token_id] = -torch.inf
        expected.append(sample_logits(logits, *values, expected_generator))
    assert all(token.item() != model.config.eos_token_id for token in expected)
    sample, calls = model._first_code_sampler.sample, []

    def record_sample(logits, settings, generators):
        calls.append((settings, len(logits), [generator is actual_generator for generator in generators]))
        return sample(logits, settings, generators)

    monkeypatch.setattr(model._first_code_sampler, "sample", record_sample)
    output = model.make_omni_output(
        hidden, model_intermediate_buffer=infos, request_token_spans=[(0, 1), (1, 2), (2, 3)]
    )
    assert calls == [(values, 1, [True]) for values in parameters]
    torch.testing.assert_close(actual_generator.get_state(), expected_generator.get_state(), atol=0, rtol=0)
    for info, token, audio in zip(infos, expected, output.multimodal_outputs["codes"]["audio"], strict=True):
        torch.testing.assert_close(audio, token[:, None].repeat(1, 3), atol=0, rtol=0)
        torch.testing.assert_close(info["breeze_state"]["history"], token[:, None], atol=0, rtol=0)
        assert info["breeze_state"]["generator"] is actual_generator
