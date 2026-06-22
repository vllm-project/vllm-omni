# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the StepAudioEditX AR wrapper."""

from unittest.mock import patch

import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio_editx.step_audio_ar import (
    StepAudioAR,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeInnerModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, hidden_size))
        self.forward_calls: list[dict] = []

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.to(torch.float32).reshape(-1, 1)
        return values.expand(-1, self.weight.shape[1])

    def __call__(self, **kwargs):
        self.forward_calls.append(kwargs)
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            return inputs_embeds + 1
        return self.embed_input_ids(kwargs["input_ids"]) + 1

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.sum(dim=-1, keepdim=True)

    def load_weights(self, _weights):
        return {
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
        }


def _make_ar() -> StepAudioAR:
    model = StepAudioAR.__new__(StepAudioAR)
    torch.nn.Module.__init__(model)
    model.model = _FakeInnerModel()
    model.have_multimodal_outputs = True
    model.has_preprocess = True
    model.has_postprocess = False
    return model


def test_make_omni_output_keeps_ref_codes_batch_aligned() -> None:
    model = _make_ar()
    hidden = torch.zeros((2, 4))
    ref_a = torch.tensor([65536, 65537], dtype=torch.long)
    ref_b = torch.tensor([65538, 65539, 65540], dtype=torch.long)

    out = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {"codes": {"ref": ref_a}},
            {"codes": {"ref": ref_b}},
        ],
    )

    assert out.text_hidden_states is hidden
    ref_list = out.multimodal_outputs["codes"]["ref"]
    assert len(ref_list) == 2
    assert torch.equal(ref_list[0], ref_a)
    assert torch.equal(ref_list[1], ref_b)


def test_make_omni_output_pads_requests_without_ref_code() -> None:
    model = _make_ar()
    ref_b = torch.tensor([65538, 65539], dtype=torch.long)

    out = model.make_omni_output(
        torch.zeros((2, 4)),
        model_intermediate_buffer=[
            {"codes": {}},
            {"codes": {"ref": ref_b}},
        ],
    )

    ref_list = out.multimodal_outputs["codes"]["ref"]
    assert len(ref_list) == 2
    assert ref_list[0].numel() == 0
    assert torch.equal(ref_list[1], ref_b)


def test_make_omni_output_omits_multimodal_payload_without_ref_code() -> None:
    model = _make_ar()

    out = model.make_omni_output(
        torch.zeros((2, 4)),
        model_intermediate_buffer=[{"codes": {}}, {"codes": {}}],
    )

    assert out.multimodal_outputs == {}


def test_preprocess_first_prefill_builds_and_caches_prompt_embeds() -> None:
    model = _make_ar()
    full_prompt = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ref_code = torch.tensor([65536, 65537], dtype=torch.long)
    tts_pad = torch.full((1, 4), -1.0)

    with patch.object(
        model,
        "_build_prompt_embeds",
        return_value=(full_prompt, ref_code, tts_pad),
    ) as build_prompt:
        out_ids, out_embeds, update = model.preprocess(
            input_ids=torch.tensor([10, 11], dtype=torch.long),
            input_embeds=None,
            additional_information={
                "text": ["hello"],
                "ref_text": ["world"],
            },
        )

    build_prompt.assert_called_once()
    assert out_ids.tolist() == [0, 0]
    torch.testing.assert_close(out_embeds, full_prompt[:2])
    assert update["meta"]["talker_prefill_offset"] == 2
    torch.testing.assert_close(update["embed"]["prefill"], full_prompt)
    torch.testing.assert_close(update["embed"]["tts_pad"], tts_pad)
    assert torch.equal(update["codes"]["ref"], ref_code)


def test_preprocess_subsequent_prefill_slices_cached_prompt_and_pads() -> None:
    model = _make_ar()
    full_prompt = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tts_pad = torch.full((1, 4), -1.0)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([10, 11], dtype=torch.long),
        input_embeds=None,
        embed={"prefill": full_prompt, "tts_pad": tts_pad},
        meta={"talker_prefill_offset": 2},
    )

    assert out_ids.tolist() == [0, 0]
    expected = torch.cat([full_prompt[2:3], tts_pad], dim=0)
    torch.testing.assert_close(out_embeds, expected)
    assert update["meta"]["talker_prefill_offset"] == 3


def test_preprocess_decode_uses_sampled_token_embedding_and_forwards_audio_code() -> None:
    model = _make_ar()
    full_prompt = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    tts_pad = torch.full((1, 4), -1.0)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        embed={"prefill": full_prompt, "tts_pad": tts_pad},
        meta={"talker_prefill_offset": 2},
    )

    assert out_ids.tolist() == [123]
    torch.testing.assert_close(out_embeds, torch.full((1, 4), 123.0))
    assert update["codes"]["audio"].tolist() == [[123]]


def test_compute_logits_unwraps_omni_output() -> None:
    model = _make_ar()
    hidden = torch.ones((2, 4))

    logits = model.compute_logits(OmniOutput(text_hidden_states=hidden, multimodal_outputs={}))

    torch.testing.assert_close(logits, torch.full((2, 1), 4.0))


def test_load_weights_fixes_step1_norm_names() -> None:
    model = _make_ar()

    loaded = model.load_weights(iter(()))

    assert "model.model.layers.0.input_layernorm.weight" in loaded
    assert "model.layers.0.self_attn.q_proj.weight" in loaded
