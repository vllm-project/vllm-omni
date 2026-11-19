# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_H = 4
_Q = 3


class _Talker(nn.Module):
    num_code_groups = _Q

    def __init__(self):
        super().__init__()
        self.text_projection = nn.Identity()
        self.code_predictor = SimpleNamespace(
            _top_k=50,
            _top_p=0.8,
            set_sampling_params=lambda **kwargs: setattr(self, "sampling", kwargs),
        )


def _talker(*, async_chunk: bool = True) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "talker"
    model.talker = _Talker()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk, use_v2_model_runner=True))
    model.tts_eos_embed = torch.full((_H,), -1.0)
    model.tts_pad_embed = torch.full((_H,), -2.0)
    model._codec_codebook_size = 2048
    return model


def _rows(*values: float) -> torch.Tensor:
    return torch.tensor([[value] * _H for value in values])


def _step(model, payload: dict) -> torch.Tensor:
    update: dict = {}
    text_step = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)
    for key, value in update.items():
        payload.setdefault(key, {}).update(value)
    return text_step.reshape(-1)


def test_mtp_frame_valid_rejects_codec_eos_and_special_ids():
    layer0 = torch.tensor([0, 2047, 2048, 2150, -1])
    assert _talker().mtp_frame_valid(layer0).tolist() == [True, True, False, False, False]


def test_decode_rows_are_consumed_one_per_step_in_arrival_order():
    model = _talker()
    # Two rows were already pending (cached at prefill); two more arrive at once.
    payload = {"embed": {"cached_decode": _rows(1, 2), "decode": _rows(3, 4)}, "meta": {}}
    steps = [_step(model, payload)[0].item()]
    for arrival in (None, _rows(5), None, None, None, None):
        payload["embed"]["decode"] = arrival
        steps.append(_step(model, payload)[0].item())
    assert steps == [1, 2, 3, 4, 5, -1, -2]
    assert payload["embed"]["cached_decode"] is None


@pytest.mark.parametrize(("v2", "resumable"), [(True, True), (False, False), (False, True)])
def test_v1_and_resumable_requests_keep_the_indexed_cache_path(v2, resumable):
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = v2
    payload = {"embed": {"cached_decode": _rows(1, 2, 3), "decode": None}, "meta": {"resumable": resumable}}
    payload["meta"]["num_processed_tokens"] = 2
    assert _step(model, payload)[0].item() == 3


def test_make_omni_output_aligns_codes_to_request_spans():
    model = _talker()
    hidden = torch.zeros((8, _H))  # 5 real tokens + graph padding
    decode_codes = torch.tensor([[5, 6, 7]])
    buffers = [
        {"req_id": "_warmup_0_"},  # no preprocess, no codes
        {"req_id": "a", "codes": {"audio": torch.zeros((3, _Q), dtype=torch.long)}},
        {"req_id": "b", "codes": {"audio": decode_codes}},
    ]
    out = model.make_omni_output(
        hidden, model_intermediate_buffer=buffers, request_token_spans=[(0, 1), (1, 4), (4, 5)]
    )
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes.shape == (5, _Q)
    assert codes[4].tolist() == [5, 6, 7]
    assert out.text_hidden_states.shape[0] == 5


def test_make_omni_output_without_spans_keeps_v1_concat():
    model = _talker()
    buffers = [{"codes": {"audio": torch.tensor([[1, 2, 3]])}}, {"codes": {"audio": torch.tensor([[4, 5, 6]])}}]
    out = model.make_omni_output(torch.zeros((4, _H)), model_intermediate_buffer=buffers)
    assert out.multimodal_outputs["codes"]["audio"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert out.text_hidden_states.shape[0] == 2


def test_talker_mtp_feeds_precomputed_sampling_noise_to_one_decode_step():
    """Residual-codebook noise must enter the predictor as a tensor.

    MRv2 captures ``talker_mtp`` in one CUDA graph. Drawing Gumbel noise
    inside that capture replays the graph RNG. The runner fills a static
    uniform buffer each step and the predictor has to consume it.
    """
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    recorded: dict = {}

    class _Predictor:
        def code_predictor_forward(self, input_ids, inputs_embeds, *, last_talker_hidden=None, **kwargs):
            recorded["seq"] = int(input_ids.shape[-1])
            recorded["kwargs"] = kwargs
            hidden = inputs_embeds.shape[-1]
            codes = torch.zeros(input_ids.shape[0], _Q, input_ids.shape[-1], dtype=torch.long)
            summed = torch.zeros(input_ids.shape[0], input_ids.shape[-1], hidden)
            return codes, summed

    model.talker = _Predictor()
    model.talker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=_H))
    uniforms = torch.rand(2, _Q - 1, 8)
    model.talker_mtp(
        torch.zeros(2, 1, dtype=torch.long),
        torch.zeros(2, _H),
        torch.zeros(2, 1, _H),
        torch.zeros(2, _H),
        sample_uniforms=uniforms,
        generator=object(),
    )
    assert recorded["seq"] == 1
    assert recorded["kwargs"]["sample_uniforms"] is uniforms
    assert recorded["kwargs"]["generator"] is None
    assert recorded["kwargs"]["generators"] is None

    model.talker_mtp(
        torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(3, _H),
        torch.zeros(1, 1, _H),
        torch.zeros(3, _H),
        sample_uniforms=uniforms[:1],
    )
    assert recorded["seq"] == 3
    assert recorded["kwargs"]["sample_uniforms"] is None


def test_eager_output_is_token_major_with_invalid_rows():
    model = _talker()
    model.eager_frames_active = True
    hidden = torch.zeros((6, _H))
    buffers = [{"req_id": "a"}, {"req_id": "b"}]
    out = model.make_omni_output(hidden, model_intermediate_buffer=buffers, request_token_spans=[(0, 3), (3, 4)])
    mm = out.multimodal_outputs
    assert mm["codes"]["audio"].shape == (4, _Q)
    assert mm["meta"]["codec_frame_valid"].tolist() == [0, 0, 0, 0]
    assert out.text_hidden_states is hidden
