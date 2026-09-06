# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _IdentityTalker:
    num_code_groups = 2

    @staticmethod
    def parameters():
        return iter((torch.empty(0),))

    @staticmethod
    def text_projection(value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def embed_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).reshape(-1, 1).repeat(1, 2)


def _model() -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = _IdentityTalker()
    model.tts_eos_embed = torch.tensor([[90.0, 91.0]])
    model.tts_pad_embed = torch.tensor([[80.0, 81.0]])
    return model


def test_talker_postprocess_batch_gathers_each_request_tail():
    model = _model()
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    key, values = model.postprocess_batch_mrv2(
        hidden_states=hidden,
        last_token_indices=torch.tensor([1, 3]),
    )

    assert key == ("hidden_states", "last")
    assert torch.equal(values, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_talker_prefill_marks_placeholder_codec_rows_invalid() -> None:
    model = _model()
    model.talker_preprocess_prefill = lambda input_ids, input_embeds, _payload: (
        input_ids,
        input_embeds,
        {},
    )

    _, _, updates = model.talker_preprocess(
        torch.tensor([1, 2], dtype=torch.long),
        torch.ones(2, 2),
        _omni_is_prefill=True,
        meta={},
    )

    assert torch.equal(updates["codes"]["audio"], torch.zeros((2, 2), dtype=torch.long))
    assert updates["meta"]["codec_frame_valid"].item() is False


def test_talker_mtp_forwards_request_sampling_state_to_code_predictor() -> None:
    calls = []

    class _Talker:
        num_code_groups = 2

        @staticmethod
        def code_predictor_forward(*args, **kwargs):
            calls.append(kwargs)
            return torch.tensor([[[1], [2]]]), torch.ones(1, 1, 2)

    model = _model()
    model.talker = _Talker()
    model.talker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
    generator = torch.Generator().manual_seed(42)

    model.talker_mtp(
        torch.tensor([1]),
        torch.ones(1, 2),
        torch.ones(1, 2),
        torch.ones(1, 2),
        do_sample=True,
        temperature=0.8,
        top_k=25,
        top_p=0.95,
        generator=generator,
    )

    assert len(calls) == 1
    assert calls[0]["do_sample"] is True
    assert calls[0]["temperature"] == 0.8
    assert calls[0]["top_k"] == 25
    assert calls[0]["top_p"] == 0.95
    assert calls[0]["generator"] is generator


def test_qwen3_omni_code_predictor_forwards_generator_for_every_position() -> None:
    calls = []

    class _CodePredictor:
        def __call__(self, layer0_code, layer0_embed, last_talker_hidden, **kwargs):
            calls.append(kwargs)
            batch_size = layer0_code.shape[0]
            proj_buf = torch.ones(batch_size, 3, 2)
            return torch.ones(batch_size, 2, 1, dtype=torch.long), proj_buf

    talker = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(codec_embedding=lambda ids: torch.ones(*ids.shape, 2))),
        config=SimpleNamespace(code_predictor_config=SimpleNamespace(hidden_size=2)),
        num_code_groups=2,
        code_predictor=_CodePredictor(),
    )
    generator = torch.Generator().manual_seed(42)

    Qwen3OmniMoeTalkerForConditionalGeneration.code_predictor_forward(
        talker,
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, 2),
        last_talker_hidden=torch.ones(1, 2),
        generator=generator,
    )

    assert len(calls) == 2
    assert all(call["generator"] is generator for call in calls)
    assert all(call["generators"] is None for call in calls)


def test_talker_decode_consumes_absolute_cached_row_before_new_delta() -> None:
    model = _model()
    cached_row = torch.tensor([[11.0, 11.5]])
    new_rows = torch.tensor(
        [
            [20.0, 20.5],
            [30.0, 30.5],
            [40.0, 40.5],
            [50.0, 50.5],
            [60.0, 60.5],
            [70.0, 70.5],
        ]
    )
    payload = {
        "embed": {
            "cached_decode": cached_row,
            "cached_decode_token_start": 1,
            "cached_decode_token_end": 2,
            "decode": new_rows,
            "decode_token_start": 2,
            "decode_token_end": 8,
        },
        "meta": {"num_processed_tokens": 1},
    }
    updates = {}

    text_step = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), updates)

    assert text_step.shape == (2,)
    assert torch.equal(text_step, cached_row[0])
    assert updates["embed"]["cached_decode_token_start"] == 1
    assert updates["embed"]["cached_decode_token_end"] == 8
    assert torch.equal(updates["embed"]["cached_decode"], torch.cat([cached_row, new_rows]))


def test_talker_decode_uses_next_absolute_row_without_new_chunk() -> None:
    model = _model()
    cached_rows = torch.tensor(
        [
            [11.0, 11.5],
            [20.0, 20.5],
            [30.0, 30.5],
        ]
    )
    payload = {
        "embed": {
            "cached_decode": cached_rows,
            "cached_decode_token_start": 1,
            "cached_decode_token_end": 4,
        },
        "meta": {"num_processed_tokens": 2},
    }

    updates = {}
    text_step = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), updates)

    assert text_step.shape == (2,)
    assert torch.equal(text_step, cached_rows[1])
    assert "cached_decode" not in updates.get("embed", {})


def test_talker_decode_can_defer_text_projection_for_mrv2_batching() -> None:
    class _ProjectingTalker:
        def text_projection(self, value: torch.Tensor) -> torch.Tensor:
            return value + 100

    model = _model()
    model.talker = _ProjectingTalker()
    cached_rows = torch.tensor([[11.0, 11.5], [20.0, 20.5]])
    payload = {
        "_omni_defer_talker_text_projection": True,
        "embed": {
            "cached_decode": cached_rows,
            "cached_decode_token_start": 1,
            "cached_decode_token_end": 3,
        },
        "meta": {"num_processed_tokens": 1},
    }
    updates = {}

    text_step = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), updates)

    assert torch.equal(text_step, cached_rows[0])
    assert updates["mtp_text_step_requires_projection"] is True


def test_talker_decode_deduplicates_overlapping_absolute_delta() -> None:
    model = _model()
    cached_rows = torch.tensor([[11.0], [20.0], [30.0]])
    overlapping_rows = torch.tensor([[30.0], [40.0], [50.0]])
    payload = {
        "embed": {
            "cached_decode": cached_rows,
            "cached_decode_token_start": 1,
            "cached_decode_token_end": 4,
            "decode": overlapping_rows,
            "decode_token_start": 3,
            "decode_token_end": 6,
        },
        "meta": {"num_processed_tokens": 3},
    }
    updates = {}

    text_step = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), updates)

    assert torch.equal(text_step, cached_rows[2])
    assert updates["embed"]["cached_decode_token_start"] == 3
    assert updates["embed"]["cached_decode_token_end"] == 6
    assert updates["embed"]["cached_decode"].flatten().tolist() == [30.0, 40.0, 50.0]


def test_talker_decode_does_not_turn_temporary_starvation_into_eos() -> None:
    model = _model()
    payload = {
        "embed": {
            "cached_decode": torch.tensor([[11.0, 11.5]]),
            "cached_decode_token_start": 0,
            "cached_decode_token_end": 1,
        },
        "meta": {
            "num_processed_tokens": 1,
            "finished": False,
        },
    }

    with pytest.raises(RuntimeError, match="conditioning row"):
        model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), {})


def test_talker_preprocess_decode_propagates_conditioning_credit_error() -> None:
    model = _model()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    payload = {
        "embed": {
            "cached_decode": torch.tensor([[11.0, 11.5]]),
            "cached_decode_token_start": 0,
            "cached_decode_token_end": 1,
        },
        "meta": {
            "num_processed_tokens": 1,
            "finished": False,
        },
        "hidden_states": {},
    }

    with pytest.raises(RuntimeError, match="conditioning row"):
        model.talker_preprocess_decode(
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, 2),
            {},
            payload,
        )


def test_talker_decode_emits_eos_once_after_finished_horizon_is_consumed() -> None:
    model = _model()
    payload = {
        "embed": {
            "cached_decode": torch.tensor([[11.0, 11.5]]),
            "cached_decode_token_start": 0,
            "cached_decode_token_end": 1,
        },
        "meta": {
            "num_processed_tokens": 1,
            "finished": True,
        },
    }
    eos_updates = {}

    eos = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), eos_updates)

    assert torch.equal(eos, model.tts_eos_embed)
    assert eos_updates["meta"]["eos_emitted"] is True

    payload["meta"].update(eos_updates["meta"])
    pad_updates = {}
    pad = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), pad_updates)

    assert torch.equal(pad, model.tts_pad_embed)
    assert "eos_emitted" not in pad_updates.get("meta", {})


def test_talker_decode_batch_matches_scalar_state_and_conditioning() -> None:
    class _ProjectingTalker(_IdentityTalker):
        @staticmethod
        def text_projection(value: torch.Tensor) -> torch.Tensor:
            return value + 100

    model = _model()
    model.talker = _ProjectingTalker()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    model.talker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
    req_infos = [
        {
            "req_id": "cached",
            "embed": {
                "cached_decode": torch.tensor([[11.0, 11.5], [12.0, 12.5]]),
                "cached_decode_token_start": 1,
                "cached_decode_token_end": 3,
            },
            "hidden_states": {"last": torch.tensor([[1.0, 1.5]])},
            "meta": {"decode_flag": True, "num_processed_tokens": 1, "finished": False},
        },
        {
            "req_id": "merged",
            "embed": {
                "cached_decode": torch.tensor([[20.0, 20.5], [30.0, 30.5]]),
                "cached_decode_token_start": 1,
                "cached_decode_token_end": 3,
                "decode": torch.tensor([[30.0, 30.5], [40.0, 40.5]]),
                "decode_token_start": 2,
                "decode_token_end": 4,
            },
            "hidden_states": {"last": torch.tensor([[2.0, 2.5]])},
            "meta": {"decode_flag": True, "num_processed_tokens": 2, "finished": False},
        },
        {
            "req_id": "eos",
            "embed": {
                "cached_decode": torch.tensor([[50.0, 50.5]]),
                "cached_decode_token_start": 0,
                "cached_decode_token_end": 1,
            },
            "hidden_states": {},
            "meta": {"decode_flag": True, "num_processed_tokens": 1, "finished": True},
        },
        {
            "req_id": "pad",
            "embed": {
                "cached_decode": torch.tensor([[60.0, 60.5]]),
                "cached_decode_token_start": 0,
                "cached_decode_token_end": 1,
            },
            "hidden_states": {"last": torch.tensor([[4.0, 4.5]])},
            "meta": {
                "decode_flag": True,
                "num_processed_tokens": 1,
                "finished": True,
                "eos_emitted": True,
                "tts_pad_steps": 2,
            },
        },
    ]
    input_ids = torch.tensor([101, 202, 303, 404], dtype=torch.long)
    input_embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    scalar_updates = []
    scalar_hidden = []
    scalar_text_steps = []

    for row, info in enumerate(req_infos):
        _ids, _embeds, updates = model.talker_preprocess(
            input_ids[row : row + 1],
            input_embeds[row : row + 1],
            **info,
        )
        hidden, text_step = updates.pop("mtp_inputs")
        scalar_hidden.append(hidden)
        scalar_text_steps.append(text_step.reshape(1, -1))
        scalar_updates.append(updates)

    out_ids, out_embeds, hidden, text_steps, updates = model.preprocess_decode_batch_mrv2(
        input_ids=input_ids,
        input_embeds=input_embeds,
        req_infos=req_infos,
    )

    assert torch.equal(out_ids, input_ids)
    assert torch.equal(out_embeds, input_embeds)
    assert torch.equal(hidden, torch.cat(scalar_hidden))
    assert torch.equal(text_steps, torch.cat(scalar_text_steps))
    for actual, expected in zip(updates, scalar_updates, strict=True):
        assert actual.get("meta", {}) == expected.get("meta", {})
        assert actual.get("embed", {}).keys() == expected.get("embed", {}).keys()
        for key, expected_value in expected.get("embed", {}).items():
            actual_value = actual["embed"][key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert all(update["meta"]["codec_frame_valid"].item() is False for update in updates)


def test_talker_decode_batch_propagates_missing_conditioning_credit() -> None:
    model = _model()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    model.talker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
    req_info = {
        "req_id": "starved",
        "embed": {
            "cached_decode": torch.tensor([[11.0, 11.5]]),
            "cached_decode_token_start": 0,
            "cached_decode_token_end": 1,
        },
        "hidden_states": {},
        "meta": {"decode_flag": True, "num_processed_tokens": 1, "finished": False},
    }

    with pytest.raises(RuntimeError, match="conditioning row"):
        model.preprocess_decode_batch_mrv2(
            input_ids=torch.tensor([101], dtype=torch.long),
            input_embeds=torch.zeros(1, 2),
            req_infos=[req_info],
        )
