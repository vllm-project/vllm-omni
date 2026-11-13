# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.common.request_outputs import (
    audio_sample_count,
    audio_value,
    coerce_int,
    coerce_int_list,
    first_completion,
    multimodal_output,
    sample_rate_hz,
    slice_audio_delta,
    text_delta,
    text_value,
    unwrap_request_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_unwrap_prefers_the_wrapped_request_output_and_its_first_completion() -> None:
    completion = SimpleNamespace(text="hi", multimodal_output={})
    inner = SimpleNamespace(request_id="req", outputs=[completion, SimpleNamespace(text="second")])
    wrapper = SimpleNamespace(request_output=inner)

    output, first = unwrap_request_output(wrapper)

    assert output is inner
    assert first is completion
    assert first_completion(SimpleNamespace(outputs=[])) is None
    assert first_completion(SimpleNamespace()) is None


def test_multimodal_output_takes_the_first_non_empty_mapping_as_a_copy() -> None:
    completion = SimpleNamespace(multimodal_output={"audio": [1.0]})
    output = SimpleNamespace(multimodal_output={})

    result = multimodal_output(output, completion)

    assert result == {"audio": [1.0]}
    assert result is not completion.multimodal_output
    assert multimodal_output(SimpleNamespace(multimodal_output=None), None) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4, 4),
        ("11", 11),
        (2.9, 2),
        (True, 1),
        (None, None),
        ("x", None),
        (object(), None),
        ([1], None),
        (torch.tensor([7]), 7),
        (torch.tensor(9), 9),
        (torch.tensor([], dtype=torch.long), None),
        (np.array([5], dtype=np.int32), 5),
        (np.array([], dtype=np.int32), None),
    ],
)
def test_coerce_int_handles_scalars_strings_tensors_and_arrays(value: object, expected: int | None) -> None:
    assert coerce_int(value) == expected


def test_coerce_int_list_flattens_tensors_and_drops_non_integers() -> None:
    assert coerce_int_list(torch.tensor([[1, 2], [3, 4]])) == [1, 2, 3, 4]
    assert coerce_int_list(np.array([5, 6])) == [5, 6]
    assert coerce_int_list([1, "2", "x", None, torch.tensor(3)]) == [1, 2, 3]
    assert coerce_int_list(None) == []
    assert coerce_int_list("12") == []


def test_audio_value_reads_the_known_keys_and_unwraps_a_single_item_list() -> None:
    assert audio_value({"model_outputs": [np.zeros(2)]}).shape == (2,)
    assert audio_value({"latent": "x"}) == "x"
    assert audio_value({"audio": [1, 2]}) == [1, 2]
    assert audio_value({}) is None


def test_text_value_prefers_multimodal_text_then_llm_text_then_the_completion() -> None:
    completion = SimpleNamespace(text="from completion")
    assert text_value({"text": "mm"}, completion) == "mm"
    assert text_value({"llm_output_text": "llm"}, completion) == "llm"
    assert text_value({}, completion) == "from completion"
    assert text_value({"text": ""}, None) == ""


@pytest.mark.parametrize(
    ("text", "previous", "expected"),
    [("", "he", ""), ("he", "he", ""), ("hello", "he", "llo"), ("new", "old", "new"), ("abc", "", "abc")],
)
def test_text_delta(text: str, previous: str, expected: str) -> None:
    assert text_delta(text, previous) == expected


def test_slice_audio_delta_returns_only_new_samples_for_cumulative_audio() -> None:
    cumulative = np.arange(6, dtype=np.float32)

    assert slice_audio_delta(cumulative, 4).tolist() == [4.0, 5.0]
    assert slice_audio_delta(cumulative, 6) is None
    assert slice_audio_delta(cumulative, 0).tolist() == cumulative.tolist()
    # A shorter waveform than the cursor means the stream restarted: send it whole.
    assert slice_audio_delta(np.arange(3, dtype=np.float32), 4).tolist() == [0.0, 1.0, 2.0]
    assert slice_audio_delta(None, 2) is None
    assert slice_audio_delta(np.zeros(0, dtype=np.float32), 0) is None


def test_slice_audio_delta_keeps_tensors_as_contiguous_tensors() -> None:
    delta = slice_audio_delta(torch.arange(6, dtype=torch.float32).reshape(2, 3), 4)

    assert isinstance(delta, torch.Tensor)
    assert delta.is_contiguous()
    assert delta.tolist() == [4.0, 5.0]
    assert audio_sample_count(torch.zeros(2, 3)) == 6
    assert audio_sample_count("not audio") is None


@pytest.mark.parametrize(
    ("multimodal", "expected"),
    [
        ({"sr": 16000}, 16000),
        ({"sample_rate_hz": 22050}, 22050),
        ({"sr": [48000]}, 48000),
        ({"sr": np.array([8000])}, 8000),
        ({"sr": "bad"}, 24000),
        ({"sr": 0}, 24000),
        ({}, 24000),
    ],
)
def test_sample_rate_hz_falls_back_to_the_default(multimodal: dict, expected: int) -> None:
    assert sample_rate_hz(multimodal, default=24000) == expected
