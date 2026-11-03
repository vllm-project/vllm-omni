# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU parser parity: integer validity, native fallback and output trimming."""

from collections import Counter
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.easymagpie.codec import payload_validity
from vllm_omni.model_executor.models.easymagpie.codec.config import EasyMagpieCodecConfig
from vllm_omni.model_executor.models.easymagpie.codec.model import EasyMagpieCodecForConditionalGeneration as Model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def model(enabled=False):
    value = Model.__new__(Model)
    nn.Module.__init__(value)
    value.config = EasyMagpieCodecConfig()
    value._batched_payload_validity = enabled
    value._payload_counts = Counter()
    value._payload_fallbacks = Counter()
    value.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    return value


def payloads(count, frames):
    return [{"codes": {"audio": torch.arange(frames * 16).reshape(frames, 16) % 1024}} for _ in range(count)]


def compare(items, spans=None, device=torch.device("cpu")):
    original, candidate = model(), model(True)
    left = original._payload_codes(items, device, spans)
    right = candidate._payload_codes(items, device, spans)
    assert torch.equal(left[0], right[0])
    assert left[0].dtype == right[0].dtype == torch.long
    assert left[1:] == right[1:]
    return candidate, right


@pytest.mark.parametrize(
    "count,frames,batched",
    [(7, 2, False), (8, 2, True), (256, 8, True), (512, 8, True), (513, 2, False), (8, 513, False)],
)
def test_exact_batch_and_frame_bounds(count, frames, batched):
    candidate, _ = compare(payloads(count, frames))
    assert bool(candidate._payload_counts["batched_calls"]) is batched
    assert candidate.codec_payload_status()["retained_tensors"] == 0


def test_ragged_flat_noncontiguous_padding_and_terminal_samples():
    items = payloads(8, 6)
    items[0]["codes"]["audio"] = torch.empty((0, 16), dtype=torch.long)
    items[1]["codes"]["audio"][3:] = -1
    items[2]["codes"]["audio"][5, 0] = 1024
    items[3]["codes"]["audio"][5, 1] = 1024
    items[4]["codes"]["audio"] = items[4]["codes"]["audio"].T.contiguous().flatten()
    items[5]["codes"]["audio"] = torch.arange(6 * 32).reshape(6, 32)[:, ::2]
    items[6]["codes"]["audio"][:] = -1
    before = [item["codes"]["audio"].clone() for item in items]
    candidate, result = compare(items)
    assert candidate._payload_counts["batched_calls"] == 1
    assert result[2][:4] == [0, 3 * 1764, 5 * 1764, 5 * 1764 + 882]
    for item, saved in zip(items, before):
        assert torch.equal(item["codes"]["audio"], saved)


def test_zero_scheduled_payload_is_not_parsed():
    items = payloads(8, 2) + [{"codes": "malformed but unscheduled"}]
    candidate, result = compare(items, [(0, 2)] * 8 + [(2, 2)])
    assert result[1] == [2] * 8
    assert candidate._payload_counts["batched_payloads"] == 8


@pytest.mark.parametrize("case", ["list", "int32", "string_device", "subclass"])
def test_unsupported_payloads_retain_native_parser(case):
    items, device = payloads(8, 2), torch.device("cpu")
    if case == "list":
        items[0]["codes"]["audio"] = items[0]["codes"]["audio"].tolist()
    elif case == "int32":
        items[0]["codes"]["audio"] = items[0]["codes"]["audio"].int()
    elif case == "string_device":
        device = "cpu"
    else:

        class TensorSubclass(torch.Tensor):
            pass

        items[0]["codes"]["audio"] = items[0]["codes"]["audio"].as_subclass(TensorSubclass)
    candidate, _ = compare(items, device=device)
    assert candidate._payload_counts["native_calls"] == 1
    assert candidate._payload_counts["batched_calls"] == 0


@pytest.mark.parametrize("case", ["shape", "missing", "scheduled", "span_length"])
def test_malformed_input_keeps_original_exception(case):
    items, spans = payloads(8, 2), None
    if case == "shape":
        items[3]["codes"]["audio"] = torch.zeros((2, 15), dtype=torch.long)
    elif case == "missing":
        items[3] = {}
    elif case == "scheduled":
        spans = [(0, 3)] * 8
    else:
        spans = [(0, 2)]
    errors = []
    for value in (model(), model(True)):
        with pytest.raises(Exception) as caught:
            value._payload_codes(items, torch.device("cpu"), spans)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]


def test_overridden_validity_and_default_off():
    candidate = model(True)
    candidate._valid_sample_count = MethodType(lambda self, rows: 7, candidate)
    assert candidate._payload_codes(payloads(8, 2), torch.device("cpu"))[2] == [7] * 8
    assert candidate._payload_fallbacks["validity_override"] == 1
    assert model().codec_payload_status()["enabled"] is False


def test_flag_default_and_invalid(monkeypatch):
    monkeypatch.delenv(payload_validity.ENV, raising=False)
    assert payload_validity.flag() == "0"
    monkeypatch.setenv(payload_validity.ENV, "invalid")
    with pytest.raises(ValueError):
        payload_validity.flag()


def test_real_forward_preserves_offsets_and_terminal_trimming():
    items = payloads(8, 2)
    items[0]["codes"]["audio"][1, 1] = 1024
    outputs = []
    for value in (model(), model(True)):
        value.codec = lambda codes: torch.arange(codes.shape[0] * 1764, dtype=torch.float32)
        result = value(input_ids=torch.zeros(16, dtype=torch.long), runtime_additional_information=items)
        outputs.append(result.multimodal_outputs["model_outputs"])
    assert len(outputs[0]) == len(outputs[1]) == 8
    for left, right in zip(*outputs):
        assert torch.equal(left, right)
    assert outputs[1][0].numel() == 1764 + 882
    assert outputs[1][1][0] == 2 * 1764
